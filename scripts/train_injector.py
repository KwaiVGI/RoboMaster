"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import logging
import math
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from packaging import version
from PIL import Image, ImageDraw
from torch.utils.data import RandomSampler
from io import BytesIO
import imageio.v2 as imageio

from torchvision import transforms
from torchvision.utils import flow_to_image
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers
from torch import nn

import tensorflow as tf
import tensorflow_datasets as tfds
from IPython import display

import datasets
import random
import time

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from cogvideox_robot_obj_v2.data.bucket_sampler import (ASPECT_RATIO_512,
                                        ASPECT_RATIO_RANDOM_CROP_512,
                                        ASPECT_RATIO_RANDOM_CROP_PROB,
                                        AspectRatioBatchImageVideoSampler,
                                        RandomSampler, get_closest_ratio)
from cogvideox_robot_obj_v2.data.dataset_image_video import (ImageVideoDataset,
                                                ImageVideoSampler,
                                                get_random_mask)
from cogvideox_robot_obj_v2.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox_robot_obj_v2.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox_robot_obj_v2.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox_robot_obj_v2.pipeline.pipeline_cogvideox_inpaint import (
    CogVideoX_Fun_Pipeline_Inpaint, add_noise_to_reference_video,
    get_3d_rotary_pos_embed, get_resize_crop_region_for_grid)
from cogvideox_robot_obj_v2.utils.lora_utils import create_network, merge_lora, unmerge_lora
from cogvideox_robot_obj_v2.utils.discrete_sampler import DiscreteSampling
from cogvideox_robot_obj_v2.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def process_points(points, frames):

    if len(points) >= frames:

        frames_interval = np.linspace(0, len(points) - 1, frames, dtype=int)
        points = [points[i] for i in frames_interval]
        return points

    else:
        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        for i in range(interval):
            insert_num_dict[i] = n

        m = insert_num % interval
        if m > 0:
            frames_interval = np.linspace(0, len(points)-1, m, dtype=int)
            if frames_interval[-1] > 0:
                frames_interval[-1] -= 1
            for i in range(interval):
                if i in frames_interval:
                    insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0

            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += points[i : i + 1] + insert_points
        res += points[-1:]
        
        return res
    

def process_traj(point_path, num_frames, video_size, device="cpu"):
    processed_points = []
    
    points = np.load(point_path)
    points = [tuple(x) for x in points.tolist()]
    h, w = video_size
    points = process_points(points, num_frames)
    points = [[int(w * x / w), int(h * y / h)] for x, y in points]
    points_resized = [] 
    for point in points:
        if point[0] >= w:
            point[0] = w - 1
        elif point[0] < 0:
            point[0] = 0
        elif point[1] >= h:
            point[1] = h - 1
        elif point[1] < 0:
            point[1] = 0
        points_resized.append(point)
    processed_points.append(points_resized)

    return processed_points


def sample_flowlatents(latents, flow_latents, mask, points, diameter, transit_start, transit_end):

    points = points[:,:,::4,:]
    radius = diameter // 2
    channels = latents.shape[1]

    for channel in range(channels):
        latent_value = latents[:, channel, :].unsqueeze(2)[mask>0.].mean()
        for frame in range(transit_start, transit_end):            
            if frame > 0:
                flow_latents[0,:,frame,:,:] = flow_latents[0,:,frame-1,:,:]
            centroid_x, centroid_y = points[0,0,frame]
            centroid_x, centroid_y = int(centroid_x), int(centroid_y)
            for i in range(centroid_y - radius, centroid_y + radius + 1):
                for j in range(centroid_x - radius, centroid_x + radius + 1):
                    if 0 <= i < flow_latents.shape[-2] and 0 <= j < flow_latents.shape[-1]: 
                        if (i - centroid_y) ** 2 + (j - centroid_x) ** 2 <= radius ** 2:
                            flow_latents[0,channel,frame,i,j] = latent_value + 1e-4
            

    return flow_latents

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--condition_drop_rate",
        type=float,
        default=0.1,
        help="Random drop of input condition to enable classifier free guidance.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--low_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--noise_share_in_frames", action="store_true", help="Whether enable noise share in frames."
    )
    parser.add_argument(
        "--noise_share_in_frames_ratio", type=float, default=0.5, help="Noise share ratio.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size_h",
        type=int,
        default=256,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_size_w",
        type=int,
        default=320,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--image_repeat_in_forward",
        type=int,
        default=0,
        help="Num of repeat image in forward.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=3072,
        help="dimension of each basic transformer block.",
    )
    parser.add_argument(
        "--block_interval",
        type=int,
        default=2,
        help="the injector at intervals in transformer blocks to reduce training parameters and improve inference speed.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=226,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    parser.add_argument(
        "--finetune_init",
        action="store_true",
        help="Remove the injector n the first finetune stage w/o loading pretrained ckpt.",
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"inpaint"`.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_optimizer(args, params_to_optimize):

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
    
    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Optional, Tuple, Union
import json
import cv2
import decord
class VideoDataset(Dataset):
    def __init__(
        self,
        train_data_dir: Optional[str] = None,
        sample_n_frames: int = 49,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        self.train_data_dir = Path(train_data_dir) if train_data_dir is not None else None
        self.tracking_dir = '/root/RoboMaster/dataset/tracking_videos_robot_obj'
        self.ground_sam_dir = '/root/RoboMaster/dataset/Grounded-Segment-Anything/outputs_bridge_obj/mask'
        self.sample_n_frames = sample_n_frames
        self.cache_dir = cache_dir
        self.dataset = 'bridge'
        
        dataset_meta_path = os.path.join(self.train_data_dir, self.dataset, 'meta_info.json')
        with open(dataset_meta_path, 'r') as file: self.meta = json.load(file)
        valid_video_names = sorted([video_name.split('.npy')[0] for video_name in os.listdir(self.ground_sam_dir) if video_name.endswith('.npy')])

        eval_path = '/root/RoboMaster/dataset/eval/bridge_eval'
        exclude_video_names = sorted([video_name.split('.mp4')[0] for video_name in os.listdir(eval_path) if video_name.endswith('.mp4')])
        self.valid_video_names = [video_name for video_name in valid_video_names if video_name not in exclude_video_names]
        self.length = len(self.valid_video_names)

        self.pixel_transforms = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        random.seed(int(time.time())+idx)
        data_idx = random.randint(0, self.length-1)
        video_name = self.valid_video_names[data_idx]
        video_path = os.path.join(os.path.dirname(self.train_data_dir), self.meta[video_name]['video_path'])
        prompt_path = os.path.join(os.path.dirname(self.train_data_dir), self.meta[video_name]['prompt_path'])
        transit_path = os.path.join(self.tracking_dir, video_name + '_transit.npy')

        while True:
            try:
                cap = cv2.VideoCapture(video_path)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                # get local rank
                ctx = decord.cpu(0)
                reader = decord.VideoReader(video_path, ctx=ctx, height=height, width=width)
                frame_indexes = [frame_idx for frame_idx in range(len(reader))]
                try:
                    video_chunk = reader.get_batch(frame_indexes).asnumpy()    
                except:
                    video_chunk = reader.get_batch(frame_indexes).numpy()

                pixel_values = np.array(video_chunk)
                pixel_values = rearrange(torch.from_numpy(pixel_values) / 255.0, "f h w c -> f c h w")
                pixel_values = self.pixel_transforms(pixel_values)
                batch_video_length = len(pixel_values)
                
                transit_start, transit_end = np.load(transit_path)
                transit_start = int(transit_start * self.sample_n_frames / batch_video_length)
                transit_end = min(int(transit_end * self.sample_n_frames / batch_video_length), self.sample_n_frames-1)
                assert transit_start >= 0 and transit_start <= transit_end and transit_end < self.sample_n_frames

                # adjust frame length
                assert batch_video_length >= 16
                if batch_video_length >= self.sample_n_frames:
                    frame_indices = np.linspace(0, batch_video_length - 1, self.sample_n_frames, dtype=int)
                    pixel_values = pixel_values[frame_indices]
                else:
                    pixel_values = torch.repeat_interleave(pixel_values, repeats=self.sample_n_frames // batch_video_length, dim=0)
                    batch_video_length = len(pixel_values)
                    frame_indices = np.linspace(0, batch_video_length - 1, self.sample_n_frames-batch_video_length, dtype=int)
                    pixel_values_extend = torch.zeros((self.sample_n_frames, 3, pixel_values.shape[2], pixel_values.shape[3])).to(pixel_values)
                    idx_extend = 0
                    for idx in range(batch_video_length):
                        pixel_values_extend[idx+idx_extend] = pixel_values[idx]
                        if idx in frame_indices:
                            pixel_values_extend[idx+idx_extend+1] = pixel_values[idx]
                            idx_extend += 1
                    pixel_values = pixel_values_extend

                # loading prompt
                with open(prompt_path, 'r') as file: prompt = file.readline().strip().lower()

                # loading tracking points
                points_obj = process_traj(os.path.join(self.tracking_dir, video_name+'_obj.npy'), self.sample_n_frames, [height, width])
                points_obj = torch.tensor(points_obj, dtype=torch.int32)
                mask_obj = torch.from_numpy(np.load(os.path.join(self.ground_sam_dir, video_name+'.npy')))

                points_robot = process_traj(os.path.join(self.tracking_dir, video_name+'_robot.npy'), self.sample_n_frames, [height, width])
                points_robot = torch.tensor(points_robot, dtype=torch.int32)
                mask_robot = torch.from_numpy(np.load(os.path.join(os.path.dirname(os.path.dirname(self.ground_sam_dir)), 'robot', 'bridge_mask.npy')))

                break
            
            except Exception as e:
                
                with open(f'invalid_example.txt', 'a+') as f:
                    f.write(f'{video_path} {prompt_path}')
                    f.write('\n')

                data_idx = random.randint(0, self.length-1)
                video_name = self.valid_video_names[data_idx]
                video_path = os.path.join(os.path.dirname(self.train_data_dir), self.meta[video_name]['video_path'])
                prompt_path = os.path.join(os.path.dirname(self.train_data_dir), self.meta[video_name]['prompt_path'])
                transit_path = os.path.join(self.tracking_dir, video_name + '_transit.npy')

        return {
            "prompt": prompt, 
            "video": pixel_values,
            "points_obj": points_obj,
            "mask_obj": mask_obj,
            "points_robot": points_robot,
            "mask_robot": mask_robot,
            "transit_start": transit_start,
            "transit_end": transit_end,
        }

def main():

    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,
            torch_dtype=weight_dtype
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # Potentially load in the weights and states from a previous save
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_paths = [ckpt_path for ckpt_path in os.listdir(args.output_dir) if ckpt_path.startswith('checkpoint')]

    if len(ckpt_paths)==0:
        accelerator.print(f"Checkpoint not found in '{args.output_dir}'. Starting a new training run.")
        initial_global_step = 0
        global_step = 0

        finetune_init = True
        transformer3d = CogVideoXTransformer3DModel.from_pretrained_2d(
            args.transformer_path, 
            finetune_init=finetune_init,
        )
        
        if finetune_init == True:
            flow_in_dim = 128
            out_dim = args.dim

            def zero_module(module):
                """
                Zero out the parameters of a module and return it.
                """
                for p in module.parameters():
                    p.detach().zero_()
                return module
            
            class FloatGroupNorm(nn.GroupNorm):
                def forward(self, x):
                    return super().forward(x.to(self.bias.dtype)).type(x.dtype)

            for idx in range(len(transformer3d.transformer_blocks)):
                if idx%args.block_interval == 0:

                    transformer3d.transformer_blocks[idx].flow_spatial = nn.Conv2d(flow_in_dim, out_dim // 4, 3, padding=1)
                    transformer3d.transformer_blocks[idx].flow_temporal = nn.Conv1d(
                        out_dim // 4,
                        out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="replicate",
                    )
                    transformer3d.transformer_blocks[idx].flow_cond_norm = FloatGroupNorm(32, out_dim)
            
            transformer3d.save_pretrained(os.path.join(args.output_dir, 'checkpoint-0'))
            del transformer3d

    ckpt_paths = [ckpt_path for ckpt_path in os.listdir(args.output_dir) if ckpt_path.startswith('checkpoint')]
    global_step = max([int(ckpt_path.split('-')[1].split('.')[0]) for ckpt_path in ckpt_paths])
    initial_global_step = global_step
    ckpt_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')

    finetune_init = False
    transformer3d = CogVideoXTransformer3DModel.from_pretrained_2d(
        ckpt_path, 
        finetune_init=finetune_init,
    )
    accelerator.print(f"Resuming from checkpoint {ckpt_path}")

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)  
    transformer3d.requires_grad_(True)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):

            if accelerator.is_main_process:
                for model in models:
                    if isinstance(model, type(unwrap_model(transformer3d))):
                        model.save_pretrained(os.path.join(output_dir))
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            print('load model hook')

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    transformer3d.train()
    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    logging.info("Add network parameters")
    trainable_params_base = [param for name, param in transformer3d.named_parameters() if ('flow' not in name) and param.requires_grad]
    trainable_params_injector = [param for name, param in transformer3d.named_parameters() if ('flow' in name) and param.requires_grad]
    params_to_optimize = [
        {"params": trainable_params_injector, "lr": args.learning_rate},
        {"params": trainable_params_base, "lr": args.low_learning_rate}
    ]

    optimizer = get_optimizer(args, params_to_optimize)

    # Dataset and DataLoader
    train_dataset = VideoDataset(
        train_data_dir=args.train_data_dir,
        sample_n_frames=args.video_sample_n_frames,
        cache_dir=args.cache_dir,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        # collate_fn=collate_fn,
        # num_workers=args.dataloader_num_workers,
        num_workers=0
    )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    patch_size_t = accelerator.unwrap_model(transformer3d).config.patch_size_t

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    
    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # convert RLDS episode dataset to individual steps & reformat
    for epoch in range(0, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            texts = batch['prompt']
            pixel_values = batch['video']                   # [B, F, 3, H, W]
            points_obj = batch['points_obj']    
            mask_obj = batch['mask_obj']    
            points_robot = batch['points_robot']    
            mask_robot = batch['mask_robot']   
            transit_start = batch['transit_start'].item()
            transit_end = batch['transit_end'].item()
            bsz, f, c, h, w = pixel_values.size()
            assert f == args.video_sample_n_frames

            vae_scale_factor_spatial = (
                2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
            )
            points_robot = (points_robot / vae_scale_factor_spatial).int()
            points_obj = (points_obj / vae_scale_factor_spatial).int()

            if args.train_mode != "normal":
                mask = torch.ones((bsz,f,1,h,w)).to(pixel_values)
                mask[:,0] = 0.
                mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask

            prompt_ids = tokenizer(
                texts, 
                max_length=args.tokenizer_max_length, 
                padding="max_length", 
                add_special_tokens=True, 
                truncation=True, 
                return_tensors="pt"
            )
            encoder_hidden_states = text_encoder(
                prompt_ids.input_ids.cuda(),
                return_dict=False
            )[0]
        
            encoder_attention_mask = prompt_ids.attention_mask
            encoder_hidden_states = encoder_hidden_states

            # Put tensors to device
            pixel_values = pixel_values.to(accelerator.device)
            mask = mask.to(accelerator.device)
            mask_pixel_values = mask_pixel_values.to(accelerator.device)
            encoder_attention_mask = encoder_attention_mask.to(accelerator.device)
            encoder_hidden_states = encoder_hidden_states.to(accelerator.device)
            
            # Data batch sanity check
            if global_step == 0:
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values.cpu(), texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                if args.train_mode != "normal":
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(mask_pixel_values.cpu(), texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.gif", rescale=True)
                pixel_values = rearrange(pixel_values, "b c f h w -> b f c h w")
                mask_pixel_values = rearrange(mask_pixel_values, "b c f h w -> b f c h w")

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = pixel_values.to(weight_dtype)
                mask = mask.to(weight_dtype)
                mask_pixel_values = mask_pixel_values.to(weight_dtype)

                torch.cuda.empty_cache()

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    
                    latents = _batch_encode_vae(pixel_values)
                    latents = latents * vae.config.scaling_factor

                    latents_obj = _batch_encode_vae(pixel_values[:,0].unsqueeze(1))
                    latents_obj = latents_obj * vae.config.scaling_factor

                    diameter_obj = max(int(torch.sqrt(mask_obj.sum()) / vae_scale_factor_spatial), 2)
                    mask_obj = F.interpolate(
                        mask_obj[None, None].float(),
                        size=latents_obj.shape[2:],
                        mode='trilinear',
                        align_corners=False
                    )

                    latents_robot = torch.load('/root/RoboMaster/robot/bridge.pth')
                    diameter_robot = max(int(torch.sqrt(mask_robot.sum()) / 2 / vae_scale_factor_spatial), 2)
                    latents_robot = latents_robot.to(device=latents_obj.device, dtype=weight_dtype)
                    mask_robot = F.interpolate(
                        mask_robot[None, None].float(),
                        size=latents_robot.shape[2:],
                        mode='trilinear',
                        align_corners=False
                    )

                    transit_start_latent = transit_start // vae.config.temporal_compression_ratio
                    transit_end_latent = transit_end // vae.config.temporal_compression_ratio

                    # pre-interaction
                    flow_latents = sample_flowlatents(
                        latents_robot, 
                        torch.zeros_like(latents),
                        mask_robot,
                        points_robot,
                        diameter_robot,
                        0,
                        transit_start_latent,
                    )

                    # interaction
                    flow_latents = sample_flowlatents(
                        latents_obj, 
                        flow_latents,
                        mask_obj,
                        points_obj,
                        diameter_obj,
                        transit_start_latent,
                        transit_end_latent,
                    )
                    
                    # post-interaction
                    flow_latents = sample_flowlatents(
                        latents_robot, 
                        flow_latents,
                        mask_robot,
                        points_robot,
                        diameter_robot,
                        transit_end_latent,
                        latents.shape[2],
                    )

                    if args.train_mode != "normal":
                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = 1 - mask
                        mask = resize_mask(mask, latents)

                        if unwrap_model(transformer3d).config.add_noise_in_inpaint_model:
                            mask_pixel_values = add_noise_to_reference_video(mask_pixel_values)
                        # Encode inpaint latents.
                        mask_latents = _batch_encode_vae(mask_pixel_values.to(weight_dtype))

                        # random drop inpaint latents
                        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                        mask_drop_latents = torch.rand([bsz]) < args.condition_drop_rate
                        mask_drop_latents = (1 - mask_drop_latents.float()).to(inpaint_latents)
                        inpaint_latents = mask_drop_latents * inpaint_latents * vae.config.scaling_factor
                        inpaint_latents = rearrange(inpaint_latents, "b c f h w -> b f c h w").to(weight_dtype)

                    latents = rearrange(latents, "b c f h w -> b f c h w")
                    flow_latents = rearrange(flow_latents, "b c f h w -> b f c h w")

                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                # Sample a random timestep for each image
                # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                timesteps = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                timesteps = timesteps.long()

                def _prepare_rotary_positional_embeddings(
                    height: int,
                    width: int,
                    num_frames: int,
                    device: torch.device,
                    dtype: torch.bfloat16
                ):
                    vae_scale_factor_spatial = (
                        2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
                    )

                    p = unwrap_model(transformer3d).config.patch_size
                    p_t = unwrap_model(transformer3d).config.patch_size_t

                    grid_height = height // (vae_scale_factor_spatial * p)
                    grid_width = width // (vae_scale_factor_spatial * p)
                    base_size_height = unwrap_model(transformer3d).config.sample_height // p
                    base_size_width = unwrap_model(transformer3d).config.sample_width // p

                    if p_t is None:
                        # CogVideoX 1.0
                        grid_crops_coords = get_resize_crop_region_for_grid(
                            (grid_height, grid_width), base_size_width, base_size_height
                        )
                        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                            embed_dim=unwrap_model(transformer3d).config.attention_head_dim,
                            crops_coords=grid_crops_coords,
                            grid_size=(grid_height, grid_width),
                            temporal_size=num_frames,
                            use_real=True,
                        )
                    else:
                        # CogVideoX 1.5
                        base_num_frames = (num_frames + p_t - 1) // p_t
                        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                            embed_dim=unwrap_model(transformer3d).config.attention_head_dim,
                            crops_coords=None,
                            grid_size=(grid_height, grid_width),
                            temporal_size=base_num_frames,
                            grid_type="slice",
                            max_size=(base_size_height, base_size_width),
                        )
                    freqs_cos = freqs_cos.to(device=device, dtype=dtype)
                    freqs_sin = freqs_sin.to(device=device, dtype=dtype)
                    return freqs_cos, freqs_sin

                # 7. Create rotary embeds if required
                image_rotary_emb = (
                    _prepare_rotary_positional_embeddings(h, w, latents.size(1), latents.device, weight_dtype)
                    if unwrap_model(transformer3d).config.use_rotary_positional_embeddings
                    else None
                )
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # random drop texts
                if False:
                    mask_drop_texts = torch.rand([bsz]) < args.condition_drop_rate
                    mask_drop_texts = (1 - mask_drop_texts.float()).to(encoder_hidden_states)
                    encoder_hidden_states = mask_drop_texts * encoder_hidden_states

                # predict the noise residual
                noise_pred = transformer3d(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents if args.train_mode != "normal" else None,
                    flow_latents=flow_latents,
                )[0]

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                if args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, 1:, :].float() - noise_pred[:, :-1, :].float()
                    pre_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params_base, args.max_grad_norm)
                    accelerator.clip_grad_norm_(trainable_params_injector, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            

    accelerator.end_training()


if __name__ == "__main__":
    main()