import json
import os
import tqdm
import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
    PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
import cv2
import decord

from robomaster.models.transformer3d import CogVideoXTransformer3DModel
from robomaster.models.autoencoder_magvit import AutoencoderKLCogVideoX
from robomaster.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from robomaster.utils.utils import get_image_to_video_latent, save_videos_grid
from utils import *

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# Model path
model_name              = "ckpts/CogVideoX-Fun-V1.5-5b-InP"
transformer_path        = "ckpts/RoboMaster"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
sampler_name            = "DDIM_Origin"

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length    = None
overlap_video_length    = 4

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16

# Configs
negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
video_length            = 37
fps                     = 12
validation_image_path   = "eval_metrics/results/bridge_eval_gt"
save_path               = "samples/bridge_eval_ours"

# Get Transformer
transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    transformer_path,
    low_cpu_mem_usage=True,
    finetune_init=False,
).to(weight_dtype)

# Get Vae
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)

text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
    model_name,
    vae=vae,
    text_encoder=text_encoder,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=weight_dtype
)

if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_images = [validation_image for validation_image in sorted(os.listdir(validation_image_path)) if validation_image.endswith('.png')]
vae_scale_factor_spatial = (2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8)
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
generator = torch.Generator(device="cuda").manual_seed(seed)

for validation_image in tqdm.tqdm(validation_images):

    if os.path.exists(os.path.join(save_path, validation_image.replace('.png','.mp4'))):
        continue

    validation_image_start  = os.path.join(validation_image_path, validation_image)
    validation_image_end    = None
    image                   = Image.open(validation_image_start).convert("RGB")
    sample_size_ori         = (image.size[1], image.size[0])
    sample_size             = (round(image.size[1]/8)*8, round(image.size[0]/8)*8)
    image                   = image.resize(sample_size)
    prompt_path             = validation_image_start.replace('.png', '.txt')
    with open(prompt_path, 'r') as file: prompt = file.readline().strip()
    obj_tracking_path     = os.path.join(validation_image_path, validation_image.replace('.png', '_obj.npy'))
    robot_tracking_path     = os.path.join(validation_image_path, validation_image.replace('.png', '_robot.npy'))
    
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    if video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
        additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
        video_length += additional_frames * vae.config.temporal_compression_ratio
    input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

    points_obj = process_traj(obj_tracking_path, video_length, [sample_size_ori[0], sample_size_ori[1]])
    points_obj = torch.tensor(points_obj)
    points_obj = (points_obj / vae_scale_factor_spatial).int()

    points_robot = process_traj(robot_tracking_path, video_length, [sample_size_ori[0], sample_size_ori[1]])
    points_robot = torch.tensor(points_robot)
    points_robot = (points_robot / vae_scale_factor_spatial).int()

    mask_obj = torch.from_numpy(np.load(os.path.join(validation_image_path, validation_image.replace('.png', '_obj_mask.npy'))))
    diameter_obj = max(int(torch.sqrt(mask_obj.sum()) / vae_scale_factor_spatial), 2)

    with torch.no_grad():        
        
        latents_obj = vae.encode((input_video[:,:,0].unsqueeze(2)*2-1).to(dtype=weight_dtype, device='cuda'))[0]
        latents_obj = latents_obj.sample()
        latents_obj = latents_obj * vae.config.scaling_factor

        mask_obj = F.interpolate(
            mask_obj[None, None, None].float(),
            size=latents_obj.shape[2:],
            mode='trilinear',
            align_corners=False
        )

        ground_sam_robot_path = './robot'
        latents_robot = torch.load(os.path.join(ground_sam_robot_path, 'bridge.pth'))
        mask_robot = torch.from_numpy(np.load(os.path.join(ground_sam_robot_path, 'bridge_mask.npy')))
        diameter_robot = max(int(torch.sqrt(mask_robot.sum()) / 2 / vae_scale_factor_spatial), 2)
        latents_robot = latents_robot.to(device=latents_obj.device, dtype=weight_dtype)
        mask_robot = F.interpolate(
            mask_robot[None, None, None].float(),
            size=latents_robot.shape[2:],
            mode='trilinear',
            align_corners=False
        )

        transit_start, transit_end = np.load(os.path.join(validation_image_path, validation_image.replace('.png', '_transit.npy')))
        video_path = os.path.join(validation_image_path, validation_image.replace('.png', '.mp4'))
        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ctx = decord.cpu(0)
        reader = decord.VideoReader(video_path, ctx=ctx, height=height, width=width)
        transit_start = int(transit_start * video_length / len(reader))
        transit_end = int(transit_end * video_length / len(reader))
        transit_start_latent = transit_start // vae.config.temporal_compression_ratio
        transit_end_latent = transit_end // vae.config.temporal_compression_ratio
        if transit_end >= (video_length - 3):
            transit_end_latent = latent_frames

        # pre-interaction
        flow_latents = sample_flowlatents(
            latents_robot, 
            torch.zeros_like(latents_obj).repeat(1,1,latent_frames,1,1),
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
            latent_frames,
        )

        flow_latents = rearrange(flow_latents, "b c f h w -> b f c h w")

        sample = pipeline(
            prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            video        = input_video,
            mask_video   = input_video_mask,
            flow_latents = flow_latents,
        ).videos

    sample = F.interpolate(
        sample, 
        size=torch.Size([video_length, sample_size_ori[0], sample_size_ori[1]]), 
        mode='trilinear', 
        align_corners=False
    )

    # save files
    video_chunk = (rearrange(sample[0], "c f h w -> f h w c").numpy()*255).astype(np.uint8)
    save_video_name = os.path.join(save_path, os.path.basename(validation_image_start).split('.png')[0])
    save_images2video(video_chunk, save_video_name, fps=12) 
    os.system(f'cp -r {prompt_path} {save_path}')
