export MODEL_NAME="/root/RoboMaster/ckpts/CogVideoX-Fun-V1.5-5b-InP"
export TRANSFORMER_NAME="/root/RoboMaster/ckpts/CogVideoX-Fun-V1.5-5b-InP/transformer"
export DATASET_NAME="root/processed_dataset"
export SAVE_NAME="output_dir/openx_bridge_injector"
export TRACKER_NAME="bridge_injector"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

NCCL_DEBUG=INFO

accelerate launch --config_file scripts/accelerate_config_machine_single.yaml --main_process_port 29500 --multi_gpu \
    scripts/train_injector.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_NAME \
    --transformer_path=$TRANSFORMER_NAME \
    --resume_from_checkpoint=$SAVE_NAME \
    --video_sample_n_frames 37 \
    --train_batch_size 1 \
    --video_repeat 1 \
    --vae_mini_batch 1 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 1000 \
    --max_train_steps 100000 \
    --validation_steps 500 \
    --seed 42 \
    --output_dir $SAVE_NAME \
    --mixed_precision "bf16" \
    --train_mode "inpaint" \
    --tracker_project_name $TRACKER_NAME \
    --optimizer AdamW \
    --lr_scheduler cosine_with_restarts \
    --learning_rate 1e-4 \
    --low_learning_rate 2e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 0.05 \
    --block_interval 1 \
    --gradient_checkpointing \
    --allow_tf32