#!/bin/bash

# ===========================
# Configurable variables
# ===========================
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct-1M}
DATA_PATH=${DATA_PATH:-/path/to/your/data.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/path/to/your/image_folder}
PRETRAIN_MM_MLP_ADAPTER=${PRETRAIN_MM_MLP_ADAPTER:-/path/to/pretrained/mm_projector.bin}
VISION_TOWER=${VISION_TOWER:-/path/to/vision_tower}
OUTPUT_DIR=${OUTPUT_DIR:-/path/to/output_dir}
CACHE_DIR=${CACHE_DIR:-/path/to/cache_dir}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-29502}
INCLUDE_DEVICES=${INCLUDE_DEVICES:-localhost:0,1,2,3}

# ===========================
# Training command
# ===========================
deepspeed --include ${INCLUDE_DEVICES} \
    --master_port ${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version qwen_1_5 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --pretrain_mm_mlp_adapter "${PRETRAIN_MM_MLP_ADAPTER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 7000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --cache_dir "${CACHE_DIR}"
