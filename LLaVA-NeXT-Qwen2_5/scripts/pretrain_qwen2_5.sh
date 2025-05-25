#!/bin/bash

# ===========================
# Configurable variables
# ===========================
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct-1M}
DATA_PATH=${DATA_PATH:-/path/to/alignment_with_image_path.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/path/to/image_folder}
VISION_TOWER=${VISION_TOWER:-/path/to/vision_tower}
OUTPUT_DIR=${OUTPUT_DIR:-/path/to/output_dir}
CACHE_DIR=${CACHE_DIR:-/path/to/cache_dir}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-29506}
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
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --cache_dir "${CACHE_DIR}"