#!/bin/bash
#
# Script for training a CLIP-like model with hard negative images using DeepSpeed.
#
# This script is designed for easy configuration and sharing. 
# You can set the following environment variables before running, or replace the placeholders below with your actual paths.
#
# Environment variables (set with `export VAR=...`):
#   MODEL_NAME      - HuggingFace model name or local path (e.g., google/siglip-so400m-patch14-384)
#   OUTPUT_DIR      - Directory to save model checkpoints and logs
#   TRAIN_FILE      - Path to your training data JSON file
#   IMAGE_FOLDER    - Path to your image folder
#   CACHE_DIR       - (Optional) Directory for model cache
#
# Example usage:
#   export MODEL_NAME=google/siglip-so400m-patch14-384
#   export OUTPUT_DIR=/path/to/output
#   export TRAIN_FILE=/path/to/train.json
#   export IMAGE_FOLDER=/path/to/images
#   export CACHE_DIR=/path/to/cache
#   bash run_negative_images.sh
#
# Alternatively, you can directly edit the variables below.

MODEL_NAME=${MODEL_NAME:-your_model_name_or_path}
OUTPUT_DIR=${OUTPUT_DIR:-your_output_directory}
TRAIN_FILE=${TRAIN_FILE:-your_training_data_json}
IMAGE_FOLDER=${IMAGE_FOLDER:-your_image_folder}
CACHE_DIR=${CACHE_DIR:-your_cache_directory}

# Launch training with DeepSpeed
deepspeed --include localhost:1 \
    --master_port 29504 \
    run_clip_multi_image.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --train_file "$TRAIN_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --per_device_train_batch_size 5 \
    --num_train_epochs 1 \
    --topk_negative 9 \
    --save_steps 3000 \
    --bf16 True \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --save_total_limit 1 \
    --logging_steps 2 \
    --learning_rate 2e-5 \
    --cache_dir "$CACHE_DIR"
