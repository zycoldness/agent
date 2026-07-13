#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
TRAIN_DATA=${TRAIN_DATA:-outputs/sft/train.jsonl}
VAL_DATA=${VAL_DATA-outputs/sft/dev.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3_vl_8b_sft}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
VAL_ARGS=()
if [[ -n "$VAL_DATA" ]]; then
  VAL_ARGS=(--val_dataset "$VAL_DATA")
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN_DATA" \
  "${VAL_ARGS[@]}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --padding_free true \
  --packing true \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --num_train_epochs 1 \
  --max_length 4096 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --warmup_ratio 0.05 \
  --deepspeed zero2 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --output_dir "$OUTPUT_DIR" \
  --report_to none
