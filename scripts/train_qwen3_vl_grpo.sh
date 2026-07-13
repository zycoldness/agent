#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
TRAIN_DATA=${TRAIN_DATA:-outputs/grpo/train.jsonl}
VAL_DATA=${VAL_DATA-outputs/grpo/dev.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3_vl_8b_grpo}
: "${SFT_ADAPTER:?set SFT_ADAPTER to the SFT LoRA checkpoint}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
VAL_ARGS=()
if [[ -n "$VAL_DATA" ]]; then
  VAL_ARGS=(--val_dataset "$VAL_DATA")
fi

swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --adapters "$SFT_ADAPTER" \
  --ref_adapters "$SFT_ADAPTER" \
  --dataset "$TRAIN_DATA" \
  "${VAL_ARGS[@]}" \
  --external_plugins plugins/ms_swift_risk_rewards.py \
  --reward_funcs risk_format_v1 risk_label_exact_v1 risk_rule_exact_v1 risk_evidence_exact_v1 \
  --reward_weights 0.05 0.60 0.20 0.15 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --vllm_server_timeout 600 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --remove_unused_columns false \
  --num_generations 4 \
  --max_length 2048 \
  --max_completion_length 256 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --deepspeed zero2 \
  --logging_steps 1 \
  --save_steps 50 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --warmup_ratio 0.03 \
  --output_dir "$OUTPUT_DIR" \
  --report_to none
