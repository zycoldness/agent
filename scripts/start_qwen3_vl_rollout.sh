#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
LORA_RANK=${LORA_RANK:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CUDA_VISIBLE_DEVICES

swift rollout \
  --model "$MODEL" \
  --vllm_tensor_parallel_size 4 \
  --vllm_data_parallel_size 1 \
  --vllm_enable_lora true \
  --vllm_max_lora_rank "$LORA_RANK" \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 4096 \
  --port 8000
