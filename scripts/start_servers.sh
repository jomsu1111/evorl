#!/bin/bash
# GPU0: policy server (Qwen3-0.6B + runtime LoRA) and helper server (Qwen3-4B-Instruct-2507).
# GPU1 is left for the RFT trainer subprocesses.
set -euo pipefail
VENV=/opt/cove-venv
LOG=/opt/cove-logs
mkdir -p "$LOG"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export HF_HUB_OFFLINE=1

wait_up() {  # port name
  for i in $(seq 1 180); do
    if curl -s "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; then echo "$2 up"; return 0; fi
    sleep 5
  done
  echo "$2 failed to start; see $LOG/$2.log" >&2; return 1
}

if ! curl -s http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=0 nohup "$VENV/bin/vllm" serve /root/cove-rl/models/Qwen3-0.6B \
    --served-model-name policy --port 8001 --seed 0 \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-cpu-loras 32 \
    --gpu-memory-utilization 0.28 --max-model-len 6144 --max-num-seqs 128 \
    > "$LOG/policy.log" 2>&1 < /dev/null &
  wait_up 8001 policy
fi

if ! curl -s http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=0 nohup "$VENV/bin/vllm" serve /opt/cove-models/Qwen3-4B-Instruct-2507 \
    --served-model-name helper --port 8002 --seed 0 \
    --gpu-memory-utilization 0.60 --max-model-len 8192 --max-num-seqs 64 \
    > "$LOG/helper.log" 2>&1 < /dev/null &
  wait_up 8002 helper
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
