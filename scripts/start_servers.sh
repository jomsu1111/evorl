#!/bin/bash
# GPU0: policy server (Qwen3-0.6B + runtime LoRA), alone on the card -> large KV cache.
# GPU1: helper server (Qwen3-4B-Instruct-2507) + the RFT trainer subprocesses
#       (helper 0.60 x 24GB = 14.4GB, trainer peak ~5.1GB).
# Before 2026-11-15 15:40 KST both servers shared GPU0 (policy 0.28, helper 0.60); GPU0 was the
# bottleneck (~94% util) while GPU1 idled between trainings.
set -euo pipefail
VENV=/opt/cove-venv
LOG=/opt/cove-logs
POLICY_GPU=${POLICY_GPU:-0}
HELPER_GPU=${HELPER_GPU:-1}
POLICY_UTIL=${POLICY_UTIL:-0.85}
HELPER_UTIL=${HELPER_UTIL:-0.60}
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
  CUDA_VISIBLE_DEVICES=$POLICY_GPU nohup "$VENV/bin/vllm" serve /root/cove-rl/models/Qwen3-0.6B \
    --served-model-name policy --port 8001 --seed 0 \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-cpu-loras 32 \
    --gpu-memory-utilization "$POLICY_UTIL" --max-model-len 6144 --max-num-seqs 128 \
    > "$LOG/policy.log" 2>&1 < /dev/null &
  wait_up 8001 policy
fi

if ! curl -s http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=$HELPER_GPU nohup "$VENV/bin/vllm" serve /opt/cove-models/Qwen3-4B-Instruct-2507 \
    --served-model-name helper --port 8002 --seed 0 \
    --gpu-memory-utilization "$HELPER_UTIL" --max-model-len 8192 --max-num-seqs 64 \
    > "$LOG/helper.log" 2>&1 < /dev/null &
  wait_up 8002 helper
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
