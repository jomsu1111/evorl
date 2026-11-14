#!/bin/bash
# Recreate the (ephemeral) /opt environment after a container restart:
#   venv -> /opt/cove-venv, helper model -> /opt/cove-models. Persistent data stays in /root/cove-rl.
set -euo pipefail
export PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/opt/uvcache UV_LINK_MODE=copy UV_HTTP_TIMEOUT=900
uv venv /opt/cove-venv --python 3.11
uv pip install --python /opt/cove-venv/bin/python "vllm==0.10.1.1" "torch==2.7.1" "transformers==4.56.1" \
  "ms_swift==3.8.2" "peft==0.17.1" "trl==0.17.0" "deepspeed==0.17.5" "accelerate==1.10.1" "datasets==3.6.0" \
  "sentence-transformers" "python-dotenv" "json5" "openai" "pandas" "math-verify[antlr4_13_2]" "scikit-learn" "tqdm"
/opt/cove-venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='/opt/cove-models/Qwen3-4B-Instruct-2507',
                  allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab*','tokenizer*'])"
echo SETUP_DONE
