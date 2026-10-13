#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_PREFIX="${WEBSHOP_AGENT_ENV_PREFIX:-/root/autodl-tmp/conda_envs/webshop-agent}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
WEBSHOP_ROOT="${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop"

source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import importlib
import torch
for name in ("transformers", "vllm", "ray", "peft", "spacy", "flash_attn", "tensordict", "faiss"):
    importlib.import_module(name)
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
test -f "${WEBSHOP_ROOT}/search_engine/indexes/segments_1"
python scripts/evaluation/test_webshop_small_index.py
echo "Environment verification passed."

