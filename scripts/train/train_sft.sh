#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_PREFIX="${WEBSHOP_AGENT_ENV_PREFIX:-/root/autodl-tmp/conda_envs/webshop-agent}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/webshop-agent-runs}"
GENERATED="${REPO_ROOT}/datasets/generated"

source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
cd "${REPO_ROOT}"
mkdir -p "${GENERATED}"

if [[ ! -s "${GENERATED}/self_expert_train4500.jsonl" ]]; then
    tar -xzf datasets/archives/self_expert_5000.tar.gz -C "${GENERATED}"
fi
if [[ ! -s "${GENERATED}/retrieval_curriculum2500.jsonl" ]]; then
    tar -xzf datasets/archives/retrieval_curriculum_2500.tar.gz -C "${GENERATED}"
fi

DIRECT_OUT="${RUN_ROOT}/sft-direct"
DIRECT_MODEL="${RUN_ROOT}/models/qwen25-3b-direct-sft"
python scripts/expert_sft/train_weighted_lora_sft.py \
    --model "${BASE_MODEL}" \
    --train-file "${GENERATED}/self_expert_train4500.jsonl" \
    --validation-file "${GENERATED}/self_expert_validation500.jsonl" \
    --output-dir "${DIRECT_OUT}" \
    --max-length 4096 --think-weight 0.2 --action-weight 1.0 \
    --lora-rank 32 --lora-alpha 64 --learning-rate 1e-4 \
    --epochs 3 --batch-size 1 --gradient-accumulation 8 \
    --max-eval-samples 128 --eval-strategy epoch --save-strategy epoch \
    --save-total-limit 3 --seed 42
python scripts/expert_sft/merge_lora_adapter.py \
    --base-model "${BASE_MODEL}" --adapter "${DIRECT_OUT}/final_adapter" --output "${DIRECT_MODEL}"

RETRIEVAL_OUT="${RUN_ROOT}/sft-retrieval"
FINAL_MODEL="${RUN_ROOT}/models/qwen25-3b-webshop-agent"
python scripts/expert_sft/train_weighted_lora_sft.py \
    --model "${DIRECT_MODEL}" \
    --train-file "${GENERATED}/retrieval_curriculum2500.jsonl" \
    --validation-file "${GENERATED}/self_expert_validation500.jsonl" \
    --output-dir "${RETRIEVAL_OUT}" \
    --max-length 4096 --think-weight 0.2 --action-weight 1.0 \
    --lora-rank 32 --lora-alpha 64 --learning-rate 5e-5 \
    --epochs 1 --batch-size 1 --gradient-accumulation 8 \
    --max-eval-samples 128 --eval-strategy epoch --save-strategy epoch \
    --save-total-limit 2 --seed 42
python scripts/expert_sft/merge_lora_adapter.py \
    --base-model "${DIRECT_MODEL}" --adapter "${RETRIEVAL_OUT}/final_adapter" --output "${FINAL_MODEL}"
echo "Final SFT model: ${FINAL_MODEL}"

