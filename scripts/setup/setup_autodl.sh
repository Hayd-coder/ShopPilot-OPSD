#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_PREFIX="${WEBSHOP_AGENT_ENV_PREFIX:-/root/autodl-tmp/conda_envs/webshop-agent}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
WEBSHOP_ROOT="${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop"
BUNDLED_DATA="${REPO_ROOT}/datasets/webshop-small"

for file in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do
    test -s "${BUNDLED_DATA}/${file}" || { echo "Missing ${BUNDLED_DATA}/${file}" >&2; exit 1; }
done
test -x "${CONDA_EXE}" || { echo "Conda was not found: ${CONDA_EXE}" >&2; exit 1; }

source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
mkdir -p "$(dirname -- "${ENV_PREFIX}")" /root/autodl-tmp/{huggingface,pip-cache,conda_pkgs,tmp}
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    conda create -y -p "${ENV_PREFIX}" python=3.10 pip
fi
conda activate "${ENV_PREFIX}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda_pkgs}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"

python -m pip install -U pip setuptools wheel packaging ninja uv
UV="${ENV_PREFIX}/bin/uv"
if ! command -v javac >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk-headless
fi

"${UV}" pip install --python "${ENV_PREFIX}/bin/python" --index-url "${PYPI_INDEX_URL}" torch==2.6.0
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" --index-url "${PYPI_INDEX_URL}" -r "${WEBSHOP_ROOT}/requirements.txt"
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" --index-url "${PYPI_INDEX_URL}" faiss-cpu==1.9.0.post1
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" --index-url "${PYPI_INDEX_URL}" \
    vllm==0.8.5.post1 ray==2.50.0 transformers==4.51.1 \
    codetiming==1.4.0 datasets==3.2.0 hydra-core==1.3.2 pyarrow==19.0.1 \
    pylatexenc==2.10 torchdata==0.11.0 tensordict==0.8.3 'wandb<0.20' \
    peft accelerate 'qwen-vl-utils<0.1'
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" --no-deps -e "${REPO_ROOT}"

if ! python -c 'import flash_attn' >/dev/null 2>&1; then
    python -m pip install --resume-retries 30 --timeout 120 flash-attn==2.7.4.post1 --no-build-isolation
fi
if ! python -c 'import en_core_web_sm' >/dev/null 2>&1; then
    python -m spacy download en_core_web_sm
fi
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" --index-url "${PYPI_INDEX_URL}" typer==0.27.3 websockets==11.0.3

mkdir -p "${WEBSHOP_ROOT}/data"
cp -f "${BUNDLED_DATA}"/*.json "${WEBSHOP_ROOT}/data/"
cd "${WEBSHOP_ROOT}/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py
if [[ ! -f indexes/segments_1 ]]; then
    bash run_indexing.sh
fi

cd "${REPO_ROOT}"
bash scripts/setup/verify_environment.sh

