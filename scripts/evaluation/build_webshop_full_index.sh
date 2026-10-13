#!/usr/bin/env bash
set -euo pipefail

# Build the full WebShop Lucene index after any active RL trainer exits.
# The final index is activated only after conversion, indexing, and a search
# validation all succeed, so a failed build cannot corrupt the working setup.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ENV_PREFIX="${WEBSHOP_AGENT_ENV_PREFIX:-/root/autodl-tmp/conda_envs/webshop-agent}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
RAW_DIR="${RAW_DIR:-/root/autodl-tmp/webshop-full-raw}"
WEBSHOP_ROOT="${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop"
DATA_DIR="${WEBSHOP_ROOT}/data"
SEARCH_DIR="${WEBSHOP_ROOT}/search_engine"
RESOURCE_DIR="${SEARCH_DIR}/resources_full"
INDEX_DIR="${SEARCH_DIR}/indexes_full"
MARKER="${INDEX_DIR}/FULL_INDEX_COMPLETE"
THREADS="${INDEX_THREADS:-8}"

log() {
    echo "[$(date '+%F %T')] $*"
}

for file in items_shuffle.json items_ins_v2.json items_human_ins.json; do
    if [[ ! -s "${RAW_DIR}/${file}" ]]; then
        echo "Missing full WebShop file: ${RAW_DIR}/${file}" >&2
        exit 1
    fi
done

if [[ -f "${MARKER}" ]]; then
    log "Full WebShop index is already complete; nothing to do."
    exit 0
fi

if [[ "${WAIT_FOR_TRAINING:-1}" == "1" ]]; then
    log "Waiting for active veRL training processes to finish"
    while pgrep -f '[v]erl.trainer.main_' >/dev/null; do
        sleep 60
    done
else
    log "Concurrent mode enabled; building alongside the active trainer"
fi

source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
cd "${REPO_ROOT}"

log "Checking disk capacity"
free_kib=$(df -Pk "${SEARCH_DIR}" | awk 'NR==2 {print $4}')
if (( free_kib < 30 * 1024 * 1024 )); then
    echo "At least 30 GiB free is required to build the full index." >&2
    exit 1
fi

log "Linking the verified full raw files into the WebShop data directory"
mkdir -p "${DATA_DIR}" "${RESOURCE_DIR}"
ln -sfn "${RAW_DIR}/items_shuffle.json" "${DATA_DIR}/items_shuffle.json"
ln -sfn "${RAW_DIR}/items_ins_v2.json" "${DATA_DIR}/items_ins_v2.json"
ln -sfn "${RAW_DIR}/items_human_ins.json" "${DATA_DIR}/items_human_ins.json"

if ! python -c 'import ijson' >/dev/null 2>&1; then
    log "Installing the small streaming JSON dependency"
    python -m pip install --index-url http://mirrors.aliyun.com/pypi/simple \
        --trusted-host mirrors.aliyun.com ijson==3.4.0.post0
fi

DOCS="${RESOURCE_DIR}/documents.jsonl"
if [[ ! -s "${DOCS}" ]]; then
    log "Converting 1.18M products to Pyserini JSONL using bounded memory"
    python scripts/evaluation/stream_webshop_index_docs.py \
        --input "${RAW_DIR}/items_shuffle.json" \
        --output "${DOCS}"
else
    log "Converted document file already exists; reusing it"
fi

doc_count=$(wc -l < "${DOCS}")
if (( doc_count < 1_000_000 )); then
    echo "Converted document count is unexpectedly small: ${doc_count}" >&2
    exit 1
fi

if [[ -d "${INDEX_DIR}" && ! -f "${MARKER}" ]]; then
    partial="${INDEX_DIR}.partial.$(date +%Y%m%d_%H%M%S)"
    log "Moving an incomplete previous index to ${partial}"
    mv "${INDEX_DIR}" "${partial}"
fi
mkdir -p "${INDEX_DIR}"

log "Building the full Lucene index with ${THREADS} threads"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms2g -Xmx16g}"
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${RESOURCE_DIR}" \
    --index "${INDEX_DIR}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${THREADS}" \
    --storePositions --storeDocvectors --storeRaw

log "Validating index document count and a real search query"
INDEX_DIR="${INDEX_DIR}" EXPECTED_COUNT="${doc_count}" python - <<'PY'
import os
from pyserini.search.lucene import LuceneSearcher

index_dir = os.environ["INDEX_DIR"]
expected = int(os.environ["EXPECTED_COUNT"])
searcher = LuceneSearcher(index_dir)
actual = int(searcher.num_docs)
hits = searcher.search("black rubber sole loafers", k=5)
print(f"index_docs={actual:,} expected={expected:,} search_hits={len(hits)}")
empty_documents = expected - actual
if actual > expected or empty_documents > 100:
    raise SystemExit(
        f"Index count mismatch: indexed={actual}, input={expected}, "
        f"empty_or_skipped={empty_documents}"
    )
if not hits:
    raise SystemExit("Full index returned no results for the validation query")
print(f"empty_or_skipped_documents={empty_documents}")
PY

printf 'completed_at=%s\ndocuments=%s\n' "$(date -Iseconds)" "${doc_count}" > "${MARKER}"

log "Activating the new full index"
cd "${SEARCH_DIR}"
if [[ -L indexes ]]; then
    rm indexes
elif [[ -d indexes ]]; then
    backup="indexes_small_backup_$(date +%Y%m%d_%H%M%S)"
    mv indexes "${backup}"
    log "Previous small index retained as ${backup}"
fi
ln -s indexes_full indexes

log "Full WebShop index build and validation completed"
du -sh "${RESOURCE_DIR}" "${INDEX_DIR}"
