#!/usr/bin/env bash

# Download the full WebShop raw JSON files into an isolated staging folder.
# This script deliberately does not build the Lucene index while RL is running.

set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/root/autodl-tmp/webshop-full-raw}"
BASE_URL="${BASE_URL:-https://hf-mirror.com/datasets/YWZBrandon/webshop-data/resolve/main}"
MIN_FREE_GIB="${MIN_FREE_GIB:-30}"
if [[ ! "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || (( MIN_FREE_GIB < 1 )); then
    echo "MIN_FREE_GIB must be a positive integer, got: ${MIN_FREE_GIB}" >&2
    exit 2
fi
MIN_FREE_BYTES=$((MIN_FREE_GIB * 1024 * 1024 * 1024))

mkdir -p "${TARGET_DIR}"
available_bytes="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d ' ')"
if (( available_bytes < MIN_FREE_BYTES )); then
    echo "Need at least ${MIN_FREE_GIB} GiB free before downloading full WebShop data."
    echo "Current free bytes: ${available_bytes}"
    echo "Expand the AutoDL data disk first; no files were downloaded."
    exit 1
fi

download() {
    local filename="$1"
    echo "[$(date '+%F %T')] Downloading ${filename}"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c \
            --continue=true \
            --max-connection-per-server=8 \
            --split=8 \
            --min-split-size=1M \
            --file-allocation=none \
            --max-tries=20 \
            --retry-wait=5 \
            --connect-timeout=30 \
            --timeout=30 \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --dir="${TARGET_DIR}" \
            --out="${filename}" \
            "${BASE_URL}/${filename}?download=true"
    else
        wget -c --tries=20 --timeout=30 \
            -O "${TARGET_DIR}/${filename}" \
            "${BASE_URL}/${filename}?download=true"
    fi
}

download items_shuffle.json
download items_ins_v2.json
download items_human_ins.json

echo "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8  ${TARGET_DIR}/items_shuffle.json" | sha256sum -c -
echo "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e  ${TARGET_DIR}/items_ins_v2.json" | sha256sum -c -

du -sh "${TARGET_DIR}"
date '+%F %T' > "${TARGET_DIR}/DOWNLOAD_COMPLETE"
echo "Full WebShop raw-data download complete. Indexing has not been started."
