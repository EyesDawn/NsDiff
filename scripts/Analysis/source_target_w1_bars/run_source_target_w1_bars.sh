#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/source_target_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/analysis/source_target_w1_bars}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/source_target_w1_bars_scaled_all.pdf}"
METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/source_target_w1_bars_scaled_all.json}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
MAX_POINTS="${MAX_POINTS:-50000}"
NUM_REPEATS="${NUM_REPEATS:-3}"
EPS="${EPS:-1e-6}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-64}"
ANALYSIS_NUM_WORKER="${ANALYSIS_NUM_WORKER:-4}"
ANALYSIS_NUM_SAMPLES="${ANALYSIS_NUM_SAMPLES:-16}"
PDN_FILTER_METHOD="${PDN_FILTER_METHOD:-none}"
PDN_FILTER_VALUE="${PDN_FILTER_VALUE:-0.99}"

if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "Missing manifest: ${MANIFEST_PATH}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python -u ./src/analysis/source_target_w1.py
    --manifest_path "${MANIFEST_PATH}"
    --output_path "${OUTPUT_PATH}"
    --metadata_path "${METADATA_PATH}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --max_points "${MAX_POINTS}"
    --num_repeats "${NUM_REPEATS}"
    --eps "${EPS}"
)

if [ -n "${ANALYSIS_BATCH_SIZE:-}" ]; then
    CMD+=(--analysis_batch_size "${ANALYSIS_BATCH_SIZE}")
fi

if [ -n "${ANALYSIS_NUM_WORKER:-}" ]; then
    CMD+=(--analysis_num_worker "${ANALYSIS_NUM_WORKER}")
fi

if [ -n "${ANALYSIS_NUM_SAMPLES:-}" ]; then
    CMD+=(--analysis_num_samples "${ANALYSIS_NUM_SAMPLES}")
fi

if [ -n "${PDN_FILTER_METHOD:-}" ]; then
    CMD+=(--pdn_filter_method "${PDN_FILTER_METHOD}")
fi

if [ -n "${PDN_FILTER_VALUE:-}" ]; then
    CMD+=(--pdn_filter_value "${PDN_FILTER_VALUE}")
fi

"${CMD[@]}"
