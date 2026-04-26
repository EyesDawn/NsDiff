#!/bin/bash

# Original manifest-driven script retained below for reference.
# #!/bin/bash
# set -euo pipefail
#
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# cd "${REPO_ROOT}"
#
# export PYTHONPATH=./
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
#
# MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/source_target_manifest.json}"
# OUTPUT_DIR="${OUTPUT_DIR:-./results/analysis/source_target_w1_bars}"
# OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/source_target_w1_bars_scaled_all.pdf}"
# METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/source_target_w1_bars_scaled_all.json}"
# SEED="${SEED:-42}"
# DEVICE="${DEVICE:-cuda:0}"
# MAX_POINTS="${MAX_POINTS:-50000}"
# NUM_REPEATS="${NUM_REPEATS:-3}"
# EPS="${EPS:-1e-6}"
# ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-64}"
# ANALYSIS_NUM_WORKER="${ANALYSIS_NUM_WORKER:-4}"
# ANALYSIS_NUM_SAMPLES="${ANALYSIS_NUM_SAMPLES:-16}"
# PDN_FILTER_METHOD="${PDN_FILTER_METHOD:-none}"
# PDN_FILTER_VALUE="${PDN_FILTER_VALUE:-0.99}"
# PDN_SIGMA_FILTER_METHOD="${PDN_SIGMA_FILTER_METHOD:-quantile_band}"
# PDN_SIGMA_FILTER_LOWER="${PDN_SIGMA_FILTER_LOWER:-0.05}"
# PDN_SIGMA_FILTER_UPPER="${PDN_SIGMA_FILTER_UPPER:-0.95}"
#
# if [ ! -f "${MANIFEST_PATH}" ]; then
#     echo "Missing manifest: ${MANIFEST_PATH}"
#     exit 1
# fi
#
# mkdir -p "${OUTPUT_DIR}"
#
# CMD=(
#     python -u ./src/analysis/source_target_w1.py
#     --manifest_path "${MANIFEST_PATH}"
#     --output_path "${OUTPUT_PATH}"
#     --metadata_path "${METADATA_PATH}"
#     --seed "${SEED}"
#     --device "${DEVICE}"
#     --max_points "${MAX_POINTS}"
#     --num_repeats "${NUM_REPEATS}"
#     --eps "${EPS}"
# )
#
# if [ -n "${ANALYSIS_BATCH_SIZE:-}" ]; then
#     CMD+=(--analysis_batch_size "${ANALYSIS_BATCH_SIZE}")
# fi
#
# if [ -n "${ANALYSIS_NUM_WORKER:-}" ]; then
#     CMD+=(--analysis_num_worker "${ANALYSIS_NUM_WORKER}")
# fi
#
# if [ -n "${ANALYSIS_NUM_SAMPLES:-}" ]; then
#     CMD+=(--analysis_num_samples "${ANALYSIS_NUM_SAMPLES}")
# fi
#
# if [ -n "${PDN_FILTER_METHOD:-}" ]; then
#     CMD+=(--pdn_filter_method "${PDN_FILTER_METHOD}")
# fi
#
# if [ -n "${PDN_FILTER_VALUE:-}" ]; then
#     CMD+=(--pdn_filter_value "${PDN_FILTER_VALUE}")
# fi
#
# if [ -n "${PDN_SIGMA_FILTER_METHOD:-}" ]; then
#     CMD+=(--pdn_sigma_filter_method "${PDN_SIGMA_FILTER_METHOD}")
# fi
#
# if [ -n "${PDN_SIGMA_FILTER_LOWER:-}" ]; then
#     CMD+=(--pdn_sigma_filter_lower "${PDN_SIGMA_FILTER_LOWER}")
# fi
#
# if [ -n "${PDN_SIGMA_FILTER_UPPER:-}" ]; then
#     CMD+=(--pdn_sigma_filter_upper "${PDN_SIGMA_FILTER_UPPER}")
# fi
#
# "${CMD[@]}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=./

INPUT_JSON="${INPUT_JSON:-./results/analysis/source_target_w1_bars/source_target_w1_bars_scaled_all.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/analysis/source_target_w1_bars}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/source_target_w1_bars_scaled_all_redraw.pdf}"

if [ ! -f "${INPUT_JSON}" ]; then
    echo "Missing result json: ${INPUT_JSON}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python - <<PY
from src.analysis.source_target_w1 import redraw_source_target_wasserstein_bars_from_metadata

result = redraw_source_target_wasserstein_bars_from_metadata(
    metadata_path="${INPUT_JSON}",
    output_path="${OUTPUT_PATH}",
)
print(result)
PY
