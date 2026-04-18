#!/bin/bash
set -euo pipefail

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET_NAME="${DATASET_NAME:-Electricity}"
WINDOWS="${WINDOWS:-96}"
HORIZON="${HORIZON:-1}"
PRED_LEN="${PRED_LEN:-192}"
RUN_SPEC="w${WINDOWS}h${HORIZON}s${PRED_LEN}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/analysis/Electricity/probabilistic_decoupling}"
SELECTION_PATH="${SELECTION_PATH:-${OUTPUT_DIR}/electricity_probabilistic_decoupling_selection.json}"
REF_RUN_DIR="${REF_RUN_DIR:-$(find "./results/runs/TimeGrad/${DATASET_NAME}/${RUN_SPEC}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)}"

NUM_VARIABLES="${NUM_VARIABLES:-10}"
VARIABLE_TAIL_QUANTILE="${VARIABLE_TAIL_QUANTILE:-0.9}"
HIGH_DRIFT_RATIO="${HIGH_DRIFT_RATIO:-0.2}"
LOW_DRIFT_QUANTILE="${LOW_DRIFT_QUANTILE:-0.3}"
HIGH_DRIFT_QUANTILE="${HIGH_DRIFT_QUANTILE:-0.7}"
PAIRS_PER_BIN="${PAIRS_PER_BIN:-256}"
RANDOM_SEED="${RANDOM_SEED:-2027}"
DEVICE="${DEVICE:-cpu}"
NUM_WORKER="${NUM_WORKER:-0}"

mkdir -p "${OUTPUT_DIR}"

if [ -z "${REF_RUN_DIR}" ] || [ ! -d "${REF_RUN_DIR}" ]; then
    echo "Missing reference run directory: ${REF_RUN_DIR}" >&2
    exit 1
fi

python3 -u ./src/analysis/build_probabilistic_decoupling_selection.py \
    --run_dir "${REF_RUN_DIR}" \
    --selection_path "${SELECTION_PATH}" \
    --num_variables "${NUM_VARIABLES}" \
    --variable_tail_quantile "${VARIABLE_TAIL_QUANTILE}" \
    --high_drift_ratio "${HIGH_DRIFT_RATIO}" \
    --low_drift_quantile "${LOW_DRIFT_QUANTILE}" \
    --high_drift_quantile "${HIGH_DRIFT_QUANTILE}" \
    --pairs_per_bin "${PAIRS_PER_BIN}" \
    --random_seed "${RANDOM_SEED}" \
    --device "${DEVICE}" \
    --num_worker "${NUM_WORKER}"
