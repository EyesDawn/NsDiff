#!/bin/bash
set -euo pipefail

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OUTPUT_DIR="${OUTPUT_DIR:-./results/analysis/Electricity/probabilistic_decoupling}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_DIR}/electricity_probabilistic_decoupling_manifest.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/electricity_probabilistic_decoupling_map.pdf}"
METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/electricity_probabilistic_decoupling_map.json}"
DATASET_NAME="${DATASET_NAME:-Electricity}"
NUM_VARIABLES="${NUM_VARIABLES:-10}"
VARIABLE_TAIL_QUANTILE="${VARIABLE_TAIL_QUANTILE:-0.9}"
HIGH_DRIFT_RATIO="${HIGH_DRIFT_RATIO:-0.2}"
LOW_DRIFT_QUANTILE="${LOW_DRIFT_QUANTILE:-0.3}"
HIGH_DRIFT_QUANTILE="${HIGH_DRIFT_QUANTILE:-0.7}"
ENERGY_BATCH_SIZE="${ENERGY_BATCH_SIZE:-128}"

if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "Missing manifest: ${MANIFEST_PATH}"
    echo "Create a JSON file that lists per-method sample artifacts exported by export_forecast_samples_on_test."
    exit 1
fi

python3 -u ./src/analysis/probabilistic_decoupling_map.py \
    --manifest_path "${MANIFEST_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --metadata_path "${METADATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --num_variables "${NUM_VARIABLES}" \
    --variable_tail_quantile "${VARIABLE_TAIL_QUANTILE}" \
    --high_drift_ratio "${HIGH_DRIFT_RATIO}" \
    --low_drift_quantile "${LOW_DRIFT_QUANTILE}" \
    --high_drift_quantile "${HIGH_DRIFT_QUANTILE}" \
    --energy_batch_size "${ENERGY_BATCH_SIZE}"
