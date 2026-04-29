#!/bin/bash
set -euo pipefail

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OUTPUT_DIR="${OUTPUT_DIR:-./fig/example}"
PREFIX="${PREFIX:-pdn_motivation_schematic}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/${PREFIX}.png}"
METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/${PREFIX}.json}"

SEED="${SEED:-7}"
BINS="${BINS:-52}"
DPI="${DPI:-1200}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-160}"
NUM_DENSITY_SAMPLES="${NUM_DENSITY_SAMPLES:-1800}"
CLIP_LOWER_QUANTILE="${CLIP_LOWER_QUANTILE:-0.005}"
CLIP_UPPER_QUANTILE="${CLIP_UPPER_QUANTILE:-0.995}"

python3 -u ./src/analysis/pdn_motivation_plot.py \
    --output_path "${OUTPUT_PATH}" \
    --metadata_path "${METADATA_PATH}" \
    --seed "${SEED}" \
    --bins "${BINS}" \
    --dpi "${DPI}" \
    --segment_length "${SEGMENT_LENGTH}" \
    --num_density_samples "${NUM_DENSITY_SAMPLES}" \
    --clip_lower_quantile "${CLIP_LOWER_QUANTILE}" \
    --clip_upper_quantile "${CLIP_UPPER_QUANTILE}"
