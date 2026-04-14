#!/bin/bash
set -euo pipefail

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

NPZ_PATH="${NPZ_PATH:-./results/analysis/ETTm1/ETTm1_decoupling_case_study_data_seed_2029.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-./results/analysis/ETTm1/ETTm1_pca_transport_path_seed_2029.pdf}"
METADATA_PATH="${METADATA_PATH:-./results/analysis/ETTm1/ETTm1_pca_transport_path_seed_2029.json}"
DATASET_NAME="${DATASET_NAME:-ETTm1}"
NUM_WINDOWS="${NUM_WINDOWS:-256}"
HIGH_DRIFT_RATIO="${HIGH_DRIFT_RATIO:-0.2}"
MIN_HIGH_DRIFT_WINDOWS="${MIN_HIGH_DRIFT_WINDOWS:-64}"
SEED="${SEED:-2029}"
PCA_COMPONENTS="${PCA_COMPONENTS:-8}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"

if [ ! -f "${NPZ_PATH}" ]; then
    echo "Missing input artifact: ${NPZ_PATH}"
    echo "Generate it first with scripts/Analysis/export_decoupling_case_study_data.sh"
    exit 1
fi

python3 -u ./src/analysis/pca_transport_path.py \
    --npz_path "${NPZ_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --metadata_path "${METADATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --num_windows "${NUM_WINDOWS}" \
    --high_drift_ratio "${HIGH_DRIFT_RATIO}" \
    --min_high_drift_windows "${MIN_HIGH_DRIFT_WINDOWS}" \
    --tau_values 0.0 0.25 0.5 0.75 1.0 \
    --seed "${SEED}" \
    --pca_components "${PCA_COMPONENTS}" \
    --sample_batch_size "${SAMPLE_BATCH_SIZE}"
