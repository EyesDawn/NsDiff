#!/bin/bash
set -euo pipefail

export PYTHONPATH=./

OUTPUT_PATH="${OUTPUT_PATH:-./results/analysis/synthetic/synthetic_drift_stress_test.pdf}"
METADATA_PATH="${METADATA_PATH:-./results/analysis/synthetic/synthetic_drift_stress_test.json}"
SEED="${SEED:-2026}"
CONTEXT_LEN="${CONTEXT_LEN:-96}"
PRED_LEN="${PRED_LEN:-96}"
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-384}"
NUM_REFERENCE_WINDOWS="${NUM_REFERENCE_WINDOWS:-2048}"
RESIDUAL_LIBRARY_SIZE="${RESIDUAL_LIBRARY_SIZE:-128}"
PDN_MU_NOISE="${PDN_MU_NOISE:-0.06}"
PDN_LOG_SIGMA_NOISE="${PDN_LOG_SIGMA_NOISE:-0.01}"
NUM_TRAJECTORY_SAMPLES="${NUM_TRAJECTORY_SAMPLES:-18}"
CONTEXT_TAIL="${CONTEXT_TAIL:-24}"

python3 -u ./src/analysis/synthetic_drift_stress_test.py \
    --output_path "${OUTPUT_PATH}" \
    --metadata_path "${METADATA_PATH}" \
    --seed "${SEED}" \
    --context_len "${CONTEXT_LEN}" \
    --pred_len "${PRED_LEN}" \
    --drift_levels 0 1 2 3 \
    --num_eval_windows "${NUM_EVAL_WINDOWS}" \
    --num_reference_windows "${NUM_REFERENCE_WINDOWS}" \
    --residual_library_size "${RESIDUAL_LIBRARY_SIZE}" \
    --pdn_mu_noise "${PDN_MU_NOISE}" \
    --pdn_log_sigma_noise "${PDN_LOG_SIGMA_NOISE}" \
    --num_trajectory_samples "${NUM_TRAJECTORY_SAMPLES}" \
    --context_tail "${CONTEXT_TAIL}"
