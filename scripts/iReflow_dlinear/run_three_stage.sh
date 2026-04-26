#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_PATH="${ROOT_PATH:-./data/}"
CHECKPOINTS="${CHECKPOINTS:-./results/runs/DLinear/}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"

DATASET="${DATASET:?DATASET is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
ENC_IN="${ENC_IN:?ENC_IN is required}"
DEC_IN="${DEC_IN:?DEC_IN is required}"
C_OUT="${C_OUT:?C_OUT is required}"
D_MODEL="${D_MODEL:?D_MODEL is required}"
N_HEADS="${N_HEADS:-8}"
E_LAYERS="${E_LAYERS:-2}"
FLOW_LAYERS="${FLOW_LAYERS:-3}"
D_FF="${D_FF:?D_FF is required}"
DROPOUT="${DROPOUT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
LR_PATIENCE="${LR_PATIENCE:-2}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LEN="${PRED_LEN:-192}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEEDS="${SEEDS:-[2027]}"
ITR="${ITR:-1}"
DES="${DES:-Exp}"
D_FEATURES="${D_FEATURES:-M}"
USE_RELATIVE_SPACE="${USE_RELATIVE_SPACE:-True}"
USE_NORM="${USE_NORM:-True}"
MOVING_AVG="${MOVING_AVG:-25}"
INDIVIDUAL="${INDIVIDUAL:-False}"
X0_DIST="${X0_DIST:-pred_gaussian}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"

STAGE1_WANDB_PROJECT="${STAGE1_WANDB_PROJECT:-DLinear-Stage1}"
STAGE2_WANDB_PROJECT="${STAGE2_WANDB_PROJECT:-iReflow-DLinear-Stage2-Uncertainty}"
STAGE3_WANDB_PROJECT="${STAGE3_WANDB_PROJECT:-iReflow-DLinear-Stage3-Velocity}"
STAGE1_LR="${STAGE1_LR:-0.0001}"
STAGE2_LR="${STAGE2_LR:-0.0001}"
STAGE3_LR="${STAGE3_LR:-0.0001}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE2_PATIENCE="${STAGE2_PATIENCE:-6}"
STAGE2_LR_PATIENCE="${STAGE2_LR_PATIENCE:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Device mapping: physical GPU ${GPU_ID} -> process-visible cuda:0"

if [[ "${RUN_STAGE1}" == "1" ]]; then
    echo "===================================================================="
    echo "Stage 1: Pretraining DLinear"
    echo "===================================================================="

    "${PYTHON_BIN}" -u ./src/experiments/DLinear.py \
        --wandb_project "${STAGE1_WANDB_PROJECT}" \
        --is_training 1 \
        --root_path "${ROOT_PATH}" \
        --data_path "${DATA_PATH}" \
        --model_id "${MODEL_ID}" \
        --model DLinear \
        --data "${DATASET}" \
        --features "${D_FEATURES}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --e_layers "${E_LAYERS}" \
        --enc_in "${ENC_IN}" \
        --dec_in "${DEC_IN}" \
        --c_out "${C_OUT}" \
        --des "${DES}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${STAGE1_LR}" \
        --itr "${ITR}" \
        --checkpoints "${CHECKPOINTS}" \
        --n_heads "${N_HEADS}" \
        --dropout "${DROPOUT}" \
        --train_epochs "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --lr_patience "${LR_PATIENCE}" \
        --moving_avg "${MOVING_AVG}" \
        --individual "${INDIVIDUAL}" \
        --use_norm "${USE_NORM}" \
        --skip_test_after_train True \
        --use_gpu True \
        --gpu 0

    echo "Stage 1 completed."
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
    echo "===================================================================="
    echo "Stage 2: Pretraining Uncertainty Estimator"
    echo "===================================================================="

    "${PYTHON_BIN}" -u ./src/experiments/pretrain_uncertainty_estimator_DLinear.py \
        --wandb_project "${STAGE2_WANDB_PROJECT}" \
        --is_training 1 \
        --root_path "${ROOT_PATH}" \
        --data_path "${DATA_PATH}" \
        --model_id "${MODEL_ID}" \
        --model iReflow_DLinear \
        --data "${DATASET}" \
        --features "${D_FEATURES}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --e_layers "${E_LAYERS}" \
        --enc_in "${ENC_IN}" \
        --dec_in "${DEC_IN}" \
        --c_out "${C_OUT}" \
        --des "${DES}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${STAGE2_LR}" \
        --itr "${ITR}" \
        --checkpoints "${CHECKPOINTS}" \
        --flow_layers "${FLOW_LAYERS}" \
        --n_heads "${N_HEADS}" \
        --dropout "${DROPOUT}" \
        --epochs "${STAGE2_EPOCHS}" \
        --patience "${STAGE2_PATIENCE}" \
        --lr_patience "${STAGE2_LR_PATIENCE}" \
        --num_sampling_steps 1 \
        --temperature 1.0 \
        --num_samples 100 \
        --device "${DEVICE}" \
        --use_relative_space "${USE_RELATIVE_SPACE}" \
        --moving_avg "${MOVING_AVG}" \
        --individual "${INDIVIDUAL}" \
        --use_norm "${USE_NORM}" \
        runs --seeds="${SEEDS}"

    echo "Stage 2 completed."
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
    echo "===================================================================="
    echo "Stage 3: Training Velocity Network"
    echo "===================================================================="

    "${PYTHON_BIN}" -u ./src/experiments/iReflow_DLinear.py \
        --wandb_project "${STAGE3_WANDB_PROJECT}" \
        --is_training 1 \
        --root_path "${ROOT_PATH}" \
        --data_path "${DATA_PATH}" \
        --model_id "${MODEL_ID}" \
        --model iReflow_DLinear \
        --data "${DATASET}" \
        --features "${D_FEATURES}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${PRED_LEN}" \
        --e_layers "${E_LAYERS}" \
        --enc_in "${ENC_IN}" \
        --dec_in "${DEC_IN}" \
        --c_out "${C_OUT}" \
        --des "${DES}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${STAGE3_LR}" \
        --itr "${ITR}" \
        --checkpoints "${CHECKPOINTS}" \
        --flow_layers "${FLOW_LAYERS}" \
        --n_heads "${N_HEADS}" \
        --dropout "${DROPOUT}" \
        --epochs "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --lr_patience "${LR_PATIENCE}" \
        --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
        --temperature "${TEMPERATURE}" \
        --num_samples "${NUM_SAMPLES}" \
        --device "${DEVICE}" \
        --use_relative_space "${USE_RELATIVE_SPACE}" \
        --x0_dist "${X0_DIST}" \
        --moving_avg "${MOVING_AVG}" \
        --individual "${INDIVIDUAL}" \
        --use_norm "${USE_NORM}" \
        runs --seeds="${SEEDS}"

    echo "Stage 3 completed."
fi
