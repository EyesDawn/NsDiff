#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_PATH="${ROOT_PATH:-./data/}"
CHECKPOINTS="${CHECKPOINTS:-./results/runs/iTransformer/}"
DEVICE="${DEVICE:-cuda:0}"
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-[2020]}"
NUM_SAMPLING_STEPS_LIST="${NUM_SAMPLING_STEPS_LIST:-5}"

RUN_STAGE1="${RUN_STAGE1:-0}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"
STAGE3_IS_TRAINING="${STAGE3_IS_TRAINING:-1}"

STAGE1_WANDB_PROJECT="${STAGE1_WANDB_PROJECT:-iReflow-Stage1-iTransformer}"
STAGE2_WANDB_PROJECT="${STAGE2_WANDB_PROJECT:-iReflow-Stage2-Uncertainty}"
STAGE3_WANDB_PROJECT="${STAGE3_WANDB_PROJECT:-iReflow-Stage3-Velocity}"

STAGE1_CHECKPOINTS="${STAGE1_CHECKPOINTS:-${CHECKPOINTS}}"
STAGE2_CHECKPOINTS="${STAGE2_CHECKPOINTS:-${CHECKPOINTS}}"
STAGE3_CHECKPOINTS="${STAGE3_CHECKPOINTS:-${CHECKPOINTS}}"

STAGE1_LR="${STAGE1_LR:-0.0001}"
STAGE1_SEED="${STAGE1_SEED:-2020}"
STAGE2_LR="${STAGE2_LR:-0.0001}"
STAGE3_LR="${STAGE3_LR:-0.0001}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE2_PATIENCE="${STAGE2_PATIENCE:-6}"
STAGE2_LR_PATIENCE="${STAGE2_LR_PATIENCE:-1}"

export PYTHONPATH="${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_stage1() {
    if [[ "${RUN_STAGE1}" != "1" ]]; then
        echo "Skipping Stage 1. Set RUN_STAGE1=1 to pretrain iTransformer in this script."
        echo ""
        return
    fi

    echo "============================================================================"
    echo "Stage 1: Pretraining iTransformer"
    echo "============================================================================"
    echo ""

    local cmd=(
        "${PYTHON_BIN}" -u "./src/experiments/iTransformer.py"
        --wandb_project "${STAGE1_WANDB_PROJECT}"
        --is_training 1
        --root_path "${ROOT_PATH}"
        --data_path "${DATA_PATH}"
        --model_id "${MODEL_ID}"
        --model iTransformer
        --data "${DATASET}"
        --features "${D_FEATURES}"
        --seq_len "${SEQ_LEN}"
        --pred_len "${PRED_LEN}"
        --e_layers "${E_LAYERS}"
        --enc_in "${ENC_IN}"
        --dec_in "${DEC_IN}"
        --c_out "${C_OUT}"
        --des "${DES}"
        --d_model "${D_MODEL}"
        --d_ff "${D_FF}"
        --batch_size "${BATCH_SIZE}"
        --learning_rate "${STAGE1_LR}"
        --itr "${ITR}"
        --checkpoints "${STAGE1_CHECKPOINTS}"
        --n_heads "${N_HEADS}"
        --dropout "${DROPOUT}"
        --train_epochs "${EPOCHS}"
        --patience "${PATIENCE}"
        --lr_patience "${LR_PATIENCE}"
        --seed "${STAGE1_SEED}"
    )

    (
        cd "${REPO_ROOT}"
        "${cmd[@]}"
    )

    echo ""
    echo "Stage 1 completed!"
    echo ""
}

run_stage2() {
    if [[ "${RUN_STAGE2}" != "1" ]]; then
        echo "Skipping Stage 2. Set RUN_STAGE2=1 to pretrain the uncertainty estimator."
        echo ""
        return
    fi

    echo "============================================================================"
    echo "Stage 2: Pretraining Uncertainty Estimator"
    echo "============================================================================"
    echo ""

    local cmd=(
        "${PYTHON_BIN}" -u "./src/experiments/pretrain_uncertainty_estimator.py"
        --wandb_project "${STAGE2_WANDB_PROJECT}"
        --is_training 1
        --root_path "${ROOT_PATH}"
        --data_path "${DATA_PATH}"
        --model_id "${MODEL_ID}"
        --model iReflow
        --data "${DATASET}"
        --features "${D_FEATURES}"
        --seq_len "${SEQ_LEN}"
        --pred_len "${PRED_LEN}"
        --e_layers "${E_LAYERS}"
        --enc_in "${ENC_IN}"
        --dec_in "${DEC_IN}"
        --c_out "${C_OUT}"
        --des "${DES}"
        --d_model "${D_MODEL}"
        --d_ff "${D_FF}"
        --batch_size "${BATCH_SIZE}"
        --lr "${STAGE2_LR}"
        --itr "${ITR}"
        --checkpoints "${STAGE2_CHECKPOINTS}"
        --flow_layers "${FLOW_LAYERS}"
        --n_heads "${N_HEADS}"
        --dropout "${DROPOUT}"
        --epochs "${STAGE2_EPOCHS}"
        --patience "${STAGE2_PATIENCE}"
        --lr_patience "${STAGE2_LR_PATIENCE}"
        --num_sampling_steps 1
        --temperature 1.0
        --num_samples 100
        --device "${DEVICE}"
        --use_relative_space "${USE_RELATIVE_SPACE}"
        runs "--seeds=${SEEDS}"
    )

    (
        cd "${REPO_ROOT}"
        "${cmd[@]}"
    )

    echo ""
    echo "Stage 2 completed!"
    echo ""
}

run_stage3_sweep() {
    if [[ "${RUN_STAGE3}" != "1" ]]; then
        echo "Skipping Stage 3. Set RUN_STAGE3=1 to run the sampling-step sweep."
        echo ""
        return
    fi

    echo "============================================================================"
    echo "Stage 3: Training Velocity Network / Step Sweep"
    echo "============================================================================"
    echo "Dataset              : ${DATASET}"
    echo "Seeds                : ${SEEDS}"
    echo "Sampling step sweep  : ${NUM_SAMPLING_STEPS_LIST}"
    echo "Stage 3 is_training  : ${STAGE3_IS_TRAINING}"
    echo ""

    local num_sampling_steps
    for num_sampling_steps in ${NUM_SAMPLING_STEPS_LIST}; do
        echo "--------------------------------------------------------------------"
        echo "Stage 3 run for NUM_SAMPLING_STEPS=${num_sampling_steps}"
        echo "--------------------------------------------------------------------"

        local cmd=(
            "${PYTHON_BIN}" -u "./src/experiments/iReflow.py"
            --wandb_project "${STAGE3_WANDB_PROJECT}"
            --is_training "${STAGE3_IS_TRAINING}"
            --root_path "${ROOT_PATH}"
            --data_path "${DATA_PATH}"
            --model_id "${MODEL_ID}"
            --model iReflow
            --data "${DATASET}"
            --features "${D_FEATURES}"
            --seq_len "${SEQ_LEN}"
            --pred_len "${PRED_LEN}"
            --e_layers "${E_LAYERS}"
            --enc_in "${ENC_IN}"
            --dec_in "${DEC_IN}"
            --c_out "${C_OUT}"
            --des "${DES}"
            --d_model "${D_MODEL}"
            --d_ff "${D_FF}"
            --batch_size "${BATCH_SIZE}"
            --lr "${STAGE3_LR}"
            --itr "${ITR}"
            --checkpoints "${STAGE3_CHECKPOINTS}"
            --flow_layers "${FLOW_LAYERS}"
            --n_heads "${N_HEADS}"
            --dropout "${DROPOUT}"
            --epochs "${EPOCHS}"
            --patience "${PATIENCE}"
            --lr_patience "${LR_PATIENCE}"
            --num_sampling_steps "${num_sampling_steps}"
            --temperature "${TEMPERATURE}"
            --num_samples "${NUM_SAMPLES}"
            --device "${DEVICE}"
            --use_relative_space "${USE_RELATIVE_SPACE}"
            --x0_dist "${X0_DIST}"
        )

        if [[ -n "${EVAL_MICRO_BATCH_SIZE:-}" ]]; then
            cmd+=(--eval_micro_batch_size "${EVAL_MICRO_BATCH_SIZE}")
        fi

        if [[ -n "${CHECKPOINT_MODE_FOR_TEST:-}" ]]; then
            cmd+=(--checkpoint_mode_for_test "${CHECKPOINT_MODE_FOR_TEST}")
        fi

        cmd+=(runs "--seeds=${SEEDS}")

        (
            cd "${REPO_ROOT}"
            "${cmd[@]}"
        )

        echo ""
    done

    echo "Stage 3 sweep completed!"
    echo ""
}
