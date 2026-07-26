#!/usr/bin/env bash

# Run one complete, seed-specific LS-Flow three-stage experiment.  This script
# is intentionally small: the dataset scripts remain the source of truth for
# all model hyperparameters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET=""
PRED_LEN=""
SEED=""
GPU_ID=""
TASK_DIR=""

usage() {
    echo "Usage: $0 --dataset <name> --pred-len <length> --seed <seed> --gpu-id <id> --task-dir <dir>" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --pred-len) PRED_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --task-dir) TASK_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${DATASET}" || -z "${PRED_LEN}" || -z "${SEED}" || -z "${GPU_ID}" || -z "${TASK_DIR}" ]]; then
    usage
    exit 2
fi

case "${DATASET}" in
    ETTh2) DATASET_SCRIPT="ETTh2.sh" ;;
    Weather) DATASET_SCRIPT="Weather.sh" ;;
    SolarEnergy) DATASET_SCRIPT="Solar.sh" ;;
    Electricity) DATASET_SCRIPT="Electricity.sh" ;;
    *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

mkdir -p "${TASK_DIR}"

on_exit() {
    local code=$?
    set +e
    printf '%s\n' "${code}" >"${TASK_DIR}/exit_code"
    return "${code}"
}
trap on_exit EXIT

export PYTHONPATH="${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export GPU_ID
export DEVICE=cuda:0
export PRED_LEN
export SEEDS="[${SEED}]"
export STAGE1_SEED="${SEED}"
export RUN_STAGE1=1
export RUN_STAGE2=1
export RUN_STAGE3=1
export STAGE3_IS_TRAINING=1
export NUM_SAMPLING_STEPS_LIST=5
export PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/${DATASET_SCRIPT}"
