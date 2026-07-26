#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/mean_only_ablation.yaml}"
DEVICE="${DEVICE:-cuda:0}"
ROOT_PATH="${ROOT_PATH:-./data/}"
CHECKPOINTS="${CHECKPOINTS:-./results/runs/iTransformer/}"
WANDB_PROJECT="${WANDB_PROJECT:-iReflow-MeanOnly}"
# Distinct GPUs run in parallel; datasets sharing one GPU remain serial.
GPU_MAP="${GPU_MAP:-ETTm1=0,ETTm2=1,Weather=2,Electricity=3}"
DATASETS="${DATASETS:-}"
SEEDS="${SEEDS:-}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${REPO_ROOT}"

command=(conda run --no-capture-output -n NsDiff python \
    "${REPO_ROOT}/src/analysis/run_mean_only_ablation.py" \
    --repo-root "${REPO_ROOT}" \
    --config "${CONFIG_PATH}" \
    --root-path "${ROOT_PATH}" \
    --checkpoints "${CHECKPOINTS}" \
    --device "${DEVICE}" \
    --wandb-project "${WANDB_PROJECT}" \
    --gpu-map "${GPU_MAP}")

if [[ -n "${DATASETS}" ]]; then
    read -r -a selected_datasets <<< "${DATASETS}"
    command+=(--datasets "${selected_datasets[@]}")
fi

if [[ -n "${SEEDS}" ]]; then
    command+=(--seeds "${SEEDS}")
fi

exec "${command[@]}"
