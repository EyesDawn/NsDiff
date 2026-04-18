#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED=1

GPU_A="${GPU_A:-cuda:0}"
GPU_B="${GPU_B:-cuda:2}"
SEEDS="${SEEDS:-[1]}"
EPOCHS_FAST="${EPOCHS_FAST:-1}"
PATIENCE_FAST="${PATIENCE_FAST:-1}"
TRAIN_NUM_SAMPLES_FAST="${TRAIN_NUM_SAMPLES_FAST:-20}"
EXPORT_NUM_SAMPLES="${EXPORT_NUM_SAMPLES:-100}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cuda:0}"
NUM_WORKER="${NUM_WORKER:-4}"
SELECTION_PATH="${SELECTION_PATH:-./results/analysis/Electricity/probabilistic_decoupling/electricity_probabilistic_decoupling_selection.json}"

run_pair() {
    local script_a="$1"
    local script_b="$2"
    local name_a="$3"
    local name_b="$4"

    echo "Starting ${name_a} on ${GPU_A} and ${name_b} on ${GPU_B}"
    DEVICE="${GPU_A}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS_FAST}" PATIENCE="${PATIENCE_FAST}" TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES_FAST}" NUM_WORKER=0 bash "${script_a}" &
    pid_a=$!
    DEVICE="${GPU_B}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS_FAST}" PATIENCE="${PATIENCE_FAST}" TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES_FAST}" NUM_WORKER=0 bash "${script_b}" &
    pid_b=$!

    wait "${pid_a}"
    wait "${pid_b}"
}

run_pair "scripts/CSDI/Electricity_w96h1s192.sh" "scripts/TimeGrad/Electricity_w96h1s192.sh" "CSDI" "TimeGrad"
run_pair "scripts/TMDM/Electricity_w96h1s192.sh" "scripts/TimeDiff/Electricity_w96h1s192.sh" "TMDM" "TimeDiff"

SELECTION_PATH="${SELECTION_PATH}" bash "scripts/Analysis/build_electricity_probabilistic_decoupling_selection.sh"

DEVICE="${EXPORT_DEVICE}" NUM_WORKER="${NUM_WORKER}" NUM_SAMPLES="${EXPORT_NUM_SAMPLES}" SELECTION_PATH="${SELECTION_PATH}" \
    bash "scripts/Analysis/export_electricity_probabilistic_decoupling_artifacts.sh"

MANIFEST_PATH="${MANIFEST_PATH:-./results/analysis/Electricity/probabilistic_decoupling/electricity_probabilistic_decoupling_manifest.json}" SELECTION_PATH="${SELECTION_PATH}" \
    bash "scripts/Analysis/probabilistic_decoupling_map.sh"
