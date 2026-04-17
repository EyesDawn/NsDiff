#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH=./:/notebooks/pytorchtimseries
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

SEEDS="${SEEDS:-[1]}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-10}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-${NUM_SAMPLES:-20}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKER="${NUM_WORKER:-0}"

python3 -u ./src/experiments/TimeGrad.py \
   --dataset_type="Electricity" \
   --device="$DEVICE" \
   --batch_size="$BATCH_SIZE" \
   --num_worker="$NUM_WORKER" \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --epochs="$EPOCHS" \
   --patience="$PATIENCE" \
   --num_samples="$TRAIN_NUM_SAMPLES" \
   runs --seeds="$SEEDS"
