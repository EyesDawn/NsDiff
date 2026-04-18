#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID
SEEDS="${SEEDS:-[1,2]}"
PRETRAIN_F_EPOCHS="${PRETRAIN_F_EPOCHS:-${PRETRAIN_EPOCHS:-50}}"
PRETRAIN_F_PATIENCE="${PRETRAIN_F_PATIENCE:-${PRETRAIN_PATIENCE:-10}}"
DEVICE="${DEVICE:-cuda:2}"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/pretrain_f.py \
   --dataset_type="Weather" \
   --device="$DEVICE" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --epochs="$PRETRAIN_F_EPOCHS" \
   --patience="$PRETRAIN_F_PATIENCE" \
   runs --seeds="$SEEDS"
