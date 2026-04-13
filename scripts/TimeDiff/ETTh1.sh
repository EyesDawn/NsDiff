#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH=./
SEEDS="${SEEDS:-[1,2]}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda:0}"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeDiff.py \
   --dataset_type="ETTh1" \
   --device="$DEVICE" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=92 \
   --epochs="$EPOCHS" \
   --patience="$PATIENCE" \
   --num_samples="$NUM_SAMPLES" \
   runs --seeds="$SEEDS"
