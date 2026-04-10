#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-[1,2]}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

bash "$REPO_ROOT/scripts/pretrain_F/ETTh1.sh"
bash "$REPO_ROOT/scripts/pretrain_G/ETTh1.sh"

export PYTHONPATH=./

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="ETTh1" \
   --device="$DEVICE" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=92 \
   --rolling_length=24 \
   --load_pretrain=True \
   --epochs="$EPOCHS" \
   --patience="$PATIENCE" \
   --num_samples="$NUM_SAMPLES" \
   runs --seeds="$SEEDS"
