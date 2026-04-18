#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-[1,2]}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

bash "$REPO_ROOT/scripts/pretrain_F/Weather.sh"
bash "$REPO_ROOT/scripts/pretrain_G/Weather.sh"

export PYTHONPATH=./

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="Weather" \
   --device="$DEVICE" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --rolling_length=48 \
   --epochs="$EPOCHS" \
   --patience="$PATIENCE" \
   --num_samples="$NUM_SAMPLES" \
   --load_pretrain=True \
   runs --seeds="$SEEDS"
