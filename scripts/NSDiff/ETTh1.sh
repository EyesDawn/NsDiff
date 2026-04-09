#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
SEEDS="${SEEDS:-[1,2,3]}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

bash "$REPO_ROOT/scripts/pretrain_F/ETTh1.sh"
bash "$REPO_ROOT/scripts/pretrain_G/ETTh1.sh"

export PYTHONPATH=./

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="ETTh1" \
   --device="cuda:4" \
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
