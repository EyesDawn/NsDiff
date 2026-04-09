#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
SEEDS="${SEEDS:-[1,2,3]}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

bash "$REPO_ROOT/scripts/pretrain_F/Exchange.sh"
bash "$REPO_ROOT/scripts/pretrain_G/Exchange.sh"

export PYTHONPATH=./

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:6" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=92 \
   --rolling_length=24 \
   --epochs="$EPOCHS" \
   --patience="$PATIENCE" \
   --num_samples="$NUM_SAMPLES" \
   --load_pretrain=True \
   runs --seeds="$SEEDS"
