#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

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
   --epochs=50 \
   --patience=10 \
   --load_pretrain=True \
   runs --seeds='[1, 2, 3]'
