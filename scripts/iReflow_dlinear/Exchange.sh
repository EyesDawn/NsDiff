#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="ExchangeRate"
export DATA_PATH="ExchangeRate/exchange_rate.csv"
export MODEL_ID="Exchange_96_192"
export ENC_IN=8
export DEC_IN=8
export C_OUT=8
export D_MODEL=128
export N_HEADS=8
export E_LAYERS=2
export FLOW_LAYERS=3
export D_FF=128
export BATCH_SIZE=32
export EPOCHS=40
export PATIENCE=8
export LR_PATIENCE=2
export SEQ_LEN=96
export PRED_LEN=192
export NUM_SAMPLING_STEPS=5
export NUM_SAMPLES=100
export GPU_ID="${GPU_ID:-0}"
export SEEDS='[2210]'

"${SCRIPT_DIR}/run_three_stage.sh"
