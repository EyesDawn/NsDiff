#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="Traffic"
export DATA_PATH="traffic/traffic.csv"
export MODEL_ID="traffic_96_192"
export ENC_IN=862
export DEC_IN=862
export C_OUT=862
export D_MODEL=512
export N_HEADS=8
export E_LAYERS=4
export FLOW_LAYERS=3
export D_FF=512
export BATCH_SIZE=16
export EPOCHS=40
export PATIENCE=8
export LR_PATIENCE=3
export SEQ_LEN=96
export PRED_LEN=192
export NUM_SAMPLING_STEPS=5
export NUM_SAMPLES=100
export GPU_ID="${GPU_ID:-2}"
export SEEDS='[2026]'

"${SCRIPT_DIR}/run_three_stage.sh"
