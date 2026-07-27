#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:2}"
NUM_WORKER="${NUM_WORKER:-0}"
EVAL_MICRO_BATCH_SIZE="${EVAL_MICRO_BATCH_SIZE:-8}"
OUTPUT_CSV="${OUTPUT_CSV:-./results/analysis/lsflow_normalized_distribution_seed2020.csv}"

# Main LS-Flow setting on the seed=2020 sweep: num_sampling_steps=5.
RUN_DIR_ETTH1="./results/runs/iReflow/ETTh1/w96h1s192/9adf4055a76188f2fb769366c51d5cd9/train_mode_1"
RUN_DIR_ETTH2="./results/runs/iReflow/ETTh2/w96h1s192/b65ba842374dc18dafa4efea68686ed3/train_mode_1"
RUN_DIR_ETTM1="./results/runs/iReflow/ETTm1/w96h1s192/9a3045cf2c9ab21e9473e896d3fb80f1/train_mode_1"
RUN_DIR_ETTM2="./results/runs/iReflow/ETTm2/w96h1s192/0615493dd777372e08579c62d40e2542/train_mode_1"
RUN_DIR_WEATHER="./results/runs/iReflow/Weather/w96h1s192/a5daaaee383532b4bb66dc439196b879/train_mode_1"
RUN_DIR_ELECTRICITY="./results/runs/iReflow/Electricity/w96h1s192/c2df97649dbdc3a7a698ea8fca66431a/train_mode_1"
RUN_DIR_SOLAR="./results/runs/iReflow/SolarEnergy/w96h1s192/3012552e94a4cf81dba5a402bbb8544a/train_mode_1"
RUN_DIR_TRAFFIC="./results/runs/iReflow/Traffic/w96h1s192/dfc91779450f15633ee03502b3c3ab9a/train_mode_1"

conda run -n NsDiff python ./src/analysis/evaluate_lsflow_normalized_distribution.py \
  --run_dirs \
  "${RUN_DIR_ETTH1}" \
  "${RUN_DIR_ETTH2}" \
  "${RUN_DIR_ETTM1}" \
  "${RUN_DIR_ETTM2}" \
  "${RUN_DIR_WEATHER}" \
  "${RUN_DIR_ELECTRICITY}" \
  "${RUN_DIR_SOLAR}" \
  "${RUN_DIR_TRAFFIC}" \
  --seed 2020 \
  --device "${DEVICE}" \
  --num_worker "${NUM_WORKER}" \
  --eval_micro_batch_size "${EVAL_MICRO_BATCH_SIZE}" \
  --output_csv "${OUTPUT_CSV}"
