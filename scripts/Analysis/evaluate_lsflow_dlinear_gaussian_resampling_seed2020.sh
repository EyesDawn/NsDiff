#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DEVICE="${DEVICE:-cuda:2}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_WORKER="${NUM_WORKER:-0}"
EVAL_MICRO_BATCH_SIZE="${EVAL_MICRO_BATCH_SIZE:-8}"
OUTPUT_CSV="${OUTPUT_CSV:-./results/analysis/lsflow_dlinear_gaussian_resampling_seed2020_steps5.csv}"

# DLinear-backbone LS-Flow setting on the seed=2020 sweep: num_sampling_steps=5.
RUN_DIR_ETTH1="./results/runs/iReflow_DLinear/ETTh1/w96h1s192/dc4e6ca045a15b166da77d90078d9c0a/train_mode_1"
RUN_DIR_ETTH2="./results/runs/iReflow_DLinear/ETTh2/w96h1s192/dd3c95b6d1358fc53e7e717a00e87e50/train_mode_1"
RUN_DIR_ETTM1="./results/runs/iReflow_DLinear/ETTm1/w96h1s192/331130ba0432bb9705f39ec2f7baf163/train_mode_1"
RUN_DIR_ETTM2="./results/runs/iReflow_DLinear/ETTm2/w96h1s192/5411e82143eb728964f8013ff705709c/train_mode_1"
RUN_DIR_WEATHER="./results/runs/iReflow_DLinear/Weather/w96h1s192/3084bb3f33661995cfcefbdf6f550e9f/train_mode_1"
RUN_DIR_ELECTRICITY="./results/runs/iReflow_DLinear/Electricity/w96h1s192/cc53020cac54c019d417c84fb9e19fef/train_mode_1"
RUN_DIR_SOLAR="./results/runs/iReflow_DLinear/SolarEnergy/w96h1s192/ba57bb17c25005250f65199f8a348c4a/train_mode_1"
RUN_DIR_TRAFFIC="./results/runs/iReflow_DLinear/Traffic/w96h1s192/e2f3307f9a9c139d1072476254adc9e5/train_mode_1"

conda run -n NsDiff python ./src/analysis/evaluate_lsflow_gaussian_resampling.py \
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
  --num_samples "${NUM_SAMPLES}" \
  --num_worker "${NUM_WORKER}" \
  --eval_micro_batch_size "${EVAL_MICRO_BATCH_SIZE}" \
  --output_csv "${OUTPUT_CSV}"
