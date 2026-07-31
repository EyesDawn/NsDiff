#!/usr/bin/env bash
# Train and evaluate NsDiff on the eight non-ExchangeRate benchmark datasets.
# A job owns one GPU-visible process and performs F pretraining, G pretraining,
# then NsDiff training and its final EnergyScore evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lsflow}"
SEED=2020
PRED_LEN=192
WINDOWS=96
# The requirement is strictly more than 8 GiB, hence the scheduler uses >.
MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB:-8192}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/NSDiff_parallel/${TIMESTAMP}}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather SolarEnergy)

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; expected environment: ${CONDA_ENV}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to discover eligible GPUs." >&2
  exit 2
fi
if ! [[ "${MAX_TASKS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TASKS_PER_GPU must be a positive integer." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
cd "${REPO_ROOT}"

declare -A PID_GPU=()
declare -A PID_JOBS=()
declare -A GPU_ACTIVE=()
FAILED=0
NEXT_JOB=0

eligible_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    while IFS=',' read -r gpu free_mib; do
      gpu="${gpu//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib > MIN_FREE_MEM_MIB )); then
        printf '%s\n' "${gpu}"
      fi
    done
}

prepare_datasets() {
  local dataset
  # torch_timeseries downloads lazily.  Serial preparation avoids two jobs
  # creating or extracting the same dataset simultaneously.
  for dataset in "${DATASETS[@]}"; do
    echo "Checking dataset: ${dataset}"
    conda run --no-capture-output -n "${CONDA_ENV}" python -c \
      "from torch_timeseries.dataset import ${dataset}; ${dataset}(root='./data')"
  done
}

dataset_parameters() {
  local dataset="$1"
  case "${dataset}" in
    ETTh1|ETTh2)
      BATCH_F=32; BATCH_G=32; BATCH_MAIN=32; ROLLING_LENGTH=24
      MAIN_EPOCHS=40; MAIN_PATIENCE=10; NUM_SAMPLES=50
      ;;
    ETTm1|ETTm2)
      BATCH_F=32; BATCH_G=32; BATCH_MAIN=32; ROLLING_LENGTH=48
      MAIN_EPOCHS=50; MAIN_PATIENCE=10; NUM_SAMPLES=""
      ;;
    Electricity)
      BATCH_F=8; BATCH_G=8; BATCH_MAIN=8; ROLLING_LENGTH=48
      MAIN_EPOCHS=50; MAIN_PATIENCE=10; NUM_SAMPLES=""
      ;;
    Traffic)
      BATCH_F=8; BATCH_G=8; BATCH_MAIN=1; ROLLING_LENGTH=48
      MAIN_EPOCHS=50; MAIN_PATIENCE=10; NUM_SAMPLES=""
      ;;
    Weather)
      BATCH_F=32; BATCH_G=32; BATCH_MAIN=32; ROLLING_LENGTH=48
      MAIN_EPOCHS=50; MAIN_PATIENCE=10; NUM_SAMPLES=50
      ;;
    SolarEnergy)
      BATCH_F=16; BATCH_G=16; BATCH_MAIN=16; ROLLING_LENGTH=48
      MAIN_EPOCHS=50; MAIN_PATIENCE=10; NUM_SAMPLES=""
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 2
      ;;
  esac
}

run_pipeline() {
  local dataset="$1"
  local physical_gpu="$2"
  local log_path="$3"
  local status_path="$4"
  local exit_code
  local -a main_args

  dataset_parameters "${dataset}"
  set +e
  (
    set -euo pipefail
    export PYTHONPATH="${REPO_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${physical_gpu}"

    echo "[$(date -u +%FT%TZ)] Starting ${dataset}, pred_len=${PRED_LEN}, windows=${WINDOWS}, seed=${SEED} on GPU ${physical_gpu}"
    conda run --no-capture-output -n "${CONDA_ENV}" python ./src/experiments/pretrain_f.py \
      --dataset_type="${dataset}" --device=cuda:0 --batch_size="${BATCH_F}" \
      --horizon=1 --pred_len="${PRED_LEN}" --windows="${WINDOWS}" \
      --epochs=50 --patience=10 runs --seeds="[${SEED}]"
    conda run --no-capture-output -n "${CONDA_ENV}" python ./src/experiments/pretrain_g.py \
      --dataset_type="${dataset}" --device=cuda:0 --batch_size="${BATCH_G}" \
      --horizon=1 --pred_len="${PRED_LEN}" --windows="${WINDOWS}" \
      --rolling_length="${ROLLING_LENGTH}" --epochs=20 --patience=5 \
      runs --seeds="[${SEED}]"

    main_args=(
      --dataset_type="${dataset}" --device=cuda:0 --batch_size="${BATCH_MAIN}"
      --horizon=1 --pred_len="${PRED_LEN}" --windows="${WINDOWS}"
      --rolling_length="${ROLLING_LENGTH}" --epochs="${MAIN_EPOCHS}" --patience="${MAIN_PATIENCE}"
      --load_pretrain=True --pretrain_seed="${SEED}"
    )
    if [[ -n "${NUM_SAMPLES}" ]]; then
      main_args+=(--num_samples="${NUM_SAMPLES}")
    fi
    conda run --no-capture-output -n "${CONDA_ENV}" python ./src/experiments/NsDiff.py \
      "${main_args[@]}" runs --seeds="[${SEED}]"
  ) >"${log_path}" 2>&1
  exit_code=$?
  set -e

  if (( exit_code == 0 )); then
    printf 'succeeded\n' >"${status_path}"
  else
    printf 'failed:exit_%s\n' "${exit_code}" >"${status_path}"
  fi
  return "${exit_code}"
}

launch_job() {
  local dataset="$1"
  local physical_gpu="$2"
  local log_path="${LOG_DIR}/${dataset}_p${PRED_LEN}.log"
  local status_path="${STATUS_DIR}/${dataset}_p${PRED_LEN}.status"

  echo "Launching ${dataset} on GPU ${physical_gpu}"
  run_pipeline "${dataset}" "${physical_gpu}" "${log_path}" "${status_path}" &
  local pid=$!
  PID_GPU["${pid}"]="${physical_gpu}"
  PID_JOBS["${pid}"]="${dataset}"
  GPU_ACTIVE["${physical_gpu}"]=$(( ${GPU_ACTIVE[${physical_gpu}]:-0} + 1 ))
}

reap_finished_jobs() {
  local gpu pid
  for pid in "${!PID_GPU[@]}"; do
    gpu="${PID_GPU[${pid}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if ! wait "${pid}"; then
        FAILED=1
        echo "Failed: ${PID_JOBS[${pid}]} on GPU ${gpu}." >&2
      else
        echo "Completed: ${PID_JOBS[${pid}]} on GPU ${gpu}."
      fi
      GPU_ACTIVE["${gpu}"]=$(( GPU_ACTIVE[${gpu}] - 1 ))
      unset 'PID_GPU['"${pid}"']' 'PID_JOBS['"${pid}"']'
    fi
  done
}

prepare_datasets

while (( NEXT_JOB < ${#DATASETS[@]} || ${#PID_GPU[@]} > 0 )); do
  reap_finished_jobs

  if (( NEXT_JOB < ${#DATASETS[@]} )); then
    while IFS= read -r gpu; do
      [[ -n "${gpu}" ]] || continue
      while (( ${GPU_ACTIVE[${gpu}]:-0} < MAX_TASKS_PER_GPU && NEXT_JOB < ${#DATASETS[@]} )); do
        launch_job "${DATASETS[${NEXT_JOB}]}" "${gpu}"
        ((NEXT_JOB += 1))
      done
    done < <(eligible_gpus)
  fi

  if (( NEXT_JOB < ${#DATASETS[@]} || ${#PID_GPU[@]} > 0 )); then
    sleep "${POLL_INTERVAL_SECONDS}"
  fi
done

SUMMARY_PATH="${RUN_DIR}/energy_score_summary.csv"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python ./src/analysis/collect_nsdiff_parallel_results.py \
  --run-dir "${RUN_DIR}" --output "${SUMMARY_PATH}" --seed "${SEED}" --metrics es
echo "EnergyScore CSV summary: ${SUMMARY_PATH}"

exit "${FAILED}"
