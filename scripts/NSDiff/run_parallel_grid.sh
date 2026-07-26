#!/usr/bin/env bash
# Run the NsDiff ETTh2/Weather/SolarEnergy/Electricity grid with one pipeline
# per eligible GPU, then collect final test metrics in a CSV file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lsflow}"
SEED="${SEED:-2022}"
MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB:-6144}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/NSDiff_parallel/${TIMESTAMP}}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"

DATASETS=(ETTh2 Weather SolarEnergy Electricity)
PRED_LENS=(48 96 336)
JOBS=()
for dataset in "${DATASETS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do
    JOBS+=("${dataset}:${pred_len}")
  done
done

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
cd "${REPO_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to discover eligible GPUs." >&2
  exit 2
fi

declare -A PID_GPU=()
declare -A PID_JOBS=()
declare -A GPU_ACTIVE=()
FAILED=0
NEXT_JOB=0

eligible_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required to discover eligible GPUs." >&2
    return 1
  }
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    while IFS=',' read -r gpu free_mib; do
      gpu="${gpu//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MEM_MIB )); then
        printf '%s\n' "${gpu}"
      fi
    done
}

prepare_datasets() {
  local dataset
  # torch_timeseries downloads datasets lazily from every process.  Preparing
  # them serially prevents the three prediction-length jobs of one dataset
  # from concurrently creating or extracting the same source file.
  for dataset in "${DATASETS[@]}"; do
    echo "Checking dataset: ${dataset}"
    conda run --no-capture-output -n "${CONDA_ENV}" python -c \
      "from torch_timeseries.dataset import ${dataset}; ${dataset}(root='./data')"
  done
}

run_pipeline() {
  local dataset="$1"
  local pred_len="$2"
  local physical_gpu="$3"
  local log_path="$4"
  local status_path="$5"
  local batch_f batch_g batch_main windows rolling_length main_epochs main_patience num_samples

  case "${dataset}" in
    ETTh2)
      batch_f=32; batch_g=32; batch_main=32; windows=92; rolling_length=24
      main_epochs=40; main_patience=10; num_samples=50
      ;;
    Weather)
      batch_f=32; batch_g=32; batch_main=32; windows=96; rolling_length=48
      main_epochs=50; main_patience=10; num_samples=50
      ;;
    SolarEnergy)
      batch_f=32; batch_g=32; batch_main=16; windows=96; rolling_length=48
      main_epochs=50; main_patience=10; num_samples=""
      ;;
    Electricity)
      batch_f=8; batch_g=8; batch_main=8; windows=96; rolling_length=48
      main_epochs=50; main_patience=10; num_samples=""
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 2
      ;;
  esac

  # Keep the outer function alive long enough to write a status file, while
  # retaining fail-fast behaviour inside the three-stage pipeline.
  set +e
  (
    set -euo pipefail
    export PYTHONPATH="${REPO_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${physical_gpu}"

    echo "[$(date -u +%FT%TZ)] Starting ${dataset}, pred_len=${pred_len}, seed=${SEED} on GPU ${physical_gpu}"
    conda run --no-capture-output -n "${CONDA_ENV}" python ./src/experiments/pretrain_f.py \
      --dataset_type="${dataset}" --device=cuda:0 --batch_size="${batch_f}" \
      --horizon=1 --pred_len="${pred_len}" --windows="${windows}" --epochs=50 --patience=10 \
      runs --seeds="[${SEED}]"
    conda run --no-capture-output -n "${CONDA_ENV}" python ./src/experiments/pretrain_g.py \
      --dataset_type="${dataset}" --device=cuda:0 --batch_size="${batch_g}" \
      --horizon=1 --pred_len="${pred_len}" --windows="${windows}" --rolling_length="${rolling_length}" \
      --epochs=20 --patience=5 \
      runs --seeds="[${SEED}]"

    main_args=(
      --dataset_type="${dataset}" --device=cuda:0 --batch_size="${batch_main}"
      --horizon=1 --pred_len="${pred_len}" --windows="${windows}"
      --rolling_length="${rolling_length}" --epochs="${main_epochs}" --patience="${main_patience}"
      --load_pretrain=True --pretrain_seed="${SEED}"
    )
    if [[ -n "${num_samples}" ]]; then
      main_args+=(--num_samples="${num_samples}")
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
  local job="$1"
  local physical_gpu="$2"
  local dataset="${job%%:*}"
  local pred_len="${job##*:}"
  local log_path="${LOG_DIR}/${dataset}_p${pred_len}.log"
  local status_path="${STATUS_DIR}/${dataset}_p${pred_len}.status"

  echo "Launching ${dataset}, pred_len=${pred_len} on GPU ${physical_gpu}"
  run_pipeline "${dataset}" "${pred_len}" "${physical_gpu}" "${log_path}" "${status_path}" &
  local pid=$!
  PID_GPU["${pid}"]="${physical_gpu}"
  PID_JOBS["${pid}"]="${job}"
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

while (( NEXT_JOB < ${#JOBS[@]} || ${#PID_GPU[@]} > 0 )); do
  reap_finished_jobs

  if (( NEXT_JOB < ${#JOBS[@]} )); then
    while IFS= read -r gpu; do
      [[ -n "${gpu}" ]] || continue
      while (( ${GPU_ACTIVE[${gpu}]:-0} < MAX_TASKS_PER_GPU && NEXT_JOB < ${#JOBS[@]} )); do
        launch_job "${JOBS[${NEXT_JOB}]}" "${gpu}"
        ((NEXT_JOB += 1))
      done
    done < <(eligible_gpus)
  fi

  if (( NEXT_JOB < ${#JOBS[@]} || ${#PID_GPU[@]} > 0 )); then
    sleep "${POLL_INTERVAL_SECONDS}"
  fi
done

SUMMARY_PATH="${RUN_DIR}/summary.csv"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python ./src/analysis/collect_nsdiff_parallel_results.py \
  --run-dir "${RUN_DIR}" --output "${SUMMARY_PATH}" --seed "${SEED}"
echo "CSV summary: ${SUMMARY_PATH}"

exit "${FAILED}"
