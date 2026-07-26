#!/usr/bin/env bash

# Schedule the 12 requested LS-Flow three-stage experiments.  Selected GPUs
# are filled in performance-priority order, with at most three batch tasks per
# physical GPU at once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-NsDiff}"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-6144}"
POLL_SECONDS="${POLL_SECONDS:-60}"
BATCH_ROOT="${BATCH_ROOT:-${REPO_ROOT}/results/lsflow_three_stage_parallel}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BATCH_DIR="${BATCH_ROOT}/${TIMESTAMP}"
DRY_RUN="${DRY_RUN:-0}"

DATASETS=(ETTh2 Weather SolarEnergy Electricity)
PRED_LENS=(48 96 336)
SEED=2022
PREFERRED_GPU_IDS=(4 5 6 2 3)
MAX_TASKS_PER_GPU=3

if ! [[ "${MIN_FREE_MEMORY_MIB}" =~ ^[0-9]+$ ]] || ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS == 0 )); then
    echo "MIN_FREE_MEMORY_MIB must be an integer and POLL_SECONDS a positive integer." >&2
    exit 2
fi

mkdir -p "${BATCH_DIR}/tasks"
MANIFEST="${BATCH_DIR}/manifest.tsv"
printf 'task_id\tdataset\tpred_len\tseed\n' >"${MANIFEST}"

TASK_IDS=()
TASK_DATASETS=()
TASK_PRED_LENS=()
for dataset in "${DATASETS[@]}"; do
    for pred_len in "${PRED_LENS[@]}"; do
        task_id="${dataset}_pl${pred_len}_seed${SEED}"
        TASK_IDS+=("${task_id}")
        TASK_DATASETS+=("${dataset}")
        TASK_PRED_LENS+=("${pred_len}")
        printf '%s\t%s\t%s\t%s\n' "${task_id}" "${dataset}" "${pred_len}" "${SEED}" >>"${MANIFEST}"
    done
done

echo "LS-Flow three-stage parallel runner"
echo "Batch directory : ${BATCH_DIR}"
echo "Tasks           : ${#TASK_IDS[@]}"
echo "Seed            : ${SEED}"
echo "GPU threshold   : free memory > ${MIN_FREE_MEMORY_MIB} MiB"
echo "GPU priority    : ${PREFERRED_GPU_IDS[*]}"
echo "Per-GPU limit   : ${MAX_TASKS_PER_GPU} tasks"

if [[ "${DRY_RUN}" == "1" ]]; then
    for index in "${!TASK_IDS[@]}"; do
        echo "DRY RUN: ${TASK_IDS[$index]} -> ${SCRIPT_DIR}/run_three_stage_job.sh --dataset ${TASK_DATASETS[$index]} --pred-len ${TASK_PRED_LENS[$index]} --seed ${SEED} --gpu-id <eligible-gpu> --task-dir ${BATCH_DIR}/tasks/${TASK_IDS[$index]}"
    done
    echo "Manifest written to ${MANIFEST}"
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to discover GPUs." >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required; expected environment '${CONDA_ENV}'." >&2
    exit 2
fi

declare -A TASK_PIDS=()
declare -A TASK_GPUS=()
declare -A GPU_ACTIVE_COUNTS=()
declare -A TASK_EXIT_CODES=()
ACTIVE_TASKS=0
NEXT_TASK=0

eligible_gpus() {
    local gpu free_memory
    local -A free_memory_by_gpu=()
    while IFS=',' read -r gpu free_memory; do
        gpu="${gpu//[[:space:]]/}"
        free_memory="${free_memory//[[:space:]]/}"
        [[ -z "${gpu}" || -z "${free_memory}" ]] && continue
        free_memory_by_gpu["${gpu}"]="${free_memory}"
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    for gpu in "${PREFERRED_GPU_IDS[@]}"; do
        free_memory="${free_memory_by_gpu[${gpu}]:-0}"
        if (( free_memory > MIN_FREE_MEMORY_MIB )) && (( ${GPU_ACTIVE_COUNTS[${gpu}]:-0} < MAX_TASKS_PER_GPU )); then
            printf '%s\n' "${gpu}"
        fi
    done
}

launch_task() {
    local index="$1"
    local gpu="$2"
    local task_id="${TASK_IDS[$index]}"
    local task_dir="${BATCH_DIR}/tasks/${task_id}"
    local log_path="${task_dir}/task.log"

    mkdir -p "${task_dir}"
    echo "[$(date -u +%FT%TZ)] Launching ${task_id} on physical GPU ${gpu}"
    (
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="${gpu}"
        conda run --no-capture-output -n "${CONDA_ENV}" \
            bash "${SCRIPT_DIR}/run_three_stage_job.sh" \
            --dataset "${TASK_DATASETS[$index]}" \
            --pred-len "${TASK_PRED_LENS[$index]}" \
            --seed "${SEED}" \
            --gpu-id "${gpu}" \
            --task-dir "${task_dir}"
    ) >"${log_path}" 2>&1 &

    TASK_PIDS["${index}"]=$!
    TASK_GPUS["${index}"]="${gpu}"
    GPU_ACTIVE_COUNTS["${gpu}"]=$(( ${GPU_ACTIVE_COUNTS[${gpu}]:-0} + 1 ))
    ACTIVE_TASKS=$((ACTIVE_TASKS + 1))
}

reap_finished_tasks() {
    local index pid gpu code_file code
    for index in "${!TASK_PIDS[@]}"; do
        code_file="${BATCH_DIR}/tasks/${TASK_IDS[$index]}/exit_code"
        [[ -f "${code_file}" ]] || continue

        pid="${TASK_PIDS[$index]}"
        code=0
        wait "${pid}" || code=$?
        code="$(tr -d '[:space:]' <"${code_file}")"
        gpu="${TASK_GPUS[$index]}"
        TASK_EXIT_CODES["${index}"]="${code}"
        unset 'TASK_PIDS[$index]'
        unset 'TASK_GPUS[$index]'
        GPU_ACTIVE_COUNTS["${gpu}"]=$(( ${GPU_ACTIVE_COUNTS[${gpu}]:-0} - 1 ))
        ACTIVE_TASKS=$((ACTIVE_TASKS - 1))
        echo "[$(date -u +%FT%TZ)] Finished ${TASK_IDS[$index]} on GPU ${gpu} (exit ${code})"
    done
}

launch_available_tasks() {
    local gpu launched
    local -a free_gpus
    while (( NEXT_TASK < ${#TASK_IDS[@]} )); do
        mapfile -t free_gpus < <(eligible_gpus)
        if (( ${#free_gpus[@]} == 0 )); then
            return
        fi

        launched=0
        for gpu in "${free_gpus[@]}"; do
            (( NEXT_TASK < ${#TASK_IDS[@]} )) || break
            launch_task "${NEXT_TASK}" "${gpu}"
            NEXT_TASK=$((NEXT_TASK + 1))
            launched=1
        done
        if (( launched == 0 )); then
            return
        fi
    done
}

while true; do
    reap_finished_tasks

    if (( NEXT_TASK == ${#TASK_IDS[@]} && ACTIVE_TASKS == 0 )); then
        break
    fi

    launch_available_tasks

    if (( NEXT_TASK < ${#TASK_IDS[@]} )); then
        echo "[$(date -u +%FT%TZ)] Waiting ${POLL_SECONDS}s for a GPU with > ${MIN_FREE_MEMORY_MIB} MiB free memory."
    fi
    sleep "${POLL_SECONDS}"
done

CSV_PATH="${BATCH_DIR}/lsflow_three_stage_results.csv"
conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/src/analysis/collect_lsflow_three_stage_results.py" \
    --batch-dir "${BATCH_DIR}" \
    --output-csv "${CSV_PATH}"

failed=0
for index in "${!TASK_IDS[@]}"; do
    if [[ "${TASK_EXIT_CODES[$index]:-1}" != "0" ]]; then
        failed=1
    fi
done

echo "CSV summary: ${CSV_PATH}"
exit "${failed}"
