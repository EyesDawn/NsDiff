#!/bin/bash
set -euo pipefail

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET_NAME="${DATASET_NAME:-Electricity}"
WINDOWS="${WINDOWS:-96}"
HORIZON="${HORIZON:-1}"
PRED_LEN="${PRED_LEN:-192}"
RUN_SPEC="w${WINDOWS}h${HORIZON}s${PRED_LEN}"

EXPORT_SEED="${EXPORT_SEED:-2027}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKER="${NUM_WORKER:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/analysis/Electricity/probabilistic_decoupling}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_DIR}/electricity_probabilistic_decoupling_manifest.json}"

TIMEGRAD_BATCH_SIZE="${TIMEGRAD_BATCH_SIZE:-${BATCH_SIZE}}"
CSDI_BATCH_SIZE="${CSDI_BATCH_SIZE:-${BATCH_SIZE}}"
TIMEDIFF_BATCH_SIZE="${TIMEDIFF_BATCH_SIZE:-${BATCH_SIZE}}"
NSDIFF_BATCH_SIZE="${NSDIFF_BATCH_SIZE:-${BATCH_SIZE}}"
TMDM_BATCH_SIZE="${TMDM_BATCH_SIZE:-${BATCH_SIZE}}"
IREFLOW_BATCH_SIZE="${IREFLOW_BATCH_SIZE:-${BATCH_SIZE}}"

mkdir -p "${OUTPUT_DIR}"

resolve_latest_run_dir() {
    local base_dir="$1"
    if [ ! -d "$base_dir" ]; then
        return 1
    fi
    find "$base_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

resolve_latest_ireflow_run_dir() {
    local base_dir="$1"
    if [ ! -d "$base_dir" ]; then
        return 1
    fi
    local candidate
    candidate="$(find "$base_dir" -mindepth 2 -maxdepth 2 -type d -name 'train_mode_1' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    if [ -n "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    candidate="$(find "$base_dir" -mindepth 2 -maxdepth 2 -type d -name 'train_mode_2' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    if [ -n "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    return 1
}

export_one() {
    local method_name="$1"
    local run_dir="$2"
    local output_path="$3"
    local batch_size="${4:-}"
    local extra_args=()

    if [ -z "${run_dir}" ] || [ ! -d "${run_dir}" ]; then
        echo "Missing run directory for ${method_name}: ${run_dir}" >&2
        exit 1
    fi

    echo "============================================================"
    echo "Exporting ${method_name}"
    echo "Run dir : ${run_dir}"
    echo "Output  : ${output_path}"
    echo "============================================================"

    if [ -n "${batch_size}" ]; then
        extra_args+=(--batch_size "${batch_size}")
    fi

    python3 -u ./src/analysis/export_forecast_samples_from_run.py \
        --run_dir "${run_dir}" \
        --output_path "${output_path}" \
        --seed "${EXPORT_SEED}" \
        --device "${DEVICE}" \
        --num_worker "${NUM_WORKER}" \
        --num_samples "${NUM_SAMPLES}" \
        "${extra_args[@]}"
}

TIMEGRAD_RUN_DIR="${TIMEGRAD_RUN_DIR:-$(resolve_latest_run_dir "./results/runs/TimeGrad/${DATASET_NAME}/${RUN_SPEC}" || true)}"
CSDI_RUN_DIR="${CSDI_RUN_DIR:-$(resolve_latest_run_dir "./results/runs/CSDI/${DATASET_NAME}/${RUN_SPEC}" || true)}"
TIMEDIFF_RUN_DIR="${TIMEDIFF_RUN_DIR:-$(resolve_latest_run_dir "./results/runs/TimeDiff/${DATASET_NAME}/${RUN_SPEC}" || true)}"
NSDIFF_RUN_DIR="${NSDIFF_RUN_DIR:-$(resolve_latest_run_dir "./results/runs/NsDiff4/${DATASET_NAME}/${RUN_SPEC}" || true)}"
TMDM_RUN_DIR="${TMDM_RUN_DIR:-$(resolve_latest_run_dir "./results/runs/TMDM/${DATASET_NAME}/${RUN_SPEC}" || true)}"
IREFLOW_RUN_DIR="${IREFLOW_RUN_DIR:-$(resolve_latest_ireflow_run_dir "./results/runs/iReflow/${DATASET_NAME}/${RUN_SPEC}" || true)}"

TIMEGRAD_NPZ="${OUTPUT_DIR}/timegrad_forecast_samples.npz"
CSDI_NPZ="${OUTPUT_DIR}/csdi_forecast_samples.npz"
TIMEDIFF_NPZ="${OUTPUT_DIR}/timediff_forecast_samples.npz"
NSDIFF_NPZ="${OUTPUT_DIR}/nsdiff_forecast_samples.npz"
TMDM_NPZ="${OUTPUT_DIR}/tmdm_forecast_samples.npz"
IREFLOW_NPZ="${OUTPUT_DIR}/ireflow_forecast_samples.npz"

export_one "TimeGrad" "${TIMEGRAD_RUN_DIR}" "${TIMEGRAD_NPZ}" "${TIMEGRAD_BATCH_SIZE}"
export_one "CSDI" "${CSDI_RUN_DIR}" "${CSDI_NPZ}" "${CSDI_BATCH_SIZE}"
export_one "TimeDiff" "${TIMEDIFF_RUN_DIR}" "${TIMEDIFF_NPZ}" "${TIMEDIFF_BATCH_SIZE}"
export_one "NsDiff" "${NSDIFF_RUN_DIR}" "${NSDIFF_NPZ}" "${NSDIFF_BATCH_SIZE}"
export_one "TMDM" "${TMDM_RUN_DIR}" "${TMDM_NPZ}" "${TMDM_BATCH_SIZE}"
export_one "PDN-Flow" "${IREFLOW_RUN_DIR}" "${IREFLOW_NPZ}" "${IREFLOW_BATCH_SIZE}"

python3 - <<PY
import json
manifest = {
    "dataset_name": "${DATASET_NAME}",
    "methods": [
        {"name": "TimeGrad", "path": "${TIMEGRAD_NPZ}"},
        {"name": "CSDI", "path": "${CSDI_NPZ}"},
        {"name": "TimeDiff", "path": "${TIMEDIFF_NPZ}"},
        {"name": "NsDiff", "path": "${NSDIFF_NPZ}"},
        {"name": "TMDM", "path": "${TMDM_NPZ}"},
        {"name": "PDN-Flow", "path": "${IREFLOW_NPZ}"}
    ]
}
with open("${MANIFEST_PATH}", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("Manifest written to: ${MANIFEST_PATH}")
PY

echo ""
echo "All Electricity forecast-sample artifacts exported."
echo "Manifest: ${MANIFEST_PATH}"
echo "Next step:"
echo "  MANIFEST_PATH=${MANIFEST_PATH} bash scripts/Analysis/probabilistic_decoupling_map.sh"
