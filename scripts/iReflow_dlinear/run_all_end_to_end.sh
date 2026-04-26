#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/configs/iReflow_e2e"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_end_to_end_from_config.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-iReflow-DLinear-E2E}"
export PYTHONPATH="${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DEVICE="${DEVICE:-cuda:0}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/results/logs/iReflow_dlinear_e2e/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
    "${CONFIG_DIR}/etth1.yaml"
    "${CONFIG_DIR}/etth2.yaml"
    "${CONFIG_DIR}/ettm1.yaml"
    "${CONFIG_DIR}/ettm2.yaml"
    "${CONFIG_DIR}/electricity.yaml"
    "${CONFIG_DIR}/exchange_rate.yaml"
    "${CONFIG_DIR}/solar_energy.yaml"
    "${CONFIG_DIR}/traffic.yaml"
    "${CONFIG_DIR}/weather.yaml"
)

echo "===================================================================="
echo "iReflow_DLinear end-to-end batch runner"
echo "Python     : ${PYTHON_BIN}"
echo "Physical GPU: ${GPU_ID}"
echo "Mapped device: ${DEVICE}"
echo "Wandb proj : ${WANDB_PROJECT}"
echo "Config dir : ${CONFIG_DIR}"
echo "Log dir    : ${LOG_DIR}"
echo "===================================================================="

for config in "${CONFIGS[@]}"; do
    dataset_name="$("${PYTHON_BIN}" - "${config}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["dataset"])
PY
)"
    dataset_slug="$(printf '%s' "${dataset_name}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//')"
    log_file="${LOG_DIR}/${dataset_slug}.log"

    echo ""
    echo "===================================================================="
    echo "Starting dataset: ${dataset_name}"
    echo "Config          : ${config}"
    echo "Log             : ${log_file}"
    echo "===================================================================="

    "${RUNNER_SCRIPT}" "${config}" 2>&1 | tee "${log_file}"

    echo ""
    echo "Finished dataset: ${dataset_name}"
done

echo ""
echo "All 9 iReflow_DLinear end-to-end experiments completed successfully."
