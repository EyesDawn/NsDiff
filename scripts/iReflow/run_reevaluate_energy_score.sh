#!/usr/bin/env bash
set -euo pipefail

# Shell entry point for EnergyScore-only LS-Flow/iReflow checkpoint
# re-evaluation.  The Python launcher performs dynamic GPU scheduling; this
# wrapper only establishes the repository and conda environment consistently.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-NsDiff}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

cd "${REPO_ROOT}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/src/analysis/reevaluate_ireflow_energy_score.py" \
  "$@"
