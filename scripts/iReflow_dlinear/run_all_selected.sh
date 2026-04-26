#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRIPTS=(
    "scripts/iReflow_dlinear/ETTm1.sh"
    "scripts/iReflow_dlinear/ETTm2.sh"
    "scripts/iReflow_dlinear/Weather.sh"
    "scripts/iReflow_dlinear/Electricity.sh"
)

cd "${REPO_ROOT}"

for script in "${SCRIPTS[@]}"; do
    echo "===================================================================="
    echo "Running ${script}"
    echo "===================================================================="
    bash "${REPO_ROOT}/${script}"
    echo ""
done

echo "All selected iReflow_DLinear scripts completed successfully."
