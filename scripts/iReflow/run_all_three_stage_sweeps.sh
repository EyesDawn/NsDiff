#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRIPTS=(
    "scripts/iReflow/ETTh1.sh"
    "scripts/iReflow/ETTh2.sh"
    "scripts/iReflow/ETTm1.sh"
    "scripts/iReflow/ETTm2.sh"
    "scripts/iReflow/Electricity.sh"
    "scripts/iReflow/Exchage.sh"
    "scripts/iReflow/Solar.sh"
    "scripts/iReflow/Traffic.sh"
    "scripts/iReflow/Weather.sh"
)

cd "${REPO_ROOT}"

for script in "${SCRIPTS[@]}"; do
    echo "===================================================================="
    echo "Running ${script}"
    echo "===================================================================="

    bash "${REPO_ROOT}/${script}"

    echo ""
    echo "Finished ${script}"
    echo ""
done

echo "All 9 iReflow three-stage step-sweep scripts completed successfully."
