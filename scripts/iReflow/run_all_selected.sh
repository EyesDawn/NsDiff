#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRIPTS=(
    "scripts/iReflow/ETTm1.sh"
    "scripts/iReflow/ETTm2.sh"
    "scripts/iReflow/Weather.sh"
    "scripts/iReflow/Electricity.sh"
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

echo "All scripts completed successfully."
