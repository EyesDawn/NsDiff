#!/usr/bin/env bash
set -euo pipefail

# Run one dataset at a time to bound host-memory usage.  GPU IDs use the
# physical numbering shown by nvidia-smi (PCI_BUS_ID ordering is enforced).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-NsDiff}"
GPU_IDS_STRING="${GPU_IDS:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STRING}"
DATASETS_STRING="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 Electricity SolarEnergy Traffic Weather}"
read -r -a DATASETS <<< "${DATASETS_STRING}"
RUNS_PER_DATASET="${RUNS_PER_DATASET:-1}"

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SHARD_ROOT="${SHARD_ROOT:-evidence/shape_decomposition_shards/${TIMESTAMP}}"
RUN_ROOT="${RUN_ROOT:-results/third_order_shape/${TIMESTAMP}}"
LOG_ROOT="${LOG_ROOT:-results/logs/third_order_shape/${TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evidence}"
mkdir -p "${REPO_ROOT}/${LOG_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/nsdiff_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  echo "Launching ${dataset} serially on physical GPU ${gpu}"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${gpu}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      bash "${REPO_ROOT}/scripts/iReflow/run_third_order_shape_pilot.sh" \
      --datasets "${dataset}" \
      --runs_per_dataset "${RUNS_PER_DATASET}" \
      --output_root "${SHARD_ROOT}/${dataset}" \
      --run_root "${RUN_ROOT}" \
      "$@"
  ) >"${REPO_ROOT}/${LOG_ROOT}/${dataset}.log" 2>&1
done

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/src/analysis/merge_third_order_shape_evidence.py" \
  --input_root "${SHARD_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --runs_per_dataset "${RUNS_PER_DATASET}" \
  --datasets "${DATASETS[@]}"

echo "Completed. Consolidated evidence: ${REPO_ROOT}/${OUTPUT_ROOT}/shape_decomposition.{md,per_seed.csv}"
