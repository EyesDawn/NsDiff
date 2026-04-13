#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/results/formal_prob/ett_exchange"
RUN_LOG="$LOG_DIR/run_prob_models_ett_exchange.log"
SUMMARY_OUTPUT="$LOG_DIR/run_prob_models_ett_exchange_summary.md"

mkdir -p "$LOG_DIR"
exec > >(tee "$RUN_LOG") 2>&1

export SEEDS="${SEEDS:-[1,2]}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"
export EPOCHS="${EPOCHS:-40}"
export PATIENCE="${PATIENCE:-10}"
export NUM_SAMPLES="${NUM_SAMPLES:-50}"
export PRETRAIN_F_EPOCHS="${PRETRAIN_F_EPOCHS:-50}"
export PRETRAIN_F_PATIENCE="${PRETRAIN_F_PATIENCE:-10}"
export PRETRAIN_G_EPOCHS="${PRETRAIN_G_EPOCHS:-20}"
export PRETRAIN_G_PATIENCE="${PRETRAIN_G_PATIENCE:-5}"
export START_MODEL="${START_MODEL:-CSDI}"
export DEVICE="${DEVICE:-cuda:0}"

run_model_group() {
    local model_name="$1"
    shift

    local pids=()
    local scripts=()
    local i

    echo "Starting ${model_name} scripts..."

    for script_name in "$@"; do
        local script_path="$SCRIPT_DIR/${model_name}/${script_name}"
        bash "$script_path" 2>&1 | sed -u "s/^/[${model_name}\/${script_name}] /" &
        pids+=("$!")
        scripts+=("$script_name")
    done

    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "Failed: ${model_name}/${scripts[$i]}" >&2
            for pid in "${pids[@]}"; do
                kill "$pid" 2>/dev/null || true
            done
            wait || true
            return 1
        fi
    done

    echo "Finished ${model_name}."
}

echo "Using seeds: ${SEEDS}"
echo "Using epochs: ${EPOCHS}, patience: ${PATIENCE}, num_samples: ${NUM_SAMPLES}"
echo "Using pretrain F epochs: ${PRETRAIN_F_EPOCHS}, patience: ${PRETRAIN_F_PATIENCE}"
echo "Using pretrain G epochs: ${PRETRAIN_G_EPOCHS}, patience: ${PRETRAIN_G_PATIENCE}"
echo "Using device: ${DEVICE}"
echo "Starting from model: ${START_MODEL}"
echo "Run log: ${RUN_LOG}"

case "$START_MODEL" in
    CSDI)
        run_model_group "CSDI" "ETTh1.sh" "ETTh2.sh" "Exchange.sh"
        ;&
    TimeDiff)
        run_model_group "TimeDiff" "ETTh1.sh" "ETTh2.sh" "Exchange.sh"
        ;&
    NSDiff)
        run_model_group "NSDiff" "ETTh1.sh" "ETTh2.sh" "Exchange.sh"
        ;;
    *)
        echo "Unsupported START_MODEL: ${START_MODEL}" >&2
        echo "Supported START_MODEL values: CSDI, TimeDiff, NSDiff" >&2
        exit 1
        ;;
esac

echo "All model groups completed."

python3 "$SCRIPT_DIR/summarize_formal_prob_run.py" \
    --log "$RUN_LOG" \
    --output "$SUMMARY_OUTPUT"

echo "Summary written to: ${SUMMARY_OUTPUT}"
cat "$SUMMARY_OUTPUT"
