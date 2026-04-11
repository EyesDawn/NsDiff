#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <config_path>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="$1"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ireflow/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-iReflow-E2E}"
ROOT_PATH="${ROOT_PATH:-./data/}"
CHECKPOINTS="${CHECKPOINTS:-./results/runs/iTransformer/}"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PYTHON_BIN}" - "${REPO_ROOT}" "${CONFIG_PATH}" "${WANDB_PROJECT}" "${ROOT_PATH}" "${CHECKPOINTS}" "${DEVICE}" <<'PY'
import json
import os
import shlex
import subprocess
import sys

import yaml

repo_root, config_path, wandb_project, root_path, checkpoints, device = sys.argv[1:]

required_fields = [
    "dataset",
    "data_path",
    "model_id",
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "n_heads",
    "e_layers",
    "flow_layers",
    "d_ff",
    "dropout",
    "batch_size",
    "lr",
    "epochs",
    "patience",
    "lr_patience",
    "seq_len",
    "pred_len",
    "num_sampling_steps",
    "temperature",
    "num_samples",
    "seeds",
    "use_relative_space",
    "x0_dist",
]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

wandb_project = str(cfg.get("wandb_project", wandb_project))

missing = [field for field in required_fields if field not in cfg]
if missing:
    raise SystemExit(f"Missing required fields in {config_path}: {missing}")

seeds = cfg["seeds"]
if isinstance(seeds, str):
    seeds_arg = seeds
else:
    seeds_arg = json.dumps(seeds, separators=(",", ":"))

def bool_arg(value):
    return "True" if value else "False"

cmd = [
    sys.executable,
    "-u",
    "./src/experiments/iReflow.py",
    "--wandb_project",
    wandb_project,
    "--is_training",
    "2",
    "--root_path",
    root_path,
    "--data_path",
    cfg["data_path"],
    "--model_id",
    str(cfg["model_id"]),
    "--model",
    "iReflow",
    "--data",
    str(cfg["dataset"]),
    "--features",
    "M",
    "--seq_len",
    str(cfg["seq_len"]),
    "--pred_len",
    str(cfg["pred_len"]),
    "--e_layers",
    str(cfg["e_layers"]),
    "--enc_in",
    str(cfg["enc_in"]),
    "--dec_in",
    str(cfg["dec_in"]),
    "--c_out",
    str(cfg["c_out"]),
    "--des",
    "Exp",
    "--d_model",
    str(cfg["d_model"]),
    "--d_ff",
    str(cfg["d_ff"]),
    "--batch_size",
    str(cfg["batch_size"]),
    "--lr",
    str(cfg["lr"]),
    "--itr",
    "1",
    "--checkpoints",
    checkpoints,
    "--flow_layers",
    str(cfg["flow_layers"]),
    "--n_heads",
    str(cfg["n_heads"]),
    "--dropout",
    str(cfg["dropout"]),
    "--epochs",
    str(cfg["epochs"]),
    "--patience",
    str(cfg["patience"]),
    "--lr_patience",
    str(cfg["lr_patience"]),
    "--num_sampling_steps",
    str(cfg["num_sampling_steps"]),
    "--temperature",
    str(cfg["temperature"]),
    "--num_samples",
    str(cfg["num_samples"]),
    "--device",
    device,
    "--use_relative_space",
    bool_arg(cfg["use_relative_space"]),
    "--x0_dist",
    str(cfg["x0_dist"]),
    "runs",
    f"--seeds={seeds_arg}",
]

print("Resolved config:", config_path, flush=True)
print("Launching command:", flush=True)
print(shlex.join(cmd), flush=True)

env = dict(os.environ)
env["PYTHONPATH"] = repo_root
env["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

subprocess.run(cmd, cwd=repo_root, env=env, check=True)
PY
