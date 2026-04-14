#!/bin/bash

set -euo pipefail

export PYTHONPATH=./

NPZ_PATH="${1:-./results/analysis/ETTm1/ETTm1_decoupling_case_study_data_seed_2029.npz}"
OUTPUT_DIR="${2:-./results/analysis/ETTm1}"
PREFIX="${3:-ETTm1_decoupling_case_study_seed_2029_all_dims}"
BINS="${BINS:-120}"
N_COLS="${N_COLS:-3}"
SEED="${SEED:-2029}"
KDE_MAX_POINTS="${KDE_MAX_POINTS:-200000}"
SHOW_STATS="${SHOW_STATS:-1}"

export NPZ_PATH OUTPUT_DIR PREFIX BINS N_COLS SEED KDE_MAX_POINTS SHOW_STATS

python3 - <<'PY'
import os

import numpy as np

from src.analysis.pdn_density_plot import plot_per_feature_separate

npz_path = os.environ["NPZ_PATH"]
output_dir = os.environ["OUTPUT_DIR"]
prefix = os.environ["PREFIX"]
bins = int(os.environ["BINS"])
n_cols = int(os.environ["N_COLS"])
seed = int(os.environ["SEED"])
kde_max_points = int(os.environ["KDE_MAX_POINTS"])
show_stats = os.environ["SHOW_STATS"] == "1"

data = np.load(npz_path)
required = {"Y", "Z_PDN"}
missing = required - set(data.files)
if missing:
    raise KeyError(f"{npz_path} is missing required arrays: {sorted(missing)}")

Y = data["Y"]
Z_PDN = data["Z_PDN"]
if Y.shape != Z_PDN.shape:
    raise ValueError(f"Shape mismatch: Y{Y.shape} vs Z_PDN{Z_PDN.shape}")

dims = list(range(Y.shape[2]))
rng = np.random.default_rng(seed)
os.makedirs(output_dir, exist_ok=True)

plot_per_feature_separate(
    Y=Y,
    Z_PDN=Z_PDN,
    output_path_y=os.path.join(output_dir, f"{prefix}_pdn_density_per_feature_Y.png"),
    output_path_zpdn=os.path.join(output_dir, f"{prefix}_pdn_density_per_feature_ZPDN.png"),
    clip_range=None,
    bins=bins,
    feature_dims=dims,
    n_cols=n_cols,
    add_normal_ref=False,
    show_stats=show_stats,
    kde_max_points=kde_max_points,
    rng=rng,
)

print(f"Finished plotting {len(dims)} feature dims from: {npz_path}")
PY
