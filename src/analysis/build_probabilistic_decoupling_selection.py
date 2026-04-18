"""
Build a reproducible window/pair selection for the probabilistic decoupling analysis.

The selection is computed from ground-truth past/future statistics only, so it can
be used to pre-filter expensive forecast-sample exports before model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.export_forecast_samples_from_run import (  # noqa: E402
    _filter_init_kwargs,
    _load_args,
    _resolve_cls,
)
from src.analysis.probabilistic_decoupling_map import _top_k_dims  # noqa: E402


def _compute_drift_from_loader(
    run_dir: str,
    device: str,
    num_worker: Optional[int],
    batch_size: Optional[int],
    eps: float,
) -> np.ndarray:
    config = _load_args(run_dir)
    model_type = str(config.get("model_type"))
    cls = _resolve_cls(model_type)
    init_kwargs = _filter_init_kwargs(cls, config)

    init_kwargs["device"] = device
    if num_worker is not None and "num_worker" in init_kwargs:
        init_kwargs["num_worker"] = num_worker
    if batch_size is not None and "batch_size" in init_kwargs:
        init_kwargs["batch_size"] = batch_size

    exp = cls(**init_kwargs)
    try:
        exp._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
    except TypeError:
        exp._init_data_loader()

    drift_batches = []
    with tqdm(total=len(exp.test_loader.dataset)) as progress_bar:
        for (
            _batch_x,
            _batch_y,
            origin_x,
            origin_y,
            _batch_x_date_enc,
            _batch_y_date_enc,
        ) in exp.test_loader:
            x_ref = origin_x.float()
            y_ref = origin_y.float()

            mu_p = x_ref.mean(dim=1)
            sigma_p = x_ref.std(dim=1).clamp_min(eps)
            mu_f = y_ref.mean(dim=1)
            sigma_f = y_ref.std(dim=1).clamp_min(eps)

            drift = torch.abs(mu_f - mu_p) / (sigma_p + eps)
            drift += torch.abs(torch.log(sigma_f + eps) - torch.log(sigma_p + eps))
            drift_batches.append(drift.cpu().numpy().astype(np.float32))
            progress_bar.update(x_ref.shape[0])

    if not drift_batches:
        raise RuntimeError("No test batches were loaded while building the selection.")
    return np.concatenate(drift_batches, axis=0)


def build_selection(
    run_dir: str,
    selection_path: str,
    num_variables: int,
    variable_tail_quantile: float,
    high_drift_ratio: float,
    low_drift_quantile: float,
    high_drift_quantile: float,
    pairs_per_bin: int,
    random_seed: int,
    device: str,
    num_worker: Optional[int],
    batch_size: Optional[int],
    eps: float,
    feature_dims: Optional[np.ndarray],
) -> Dict[str, Any]:
    drift = _compute_drift_from_loader(
        run_dir=run_dir,
        device=device,
        num_worker=num_worker,
        batch_size=batch_size,
        eps=eps,
    )

    if feature_dims is None:
        tail = np.quantile(drift, variable_tail_quantile, axis=0)
        selected_dims = _top_k_dims(tail, num_variables)
        per_dim_tail_score = tail[selected_dims]
    else:
        selected_dims = np.asarray(feature_dims, dtype=np.int64).reshape(-1)
        per_dim_tail_score = np.quantile(drift, variable_tail_quantile, axis=0)[selected_dims]

    pair_scores = drift[:, selected_dims].reshape(-1)
    window_idx = np.repeat(np.arange(drift.shape[0], dtype=np.int64), selected_dims.shape[0])
    dim_idx = np.tile(selected_dims, drift.shape[0]).astype(np.int64)

    high_count = max(1, int(np.ceil(pair_scores.shape[0] * high_drift_ratio)))
    ranking = np.argsort(-pair_scores, kind="mergesort")
    high_pair_indices = ranking[:high_count]
    high_threshold = float(np.min(pair_scores[high_pair_indices]))
    low_threshold = float(np.quantile(pair_scores, low_drift_quantile))
    mid_threshold = float(np.quantile(pair_scores, high_drift_quantile))

    low_pair_indices = np.flatnonzero(pair_scores <= low_threshold).astype(np.int64)
    mid_pair_indices = np.flatnonzero(
        (pair_scores > low_threshold) & (pair_scores < mid_threshold)
    ).astype(np.int64)
    high_bin_indices = np.flatnonzero(pair_scores >= mid_threshold).astype(np.int64)

    rng = np.random.default_rng(random_seed)

    def _sample(indices: np.ndarray) -> np.ndarray:
        if indices.size <= pairs_per_bin:
            return np.sort(indices)
        return np.sort(rng.choice(indices, size=pairs_per_bin, replace=False))

    sampled_low = _sample(low_pair_indices)
    sampled_mid = _sample(mid_pair_indices)
    sampled_high = _sample(high_bin_indices)

    selected_window_indices = np.unique(
        np.concatenate(
            [
                window_idx[sampled_low],
                window_idx[sampled_mid],
                window_idx[sampled_high],
            ],
            axis=0,
        )
    ).astype(np.int64)

    payload = {
        "run_dir": os.path.abspath(run_dir),
        "window_indices": selected_window_indices.tolist(),
        "selected_dims": selected_dims.tolist(),
        "per_dim_tail_score": per_dim_tail_score.astype(np.float32).tolist(),
        "high_drift_threshold": high_threshold,
        "low_drift_threshold": low_threshold,
        "mid_drift_threshold": mid_threshold,
        "low_pairs": {
            "window_idx": window_idx[sampled_low].tolist(),
            "dim_idx": dim_idx[sampled_low].tolist(),
            "drift_scores": pair_scores[sampled_low].astype(np.float32).tolist(),
        },
        "mid_pairs": {
            "window_idx": window_idx[sampled_mid].tolist(),
            "dim_idx": dim_idx[sampled_mid].tolist(),
            "drift_scores": pair_scores[sampled_mid].astype(np.float32).tolist(),
        },
        "high_pairs": {
            "window_idx": window_idx[sampled_high].tolist(),
            "dim_idx": dim_idx[sampled_high].tolist(),
            "drift_scores": pair_scores[sampled_high].astype(np.float32).tolist(),
        },
    }

    out_dir = os.path.dirname(selection_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reusable sampled selection for the probabilistic decoupling analysis."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--selection_path", type=str, required=True)
    parser.add_argument("--num_variables", type=int, default=10)
    parser.add_argument("--variable_tail_quantile", type=float, default=0.90)
    parser.add_argument("--high_drift_ratio", type=float, default=0.20)
    parser.add_argument("--low_drift_quantile", type=float, default=0.30)
    parser.add_argument("--high_drift_quantile", type=float, default=0.70)
    parser.add_argument("--pairs_per_bin", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--feature_dims", type=int, nargs="*", default=None)

    args = parser.parse_args()
    result = build_selection(
        run_dir=args.run_dir,
        selection_path=args.selection_path,
        num_variables=args.num_variables,
        variable_tail_quantile=args.variable_tail_quantile,
        high_drift_ratio=args.high_drift_ratio,
        low_drift_quantile=args.low_drift_quantile,
        high_drift_quantile=args.high_drift_quantile,
        pairs_per_bin=args.pairs_per_bin,
        random_seed=args.random_seed,
        device=args.device,
        num_worker=args.num_worker,
        batch_size=args.batch_size,
        eps=args.eps,
        feature_dims=None if args.feature_dims is None else np.asarray(args.feature_dims, dtype=np.int64),
    )
    print("Selection written to:", args.selection_path)
    print("Selected windows:", len(result["window_indices"]))


if __name__ == "__main__":
    main()
