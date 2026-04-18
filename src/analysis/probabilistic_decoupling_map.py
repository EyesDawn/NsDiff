"""
Distribution-level mechanism analysis on probabilistic forecasts.

This script implements the two real-benchmark analysis figures described in
`docs/Analytical_experiments.md`:

1. Probabilistic Decoupling Map
2. Drift-Conditioned Centroid Shift

Input format
------------
The script expects a manifest JSON that points to per-method `.npz` artifacts
exported by `ProbForecastExp.export_forecast_samples_on_test`, where each file
contains:

  - Y:       [N, P, D]
  - samples: [N, P, D, S]
  - mu_X:    [N, 1, D]
  - sigma_X: [N, 1, D]

Example manifest:
{
  "dataset_name": "Electricity",
  "methods": [
    {"name": "TimeGrad", "path": ".../timegrad_samples.npz"},
    {"name": "CSDI", "path": ".../csdi_samples.npz"},
    {"name": "TimeDiff", "path": ".../timediff_samples.npz"},
    {"name": "NsDiff", "path": ".../nsdiff_samples.npz"},
    {"name": "TMDM", "path": ".../tmdm_samples.npz"},
    {"name": "PDN-Flow", "path": ".../ireflow_samples.npz"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


DEFAULT_MARKERS = {
    "TimeGrad": "o",
    "CSDI": "s",
    "TimeDiff": "^",
    "NsDiff": "D",
    "TMDM": "p",
    "PDN-Flow": "*",
    "iReflow": "*",
}

DEFAULT_COLORS = {
    "TimeGrad": "#4C78A8",
    "CSDI": "#72B7B2",
    "TimeDiff": "#9ECAE1",
    "NsDiff": "#F58518",
    "TMDM": "#ECA82C",
    "PDN-Flow": "#D62728",
    "iReflow": "#D62728",
}

DEFAULT_FAMILY = {
    "TimeGrad": "standard_diffusion",
    "CSDI": "standard_diffusion",
    "TimeDiff": "standard_diffusion",
    "NsDiff": "informative_prior",
    "TMDM": "informative_prior",
    "PDN-Flow": "pdn_flow",
    "iReflow": "pdn_flow",
}

DISPLAY_ALIASES = {
    "iReflow": "PDN-Flow",
}


@dataclass
class MethodSpec:
    name: str
    display_name: str
    path: str
    family: str
    color: str
    marker: str


@dataclass
class MethodArtifact:
    spec: MethodSpec
    y: np.ndarray
    samples: np.ndarray
    mu_x: np.ndarray
    sigma_x: np.ndarray
    window_index: np.ndarray


@dataclass
class SelectedPairs:
    window_idx: np.ndarray
    dim_idx: np.ndarray
    drift_scores: np.ndarray
    selected_dims: np.ndarray
    per_dim_tail_score: np.ndarray
    high_pair_indices: np.ndarray
    low_pair_indices: np.ndarray
    mid_pair_indices: np.ndarray
    high_drift_threshold: float
    low_drift_threshold: float
    mid_drift_threshold: float


@dataclass
class MethodScores:
    spec: MethodSpec
    macro_all: np.ndarray
    micro_all: np.ndarray
    macro_mid: np.ndarray
    micro_mid: np.ndarray
    centroids: Dict[str, Tuple[float, float]]


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.15,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "savefig.dpi": 600,
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid")


def _normalize_display_name(name: str) -> str:
    return DISPLAY_ALIASES.get(name, name)


def _infer_family(name: str) -> str:
    return DEFAULT_FAMILY.get(name, "other")


def _infer_color(name: str) -> str:
    return DEFAULT_COLORS.get(name, "#444444")


def _infer_marker(name: str) -> str:
    return DEFAULT_MARKERS.get(name, "o")


def _load_manifest(manifest_path: str) -> Tuple[str, List[MethodSpec]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset_name = manifest.get("dataset_name", "Dataset")
    methods_cfg = manifest.get("methods", [])
    if not methods_cfg:
        raise ValueError("Manifest must contain a non-empty `methods` list.")

    methods = []
    for item in methods_cfg:
        if "name" not in item or "path" not in item:
            raise ValueError("Each method entry must contain `name` and `path`.")
        name = str(item["name"])
        display_name = str(item.get("display_name", _normalize_display_name(name)))
        methods.append(
            MethodSpec(
                name=name,
                display_name=display_name,
                path=str(item["path"]),
                family=str(item.get("family", _infer_family(name))),
                color=str(item.get("color", _infer_color(name))),
                marker=str(item.get("marker", _infer_marker(name))),
            )
        )
    return dataset_name, methods


def _as_horizon_tensor(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("%s must have shape [N, P, D], got %s." % (name, arr.shape))
    return np.asarray(arr, dtype=np.float32)


def _as_sample_tensor(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError("%s must have shape [N, P, D, S], got %s." % (name, arr.shape))
    return np.asarray(arr, dtype=np.float32)


def _as_stats_tensor(arr: np.ndarray, horizon: int, name: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("%s must be a 3D tensor, got %s." % (name, arr.shape))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[1] == 1:
        return arr
    if arr.shape[1] == horizon:
        return np.mean(arr, axis=1, keepdims=True)
    raise ValueError(
        "%s must have shape [N, 1, D] or [N, P, D], got %s with P=%d."
        % (name, arr.shape, horizon)
    )


def _load_artifacts(methods: Sequence[MethodSpec], atol: float = 1e-5) -> List[MethodArtifact]:
    artifacts = []
    ref_y = None
    ref_mu_x = None
    ref_sigma_x = None
    ref_window_index = None

    for spec in methods:
        if not os.path.isfile(spec.path):
            raise FileNotFoundError("Missing artifact for %s: %s" % (spec.display_name, spec.path))
        data = np.load(spec.path)
        required = {"Y", "samples", "mu_X", "sigma_X"}
        missing = required - set(data.files)
        if missing:
            raise KeyError("%s is missing arrays: %s" % (spec.path, sorted(missing)))

        y = _as_horizon_tensor(data["Y"], "Y")
        samples = _as_sample_tensor(data["samples"], "samples")
        mu_x = _as_stats_tensor(data["mu_X"], horizon=y.shape[1], name="mu_X")
        sigma_x = _as_stats_tensor(data["sigma_X"], horizon=y.shape[1], name="sigma_X")
        if "window_index" in data.files:
            window_index = np.asarray(data["window_index"], dtype=np.int64).reshape(-1)
        else:
            window_index = np.arange(y.shape[0], dtype=np.int64)

        if samples.shape[:3] != y.shape:
            raise ValueError(
                "Shape mismatch for %s: Y%s vs samples%s."
                % (spec.display_name, y.shape, samples.shape)
            )
        if window_index.shape[0] != y.shape[0]:
            raise ValueError(
                "Shape mismatch for %s: window_index%s vs Y%s."
                % (spec.display_name, window_index.shape, y.shape)
            )

        if ref_y is None:
            ref_y = y
            ref_mu_x = mu_x
            ref_sigma_x = sigma_x
            ref_window_index = window_index
        else:
            if not np.allclose(y, ref_y, atol=atol, rtol=0.0):
                raise ValueError(
                    "Artifact `%s` does not share the same realized futures as the reference artifact. "
                    "All methods must be exported on the same test windows in the same order."
                    % spec.path
                )
            if not np.allclose(mu_x, ref_mu_x, atol=atol, rtol=0.0):
                raise ValueError(
                    "Artifact `%s` does not share the same history means as the reference artifact."
                    % spec.path
                )
            if not np.allclose(sigma_x, ref_sigma_x, atol=atol, rtol=0.0):
                raise ValueError(
                    "Artifact `%s` does not share the same history scales as the reference artifact."
                    % spec.path
                )
            if not np.array_equal(window_index, ref_window_index):
                raise ValueError(
                    "Artifact `%s` does not share the same exported window indices as the reference artifact."
                    % spec.path
                )

        artifacts.append(
            MethodArtifact(
                spec=spec,
                y=y,
                samples=samples,
                mu_x=mu_x,
                sigma_x=sigma_x,
                window_index=window_index,
            )
        )

    return artifacts


def _load_selected_pairs_from_file(
    selection_path: str,
    reference: MethodArtifact,
) -> SelectedPairs:
    with open(selection_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    exported_window_index = np.asarray(reference.window_index, dtype=np.int64).reshape(-1)
    row_lookup = {int(idx): pos for pos, idx in enumerate(exported_window_index.tolist())}

    def _extract_bin(bin_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = payload.get(bin_name, {})
        window_idx = np.asarray(cfg.get("window_idx", []), dtype=np.int64).reshape(-1)
        dim_idx = np.asarray(cfg.get("dim_idx", []), dtype=np.int64).reshape(-1)
        drift_scores = np.asarray(cfg.get("drift_scores", []), dtype=np.float32).reshape(-1)
        if window_idx.shape != dim_idx.shape or window_idx.shape != drift_scores.shape:
            raise ValueError("Selection bin `%s` has inconsistent shapes." % bin_name)
        if window_idx.size == 0:
            raise ValueError("Selection bin `%s` is empty." % bin_name)

        local_window_idx = []
        local_dim_idx = []
        local_drift_scores = []
        for orig_idx, dim, drift in zip(window_idx.tolist(), dim_idx.tolist(), drift_scores.tolist()):
            if int(orig_idx) not in row_lookup:
                raise KeyError(
                    "Selection references window %d, but it is missing from exported artifacts."
                    % int(orig_idx)
                )
            local_window_idx.append(row_lookup[int(orig_idx)])
            local_dim_idx.append(int(dim))
            local_drift_scores.append(float(drift))
        return (
            np.asarray(local_window_idx, dtype=np.int64),
            np.asarray(local_dim_idx, dtype=np.int64),
            np.asarray(local_drift_scores, dtype=np.float32),
        )

    low_window_idx, low_dim_idx, low_drift = _extract_bin("low_pairs")
    mid_window_idx, mid_dim_idx, mid_drift = _extract_bin("mid_pairs")
    high_window_idx, high_dim_idx, high_drift = _extract_bin("high_pairs")

    low_size = low_window_idx.shape[0]
    mid_size = mid_window_idx.shape[0]
    high_size = high_window_idx.shape[0]

    return SelectedPairs(
        window_idx=np.concatenate([low_window_idx, mid_window_idx, high_window_idx], axis=0),
        dim_idx=np.concatenate([low_dim_idx, mid_dim_idx, high_dim_idx], axis=0),
        drift_scores=np.concatenate([low_drift, mid_drift, high_drift], axis=0),
        selected_dims=np.asarray(payload.get("selected_dims", []), dtype=np.int64).reshape(-1),
        per_dim_tail_score=np.asarray(
            payload.get("per_dim_tail_score", []), dtype=np.float32
        ).reshape(-1),
        high_pair_indices=np.arange(low_size + mid_size, low_size + mid_size + high_size, dtype=np.int64),
        low_pair_indices=np.arange(0, low_size, dtype=np.int64),
        mid_pair_indices=np.arange(low_size, low_size + mid_size, dtype=np.int64),
        high_drift_threshold=float(payload.get("high_drift_threshold", 0.0)),
        low_drift_threshold=float(payload.get("low_drift_threshold", 0.0)),
        mid_drift_threshold=float(payload.get("mid_drift_threshold", 0.0)),
    )


def _compute_drift_scores(
    y: np.ndarray,
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    eps: float,
) -> np.ndarray:
    mu_f = np.mean(y, axis=1)
    sigma_f = np.maximum(np.std(y, axis=1), eps)
    mu_p = mu_x[:, 0, :]
    sigma_p = np.maximum(sigma_x[:, 0, :], eps)
    drift = np.abs(mu_f - mu_p) / (sigma_p + eps)
    drift += np.abs(np.log(sigma_f + eps) - np.log(sigma_p + eps))
    return drift.astype(np.float32)


def _top_k_dims(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), scores.shape[0]))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64)


def _select_pairs(
    y: np.ndarray,
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    num_variables: int,
    variable_tail_quantile: float,
    high_drift_ratio: float,
    low_drift_quantile: float,
    high_drift_quantile: float,
    eps: float,
    feature_dims: Optional[Sequence[int]] = None,
) -> SelectedPairs:
    drift = _compute_drift_scores(y=y, mu_x=mu_x, sigma_x=sigma_x, eps=eps)

    if feature_dims is None or len(feature_dims) == 0:
        tail = np.quantile(drift, variable_tail_quantile, axis=0)
        selected_dims = _top_k_dims(tail, num_variables)
        per_dim_tail_score = tail
    else:
        selected_dims = np.asarray(feature_dims, dtype=np.int64)
        if np.any(selected_dims < 0) or np.any(selected_dims >= drift.shape[1]):
            raise ValueError("feature_dims contains out-of-range feature indices.")
        per_dim_tail_score = np.quantile(drift, variable_tail_quantile, axis=0)

    pair_scores = drift[:, selected_dims].reshape(-1)
    window_idx = np.repeat(np.arange(drift.shape[0], dtype=np.int64), selected_dims.shape[0])
    dim_idx = np.tile(selected_dims, drift.shape[0]).astype(np.int64)

    high_count = max(1, int(math.ceil(pair_scores.shape[0] * high_drift_ratio)))
    top_local = np.argpartition(pair_scores, -high_count)[-high_count:]
    high_pair_indices = top_local[np.argsort(pair_scores[top_local])[::-1]].astype(np.int64)
    high_drift_threshold = float(np.min(pair_scores[high_pair_indices]))

    low_threshold = float(np.quantile(pair_scores, low_drift_quantile))
    high_threshold = float(np.quantile(pair_scores, high_drift_quantile))
    low_pair_indices = np.flatnonzero(pair_scores <= low_threshold).astype(np.int64)
    mid_pair_indices = np.flatnonzero(
        (pair_scores > low_threshold) & (pair_scores < high_threshold)
    ).astype(np.int64)
    high_bin_indices = np.flatnonzero(pair_scores >= high_threshold).astype(np.int64)

    if low_pair_indices.size == 0 or mid_pair_indices.size == 0 or high_bin_indices.size == 0:
        raise ValueError("Drift binning produced an empty split. Relax the quantiles.")

    return SelectedPairs(
        window_idx=window_idx,
        dim_idx=dim_idx,
        drift_scores=pair_scores.astype(np.float32),
        selected_dims=selected_dims.astype(np.int64),
        per_dim_tail_score=per_dim_tail_score.astype(np.float32),
        high_pair_indices=high_pair_indices,
        low_pair_indices=low_pair_indices,
        mid_pair_indices=mid_pair_indices,
        high_drift_threshold=high_drift_threshold,
        low_drift_threshold=low_threshold,
        mid_drift_threshold=high_threshold,
    )


def _extract_pair_samples(
    samples: np.ndarray,
    y: np.ndarray,
    window_idx: np.ndarray,
    dim_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pair_samples = samples[window_idx, :, dim_idx, :]
    pair_truth = y[window_idx, :, dim_idx]
    pair_samples = np.transpose(pair_samples, (0, 2, 1))
    return pair_samples.astype(np.float32), pair_truth.astype(np.float32)


def _energy_score(
    sample_cloud: np.ndarray,
    truth: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    if sample_cloud.ndim != 3:
        raise ValueError("sample_cloud must have shape [M, S, K].")
    if truth.ndim != 2:
        raise ValueError("truth must have shape [M, K].")
    if sample_cloud.shape[0] != truth.shape[0]:
        raise ValueError("sample_cloud and truth must share the same leading dimension.")

    num_items = sample_cloud.shape[0]
    scores = np.empty((num_items,), dtype=np.float32)

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        cloud = sample_cloud[start:end].astype(np.float32, copy=False)
        obs = truth[start:end].astype(np.float32, copy=False)

        diff_obs = cloud - obs[:, None, :]
        first_term = np.linalg.norm(diff_obs, axis=-1).mean(axis=1)

        sq_norm = np.sum(cloud * cloud, axis=-1, keepdims=True)
        gram = np.matmul(cloud, np.transpose(cloud, (0, 2, 1)))
        sq_dist = sq_norm + np.transpose(sq_norm, (0, 2, 1)) - 2.0 * gram
        np.maximum(sq_dist, 0.0, out=sq_dist)
        np.sqrt(sq_dist, out=sq_dist)
        pair_term = sq_dist.mean(axis=(1, 2))
        scores[start:end] = first_term - 0.5 * pair_term

    return scores


def _obs_distance_score(
    sample_cloud: np.ndarray,
    truth: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    if sample_cloud.ndim != 3:
        raise ValueError("sample_cloud must have shape [M, S, K].")
    if truth.ndim != 2:
        raise ValueError("truth must have shape [M, K].")
    if sample_cloud.shape[0] != truth.shape[0]:
        raise ValueError("sample_cloud and truth must share the same leading dimension.")

    num_items = sample_cloud.shape[0]
    scores = np.empty((num_items,), dtype=np.float32)

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        cloud = sample_cloud[start:end].astype(np.float32, copy=False)
        obs = truth[start:end].astype(np.float32, copy=False)
        diff_obs = cloud - obs[:, None, :]
        scores[start:end] = np.linalg.norm(diff_obs, axis=-1).mean(axis=1)

    return scores


def _compute_method_scores(
    artifact: MethodArtifact,
    selected: SelectedPairs,
    eps: float,
    batch_size: int,
) -> MethodScores:
    pair_samples, pair_truth = _extract_pair_samples(
        samples=artifact.samples,
        y=artifact.y,
        window_idx=selected.window_idx,
        dim_idx=selected.dim_idx,
    )

    sample_mean = np.mean(pair_samples, axis=-1)
    sample_std = np.maximum(np.std(pair_samples, axis=-1), eps)
    truth_mean = np.mean(pair_truth, axis=-1)
    truth_std = np.maximum(np.std(pair_truth, axis=-1), eps)

    macro_samples = np.stack(
        [sample_mean, np.log(sample_std + eps)],
        axis=-1,
    )
    macro_truth = np.stack(
        [truth_mean, np.log(truth_std + eps)],
        axis=-1,
    )

    sample_residual = (pair_samples - sample_mean[:, :, None]) / sample_std[:, :, None]
    truth_residual = (pair_truth - truth_mean[:, None]) / truth_std[:, None]

    macro_all = _energy_score(macro_samples, macro_truth, batch_size=batch_size)
    micro_all = _obs_distance_score(sample_residual, truth_residual, batch_size=batch_size)
    low_idx = selected.low_pair_indices
    mid_idx = selected.mid_pair_indices

    centroids = {
        "low": (
            float(np.mean(macro_all[low_idx])),
            float(np.mean(micro_all[low_idx])),
        ),
        "mid": (
            float(np.mean(macro_all[mid_idx])),
            float(np.mean(micro_all[mid_idx])),
        ),
    }

    return MethodScores(
        spec=artifact.spec,
        macro_all=macro_all,
        micro_all=micro_all,
        macro_mid=macro_all[mid_idx],
        micro_mid=micro_all[mid_idx],
        centroids=centroids,
    )


def _compute_centroid_axis_limits(
    method_scores: Sequence[MethodScores],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = []
    ys = []
    for item in method_scores:
        for centroid in item.centroids.values():
            xs.append(np.asarray([centroid[0]], dtype=np.float32))
            ys.append(np.asarray([centroid[1]], dtype=np.float32))

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    def _limits(values: np.ndarray) -> Tuple[float, float]:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if math.isclose(vmin, vmax):
            pad = 0.05 * max(abs(vmin), 1.0)
            return vmin - pad, vmax + pad
        pad = 0.06 * (vmax - vmin)
        return vmin - pad, vmax + pad

    return _limits(x_all), _limits(y_all)


def _kde_thresholds(z: np.ndarray, masses: Sequence[float]) -> List[float]:
    flat = np.sort(z.ravel())[::-1]
    cdf = np.cumsum(flat)
    cdf = cdf / max(float(cdf[-1]), 1e-12)
    thresholds = []
    for mass in masses:
        idx = min(np.searchsorted(cdf, mass, side="left"), flat.size - 1)
        thresholds.append(float(flat[idx]))
    thresholds = sorted(set(thresholds))
    return thresholds


def _draw_density_contours(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if gaussian_kde is None:
        return
    if x.size < 16:
        return
    try:
        values = np.vstack([x, y])
        kde = gaussian_kde(values)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        xpad = 0.08 * max(xmax - xmin, 1e-6)
        ypad = 0.08 * max(ymax - ymin, 1e-6)
        grid_x, grid_y = np.meshgrid(
            np.linspace(xmin - xpad, xmax + xpad, 180),
            np.linspace(ymin - ypad, ymax + ypad, 180),
        )
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        z = kde(positions).reshape(grid_x.shape)
        levels = _kde_thresholds(z, masses=[0.8, 0.5])
        if len(levels) >= 1:
            ax.contour(
                grid_x,
                grid_y,
                z,
                levels=levels,
                colors=[color],
                linewidths=1.2,
                alpha=0.9,
            )
    except Exception:
        return


def _add_better_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "better",
        xy=(0.06, 0.08),
        xytext=(0.20, 0.22),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=13.5,
        color="#666666",
        arrowprops=dict(arrowstyle="->", color="#B0B0B0", lw=2.1),
    )


def _plot_centroid_shift(
    ax: plt.Axes,
    method_scores: Sequence[MethodScores],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    for item in method_scores:
        xs = [
            item.centroids["low"][0],
            item.centroids["mid"][0],
        ]
        ys = [
            item.centroids["low"][1],
            item.centroids["mid"][1],
        ]
        ax.plot(xs, ys, color=item.spec.color, lw=2.3, alpha=0.95, zorder=2)
        ax.annotate(
            "",
            xy=(xs[1], ys[1]),
            xytext=(xs[0], ys[0]),
            arrowprops=dict(arrowstyle="->", color=item.spec.color, lw=2.1, alpha=0.95),
        )
        ax.scatter(
            [xs[0]],
            [ys[0]],
            s=92,
            facecolors="none",
            edgecolors=item.spec.color,
            marker=item.spec.marker,
            linewidths=1.9,
            zorder=3,
        )
        ax.scatter(
            [xs[1]],
            [ys[1]],
            s=235 if item.spec.marker != "*" else 300,
            color=item.spec.color,
            marker=item.spec.marker,
            edgecolors="black",
            linewidths=1.15,
            zorder=5,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Macro distribution error")
    ax.set_ylabel("Micro distribution error")
    ax.tick_params(axis="both", which="major", width=1.15, length=5.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("#4A4A4A")
    _add_better_arrow(ax)


def _build_legend(method_scores: Sequence[MethodScores]) -> List[Line2D]:
    handles = []
    for item in method_scores:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=item.spec.marker,
                color=item.spec.color,
                markerfacecolor=item.spec.color,
                markeredgecolor="black" if item.spec.marker == "*" else item.spec.color,
                markersize=11.5 if item.spec.marker != "*" else 13.5,
                linewidth=2.3,
                label=item.spec.display_name,
            )
        )
    return handles


def _reorder_legend_handles(handles: Sequence[Line2D]) -> List[Line2D]:
    handle_map = {str(handle.get_label()): handle for handle in handles}
    preferred_order = [
        "TimeGrad",
        "TMDM",
        "TimeDiff",
        "NsDiff",
        "CSDI",
        "PDN-Flow",
    ]
    reordered = [handle_map[name] for name in preferred_order if name in handle_map]
    seen = {handle.get_label() for handle in reordered}
    reordered.extend(handle for handle in handles if handle.get_label() not in seen)
    return reordered


def _scores_to_metadata(
    dataset_name: str,
    selected: SelectedPairs,
    method_scores: Sequence[MethodScores],
) -> Dict[str, Any]:
    metadata = {
        "dataset_name": dataset_name,
        "selected_feature_dims": selected.selected_dims.tolist(),
        "num_pairs": int(selected.drift_scores.shape[0]),
        "num_low_pairs": int(selected.low_pair_indices.shape[0]),
        "num_mid_pairs": int(selected.mid_pair_indices.shape[0]),
        "low_drift_threshold": float(selected.low_drift_threshold),
        "mid_drift_threshold": float(selected.mid_drift_threshold),
        "methods": {},
    }
    for item in method_scores:
        metadata["methods"][item.spec.display_name] = {
            "path": item.spec.path,
            "family": item.spec.family,
            "mid_drift_centroid": {
                "macro": float(np.mean(item.macro_mid)),
                "micro": float(np.mean(item.micro_mid)),
            },
            "centroids": {
                key: {"macro": float(value[0]), "micro": float(value[1])}
                for key, value in item.centroids.items()
            },
        }
    return metadata


def run_analysis(
    manifest_path: str,
    output_path: str,
    metadata_path: Optional[str],
    selection_path: Optional[str],
    dataset_name_override: Optional[str],
    num_variables: int,
    variable_tail_quantile: float,
    high_drift_ratio: float,
    low_drift_quantile: float,
    high_drift_quantile: float,
    feature_dims: Optional[Sequence[int]],
    eps: float,
    energy_batch_size: int,
) -> Dict[str, Any]:
    _configure_style()
    dataset_name, methods = _load_manifest(manifest_path)
    if dataset_name_override:
        dataset_name = dataset_name_override

    artifacts = _load_artifacts(methods)
    reference = artifacts[0]
    if selection_path is not None:
        selected = _load_selected_pairs_from_file(selection_path=selection_path, reference=reference)
    else:
        selected = _select_pairs(
            y=reference.y,
            mu_x=reference.mu_x,
            sigma_x=reference.sigma_x,
            num_variables=num_variables,
            variable_tail_quantile=variable_tail_quantile,
            high_drift_ratio=high_drift_ratio,
            low_drift_quantile=low_drift_quantile,
            high_drift_quantile=high_drift_quantile,
            eps=eps,
            feature_dims=feature_dims,
        )

    scores = []
    for artifact in artifacts:
        scores.append(
            _compute_method_scores(
                artifact=artifact,
                selected=selected,
                eps=eps,
                batch_size=energy_batch_size,
            )
        )

    xlim, ylim = _compute_centroid_axis_limits(scores)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.0), constrained_layout=False)
    _plot_centroid_shift(
        ax=ax,
        method_scores=scores,
        xlim=xlim,
        ylim=ylim,
    )

    handles = _reorder_legend_handles(_build_legend(scores))
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=3,
        bbox_to_anchor=(0.985, 0.03),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#D0D0D0",
        facecolor="white",
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.9,
        handletextpad=0.55,
        columnspacing=1.0,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = _scores_to_metadata(dataset_name=dataset_name, selected=selected, method_scores=scores)
    if metadata_path:
        metadata_dir = os.path.dirname(metadata_path)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Probabilistic Decoupling Map and the Drift-Conditioned "
            "Centroid Shift figure from aligned per-method forecast-sample artifacts."
        )
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--selection_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_variables", type=int, default=10)
    parser.add_argument("--variable_tail_quantile", type=float, default=0.90)
    parser.add_argument("--high_drift_ratio", type=float, default=0.20)
    parser.add_argument("--low_drift_quantile", type=float, default=0.30)
    parser.add_argument("--high_drift_quantile", type=float, default=0.70)
    parser.add_argument("--feature_dims", type=int, nargs="*", default=None)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--energy_batch_size", type=int, default=128)

    args = parser.parse_args()
    run_analysis(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        selection_path=args.selection_path,
        dataset_name_override=args.dataset_name,
        num_variables=args.num_variables,
        variable_tail_quantile=args.variable_tail_quantile,
        high_drift_ratio=args.high_drift_ratio,
        low_drift_quantile=args.low_drift_quantile,
        high_drift_quantile=args.high_drift_quantile,
        feature_dims=args.feature_dims,
        eps=args.eps,
        energy_batch_size=args.energy_batch_size,
    )


if __name__ == "__main__":
    main()
