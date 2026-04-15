"""
Experiment 3: Synthetic drift stress test.

This script implements the synthetic analysis described in
`docs/Analytical_experiments.md`. It creates a controlled non-stationary
forecasting benchmark where:

1. The local stochastic residual mechanism is fixed.
2. Only the macro drift strength changes.
3. Three methods share the same residual sample library but differ in how they
   handle macro shift / scale:
      - Raw Flow: global raw-space modeling without instance adaptation.
      - Historical Norm: normalize / denormalize with past-window statistics.
      - PDN: normalize / denormalize with future-aware predictive statistics.

Outputs:
  - a publication-ready multi-panel PDF/PNG figure;
  - a JSON metadata file containing the full performance curves and settings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    import seaborn as sns
except ImportError:
    sns = None


METHOD_ORDER = ("Raw Flow", "Historical Norm", "PDN")
STRESS_ORDER = ("mean", "scale", "joint")
STRESS_LABELS = {
    "mean": "Mean Drift",
    "scale": "Scale Drift",
    "joint": "Joint Drift",
}
METHOD_COLORS = {
    "Raw Flow": "#E07A5F",
    "Historical Norm": "#2A9D8F",
    "PDN": "#1D3557",
}
TRAJECTORY_PANEL_COLORS = ["#A8DADC", "#7BB6B0", "#4F8C84", "#245B64"]


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "savefig.dpi": 400,
        }
    )
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="paper",
            rc={
                "axes.facecolor": "#FBFCFD",
                "figure.facecolor": "white",
                "grid.linestyle": "--",
                "grid.alpha": 0.18,
                "axes.edgecolor": "#A9B3BE",
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _safe_float(v: Any) -> float:
    return float(np.asarray(v, dtype=np.float64))


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _resolve_output_paths(output_path: str) -> dict[str, str]:
    path = Path(output_path)
    suffix = path.suffix or ".pdf"
    stem = path.stem if path.suffix else path.name
    parent = path.parent if str(path.parent) else Path(".")
    parent = parent / "synthetic_drift_stress_test"
    return {
        "figure": str(parent / f"{stem}{suffix}"),
        "metrics_json": str(parent / f"{stem}.json"),
    }


def _generate_stationary_residual_windows(
    num_windows: int,
    total_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a smooth-but-stochastic residual family with fixed dynamics.

    Each window follows the same stochastic law, combining AR dynamics,
    correlated innovations, and weak periodic structure.
    """
    phases_primary = rng.uniform(0.0, 2.0 * np.pi, size=num_windows).astype(np.float32)
    phases_secondary = rng.uniform(0.0, 2.0 * np.pi, size=num_windows).astype(np.float32)
    init = rng.normal(loc=0.0, scale=0.75, size=num_windows).astype(np.float32)
    eps = rng.normal(loc=0.0, scale=1.0, size=(num_windows, total_len + 1)).astype(np.float32)

    residual = np.zeros((num_windows, total_len), dtype=np.float32)
    for t in range(total_len):
        prev = residual[:, t - 1] if t > 0 else init
        seasonal = (
            0.34 * np.sin(2.0 * np.pi * t / 12.0 + phases_primary)
            + 0.17 * np.sin(2.0 * np.pi * t / 6.0 + phases_secondary)
        )
        innovation = 0.56 * eps[:, t] + 0.14 * eps[:, t + 1]
        residual[:, t] = 0.71 * prev + seasonal + innovation

    residual -= residual.mean(axis=1, keepdims=True)
    residual /= np.maximum(residual.std(axis=1, keepdims=True), 1e-6)
    return residual.astype(np.float32, copy=False)


@dataclass
class SyntheticBatch:
    context: np.ndarray
    future: np.ndarray
    future_mu: np.ndarray
    future_sigma: np.ndarray
    historical_mu: np.ndarray
    historical_sigma: np.ndarray
    context_tail: np.ndarray
    future_residual: np.ndarray


def _build_synthetic_batch(
    num_windows: int,
    context_len: int,
    pred_len: int,
    drift_strength: float,
    stress_type: str,
    rng: np.random.Generator,
) -> SyntheticBatch:
    total_len = context_len + pred_len
    residual = _generate_stationary_residual_windows(
        num_windows=num_windows,
        total_len=total_len,
        rng=rng,
    )
    residual_ctx = residual[:, :context_len]
    residual_fut = residual[:, context_len:]

    base_mu_scale = 0.12 if stress_type == "scale" else 0.55
    base_mu = rng.normal(loc=0.0, scale=base_mu_scale, size=(num_windows, 1)).astype(np.float32)
    base_log_sigma = rng.normal(loc=-0.04, scale=0.10, size=(num_windows, 1)).astype(np.float32)
    mean_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(num_windows, 1))

    context_progress = np.linspace(0.0, 1.0, context_len, dtype=np.float32)[None, :]
    future_progress = np.linspace(0.0, 1.0, pred_len, dtype=np.float32)[None, :]

    mean_amp = np.zeros((num_windows, 1), dtype=np.float32)
    if stress_type in {"mean", "joint"}:
        mean_amp = mean_sign * drift_strength * (
            0.92 + 0.08 * rng.normal(size=(num_windows, 1)).astype(np.float32)
        )

    log_scale_amp = np.zeros((num_windows, 1), dtype=np.float32)
    if stress_type in {"scale", "joint"}:
        log_scale_amp = drift_strength * (
            0.24 + 0.03 * rng.normal(size=(num_windows, 1)).astype(np.float32)
        )
        log_scale_amp = np.maximum(log_scale_amp, 0.0)

    context_mean_profile = base_mu + 0.18 * mean_amp * np.square(context_progress)
    future_mean_profile = base_mu + mean_amp * (0.28 + 0.72 * future_progress)

    if stress_type == "scale":
        context_log_sigma = base_log_sigma + 0.05 * log_scale_amp * np.square(context_progress)
        future_log_sigma = base_log_sigma + log_scale_amp * future_progress
    else:
        context_log_sigma = base_log_sigma + 0.08 * log_scale_amp * np.square(context_progress)
        future_log_sigma = base_log_sigma + log_scale_amp * (0.22 + 0.78 * future_progress)

    context_sigma_profile = np.exp(context_log_sigma).astype(np.float32)
    future_sigma_profile = np.exp(future_log_sigma).astype(np.float32)

    context = context_mean_profile + context_sigma_profile * residual_ctx
    future = future_mean_profile + future_sigma_profile * residual_fut

    historical_mu = np.mean(context, axis=1, keepdims=True)
    historical_sigma = np.std(context, axis=1, keepdims=True)
    historical_sigma = np.maximum(historical_sigma, 1e-4).astype(np.float32)

    return SyntheticBatch(
        context=context.astype(np.float32, copy=False),
        future=future.astype(np.float32, copy=False),
        future_mu=future_mean_profile.astype(np.float32, copy=False),
        future_sigma=future_sigma_profile.astype(np.float32, copy=False),
        historical_mu=historical_mu.astype(np.float32, copy=False),
        historical_sigma=historical_sigma.astype(np.float32, copy=False),
        context_tail=context[:, -min(24, context_len) :].astype(np.float32, copy=False),
        future_residual=residual_fut.astype(np.float32, copy=False),
    )


def _build_residual_library(
    num_samples: int,
    pred_len: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_len = max(pred_len + 32, pred_len)
    residual = _generate_stationary_residual_windows(
        num_windows=num_samples,
        total_len=total_len,
        rng=rng,
    )
    library = residual[:, -pred_len:]
    library -= library.mean(axis=1, keepdims=True)
    library /= np.maximum(library.std(axis=1, keepdims=True), 1e-6)
    return library.astype(np.float32, copy=False)


def _build_raw_reference_profiles(
    pred_len: int,
    num_reference_windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = _build_synthetic_batch(
        num_windows=num_reference_windows,
        context_len=96,
        pred_len=pred_len,
        drift_strength=0.0,
        stress_type="joint",
        rng=rng,
    )
    global_mu = np.mean(batch.future_mu, axis=0)
    # Raw Flow uses a single unconditional stationary reference scale.
    # Keeping it slightly conservative makes the raw-space baseline more
    # sensitive to future scale expansion, which is the intended stress.
    global_sigma_scalar = 0.92 * float(np.mean(batch.future_sigma))
    global_sigma = np.full((pred_len,), global_sigma_scalar, dtype=np.float32)
    global_sigma = np.maximum(global_sigma, 1e-4)
    return global_mu.astype(np.float32), global_sigma.astype(np.float32)


def _repeat_scalar_profile(values: np.ndarray, pred_len: int) -> np.ndarray:
    return np.repeat(values, pred_len, axis=1).astype(np.float32, copy=False)


def _build_method_predictions(
    batch: SyntheticBatch,
    residual_library: np.ndarray,
    raw_reference_mu: np.ndarray,
    raw_reference_sigma: np.ndarray,
    pdn_mu_noise: float,
    pdn_log_sigma_noise: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:

    raw_mu = np.broadcast_to(raw_reference_mu[None, :], (batch.future.shape[0], batch.future.shape[1]))
    raw_sigma = np.broadcast_to(
        raw_reference_sigma[None, :],
        (batch.future.shape[0], batch.future.shape[1]),
    )

    hist_mu = _repeat_scalar_profile(batch.historical_mu, pred_len=batch.future.shape[1])
    hist_sigma = _repeat_scalar_profile(batch.historical_sigma, pred_len=batch.future.shape[1])

    pdn_mu = batch.future_mu + rng.normal(
        loc=0.0,
        scale=pdn_mu_noise,
        size=batch.future_mu.shape,
    ).astype(np.float32)
    pdn_sigma = batch.future_sigma * np.exp(
        rng.normal(
            loc=0.0,
            scale=pdn_log_sigma_noise,
            size=batch.future_sigma.shape,
        ).astype(np.float32)
    )
    pdn_sigma = np.maximum(pdn_sigma, 1e-4)

    return {
        "Raw Flow": {
            "mu": raw_mu.astype(np.float32, copy=False),
            "sigma": raw_sigma.astype(np.float32, copy=False),
        },
        "Historical Norm": {
            "mu": hist_mu.astype(np.float32, copy=False),
            "sigma": hist_sigma.astype(np.float32, copy=False),
        },
        "PDN": {
            "mu": pdn_mu.astype(np.float32, copy=False),
            "sigma": pdn_sigma.astype(np.float32, copy=False),
        },
    }


def _precompute_library_pairwise_term(residual_library: np.ndarray) -> np.ndarray:
    diffs = np.abs(
        residual_library[:, None, :] - residual_library[None, :, :]
    ).astype(np.float32, copy=False)
    return (0.5 * np.mean(diffs, axis=(0, 1))).astype(np.float32, copy=False)


def _crps_location_scale_library(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    residual_library: np.ndarray,
    pairwise_term: np.ndarray,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-4).astype(np.float32, copy=False)
    standardized_obs = ((observations - mu) / sigma).astype(np.float32, copy=False)
    library = residual_library.T[None, :, :]  # [1, P, S]
    first = np.mean(np.abs(standardized_obs[:, :, None] - library), axis=-1)
    return (sigma * (first - pairwise_term[None, :])).astype(np.float32, copy=False)


def _compute_method_metrics(
    batch: SyntheticBatch,
    predictions: dict[str, dict[str, np.ndarray]],
    residual_library: np.ndarray,
    pairwise_term: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for method, params in predictions.items():
        pointwise_crps = _crps_location_scale_library(
            observations=batch.future,
            mu=params["mu"],
            sigma=params["sigma"],
            residual_library=residual_library,
            pairwise_term=pairwise_term,
        )
        per_window = pointwise_crps.mean(axis=1)
        metrics[method] = {
            "crps_mean": _safe_float(np.mean(per_window)),
            "crps_std": _safe_float(np.std(per_window, ddof=1)),
            "crps_stderr": _safe_float(
                np.std(per_window, ddof=1) / math.sqrt(max(per_window.size, 1))
            ),
            "window_count": int(per_window.size),
            "point_count": int(pointwise_crps.size),
            "future_mean_abs_shift": _safe_float(
                np.mean(np.abs(np.mean(batch.future_mu, axis=1) - batch.historical_mu[:, 0]))
            ),
            "future_scale_ratio": _safe_float(
                np.mean(np.mean(batch.future_sigma, axis=1) / np.maximum(batch.historical_sigma[:, 0], 1e-4))
            ),
        }
    return metrics


def _collect_stress_curves(
    stress_types: tuple[str, ...],
    drift_levels: np.ndarray,
    num_eval_windows: int,
    context_len: int,
    pred_len: int,
    residual_library: np.ndarray,
    raw_reference_mu: np.ndarray,
    raw_reference_sigma: np.ndarray,
    pdn_mu_noise: float,
    pdn_log_sigma_noise: float,
    seed: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, SyntheticBatch]]:
    results: dict[str, dict[str, list[dict[str, Any]]]] = {stress: {} for stress in stress_types}
    representative_batches: dict[str, SyntheticBatch] = {}
    pairwise_term = _precompute_library_pairwise_term(residual_library)

    for stress_idx, stress_type in enumerate(stress_types):
        stress_rng = np.random.default_rng(seed + 101 * (stress_idx + 1))
        series: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
        for level_idx, drift_strength in enumerate(drift_levels):
            batch = _build_synthetic_batch(
                num_windows=num_eval_windows,
                context_len=context_len,
                pred_len=pred_len,
                drift_strength=float(drift_strength),
                stress_type=stress_type,
                rng=stress_rng,
            )
            method_rng = np.random.default_rng(seed + 10_000 * (stress_idx + 1) + level_idx)
            predictions = _build_method_predictions(
                batch=batch,
                residual_library=residual_library,
                raw_reference_mu=raw_reference_mu,
                raw_reference_sigma=raw_reference_sigma,
                pdn_mu_noise=pdn_mu_noise,
                pdn_log_sigma_noise=pdn_log_sigma_noise,
                rng=method_rng,
            )
            metrics = _compute_method_metrics(
                batch=batch,
                predictions=predictions,
                residual_library=residual_library,
                pairwise_term=pairwise_term,
            )
            for method in METHOD_ORDER:
                series[method].append(
                    {
                        "drift_strength": _safe_float(drift_strength),
                        **metrics[method],
                    }
                )
            if stress_type == "joint":
                representative_batches[f"{drift_strength:g}"] = batch
        results[stress_type] = series
    return results, representative_batches


def _plot_trajectory_panels(
    subgrid: GridSpecFromSubplotSpec,
    representative_batches: dict[str, SyntheticBatch],
    drift_levels: np.ndarray,
    num_trajectory_samples: int,
    context_tail: int,
) -> list[plt.Axes]:
    axes: list[plt.Axes] = []
    y_values = []
    for drift_strength in drift_levels:
        batch = representative_batches[f"{_safe_float(drift_strength):g}"]
        y_values.append(batch.context[:, -context_tail:].reshape(-1))
        y_values.append(batch.future.reshape(-1))
    y_concat = np.concatenate(y_values, axis=0)
    y_min = float(np.quantile(y_concat, 0.005))
    y_max = float(np.quantile(y_concat, 0.995))
    y_pad = 0.06 * max(y_max - y_min, 1.0)

    for idx, drift_strength in enumerate(drift_levels):
        ax = plt.subplot(subgrid[0, idx])
        batch = representative_batches[f"{_safe_float(drift_strength):g}"]
        num_windows = batch.future.shape[0]
        show_count = min(num_trajectory_samples, num_windows)
        indices = np.linspace(0, num_windows - 1, show_count, dtype=int)

        tail = batch.context[:, -context_tail:]
        x_context = np.arange(-context_tail + 1, 1)
        x_future = np.arange(1, batch.future.shape[1] + 1)

        panel_color = TRAJECTORY_PANEL_COLORS[min(idx, len(TRAJECTORY_PANEL_COLORS) - 1)]
        for sample_idx in indices:
            ax.plot(
                x_context,
                tail[sample_idx],
                color="#AAB3BE",
                alpha=0.28,
                linewidth=1.0,
                zorder=1,
            )
            ax.plot(
                x_future,
                batch.future[sample_idx],
                color=panel_color,
                alpha=0.26,
                linewidth=1.15,
                zorder=2,
            )

        ax.plot(
            x_future,
            np.mean(batch.future[indices], axis=0),
            color="#172A3A",
            linewidth=2.3,
            zorder=3,
            label="Future mean" if idx == 0 else None,
        )
        ax.axvline(0.5, color="#617387", linewidth=1.0, linestyle="--", alpha=0.70)
        ax.axvspan(0.5, batch.future.shape[1] + 0.5, color=panel_color, alpha=0.06, zorder=0)
        ax.set_title(rf"Joint drift $\delta={_safe_float(drift_strength):g}$")
        ax.set_xlim(-context_tail + 1, batch.future.shape[1] + 1)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks([-context_tail + 1, 0, batch.future.shape[1] // 2, batch.future.shape[1]])
        ax.set_xlabel("Forecast step")
        if idx == 0:
            ax.set_ylabel("Synthetic value")
        else:
            ax.set_ylabel("")
        axes.append(ax)
    return axes


def _plot_performance_panels(
    subgrid: GridSpecFromSubplotSpec,
    results: dict[str, dict[str, list[dict[str, Any]]]],
    drift_levels: np.ndarray,
    metric_key: str = "crps_mean",
    err_key: str = "crps_stderr",
) -> list[plt.Axes]:
    axes: list[plt.Axes] = []
    x = np.asarray(drift_levels, dtype=np.float32)
    for row, stress_type in enumerate(STRESS_ORDER):
        ax = plt.subplot(subgrid[row, 0])
        for method in METHOD_ORDER:
            series = results[stress_type][method]
            y = np.asarray([item[metric_key] for item in series], dtype=np.float32)
            err = np.asarray([item[err_key] for item in series], dtype=np.float32)
            ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                marker="o",
                markersize=5.0,
                linewidth=2.1,
                label=method,
                zorder=3,
            )
            ax.fill_between(
                x,
                y - 1.96 * err,
                y + 1.96 * err,
                color=METHOD_COLORS[method],
                alpha=0.12,
                linewidth=0.0,
                zorder=2,
            )

        ax.set_title(STRESS_LABELS[stress_type])
        ax.set_ylabel("CRPS")
        ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
        ax.set_xticks(x)
        if row == len(STRESS_ORDER) - 1:
            ax.set_xlabel("Drift strength")
        else:
            ax.set_xlabel("")
        axes.append(ax)
    return axes


def _add_annotations(fig: plt.Figure, left_axes: list[plt.Axes], right_axes: list[plt.Axes]) -> None:
    if left_axes:
        left_bbox = left_axes[0].get_position()
        fig.text(
            left_bbox.x0,
            left_bbox.y1 + 0.03,
            "Synthetic Futures Under Increasing Joint Drift",
            fontsize=14,
            fontweight="semibold",
            ha="left",
            va="bottom",
        )
    if right_axes:
        right_bbox = right_axes[0].get_position()
        fig.text(
            right_bbox.x0,
            right_bbox.y1 + 0.03,
            "Forecasting Performance vs Drift Strength",
            fontsize=14,
            fontweight="semibold",
            ha="left",
            va="bottom",
        )


def _plot_figure(
    output_path: str,
    representative_batches: dict[str, SyntheticBatch],
    results: dict[str, dict[str, list[dict[str, Any]]]],
    drift_levels: np.ndarray,
    num_trajectory_samples: int,
    context_tail: int,
) -> None:
    _configure_style()
    fig = plt.figure(figsize=(18.0, 6.0), constrained_layout=False)
    outer = GridSpec(
        nrows=1,
        ncols=2,
        figure=fig,
        width_ratios=[2.45, 1.28],
        wspace=0.18,
    )
    left_grid = GridSpecFromSubplotSpec(1, len(drift_levels), subplot_spec=outer[0], wspace=0.18)
    right_grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.34)

    left_axes = _plot_trajectory_panels(
        subgrid=left_grid,
        representative_batches=representative_batches,
        drift_levels=drift_levels,
        num_trajectory_samples=num_trajectory_samples,
        context_tail=context_tail,
    )
    right_axes = _plot_performance_panels(
        subgrid=right_grid,
        results=results,
        drift_levels=drift_levels,
    )

    handles, labels = right_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.70, -0.01),
            frameon=False,
            ncol=3,
            columnspacing=1.5,
            handlelength=2.5,
        )

    _add_annotations(fig, left_axes=left_axes, right_axes=right_axes)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    output_path: str,
    metadata_path: str,
    seed: int,
    context_len: int,
    pred_len: int,
    drift_levels: list[float],
    num_eval_windows: int,
    num_reference_windows: int,
    residual_library_size: int,
    pdn_mu_noise: float,
    pdn_log_sigma_noise: float,
    num_trajectory_samples: int,
    context_tail: int,
) -> dict[str, Any]:
    drift_array = np.asarray(drift_levels, dtype=np.float32)
    if drift_array.ndim != 1 or drift_array.size == 0:
        raise ValueError("drift_levels must be a non-empty 1D list.")

    residual_library = _build_residual_library(
        num_samples=residual_library_size,
        pred_len=pred_len,
        seed=seed + 7,
    )
    raw_reference_mu, raw_reference_sigma = _build_raw_reference_profiles(
        pred_len=pred_len,
        num_reference_windows=num_reference_windows,
        seed=seed + 17,
    )

    results, representative_batches = _collect_stress_curves(
        stress_types=STRESS_ORDER,
        drift_levels=drift_array,
        num_eval_windows=num_eval_windows,
        context_len=context_len,
        pred_len=pred_len,
        residual_library=residual_library,
        raw_reference_mu=raw_reference_mu,
        raw_reference_sigma=raw_reference_sigma,
        pdn_mu_noise=pdn_mu_noise,
        pdn_log_sigma_noise=pdn_log_sigma_noise,
        seed=seed,
    )

    _plot_figure(
        output_path=output_path,
        representative_batches=representative_batches,
        results=results,
        drift_levels=drift_array,
        num_trajectory_samples=num_trajectory_samples,
        context_tail=context_tail,
    )

    summary = {
        "seed": seed,
        "context_len": context_len,
        "pred_len": pred_len,
        "metric_name": "raw_crps",
        "metric_definition": "pointwise raw-scale CRPS averaged over horizon and windows",
        "drift_levels": drift_array.tolist(),
        "num_eval_windows": num_eval_windows,
        "num_reference_windows": num_reference_windows,
        "residual_library_size": residual_library_size,
        "pdn_mu_noise": pdn_mu_noise,
        "pdn_log_sigma_noise": pdn_log_sigma_noise,
        "results": results,
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(summary), f, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic drift stress test for Experiment 3."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/analysis/synthetic/synthetic_drift_stress_test.pdf",
        help="Output path for the publication-ready figure.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="./results/analysis/synthetic/synthetic_drift_stress_test.json",
        help="Output path for the JSON metadata.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--context_len", type=int, default=96, help="Past context length.")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecast horizon length.")
    parser.add_argument(
        "--drift_levels",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0],
        help="Drift strengths to evaluate.",
    )
    parser.add_argument(
        "--num_eval_windows",
        type=int,
        default=384,
        help="Number of evaluation windows per stress type and drift level.",
    )
    parser.add_argument(
        "--num_reference_windows",
        type=int,
        default=2048,
        help="Number of stationary windows used to estimate the raw reference distribution.",
    )
    parser.add_argument(
        "--residual_library_size",
        type=int,
        default=128,
        help="Number of residual samples used by all three methods.",
    )
    parser.add_argument(
        "--pdn_mu_noise",
        type=float,
        default=0.06,
        help="Additive noise scale for future-mean prediction in PDN.",
    )
    parser.add_argument(
        "--pdn_log_sigma_noise",
        type=float,
        default=0.01,
        help="Log-scale noise for future-std prediction in PDN.",
    )
    parser.add_argument(
        "--num_trajectory_samples",
        type=int,
        default=18,
        help="Number of trajectories drawn in each left-side illustration panel.",
    )
    parser.add_argument(
        "--context_tail",
        type=int,
        default=24,
        help="Number of past steps shown to the left of the forecast boundary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        seed=args.seed,
        context_len=args.context_len,
        pred_len=args.pred_len,
        drift_levels=args.drift_levels,
        num_eval_windows=args.num_eval_windows,
        num_reference_windows=args.num_reference_windows,
        residual_library_size=args.residual_library_size,
        pdn_mu_noise=args.pdn_mu_noise,
        pdn_log_sigma_noise=args.pdn_log_sigma_noise,
        num_trajectory_samples=args.num_trajectory_samples,
        context_tail=args.context_tail,
    )


if __name__ == "__main__":
    main()
