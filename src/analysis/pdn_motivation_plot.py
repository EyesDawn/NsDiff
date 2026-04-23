"""
Create a schematic motivation figure for location-scale decomposition.

This figure is intentionally synthetic. It illustrates that:
  1. raw-space temporal segments can have different local probability
     distributions, hence non-stationary;
  2. after segment-wise location-scale standardization, the distributions
     become much better aligned in the standardized space.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None

from src.analysis.pdn_density_plot import _plot_single_on_ax


@dataclass
class SyntheticMotivationMetadata:
    seed: int
    segment_length: int
    num_segments: int
    num_density_samples: int
    raw_locations: list[float]
    raw_scales: list[float]
    standardized_mean_per_segment: list[float]
    standardized_std_per_segment: list[float]
    raw_mean_per_segment: list[float]
    raw_std_per_segment: list[float]


def _configure_style(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 17,
            "savefig.dpi": dpi,
        }
    )
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="paper",
            palette="colorblind",
            rc={
                "axes.facecolor": "#FBFBFC",
                "figure.facecolor": "white",
                "grid.linestyle": "--",
                "grid.alpha": 0.18,
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _sample_base_distribution(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    mixture_mask = rng.random(num_samples) < 0.70
    base = np.empty(num_samples, dtype=np.float64)
    base[mixture_mask] = rng.normal(loc=-0.35, scale=0.62, size=int(np.sum(mixture_mask)))
    base[~mixture_mask] = rng.normal(loc=1.10, scale=0.38, size=int(np.sum(~mixture_mask)))
    base = (base - np.mean(base)) / np.std(base)
    return base


def _smooth_latent_curve(length: int, phase_shift: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length, endpoint=False)
    curve = (
        0.90 * np.sin(2.0 * np.pi * (2.2 * x + phase_shift))
        + 0.30 * np.sin(2.0 * np.pi * (5.4 * x + 0.15 + 0.5 * phase_shift))
    )
    curve = (curve - np.mean(curve)) / np.std(curve)
    return curve


def _build_synthetic_motivation_data(
    seed: int,
    segment_length: int,
    num_density_samples: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], SyntheticMotivationMetadata]:
    rng = np.random.default_rng(seed)
    raw_locations = [-2.6, 0.2, 2.9]
    raw_scales = [0.55, 1.00, 1.55]
    num_segments = len(raw_locations)

    standardized_density_segments: list[np.ndarray] = []
    raw_density_segments: list[np.ndarray] = []
    raw_curve_segments: list[np.ndarray] = []

    for seg_idx, (loc, scale) in enumerate(zip(raw_locations, raw_scales)):
        z_density = _sample_base_distribution(num_density_samples, rng)
        y_density = loc + scale * z_density
        standardized_density_segments.append(z_density)
        raw_density_segments.append(y_density)

        z_curve = _smooth_latent_curve(segment_length, phase_shift=0.06 * seg_idx)
        y_curve = loc + scale * z_curve
        raw_curve_segments.append(y_curve)

    raw_curve = np.concatenate(raw_curve_segments, axis=0)
    metadata = SyntheticMotivationMetadata(
        seed=seed,
        segment_length=segment_length,
        num_segments=num_segments,
        num_density_samples=num_density_samples,
        raw_locations=[float(v) for v in raw_locations],
        raw_scales=[float(v) for v in raw_scales],
        standardized_mean_per_segment=[float(np.mean(v)) for v in standardized_density_segments],
        standardized_std_per_segment=[float(np.std(v)) for v in standardized_density_segments],
        raw_mean_per_segment=[float(np.mean(v)) for v in raw_density_segments],
        raw_std_per_segment=[float(np.std(v)) for v in raw_density_segments],
    )
    return raw_curve, raw_density_segments, standardized_density_segments, metadata


def _robust_clip(
    arrays: list[np.ndarray],
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in arrays], axis=0)
    x_min = float(np.quantile(flat, lower_quantile))
    x_max = float(np.quantile(flat, upper_quantile))
    if x_min == x_max:
        delta = max(abs(x_min) * 0.1, 1e-3)
        return x_min - delta, x_max + delta
    return x_min, x_max


def _plot_density_panel(
    ax: plt.Axes,
    data: np.ndarray,
    clip_range: tuple[float, float],
    bins: int,
    color: str,
    add_normal_ref: bool,
    stat_color: str,
) -> None:
    _plot_single_on_ax(
        ax=ax,
        data=data,
        clip_range=clip_range,
        bins=bins,
        color=color,
        label="distribution",
        add_normal_ref=add_normal_ref,
        title="",
        show_stats=False,
        kde_max_points=50000,
        rng=np.random.default_rng(0),
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xlabel("")
    ax.set_ylabel("")

    stats = np.asarray(data, dtype=np.float64).ravel()
    ax.text(
        0.03,
        0.95,
        rf"$\mu={np.mean(stats):.2f},\ \sigma={np.std(stats):.2f}$",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        color=stat_color,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.82),
    )


def _style_density_strip(
    axes: list[plt.Axes],
    edge_color: str,
    panel_face: str,
    separator_color: str,
) -> None:
    for idx, ax in enumerate(axes):
        ax.set_facecolor(panel_face)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(axis="y", linestyle="--", alpha=0.10)
        ax.grid(axis="x", visible=False)

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["top"].set_linewidth(1.25)
        ax.spines["bottom"].set_linewidth(1.25)
        ax.spines["top"].set_color(edge_color)
        ax.spines["bottom"].set_color(edge_color)

        if idx == 0:
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color(edge_color)
            ax.spines["left"].set_linewidth(1.45)
        else:
            ax.spines["left"].set_visible(False)

        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color(separator_color if idx < len(axes) - 1 else edge_color)
        ax.spines["right"].set_linewidth(1.15 if idx < len(axes) - 1 else 1.45)


def _add_row_transition_arrows(
    fig: plt.Figure,
    raw_axes: list[plt.Axes],
    std_axes: list[plt.Axes],
    arrow_color: str,
) -> None:
    for idx, (ax_raw, ax_std) in enumerate(zip(raw_axes, std_axes)):
        raw_bbox = ax_raw.get_position()
        std_bbox = ax_std.get_position()

        start_xy = (
            raw_bbox.x0 + 0.5 * raw_bbox.width,
            raw_bbox.y0 - 0.016,
        )
        end_xy = (
            std_bbox.x0 + 0.5 * std_bbox.width,
            std_bbox.y1 + 0.012,
        )
        fig.add_artist(
            FancyArrowPatch(
                start_xy,
                end_xy,
                transform=fig.transFigure,
                arrowstyle="simple",
                mutation_scale=22,
                linewidth=0.0,
                color=arrow_color,
                alpha=0.48,
            )
        )


def _add_row_label(fig: plt.Figure, axes: list[plt.Axes], text: str, color: str) -> None:
    bbox = axes[0].get_position()
    x = bbox.x0 - 0.05
    y = bbox.y0 + 0.5 * bbox.height
    fig.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=13.0,
        fontweight="semibold",
        color=color,
    )


def _save_metadata(metadata_path: str | None, metadata: SyntheticMotivationMetadata) -> None:
    if metadata_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
    print(f"[Metadata] Saved to: {metadata_path}")


def plot_schematic_motivation(
    output_path: str,
    metadata_path: str | None,
    seed: int,
    bins: int,
    dpi: int,
    segment_length: int,
    num_density_samples: int,
    clip_lower_quantile: float,
    clip_upper_quantile: float,
) -> None:
    _configure_style(dpi=dpi)

    _, raw_segments, standardized_segments, metadata = _build_synthetic_motivation_data(
        seed=seed,
        segment_length=segment_length,
        num_density_samples=num_density_samples,
    )

    segment_colors = ["#2E5EAA", "#C47A2C", "#2C8A72"]
    edge_color = "#2A3655"
    separator_color = "#A6B2C8"
    arrow_color = "#E7A2AC"
    panel_face = "#FCFCFE"
    raw_clip = _robust_clip(raw_segments, clip_lower_quantile, clip_upper_quantile)
    std_clip = _robust_clip(standardized_segments, clip_lower_quantile, clip_upper_quantile)

    fig = plt.figure(figsize=(14.0, 5.9), dpi=dpi)
    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.40,
        figure=fig,
    )

    raw_gs = gs[0, 0].subgridspec(1, 3, wspace=0.0)
    std_gs = gs[1, 0].subgridspec(1, 3, wspace=0.0)
    raw_axes: list[plt.Axes] = []
    std_axes: list[plt.Axes] = []
    for seg_idx in range(3):
        ax_raw = fig.add_subplot(raw_gs[0, seg_idx])
        _plot_density_panel(
            ax=ax_raw,
            data=raw_segments[seg_idx],
            clip_range=raw_clip,
            bins=bins,
            color=segment_colors[seg_idx],
            add_normal_ref=False,
            stat_color="#485466",
        )
        raw_axes.append(ax_raw)

        ax_std = fig.add_subplot(std_gs[0, seg_idx])
        _plot_density_panel(
            ax=ax_std,
            data=standardized_segments[seg_idx],
            clip_range=std_clip,
            bins=bins,
            color=segment_colors[seg_idx],
            add_normal_ref=True,
            stat_color="#485466",
        )
        std_axes.append(ax_std)

    raw_ymax = max(ax.get_ylim()[1] for ax in raw_axes)
    for ax in raw_axes:
        ax.set_ylim(0.0, raw_ymax * 1.06)

    std_ymax = max(ax.get_ylim()[1] for ax in std_axes)
    for ax in std_axes:
        ax.set_ylim(0.0, std_ymax * 1.06)
    _style_density_strip(raw_axes, edge_color=edge_color, panel_face=panel_face, separator_color=separator_color)
    _style_density_strip(std_axes, edge_color=edge_color, panel_face=panel_face, separator_color=separator_color)

    _add_row_label(fig=fig, axes=raw_axes, text="Raw space", color="#1F2937")
    _add_row_label(fig=fig, axes=std_axes, text="Standardized space", color="#1F2937")

    _add_row_transition_arrows(
        fig=fig,
        raw_axes=raw_axes,
        std_axes=std_axes,
        arrow_color=arrow_color,
    )

    if sns is not None:
        sns.despine(fig=fig, offset=2)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.96, bottom=0.085, hspace=0.40)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Figure] Saved to: {output_path}")

    _save_metadata(metadata_path=metadata_path, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a synthetic schematic figure for location-scale decomposition."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output figure (.png / .pdf).",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Optional JSON metadata path for reproducibility.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic sampling.")
    parser.add_argument("--bins", type=int, default=52, help="Histogram bin count.")
    parser.add_argument("--dpi", type=int, default=600, help="Figure dpi.")
    parser.add_argument(
        "--segment_length",
        type=int,
        default=160,
        help="Number of time steps per segment in the top schematic time series.",
    )
    parser.add_argument(
        "--num_density_samples",
        type=int,
        default=1800,
        help="Number of synthetic samples per segment used in density plots.",
    )
    parser.add_argument(
        "--clip_lower_quantile",
        type=float,
        default=0.005,
        help="Lower quantile used to set the density x-limits.",
    )
    parser.add_argument(
        "--clip_upper_quantile",
        type=float,
        default=0.995,
        help="Upper quantile used to set the density x-limits.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.clip_lower_quantile < args.clip_upper_quantile <= 1.0):
        raise ValueError(
            "clip quantiles must satisfy 0 <= lower < upper <= 1, got "
            f"{args.clip_lower_quantile} and {args.clip_upper_quantile}."
        )

    plot_schematic_motivation(
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        seed=args.seed,
        bins=args.bins,
        dpi=args.dpi,
        segment_length=args.segment_length,
        num_density_samples=args.num_density_samples,
        clip_lower_quantile=args.clip_lower_quantile,
        clip_upper_quantile=args.clip_upper_quantile,
    )


if __name__ == "__main__":
    main()
