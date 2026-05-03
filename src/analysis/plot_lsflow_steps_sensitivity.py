from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - seaborn is optional at runtime.
    sns = None


REQUIRED_COLUMNS = ("data", "num_sampling_steps", "test_crps", "test_crps_sum")
STEP_ORDER = (1, 2, 5, 10, 20, 50)
MAIN_EXPERIMENT_STEP = 5
DATASET_GROUPS = (
    ("ETTh1", "ETTh2", "ETTm1", "ETTm2"),
    ("Electricity", "SolarEnergy", "Traffic", "Weather"),
)
METRICS = {
    "crps": {"column": "crps", "label": "CRPS", "color": "#1F5A93", "marker": "s"},
    "crpssum": {
        "column": "crps_sum",
        "label": "CRPSsum",
        "color": "#B13A32",
        "marker": "o",
    },
}
DATASET_LABELS = {
    "Electricity": "ECL",
    "SolarEnergy": "Solar",
}


def configure_style(dpi: int) -> None:
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="paper",
            rc={
                "axes.facecolor": "#FCFCFF",
                "figure.facecolor": "white",
                "grid.color": "#BFC7D5",
                "grid.linestyle": "--",
                "grid.alpha": 0.24,
                "axes.edgecolor": "#2A3140",
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.05,
            "lines.linewidth": 2.15,
            "lines.markersize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
        }
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return slug or "dataset"


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def load_sensitivity_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_columns(df)

    for column in ("num_sampling_steps", "test_crps", "test_crps_sum"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["num_sampling_steps"] = df["num_sampling_steps"].astype(int)
    df = df[df["num_sampling_steps"].isin(STEP_ORDER)]

    grouped = (
        df.groupby(["data", "num_sampling_steps"], as_index=False)
        .agg(
            crps=("test_crps", "mean"),
            crps_sum=("test_crps_sum", "mean"),
            crps_std=("test_crps", "std"),
            crps_sum_std=("test_crps_sum", "std"),
            n=("test_crps", "size"),
        )
        .sort_values(["data", "num_sampling_steps"])
    )
    grouped[["crps_std", "crps_sum_std"]] = grouped[["crps_std", "crps_sum_std"]].fillna(0.0)
    return grouped


def pad_limits(values: np.ndarray, fraction: float = 0.14) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * fraction, 1e-3)
    else:
        pad = (vmax - vmin) * fraction
    return vmin - pad, vmax + pad


def prepare_dataset_metric(
    sensitivity_df: pd.DataFrame,
    dataset: str,
    metric_column: str,
) -> np.ndarray:
    dataset_df = sensitivity_df[sensitivity_df["data"] == dataset]
    if dataset_df.empty:
        raise ValueError(f"Dataset '{dataset}' was not found in the CSV file.")

    data = dataset_df.set_index("num_sampling_steps").reindex(STEP_ORDER)
    if data[metric_column].isna().any():
        missing_steps = data.index[data[metric_column].isna()].tolist()
        raise ValueError(f"{dataset} is missing {metric_column} values for steps: {missing_steps}")
    return data[metric_column].to_numpy(dtype=np.float64)


def style_axis(ax: plt.Axes) -> None:
    grid_color = "#8D99AA"
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28, color=grid_color)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.16, color=grid_color)
    ax.tick_params(axis="both", length=3.5, width=0.9, pad=4)
    ax.spines["left"].set_color("#2A3140")
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_color("#2A3140")
    ax.spines["bottom"].set_linewidth(1.05)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_metric_group(
    sensitivity_df: pd.DataFrame,
    metric_key: str,
    datasets: tuple[str, ...],
    output_path: Path,
    dpi: int,
    figure_size: tuple[float, float] = (6.8, 4.8),
) -> None:
    metric = METRICS[metric_key]
    metric_column = metric["column"]
    metric_label = metric["label"]
    x = np.arange(len(STEP_ORDER), dtype=np.float64)
    main_step_x = STEP_ORDER.index(MAIN_EXPERIMENT_STEP)
    colors = ["#1F5A93", "#B13A32", "#2F7D5B", "#7A4EAB"]
    markers = ["s", "o", "^", "D"]

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    fig.patch.set_facecolor("white")

    all_values: list[np.ndarray] = []
    for dataset_idx, dataset in enumerate(datasets):
        values = prepare_dataset_metric(
            sensitivity_df=sensitivity_df,
            dataset=dataset,
            metric_column=metric_column,
        )
        all_values.append(values)
        ax.plot(
            x,
            values,
            marker=markers[dataset_idx % len(markers)],
            color=colors[dataset_idx % len(colors)],
            markerfacecolor=colors[dataset_idx % len(colors)],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=DATASET_LABELS.get(dataset, dataset),
            zorder=4,
        )

    ax.axvline(
        main_step_x,
        color="#222222",
        linestyle="--",
        linewidth=1.35,
        alpha=0.82,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(step) for step in STEP_ORDER])
    ax.set_xlim(-0.35, len(STEP_ORDER) - 0.65)
    ax.set_ylim(*pad_limits(np.concatenate(all_values, axis=0)))
    ax.set_xlabel("ODE Steps", labelpad=6)
    ax.set_ylabel(metric_label, labelpad=8)
    style_axis(ax)

    legend = ax.legend(
        loc="best",
        ncol=1,
        frameon=True,
        framealpha=0.94,
        borderpad=0.36,
        handlelength=1.75,
        handletextpad=0.55,
    )
    legend.get_frame().set_edgecolor("#A8B2C2")
    legend.get_frame().set_linewidth(0.75)
    legend.get_frame().set_facecolor("white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot LS-Flow NUM_SAMPLING_STEPS sensitivity figures."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/LS-Flow_steps_sensitivity.csv"),
        help="Path to the sensitivity CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/LS-Flow_steps_sensitivity_grouped"),
        help="Directory to save grouped PDF figures.",
    )
    parser.add_argument("--dpi", type=int, default=900, help="PDF output DPI.")
    args = parser.parse_args()

    configure_style(dpi=args.dpi)
    sensitivity_df = load_sensitivity_data(args.csv_path)
    if sensitivity_df.empty:
        raise ValueError("No valid sensitivity rows were found in the CSV file.")

    for metric_key in METRICS:
        for group_idx, dataset_group in enumerate(DATASET_GROUPS, start=1):
            group_slug = "_".join(slugify(dataset) for dataset in dataset_group)
            output_path = args.output_dir / f"{metric_key}_group{group_idx}_{group_slug}.pdf"
            plot_metric_group(
                sensitivity_df=sensitivity_df,
                metric_key=metric_key,
                datasets=dataset_group,
                output_path=output_path,
                dpi=args.dpi,
            )
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
