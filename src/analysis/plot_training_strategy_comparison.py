from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

try:
    import seaborn as sns
except ImportError:
    sns = None


TABLE_LABEL = "tab:lsflow_training_strategy_comparison"
SELECTED_DATASETS = ("ETTh1", "Weather", "Solar", "ECL", "Traffic")


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
                "grid.alpha": 0.25,
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
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.linewidth": 1.1,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def extract_target_table(tex_content: str, table_label: str) -> str:
    table_pattern = re.compile(
        r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}",
        flags=re.DOTALL,
    )
    for match in table_pattern.finditer(tex_content):
        block = match.group(0)
        if table_label in block:
            return block
    raise ValueError(f"Could not find table with label '{table_label}'.")


def strip_latex(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\\\\", "").strip()
    cleaned = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", cleaned)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("\\_", "_")
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_numeric_cell(cell: str) -> tuple[float, float]:
    normalized = cell
    normalized = re.sub(r"\\?\[-?\d+pt\]", " ", normalized)
    normalized = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", normalized)
    normalized = normalized.replace("\\tiny", " ")
    normalized = normalized.replace("\\pm", " ")
    normalized = normalized.replace("$", " ")
    normalized = normalized.replace("{", " ")
    normalized = normalized.replace("}", " ")
    normalized = normalized.replace("\\\\", " ")
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", normalized)
    if len(numbers) < 2:
        raise ValueError(f"Could not parse mean/std from cell: {cell}")
    return float(numbers[0]), float(numbers[1])


def parse_table(
    table_block: str,
) -> tuple[list[str], dict[str, dict[str, dict[str, list[float]]]]]:
    raw_lines = [line.strip() for line in table_block.splitlines() if line.strip()]
    lines: list[str] = []
    current_parts: list[str] = []
    for line in raw_lines:
        current_parts.append(line)
        if line.endswith("\\\\"):
            lines.append(" ".join(current_parts))
            current_parts = []
    if current_parts:
        lines.append(" ".join(current_parts))

    header_line = next(
        (line for line in lines if "Metric & Training Strategy &" in line),
        None,
    )
    if header_line is None:
        raise ValueError("Header line not found in target table.")

    header_line = header_line[header_line.index("Metric & Training Strategy &") :]
    header_parts = [part.strip() for part in header_line.replace("\\\\", "").split("&")]
    datasets = header_parts[2:]

    metrics: dict[str, dict[str, dict[str, list[float]]]] = {}
    current_metric: str | None = None

    for line in lines:
        if "\\multirow" in line:
            metric_match = re.search(r"\\multirow\{2\}\{\*\}\{(.+?)\}\s*&", line)
            if metric_match is None:
                raise ValueError(f"Could not parse metric line: {line}")
            current_metric = strip_latex(metric_match.group(1))
            metrics[current_metric] = {}

        if current_metric is None:
            continue
        if "& Pre-trained &" not in line and "& End-to-End &" not in line:
            continue

        parts = [part.strip() for part in line.replace("\\\\", "").split("&")]
        strategy = parts[1]
        means: list[float] = []
        stds: list[float] = []
        for cell in parts[2:]:
            mean, std = parse_numeric_cell(cell)
            means.append(mean)
            stds.append(std)
        metrics[current_metric][strategy] = {"mean": means, "std": stds}

    for metric_name, strategy_dict in metrics.items():
        missing = {"Pre-trained", "End-to-End"} - set(strategy_dict)
        if missing:
            raise ValueError(f"Metric '{metric_name}' is missing rows: {sorted(missing)}")

    return datasets, metrics


def metric_to_filename(metric_name: str) -> str:
    safe = metric_name.lower().replace(" ", "_")
    safe = safe.replace("_", "-")
    return safe


def metric_to_ylabel(metric_name: str) -> str:
    if metric_name == "CRPS_sum":
        return r"$\mathrm{CRPS}_{\mathrm{sum}}$"
    return metric_name


def add_delta_labels(
    ax: plt.Axes,
    xs: np.ndarray,
    pre_values: np.ndarray,
    pre_stds: np.ndarray,
    e2e_values: np.ndarray,
    e2e_stds: np.ndarray,
    upper_limit: float,
) -> None:
    span = upper_limit - 0.0
    for idx, x_center in enumerate(xs):
        top = max(pre_values[idx] + pre_stds[idx], e2e_values[idx] + e2e_stds[idx])
        delta = e2e_values[idx] - pre_values[idx]
        label_y = top + span * 0.035
        label_text = rf"$\Delta={delta:+.3f}$"
        if delta < 0:
            color = "#1B7F5A"
        elif delta > 0:
            color = "#A23B3B"
        else:
            color = "#5C6773"
        ax.text(
            x_center,
            label_y,
            label_text,
            ha="center",
            va="bottom",
            fontsize=13,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )


def plot_single_metric(
    metric_name: str,
    datasets: list[str],
    metric_values: dict[str, dict[str, list[float]]],
    output_path: Path,
    figure_size: tuple[float, float] = (8.0, 6.0),
) -> None:
    pre_values = np.asarray(metric_values["Pre-trained"]["mean"], dtype=np.float64)
    pre_stds = np.asarray(metric_values["Pre-trained"]["std"], dtype=np.float64)
    e2e_values = np.asarray(metric_values["End-to-End"]["mean"], dtype=np.float64)
    e2e_stds = np.asarray(metric_values["End-to-End"]["std"], dtype=np.float64)
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.30
    offsets = np.array([-0.5, 0.5], dtype=np.float64) * width

    pre_color = "#1F77B4"
    e2e_color = "#4C78A8"
    strategy_styles = {
        "Pre-trained": {"color": pre_color, "hatch": "***"},
        "End-to-End": {"color": e2e_color, "hatch": "ooo"},
    }

    max_value = float(max(np.max(pre_values + pre_stds), np.max(e2e_values + e2e_stds)))
    min_value = 0.0
    span = max(max_value - min_value, max_value * 0.14, 0.02)
    upper_limit = max_value + span * 0.34

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    fig.patch.set_facecolor("white")

    errorbar_style = {
        "fmt": "none",
        "ecolor": "#111827",
        "elinewidth": 1.35,
        "capsize": 4.2,
        "capthick": 1.35,
        "zorder": 5,
    }

    bars_pre = ax.bar(
        x + offsets[0],
        pre_values,
        width=width,
        color=strategy_styles["Pre-trained"]["color"],
        edgecolor="white",
        linewidth=0.8,
        hatch=strategy_styles["Pre-trained"]["hatch"],
        label="Pre-trained",
        zorder=3,
    )
    bars_e2e = ax.bar(
        x + offsets[1],
        e2e_values,
        width=width,
        color=strategy_styles["End-to-End"]["color"],
        edgecolor="white",
        linewidth=0.8,
        hatch=strategy_styles["End-to-End"]["hatch"],
        label="End-to-End",
        zorder=3,
    )

    for bars in (bars_pre, bars_e2e):
        for bar in bars:
            bar.set_joinstyle("miter")

    ax.errorbar(
        x + offsets[0],
        pre_values,
        yerr=pre_stds,
        **errorbar_style,
    )
    ax.errorbar(
        x + offsets[1],
        e2e_values,
        yerr=e2e_stds,
        **errorbar_style,
    )

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.24)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=15)
    ax.set_ylabel(metric_to_ylabel(metric_name), fontsize=20, labelpad=10)
    ax.set_xlabel("Dataset", fontsize=20, labelpad=10)
    ax.set_ylim(0.0, upper_limit)
    ax.margins(x=0.06)

    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_color("#2A3140")
    ax.spines["bottom"].set_color("#2A3140")
    ax.tick_params(axis="x", pad=6, length=4, width=0.9, labelsize=15)
    ax.tick_params(axis="y", pad=4, length=4, width=0.9, labelsize=15)

    add_delta_labels(ax, x, pre_values, pre_stds, e2e_values, e2e_stds, upper_limit)

    legend_handles = [
        Patch(
            facecolor=strategy_styles["Pre-trained"]["color"],
            edgecolor="white",
            linewidth=0.8,
            hatch=strategy_styles["Pre-trained"]["hatch"],
            label="Pre-trained",
        ),
        Patch(
            facecolor=strategy_styles["End-to-End"]["color"],
            edgecolor="white",
            linewidth=0.8,
            hatch=strategy_styles["End-to-End"]["hatch"],
            label="End-to-End",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        framealpha=0.95,
        borderpad=0.42,
        handlelength=1.7,
        handletextpad=0.55,
        prop={"size": 14},
    )
    legend.get_frame().set_edgecolor("#A8B2C2")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")

    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=900)
    plt.close(fig)


def select_datasets(
    datasets: list[str],
    metric_values: dict[str, dict[str, list[float]]],
    selected_datasets: tuple[str, ...],
) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    keep_indices = [datasets.index(dataset) for dataset in selected_datasets]
    filtered_values = {
        strategy: {
            "mean": [stats["mean"][idx] for idx in keep_indices],
            "std": [stats["std"][idx] for idx in keep_indices],
        }
        for strategy, stats in metric_values.items()
    }
    return list(selected_datasets), filtered_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot grouped bar charts for LS-Flow training strategy comparison."
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=Path("results/all_results.tex"),
        help="Path to the LaTeX file that contains the comparison table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/training_strategy_comparison"),
        help="Directory to save per-metric PDF figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=900,
        help="Output DPI used for PDF metadata and rasterized elements.",
    )
    args = parser.parse_args()

    configure_style(dpi=args.dpi)
    tex_content = args.tex_path.read_text(encoding="utf-8")
    table_block = extract_target_table(tex_content, TABLE_LABEL)
    datasets, metrics = parse_table(table_block)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric_name, metric_values in metrics.items():
        output_path = args.output_dir / f"{metric_to_filename(metric_name)}.pdf"
        plot_single_metric(
            metric_name,
            datasets,
            metric_values,
            output_path,
            figure_size=(10.5, 5.2),
        )
        print(f"Saved {output_path}")

    crps_subset_datasets, crps_subset_values = select_datasets(
        datasets=datasets,
        metric_values=metrics["CRPS"],
        selected_datasets=SELECTED_DATASETS,
    )
    subset_output_path = args.output_dir / "crps-selected.pdf"
    plot_single_metric(
        "CRPS",
        crps_subset_datasets,
        crps_subset_values,
        subset_output_path,
        figure_size=(8.0, 6.0),
    )
    print(f"Saved {subset_output_path}")


if __name__ == "__main__":
    main()
