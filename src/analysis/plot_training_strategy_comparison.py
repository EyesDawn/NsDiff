from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None


TABLE_LABEL = "tab:lsflow_training_strategy_comparison"


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
            "axes.labelsize": 22,
            "axes.titlesize": 18,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "axes.linewidth": 1.1,
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


def parse_table(table_block: str) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    lines = [line.strip() for line in table_block.splitlines() if line.strip()]

    header_line = next(
        (line for line in lines if line.startswith("Metric & Training Strategy &")),
        None,
    )
    if header_line is None:
        raise ValueError("Header line not found in target table.")

    header_parts = [part.strip() for part in header_line.replace("\\\\", "").split("&")]
    datasets = header_parts[2:]

    metrics: dict[str, dict[str, list[float]]] = {}
    current_metric: str | None = None

    for line in lines:
        if "\\multirow" in line:
            metric_match = re.search(r"\\multirow\{2\}\{\*\}\{(.*)\}", line)
            if metric_match is None:
                raise ValueError(f"Could not parse metric line: {line}")
            current_metric = strip_latex(metric_match.group(1))
            metrics[current_metric] = {}
            continue

        if current_metric is None:
            continue
        if "& Pre-trained &" not in line and "& End-to-End &" not in line:
            continue

        parts = [part.strip() for part in line.replace("\\\\", "").split("&")]
        strategy = parts[1]
        values = [float(strip_latex(cell)) for cell in parts[2:]]
        metrics[current_metric][strategy] = values

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
    e2e_values: np.ndarray,
    upper_limit: float,
) -> None:
    span = upper_limit - 0.0
    for idx, x_center in enumerate(xs):
        top = max(pre_values[idx], e2e_values[idx])
        delta = e2e_values[idx] - pre_values[idx]
        label_y = top + span * 0.025
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
        )


def build_stacked_segments(
    pre_values: np.ndarray,
    e2e_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre_base = np.zeros_like(pre_values)
    pre_height = np.zeros_like(pre_values)
    e2e_base = np.zeros_like(e2e_values)
    e2e_height = np.zeros_like(e2e_values)

    for idx, (pre_val, e2e_val) in enumerate(zip(pre_values, e2e_values)):
        if e2e_val >= pre_val:
            pre_base[idx] = 0.0
            pre_height[idx] = pre_val
            e2e_base[idx] = pre_val
            e2e_height[idx] = e2e_val - pre_val
        else:
            e2e_base[idx] = 0.0
            e2e_height[idx] = e2e_val
            pre_base[idx] = e2e_val
            pre_height[idx] = pre_val - e2e_val

    return pre_base, pre_height, e2e_base, e2e_height


def plot_single_metric(
    metric_name: str,
    datasets: list[str],
    metric_values: dict[str, list[float]],
    output_path: Path,
    figure_size: tuple[float, float] = (12.0, 5.8),
) -> None:
    pre_values = np.asarray(metric_values["Pre-trained"], dtype=np.float64)
    e2e_values = np.asarray(metric_values["End-to-End"], dtype=np.float64)
    x = np.arange(len(datasets), dtype=np.float64) * 0.86
    width = 0.50

    if sns is not None:
        light_color = "#6FB7D6"
        dark_color = "#2D6FA3"
    else:
        light_color = "#6FB7D6"
        dark_color = "#2D6FA3"

    edge_color = "#1E3557"
    max_value = float(max(np.max(pre_values), np.max(e2e_values)))
    min_value = float(min(np.min(pre_values), np.min(e2e_values)))
    span = max(max_value - min_value, max_value * 0.12, 0.02)
    upper_limit = max_value + span * 0.32
    pre_base, pre_height, e2e_base, e2e_height = build_stacked_segments(
        pre_values=pre_values,
        e2e_values=e2e_values,
    )

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    fig.patch.set_facecolor("white")

    bars_pre = ax.bar(
        x,
        pre_height,
        width=width,
        bottom=pre_base,
        color=light_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Pre-trained",
        zorder=3,
    )
    bars_e2e = ax.bar(
        x,
        e2e_height,
        width=width,
        bottom=e2e_base,
        color=dark_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="End-to-End",
        zorder=3,
    )

    for bars in (bars_pre, bars_e2e):
        for bar in bars:
            bar.set_joinstyle("miter")

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.24)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=18)
    ax.set_ylabel(metric_to_ylabel(metric_name), fontsize=22, labelpad=10)
    ax.set_xlabel("Dataset", fontsize=22, labelpad=10)
    ax.set_ylim(0.0, upper_limit)
    ax.margins(x=0.015)

    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_color("#2A3140")
    ax.spines["bottom"].set_color("#2A3140")
    ax.tick_params(axis="x", pad=6, length=4, width=0.9, labelsize=18)
    ax.tick_params(axis="y", pad=4, length=4, width=0.9, labelsize=18)

    add_delta_labels(ax, x, pre_values, e2e_values, upper_limit)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        borderpad=0.35,
        handlelength=1.6,
        columnspacing=1.5,
        prop={"size": 17},
    )
    legend.get_frame().set_edgecolor("#A8B2C2")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")

    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)


def select_datasets(
    datasets: list[str],
    metric_values: dict[str, list[float]],
    excluded_datasets: set[str],
) -> tuple[list[str], dict[str, list[float]]]:
    keep_indices = [idx for idx, dataset in enumerate(datasets) if dataset not in excluded_datasets]
    filtered_datasets = [datasets[idx] for idx in keep_indices]
    filtered_values = {
        strategy: [values[idx] for idx in keep_indices]
        for strategy, values in metric_values.items()
    }
    return filtered_datasets, filtered_values


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
        default=600,
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
        plot_single_metric(metric_name, datasets, metric_values, output_path)
        print(f"Saved {output_path}")

    crps_subset_datasets, crps_subset_values = select_datasets(
        datasets=datasets,
        metric_values=metrics["CRPS"],
        excluded_datasets={"ETTm1", "ETTm2", "ETTh2"},
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
