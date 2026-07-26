#!/usr/bin/env python3
"""Collect Mean-only run logs into the evidence artifacts required by the ablation."""

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("ETTm1", "ETTm2", "Weather", "Electricity")
SEEDS = (2022, 2023, 2024)
METRICS = (
    "crps", "crps_sum", "energy_score", "mse", "mae", "coverage_90",
    "coverage_95", "interval_width_90", "interval_width_95", "residual_mean",
    "residual_variance", "residual_skewness",
)
RESULT_RE = re.compile(r"test_results:\s*(\{.*\})")


def read_last_result(log_path: Path):
    if not log_path.exists():
        return None
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_RE.search(line)
        if match:
            try:
                result = ast.literal_eval(match.group(1))
            except (SyntaxError, ValueError):
                continue
    return result


def find_runs(runs_root: Path):
    found = {}
    for args_path in runs_root.rglob("train_mode_1/args.json"):
        try:
            config = json.loads(args_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config.get("ablation_mode") != "mean_only":
            continue
        dataset = config.get("dataset_type", config.get("data"))
        seed = config.get("seed")
        # The runner's settings carry the seed separately; fall back to log parsing.
        log_path = args_path.with_name("output.log")
        if seed is None and log_path.exists():
            seed_match = re.search(r"seed:\s*(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
            seed = int(seed_match.group(1)) if seed_match else None
        if dataset in DATASETS and seed in SEEDS:
            found[(dataset, int(seed))] = (args_path.parent, config, read_last_result(log_path))
    return found


def value(row, metric):
    raw = row.get(metric, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def render_summary(rows):
    headers = ["Dataset", "CRPS", "CRPS-sum", "Energy Score", "MSE", "MAE", "90% cov.", "95% cov.", "90% width", "95% width", "Residual mean", "Residual var.", "Residual skew"]
    metric_headers = dict(zip(headers[1:], METRICS))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset and row["status"] == "success"]
        cells = [dataset]
        for header in headers[1:]:
            values = [value(row, metric_headers[header]) for row in selected]
            values = [item for item in values if math.isfinite(item)]
            if not values:
                cells.append("—")
            elif len(values) == 1:
                cells.append(f"{values[0]:.6g} ± 0")
            else:
                cells.append(f"{mean(values):.6g} ± {stdev(values):.3g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="results/runs/iReflow")
    parser.add_argument("--output-csv", default="evidence/mean_only_ablation_per_seed.csv")
    parser.add_argument("--output-md", default="evidence/mean_only_ablation.md")
    args = parser.parse_args()

    found = find_runs(Path(args.runs_root))
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = found.get((dataset, seed))
            row = {
                "dataset": dataset,
                "seed": seed,
                "design": "Mean-only",
                "status": "missing",
                "failure_reason": "run directory not found",
            }
            for metric in METRICS:
                row[metric] = ""
            row.update({"run_dir": "", "stage1_checkpoint": "", "stage1_sha256": "", "config_path": ""})
            if run is not None:
                run_dir, config, result = run
                row.update({"run_dir": str(run_dir), "config_path": str(run_dir / "args.json")})
                provenance_path = run_dir / "stage1_checkpoint.json"
                if provenance_path.exists():
                    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                    row["stage1_checkpoint"] = provenance.get("path", "")
                    row["stage1_sha256"] = provenance.get("sha256", "")
                if result is not None:
                    row["status"] = "success"
                    row["failure_reason"] = ""
                    for metric in METRICS:
                        row[metric] = result.get(metric, "")
                else:
                    row["status"] = "incomplete"
                    row["failure_reason"] = "test result not found in output.log"
            rows.append(row)

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "seed", "design", "status", "failure_reason", *METRICS, "run_dir", "stage1_checkpoint", "stage1_sha256", "config_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        dataset: sum(
            row["dataset"] == dataset and row["status"] == "success" for row in rows
        )
        for dataset in DATASETS
    }
    completion_status = ", ".join(
        f"{dataset} {completed[dataset]}/{len(SEEDS)}" for dataset in DATASETS
    )
    md_path.write_text(
        "# Mean-only ablation evidence\n\n"
        "Mean-only uses a fixed unit scale for residual coordinates, source noise, and inverse mapping. "
        "The Location--scale comparison is supplied externally and is not reproduced on this machine.\n\n"
        "## Mean-only (mean ± sample standard deviation across completed seeds)\n\n"
        + render_summary(rows)
        + "\n\n## Status\n\n"
        f"- Completed Mean-only seeds: {completion_status}.\n"
        "- Not run in this batch: ETTh1, ETTh2, SolarEnergy, Traffic.\n"
        "- Do not infer a Location--scale advantage until compatible external results are merged.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
