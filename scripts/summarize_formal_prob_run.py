#!/usr/bin/env python3
import argparse
import ast
import datetime as dt
import os
import re
from typing import Dict, List, Optional


PREFIX_RE = re.compile(r"^\[(?P<label>[^\]]+)\]\s?(?P<message>.*)$")
RUN_DIR_CREATE_RE = re.compile(r"Creating running results saving dir:\s+'(?P<path>[^']+)'\.")
RUN_DIR_EXISTS_RE = re.compile(r"result directory exists:\s+(?P<path>\S+)")
SEED_RE = re.compile(r"run\s*:\s*\d+\s+in seed:\s*(?P<seed>.+)$")
TEST_RE = re.compile(r"test_results:\s*(?P<payload>\{.*\})$")
CONFIG_RE = re.compile(r"^(Using seeds|Using epochs|Using pretrain F epochs|Using pretrain G epochs|Starting from model):")
FAILED_RE = re.compile(r"^Failed:\s+(?P<label>.+)$")


def normalize_run_dir(path: str) -> str:
    path = path.strip().strip("'").strip('"')
    if path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def stage_from_run_dir(run_dir: str) -> str:
    parts = run_dir.split("/")
    if len(parts) < 4:
        return "unknown"
    model_dir = parts[2]
    mapping = {
        "TimeGrad": "TimeGrad",
        "CSDI": "CSDI",
        "TimeDiff": "TimeDiff",
        "NsDiff4": "NSDiff",
        "F": "NSDiff-pretrain-F",
        "G": "NSDiff-pretrain-G",
    }
    return mapping.get(model_dir, model_dir)


def dataset_from_run_dir(run_dir: str) -> str:
    parts = run_dir.split("/")
    return parts[3] if len(parts) > 3 else "-"


def model_from_stage(stage: str) -> str:
    if stage.startswith("NSDiff-pretrain"):
        return "NSDiff"
    return stage


def notes_from_label(label: str) -> str:
    return label


def make_row(stage: str, dataset: str, seed: str, run_dir: str, status: str, metrics: Optional[Dict], notes: str) -> Dict[str, str]:
    metrics = metrics or {}
    return {
        "Model": model_from_stage(stage),
        "Dataset": dataset,
        "Stage": stage,
        "Seeds": seed or "-",
        "Run Dir": run_dir,
        "Status": status,
        "CRPS": str(metrics.get("crps", "-")),
        "CRPS_SUM": str(metrics.get("crps_sum", "-")),
        "MAE": str(metrics.get("mae", "-")),
        "MSE": str(metrics.get("mse", "-")),
        "RMSE": str(metrics.get("rmse", "-")),
        "PICP": str(metrics.get("picp", "-")),
        "QICE": str(metrics.get("qice", "-")),
        "Notes": notes or "-",
    }


def render_markdown_table(rows: List[Dict[str, str]]) -> str:
    headers = [
        "Model",
        "Dataset",
        "Stage",
        "Seeds",
        "Run Dir",
        "Status",
        "CRPS",
        "CRPS_SUM",
        "MAE",
        "MSE",
        "RMSE",
        "PICP",
        "QICE",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to the run log file.")
    parser.add_argument("--output", required=True, help="Path to the markdown summary file.")
    args = parser.parse_args()

    contexts: Dict[str, Dict[str, Optional[str]]] = {}
    run_records: Dict[str, Dict[str, str]] = {}
    ordered_run_dirs: List[str] = []
    config_lines: List[str] = []
    failed_labels: List[str] = []

    with open(args.log, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n").replace("\r", "")
            if not line:
                continue

            if CONFIG_RE.search(line):
                config_lines.append(line)

            failed_match = FAILED_RE.search(line)
            if failed_match:
                failed_labels.append(failed_match.group("label"))

            prefix_match = PREFIX_RE.match(line)
            if not prefix_match:
                continue

            label = prefix_match.group("label")
            message = prefix_match.group("message").strip()
            context = contexts.setdefault(label, {"run_dir": None, "seed": None})

            run_dir_match = RUN_DIR_CREATE_RE.search(message) or RUN_DIR_EXISTS_RE.search(message)
            if run_dir_match:
                run_dir = normalize_run_dir(run_dir_match.group("path"))
                context["run_dir"] = run_dir
                if run_dir not in run_records:
                    stage = stage_from_run_dir(run_dir)
                    run_records[run_dir] = {
                        "stage": stage,
                        "dataset": dataset_from_run_dir(run_dir),
                        "seed": context.get("seed") or "-",
                        "notes": notes_from_label(label),
                    }
                    ordered_run_dirs.append(run_dir)
                continue

            seed_match = SEED_RE.search(message)
            if seed_match:
                seed = seed_match.group("seed").strip()
                context["seed"] = seed
                if context.get("run_dir") and context["run_dir"] in run_records:
                    run_records[context["run_dir"]]["seed"] = seed
                continue

            test_match = TEST_RE.search(message)
            if test_match and context.get("run_dir"):
                run_dir = context["run_dir"]
                try:
                    metrics = ast.literal_eval(test_match.group("payload"))
                except Exception:
                    metrics = {}
                run_records.setdefault(
                    run_dir,
                    {
                        "stage": stage_from_run_dir(run_dir),
                        "dataset": dataset_from_run_dir(run_dir),
                        "seed": context.get("seed") or "-",
                        "notes": notes_from_label(label),
                    },
                )["metrics"] = metrics
                continue

    success_rows: List[Dict[str, str]] = []
    checkpoint_only_rows: List[Dict[str, str]] = []

    for run_dir in ordered_run_dirs:
        record = run_records[run_dir]
        row = make_row(
            stage=record["stage"],
            dataset=record["dataset"],
            seed=record.get("seed", "-"),
            run_dir=run_dir,
            status="success" if "metrics" in record else "checkpoint_only",
            metrics=record.get("metrics"),
            notes=record.get("notes", "-"),
        )
        if "metrics" in record:
            success_rows.append(row)
        else:
            checkpoint_only_rows.append(row)

    failed_rows = [
        make_row(
            stage=label.split("/")[0] if "/" in label else "unknown",
            dataset="-",
            seed="-",
            run_dir="-",
            status="failed",
            metrics=None,
            notes=label,
        )
        for label in failed_labels
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# Formal Probabilistic Run Summary\n\n")
        handle.write(f"Generated at: {timestamp}\n\n")
        handle.write(f"Source log: `{args.log}`\n\n")
        if config_lines:
            handle.write("## Run Configuration\n\n")
            for line in config_lines:
                handle.write(f"- {line}\n")
            handle.write("\n")

        handle.write("## Successful Results\n\n")
        if success_rows:
            handle.write(render_markdown_table(success_rows))
            handle.write("\n\n")
        else:
            handle.write("No successful test results found in the log.\n\n")

        if checkpoint_only_rows:
            handle.write("## Checkpoint-Only Entries\n\n")
            handle.write(render_markdown_table(checkpoint_only_rows))
            handle.write("\n\n")

        if failed_rows:
            handle.write("## Failed Entries\n\n")
            handle.write(render_markdown_table(failed_rows))
            handle.write("\n")


if __name__ == "__main__":
    main()
