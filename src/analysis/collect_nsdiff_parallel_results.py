#!/usr/bin/env python3
"""Collect final NsDiff metrics from a parallel-grid run into a CSV file."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_FIELDS = ("crps", "crps_sum", "qice", "picp", "mse", "mae", "rmse", "es")
CSV_FIELDS = ("dataset", "pred_len", "seed", "status", *METRIC_FIELDS, "log_path")
STATUS_PATTERN = re.compile(r"^(?P<dataset>.+)_p(?P<pred_len>\d+)\.status$")


def _parse_final_metrics(log_path: Path) -> Dict[str, float] | None:
    """Return the last test_results mapping written by the final pipeline stage."""
    if not log_path.is_file():
        return None

    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if "test_results:" not in line:
            continue
        payload = line.split("test_results:", 1)[1].strip()
        try:
            result = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if isinstance(result, dict):
            return result
    return None


def _status_jobs(status_dir: Path) -> Iterable[tuple[str, int, Path]]:
    for status_path in sorted(status_dir.glob("*.status")):
        match = STATUS_PATTERN.match(status_path.name)
        if match is None:
            continue
        yield match.group("dataset"), int(match.group("pred_len")), status_path


def build_rows(run_dir: Path, seed: int) -> List[Dict[str, object]]:
    """Build one stable CSV row for every completed or failed grid job."""
    status_dir = run_dir / "status"
    log_dir = run_dir / "logs"
    rows: List[Dict[str, object]] = []

    for dataset, pred_len, status_path in _status_jobs(status_dir):
        log_path = log_dir / f"{dataset}_p{pred_len}.log"
        status = status_path.read_text(encoding="utf-8").strip() or "unknown"
        row: Dict[str, object] = {
            "dataset": dataset,
            "pred_len": pred_len,
            "seed": seed,
            "status": status,
            "log_path": str(log_path),
        }
        for field in METRIC_FIELDS:
            row[field] = ""

        if status == "succeeded":
            metrics = _parse_final_metrics(log_path)
            if metrics is None:
                row["status"] = "missing_test_results"
            else:
                for field in METRIC_FIELDS:
                    if field not in metrics:
                        row["status"] = f"missing_metric:{field}"
                        break
                    row[field] = metrics[field]
        rows.append(row)

    return rows


def write_summary(run_dir: Path, output_path: Path, seed: int) -> List[Dict[str, object]]:
    rows = build_rows(run_dir, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows = write_summary(args.run_dir, args.output, args.seed)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
