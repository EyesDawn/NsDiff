#!/usr/bin/env python3
"""Collect final Stage-3 LS-Flow test metrics from a parallel batch."""

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"test_results:\s*(\{.*\})")
BASE_COLUMNS = [
    "task_id",
    "dataset",
    "pred_len",
    "seed",
    "status",
    "failure_reason",
    "ES",
]
TRAILING_COLUMNS = ["run_dir", "log_path"]


def final_test_result(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        try:
            candidate = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(candidate, dict):
            result = candidate
    return result


def failure_detail(log_path: Path, exit_code: str | None) -> str:
    reason = f"task exited with code {exit_code}" if exit_code is not None else "task did not write exit_code"
    if not log_path.exists():
        return reason + "; task log not found"
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if lines:
        return f"{reason}; last log line: {lines[-1][:500]}"
    return reason


def stringify(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return "" if value is None else value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.batch_dir / "manifest.tsv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    output_csv = args.output_csv or args.batch_dir / "lsflow_three_stage_results.csv"

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))

    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    for task in tasks:
        task_dir = args.batch_dir / "tasks" / task["task_id"]
        log_path = task_dir / "task.log"
        exit_code_path = task_dir / "exit_code"
        exit_code = exit_code_path.read_text(encoding="utf-8").strip() if exit_code_path.exists() else None
        result = final_test_result(log_path)

        row: dict[str, Any] = {
            **task,
            "status": "success",
            "failure_reason": "",
            "ES": "",
            "run_dir": str(task_dir.resolve()),
            "log_path": str(log_path.resolve()),
        }
        if exit_code != "0":
            row["status"] = "failed"
            row["failure_reason"] = failure_detail(log_path, exit_code)
        elif result is None:
            row["status"] = "incomplete"
            row["failure_reason"] = "final Stage-3 test_results not found in task log"

        if result is not None:
            row["ES"] = stringify(result.get("energy_score"))
            for name, value in result.items():
                if name == "energy_score":
                    continue
                metric_names.add(name)
                row[name] = stringify(value)
        rows.append(row)

    fieldnames = [*BASE_COLUMNS, *sorted(metric_names), *TRAILING_COLUMNS]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    success_count = sum(row["status"] == "success" for row in rows)
    print(f"Wrote {len(rows)} rows ({success_count} successful) to {output_csv}")


if __name__ == "__main__":
    main()
