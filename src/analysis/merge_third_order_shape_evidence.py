#!/usr/bin/env python3
"""Merge per-dataset third-order pilot evidence shards deterministically."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.analysis.third_order_shape_pilot import CSV_COLUMNS, DATASETS, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", default="evidence/shape_decomposition_shards")
    parser.add_argument("--output_root", default="evidence")
    parser.add_argument("--runs_per_dataset", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    rows = []
    seen = set()
    for dataset in DATASETS:
        shard = repo_root / args.input_root / dataset / "shape_decomposition_per_seed.csv"
        if not shard.is_file():
            raise FileNotFoundError(f"Missing evidence shard for {dataset}: {shard}")
        with shard.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_COLUMNS:
                raise ValueError(f"Unexpected CSV schema in {shard}")
            for row in reader:
                key = (row["dataset"], row["run_id"], row["design"])
                if key in seen:
                    raise ValueError(f"Duplicate evidence row: {key}")
                seen.add(key)
                rows.append(row)
    write_outputs(rows, repo_root / args.output_root, DATASETS, args.runs_per_dataset)


if __name__ == "__main__":
    main()
