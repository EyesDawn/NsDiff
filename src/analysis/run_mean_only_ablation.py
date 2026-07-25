#!/usr/bin/env python3
"""Launch the four-dataset Mean-only ablation with one worker per GPU."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def parse_gpu_map(value):
    gpu_map = {}
    for assignment in value.split(","):
        dataset, separator, gpu = assignment.strip().partition("=")
        if not separator or not dataset or not gpu:
            raise ValueError(
                "GPU map must be comma-separated dataset=physical_gpu assignments; "
                "for example ETTm1=4,ETTm2=5,Weather=6,Electricity=0."
            )
        gpu_map[dataset] = gpu
    return gpu_map


def make_command(dataset, common, args, seeds):
    command = [
        sys.executable, "-u", "./src/experiments/iReflow.py",
        "--wandb_project", args.wandb_project,
        "--is_training", "1",
        "--root_path", args.root_path,
        "--data_path", dataset["data_path"],
        "--model_id", dataset["model_id"],
        "--model", "iReflow",
        "--data", dataset["dataset"],
        "--features", "M",
        "--seq_len", "96", "--pred_len", "192", "--des", "Exp", "--itr", "1",
        "--checkpoints", args.checkpoints, "--device", args.device,
    ]
    for key in (
        "enc_in", "dec_in", "c_out", "d_model", "n_heads", "e_layers",
        "flow_layers", "d_ff", "batch_size", "lr", "epochs", "patience",
        "lr_patience", "dropout",
    ):
        command.extend([f"--{key}", str(dataset[key])])
    for key, value in common.items():
        if isinstance(value, bool):
            value = "True" if value else "False"
        command.extend([f"--{key}", str(value)])
    command.extend(["runs", f"--seeds={seeds}"])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root-path", default="./data/")
    parser.add_argument("--checkpoints", default="./results/runs/iTransformer/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="iReflow-MeanOnly")
    parser.add_argument("--gpu-map", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    common = {key: config[key] for key in (
        "ablation_mode", "num_sampling_steps", "temperature", "num_samples",
        "x0_dist", "use_relative_space",
    )}
    seeds = json.dumps(config["seeds"], separators=(",", ","))
    gpu_map = parse_gpu_map(args.gpu_map)
    datasets = config["datasets"]
    missing = [dataset["dataset"] for dataset in datasets if dataset["dataset"] not in gpu_map]
    if missing:
        raise SystemExit(f"GPU map has no assignment for: {', '.join(missing)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_root = repo_root / "results" / "logs" / "mean_only_ablation" / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for dataset in datasets:
        groups[gpu_map[dataset["dataset"]]].append(dataset)

    def run_gpu_group(gpu, group):
        results = []
        for dataset in group:
            dataset_name = dataset["dataset"]
            command = make_command(dataset, common, args, seeds)
            log_path = log_root / f"{dataset_name}.log"
            environment = {
                **os.environ,
                "PYTHONPATH": str(repo_root),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu),
            }
            print(f"Launching {dataset_name} on physical GPU {gpu}; log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write("Command: " + " ".join(command) + "\n\n")
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            results.append((dataset_name, completed.returncode, str(log_path)))
        return results

    failures = []
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(run_gpu_group, gpu, group) for gpu, group in groups.items()]
        for future in as_completed(futures):
            for dataset_name, returncode, log_path in future.result():
                if returncode:
                    failures.append((dataset_name, returncode, log_path))

    subprocess.run([
        sys.executable, "./src/analysis/collect_mean_only_ablation.py",
        "--runs-root", "results/runs/iReflow",
        "--output-csv", "evidence/mean_only_ablation_per_seed.csv",
        "--output-md", "evidence/mean_only_ablation.md",
    ], cwd=repo_root, check=True)
    if failures:
        print("Failed datasets:", failures, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
