#!/usr/bin/env python3
"""Run and report the manifest-driven third-order LS-Flow pilot.

The script intentionally refuses to infer a baseline from ``results/``.  A
checked manifest is the sole source of the five fixed L2 checkpoints per
dataset, which prevents test-set-driven checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import traceback
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import properscoring as ps
import torch
import yaml
from setproctitle import setproctitle



DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "SolarEnergy", "Traffic", "Weather")
REQUIRED_COLUMNS = [
    "dataset", "seed", "design", "CRPS", "CRPS_sum", "ES", "MSE", "MAE",
    "coverage_90", "coverage_95", "coverage_99", "coverage_error_90", "coverage_error_95",
    "coverage_error_99", "interval_width_90", "interval_width_95", "interval_width_99",
    "skewness_before", "skewness_after", "excess_kurtosis_before", "excess_kurtosis_after",
    "checkpoint", "commit", "config_path",
]
EXTRA_COLUMNS = [
    "run_id", "l3_seed", "l3_checkpoint", "l3_parameters", "validation_alpha_nll",
    "validation_mean_abs_alpha", "status", "failure_reason",
]
CSV_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS


def set_process_title(datasets: Sequence[str], runs_per_dataset: int) -> str:
    """Expose the pilot's dataset and assigned CUDA device in process listings."""
    dataset_label = ",".join(datasets)
    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    title = f"iReflow-L3:{dataset_label}:gpu={gpu_label}:runs={runs_per_dataset}"
    setproctitle(title)
    return title


def reproducible(seed: int) -> None:
    """Use the project helper when available, with a lightweight fallback."""
    try:
        from torch_timeseries.utils.reproduce import reproducible as project_reproducible
    except ModuleNotFoundError:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        project_reproducible(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_ids(runs_per_dataset: int) -> Tuple[str, ...]:
    if runs_per_dataset < 1:
        raise ValueError("runs_per_dataset must be at least one.")
    return tuple(f"l2_{index}" for index in range(1, runs_per_dataset + 1))


def load_manifest(
    path: Path, repo_root: Path, datasets: Sequence[str], selected_run_ids: Sequence[str]
) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    entries = manifest.get("datasets")
    if not isinstance(entries, dict):
        raise ValueError("Manifest must contain a top-level 'datasets' mapping.")
    for dataset in datasets:
        if dataset not in entries:
            raise ValueError(f"Manifest is missing dataset '{dataset}'.")
        for run_id in selected_run_ids:
            entry = entries[dataset].get(run_id)
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest is missing {dataset}/{run_id}.")
            for key in ("checkpoint", "sha256", "config_path", "l3_seed"):
                if not entry.get(key):
                    raise ValueError(f"Manifest entry {dataset}/{run_id} has no '{key}'.")
            checkpoint = repo_root / str(entry["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing L2 checkpoint: {checkpoint}")
            expected_sha = str(entry["sha256"]).lower()
            actual_sha = sha256(checkpoint)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {checkpoint}: expected {expected_sha}, got {actual_sha}.")
            config_path = repo_root / str(entry["config_path"])
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing L2 configuration: {config_path}")
    return entries


def _supported_experiment_kwargs(payload: Mapping[str, object]) -> Dict[str, object]:
    from src.experiments.iReflow import iReflowExp

    known = {item.name for item in fields(iReflowExp)}
    return {key: value for key, value in payload.items() if key in known}


def experiment_kwargs_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Adapt a checked e2e YAML configuration to ``iReflowExp`` fields."""
    payload = dict(config)
    # The checked e2e configurations use ``dataset`` while iReflowExp uses
    # the iTransformer-compatible ``data`` field.  Preserve that dataset
    # selection before filtering unknown YAML keys; otherwise the dataclass
    # default ("custom") reaches parse_type() and is evaluated as a name.
    if "dataset" in payload and "data" not in payload:
        payload["data"] = payload["dataset"]
    return _supported_experiment_kwargs(payload)


def build_experiment(config_path: Path, data_root: str, seed: int) -> iReflowExp:
    from src.experiments.iReflow import iReflowExp

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    kwargs = experiment_kwargs_from_config(config)
    kwargs["root_path"] = data_root
    kwargs["wandb_project"] = None
    kwargs["is_training"] = 0
    experiment = iReflowExp(**kwargs)
    reproducible(seed)
    experiment._setup_run(seed)
    return experiment


def state_dict_from_checkpoint(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} does not contain a model state dict.")
    return payload


class OnlineMoments:
    """Streaming raw moments for flattened residual skewness and kurtosis."""

    def __init__(self):
        self.count = 0
        self.sums = np.zeros(4, dtype=np.float64)

    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().double().reshape(-1)
        self.count += int(flat.numel())
        for power in range(1, 5):
            self.sums[power - 1] += float(flat.pow(power).sum().cpu())

    def shape(self) -> Tuple[float, float]:
        if self.count == 0:
            return float("nan"), float("nan")
        raw = self.sums / self.count
        mean = raw[0]
        variance = raw[1] - mean ** 2
        if variance <= 0:
            return float("nan"), float("nan")
        central_3 = raw[2] - 3 * mean * raw[1] + 2 * mean ** 3
        central_4 = raw[3] - 4 * mean * raw[2] + 6 * mean ** 2 * raw[1] - 3 * mean ** 4
        return float(central_3 / variance ** 1.5), float(central_4 / variance ** 2 - 3.0)


def energy_score(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Exact empirical energy score, evaluated one ensemble member at a time."""
    flat_samples = samples.flatten(start_dim=2)
    flat_truth = truth.flatten(start_dim=1)
    first = torch.linalg.vector_norm(flat_samples - flat_truth.unsqueeze(1), dim=-1).mean(dim=1)
    pair_sum = torch.zeros_like(first)
    for index in range(flat_samples.shape[1]):
        distances = torch.linalg.vector_norm(flat_samples[:, index:index + 1] - flat_samples, dim=-1)
        pair_sum += distances.sum(dim=1)
    second = pair_sum / (flat_samples.shape[1] ** 2)
    return first - 0.5 * second


def _batch_to_device(batch, device: torch.device):
    batch_x, batch_y, _, origin_y, batch_x_date, _ = batch
    return (
        batch_x.to(device).float(), batch_y.to(device).float(), origin_y.to(device).float(), batch_x_date.to(device).float()
    )


@torch.no_grad()
def evaluate(
    model, experiment: iReflowExp, dataloader, design: str, num_samples: int, temperature: float
) -> Dict[str, float]:
    totals = {"crps": 0.0, "crps_sum": 0.0, "crps_sum_denom": 0.0, "mse": 0.0, "mae": 0.0, "es": 0.0}
    point_count = 0
    window_count = 0
    coverage_counts = {0.90: 0.0, 0.95: 0.0, 0.99: 0.0}
    width_sums = {0.90: 0.0, 0.95: 0.0, 0.99: 0.0}
    before, after = OnlineMoments(), OnlineMoments()

    model.eval()
    for batch in dataloader:
        batch_x, batch_y, origin_y, batch_x_date = _batch_to_device(batch, experiment.device)
        if design == "l2":
            samples, mu, sigma, _, _ = model.forecast(batch_x, batch_x_date, num_samples=num_samples, temperature=temperature)
            z2 = (batch_y - mu) / sigma
            z3 = z2
        else:
            samples, mu, sigma, alpha = model.forecast(batch_x, batch_x_date, num_samples=num_samples, temperature=temperature)
            z2 = (batch_y - mu) / sigma
            z3 = model.transform(z2, alpha)
        before.update(z2)
        after.update(z3)

        prediction = samples.permute(0, 2, 3, 1).contiguous()
        truth = batch_y
        if experiment.invtrans_loss:
            prediction = experiment.scaler.inverse_transform(prediction)
            truth = origin_y
        samples_eval = prediction.permute(0, 3, 1, 2).contiguous()
        prediction_cpu = prediction.detach().cpu()
        truth_cpu = truth.detach().cpu()
        prediction_np = prediction_cpu.reshape(-1, num_samples).numpy()
        truth_np = truth_cpu.reshape(-1).numpy()
        totals["crps"] += float(ps.crps_ensemble(truth_np, prediction_np).sum())
        summed_prediction = prediction_cpu.sum(dim=2).reshape(-1, num_samples).numpy()
        summed_truth = truth_cpu.sum(dim=2).reshape(-1).numpy()
        totals["crps_sum"] += float(ps.crps_ensemble(summed_truth, summed_prediction).sum())
        totals["crps_sum_denom"] += float(np.abs(summed_truth).sum())
        ensemble_mean = samples_eval.mean(dim=1)
        totals["mse"] += float((ensemble_mean - truth).square().sum().cpu())
        totals["mae"] += float((ensemble_mean - truth).abs().sum().cpu())
        totals["es"] += float(energy_score(samples_eval, truth).sum().cpu())
        point_count += int(truth.numel())
        window_count += int(truth.shape[0])

        for level in coverage_counts:
            lower = torch.quantile(samples_eval, (1.0 - level) / 2.0, dim=1)
            upper = torch.quantile(samples_eval, 1.0 - (1.0 - level) / 2.0, dim=1)
            coverage_counts[level] += float(((truth >= lower) & (truth <= upper)).sum().cpu())
            width_sums[level] += float((upper - lower).sum().cpu())

    skew_before, kurt_before = before.shape()
    skew_after, kurt_after = after.shape()
    result = {
        "CRPS": totals["crps"] / point_count,
        "CRPS_sum": totals["crps_sum"] / totals["crps_sum_denom"] if totals["crps_sum_denom"] else float("nan"),
        "ES": totals["es"] / window_count,
        "MSE": totals["mse"] / point_count,
        "MAE": totals["mae"] / point_count,
        "skewness_before": skew_before,
        "skewness_after": skew_after,
        "excess_kurtosis_before": kurt_before,
        "excess_kurtosis_after": kurt_after,
    }
    for level in coverage_counts:
        suffix = str(int(level * 100))
        coverage = coverage_counts[level] / point_count
        result[f"coverage_{suffix}"] = coverage
        result[f"coverage_error_{suffix}"] = abs(coverage - level)
        result[f"interval_width_{suffix}"] = width_sums[level] / point_count
    return result


def validation_loss(model: ThirdOrderShapeModel, experiment: iReflowExp, nll_weight: float) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    model.eval()
    with torch.no_grad():
        for batch in experiment.val_loader:
            batch_x, batch_y, _, batch_x_date = _batch_to_device(batch, experiment.device)
            _, diagnostics = model.compute_loss(batch_x, batch_x_date, batch_y, nll_weight=nll_weight)
            for key, value in diagnostics.items():
                values.setdefault(key, []).append(value)
    return {key: float(np.mean(value)) for key, value in values.items()}


def train_l3(
    model: ThirdOrderShapeModel, experiment: iReflowExp, seed: int, output_dir: Path, nll_weight: float
) -> Tuple[Path, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "l3_best_model.pth"
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=experiment.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=experiment.lr_patience)
    best = float("inf")
    wait = 0
    best_diagnostics: Dict[str, float] = {}
    for epoch in range(experiment.epochs):
        reproducible(seed + epoch)
        model.train()
        for batch in experiment.train_loader:
            batch_x, batch_y, _, batch_x_date = _batch_to_device(batch, experiment.device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.compute_loss(batch_x, batch_x_date, batch_y, nll_weight=nll_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), experiment.max_grad_norm)
            optimizer.step()
        diagnostics = validation_loss(model, experiment, nll_weight)
        scheduler.step(diagnostics["total_loss"])
        if diagnostics["total_loss"] < best:
            best = diagnostics["total_loss"]
            wait = 0
            best_diagnostics = diagnostics
            torch.save({"model": model.state_dict(), "validation": diagnostics, "epoch": epoch}, best_path)
        else:
            wait += 1
            if wait >= experiment.patience:
                break
    if not best_path.exists():
        raise RuntimeError("L3 training did not produce a checkpoint.")
    model.load_state_dict(torch.load(best_path, map_location=experiment.device, weights_only=True)["model"])
    model.freeze_l2()
    return best_path, best_diagnostics


def row_base(dataset: str, entry: Mapping[str, object], design: str, commit: str) -> Dict[str, object]:
    row = {column: "" for column in CSV_COLUMNS}
    row.update({
        "dataset": dataset, "seed": entry["l3_seed"], "design": design, "checkpoint": entry["checkpoint"],
        "commit": commit, "config_path": entry["config_path"], "run_id": entry.get("run_id", ""),
        "l3_seed": entry["l3_seed"], "status": "success",
    })
    return row


def write_outputs(
    rows: Sequence[Mapping[str, object]], output_root: Path, datasets: Sequence[str] = DATASETS, runs_per_dataset: int = 1
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "shape_decomposition_per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Third-order shape decomposition pilot", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    numeric = [column for column in REQUIRED_COLUMNS if column not in {"dataset", "seed", "design", "checkpoint", "commit", "config_path"}]
    for dataset in datasets:
        lines.extend([f"## {dataset}", "", "| Design | Runs | " + " | ".join(numeric) + " |", "|---|---:|" + "---:|" * len(numeric)])
        for design in ("l2", "l3"):
            selected = [row for row in rows if row["dataset"] == dataset and row["design"] == design and row["status"] == "success"]
            cells = []
            for column in numeric:
                values = np.asarray([float(row[column]) for row in selected], dtype=float) if selected else np.asarray([])
                cells.append("—" if not len(values) else f"{values.mean():.6g} ± {values.std(ddof=0):.3g}")
            lines.append(f"| {design.upper()} | {len(selected)}/{runs_per_dataset} | " + " | ".join(cells) + " |")
        failures = [row for row in rows if row["dataset"] == dataset and row["status"] != "success"]
        if failures:
            lines.extend(["", "Failures:"] + [f"- {row['design']}/{row['run_id']}: {row['failure_reason']}" for row in failures])
        lines.append("")
    lines.extend(["## Interpretation boundary", "", "This pilot concerns conditional skewness only. It does not claim heavy-tail modeling or general superiority of L3 over L2.", ""])
    (output_root / "shape_decomposition.md").write_text("\n".join(lines), encoding="utf-8")


def run_entry(repo_root: Path, dataset: str, run_id: str, entry: Mapping[str, object], args, commit: str) -> List[Dict[str, object]]:
    from src.models.third_order_shape import ThirdOrderShapeModel

    mutable_entry = dict(entry)
    mutable_entry["run_id"] = run_id
    seed = int(entry["l3_seed"])
    experiment = build_experiment(repo_root / str(entry["config_path"]), args.data_root, seed)
    checkpoint = repo_root / str(entry["checkpoint"])
    experiment.model.load_state_dict(state_dict_from_checkpoint(checkpoint, experiment.device), strict=True)
    l2_row = row_base(dataset, mutable_entry, "l2", commit)
    reproducible(seed + 10_000)
    l2_row.update(evaluate(experiment.model, experiment, experiment.test_loader, "l2", experiment.num_samples, experiment.temperature))
    l3_row = row_base(dataset, mutable_entry, "l3", commit)
    try:
        l3_model = ThirdOrderShapeModel(experiment.model, experiment.model_configs).to(experiment.device)
        if l3_model.generator_parameter_count() != l3_model.l2_generator_parameter_count():
            raise AssertionError("L3 generator capacity differs from the frozen L2 generator.")
        output_dir = repo_root / args.run_root / dataset / run_id
        l3_checkpoint, validation = train_l3(l3_model, experiment, seed, output_dir, args.nll_weight)
        reproducible(seed + 10_000)
        l3_row.update(evaluate(l3_model, experiment, experiment.test_loader, "l3", experiment.num_samples, experiment.temperature))
        l3_row.update({
            "l3_checkpoint": str(l3_checkpoint.relative_to(repo_root)), "l3_parameters": l3_model.trainable_parameter_count(),
            "validation_alpha_nll": validation.get("alpha_nll", ""),
            "validation_mean_abs_alpha": validation.get("mean_abs_alpha", ""),
        })
        (output_dir / "metadata.json").write_text(json.dumps({
            "dataset": dataset, "run_id": run_id, "l2_checkpoint": entry["checkpoint"], "l2_sha256": entry["sha256"],
            "config_path": entry["config_path"], "l3_seed": seed, "nll_weight": args.nll_weight,
            "l3_trainable_parameters": l3_model.trainable_parameter_count(), "validation": validation, "commit": commit,
        }, indent=2), encoding="utf-8")
    except Exception as error:
        l3_row.update({"status": "failed", "failure_reason": "".join(traceback.format_exception_only(type(error), error)).strip()})
    return [l2_row, l3_row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/third_order_shape_manifest.yaml")
    parser.add_argument("--data_root", default="./data/")
    parser.add_argument("--output_root", default="evidence")
    parser.add_argument("--run_root", default="results/third_order_shape")
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--runs_per_dataset", type=int, default=1)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    set_process_title(args.datasets, args.runs_per_dataset)
    repo_root = Path(__file__).resolve().parents[2]
    selected_run_ids = run_ids(args.runs_per_dataset)
    entries = load_manifest(repo_root / args.manifest, repo_root, args.datasets, selected_run_ids)
    commit = git_commit(repo_root)
    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        for run_id in selected_run_ids:
            entry = entries[dataset][run_id]
            try:
                rows.extend(run_entry(repo_root, dataset, run_id, entry, args, commit))
            except Exception as error:
                reason = "".join(traceback.format_exception_only(type(error), error)).strip()
                # Setup/checkpoint failures prevent both designs from being
                # evaluated, so retain an explicit auditable row for each.
                for design in ("l2", "l3"):
                    failure = row_base(dataset, {**entry, "run_id": run_id}, design, commit)
                    failure.update({"status": "failed", "failure_reason": reason})
                    rows.append(failure)
                write_outputs(rows, repo_root / args.output_root, args.datasets, args.runs_per_dataset)
                print(f"FAILED {dataset}/{run_id}: {error}", flush=True)
    write_outputs(rows, repo_root / args.output_root, args.datasets, args.runs_per_dataset)


if __name__ == "__main__":
    main()
