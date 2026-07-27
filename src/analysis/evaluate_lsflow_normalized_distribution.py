"""Measure LS-Flow predictive-Gaussian normalized target statistics on test data.

For each test target point, this script evaluates the conditioner Gaussian
parameters and computes the probability-distribution-normalized residual

    z = (y - predictive_mean) / predictive_sigma.

It reports the global population mean and variance of ``z`` for each supplied
checkpoint. Statistics are accumulated in float64, so the entire test set does
not need to be retained in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


CLASS_REGISTRY: Dict[str, str] = {
    "iReflow": "src.experiments.iReflow:iReflowExp",
    "PDN-Flow": "src.experiments.iReflow:iReflowExp",
    "LS-Flow": "src.experiments.iReflow:iReflowExp",
}
CHECKPOINT_NAMES = ("best_model.pth",)
TRAIN_MODE_DIRS = ("train_mode_2", "train_mode_1")
SIGMA_EPS = 1e-6
CSV_COLUMNS = [
    "data",
    "seed",
    "normalized_mean",
    "normalized_variance",
    "num_points",
    "run_dir",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _dataset_from_config(config: Dict[str, Any]) -> str:
    return str(config.get("dataset_type") or config.get("data") or config.get("dataset") or "")


def _resolve_cls(model_name: str) -> Type[Any]:
    if model_name not in CLASS_REGISTRY:
        raise KeyError(f"Unsupported model type `{model_name}`. Supported: {sorted(CLASS_REGISTRY)}")
    module_name, cls_name = CLASS_REGISTRY[model_name].split(":")
    module = __import__(module_name, fromlist=[cls_name])
    return getattr(module, cls_name)


def _filter_init_kwargs(cls: Type[Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not is_dataclass(cls):
        return dict(config)
    valid_names = {field.name for field in fields(cls)}
    return {key: value for key, value in config.items() if key in valid_names}


def _find_args_json(run_dir: Path) -> Path:
    for path in (
        run_dir / "args.json",
        run_dir.parent / "args.json",
        run_dir / "train_mode_2" / "args.json",
        run_dir / "train_mode_1" / "args.json",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find args.json for run directory `{run_dir}`.")


def _find_checkpoint_dir(run_dir: Path) -> Path:
    for name in CHECKPOINT_NAMES:
        if (run_dir / name).is_file():
            return run_dir
    for mode_dir in TRAIN_MODE_DIRS:
        candidate = run_dir / mode_dir
        if (candidate / "best_model.pth").is_file():
            return candidate
    if run_dir.name in TRAIN_MODE_DIRS and (run_dir / "best_model.pth").is_file():
        return run_dir
    raise FileNotFoundError(f"Could not find best_model.pth under `{run_dir}`.")


def _scale_sigma_to_origin(exp: Any, sigma: torch.Tensor) -> torch.Tensor:
    if not hasattr(exp, "_get_scaler_mean_std"):
        return sigma
    _, std = exp._get_scaler_mean_std(dtype=sigma.dtype, device=sigma.device)
    if std is None:
        return sigma
    return sigma * std.view(1, 1, -1)


def _init_experiment_from_run(
    run_dir: Path,
    seed: int,
    device: Optional[str],
    batch_size: Optional[int],
    num_worker: int,
    eval_micro_batch_size: Optional[int],
) -> Tuple[Any, Dict[str, Any], Path]:
    config = _load_json(_find_args_json(run_dir))
    model_type = str(config.get("model_type") or config.get("model") or "iReflow")
    init_kwargs = _filter_init_kwargs(_resolve_cls(model_type), config)

    if device is not None:
        init_kwargs["device"] = device
    if batch_size is not None and "batch_size" in init_kwargs:
        init_kwargs["batch_size"] = batch_size
    if "num_worker" in init_kwargs:
        init_kwargs["num_worker"] = num_worker
    if eval_micro_batch_size is not None and "eval_micro_batch_size" in init_kwargs:
        init_kwargs["eval_micro_batch_size"] = eval_micro_batch_size

    exp = _resolve_cls(model_type)(**init_kwargs)
    exp.current_seed = seed
    if hasattr(exp, "num_worker"):
        exp.num_worker = num_worker

    checkpoint_dir = _find_checkpoint_dir(run_dir)
    exp.run_save_dir = str(checkpoint_dir)
    exp.best_checkpoint_filepath = str(checkpoint_dir / "best_model.pth")
    exp.run_checkpoint_filepath = str(checkpoint_dir / "run_checkpoint.pth")
    try:
        exp._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
    except TypeError:
        exp._init_data_loader()
    exp._init_model()
    exp._load_best_model()
    exp.model.eval()
    return exp, config, checkpoint_dir


def _iter_micro_batches(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    origin_y: torch.Tensor,
    batch_x_date_enc: torch.Tensor,
    micro_batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    for start in range(0, batch_x.shape[0], micro_batch_size):
        end = min(start + micro_batch_size, batch_x.shape[0])
        yield batch_x[start:end], batch_y[start:end], origin_y[start:end], batch_x_date_enc[start:end]


@torch.no_grad()
def evaluate_normalized_distribution(
    run_dir: Path,
    seed: int,
    device: Optional[str],
    batch_size: Optional[int],
    num_worker: int,
    eval_micro_batch_size: Optional[int],
    max_batches: Optional[int],
) -> Dict[str, Any]:
    _set_seed(seed)
    exp, config, checkpoint_dir = _init_experiment_from_run(
        run_dir, seed, device, batch_size, num_worker, eval_micro_batch_size
    )
    micro_batch_size = max(
        1,
        int(
            eval_micro_batch_size
            if eval_micro_batch_size is not None
            else getattr(exp, "eval_micro_batch_size", getattr(exp, "batch_size", 32))
        ),
    )
    dataset = _dataset_from_config(config) or str(getattr(exp, "dataset_type", ""))

    num_points = 0
    sum_z = 0.0
    sum_z_squared = 0.0
    seen_batches = 0
    total_windows = len(exp.test_loader.dataset)
    with tqdm(total=total_windows, desc=dataset or checkpoint_dir.name, leave=False) as progress_bar:
        for batch in exp.test_loader:
            batch_x, batch_y, _origin_x, origin_y, batch_x_date_enc, _batch_y_date_enc = batch
            seen_batches += 1
            for mb_x, mb_y, mb_origin_y, mb_x_date_enc in _iter_micro_batches(
                batch_x, batch_y, origin_y, batch_x_date_enc, micro_batch_size
            ):
                mb_x = mb_x.to(exp.device).float()
                mb_y = mb_y.to(exp.device).float()
                mb_origin_y = mb_origin_y.to(exp.device).float()
                mb_x_date_enc = mb_x_date_enc.to(exp.device).float()

                _features, predictive_mean, predictive_sigma = exp.model.get_encoder_features(
                    mb_x, mb_x_date_enc
                )
                target = mb_y
                if getattr(exp, "invtrans_loss", False):
                    target = mb_origin_y
                    predictive_mean = exp._inverse_transform_last_dim(predictive_mean)
                    predictive_sigma = _scale_sigma_to_origin(exp, predictive_sigma)

                normalized = (target - predictive_mean) / predictive_sigma.clamp_min(SIGMA_EPS)
                if not torch.isfinite(normalized).all():
                    raise FloatingPointError(
                        f"Non-finite normalized targets encountered in `{checkpoint_dir}`."
                    )
                normalized = normalized.to(dtype=torch.float64)
                num_points += normalized.numel()
                sum_z += normalized.sum().item()
                sum_z_squared += normalized.square().sum().item()
                progress_bar.update(mb_x.shape[0])

            if max_batches is not None and seen_batches >= max_batches:
                break

    if num_points == 0:
        raise RuntimeError("No test points were evaluated.")
    normalized_mean = sum_z / num_points
    normalized_variance = max(0.0, sum_z_squared / num_points - normalized_mean**2)
    return {
        "data": dataset,
        "seed": seed,
        "normalized_mean": normalized_mean,
        "normalized_variance": normalized_variance,
        "num_points": num_points,
        "run_dir": str(checkpoint_dir),
    }


def _append_csv_row(row: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in CSV_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate global test-set mean and population variance of LS-Flow normalized targets."
    )
    parser.add_argument("--run_dirs", nargs="+", required=True, help="iReflow run or checkpoint directories.")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--eval_micro_batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Debug only: limit test batches.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./results/analysis/lsflow_normalized_distribution_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    rows = []
    for raw_run_dir in args.run_dirs:
        row = evaluate_normalized_distribution(
            run_dir=Path(raw_run_dir).resolve(),
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            num_worker=args.num_worker,
            eval_micro_batch_size=args.eval_micro_batch_size,
            max_batches=args.max_batches,
        )
        _append_csv_row(row, output_path)
        rows.append(row)

    print("\nNormalized test-target distribution results:")
    for row in rows:
        print(
            "{data} seed={seed} N={num_points} mean={normalized_mean:.8f} "
            "variance={normalized_variance:.8f}".format(**row)
        )
    print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
