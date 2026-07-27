"""
Evaluate the simple Gaussian source distribution used by iReflow/LS-Flow.

This script bypasses the velocity network and evaluates the Gaussian

    Y ~ Normal(y_hat, sigma^2)

analytically. CRPS is computed with the closed-form Gaussian expression, and
CRPSsum is computed from the summed independent Gaussian, whose mean and
variance are:

    mu_sum = sum_i mu_i
    var_sum = sum_i sigma_i^2

MSE and MAE are computed on the predictive mean y_hat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
    "iReflow_DLinear": "src.experiments.iReflow_DLinear:iReflowDLinearExp",
}

DEFAULT_DATASETS = ("ETTh1", "ETTh2", "Weather", "Electricity")
CHECKPOINT_NAMES = ("best_model.pth",)
TRAIN_MODE_DIRS = ("train_mode_2", "train_mode_1")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_cls(model_name: str) -> Type[Any]:
    if model_name not in CLASS_REGISTRY:
        raise KeyError(
            "Unsupported model_type `%s`. Supported keys: %s"
            % (model_name, sorted(CLASS_REGISTRY.keys()))
        )
    module_name, cls_name = CLASS_REGISTRY[model_name].split(":")
    module = __import__(module_name, fromlist=[cls_name])
    return getattr(module, cls_name)


def _filter_init_kwargs(cls: Type[Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not is_dataclass(cls):
        return dict(config)
    valid_names = {field.name for field in fields(cls)}
    return {key: value for key, value in config.items() if key in valid_names}


def _dataset_from_config(config: Dict[str, Any]) -> str:
    return str(config.get("dataset_type") or config.get("data") or config.get("dataset") or "")


def _seed_from_run_dir(run_dir: Path, default: int) -> int:
    for part in reversed(run_dir.parts):
        if part.startswith("seed_"):
            try:
                return int(part.split("seed_", 1)[1])
            except ValueError:
                return default
    return default


def _find_args_json(run_dir: Path) -> Path:
    candidates = [
        run_dir / "args.json",
        run_dir.parent / "args.json",
        run_dir / "train_mode_2" / "args.json",
        run_dir / "train_mode_1" / "args.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find args.json for run directory `%s`. "
        "Pass a seed/train_mode directory or a parent containing train_mode_*/args.json."
        % run_dir
    )


def _find_checkpoint_dir(run_dir: Path, checkpoint_mode: Optional[int]) -> Path:
    if checkpoint_mode is not None:
        mode_name = "train_mode_%d" % checkpoint_mode
        candidate = run_dir / mode_name
        if (candidate / "best_model.pth").is_file():
            return candidate
        if run_dir.name == mode_name and (run_dir / "best_model.pth").is_file():
            return run_dir
        if run_dir.name in TRAIN_MODE_DIRS:
            sibling = run_dir.parent / mode_name
            if (sibling / "best_model.pth").is_file():
                return sibling
        raise FileNotFoundError("Could not find train_mode_%d/best_model.pth under %s" % (checkpoint_mode, run_dir))

    for name in CHECKPOINT_NAMES:
        if (run_dir / name).is_file():
            return run_dir

    for mode_dir in TRAIN_MODE_DIRS:
        candidate = run_dir / mode_dir
        if (candidate / "best_model.pth").is_file():
            return candidate

    if run_dir.name in TRAIN_MODE_DIRS and (run_dir / "best_model.pth").is_file():
        return run_dir

    raise FileNotFoundError(
        "Could not find best_model.pth under `%s` or its train_mode_2/train_mode_1 children."
        % run_dir
    )


def _discover_run_dirs(
    runs_root: Path,
    datasets: Sequence[str],
    checkpoint_mode: Optional[int],
) -> List[Path]:
    dataset_set = set(datasets)
    discovered: List[Path] = []

    for args_path in runs_root.rglob("args.json"):
        try:
            config = _load_json(args_path)
        except json.JSONDecodeError:
            continue

        model_type = str(config.get("model_type") or config.get("model") or "")
        if model_type not in CLASS_REGISTRY:
            continue

        dataset = _dataset_from_config(config)
        if dataset not in dataset_set:
            continue

        run_dir = args_path.parent
        try:
            checkpoint_dir = _find_checkpoint_dir(run_dir, checkpoint_mode=checkpoint_mode)
        except FileNotFoundError:
            continue
        discovered.append(checkpoint_dir)

    return sorted(set(discovered), key=lambda p: str(p))


def _init_experiment_from_run(
    run_dir: Path,
    device: Optional[str],
    batch_size: Optional[int],
    num_worker: Optional[int],
    eval_micro_batch_size: Optional[int],
    num_samples: Optional[int],
    seed: int,
) -> Tuple[Any, Dict[str, Any], Path]:
    args_path = _find_args_json(run_dir)
    config = _load_json(args_path)
    model_type = str(config.get("model_type") or config.get("model") or "iReflow")
    cls = _resolve_cls(model_type)
    init_kwargs = _filter_init_kwargs(cls, config)

    if device is not None:
        init_kwargs["device"] = device
    if batch_size is not None and "batch_size" in init_kwargs:
        init_kwargs["batch_size"] = batch_size
    if num_worker is not None and "num_worker" in init_kwargs:
        init_kwargs["num_worker"] = num_worker
    if eval_micro_batch_size is not None and "eval_micro_batch_size" in init_kwargs:
        init_kwargs["eval_micro_batch_size"] = eval_micro_batch_size
    if num_samples is not None and "num_samples" in init_kwargs:
        init_kwargs["num_samples"] = num_samples

    exp = cls(**init_kwargs)
    exp.current_seed = seed

    # Force a single-process dataloader by default for analysis jobs.
    resolved_num_worker = 0 if num_worker is None else int(num_worker)
    if hasattr(exp, "num_worker"):
        exp.num_worker = resolved_num_worker

    checkpoint_dir = _find_checkpoint_dir(run_dir, checkpoint_mode=None)
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


def _gaussian_crps_sum(
    true: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    sigma = sigma.clamp_min(eps)
    z = (true - mean) / sigma
    normal = torch.distributions.Normal(
        loc=torch.zeros((), device=true.device, dtype=true.dtype),
        scale=torch.ones((), device=true.device, dtype=true.dtype),
    )
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.sum()


def _scale_sigma_to_origin(exp: Any, sigma: torch.Tensor) -> torch.Tensor:
    if not hasattr(exp, "_get_scaler_mean_std"):
        return sigma
    _, std = exp._get_scaler_mean_std(dtype=sigma.dtype, device=sigma.device)
    if std is None:
        return sigma
    return sigma * std.view(1, 1, -1)


def _iter_micro_batches(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    origin_y: torch.Tensor,
    batch_x_date_enc: torch.Tensor,
    micro_batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch_size = batch_x.shape[0]
    micro_batch_size = max(1, min(int(micro_batch_size), batch_size))
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        yield (
            batch_x[start:end],
            batch_y[start:end],
            origin_y[start:end],
            batch_x_date_enc[start:end],
        )


@torch.no_grad()
def evaluate_gaussian_resampling(
    run_dir: Path,
    seed: int,
    device: Optional[str],
    batch_size: Optional[int],
    num_worker: Optional[int],
    eval_micro_batch_size: Optional[int],
    num_samples: Optional[int],
    max_batches: Optional[int],
) -> Dict[str, Any]:
    _set_seed(seed)
    exp, config, checkpoint_dir = _init_experiment_from_run(
        run_dir=run_dir,
        device=device,
        batch_size=batch_size,
        num_worker=num_worker,
        eval_micro_batch_size=eval_micro_batch_size,
        num_samples=num_samples,
        seed=seed,
    )

    samples_count = int(num_samples if num_samples is not None else getattr(exp, "num_samples", 100))
    micro_batch_size = int(
        eval_micro_batch_size
        if eval_micro_batch_size is not None
        else getattr(exp, "eval_micro_batch_size", getattr(exp, "batch_size", 32))
    )

    dataset = _dataset_from_config(config) or str(getattr(exp, "dataset_type", ""))
    if num_samples is not None:
        print(
            "Warning: --num_samples is ignored in analytic Gaussian mode; "
            "it is kept only for CLI compatibility."
        )

    total_points = 0
    total_crps = 0.0
    total_crps_sum = 0.0
    total_crps_sum_denom = 0.0
    total_mse = 0.0
    total_mae = 0.0

    total_windows = len(exp.test_loader.dataset)
    seen_batches = 0
    with tqdm(total=total_windows, desc=f"{dataset or checkpoint_dir.name}", leave=False) as progress_bar:
        for batch in exp.test_loader:
            batch_x, batch_y, _origin_x, origin_y, batch_x_date_enc, _batch_y_date_enc = batch
            seen_batches += 1

            for mb_x, mb_y, mb_origin_y, mb_x_date_enc in _iter_micro_batches(
                batch_x=batch_x,
                batch_y=batch_y,
                origin_y=origin_y,
                batch_x_date_enc=batch_x_date_enc,
                micro_batch_size=micro_batch_size,
            ):
                mb_x = mb_x.to(exp.device).float()
                mb_y = mb_y.to(exp.device).float()
                mb_origin_y = mb_origin_y.to(exp.device).float()
                mb_x_date_enc = mb_x_date_enc.to(exp.device).float()

                _enc_features, y_hat, sigma = exp.model.get_encoder_features(mb_x, mb_x_date_enc)
                truths = mb_y

                if getattr(exp, "invtrans_loss", False):
                    truths = mb_origin_y
                    y_hat = exp._inverse_transform_last_dim(y_hat)
                    sigma = _scale_sigma_to_origin(exp, sigma)

                sigma = sigma.clamp_min(1e-6)
                total_points += truths.numel()
                total_crps += float(_gaussian_crps_sum(truths, y_hat, sigma).item())

                true_sum = truths.sum(dim=2)
                mean_sum = y_hat.sum(dim=2)
                sigma_sum = sigma.square().sum(dim=2).sqrt()
                total_crps_sum += float(_gaussian_crps_sum(true_sum, mean_sum, sigma_sum).item())
                total_crps_sum_denom += float(true_sum.abs().sum().item())

                diff = y_hat - truths
                total_mse += float(diff.square().sum().item())
                total_mae += float(diff.abs().sum().item())
                progress_bar.update(mb_x.shape[0])

            if max_batches is not None and seen_batches >= max_batches:
                break

    if total_points == 0:
        raise RuntimeError("No test points were evaluated.")

    result = {
        "crps": total_crps / float(total_points),
        "crps_sum": (
            total_crps_sum / total_crps_sum_denom if total_crps_sum_denom > 0.0 else 0.0
        ),
        "mse": total_mse / float(total_points),
        "mae": total_mae / float(total_points),
    }
    result.update(
        {
            "data": dataset,
            "seed": seed,
            "num_samples": samples_count,
            "run_dir": str(checkpoint_dir),
        }
    )
    return result


CSV_COLUMNS = ["data", "seed", "num_samples", "crps", "crps_sum", "mse", "mae", "run_dir"]


def _append_csv_row(row: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _print_summary(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    print("\nGaussian resampling results:")
    for row in rows:
        print(
            "{data} seed={seed} S={num_samples} "
            "CRPS={crps:.6f} CRPSsum={crps_sum:.6f} "
            "MSE={mse:.6f} MAE={mae:.6f}".format(**row)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resample the iReflow/LS-Flow conditioner Gaussian N(y_hat, sigma^2) "
            "and evaluate CRPS, CRPSsum, MSE, and MAE on the test set."
        )
    )
    parser.add_argument("--run_dirs", nargs="*", default=None, help="Explicit iReflow run directories.")
    parser.add_argument("--runs_root", type=str, default="./results/runs", help="Root used by --discover.")
    parser.add_argument("--discover", action="store_true", help="Discover runs under --runs_root.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--checkpoint_mode", type=int, choices=[1, 2], default=None)
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed. Defaults to seed_* in run_dir or 42.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--num_worker",
        type=int,
        default=0,
        help="Dataloader workers for analysis. Default 0 avoids multiprocessing issues.",
    )
    parser.add_argument("--eval_micro_batch_size", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Debug only: limit evaluated test batches.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./results/analysis/lsflow_gaussian_resampling_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs: List[Path] = []
    output_path = Path(args.output_csv)

    if args.run_dirs:
        run_dirs.extend(Path(path).resolve() for path in args.run_dirs)

    if args.discover:
        run_dirs.extend(
            _discover_run_dirs(
                runs_root=Path(args.runs_root).resolve(),
                datasets=args.datasets,
                checkpoint_mode=args.checkpoint_mode,
            )
        )

    run_dirs = sorted(set(run_dirs), key=lambda p: str(p))
    if not run_dirs:
        raise SystemExit(
            "No run directories found. Pass --run_dirs or use --discover with a populated --runs_root."
        )

    rows = []
    for run_dir in run_dirs:
        checkpoint_dir = _find_checkpoint_dir(run_dir, checkpoint_mode=args.checkpoint_mode)
        seed = int(args.seed if args.seed is not None else _seed_from_run_dir(checkpoint_dir, 42))
        row = evaluate_gaussian_resampling(
            run_dir=checkpoint_dir,
            seed=seed,
            device=args.device,
            batch_size=args.batch_size,
            num_worker=args.num_worker,
            eval_micro_batch_size=args.eval_micro_batch_size,
            num_samples=args.num_samples,
            max_batches=args.max_batches,
        )
        rows.append(row)
        _append_csv_row(row, output_path)

    _print_summary(rows)
    print("\nSaved CSV:", args.output_csv)


if __name__ == "__main__":
    main()
