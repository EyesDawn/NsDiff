"""
Export aligned forecast samples from an existing experiment run directory.

This helper reconstructs an experiment instance from `<run_dir>/args.json`,
loads the run's checkpoint(s), and writes a normalized `.npz` artifact for the
distribution-level decoupling analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional, Type

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CLASS_REGISTRY: Dict[str, str] = {
    "TimeGrad": "src.experiments.TimeGrad:TimeGradForecast",
    "CSDI": "src.experiments.CSDI:CSDIForecast",
    "TimeDiff": "src.experiments.TimeDiff:TimeDiffForecast",
    "NsDiff4": "src.experiments.NsDiff:NsDiffForecast",
    "NsDiff": "src.experiments.NsDiff:NsDiffForecast",
    "TMDM": "src.experiments.TMDM:TMDMForecast",
    "iReflow": "src.experiments.iReflow:iReflowExp",
    "PDN-Flow": "src.experiments.iReflow:iReflowExp",
}


def _load_args(run_dir: str) -> Dict[str, Any]:
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError("Missing args.json under run directory: %s" % run_dir)
    with open(args_path, "r", encoding="utf-8") as f:
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


def export_from_run(
    run_dir: str,
    output_path: str,
    seed: int,
    use_origin_scale: bool,
    eps: float,
    max_windows: Optional[int],
    device: Optional[str],
    num_worker: Optional[int],
    batch_size: Optional[int],
    num_samples: Optional[int],
) -> Optional[Dict[str, Any]]:
    run_dir = os.path.abspath(run_dir)
    config = _load_args(run_dir)
    model_type = str(config.get("model_type"))
    cls = _resolve_cls(model_type)
    init_kwargs = _filter_init_kwargs(cls, config)

    if device is not None:
        init_kwargs["device"] = device
    if num_worker is not None and "num_worker" in init_kwargs:
        init_kwargs["num_worker"] = num_worker
    if batch_size is not None and "batch_size" in init_kwargs:
        init_kwargs["batch_size"] = batch_size
    if num_samples is not None and "num_samples" in init_kwargs:
        init_kwargs["num_samples"] = num_samples

    exp = cls(**init_kwargs)
    return exp.export_forecast_samples_on_test(
        seed=seed,
        save_path=output_path,
        use_origin_scale=use_origin_scale,
        eps=eps,
        max_windows=max_windows,
        run_dir_override=run_dir,
        return_result=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export forecast samples from an existing run directory whose args.json "
            "matches one of the probabilistic forecasting experiment classes."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_worker", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--normalized_scale", action="store_true")

    args = parser.parse_args()
    export_from_run(
        run_dir=args.run_dir,
        output_path=args.output_path,
        seed=args.seed,
        use_origin_scale=not args.normalized_scale,
        eps=args.eps,
        max_windows=args.max_windows,
        device=args.device,
        num_worker=args.num_worker,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
