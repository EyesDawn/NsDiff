"""
Source-to-target conditional Wasserstein-1 bar analysis.

This experiment compares four methods under the source/target definitions:

1. TimeGrad:    source = N(0, I),                    target = Y in origin space
2. TMDM:        source = N(mu_Y_hat, I),             target = Y in origin space
3. NsDiff:      source = N(mu_Y_hat, sigma_Y_hat^2), target = Y in origin space
4. PDN-Flow:    source/target measured in PDN, scaled, or origin space

For each test window, we treat the observed target window as a Dirac conditional
target distribution and estimate W1(source | x, delta_target | x) with the
coordinate-wise L1 ground cost. The reported bar height is the mean conditional
W1 across windows, and the error bar is the standard deviation across windows.

The manifest supports explicit per-method run paths:
  - TimeGrad: `config_run_dir`
  - TMDM: `mu_run_dir`
  - NsDiff: `mu_run_dir` + `sigma_run_dir`
  - PDN-Flow: `run_dir`
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional, Sequence, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

try:
    import seaborn as sns
except ImportError:
    sns = None


FONT_SCALE = 1.2
PCA_TEXT_SIZE = 12 * FONT_SCALE

SOURCE_TARGET_METHOD_ORDER = ("TimeGrad", "TMDM", "NsDiff", "PDN-Flow")
SOURCE_TARGET_METHOD_COLORS = {
    "TimeGrad": "#4C78A8",
    "TMDM": "#ECA82C",
    "NsDiff": "#F58518",
    "PDN-Flow": "#D62728",
}
SOURCE_TARGET_METHOD_HATCHES = {
    "TimeGrad": "ooo",
    "TMDM": "////",
    "NsDiff": "xx",
    "PDN-Flow": "***",
}
SOURCE_TARGET_DISPLAY_ALIASES = {
    "NsDiff4": "NsDiff",
    "iReflow": "PDN-Flow",
}
SOURCE_TARGET_CLASS_REGISTRY: dict[str, str] = {
    "TimeGrad": "src.experiments.TimeGrad:TimeGradForecast",
    "TMDM": "src.experiments.TMDM:TMDMForecast",
    "NsDiff4": "src.experiments.NsDiff:NsDiffForecast",
    "NsDiff": "src.experiments.NsDiff:NsDiffForecast",
    "iReflow": "src.experiments.iReflow:iReflowExp",
    "PDN-Flow": "src.experiments.iReflow:iReflowExp",
    "F": "src.experiments.pretrain_f:FForecast",
    "G": "src.experiments.pretrain_g:GForecast",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12 * FONT_SCALE,
            "xtick.labelsize": 10.5 * FONT_SCALE,
            "ytick.labelsize": 10.5 * FONT_SCALE,
            "legend.fontsize": 10 * FONT_SCALE,
            "axes.titlesize": 12 * FONT_SCALE,
            "legend.title_fontsize": 10 * FONT_SCALE,
            "savefig.dpi": 600,
        }
    )
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="paper",
            rc={
                "axes.facecolor": "#FBFBFC",
                "figure.facecolor": "white",
                "grid.linestyle": "--",
                "grid.alpha": 0.20,
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class SourceTargetMethodSpec:
    name: str
    space: str
    display_name: str
    run_dir: str | None = None
    config_run_dir: str | None = None
    mu_run_dir: str | None = None
    sigma_run_dir: str | None = None


@dataclass
class SourceTargetDatasetSpec:
    name: str
    display_name: str
    methods: list[SourceTargetMethodSpec]


@dataclass
class SourceTargetReservoir:
    target: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    sampled_windows: int
    total_windows: int


@dataclass
class SourceTargetExperimentHandle:
    exp: Any
    model_type: str
    model_loaded: bool


def _mean_abs_per_window(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"Expected at least 2 dims to compute per-window magnitude, got {values.shape}.")
    reduce_axes = tuple(range(1, values.ndim))
    return np.mean(np.abs(values), axis=reduce_axes, dtype=np.float64)


def _normalize_source_target_method_name(name: str) -> str:
    return SOURCE_TARGET_DISPLAY_ALIASES.get(name, name)


def _default_source_target_space(method_name: str) -> str:
    normalized = _normalize_source_target_method_name(method_name)
    if normalized == "PDN-Flow":
        return "pdn"
    return "origin"


def _load_source_target_manifest(manifest_path: str) -> list[SourceTargetDatasetSpec]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    datasets_cfg = manifest.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("Manifest must contain a non-empty `datasets` list.")

    dataset_specs: list[SourceTargetDatasetSpec] = []
    for dataset_cfg in datasets_cfg:
        dataset_name = str(dataset_cfg.get("name", dataset_cfg.get("dataset_name", ""))).strip()
        if not dataset_name:
            raise ValueError("Each dataset entry must define `name` or `dataset_name`.")

        methods_cfg = dataset_cfg.get("methods")
        if methods_cfg is None:
            raise ValueError(f"Dataset `{dataset_name}` is missing `methods`.")
        if isinstance(methods_cfg, dict):
            normalized_methods_cfg = []
            for method_name, method_cfg in methods_cfg.items():
                item = {} if method_cfg is None else dict(method_cfg)
                item.setdefault("name", method_name)
                normalized_methods_cfg.append(item)
            methods_cfg = normalized_methods_cfg
        if not isinstance(methods_cfg, list) or not methods_cfg:
            raise ValueError(
                f"Dataset `{dataset_name}` must provide `methods` as a non-empty list or dict."
            )

        method_specs: list[SourceTargetMethodSpec] = []
        for method_cfg in methods_cfg:
            if "name" not in method_cfg:
                raise ValueError(f"Dataset `{dataset_name}` has a method entry without `name`.")
            raw_name = str(method_cfg["name"])
            display_name = str(
                method_cfg.get("display_name", _normalize_source_target_method_name(raw_name))
            )
            run_dir = str(method_cfg.get("run_dir", "")).strip() or None
            config_run_dir = str(method_cfg.get("config_run_dir", "")).strip() or None
            mu_run_dir = str(method_cfg.get("mu_run_dir", "")).strip() or None
            sigma_run_dir = str(method_cfg.get("sigma_run_dir", "")).strip() or None
            space = str(method_cfg.get("space", _default_source_target_space(display_name))).lower()

            if display_name == "PDN-Flow":
                if run_dir is None:
                    raise ValueError(
                        f"Dataset `{dataset_name}` / method `{display_name}` is missing `run_dir`."
                    )
            elif display_name == "TimeGrad":
                if config_run_dir is None and run_dir is None:
                    raise ValueError(
                        f"Dataset `{dataset_name}` / method `{display_name}` must provide "
                        "`config_run_dir` (or legacy `run_dir`)."
                    )
            elif display_name == "TMDM":
                if mu_run_dir is None and run_dir is None:
                    raise ValueError(
                        f"Dataset `{dataset_name}` / method `{display_name}` must provide "
                        "`mu_run_dir` (or legacy `run_dir`)."
                    )
            elif display_name == "NsDiff":
                has_new_paths = mu_run_dir is not None and sigma_run_dir is not None
                if not has_new_paths and run_dir is None:
                    raise ValueError(
                        f"Dataset `{dataset_name}` / method `{display_name}` must provide "
                        "`mu_run_dir` + `sigma_run_dir` (or legacy `run_dir`)."
                    )

            method_specs.append(
                SourceTargetMethodSpec(
                    name=raw_name,
                    space=space,
                    display_name=display_name,
                    run_dir=run_dir,
                    config_run_dir=config_run_dir,
                    mu_run_dir=mu_run_dir,
                    sigma_run_dir=sigma_run_dir,
                )
            )

        available_names = {spec.display_name for spec in method_specs}
        missing_names = [
            method_name for method_name in SOURCE_TARGET_METHOD_ORDER if method_name not in available_names
        ]
        if missing_names:
            raise ValueError(
                f"Dataset `{dataset_name}` is missing required methods: {missing_names}."
            )
        ordered_specs = sorted(
            method_specs,
            key=lambda spec: SOURCE_TARGET_METHOD_ORDER.index(spec.display_name),
        )
        dataset_specs.append(
            SourceTargetDatasetSpec(
                name=dataset_name,
                display_name=str(dataset_cfg.get("display_name", dataset_name)),
                methods=ordered_specs,
            )
        )
    return dataset_specs


def _load_run_args(run_dir: str) -> dict[str, Any]:
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Missing args.json under run directory: {run_dir}")
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_run_class(model_name: str) -> Type[Any]:
    if model_name not in SOURCE_TARGET_CLASS_REGISTRY:
        raise KeyError(
            f"Unsupported model_type `{model_name}`. Supported keys: {sorted(SOURCE_TARGET_CLASS_REGISTRY)}"
        )
    module_name, cls_name = SOURCE_TARGET_CLASS_REGISTRY[model_name].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _filter_init_kwargs_for_class(cls: Type[Any], config: dict[str, Any]) -> dict[str, Any]:
    if not is_dataclass(cls):
        return dict(config)
    valid_names = {field.name for field in fields(cls)}
    return {key: value for key, value in config.items() if key in valid_names}


def _override_experiment_run_paths(exp: Any, run_dir: str) -> None:
    resolved_run_dir = os.path.abspath(run_dir)
    exp.run_save_dir = resolved_run_dir
    exp.run_checkpoint_filepath = os.path.join(
        resolved_run_dir,
        os.path.basename(getattr(exp, "run_checkpoint_filepath", "run_checkpoint.pth")),
    )
    if hasattr(exp, "best_checkpoint_filepath"):
        exp.best_checkpoint_filepath = os.path.join(
            resolved_run_dir,
            os.path.basename(getattr(exp, "best_checkpoint_filepath", "best_model.pth")),
        )
    if hasattr(exp, "best_cond_checkpoint_filepath"):
        exp.best_cond_checkpoint_filepath = os.path.join(
            resolved_run_dir,
            os.path.basename(getattr(exp, "best_cond_checkpoint_filepath", "cond_pred_model.pth")),
        )
    if hasattr(exp, "best_cond_g_checkpoint_filepath"):
        exp.best_cond_g_checkpoint_filepath = os.path.join(
            resolved_run_dir,
            os.path.basename(
                getattr(exp, "best_cond_g_checkpoint_filepath", "cond_pred_model_g.pth")
            ),
        )
    if hasattr(exp, "_base_run_save_dir"):
        exp._base_run_save_dir = os.path.dirname(resolved_run_dir)


def _build_experiment_from_run_dir(
    run_dir: str,
    device: str | None,
    batch_size: int | None,
    num_worker: int | None,
    num_samples: int | None,
) -> tuple[Any, str]:
    config = _load_run_args(run_dir)
    model_type = str(config.get("model_type"))
    cls = _resolve_run_class(model_type)
    init_kwargs = _filter_init_kwargs_for_class(cls, config)
    if device is not None:
        init_kwargs["device"] = device
    if batch_size is not None and "batch_size" in init_kwargs:
        init_kwargs["batch_size"] = batch_size
    if num_worker is not None and "num_worker" in init_kwargs:
        init_kwargs["num_worker"] = num_worker
    if num_samples is not None and "num_samples" in init_kwargs:
        init_kwargs["num_samples"] = num_samples
    exp = cls(**init_kwargs)
    return exp, model_type


def _set_experiment_eval_mode(exp: Any) -> None:
    if hasattr(exp, "model"):
        exp.model.eval()
    if hasattr(exp, "cond_pred_model"):
        exp.cond_pred_model.eval()
    if hasattr(exp, "cond_pred_model_g"):
        exp.cond_pred_model_g.eval()


def _setup_source_target_experiment(
    run_dir: str,
    seed: int,
    device: str | None,
    batch_size: int | None,
    num_worker: int | None,
    num_samples: int | None,
    load_model: bool,
) -> tuple[Any, str]:
    exp, model_type = _build_experiment_from_run_dir(
        run_dir=run_dir,
        device=device,
        batch_size=batch_size,
        num_worker=num_worker,
        num_samples=num_samples,
    )
    exp._setup_run(seed)
    _override_experiment_run_paths(exp, run_dir)
    try:
        exp._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
    except TypeError:
        exp._init_data_loader()
        dataloader = getattr(exp, "dataloader", None)
        if dataloader is not None and hasattr(dataloader, "fast_test") and hasattr(dataloader, "fast_val"):
            dataloader.fast_test = False
            dataloader.fast_val = False
            if hasattr(dataloader, "shuffle_train"):
                dataloader.shuffle_train = False
            dataloader._load_dataset()
            dataloader._load_dataloader()
            exp.train_loader = dataloader.train_loader
            exp.val_loader = dataloader.val_loader
            exp.test_loader = dataloader.test_loader
            exp.train_steps = len(exp.train_loader.dataset)
            exp.val_steps = len(exp.val_loader.dataset)
            exp.test_steps = len(exp.test_loader.dataset)
    if load_model:
        exp._load_best_model()
        _set_experiment_eval_mode(exp)
    return exp, model_type


def _get_or_create_source_target_experiment(
    run_dir: str,
    seed: int,
    device: str | None,
    batch_size: int | None,
    num_worker: int | None,
    num_samples: int | None,
    load_model: bool,
    cache: dict[tuple[str, str | None, int | None, int | None, int | None], SourceTargetExperimentHandle]
    | None,
) -> tuple[Any, str]:
    resolved_run_dir = os.path.abspath(run_dir)
    cache_key = (resolved_run_dir, device, batch_size, num_worker, num_samples)

    if cache is None:
        return _setup_source_target_experiment(
            run_dir=resolved_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=load_model,
        )

    handle = cache.get(cache_key)
    if handle is None:
        exp, model_type = _setup_source_target_experiment(
            run_dir=resolved_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=load_model,
        )
        cache[cache_key] = SourceTargetExperimentHandle(
            exp=exp,
            model_type=model_type,
            model_loaded=load_model,
        )
        return exp, model_type

    if load_model and not handle.model_loaded:
        handle.exp._load_best_model()
        _set_experiment_eval_mode(handle.exp)
        handle.model_loaded = True

    return handle.exp, handle.model_type


def _analysis_feature_slice(exp: Any) -> slice:
    features = getattr(exp, "features", None)
    if features is None and hasattr(exp, "args"):
        features = getattr(exp.args, "features", None)
    if features == "MS":
        return slice(-1, None)
    return slice(None)


def _get_experiment_scaler_mean_std(
    exp: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if hasattr(exp, "_get_scaler_mean_std"):
        mean, std = exp._get_scaler_mean_std(dtype=dtype, device=device)
        if mean is not None and std is not None:
            return mean, std

    scaler = getattr(exp, "scaler", None)
    if scaler is None:
        return None, None

    mean = None
    std = None
    if hasattr(scaler, "mean"):
        mean = getattr(scaler, "mean")
    elif hasattr(scaler, "mean_"):
        mean = getattr(scaler, "mean_")

    if hasattr(scaler, "std"):
        std = getattr(scaler, "std")
    elif hasattr(scaler, "std_"):
        std = getattr(scaler, "std_")
    elif hasattr(scaler, "scale_"):
        std = getattr(scaler, "scale_")

    if mean is None or std is None:
        return None, None

    mean_tensor = torch.as_tensor(mean, device=device, dtype=dtype).view(1, 1, -1)
    std_tensor = torch.as_tensor(std, device=device, dtype=dtype).view(1, 1, -1)
    return mean_tensor, std_tensor


def _transform_gaussian_to_origin_space(
    exp: Any,
    mean_scaled: torch.Tensor,
    std_scaled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaler_mean, scaler_std = _get_experiment_scaler_mean_std(
        exp=exp,
        dtype=mean_scaled.dtype,
        device=mean_scaled.device,
    )
    if scaler_mean is None or scaler_std is None:
        return mean_scaled, std_scaled
    feature_slice = _analysis_feature_slice(exp)
    scaler_mean = scaler_mean[:, :, feature_slice]
    scaler_std = scaler_std[:, :, feature_slice]
    mean_origin = mean_scaled * scaler_std + scaler_mean
    std_origin = std_scaled * scaler_std
    return mean_origin, std_origin


def _transform_mean_to_origin_space(
    exp: Any,
    mean_scaled: torch.Tensor,
) -> torch.Tensor:
    scaler_mean, scaler_std = _get_experiment_scaler_mean_std(
        exp=exp,
        dtype=mean_scaled.dtype,
        device=mean_scaled.device,
    )
    if scaler_mean is None or scaler_std is None:
        return mean_scaled
    feature_slice = _analysis_feature_slice(exp)
    scaler_mean = scaler_mean[:, :, feature_slice]
    scaler_std = scaler_std[:, :, feature_slice]
    return mean_scaled * scaler_std + scaler_mean


def _predict_mu_from_f_model(
    exp: Any,
    batch_x: torch.Tensor,
    batch_x_date_enc: torch.Tensor,
    batch_y_date_enc: torch.Tensor,
) -> torch.Tensor:
    label_len = getattr(exp, "label_len", exp.windows // 2)
    batch_y_mark_input = torch.concat(
        [batch_x_date_enc[:, -label_len:, :], batch_y_date_enc],
        dim=1,
    )
    dec_inp_pred = torch.zeros(
        [batch_x.size(0), exp.pred_len, exp.dataset.num_features],
        device=batch_x.device,
    )
    dec_inp_label = batch_x[:, -label_len:, :]
    dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
    source_mean, _ = exp.model(
        batch_x,
        batch_x_date_enc,
        dec_inp,
        batch_y_mark_input,
    )
    feature_slice = _analysis_feature_slice(exp)
    return source_mean[:, :, feature_slice]


def _prepare_legacy_source_target_batch(
    exp: Any,
    display_name: str,
    space: str,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    origin_y: torch.Tensor,
    batch_x_date_enc: torch.Tensor,
    batch_y_date_enc: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_slice = _analysis_feature_slice(exp)
    if display_name == "TimeGrad":
        if space == "pdn":
            raise ValueError("TimeGrad does not support `pdn` comparison space.")
        target = origin_y[:, :, feature_slice] if space == "origin" else batch_y[:, :, feature_slice]
        source_mean = torch.zeros_like(target)
        source_std = torch.ones_like(target)
        if space == "origin":
            source_mean, source_std = _transform_gaussian_to_origin_space(
                exp=exp,
                mean_scaled=source_mean,
                std_scaled=source_std,
            )
        return target, source_mean, source_std

    if display_name == "TMDM":
        if not hasattr(exp, "cond_pred_model"):
            raise AttributeError("TMDM experiment is missing `cond_pred_model`.")
        label_len = getattr(exp, "label_len", exp.windows // 2)
        batch_y_mark_input = torch.concat(
            [batch_x_date_enc[:, -label_len:, :], batch_y_date_enc],
            dim=1,
        )
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), exp.pred_len, exp.dataset.num_features],
            device=batch_x.device,
        )
        dec_inp_label = batch_x[:, -label_len:, :]
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        _, y_0_hat_full, _, _ = exp.cond_pred_model(
            batch_x,
            batch_x_date_enc,
            dec_inp,
            batch_y_mark_input,
        )
        source_mean = y_0_hat_full[:, -exp.pred_len:, feature_slice]
        source_std = torch.ones_like(source_mean)
        target = batch_y[:, :, feature_slice]
        if space == "origin":
            target = origin_y[:, :, feature_slice]
            source_mean, source_std = _transform_gaussian_to_origin_space(
                exp=exp,
                mean_scaled=source_mean,
                std_scaled=source_std,
            )
        elif space != "scaled":
            raise ValueError(f"Unsupported comparison space for TMDM: {space}")
        return target, source_mean, source_std

    if display_name == "NsDiff":
        if not hasattr(exp, "cond_pred_model") or not hasattr(exp, "cond_pred_model_g"):
            raise AttributeError("NsDiff experiment is missing conditional prior modules.")
        label_len = getattr(exp, "label_len", exp.windows // 2)
        batch_y_mark_input = torch.concat(
            [batch_x_date_enc[:, -label_len:, :], batch_y_date_enc],
            dim=1,
        )
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), exp.pred_len, exp.dataset.num_features],
            device=batch_x.device,
        )
        dec_inp_label = batch_x[:, -label_len:, :]
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        source_mean, _ = exp.cond_pred_model(
            batch_x,
            batch_x_date_enc,
            dec_inp,
            batch_y_mark_input,
        )
        gx = torch.clamp_min(exp.cond_pred_model_g(batch_x), eps)
        source_mean = source_mean[:, :, feature_slice]
        source_std = torch.sqrt(gx[:, :, feature_slice])
        target = batch_y[:, :, feature_slice]
        if space == "origin":
            target = origin_y[:, :, feature_slice]
            source_mean, source_std = _transform_gaussian_to_origin_space(
                exp=exp,
                mean_scaled=source_mean,
                std_scaled=source_std,
            )
        elif space != "scaled":
            raise ValueError(f"Unsupported comparison space for NsDiff: {space}")
        return target, source_mean, source_std

    if display_name == "PDN-Flow":
        _, y_hat, sigma = exp.model.get_encoder_features(batch_x, batch_x_date_enc)
        y_hat = y_hat[:, :, feature_slice]
        sigma = torch.clamp_min(sigma[:, :, feature_slice], eps)
        if space == "pdn":
            target = (batch_y[:, :, feature_slice] - y_hat) / sigma
            source_mean = torch.zeros_like(target)
            source_std = torch.ones_like(target)
        elif space == "scaled":
            target = batch_y[:, :, feature_slice]
            source_mean = y_hat
            source_std = sigma
        elif space == "origin":
            target = origin_y[:, :, feature_slice]
            source_mean, source_std = _transform_gaussian_to_origin_space(
                exp=exp,
                mean_scaled=y_hat,
                std_scaled=sigma,
            )
        else:
            raise ValueError(
                f"Unsupported comparison space for PDN-Flow: {space}. "
                "Expected one of: pdn, scaled, origin."
            )
        return target, source_mean, source_std

    raise ValueError(f"Unsupported method `{display_name}` for source-target analysis.")


def _reservoir_update(
    payload: dict[str, np.ndarray] | None,
    keys: np.ndarray | None,
    batch_payload: dict[str, np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    first_key = next(iter(batch_payload))
    num_values = int(batch_payload[first_key].shape[0])
    if num_values == 0:
        if payload is None or keys is None:
            empty_payload = {
                name: np.empty((0,) + tuple(values.shape[1:]), dtype=np.float32)
                for name, values in batch_payload.items()
            }
            return empty_payload, np.empty((0,), dtype=np.float64)
        return payload, keys

    flat_batch_payload = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in batch_payload.items()
    }
    batch_keys = rng.random(num_values, dtype=np.float64)

    if payload is None or keys is None:
        merged_payload = flat_batch_payload
        merged_keys = batch_keys
    else:
        merged_payload = {
            name: np.concatenate([payload[name], flat_batch_payload[name]], axis=0).astype(
                np.float32, copy=False
            )
            for name in flat_batch_payload
        }
        merged_keys = np.concatenate([keys, batch_keys], axis=0)

    if merged_keys.size > sample_size:
        keep_idx = np.argpartition(merged_keys, -sample_size)[-sample_size:]
        keep_idx = keep_idx[np.argsort(merged_keys[keep_idx])]
    else:
        keep_idx = np.arange(merged_keys.size)
    kept_payload = {name: values[keep_idx] for name, values in merged_payload.items()}
    kept_keys = merged_keys[keep_idx]
    return kept_payload, kept_keys


def _collect_source_target_reservoir(
    method_spec: SourceTargetMethodSpec,
    seed: int,
    max_points: int,
    eps: float,
    device: str | None,
    batch_size: int | None,
    num_worker: int | None,
    num_samples: int | None,
    experiment_cache: dict[
        tuple[str, str | None, int | None, int | None, int | None], SourceTargetExperimentHandle
    ]
    | None = None,
) -> tuple[SourceTargetReservoir, str]:
    legacy_run_dir = method_spec.run_dir
    exp = None
    aux_exp_sigma = None

    if method_spec.display_name == "PDN-Flow":
        if method_spec.run_dir is None:
            raise ValueError("PDN-Flow requires `run_dir`.")
        exp, model_type = _get_or_create_source_target_experiment(
            run_dir=method_spec.run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=True,
            cache=experiment_cache,
        )
    elif method_spec.display_name == "TimeGrad" and method_spec.config_run_dir is not None:
        exp, _ = _get_or_create_source_target_experiment(
            run_dir=method_spec.config_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=False,
            cache=experiment_cache,
        )
        model_type = "TimeGrad"
    elif method_spec.display_name == "TMDM" and method_spec.mu_run_dir is not None:
        exp, _ = _get_or_create_source_target_experiment(
            run_dir=method_spec.mu_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=True,
            cache=experiment_cache,
        )
        model_type = "TMDM"
    elif (
        method_spec.display_name == "NsDiff"
        and method_spec.mu_run_dir is not None
        and method_spec.sigma_run_dir is not None
    ):
        exp, _ = _get_or_create_source_target_experiment(
            run_dir=method_spec.mu_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=True,
            cache=experiment_cache,
        )
        aux_exp_sigma, _ = _get_or_create_source_target_experiment(
            run_dir=method_spec.sigma_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=True,
            cache=experiment_cache,
        )
        model_type = "NsDiff"
    elif legacy_run_dir is not None:
        load_model = method_spec.display_name != "TimeGrad"
        exp, model_type = _get_or_create_source_target_experiment(
            run_dir=legacy_run_dir,
            seed=seed,
            device=device,
            batch_size=batch_size,
            num_worker=num_worker,
            num_samples=num_samples,
            load_model=load_model,
            cache=experiment_cache,
        )
    else:
        raise ValueError(
            f"Unsupported source-target method specification for `{method_spec.display_name}`."
        )

    if exp is None:
        raise RuntimeError("Source-target experiment setup failed.")

    rng = np.random.default_rng(seed)
    reservoir_payload: dict[str, np.ndarray] | None = None
    reservoir_keys: np.ndarray | None = None
    total_windows = 0

    with torch.no_grad():
        for (
            batch_x,
            batch_y,
            origin_x,
            origin_y,
            batch_x_date_enc,
            batch_y_date_enc,
        ) in exp.test_loader:
            del origin_x
            batch_x = batch_x.to(exp.device).float()
            batch_y = batch_y.to(exp.device).float()
            origin_y = origin_y.to(exp.device).float()
            batch_x_date_enc = batch_x_date_enc.to(exp.device).float()
            batch_y_date_enc = batch_y_date_enc.to(exp.device).float()

            if method_spec.display_name == "TimeGrad" and method_spec.config_run_dir is not None:
                feature_slice = _analysis_feature_slice(exp)
                if method_spec.space == "origin":
                    target = origin_y[:, :, feature_slice]
                elif method_spec.space == "scaled":
                    target = batch_y[:, :, feature_slice]
                else:
                    raise ValueError(
                        f"Unsupported comparison space for TimeGrad: {method_spec.space}"
                    )
                source_mean = torch.zeros_like(target)
                source_std = torch.ones_like(target)
            elif method_spec.display_name == "TMDM" and method_spec.mu_run_dir is not None:
                source_mean = _predict_mu_from_f_model(
                    exp=exp,
                    batch_x=batch_x,
                    batch_x_date_enc=batch_x_date_enc,
                    batch_y_date_enc=batch_y_date_enc,
                )
                source_std = torch.ones_like(source_mean)
                if method_spec.space == "origin":
                    feature_slice = _analysis_feature_slice(exp)
                    target = origin_y[:, :, feature_slice]
                    source_mean = _transform_mean_to_origin_space(exp=exp, mean_scaled=source_mean)
                elif method_spec.space == "scaled":
                    feature_slice = _analysis_feature_slice(exp)
                    target = batch_y[:, :, feature_slice]
                else:
                    raise ValueError(
                        f"Unsupported comparison space for TMDM: {method_spec.space}"
                    )
            elif (
                method_spec.display_name == "NsDiff"
                and method_spec.mu_run_dir is not None
                and method_spec.sigma_run_dir is not None
            ):
                if aux_exp_sigma is None:
                    raise RuntimeError("NsDiff requires an initialized sigma experiment.")
                feature_slice = _analysis_feature_slice(exp)
                source_mean = _predict_mu_from_f_model(
                    exp=exp,
                    batch_x=batch_x,
                    batch_x_date_enc=batch_x_date_enc,
                    batch_y_date_enc=batch_y_date_enc,
                )
                source_std = torch.clamp_min(aux_exp_sigma.model(batch_x)[:, :, feature_slice], eps)
                if method_spec.space == "origin":
                    target = origin_y[:, :, feature_slice]
                    source_mean, source_std = _transform_gaussian_to_origin_space(
                        exp=exp,
                        mean_scaled=source_mean,
                        std_scaled=source_std,
                    )
                elif method_spec.space == "scaled":
                    target = batch_y[:, :, feature_slice]
                else:
                    raise ValueError(
                        f"Unsupported comparison space for NsDiff: {method_spec.space}"
                    )
            else:
                target, source_mean, source_std = _prepare_legacy_source_target_batch(
                    exp=exp,
                    display_name=method_spec.display_name,
                    space=method_spec.space,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    origin_y=origin_y,
                    batch_x_date_enc=batch_x_date_enc,
                    batch_y_date_enc=batch_y_date_enc,
                    eps=eps,
                )

            total_windows += int(target.shape[0])
            reservoir_payload, reservoir_keys = _reservoir_update(
                payload=reservoir_payload,
                keys=reservoir_keys,
                batch_payload={
                    "target": target.detach().cpu().numpy(),
                    "source_mean": source_mean.detach().cpu().numpy(),
                    "source_std": source_std.detach().cpu().numpy(),
                },
                sample_size=max_points,
                rng=rng,
            )

    if reservoir_payload is None:
        reservoir_payload = {
            "target": np.empty((0, 0, 0), dtype=np.float32),
            "source_mean": np.empty((0, 0, 0), dtype=np.float32),
            "source_std": np.empty((0, 0, 0), dtype=np.float32),
        }

    return (
        SourceTargetReservoir(
            target=reservoir_payload["target"],
            source_mean=reservoir_payload["source_mean"],
            source_std=reservoir_payload["source_std"],
            sampled_windows=int(reservoir_payload["target"].shape[0]),
            total_windows=total_windows,
        ),
        model_type,
    )


def _per_window_conditional_w1(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"source and target must share shape, got {source.shape} vs {target.shape}.")
    if source.ndim < 2:
        raise ValueError(f"Expected at least 2 dims for per-window conditional W1, got {source.shape}.")
    reduce_axes = tuple(range(1, source.ndim))
    return np.mean(np.abs(source - target), axis=reduce_axes, dtype=np.float64)


def _estimate_source_target_wasserstein(
    reservoir: SourceTargetReservoir,
    num_repeats: int,
    seed: int,
) -> dict[str, Any]:
    if reservoir.sampled_windows == 0:
        raise ValueError("No target/source windows were collected for Wasserstein estimation.")

    target_all = np.asarray(reservoir.target, dtype=np.float64)
    source_mean_all = np.asarray(reservoir.source_mean, dtype=np.float64)
    source_std_all = np.maximum(np.asarray(reservoir.source_std, dtype=np.float64), 0.0)
    num_windows = target_all.shape[0]
    rng = np.random.default_rng(seed)

    per_repeat_window_values = []
    for _ in range(max(1, num_repeats)):
        eps_noise = rng.standard_normal(size=target_all.shape).astype(np.float64, copy=False)
        source_noise = source_mean_all + source_std_all * eps_noise
        per_repeat_window_values.append(_per_window_conditional_w1(source_noise, target_all))

    per_repeat_window_values = np.asarray(per_repeat_window_values, dtype=np.float64)
    window_mean_values = np.mean(per_repeat_window_values, axis=0)
    repeat_mean_values = np.mean(per_repeat_window_values, axis=1)
    window_std = float(np.std(window_mean_values, dtype=np.float64))
    window_sem = float(window_std / np.sqrt(max(num_windows, 1)))
    return {
        "mean": float(np.mean(window_mean_values, dtype=np.float64)),
        "std": window_std,
        "sem": window_sem,
        "repeats": [float(v) for v in repeat_mean_values.tolist()],
        "window_values": [float(v) for v in window_mean_values.tolist()],
        "repeat_estimator": "source_resampling_per_window",
        "aggregation": "mean_over_windows_of_conditional_w1",
    }


def _filter_pdn_outlier_windows(
    reservoir: SourceTargetReservoir,
    filter_method: str,
    filter_value: float,
) -> tuple[SourceTargetReservoir, dict[str, Any] | None]:
    if filter_method == "none":
        return reservoir, None

    scores = _mean_abs_per_window(reservoir.target)
    if scores.size == 0:
        return reservoir, None

    if filter_method == "quantile":
        if not (0.0 < filter_value <= 1.0):
            raise ValueError("PDN quantile filter expects `filter_value` in (0, 1].")
        threshold = float(np.quantile(scores, filter_value))
        keep_mask = scores <= threshold
        extra_info: dict[str, Any] = {"quantile": float(filter_value)}
    elif filter_method == "robust_zscore":
        if filter_value <= 0.0:
            raise ValueError("PDN robust z-score filter expects a positive `filter_value`.")
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        denom = 1.4826 * mad
        if denom <= 0.0:
            keep_mask = np.ones_like(scores, dtype=bool)
        else:
            robust_z = np.abs(scores - median) / denom
            keep_mask = robust_z <= filter_value
        threshold = float(filter_value)
        extra_info = {"median": median, "mad": mad}
    else:
        raise ValueError(
            f"Unsupported PDN filter method `{filter_method}`. "
            "Expected one of: none, quantile, robust_zscore."
        )

    kept = int(np.sum(keep_mask))
    if kept <= 0:
        raise ValueError("PDN outlier filter removed all sampled windows.")

    filtered_reservoir = SourceTargetReservoir(
        target=reservoir.target[keep_mask],
        source_mean=reservoir.source_mean[keep_mask],
        source_std=reservoir.source_std[keep_mask],
        sampled_windows=kept,
        total_windows=reservoir.total_windows,
    )
    filter_info = {
        "method": filter_method,
        "value": float(filter_value),
        "threshold": threshold,
        "score": "mean_abs_pdn_target",
        "sampled_windows_before_filter": int(reservoir.sampled_windows),
        "sampled_windows_after_filter": kept,
        "filtered_sampled_windows": int(reservoir.sampled_windows - kept),
    }
    filter_info.update(extra_info)
    return filtered_reservoir, filter_info


def _plot_source_target_wasserstein_bars(
    dataset_specs: Sequence[SourceTargetDatasetSpec],
    dataset_results: Sequence[dict[str, Any]],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=600, constrained_layout=True)
    x = np.arange(len(dataset_specs), dtype=np.float64)
    width = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    errorbar_style = {
        "fmt": "none",
        "elinewidth": 1.5,
        "ecolor": "#111827",
        "capsize": 4.5,
        "capthick": 1.5,
        "zorder": 5,
    }

    for method_idx, method_name in enumerate(SOURCE_TARGET_METHOD_ORDER):
        heights = np.asarray(
            [dataset_result["methods"][method_name]["w1_mean"] for dataset_result in dataset_results],
            dtype=np.float64,
        )
        errors = np.asarray(
            [dataset_result["methods"][method_name]["w1_std"] for dataset_result in dataset_results],
            dtype=np.float64,
        )
        x_pos = x + offsets[method_idx]
        ax.bar(
            x_pos,
            heights,
            width=width,
            color=SOURCE_TARGET_METHOD_COLORS[method_name],
            edgecolor="white",
            linewidth=0.8,
            hatch=SOURCE_TARGET_METHOD_HATCHES[method_name],
            label=method_name,
            zorder=3,
        )
        ax.errorbar(x_pos, heights, yerr=errors, **errorbar_style)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean Conditional W1 to Target")
    ax.set_xticks(x)
    ax.set_xticklabels([spec.display_name for spec in dataset_specs])
    ax.xaxis.label.set_size(PCA_TEXT_SIZE)
    ax.yaxis.label.set_size(PCA_TEXT_SIZE)
    ax.tick_params(axis="both", labelsize=PCA_TEXT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.24, zorder=0)
    ax.grid(False, axis="x")
    ymax = max(
        max(
            dataset_result["methods"][method_name]["w1_mean"]
            + dataset_result["methods"][method_name]["w1_std"]
            for method_name in SOURCE_TARGET_METHOD_ORDER
        )
        for dataset_result in dataset_results
    )
    ax.set_ylim(0.0, max(1e-8, float(ymax)) * 1.28)
    ax.margins(x=0.03)

    legend_handles = [
        Patch(
            facecolor=SOURCE_TARGET_METHOD_COLORS[method_name],
            edgecolor="white",
            linewidth=0.8,
            hatch=SOURCE_TARGET_METHOD_HATCHES[method_name],
            label=method_name,
        )
        for method_name in SOURCE_TARGET_METHOD_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=10.5 * FONT_SCALE,
        ncol=2,
    )

    if sns is not None:
        sns.despine(ax=ax, left=False, bottom=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_source_target_metadata(
    path: str,
    dataset_results: Sequence[dict[str, Any]],
    manifest_path: str,
    max_points: int,
    num_repeats: int,
    seed: int,
    pdn_filter_method: str,
    pdn_filter_value: float,
) -> None:
    payload = {
        "analysis": "source_target_conditional_wasserstein_bars",
        "manifest_path": os.path.abspath(manifest_path),
        "seed": int(seed),
        "max_points": int(max_points),
        "num_repeats": int(num_repeats),
        "pdn_filter_method": pdn_filter_method,
        "pdn_filter_value": float(pdn_filter_value),
        "repeat_estimator": "source_resampling_per_window",
        "aggregation": "mean_over_windows_of_conditional_w1",
        "datasets": dataset_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _summarize_result_for_stdout(result: dict[str, Any]) -> dict[str, Any]:
    datasets_summary = []
    for dataset in result.get("datasets", []):
        methods_summary: dict[str, Any] = {}
        for method_name, method_payload in dataset.get("methods", {}).items():
            summarized_payload = dict(method_payload)
            window_values = summarized_payload.pop("window_w1_values", None)
            if window_values is not None:
                summarized_payload["window_w1_values_count"] = len(window_values)
                if len(window_values) > 0:
                    window_values_np = np.asarray(window_values, dtype=np.float64)
                    summarized_payload["window_w1_values_summary"] = {
                        "min": float(np.min(window_values_np)),
                        "max": float(np.max(window_values_np)),
                        "mean": float(np.mean(window_values_np, dtype=np.float64)),
                    }
            methods_summary[method_name] = summarized_payload

        datasets_summary.append(
            {
                "dataset_name": dataset.get("dataset_name"),
                "display_name": dataset.get("display_name"),
                "methods": methods_summary,
            }
        )

    return {
        "analysis": result.get("analysis"),
        "manifest_path": result.get("manifest_path"),
        "output_path": result.get("output_path"),
        "metadata_path": result.get("metadata_path"),
        "seed": result.get("seed"),
        "max_points": result.get("max_points"),
        "num_repeats": result.get("num_repeats"),
        "repeat_estimator": result.get("repeat_estimator"),
        "aggregation": result.get("aggregation"),
        "datasets": datasets_summary,
    }


def plot_source_target_wasserstein_bars(
    manifest_path: str,
    output_path: str,
    metadata_path: str | None = None,
    seed: int = 42,
    max_points: int = 250_000,
    num_repeats: int = 8,
    eps: float = 1e-6,
    device: str | None = None,
    batch_size: int | None = None,
    num_worker: int | None = None,
    num_samples: int | None = None,
    pdn_filter_method: str = "none",
    pdn_filter_value: float = 0.99,
) -> dict[str, Any]:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    if num_repeats <= 0:
        raise ValueError("num_repeats must be positive.")

    _configure_style()
    dataset_specs = _load_source_target_manifest(manifest_path)
    dataset_results: list[dict[str, Any]] = []
    experiment_cache: dict[
        tuple[str, str | None, int | None, int | None, int | None], SourceTargetExperimentHandle
    ] = {}

    for dataset_idx, dataset_spec in enumerate(dataset_specs):
        method_results: dict[str, Any] = {}
        for method_idx, method_spec in enumerate(dataset_spec.methods):
            reservoir, model_type = _collect_source_target_reservoir(
                method_spec=method_spec,
                seed=seed + 1009 * dataset_idx + 97 * method_idx,
                max_points=max_points,
                eps=eps,
                device=device,
                batch_size=batch_size,
                num_worker=num_worker,
                num_samples=num_samples,
                experiment_cache=experiment_cache,
            )
            pdn_filter_info = None
            if method_spec.display_name == "PDN-Flow" and method_spec.space == "pdn":
                reservoir, pdn_filter_info = _filter_pdn_outlier_windows(
                    reservoir=reservoir,
                    filter_method=pdn_filter_method,
                    filter_value=pdn_filter_value,
                )
            metric = _estimate_source_target_wasserstein(
                reservoir=reservoir,
                num_repeats=num_repeats,
                seed=seed + 7919 * dataset_idx + 389 * method_idx,
            )
            method_results[method_spec.display_name] = {
                "model_type": model_type,
                "run_dir": None if method_spec.run_dir is None else os.path.abspath(method_spec.run_dir),
                "config_run_dir": None if method_spec.config_run_dir is None else os.path.abspath(method_spec.config_run_dir),
                "mu_run_dir": None if method_spec.mu_run_dir is None else os.path.abspath(method_spec.mu_run_dir),
                "sigma_run_dir": None if method_spec.sigma_run_dir is None else os.path.abspath(method_spec.sigma_run_dir),
                "space": method_spec.space,
                "w1_mean": metric["mean"],
                "w1_std": metric["std"],
                "w1_sem": metric["sem"],
                "w1_repeats": metric["repeats"],
                "w1_repeat_estimator": metric["repeat_estimator"],
                "w1_aggregation": metric["aggregation"],
                "window_w1_values": metric["window_values"],
                "sampled_windows": int(reservoir.sampled_windows),
                "total_windows": int(reservoir.total_windows),
                "pdn_filter": pdn_filter_info,
            }

        dataset_results.append(
            {
                "dataset_name": dataset_spec.name,
                "display_name": dataset_spec.display_name,
                "methods": method_results,
            }
        )

    _plot_source_target_wasserstein_bars(
        dataset_specs=dataset_specs,
        dataset_results=dataset_results,
        output_path=output_path,
    )
    if metadata_path is not None:
        _save_source_target_metadata(
            path=metadata_path,
            dataset_results=dataset_results,
            manifest_path=manifest_path,
            max_points=max_points,
            num_repeats=num_repeats,
            seed=seed,
            pdn_filter_method=pdn_filter_method,
            pdn_filter_value=pdn_filter_value,
        )
    return {
        "analysis": "source_target_conditional_wasserstein_bars",
        "manifest_path": os.path.abspath(manifest_path),
        "output_path": os.path.abspath(output_path),
        "metadata_path": None if metadata_path is None else os.path.abspath(metadata_path),
        "seed": int(seed),
        "max_points": int(max_points),
        "num_repeats": int(num_repeats),
        "pdn_filter_method": pdn_filter_method,
        "pdn_filter_value": float(pdn_filter_value),
        "repeat_estimator": "source_resampling_per_window",
        "aggregation": "mean_over_windows_of_conditional_w1",
        "datasets": dataset_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source-target Wasserstein-1 bar figures.")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points", type=int, default=250_000)
    parser.add_argument("--num_repeats", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--analysis_batch_size", type=int, default=None)
    parser.add_argument("--analysis_num_worker", type=int, default=None)
    parser.add_argument("--analysis_num_samples", type=int, default=None)
    parser.add_argument(
        "--pdn_filter_method",
        type=str,
        default="none",
        choices=("none", "quantile", "robust_zscore"),
    )
    parser.add_argument("--pdn_filter_value", type=float, default=0.99)
    args = parser.parse_args()

    result = plot_source_target_wasserstein_bars(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        seed=args.seed,
        max_points=args.max_points,
        num_repeats=args.num_repeats,
        eps=args.eps,
        device=args.device,
        batch_size=args.analysis_batch_size,
        num_worker=args.analysis_num_worker,
        num_samples=args.analysis_num_samples,
        pdn_filter_method=args.pdn_filter_method,
        pdn_filter_value=args.pdn_filter_value,
    )
    print(json.dumps(_summarize_result_for_stdout(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
