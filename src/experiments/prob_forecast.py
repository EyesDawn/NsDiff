# import codecs
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim import *
from torch_timeseries.dataset import *
from src.datasets import *
from torch_timeseries.scaler import *
from src.metrics import CRPS, CRPSSum, QICE, PICP
from src.metrics import ProbMAE, ProbMSE, ProbRMSE

from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
from torch_timeseries.experiments import ForecastExp

try:
    import wandb
except:
    print("Warning: wandb is not installed, some funtionality may not work.")



def update_metrics(preds, truths, metrics):
    """Function to update metrics in a separate process."""
    metrics.update(preds, truths)


@dataclass
class ProbForecastExp(ForecastExp):
    loss_func_type : str = 'mse'
    epochs : int = 10
    
    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "crps": CRPS(),
                "crps_sum": CRPSSum(normalize=True),
                "qice": QICE(),
                "picp": PICP(),
                "mse": ProbMSE(),
                "mae":ProbMAE(),
                "rmse": ProbRMSE(),
            }
        )
        self.metrics.to("cpu")
        # 之前尝试使用多进程池异步更新 metrics，但由于 torchmetrics 的度量对象
        # 在子进程中的状态不会回传到主进程，导致主进程中的 metrics 没有被真正 update，
        # 从而在 compute() 时出现 "compute called before update" 的警告。
        # 为保证正确性，这里改为在主进程中同步更新 metrics，不再使用多进程池。

    def _init_dataset(self):
        self.dataset: TimeSeriesDataset = parse_type(self.dataset_type, globals())(
            root=self.data_path
        )

    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                self.model_optim.zero_grad()
                pred, true = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                loss = self.loss_func(pred, true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                progress_bar.update(batch_x.size(0))
                
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()

            return train_loss
    def _process_train_batch(
        self,
        batch_x,
        batch_y,
        batch_origin_x,
        batch_origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        # for single step you should output (B, N)
        # for multiple steps you should output (B, O, N)
        raise NotImplementedError()

    def _process_val_batch(
        self,
        batch_x,
        batch_origin_x,
        batch_x_date_enc,
        batch_y_date_enc,
    ):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        # for single step you should output (B, N)
        # for multiple steps you should output (B, O, N)
        raise NotImplementedError()




    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                batch_size = batch_x.size(0)
                origin_x = origin_x.to(self.device)
                origin_y = origin_y.to(self.device)
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                start = time.time()
                print(batch_x.shape)
                print(batch_x_date_enc.shape)
                
                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                end = time.time()
                print(end-start)
                assert 0
                
                if self.invtrans_loss:
                    preds = self.scaler.inverse_transform(preds)
                    truths = origin_y

                # 在主进程中同步更新 metrics，避免多进程导致的状态不同步问题
                self.metrics.update(
                    preds.contiguous().cpu().detach(),
                    truths.contiguous().cpu().detach(),
                )

                progress_bar.update(batch_x.shape[0])

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result

    def _get_scaler_mean_std(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return dataset-level scaler statistics as broadcastable tensors."""
        if not hasattr(self, "scaler"):
            return None, None

        mean = None
        std = None
        if hasattr(self.scaler, "mean"):
            mean = getattr(self.scaler, "mean")
        elif hasattr(self.scaler, "mean_"):
            mean = getattr(self.scaler, "mean_")

        if hasattr(self.scaler, "std"):
            std = getattr(self.scaler, "std")
        elif hasattr(self.scaler, "std_"):
            std = getattr(self.scaler, "std_")
        elif hasattr(self.scaler, "scale_"):
            std = getattr(self.scaler, "scale_")

        if mean is None or std is None:
            return None, None

        mean_tensor = torch.as_tensor(mean, device=device, dtype=dtype).view(1, 1, -1)
        std_tensor = torch.as_tensor(std, device=device, dtype=dtype).view(1, 1, -1)
        return mean_tensor, std_tensor

    def _inverse_transform_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform any tensor whose feature dimension is the penultimate axis.

        Supported shapes include [B, P, D] and [B, P, D, S].
        """
        if not hasattr(self, "scaler"):
            return tensor
        if tensor.ndim < 3:
            return self.scaler.inverse_transform(tensor)

        mean, std = self._get_scaler_mean_std(dtype=tensor.dtype, device=tensor.device)
        if mean is None or std is None:
            return self.scaler.inverse_transform(tensor)

        view_shape = [1] * tensor.ndim
        feature_axis = tensor.ndim - 2
        view_shape[feature_axis] = mean.shape[-1]
        mean = mean.view(*view_shape)
        std = std.view(*view_shape)
        return tensor * std + mean

    @torch.no_grad()
    def export_forecast_samples_on_test(
        self,
        seed: int = 42,
        save_path: str = "./results/analysis/forecast_samples_on_test.npz",
        use_origin_scale: bool = True,
        eps: float = 1e-6,
        max_windows: Optional[int] = None,
        selected_window_indices: Optional[Sequence[int]] = None,
        run_dir_override: Optional[str] = None,
        return_result: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Export aligned forecast samples on the full test set.

        Saved tensors:
          - Y: realized future windows, shape [N, P, D]
          - samples: predictive samples, shape [N, P, D, S]
          - mu_X / sigma_X: history-window mean/std, shape [N, 1, D]

        The output is designed for downstream cross-method analysis scripts that
        need a shared window pool and comparable probabilistic forecasts.
        """
        if hasattr(self, "_get_setting"):
            setting = self._get_setting(seed)
        else:
            setting = "N/A"

        print("=" * 80)
        print("Analysis Export: Forecast Samples on Test Set")
        print(f"Dataset: {getattr(self, 'dataset_type', getattr(self, 'data', 'custom'))}")
        print(f"Setting: {setting}")
        print(f"Seed   : {seed}")
        print("=" * 80)

        self._setup_run(seed)
        if run_dir_override is not None:
            resolved_run_dir = os.path.abspath(run_dir_override)
            self.run_save_dir = resolved_run_dir
            self.run_checkpoint_filepath = os.path.join(
                resolved_run_dir,
                os.path.basename(getattr(self, "run_checkpoint_filepath", "run_checkpoint.pth")),
            )
            if hasattr(self, "best_checkpoint_filepath"):
                self.best_checkpoint_filepath = os.path.join(
                    resolved_run_dir,
                    os.path.basename(getattr(self, "best_checkpoint_filepath", "best_model.pth")),
                )
            if hasattr(self, "best_cond_checkpoint_filepath"):
                self.best_cond_checkpoint_filepath = os.path.join(
                    resolved_run_dir,
                    os.path.basename(getattr(self, "best_cond_checkpoint_filepath", "cond_pred_model.pth")),
                )
            if hasattr(self, "best_cond_g_checkpoint_filepath"):
                self.best_cond_g_checkpoint_filepath = os.path.join(
                    resolved_run_dir,
                    os.path.basename(getattr(self, "best_cond_g_checkpoint_filepath", "cond_pred_model_g.pth")),
                )
            if hasattr(self, "_base_run_save_dir"):
                self._base_run_save_dir = os.path.dirname(resolved_run_dir)
        try:
            self._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
        except TypeError:
            self._init_data_loader()
        self._load_best_model()
        self.model.eval()

        ys = []
        samples_all = []
        mu_xs = []
        sigma_xs = []
        window_indices_all = []

        collected = 0
        dataset_size = len(self.test_loader.dataset)
        selected_positions = None
        if selected_window_indices is not None:
            selected_positions = np.asarray(selected_window_indices, dtype=np.int64).reshape(-1)
            if selected_positions.size == 0:
                raise ValueError("selected_window_indices must not be empty when provided.")
            selected_positions = np.unique(selected_positions)
            if selected_positions[0] < 0 or selected_positions[-1] >= dataset_size:
                raise IndexError(
                    "selected_window_indices must lie within [0, %d), got [%d, %d]."
                    % (dataset_size, int(selected_positions[0]), int(selected_positions[-1]))
                )
        progress_total = (
            int(selected_positions.shape[0]) if selected_positions is not None else dataset_size
        )
        current_offset = 0

        pending_batches = None
        pending_window_indices = []
        pending_count = 0

        if selected_positions is not None:
            pending_batches = {
                "batch_x": [],
                "batch_y": [],
                "origin_x": [],
                "origin_y": [],
                "batch_x_date_enc": [],
                "batch_y_date_enc": [],
            }

        def _flush_pending() -> int:
            nonlocal pending_count
            if pending_batches is None or pending_count == 0:
                return 0

            batch_x = torch.cat(pending_batches["batch_x"], dim=0).to(self.device).float()
            batch_y = torch.cat(pending_batches["batch_y"], dim=0).to(self.device).float()
            origin_x = torch.cat(pending_batches["origin_x"], dim=0).to(self.device).float()
            origin_y = torch.cat(pending_batches["origin_y"], dim=0).to(self.device).float()
            batch_x_date_enc = (
                torch.cat(pending_batches["batch_x_date_enc"], dim=0).to(self.device).float()
            )
            batch_y_date_enc = (
                torch.cat(pending_batches["batch_y_date_enc"], dim=0).to(self.device).float()
            )
            exported_window_indices = np.concatenate(pending_window_indices, axis=0).astype(
                np.int64, copy=False
            )

            preds, _ = self._process_val_batch(
                batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
            )
            if preds.ndim != 4:
                raise ValueError(
                    "Expected probabilistic predictions with shape [B, P, D, S], "
                    f"got {tuple(preds.shape)}."
                )

            x_ref = origin_x if use_origin_scale else batch_x
            y_ref = origin_y if use_origin_scale else batch_y
            if use_origin_scale:
                preds = self._inverse_transform_last_dim(preds)

            mu_x = x_ref.mean(dim=1, keepdim=True)
            sigma_x = x_ref.std(dim=1, keepdim=True).clamp_min(eps)

            ys.append(y_ref.detach().cpu())
            samples_all.append(preds.detach().cpu())
            mu_xs.append(mu_x.detach().cpu())
            sigma_xs.append(sigma_x.detach().cpu())
            window_indices_all.append(
                torch.as_tensor(exported_window_indices, dtype=torch.long).cpu()
            )

            for key in pending_batches:
                pending_batches[key].clear()
            pending_window_indices.clear()
            flushed = int(exported_window_indices.shape[0])
            pending_count = 0
            return flushed

        with tqdm(total=progress_total) as progress_bar:
            for (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in self.test_loader:
                batch_start = current_offset
                batch_size = batch_x.shape[0]
                batch_end = batch_start + batch_size
                current_offset = batch_end

                if selected_positions is not None:
                    left = int(np.searchsorted(selected_positions, batch_start, side="left"))
                    right = int(np.searchsorted(selected_positions, batch_end, side="left"))
                    if right <= left:
                        continue
                    selected_in_batch = selected_positions[left:right]
                    local_indices_np = selected_in_batch - batch_start
                    local_indices = torch.as_tensor(
                        local_indices_np,
                        dtype=torch.long,
                        device=batch_x.device,
                    )
                    batch_x = batch_x.index_select(0, local_indices)
                    batch_y = batch_y.index_select(0, local_indices)
                    origin_x = origin_x.index_select(0, local_indices)
                    origin_y = origin_y.index_select(0, local_indices)
                    batch_x_date_enc = batch_x_date_enc.index_select(0, local_indices)
                    batch_y_date_enc = batch_y_date_enc.index_select(0, local_indices)
                    pending_batches["batch_x"].append(batch_x)
                    pending_batches["batch_y"].append(batch_y)
                    pending_batches["origin_x"].append(origin_x)
                    pending_batches["origin_y"].append(origin_y)
                    pending_batches["batch_x_date_enc"].append(batch_x_date_enc)
                    pending_batches["batch_y_date_enc"].append(batch_y_date_enc)
                    pending_window_indices.append(selected_in_batch.astype(np.int64, copy=False))
                    pending_count += int(selected_in_batch.shape[0])

                    should_flush = pending_count >= int(self.batch_size)
                    if max_windows is not None and collected + pending_count >= max_windows:
                        should_flush = True
                    if should_flush:
                        exported_batch_size = _flush_pending()
                        collected += exported_batch_size
                        progress_bar.update(exported_batch_size)
                        if max_windows is not None and collected >= max_windows:
                            break
                    continue

                exported_window_indices = np.arange(batch_start, batch_end, dtype=np.int64)

                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                origin_x = origin_x.to(self.device).float()
                origin_y = origin_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, _ = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                if preds.ndim != 4:
                    raise ValueError(
                        "Expected probabilistic predictions with shape [B, P, D, S], "
                        f"got {tuple(preds.shape)}."
                    )

                x_ref = origin_x if use_origin_scale else batch_x
                y_ref = origin_y if use_origin_scale else batch_y
                if use_origin_scale:
                    preds = self._inverse_transform_last_dim(preds)

                mu_x = x_ref.mean(dim=1, keepdim=True)
                sigma_x = x_ref.std(dim=1, keepdim=True).clamp_min(eps)

                ys.append(y_ref.detach().cpu())
                samples_all.append(preds.detach().cpu())
                mu_xs.append(mu_x.detach().cpu())
                sigma_xs.append(sigma_x.detach().cpu())
                window_indices_all.append(
                    torch.as_tensor(exported_window_indices, dtype=torch.long).cpu()
                )

                exported_batch_size = batch_x.shape[0]
                collected += exported_batch_size
                progress_bar.update(exported_batch_size)
                if max_windows is not None and collected >= max_windows:
                    break

            if selected_positions is not None and (max_windows is None or collected < max_windows):
                exported_batch_size = _flush_pending()
                collected += exported_batch_size
                progress_bar.update(exported_batch_size)

        def _cat_to_numpy(tensor_list):
            return torch.cat(tensor_list, dim=0).numpy() if tensor_list else None

        y_all = _cat_to_numpy(ys)
        samples_np = _cat_to_numpy(samples_all)
        mu_x_all = _cat_to_numpy(mu_xs)
        sigma_x_all = _cat_to_numpy(sigma_xs)
        window_index_all = _cat_to_numpy(window_indices_all)

        if max_windows is not None and y_all is not None:
            limit = min(int(max_windows), y_all.shape[0])
            y_all = y_all[:limit]
            samples_np = samples_np[:limit]
            mu_x_all = mu_x_all[:limit]
            sigma_x_all = sigma_x_all[:limit]
            window_index_all = window_index_all[:limit]

        result = {
            "Y": y_all,
            "samples": samples_np,
            "mu_X": mu_x_all,
            "sigma_X": sigma_x_all,
            "window_index": window_index_all,
        }

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(save_path, **{k: v for k, v in result.items() if v is not None})

        print(f"Forecast samples saved to: {save_path}")
        print(f"Y shape        : {None if y_all is None else y_all.shape}")
        print(f"samples shape  : {None if samples_np is None else samples_np.shape}")
        print(f"mu_X / sigma_X : {None if mu_x_all is None else mu_x_all.shape}")
        print(f"window_index   : {None if window_index_all is None else window_index_all.shape}")

        if return_result:
            return result
        return None
    
    
    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        
        self._init_dataset()
        
        self.scaler = parse_type(self.scaler_type, globals=globals())()
        if self.dataset_type[0:3] == "ETT":
            if self.dataset_type[0:4] == "ETTh":
                self.dataloader = ETTHLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=shuffle,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    fast_test=fast_test,
                    fast_val=fast_val,
                )
            elif  self.dataset_type[0:4] == "ETTm":
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=shuffle,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    fast_test=fast_test,
                    fast_val=fast_val,
                )
        else:
            self.dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=True,
                shuffle_train=shuffle,
                freq=self.dataset.freq,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                num_worker=self.num_worker,
                fast_test=fast_test,
                fast_val=fast_val,
            )

        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")
        

    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)

        # if self._use_wandb():
        #     import wandb
        #     result = {}
        #     for name, metric_value in test_result.items():
        #         wandb.run.summary["test_" + name] = metric_value
        #         result["test_" + name] = metric_value
        #     wandb.log(result, step=self.current_epoch)

        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Validating .... ")
        val_result = self._evaluate(self.val_loader)

        # # log to wandb
        # if self._use_wandb():
        #     import wandb
        #     result = {}
        #     for name, metric_value in val_result.items():
        #         wandb.run.summary["val_" + name] = metric_value
        #         result["val_" + name] = metric_value
        #     wandb.log(result, step=self.current_epoch)

        self._run_print(f"vali_results: {val_result}")
        return val_result

    # def _train(self):
    #     with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
    #         self.model.train()
    #         train_loss = []
    #         for i, (
    #             batch_x,
    #             batch_y,
    #             origin_x,
    #             origin_y,
    #             batch_x_date_enc,
    #             batch_y_date_enc,
    #         ) in enumerate(self.train_loader):
    #             start = time.time()
    #             origin_y = origin_y.to(self.device)
    #             self.model_optim.zero_grad()
                
    #             origin_x = origin_x.to(self.device)
    #             origin_y = origin_y.to(self.device)
    #             batch_x = batch_x.to(self.device).float()
    #             batch_y = batch_y.to(self.device).float()
    #             batch_x_date_enc = batch_x_date_enc.to(self.device).float()
    #             batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                
    #             pred, true = self._process_train_batch(
    #                 batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
    #             )
    #             if self.invtrans_loss:
    #                 pred = self.scaler.inverse_transform(pred)
    #                 true = origin_y
    #             loss = self.loss_func(pred, true)
    #             loss.backward()

    #             torch.nn.utils.clip_grad_norm_(
    #                 self.model.parameters(), self.max_grad_norm
    #             )
    #             progress_bar.update(batch_x.size(0))
    #             train_loss.append(loss.item())
    #             progress_bar.set_postfix(
    #                 loss=loss.item(),
    #                 lr=self.model_optim.param_groups[0]["lr"],
    #                 epoch=self.current_epoch,
    #                 refresh=True,
    #             )
    #             self.model_optim.step()

    #         return train_loss

    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with open(
            os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists

    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device)
        )

    def _run_print(self, *args, **kwargs):
        time = (
            "["
            + str(datetime.datetime.now() + datetime.timedelta(hours=8))[:19]
            + "] -"
        )
        print(*args, **kwargs)
        # 确保日志目录存在
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir, exist_ok=True)
        with open(os.path.join(self.run_save_dir, "output.log"), "a+") as f:
            print(time, *args, flush=True, file=f)

    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _use_wandb(self):
        return hasattr(self, "wandb")

    def run(self, seed=42) -> Dict[str, float]:
        
        if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            wandb.run.summary["parameters"] = model_parameters_num

        # for resumable reproducibility_
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            # test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result['crps'], model=self.model)

            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
                wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                # wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

            # self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  wandb.finish()
        return best_test_result
    
    

    def runs(self, seeds: List[int] = [1, 2, 3, 4, 5]):
        results = []
        for i, seed in enumerate(seeds):
            result = self.run(seed=seed)
            results.append(result)

        return results

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
