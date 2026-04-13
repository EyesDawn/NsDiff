"""
Stage 2: Uncertainty Estimator 预训练脚本
用于在冻结 iTransformer 的情况下，单独训练 uncertainty_estimator
"""
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from typing import Dict
import argparse
import time
import numpy as np
import setproctitle
from tqdm import tqdm

from src.experiments.iReflow import iReflowExp, iReflowEarlyStopping
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
from src.utils.revin import RevIN
from src.utils.uncertainty_eval import compute_sigma_metrics

try:
    import wandb
except:
    print("Warning: wandb is not installed, some functionality may not work.")


@dataclass
class UncertaintyEstimatorPretrainExp(iReflowExp):
    """
    Stage 2: Uncertainty Estimator 预训练实验类
    
    在这个阶段：
    1. 冻结 iTransformer（使用 Stage 1 预训练的权重）
    2. 只训练 uncertainty_estimator
    3. 使用 NLL Loss 来学习预测误差分布
    """
    
    # 覆盖默认配置
    is_training: int = 1  # 固定为1，表示只训练部分模型

    def __post_init__(self):
        super().__post_init__()
        self.gaussian_nll_loss = torch.nn.GaussianNLLLoss()

    def _get_scaler_std_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        """
        Return dataset-level scaler std as a tensor for inverse-transforming sigma.

        Supports both the local `torch_timeseries` StandardScaler (`std`) and
        sklearn-like naming (`std_` / `scale_`) for robustness.
        """
        if not hasattr(self, "scaler"):
            return None

        std = None
        if hasattr(self.scaler, "std"):
            std = getattr(self.scaler, "std")
        elif hasattr(self.scaler, "std_"):
            std = getattr(self.scaler, "std_")
        elif hasattr(self.scaler, "scale_"):
            std = getattr(self.scaler, "scale_")

        if std is None:
            return None
        return torch.as_tensor(std, device=device, dtype=dtype).view(1, 1, -1)
    
    def _init_optimizer(self):
        """
        初始化优化器：只优化 uncertainty_estimator 的参数
        """
        # 只优化 uncertainty_estimator 的参数
        trainable_params = list(self.model.uncertainty_estimator.parameters())
        self.model_optim = torch.optim.Adam(trainable_params, lr=self.lr)
        print(
            "Initialized optimizer for Stage 2: "
            "will freeze iTransformer after loading weights, "
            "only training Uncertainty Estimator"
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optim, mode="min", factor=0.5, patience=self.lr_patience
        )
    
    def _freeze_itransformer(self):
        """冻结 iTransformer 参数（在加载预训练权重后调用）"""
        for param in self.model.itransformer.parameters():
            param.requires_grad = False
        print("iTransformer parameters frozen for Stage 2 (Uncertainty Estimator pretraining)")
    
    def _freeze_velocity_net(self):
        """冻结 Velocity Network 参数（Stage 2 不训练 velocity_net）"""
        for param in self.model.velocity_net.parameters():
            param.requires_grad = False
        print("Velocity Network parameters frozen for Stage 2")
    
    def _process_train_batch(
        self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
    ):
        """
        处理训练批次：只计算 NLL Loss
        
        Args:
            batch_x: [B, L, D] 历史序列
            batch_y: [B, P, D] 未来序列
            batch_x_date_enc: [B, L, T] 历史时间标记
            batch_y_date_enc: [B, P, T] 未来时间标记
        Returns:
            pred: [B, P, D] 预测（点预测）
            true: [B, P, D] 真实值
            loss: scalar 损失（只包含 NLL Loss）
            loss_dict: dict 包含详细损失和指标
        """
        # 获取编码器特征和预测
        _, y_hat, sigma = self.model.get_encoder_features(
            batch_x, batch_x_date_enc
        )
        
        # 只计算 NLL Loss（Gaussian Negative Log-Likelihood）
        nll_loss = self.gaussian_nll_loss(y_hat, batch_y, sigma.pow(2))
        
        # 记录详细指标
        loss_dict = {
            'nll_loss': nll_loss.item(),
            'mean_sigma': sigma.mean().item(),
            'min_sigma': sigma.min().item(),
            'max_sigma': sigma.max().item(),
            'mae_point': F.l1_loss(y_hat, batch_y).item(),
            'mse_point': F.mse_loss(y_hat, batch_y).item(),
        }
        
        return y_hat, batch_y, nll_loss, loss_dict
    
    def _train(self):
        """训练一个epoch：只优化 uncertainty_estimator"""
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            # 确保 iTransformer 和 velocity_net 保持 eval 模式
            self.model.itransformer.eval()
            self.model.velocity_net.eval()
            
            train_losses = []
            train_metrics = {
                'nll_loss': [],
                'mean_sigma': [],
                'min_sigma': [],
                'max_sigma': [],
                'mae_point': [],
                'mse_point': [],
            }
            
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                # 转换到设备
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                
                # 梯度清零
                self.model_optim.zero_grad()
                
                # 前向传播（只计算 NLL Loss）
                pred, true, loss, loss_dict = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.uncertainty_estimator.parameters(), 
                    self.max_grad_norm
                )
                
                # 优化器步进
                self.model_optim.step()
                
                train_losses.append(loss.item())
                
                # 收集详细指标
                for key in train_metrics.keys():
                    if key in loss_dict:
                        train_metrics[key].append(loss_dict[key])
                
                # 更新进度条
                progress_bar.set_postfix(
                    nll_loss=loss.item(),
                    avg_loss=np.mean(train_losses),
                    mean_sigma=loss_dict['mean_sigma']
                )
                progress_bar.update(batch_x.shape[0])
        
        avg_train_loss = np.mean(train_losses)
        avg_metrics = {key: np.mean(values) for key, values in train_metrics.items() if values}
        return avg_train_loss, avg_metrics
    
    def _val(self):
        """
        验证：计算 NLL Loss 和概率预测指标
        """
        # 设置验证时使用的样本数
        self._num_samples_for_eval = min(self.num_samples, 30)
        
        # 计算验证 NLL Loss
        self.model.eval()
        val_losses = []
        val_metrics = {
            'nll_loss': [],
            'mean_sigma': [],
            'mae_point': [],
        }
        
        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in self.val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                
                # 获取预测和 sigma
                _, y_hat, sigma = self.model.get_encoder_features(
                    batch_x, batch_x_date_enc
                )
                
                # 计算 NLL Loss（与训练阶段保持一致）
                nll_loss = self.gaussian_nll_loss(y_hat, batch_y, sigma.pow(2))
                
                val_losses.append(nll_loss.item())
                val_metrics['nll_loss'].append(nll_loss.item())
                val_metrics['mean_sigma'].append(sigma.mean().item())
                val_metrics['mae_point'].append(F.l1_loss(y_hat, batch_y).item())
        
        # 使用当前类的 _evaluate（已重写，避免经过 Velocity Network）
        result = self._evaluate(self.val_loader)
        
        # 添加 NLL Loss 和 sigma 统计
        result['loss'] = np.mean(val_losses)
        result['nll_loss'] = np.mean(val_metrics['nll_loss'])
        result['mean_sigma'] = np.mean(val_metrics['mean_sigma'])
        result['mae_point'] = np.mean(val_metrics['mae_point'])
        
        # 清理标志
        delattr(self, '_num_samples_for_eval')
        return result
    
    def _evaluate(self, dataloader, plot=False):
        """
        Stage 2 专用评估逻辑（方案 A）：
        
        - 完全绕过 Velocity Network / Rectified Flow，只基于冻结的 iTransformer 点预测 y_hat
        - 使用 ProbForecastExp 中定义的概率指标（CRPS / ProbMAE / ProbMSE 等），其中分布通过
          一个退化分布近似：所有样本都等于 y_hat（不注入额外随机噪声）
        - 同时保留基于 (y_hat, sigma) 的高斯校准指标（sigma_*），与 Stage 3 的命名保持一致
        """
        self.model.eval()
        self.metrics.reset()

        # 与 iReflowExp._evaluate 保持一致的 sigma 评估配置
        interval_levels = getattr(self, "sigma_interval_levels", [0.9])
        pit_bins = int(getattr(self, "pit_bins", 20))
        default_sigma_metric_keys = [
            "gauss_nll",
            "gauss_crps",
            "cov_90",
            "width_90",
            "pit_ks",
            "sharpness_sigma_mean",
        ]
        sigma_metric_keys = getattr(self, "sigma_metric_keys", default_sigma_metric_keys)

        sigma_sums = {}
        sigma_counts = 0

        # 当前评估使用的样本数（至少为 1）
        num_samples = max(1, int(getattr(self, "_num_samples_for_eval", self.num_samples)))

        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            with torch.no_grad():
                for (
                    batch_x,
                    batch_y,
                    origin_x,
                    origin_y,
                    batch_x_date_enc,
                    batch_y_date_enc,
                ) in dataloader:
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()
                    origin_y = origin_y.to(self.device).float()
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()

                    # 仅通过冻结的 iTransformer + uncertainty_estimator 获取 y_hat 与 sigma
                    enc_features, y_hat, sigma = self.model.get_encoder_features(
                        batch_x, batch_x_date_enc
                    )

                    # 构造退化“分布”：所有样本都等于 y_hat，完全不经过 Velocity Network
                    # 形状: [B, P, D, S]
                    preds = y_hat.unsqueeze(-1).expand(-1, -1, -1, num_samples)
                    truths = batch_y
                    if getattr(self, "invtrans_loss", False):
                        preds = self.scaler.inverse_transform(preds)
                        truths = origin_y

                    self.metrics.update(
                        preds.contiguous().cpu().detach(),
                        truths.contiguous().cpu().detach(),
                    )

                    # sigma 指标：在 batch_y / y_hat / sigma 同一尺度上计算（不做 inverse_transform）
                    sigma_metrics = compute_sigma_metrics(
                        y=batch_y.detach().cpu(),
                        mu=y_hat.detach().cpu(),
                        sigma=sigma.detach().cpu(),
                        interval_levels=list(interval_levels),
                        pit_bins=pit_bins,
                    )
                    # 只保留关键指标，避免日志过多
                    for k, v in sigma_metrics.items():
                        if sigma_metric_keys is not None and k not in sigma_metric_keys:
                            continue
                        sigma_sums[k] = sigma_sums.get(k, 0.0) + float(v)
                    sigma_counts += 1

                    progress_bar.update(batch_x.shape[0])

        # 概率预测指标
        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}

        # 汇总 sigma 指标（沿 batch 取平均），并加上 "sigma_" 前缀
        if sigma_counts > 0:
            result.update(
                {f"sigma_{k}": float(v / sigma_counts) for k, v in sigma_sums.items()}
            )

        return result
    
    def _setup_estimator_run_paths(self, setting: str, seed: int):
        """
        为 Uncertainty Estimator 显式设置运行目录与各类路径。
        
        目录格式：
            ./results/runs/estimator/{dataset}/{setting}/seed_{seed}/
        其中包含：
            - args.json
            - best_model.pth
            - run_checkpoint.pth
            - output.log
        """
        # dataset 使用 dataset_type（与 --data 对应，例如 ETTm1）
        dataset = getattr(self, "dataset_type", getattr(self, "data", "custom"))
        base_dir = "./results/runs/estimator"
        run_dir = os.path.join(base_dir, dataset, setting, f"seed_{seed}")

        # 显式设置运行目录和关键文件路径
        self.run_save_dir = run_dir
        os.makedirs(self.run_save_dir, exist_ok=True)

        # 运行检查点与最佳模型路径（文件名固定）
        self.run_checkpoint_filepath = os.path.join(self.run_save_dir, "run_checkpoint.pth")
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "best_model.pth")

        # 若 early_stopping 已经初始化，则同步其保存路径
        if hasattr(self, "early_stopping") and hasattr(self.early_stopping, "path"):
            self.early_stopping.path = self.best_checkpoint_filepath

    def run(self, seed=42) -> Dict[str, float]:
        """
        运行 Stage 2 训练流程
        
        流程：
        1. 加载 Stage 1 预训练的 iTransformer 权重
        2. 冻结 iTransformer 和 velocity_net
        3. 只训练 uncertainty_estimator（使用 NLL Loss）
        4. 保存训练好的模型
        """
        # 生成 setting 字符串
        setting = self._get_setting(seed)
        
        print('=' * 80)
        print('Stage 2: Pretraining Uncertainty Estimator (iTransformer frozen)')
        print(f'Setting: {setting}')
        print('=' * 80)
        
        # 初始化 wandb（如果启用）
        if self._use_wandb() and not self._init_wandb(self.project, seed):
            return {}
        
        # 设置运行环境（由父类完成数据加载、模型初始化等）
        self._setup_run(seed)
        # 覆盖默认的保存路径到 Uncertainty Estimator 专用目录
        # ./results/runs/estimator/{dataset}/{setting}/seed_{seed}/
        self._setup_estimator_run_paths(setting, seed)
        
        # 检查并恢复运行检查点
        if self._check_run_exist(seed):
            # 如果检查点存在，直接恢复
            self._resume_run(seed)
            # 恢复后需要重新冻结
            self._freeze_itransformer()
            self._freeze_velocity_net()
        else:
            # 加载预训练的 iTransformer 权重
            self._load_itransformer_only(setting)
            # 冻结 iTransformer 和 velocity_net
            self._freeze_itransformer()
            self._freeze_velocity_net()
        
        self._run_print(f"Stage 2 - Uncertainty Estimator Pretraining (seed: {seed})")
        
        # 统计参数
        parameter_tables, model_parameters_num = count_parameters(self.model)
        trainable_params = sum(p.numel() for p in self.model.uncertainty_estimator.parameters() if p.requires_grad)
        self._run_print(f"Total parameters: {model_parameters_num}")
        self._run_print(f"Trainable parameters (Uncertainty Estimator): {trainable_params}")
        
        if self._use_wandb():
            wandb.run.summary["total_parameters"] = model_parameters_num
            wandb.run.summary["trainable_parameters"] = trainable_params
        
        # 训练循环
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopping.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs, early stopping ...."
                )
                break
            
            # 可恢复的随机性
            reproducible(seed + self.current_epoch)
            train_loss, train_metrics = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Training NLL loss: {train_loss:.6f}")
            self._run_print(f"Mean sigma: {train_metrics.get('mean_sigma', 0):.6f}")
            
            val_result = self._val()
            # test_result = self._test()
            
            self.current_epoch = self.current_epoch + 1
            
            # 使用 NLL Loss 作为早停指标（Stage 2 主要关注校准质量）
            self.early_stopping(val_result['nll_loss'], self.model)
            
            # 学习率调度
            old_lr = self.model_optim.param_groups[0]['lr']
            self.scheduler.step(val_result['nll_loss'])
            current_lr = self.model_optim.param_groups[0]['lr']
            
            # 记录学习率变化
            if old_lr != current_lr:
                self._run_print(f"Learning rate updated: {old_lr:.2e} --> {current_lr:.2e}")
            else:
                self._run_print(f"Learning rate: {current_lr:.2e}")
            
            self._save_run_check_point(seed)
            
            if self._use_wandb():
                # 记录训练指标
                wandb.log({'training_nll_loss': train_loss}, step=self.current_epoch)
                for key, value in train_metrics.items():
                    wandb.log({f"train_{key}": value}, step=self.current_epoch)
                wandb.log({f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                # wandb.log({f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)
                wandb.log({'learning_rate': current_lr}, step=self.current_epoch)
        
        # 加载最佳模型
        self._load_best_model()
        best_test_result = self._test()
        
        if self._use_wandb():
            for k, v in best_test_result.items():
                wandb.run.summary[f"best_test_{k}"] = v
            wandb.finish()
        
        print('=' * 80)
        print('Stage 2 Training Completed!')
        print(f'Best Test NLL Loss: {best_test_result.get("sigma_gauss_nll", "N/A")}')
        print(f'Best Test CRPS: {best_test_result.get("crps", "N/A")}')
        print('=' * 80)
        
        return best_test_result

    @torch.no_grad()
    def extract_residuals_on_test(
        self,
        seed: int = 42,
        save_path: str = "./results/analysis/electricity_residuals_fast.npz",
        use_origin_scale: bool = False,
        eps: float = 1e-6,
        return_result: bool = False,
    ) -> Dict[str, np.ndarray] | None:
        """
        在指定配置和 seed 下，对 Test Set 进行一次前向推理，提取：
            - Y: 真实未来序列，shape [N, P, D]
            - RevIN 统计量: mu_X, sigma_X，shape [N, 1, D]
            - iReflow 预测统计量: mu_Y_hat, sigma_Y_hat，shape [N, 1, D]（对预测时间步求均值）
            - 残差空间变量:
                  Z_RevIN = (Y - mu_X) / sigma_X
                  Z_PDN   = (Y - mu_Y_hat) / sigma_Y_hat

        Args:
            seed: 与训练时一致的随机种子，用于定位同一个 setting 与 checkpoint。
            save_path: 保存 .npz 文件的路径。
            use_origin_scale: 若为 True，则在原始尺度上计算 Y 和 RevIN 统计量；
                              否则在标准化后的尺度（dataloader 输出的 batch_x/batch_y）上计算。
            eps: 数值稳定性用的小常数，防止除零。

        Returns:
            若 return_result=True，则返回一个包含上述所有张量（转为 numpy）的字典；
            否则返回 None（适合命令行调用，避免在终端打印巨大数组）。
        """
        # 1. 生成 setting，并初始化运行环境与 dataloader / 模型
        setting = self._get_setting(seed)

        print("=" * 80)
        print("Stage 2 Analysis: Extract residuals on Test Set")
        print(f"Dataset: {getattr(self, 'dataset_type', getattr(self, 'data', 'custom'))}")
        print(f"Setting: {setting}")
        print(f"Seed   : {seed}")
        print("=" * 80)

        # 不做训练，只做一次完整的 _setup_run，复用 ProbForecastExp 的数据管线
        self._setup_run(seed)
        # 对于分析实验，希望在 Test Set 上遍历全部样本，因此强制关闭 fast_test
        # 重新初始化 dataloader（仅影响当前实例，不改变训练阶段默认行为）
        try:
            self._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
        except TypeError:
            # 兼容万一父类签名不同的情况，退回默认调用
            self._init_data_loader()
        # 将 run_save_dir / best_checkpoint_filepath 重定向到 Stage 2 的 estimator 目录
        self._setup_estimator_run_paths(setting, seed)

        # 加载 Stage 2 训练得到的 best_model.pth（只包含模型 state_dict）
        self._load_best_model()
        self.model.eval()

        device = self.device

        # 2. RevIN 用于统计历史序列的 μ_X, σ_X（不干预模型输入，只做统计）
        #    这里使用特征维度 = dataset.num_features（即最后一个维度）
        num_features = getattr(self, "enc_in", None)
        if num_features is None and hasattr(self, "dataset"):
            num_features = getattr(self.dataset, "num_features", None)
        if num_features is None:
            raise ValueError(
                "无法确定时间序列特征维度 num_features，请确保 enc_in 或 dataset.num_features 可用。"
            )
        revin = RevIN(num_features=num_features, affine=False).to(device)

        # 收集容器
        Ys = []
        mu_Xs = []
        sigma_Xs = []
        mu_Y_hats = []
        sigma_Y_hats = []
        Z_RevINs = []
        Z_PDNs = []

        # 3. 遍历 Test Set，逐批收集统计量与残差
        with tqdm(total=len(self.test_loader.dataset)) as progress_bar:
            for (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in self.test_loader:
                # dataloader 输出均为 [B, L, D] / [B, P, D] 形式
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                origin_x = origin_x.to(device).float()
                origin_y = origin_y.to(device).float()
                batch_x_date_enc = batch_x_date_enc.to(device).float()

                # 选择用于统计 RevIN 的尺度
                if use_origin_scale:
                    X_for_stats = origin_x
                    Y = origin_y
                else:
                    X_for_stats = batch_x
                    Y = batch_y

                # 3.1 RevIN 统计量 μ_X, σ_X （形状 [B, 1, D]）
                _ = revin(X_for_stats, mode="norm")
                mu_X = revin.mean  # [B, 1, D]
                sigma_X = revin.stdev  # [B, 1, D]

                # 3.2 iReflow 的点预测 y_hat 和不确定性 sigma （初始为 [B, P, D]）
                enc_features, y_hat, sigma = self.model.get_encoder_features(
                    batch_x, batch_x_date_enc
                )

                # 若在原始尺度上分析，则需要把 y_hat / sigma 从标准化尺度恢复到原始尺度
                # 这里复用 scaler.inverse_transform，仅对时间维度做逐步还原
                if use_origin_scale and hasattr(self, "scaler"):
                    # scaler 接受 [..., D] 形状，这里合并 batch 与 step 维度再还原
                    B, P, D = y_hat.shape
                    y_hat_flat = y_hat.reshape(B * P, D)
                    y_hat_orig = self.scaler.inverse_transform(y_hat_flat).reshape(B, P, D)
                    # 对 sigma，仅按尺度因子放大，不做平移。
                    std = self._get_scaler_std_tensor(dtype=sigma.dtype, device=device)
                    sigma_orig = sigma * std if std is not None else sigma
                    y_hat = y_hat_orig
                    sigma = sigma_orig

                # 将 μ_Y_hat, σ_Y_hat 聚合到时间维度的均值： [B, P, D] -> [B, 1, D]
                # mu_Y_hat = y_hat.mean(dim=1, keepdim=True)
                # sigma_Y_hat = sigma.mean(dim=1, keepdim=True)
                mu_Y_hat = y_hat
                sigma_Y_hat = sigma

                # 3.3 计算残差空间变量
                #     Z_RevIN: 使用历史统计量 (μ_X, σ_X)
                #     Z_PDN  : 使用预测统计量 (μ_Y_hat, σ_Y_hat)
                # 广播: μ_X, σ_X 为 [B, 1, D]，自动广播到 [B, P, D]
                Z_RevIN = (Y - mu_X) / (sigma_X + eps)
                Z_PDN = (Y - mu_Y_hat) / (sigma_Y_hat + eps)

                # 3.4 收集到 CPU / numpy
                Ys.append(Y.detach().cpu())
                mu_Xs.append(mu_X.detach().cpu())
                sigma_Xs.append(sigma_X.detach().cpu())
                mu_Y_hats.append(mu_Y_hat.detach().cpu())
                sigma_Y_hats.append(sigma_Y_hat.detach().cpu())
                Z_RevINs.append(Z_RevIN.detach().cpu())
                Z_PDNs.append(Z_PDN.detach().cpu())

                progress_bar.update(batch_x.shape[0])

        # 4. 拼接所有 batch，得到全 Test Set 上的结果
        def _cat_to_numpy(tensor_list):
            return torch.cat(tensor_list, dim=0).numpy() if tensor_list else None

        Y_all = _cat_to_numpy(Ys)
        mu_X_all = _cat_to_numpy(mu_Xs)
        sigma_X_all = _cat_to_numpy(sigma_Xs)
        mu_Y_hat_all = _cat_to_numpy(mu_Y_hats)
        sigma_Y_hat_all = _cat_to_numpy(sigma_Y_hats)
        Z_RevIN_all = _cat_to_numpy(Z_RevINs)
        Z_PDN_all = _cat_to_numpy(Z_PDNs)

        result = {
            "Y": Y_all,
            "mu_X": mu_X_all,
            "sigma_X": sigma_X_all,
            "mu_Y_hat": mu_Y_hat_all,
            "sigma_Y_hat": sigma_Y_hat_all,
            "Z_RevIN": Z_RevIN_all,
            "Z_PDN": Z_PDN_all,
        }

        # 5. 保存到 .npz 方便后续分析
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **{k: v for k, v in result.items() if v is not None})

        print(f"Residuals and statistics saved to: {save_path}")
        print(f"Y shape           : {None if Y_all is None else Y_all.shape}")
        print(f"mu_X / sigma_X    : {None if mu_X_all is None else mu_X_all.shape}")
        print(f"mu_Y_hat / sigma_Y_hat: {None if mu_Y_hat_all is None else mu_Y_hat_all.shape}")
        print(f"Z_RevIN / Z_PDN   : "
              f"{None if Z_RevIN_all is None else Z_RevIN_all.shape}, "
              f"{None if Z_PDN_all is None else Z_PDN_all.shape}")

        # 命令行场景通常不需要在终端打印全部结果，默认不返回字典，避免 Fire 把 result 打印出来
        if return_result:
            return result
        else:
            return None

    @torch.no_grad()
    def export_decoupling_case_study_data_on_test(
        self,
        seed: int = 1,
        save_path: str = "./results/analysis/Traffic/Traffic_decoupling_case_study_data.npz",
        eps: float = 1e-6,
        return_result: bool = False,
    ) -> Dict[str, np.ndarray] | None:
        """
        Dedicated exporter for Experiment 1.

        Always writes raw-scale future targets and raw-scale PDN macro statistics:
          - Y: raw future target
          - mu_X / sigma_X: raw historical RevIN statistics
          - mu_Y_hat / sigma_Y_hat: raw-scale predictive macro components
          - Z_PDN: PDN residuals computed in raw scale

        This avoids ambiguity around whether the saved tensors are in normalized
        or original scale and is intended for `src/analysis/decoupling_case_study.py`.
        """
        return self.extract_residuals_on_test(
            seed=seed,
            save_path=save_path,
            use_origin_scale=True,
            eps=eps,
            return_result=return_result,
        )


if __name__ == '__main__':
    setproctitle.setproctitle('UncertaintyEstimator_Pretrain')
    
    import fire
    fire.Fire(UncertaintyEstimatorPretrainExp)
