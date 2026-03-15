"""
iReflow实验脚本
用于训练和评估iReflow模型
"""
from dataclasses import dataclass, field
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
import argparse
from src.models.iReflow import iReflow
from src.experiments.prob_forecast import ProbForecastExp
from torch.optim import *
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
from torch_timeseries.utils.early_stop import EarlyStopping
import numpy as np
import setproctitle
try:
    import wandb
except:
    print("Warning: wandb is not installed, some functionality may not work.")

from src.utils.uncertainty_eval import compute_sigma_metrics


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class iReflowEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """保存模型检查点"""
        if self.verbose:
            self.trace_func(
                f"Validation CRPS decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


@dataclass
class iReflowExp(ProbForecastExp):
    """
    iReflow实验类
    """
    # 基本/兼容 iTransformer bash 的配置
    wandb_project: str = None              # 对应 --wandb_project
    model_id: str = "test"                 # 对应 --model_id
    model: str = "iReflow"                 # 对应 --model（保持与 model_type 一致）
    data: str = "custom"                   # 对应 --data，用于映射到 dataset_type
    root_path: str = "./data/"             # 对应 --root_path
    des: str = "Exp"                       # 对应 --des
    itr: int = 1                           # 对应 --itr（目前主要用于兼容脚本参数）
    checkpoints: str = "./results/runs/iTransformer/"  # 对应 --checkpoints
    seq_len: int = 96                      # 对应 --seq_len，将与父类的 windows 对齐

    # 模型配置
    model_type: str = "iReflow"
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2  # num of iTransformer encoder layers
    flow_layers: int = 3  # num of Velocity Network layers
    features: str = 'M'
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    embed: str = 'timeF'
    freq: str = 'h'
    activation: str = 'gelu'
    output_attention: bool = False
    use_norm: bool = True
    class_strategy: str = 'projection'
    factor: int = 1
    use_relative_space: bool = True
    use_itransformer_enc: bool = True  # False 时改用轻量的 RevIN 倒置嵌入作为 Cross-Attention 条件
    
    # 训练配置
    is_training: int = 1
    lr: float = 0.0001
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    lr_patience: int = 1  # 学习率调度器的patience
    
    # Flow配置
    num_sampling_steps: int = 1  # ODE求解步数，1表示one-step generation
    temperature: float = 1.0  # 采样温度
    num_samples: int = 100  # 测试时生成的样本数
    
    # 损失函数
    loss_func_type: str = 'mse'
    
    def __post_init__(self):
        # ---------- 与父类配置的映射 ----------
        # 1) model / model_type 对齐，便于 wandb 命名等
        if self.model:
            self.model_type = self.model
        
        # 2) data -> dataset_type（ProbForecastExp/FörercastExp 使用 dataset_type）
        if getattr(self, "data", None):
            self.dataset_type = self.data
        
        # 3) root_path -> data_path（ProbForecastExp._init_dataset 使用 data_path 作为 root）
        #    兼容 iTransformer 脚本：iTransformer 直接使用 root_path 作为数据集类的 root
        #    数据集类会在 root_path 下创建数据集子目录（如 ETTm2/ETTm2.csv）
        if getattr(self, "root_path", None) is not None:
            # 直接使用 root_path 作为 data_path（数据集类的 root 参数）
            # 这样数据集类会在 root_path 下自动创建对应的数据集目录
            self.data_path = self.root_path
        
        # 4) seq_len -> windows（ForecastSettings 中的窗口长度）
        if getattr(self, "seq_len", None) is not None:
            self.windows = self.seq_len
        
        # 5) checkpoints -> save_dir（ForecastExp 用 save_dir 作为根目录）
        if getattr(self, "checkpoints", None):
            norm_cp = os.path.normpath(self.checkpoints)
            parts = norm_cp.split(os.sep)
            if "runs" in parts:
                # 例如 ./results/runs/iTransformer/ -> ./results
                runs_idx = parts.index("runs")
                base_parts = parts[:runs_idx] or [os.curdir]
                self.save_dir = os.path.join(*base_parts)
            else:
                # 没有 runs 就取上一层目录
                self.save_dir = os.path.dirname(norm_cp) or "."
        
        # 6) wandb_project -> project & wandb 开关
        if getattr(self, "wandb_project", None):
            # ForecastExp.config_wandb 会设置 self.project 与 self.wandb
            self.config_wandb(self.wandb_project)
        
        # 7) learning_rate -> lr（父类 ForecastExp 使用 lr 参数）
        # 将 learning_rate 同步到 lr，确保使用统一的学习率参数
        # 这样既兼容 --learning_rate 命令行参数，又统一使用父类的 lr
        if hasattr(self, "learning_rate"):
            self.lr = self.learning_rate

        # 创建模型配置
        self.model_configs = argparse.Namespace()
        self.model_configs.seq_len = self.windows
        self.model_configs.pred_len = self.pred_len
        self.model_configs.d_model = self.d_model
        self.model_configs.n_heads = self.n_heads
        self.model_configs.e_layers = self.e_layers
        self.model_configs.flow_layers = self.flow_layers
        self.model_configs.d_ff = self.d_ff
        self.model_configs.dropout = self.dropout
        self.model_configs.embed = self.embed
        self.model_configs.freq = self.freq
        self.model_configs.activation = self.activation
        self.model_configs.output_attention = self.output_attention
        self.model_configs.use_norm = self.use_norm
        self.model_configs.class_strategy = self.class_strategy
        self.model_configs.factor = self.factor
        self.model_configs.num_sampling_steps = self.num_sampling_steps
        # Loss 配置与梯度通路控制
        self.model_configs.nll_loss_weight = getattr(self, "nll_loss_weight", 1.0)
        self.model_configs.velocity_loss_weight = getattr(self, "velocity_loss_weight", 1.0)
        self.model_configs.use_relative_space = self.use_relative_space
        self.model_configs.use_itransformer_enc = self.use_itransformer_enc
    
    def _init_model(self):
        """初始化模型"""
        self.model = iReflow(self.model_configs).to(self.device)
        # 注意：在 torch_timeseries 的 _setup_run() 中，会在 _init_model() 之后调用 _init_optimizer()
        # 因此 optimizer / scheduler 必须在 _init_optimizer() 里创建，避免被父类覆盖导致 scheduler 绑定错误的 optimizer。

    def _init_optimizer(self):
        """初始化优化器与学习率调度器（遵循 torch_timeseries 的 _setup_run 调用顺序）"""
        # 根据 is_training 参数决定训练哪些部分
        # 注意：对于 is_training=1，冻结操作应该在加载预训练权重之后进行
        # 因此这里先不冻结，冻结操作将在 _freeze_itransformer() 和 _freeze_uncertainty_estimator() 中进行
        if self.is_training == 1:
            # Stage 3: 只优化 velocity_net 的参数
            # iTransformer 和 uncertainty_estimator 将从预训练权重加载并冻结
            trainable_params = list(self.model.velocity_net.parameters())
            self.model_optim = torch.optim.Adam(trainable_params, lr=self.lr)
            print(
                "Initialized optimizer for Stage 3: "
                "will freeze iTransformer and Uncertainty Estimator after loading weights, "
                "only training Velocity Network"
            )
        else:
            # 训练整个模型（is_training=2 或默认情况）
            self.model_optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            if self.is_training == 2:
                print("Initialized optimizer: training entire model (iTransformer + Uncertainty Estimator + Velocity Network)")

        # 学习率调度器（必须绑定到最终用于训练的 optimizer）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optim, mode="min", factor=0.5, patience=self.lr_patience
        )
    
    def _freeze_itransformer(self):
        """冻结 iTransformer 参数（在加载预训练权重后调用）"""
        if self.is_training == 1:
            if not hasattr(self.model, 'itransformer'):
                print("Model has no itransformer (use_itransformer_enc=False), skipping freeze.")
                return
            for param in self.model.itransformer.parameters():
                param.requires_grad = False
            print("iTransformer parameters frozen after loading pretrained weights")
        
        # 打印模型参数
        # num_params = count_parameters(self.model)
        # print(f"Model initialized with {num_params} parameters")
    
    def _freeze_uncertainty_estimator(self):
        """冻结 Uncertainty Estimator 参数（在加载预训练权重后调用）"""
        if self.is_training == 1:
            if not hasattr(self.model, 'uncertainty_estimator'):
                print("Model has no uncertainty_estimator, skipping freeze.")
                return
            for param in self.model.uncertainty_estimator.parameters():
                param.requires_grad = False
            print("Uncertainty Estimator parameters frozen after loading pretrained weights")
    
    def _setup_early_stopper(self):
        """设置早停和检查点路径"""
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )
        # Early Stopping
        self.early_stopping = iReflowEarlyStopping(
            patience=self.patience, verbose=True, path=self.best_checkpoint_filepath
        )
    
    def _process_train_batch(
        self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
    ):
        """
        处理训练批次
        
        Args:
            batch_x: [B, L, D] 历史序列
            batch_y: [B, P, D] 未来序列
            batch_x_date_enc: [B, L, T] 历史时间标记
            batch_y_date_enc: [B, P, T] 未来时间标记
        Returns:
            pred: [B, P, D] 预测（点预测）
            true: [B, P, D] 真实值
            loss: scalar 损失
            loss_dict: dict 包含详细损失和指标
        """
        # 前向传播
        loss, loss_dict, y_hat = self.model(
            x_enc=batch_x,
            x_mark_enc=batch_x_date_enc,
            x_dec=None,
            x_mark_dec=batch_y_date_enc,
            y_gt=batch_y,
            mode='train'
        )
        
        return y_hat, batch_y, loss, loss_dict
    
    def _train(self):
        """训练一个epoch"""
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_losses = []
            # 收集所有批次的详细指标
            train_metrics = {
                'total_loss': [],
                'velocity_loss': [],
                'nll_loss': [],
                'mean_sigma': [],
                'min_sigma': [],
                'max_sigma': [],
                'mae_point': [],
                'gate_mean': [],
                'gate_var': [],
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
                
                # 前向传播
                pred, true, loss, loss_dict = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                # 优化器步进
                self.model_optim.step()
                
                train_losses.append(loss.item())
                
                # 动态收集所有指标（包括log_sigma_stats添加的额外统计量）
                for key, value in loss_dict.items():
                    if key not in train_metrics:
                        train_metrics[key] = []
                    train_metrics[key].append(value)
                
                # 更新进度条
                progress_bar.set_postfix(
                    loss=loss.item(),
                    avg_loss=np.mean(train_losses)
                )
                progress_bar.update(batch_x.shape[0])
        
        avg_train_loss = np.mean(train_losses)
        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in train_metrics.items() if values}
        return avg_train_loss, avg_metrics
    
    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        处理验证/测试批次，生成样本并返回预测和真实值
        
        Args:
            batch_x: [B, L, D] 历史序列
            batch_y: [B, P, D] 未来序列（用于获取真实值）
            batch_x_date_enc: [B, L, T] 历史时间标记
            batch_y_date_enc: [B, P, T] 未来时间标记
            
        Returns:
            preds: [B, P, D, num_samples] 预测样本
            truths: [B, P, D] 真实值
        """
        # 获取当前评估时使用的样本数（如果设置了，否则使用全部样本数）
        num_samples = getattr(self, '_num_samples_for_eval', self.num_samples)
        
        # 生成样本
        samples, y_hat, sigma = self.model.forecast(
            x_enc=batch_x,
            x_mark_enc=batch_x_date_enc,
            num_samples=num_samples,
            temperature=self.temperature
        )
        
        # samples: [B, num_samples, P, D]
        # 转换维度以匹配指标期望的格式: [B, P, D, num_samples]
        samples = samples.permute(0, 2, 3, 1)  # [B, P, D, num_samples]
        
        # 返回预测和真实值（注意：基类的_evaluate会处理反归一化）
        preds = samples  # [B, P, D, num_samples]
        truths = batch_y  # [B, P, D]
        
        return preds, truths

    def _evaluate(self, dataloader):
        """
        重写评估逻辑：
        - 保持父类的采样型概率指标（CRPS/QICE/PICP/...）
        - 额外评估由 uncertainty_estimator 输出的 sigma 的“校准/锐度”

        注意：sigma/y_hat 是模型输出的同一尺度（通常是数据集 scaler 的缩放后尺度）。
        为避免依赖 scaler 的内部参数（sigma 需要乘缩放但不平移），这里的 sigma 指标默认在
        batch_y 的尺度上计算（即 invtrans_loss 为 True 时也不做 inverse_transform）。
        """
        self.model.eval()
        self.metrics.reset()

        # 默认只保留一个关键区间（90%）来评估 coverage + sharpness 的权衡，避免指标过多
        interval_levels = getattr(self, "sigma_interval_levels", [0.9])
        pit_bins = int(getattr(self, "pit_bins", 20))
        # 默认仅记录最关键的一些 sigma 指标；你也可以在配置里覆写 sigma_metric_keys
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

        # 获取当前评估时使用的样本数（如果设置了，否则使用全部样本数）
        num_samples = getattr(self, "_num_samples_for_eval", self.num_samples)

        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            with torch.no_grad():
                for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()
                    origin_y = origin_y.to(self.device).float()
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()

                    # 生成采样预测 + 点预测与 sigma
                    samples, y_hat, sigma = self.model.forecast(
                        x_enc=batch_x,
                        x_mark_enc=batch_x_date_enc,
                        num_samples=num_samples,
                        temperature=self.temperature,
                    )  # samples: [B, S, P, D], y_hat/sigma: [B, P, D]

                    # 采样型概率指标：转换为 [B, P, D, S]
                    preds = samples.permute(0, 2, 3, 1).contiguous()
                    truths = batch_y
                    if self.invtrans_loss:
                        # 采样指标仍按父类逻辑：反归一化后与 origin_y 对齐
                        preds = self.scaler.inverse_transform(preds)
                        truths = origin_y

                    self.metrics.update(
                        preds.contiguous().cpu().detach(),
                        truths.contiguous().cpu().detach(),
                    )

                    # sigma 指标：在 batch_y/y_hat/sigma 的同一尺度上计算（不 inverse_transform）
                    sigma_metrics = compute_sigma_metrics(
                        y=batch_y.detach().cpu(),
                        mu=y_hat.detach().cpu(),
                        sigma=sigma.detach().cpu(),
                        interval_levels=list(interval_levels),
                        pit_bins=pit_bins,
                    )
                    # 只保留关键指标，避免日志/面板过于拥挤
                    for k, v in sigma_metrics.items():
                        if sigma_metric_keys is not None and k not in sigma_metric_keys:
                            continue
                        sigma_sums[k] = sigma_sums.get(k, 0.0) + float(v)
                    sigma_counts += 1

                    progress_bar.update(batch_x.shape[0])

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        if sigma_counts > 0:
            result.update({f"sigma_{k}": float(v / sigma_counts) for k, v in sigma_sums.items()})
        return result
    
    def _diagnose_sample_diversity(self, n_diag_samples: int = 30) -> dict:
        """
        Diagnostic 1: Sample Collapse Detection.

        Generates `n_diag_samples` predictions for the first validation batch and
        measures inter-sample variance.  If the variance is close to zero the
        velocity network has learned to cancel epsilon without conditioning on
        enc_features, causing all samples to collapse to the same point.

        Key metrics
        -----------
        inter_sample_var
            Mean variance across the sample dimension [B, n, P, D] → scalar.
            Expected source variance  ≈  sigma_X².
            collapse_ratio = inter_sample_var / sigma_X²:
              ~0  →  full collapse (samples identical, diversity lost)
              ~1  →  diverse samples (source spread is preserved)
              >1  →  samples more spread than source (unlikely with well-trained net)

        dist_sample_mean_to_mu_X
            MAE between the mean prediction and the RevIN historical mean μ_X.
            A near-zero value means the network predicts the historical mean
            unconditionally (ignores enc_features conditioning).

        dist_sample_mean_to_truth
            MAE between the mean prediction and the ground-truth future.
            Indicates point-prediction quality independent of sample diversity.

        Returns
        -------
        dict with keys: collapse_ratio, inter_sample_var, source_var,
                        dist_sample_mean_to_mu_X, dist_sample_mean_to_truth
        """
        self.model.eval()
        with torch.no_grad():
            # ── grab the first validation batch only ──────────────────────────
            batch = next(iter(self.val_loader))
            batch_x          = batch[0].to(self.device).float()
            batch_y          = batch[1].to(self.device).float()
            batch_x_date_enc = batch[4].to(self.device).float()

            # ── generate n_diag_samples predictions for the SAME input ────────
            # samples: [B, n_diag_samples, P, D]
            # mu_X:    [B, 1, D]   sigma_X: [B, 1, D]
            samples, mu_X, sigma_X = self.model.sample(
                x_enc=batch_x,
                x_mark_enc=batch_x_date_enc,
                num_samples=n_diag_samples,
                temperature=self.temperature,
            )

            # ── inter-sample variance (across the sample dimension) ────────────
            # var over dim=1 → [B, P, D], then mean to scalar
            inter_sample_var = samples.var(dim=1).mean().item()

            # expected source variance: E[sigma_X²]
            source_var = (sigma_X ** 2).mean().item()

            # collapse ratio: 0 = fully collapsed, 1 = source-level diversity
            collapse_ratio = inter_sample_var / (source_var + 1e-8)

            # ── mean prediction vs μ_X and ground truth ───────────────────────
            sample_mean = samples.mean(dim=1)                        # [B, P, D]
            mu_X_exp    = mu_X.expand_as(sample_mean)               # [B, P, D]

            dist_to_mu_X = (sample_mean - mu_X_exp).abs().mean().item()
            dist_to_truth = (sample_mean - batch_y).abs().mean().item()

        # ── print diagnostics ─────────────────────────────────────────────────
        self._run_print("=" * 64)
        self._run_print("[Diag-1] Sample Diversity (Collapse) Analysis")
        self._run_print(f"  n_diag_samples              : {n_diag_samples}")
        self._run_print(f"  inter-sample variance       : {inter_sample_var:.6f}")
        self._run_print(f"  source variance (sigma_X²)  : {source_var:.6f}")
        self._run_print(f"  collapse ratio              : {collapse_ratio:.4f}"
                        f"  (≈0 → collapsed | ≈1 → diverse)")
        self._run_print(f"  |sample_mean - mu_X|  (MAE) : {dist_to_mu_X:.6f}"
                        f"  (≈0 → predicts historical mean unconditionally)")
        self._run_print(f"  |sample_mean - truth| (MAE) : {dist_to_truth:.6f}")
        self._run_print("=" * 64)

        return {
            "diag_collapse_ratio":              collapse_ratio,
            "diag_inter_sample_var":            inter_sample_var,
            "diag_source_var":                  source_var,
            "diag_dist_sample_mean_to_mu_X":    dist_to_mu_X,
            "diag_dist_sample_mean_to_truth":   dist_to_truth,
        }

    def _val(self):
        """验证：使用较少的样本数以加快验证速度"""
        # 设置验证时使用的样本数
        self._num_samples_for_eval = min(self.num_samples, 30)
        
        # 计算验证损失（用于学习率调度）
        self.model.eval()
        val_losses = []
        # 动态收集验证阶段的详细指标
        val_metrics = {}
        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in self.val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                
                # 计算损失
                loss, loss_dict, y_hat = self.model(
                    x_enc=batch_x,
                    x_mark_enc=batch_x_date_enc,
                    x_dec=None,
                    x_mark_dec=batch_y_date_enc,
                    y_gt=batch_y,
                    mode='train'
                )
                val_losses.append(loss.item())
                
                # 收集验证阶段的详细指标（包括log_sigma_stats添加的额外统计量）
                for key, value in loss_dict.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)
        
        # 调用基类的_val()方法获取概率预测指标
        result = super()._val()
        
        # 添加平均损失和详细指标
        result['loss'] = np.mean(val_losses)
        # 计算平均指标并添加到结果中（不添加val_前缀，因为wandb.log会统一添加）
        for key, values in val_metrics.items():
            if values:
                result[key] = np.mean(values)

        # ── Diagnostic 1: sample collapse detection ───────────────────────────
        diag = self._diagnose_sample_diversity(n_diag_samples=30)
        result.update(diag)
        
        # ── Diagnostic 2: enc_features effectiveness ─────────────────────────
        if 'enc_features_diff' in result:
            enc_diff = result['enc_features_diff']
            self._run_print("=" * 64)
            self._run_print("[Diag-2] Encoder Features Effectiveness")
            self._run_print(f"  |v_pred - v_zero| (MAE) : {enc_diff:.6f}")
            self._run_print(f"  Interpretation:")
            self._run_print(f"    ≈ 0 → enc_features ignored (cross-attention ineffective)")
            self._run_print(f"    > 0 → enc_features contribute to velocity prediction")
            self._run_print("=" * 64)
        
        # 清理标志
        delattr(self, '_num_samples_for_eval')
        return result
    
    @property
    def result_related_configs(self):
        """
        重写 result_related_configs 属性，确保所有值都可以被 JSON 序列化
        排除不可序列化的对象（如 argparse.Namespace, 模型对象等）
        """
        from torch_timeseries.utils import asdict_exc
        from torch_timeseries.core.experiments.settings import BaseIrrelevant
        import json
        
        ident = asdict_exc(self, BaseIrrelevant)
        
        # 过滤掉不可序列化的对象
        serializable_ident = {}
        for k, v in ident.items():
            try:
                # 尝试序列化以检查是否可序列化
                json.dumps(v)
                serializable_ident[k] = v
            except (TypeError, ValueError):
                # 如果不可序列化，转换为字符串表示
                # 对于 argparse.Namespace 等对象，转换为字符串
                if isinstance(v, (argparse.Namespace,)):
                    # 对于 Namespace 对象，转换为字典
                    serializable_ident[k] = vars(v) if hasattr(v, '__dict__') else str(v)
                elif hasattr(v, '__class__'):
                    # 对于其他对象，使用类型名称
                    serializable_ident[k] = type(v).__name__
                else:
                    serializable_ident[k] = str(v)
        
        return serializable_ident
    
    def _check_run_exist(self, seed: str):
        """
        重写 _check_run_exist 方法，使用 result_related_configs 而不是 asdict(self)
        以避免序列化模型对象等不可序列化的对象
        """
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        import json
        with open(
            os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            # 使用 result_related_configs 而不是 asdict(self)，避免序列化模型对象
            json.dump(self.result_related_configs, f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists
    
    def _get_setting(self, seed=0):
        """
        生成实验设置字符串，用于 checkpoints 路径命名
        格式参考 iTransformer 的 setting 格式
        """
        # 使用 seq_len 或 windows（已对齐）
        seq_len = getattr(self, 'seq_len', self.windows)
        label_len = 48  # iReflow 不使用 label_len，设为 48 以保持格式一致
        
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.model_id,
            self.dataset_type,
            self.features,
            seq_len,
            label_len,
            self.pred_len,
            self.d_model,
            self.n_heads,
            self.e_layers,
            self.d_layers,
            self.d_ff,
            self.factor,
            self.embed,
            True,  # distil (iReflow 不使用，设为 True)
            self.des,
            self.class_strategy,
            0
        )
        return setting
    
    def _load_uncertainty_estimator(self, setting):
        """
        加载预训练的 Uncertainty Estimator 权重（用于 Stage 3: is_training=1 模式）
        
        路径规则：从 Stage 2 的 run_save_dir 加载 best_model.pth
        格式：{save_dir}/runs/{model_type}/{dataset_type}/{setting}/seed_{seed}/best_model.pth
        
        Args:
            setting: 实验设置字符串
        """
        # 构建 Stage 2 的 checkpoint 路径
        # 注意：这里假设 Stage 2 使用了相同的 setting 和 seed
        # 路径格式：./results/runs/estimator/{dataset}/{setting}/seed_{seed}/best_model.pth
        dataset = getattr(self, "dataset_type", getattr(self, "data", "custom"))
        seed = getattr(self, "current_seed", 42)
        stage2_run_dir = os.path.join(
            "./results", "runs", "estimator", dataset, setting, f"seed_{seed}"
        )
        best_model_path = os.path.join(stage2_run_dir, 'best_model.pth')
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Stage 2 checkpoint not found at {best_model_path}. "
                f"Please ensure the Uncertainty Estimator has been pretrained using "
                f"pretrain_uncertainty_estimator.py first."
            )
        
        # revin 版本的模型已移除 uncertainty_estimator，跳过加载
        if not hasattr(self.model, 'uncertainty_estimator'):
            print("Model has no uncertainty_estimator (revin version), skipping Stage 2 checkpoint loading.")
            return

        print(f'Loading pretrained Uncertainty Estimator from {best_model_path}')
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        
        # 提取 uncertainty_estimator 的权重
        uncertainty_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('uncertainty_estimator.'):
                # 移除 'uncertainty_estimator.' 前缀
                new_key = key[len('uncertainty_estimator.'):]
                uncertainty_state_dict[new_key] = value
        
        if not uncertainty_state_dict:
            raise ValueError(
                f"Could not find uncertainty_estimator weights in checkpoint. "
                f"Available keys: {list(checkpoint.keys())[:10]}..."
            )
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.uncertainty_estimator.load_state_dict(
            uncertainty_state_dict, strict=True
        )
        
        if missing_keys:
            print(f'Warning: Missing keys in uncertainty_estimator: {missing_keys}')
        if unexpected_keys:
            print(f'Warning: Unexpected keys: {unexpected_keys}')
        
        print('Uncertainty Estimator weights loaded successfully from Stage 2 checkpoint.')
    
    def _load_itransformer_only(self, setting):
        """
        只加载 iTransformer 的权重（用于 Stage 2 和 Stage 3）
        路径规则：os.path.join(self.checkpoints, setting) + '/checkpoint.pth'
        """
        # revin 版本 use_itransformer_enc=False 时模型无 itransformer，跳过加载
        if not hasattr(self.model, 'itransformer'):
            print("Model has no itransformer (use_itransformer_enc=False), skipping iTransformer weight loading.")
            return

        path = os.path.join(self.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {best_model_path}. "
                f"Please ensure the iTransformer model has been trained and saved."
            )
        
        print(f'Loading iTransformer weights from {best_model_path}')
        # 使用 weights_only=True 因为 iTransformer checkpoint 只包含模型权重
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        
        # 处理嵌套字典的情况（checkpoint 可能包含 'model' 键）
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print('Found nested checkpoint structure, extracting model state_dict...')
            checkpoint = checkpoint['model']
        
        # 打印 checkpoint 的键名以便调试
        # checkpoint_keys = list(checkpoint.keys())
        # print(f'Checkpoint keys (all {len(checkpoint_keys)} keys): {checkpoint_keys}')
        
        # 检查是否是 iTransformer checkpoint（键名包含 enc_embedding, encoder, projector）
        # 还是 iReflow checkpoint（包含 itransformer, velocity_net 等）
        # checkpoint_keys_str = ' '.join(str(k) for k in checkpoint.keys())
        is_itransformer_checkpoint = any(
            key.startswith('enc_embedding') or 
            key.startswith('encoder') or 
            key.startswith('projector')
            for key in checkpoint.keys()
        )
        
        if is_itransformer_checkpoint:
            # 这是 iTransformer checkpoint，需要提取 itransformer 部分的权重
            print('Detected iTransformer checkpoint, loading itransformer weights...')
            
            # 构建 itransformer 的 state_dict（直接使用原始键名，因为 load_state_dict 是直接加载到子模块）
            itransformer_state_dict = {}
            for key, value in checkpoint.items():
                # 跳过非模型参数（如 optimizer, epoch 等）
                if key in ['optimizer', 'scheduler', 'epoch', 'current_epoch', 'rng_state', 'early_stopping']:
                    continue
                # 直接使用原始键名（不需要添加 'itransformer.' 前缀，因为是直接加载到子模块）
                itransformer_state_dict[key] = value
            
            if not itransformer_state_dict:
                raise ValueError("Could not extract itransformer weights from iTransformer checkpoint. Checkpoint may be empty or in unexpected format.")
            
            # 只加载 itransformer 部分的权重
            missing_keys, unexpected_keys = self.model.itransformer.load_state_dict(
                itransformer_state_dict, strict=False
            )
            
            if missing_keys:
                print(f'Warning: Missing keys in itransformer: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Warning: Unexpected keys: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else f'Warning: Unexpected keys: {unexpected_keys}')
            
            print('iTransformer weights loaded successfully. Velocity network and uncertainty estimator will be trained.')
        else:
            # 这是 iReflow checkpoint，尝试提取 itransformer 部分
            print('Detected iReflow checkpoint, extracting itransformer weights...')
            itransformer_state_dict = {}
            for key, value in checkpoint.items():
                # 跳过非模型参数
                if key in ['optimizer', 'scheduler', 'epoch', 'current_epoch', 'rng_state', 'early_stopping']:
                    continue
                if key.startswith('itransformer.'):
                    # 移除 'itransformer.' 前缀
                    new_key = key[len('itransformer.'):]
                    itransformer_state_dict[new_key] = value
            
            if not itransformer_state_dict:
                # 尝试其他可能的键名格式
                print('Trying alternative key formats...')
                for key, value in checkpoint.items():
                    if key in ['optimizer', 'scheduler', 'epoch', 'current_epoch', 'rng_state', 'early_stopping']:
                        continue
                    # 检查是否是 iTransformer 的直接键（没有前缀）
                    if any(k in key for k in ['enc_embedding', 'encoder']):
                        itransformer_state_dict[key] = value
                
                if not itransformer_state_dict:
                    all_keys = list(checkpoint.keys())
                    raise ValueError(
                        f"Could not find itransformer weights in the checkpoint. "
                        f"Available keys ({len(all_keys)} total): {all_keys}. "
                        f"Please ensure the checkpoint contains iTransformer weights. "
                        f"Expected keys: 'enc_embedding', 'encoder', 'projector' for iTransformer checkpoint, "
                        f"or 'itransformer.*' for iReflow checkpoint."
                    )
            
            missing_keys, unexpected_keys = self.model.itransformer.load_state_dict(
                itransformer_state_dict, strict=False
            )
            
            if missing_keys:
                print(f'Warning: Missing keys in itransformer: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Warning: Unexpected keys: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else f'Warning: Unexpected keys: {unexpected_keys}')
            
            print('iTransformer weights extracted successfully.')
    
    def _load_checkpoint_model(self, setting):
        """
        从 checkpoints 加载模型
        路径规则：os.path.join(self.checkpoints, setting) + '/checkpoint.pth'
        
        支持两种 checkpoint 格式：
        1. iReflow checkpoint: 直接加载整个模型
        2. iTransformer checkpoint: 只加载 itransformer 部分的权重
        """
        path = os.path.join(self.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {best_model_path}. "
                f"Please ensure the model has been trained and saved."
            )
        
        print(f'Loading model from {best_model_path}')
        # 使用 weights_only=True 因为 checkpoint 只包含模型权重（state_dict）
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        
        # 处理嵌套字典的情况（checkpoint 可能包含 'model' 键）
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print('Found nested checkpoint structure, extracting model state_dict...')
            checkpoint = checkpoint['model']
        
        # 检查是否是 iTransformer checkpoint（键名包含 enc_embedding, encoder, projector）
        # 还是 iReflow checkpoint（包含 itransformer, velocity_net 等）
        is_itransformer_checkpoint = any(
            key.startswith('enc_embedding') or 
            key.startswith('encoder') or 
            key.startswith('projector')
            for key in checkpoint.keys()
        )
        
        if is_itransformer_checkpoint:
            # 这是 iTransformer checkpoint，需要提取 itransformer 部分的权重
            print('Detected iTransformer checkpoint, loading itransformer weights...')
            
            # 构建 itransformer 的 state_dict（直接使用原始键名，因为 load_state_dict 是直接加载到子模块）
            itransformer_state_dict = {}
            for key, value in checkpoint.items():
                # 跳过 projector（iReflow 不使用）
                if key.startswith('projector'):
                    continue
                # 跳过非模型参数
                if key in ['optimizer', 'scheduler', 'epoch', 'current_epoch', 'rng_state', 'early_stopping']:
                    continue
                # 直接使用原始键名（不需要添加 'itransformer.' 前缀，因为是直接加载到子模块）
                itransformer_state_dict[key] = value
            
            if not itransformer_state_dict:
                raise ValueError("Could not extract itransformer weights from iTransformer checkpoint. Checkpoint may be empty or in unexpected format.")
            
            # 只加载 itransformer 部分的权重
            missing_keys, unexpected_keys = self.model.itransformer.load_state_dict(
                itransformer_state_dict, strict=False
            )
            
            if missing_keys:
                print(f'Warning: Missing keys in itransformer: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Warning: Unexpected keys: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else f'Warning: Unexpected keys: {unexpected_keys}')
            
            print('iTransformer weights loaded successfully. Velocity network and uncertainty estimator remain untrained.')
        else:
            # 这是 iReflow checkpoint，直接加载整个模型
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            
            if missing_keys:
                print(f'Warning: Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'Warning: Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Warning: Unexpected keys: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else f'Warning: Unexpected keys: {unexpected_keys}')
            
            print('iReflow model loaded successfully')
    
    def _resume_run(self, seed):
        """恢复运行检查点（重写父类方法以支持调度器状态恢复）"""
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]
        
        # 恢复调度器状态（如果存在）
        if "scheduler" in check_point:
            self.scheduler.load_state_dict(check_point["scheduler"])
            print("Scheduler state restored from checkpoint")
        else:
            print("Warning: Scheduler state not found in checkpoint, using default state")

        self.early_stopping.set_state(check_point["early_stopping"])
    
    def _load_best_model(self):
        """加载最佳模型（从 run_save_dir）"""
        # 使用 weights_only=True 因为 best_model.pth 只包含模型权重（state_dict）
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device, weights_only=True)
        )
    
    def _save_run_check_point(self, seed):
        """保存运行检查点"""
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopping.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
    
    def run(self, seed=42) -> Dict[str, float]:
        """
        运行实验，支持wandb追踪和检查点恢复
        
        训练模式说明：
        - is_training=0: 不训练任何模型，只加载已保存的权重并进行测试评估
        - is_training=1: Stage 3 - 只训练 Velocity Network
                        需要加载 Stage 1 预训练的 iTransformer 和 Stage 2 预训练的 Uncertainty Estimator
                        最后进行测试评估
        - is_training=2: 端到端训练整个模型 (iTransformer + Uncertainty Estimator + Velocity Network)
        
        注意：Stage 2 (Uncertainty Estimator 预训练) 请使用 pretrain_uncertainty_estimator.py
        """
        # 记录当前 seed，便于在加载 Stage 2 权重等场景中复用
        self.current_seed = seed
        # 生成 setting 字符串（用于 checkpoints 路径）
        setting = self._get_setting(seed)
        
        # 模式 0: 只测试，不训练
        if self.is_training == 0:
            print('>>>>>>>testing (no training) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            # 先初始化 wandb（如果启用），以便记录测试结果
            if self._use_wandb() and not self._init_wandb(self.project, seed):
                return {}
            
            # 需要先 setup_run 以初始化必要的路径和配置
            self._setup_run(seed)
            
            # 从 checkpoints 加载模型
            self._load_checkpoint_model(setting)
            
            # 直接测试
            test_result = self._test()
            
            # 记录测试结果到 wandb
            if self._use_wandb():
                for k, v in test_result.items():
                    wandb.run.summary[f"test_{k}"] = v
                wandb.finish()
            
            return test_result
        
        # 模式 1: Stage 3 - 只训练 Velocity Network（加载 iTransformer 和 Uncertainty Estimator 权重）
        if self.is_training == 1:
            print('>>>>>>>Stage 3: training Velocity Network only (iTransformer + Uncertainty Estimator frozen) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            if self._use_wandb() and not self._init_wandb(self.project, seed): 
                return {}
            
            self._setup_run(seed)
            
            # 检查并恢复运行检查点（需要在模型初始化之后）
            if self._check_run_exist(seed):
                # 如果检查点存在，直接恢复（包含整个模型状态）
                self._resume_run(seed)
                # 恢复后需要重新冻结 iTransformer 和 uncertainty_estimator
                # （因为 load_state_dict 不会保持 requires_grad=False）
                self._freeze_itransformer()
                self._freeze_uncertainty_estimator()
            else:
                # 如果检查点不存在，加载预训练的权重
                # Stage 3 需要加载：
                # 1. Stage 1 预训练的 iTransformer
                # 2. Stage 2 预训练的 Uncertainty Estimator
                self._load_itransformer_only(setting)
                self._load_uncertainty_estimator(setting)
                # 加载权重后冻结它们（确保冻结的是预训练权重，而不是随机初始化）
                self._freeze_itransformer()
                self._freeze_uncertainty_estimator()

            self._run_print(f"run : nss{self.num_sampling_steps}_temp{self.temperature} in seed: {seed}")

            parameter_tables, model_parameters_num = count_parameters(self.model)
            # self._run_print(f"parameter_tables: {parameter_tables}")
            # self._run_print(f"model parameters: {model_parameters_num}")

            if self._use_wandb():
                wandb.run.summary["parameters"] = model_parameters_num

            # 训练循环
            while self.current_epoch < self.epochs:
                epoch_start_time = time.time()
                if self.early_stopping.early_stop is True:
                    self._run_print(
                        f"val CRPS no decreased for patience={self.patience} epochs,  early stopping ...."
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
                self._run_print(f"Training loss : {train_loss}")

                val_result = self._val()
                # test_result = self._test()

                self.current_epoch = self.current_epoch + 1
                
                # 使用CRPS作为早停指标
                self.early_stopping(val_result['crps'], self.model)
                
                # 学习率调度
                old_lr = self.model_optim.param_groups[0]['lr']
                self.scheduler.step(val_result['loss'])
                current_lr = self.model_optim.param_groups[0]['lr']
                
                # 记录学习率变化
                if old_lr != current_lr:
                    self._run_print(f"Learning rate updated: {old_lr:.2e} --> {current_lr:.2e}")
                else:
                    self._run_print(f"Learning rate: {current_lr:.2e}")

                self._save_run_check_point(seed)

                if self._use_wandb():
                    # 记录训练损失和详细指标
                    wandb.log({'training_loss': train_loss}, step=self.current_epoch)
                    for key, value in train_metrics.items():
                        wandb.log({f"train_{key}": value}, step=self.current_epoch)
                    wandb.log({f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                    # wandb.log({f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)
                    wandb.log({'learning_rate': current_lr}, step=self.current_epoch)

            self._load_best_model()
            best_test_result = self._test()
            if self._use_wandb():
                for k, v in best_test_result.items(): 
                    wandb.run.summary[f"best_test_{k}"] = v 
            
            if self._use_wandb():  
                wandb.finish()
            return best_test_result
        
        # 模式 2: 训练整个模型 (is_training=2 或默认)
        if self.is_training == 2:
            print('>>>>>>>training entire model (iTransformer + Velocity Network) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        else:
            # 兼容旧代码：如果 is_training 不是 0, 1, 2，默认当作 2 处理
            print('>>>>>>>training entire model (default mode) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            self.is_training = 2
        
        if self._use_wandb() and not self._init_wandb(self.project, seed): 
            return {}
        
        self._setup_run(seed)
        
        # 检查并恢复运行检查点（需要在模型初始化之后）
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : nss{self.num_sampling_steps}_temp{self.temperature} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        # self._run_print(f"parameter_tables: {parameter_tables}")
        # self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            wandb.run.summary["parameters"] = model_parameters_num

        # 训练循环
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopping.early_stop is True:
                self._run_print(
                    f"val CRPS no decreased for patience={self.patience} epochs,  early stopping ...."
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
            self._run_print(f"Training loss : {train_loss}")

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            
            # 使用CRPS作为早停指标
            self.early_stopping(val_result['crps'], self.model)
            
            # 学习率调度
            old_lr = self.model_optim.param_groups[0]['lr']
            self.scheduler.step(val_result['loss'])
            current_lr = self.model_optim.param_groups[0]['lr']
            
            # 记录学习率变化
            if old_lr != current_lr:
                self._run_print(f"Learning rate updated: {old_lr:.2e} --> {current_lr:.2e}")
            else:
                self._run_print(f"Learning rate: {current_lr:.2e}")

            self._save_run_check_point(seed)

            if self._use_wandb():
                # 记录训练损失和详细指标
                wandb.log({'training_loss': train_loss}, step=self.current_epoch)
                for key, value in train_metrics.items():
                    wandb.log({f"train_{key}": value}, step=self.current_epoch)
                wandb.log({f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                wandb.log({f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)
                wandb.log({'learning_rate': current_lr}, step=self.current_epoch)

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): 
                wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  
            wandb.finish()
        return best_test_result
    
    def train(self):
        """完整训练流程"""
        print("=" * 50)
        print("Starting iReflow Training")
        print("=" * 50)
        
        # 初始化数据加载器
        print("\nInitializing data loaders...")
        self._init_data_loader(fast_test=False, fast_val = False)
        
        # 初始化模型
        print("Initializing model...")
        self._init_model()
        
        # 初始化指标
        print("Initializing metrics...")
        self._init_metrics()
        
        # 设置早停和检查点路径（需要run_save_dir已设置）
        if hasattr(self, 'run_save_dir') and self.run_save_dir:
            self._setup_early_stopper()
        else:
            raise ValueError("run_save_dir must be set before training. Please call _setup_run(seed) first or use the run() method.")
        
        print("\nStarting training...")
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # 训练
            train_loss, train_metrics = self._train()
            print(f"Train Loss: {train_loss:.6f}")
            if train_metrics:
                print(f"Train Metrics: {train_metrics}")
            
            # 验证
            val_result = self._val()
            print(f"Val Loss: {val_result['loss']:.6f}")
            print(f"Val CRPS: {val_result['crps']:.6f}")
            
            # 学习率调度（使用验证损失）
            old_lr = self.model_optim.param_groups[0]['lr']
            self.scheduler.step(val_result['loss'])
            current_lr = self.model_optim.param_groups[0]['lr']
            
            # 记录学习率变化
            if old_lr != current_lr:
                print(f"Learning rate updated: {old_lr:.2e} --> {current_lr:.2e}")
            else:
                print(f"Learning rate: {current_lr:.2e}")
            
            # Early Stopping（使用CRPS作为早停指标，与run()方法保持一致）
            self.early_stopping(val_result['crps'], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # 加载最佳模型
        self._load_best_model()
        
        # 测试
        print("\n" + "=" * 50)
        print("Testing on best model")
        print("=" * 50)
        test_results = self._test()
        
        print("\nTest Results:")
        for k, v in test_results.items():
            print(f"{k}: {v:.6f}")
        
        return test_results


if __name__ == '__main__':
    setproctitle.setproctitle('iReflow_main')

    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(iReflowExp)

