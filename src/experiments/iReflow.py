"""
iReflow实验脚本
用于训练和评估iReflow模型
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
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
        """保存模型检查点
        
        注意：虽然参数名为 val_loss，但实际传入的是 CRPS 指标值
        """
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
    
    # 训练配置
    is_training: int = 1
    lr: float = 0.0001
    epochs: int = 100
    batch_size: int = 32
    num_worker: int = 1
    patience: int = 10
    lr_patience: int = 1  # 学习率调度器的patience
    point_loss_weight: float = 1.0
    nll_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    checkpoint_mode_for_test: Optional[int] = None
    
    # Flow配置
    num_sampling_steps: int = 1  # ODE求解步数，1表示one-step generation
    temperature: float = 1.0  # 采样温度
    num_samples: int = 100  # 测试时生成的样本数
    val_num_samples: int = 100  # 验证时用于估计 CRPS 的样本数
    x0_dist: str = 'pred_gaussian'  # X_0 分布: pred_gaussian | standard_normal
    
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
        self.model_configs.x0_dist = self.x0_dist
        # Loss 配置与梯度通路控制
        self.model_configs.is_training = self.is_training
        self.model_configs.point_loss_weight = self.point_loss_weight
        self.model_configs.nll_loss_weight = self.nll_loss_weight
        self.model_configs.velocity_loss_weight = self.velocity_loss_weight
        self.model_configs.use_relative_space = self.use_relative_space
    
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
            for param in self.model.itransformer.parameters():
                param.requires_grad = False
            print("iTransformer parameters frozen after loading pretrained weights")
        
        # 打印模型参数
        # num_params = count_parameters(self.model)
        # print(f"Model initialized with {num_params} parameters")
    
    def _freeze_uncertainty_estimator(self):
        """冻结 Uncertainty Estimator 参数（在加载预训练权重后调用）"""
        if self.is_training == 1:
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
                'point_loss': [],
                'velocity_loss': [],
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
                
                # 收集详细指标
                for key in train_metrics.keys():
                    if key in loss_dict:
                        train_metrics[key].append(loss_dict[key])
                
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
        samples, y_hat, sigma, _, _ = self.model.forecast(
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

    def _evaluate(self, dataloader, plot=False):
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
                for i, (batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc) in enumerate(dataloader):
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()
                    origin_y = origin_y.to(self.device).float()
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()

                    # start = time.time()
                    # print(batch_x.shape)
                    # print(batch_x_date_enc.shape)
                    # 生成采样预测 + 点预测与 sigma
                    samples, y_hat, sigma, z_samples, x_samples = self.model.forecast(
                        x_enc=batch_x,
                        x_mark_enc=batch_x_date_enc,
                        num_samples=num_samples,
                        temperature=self.temperature,
                    )  # samples: [B, S, P, D], y_hat/sigma: [B, P, D]
                    # end = time.time()
                    # print(end-start)
                    # assert 0

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

                    if plot:
                        voutput = preds.permute(0,3,1,2).detach().cpu().numpy()
                        true = truths.detach().cpu().numpy()
                        vx = batch_x.detach().cpu().numpy()[0, :, -1]

                        vtrue = np.concatenate((vx, true[0, :, -1]))
                        data = voutput[0,:,:,-1]
                        prob_visual(data, vtrue, name=os.path.join(os.path.join('./plot_results', self.dataset_type), str(i) + '.pdf'))

                        # sigma = sigma.detach().cpu().numpy()
                        # vsigma = sigma[0,:,-1]
                        # std_visual(vx, true[0, :, -1], vsigma, name=os.path.join(os.path.join('./plot_results', self.dataset_type), str(i) + '_std' + '.pdf'))

                        # z = z_samples[0,:,:,-1].detach().cpu().numpy()
                        # x = x_samples[0,:,:,-1].detach().cpu().numpy()
                        # zx_visual(z,x, name=os.path.join(os.path.join('./plot_results', self.dataset_type), str(i) + '_std' + '.pdf'))

                        max_plot_steps = min(5, z_samples.shape[0])
                        for j in range(max_plot_steps):
                            z = z_samples[j,0,:,:,-1].detach().cpu().numpy()
                            x = x_samples[j,0,:,:,-1].detach().cpu().numpy()
                            zx_visual(z,x, name=os.path.join(os.path.join('./plot_results', self.dataset_type), str(i) + '_std' + str(j) + '.pdf'))

                    progress_bar.update(batch_x.shape[0])

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        if sigma_counts > 0:
            result.update({f"sigma_{k}": float(v / sigma_counts) for k, v in sigma_sums.items()})
        return result
    
    def _val(self):
        """验证：使用固定的采样数来稳定 CRPS 估计。"""
        # 设置验证时使用的样本数
        self._num_samples_for_eval = min(self.num_samples, self.val_num_samples)
        
        # 计算验证损失（用于学习率调度）
        self.model.eval()
        val_losses = []
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
        
        # 调用基类的_val()方法获取概率预测指标
        result = super()._val()
        
        # 添加平均损失
        result['loss'] = np.mean(val_losses)
        
        # 清理标志
        delattr(self, '_num_samples_for_eval')
        return result
    
    # 不参与 run_save_dir hash 计算的字段：这些字段只影响运行行为，不影响模型结构/数据
    _run_irrelevant_fields = {
        'is_training',   # 训练/测试模式切换，is_training=0 需要能找到 is_training=1 训练出的模型
        'wandb_project', # 日志项目名，不影响实验结果
        'checkpoint_mode_for_test', # 仅影响测试时从哪个训练模式目录加载
    }

    @property
    def result_related_configs(self):
        """
        重写 result_related_configs 属性，确保所有值都可以被 JSON 序列化
        排除不可序列化的对象（如 argparse.Namespace, 模型对象等）
        同时排除 _run_irrelevant_fields 中的字段，使 run_save_dir hash 在不同运行模式下保持一致
        """
        from torch_timeseries.utils import asdict_exc
        from torch_timeseries.core.experiments.settings import BaseIrrelevant
        import json
        
        ident = asdict_exc(self, BaseIrrelevant)
        
        # 过滤掉不可序列化的对象
        serializable_ident = {}
        for k, v in ident.items():
            if k in self._run_irrelevant_fields:
                continue
            try:
                # 尝试序列化以检查是否可序列化
                json.dumps(v)
                serializable_ident[k] = v
            except (TypeError, ValueError):
                # 如果不可序列化，转换为字符串表示
                # 对于 argparse.Namespace 等对象，转换为字典
                if isinstance(v, (argparse.Namespace,)):
                    serializable_ident[k] = vars(v) if hasattr(v, '__dict__') else str(v)
                elif hasattr(v, '__class__'):
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
    
    # TODO: 修改 setting 格式，使其符合 iTransformer 的 setting 格式
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

    def _set_mode_specific_run_paths(self, mode: int):
        """将训练/测试产物重定向到 train_mode_{mode} 子目录。"""
        if mode not in (1, 2):
            raise ValueError(f"Unsupported mode for checkpoint paths: {mode}. Expected 1 or 2.")

        base_run_dir = getattr(self, "_base_run_save_dir", self.run_save_dir)
        self._base_run_save_dir = base_run_dir
        self.run_save_dir = os.path.join(base_run_dir, f"train_mode_{mode}")
        self.run_checkpoint_filepath = os.path.join(self.run_save_dir, "run_checkpoint.pth")
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "best_model.pth")

        if hasattr(self, "early_stopping") and hasattr(self.early_stopping, "path"):
            self.early_stopping.path = self.best_checkpoint_filepath

    def _resolve_test_checkpoint_mode(self) -> int:
        """解析 is_training=0 时应该加载哪个训练模式的最佳模型。"""
        requested_mode = self.checkpoint_mode_for_test
        base_run_dir = getattr(self, "_base_run_save_dir", self.run_save_dir)

        if requested_mode is not None:
            if requested_mode not in (1, 2):
                raise ValueError(
                    f"Invalid checkpoint_mode_for_test={requested_mode}. Expected 1 or 2."
                )
            requested_best_model = os.path.join(
                base_run_dir, f"train_mode_{requested_mode}", "best_model.pth"
            )
            if not os.path.exists(requested_best_model):
                raise FileNotFoundError(
                    f"Requested test checkpoint mode {requested_mode} not found at {requested_best_model}."
                )
            return requested_mode

        available_modes = []
        for mode in (1, 2):
            best_model_path = os.path.join(
                base_run_dir, f"train_mode_{mode}", "best_model.pth"
            )
            if os.path.exists(best_model_path):
                available_modes.append(mode)

        if len(available_modes) == 1:
            return available_modes[0]
        if len(available_modes) == 0:
            raise FileNotFoundError(
                "No trained iReflow checkpoint found for testing. "
                f"Searched under {os.path.join(base_run_dir, 'train_mode_1')} and "
                f"{os.path.join(base_run_dir, 'train_mode_2')}."
            )

        raise ValueError(
            "Multiple trained iReflow checkpoints found for testing. "
            "Please set checkpoint_mode_for_test=1 or checkpoint_mode_for_test=2 explicitly."
        )
    
    
    def _resume_run(self, seed):
        """恢复运行检查点（重写父类方法以支持调度器状态恢复）"""
        print(f"resuming from {self.run_checkpoint_filepath}")

        # torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device, weights_only=False)

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
            self._base_run_save_dir = self.run_save_dir
            resolved_mode = self._resolve_test_checkpoint_mode()
            self._set_mode_specific_run_paths(resolved_mode)
            
            # 加载已训练好的 iReflow 模型（best_model.pth 由训练阶段保存）
            if not os.path.exists(self.best_checkpoint_filepath):
                raise FileNotFoundError(
                    f"iReflow best model not found at {self.best_checkpoint_filepath}. "
                    f"Please ensure the model has been trained (is_training=1 or is_training=2) before testing."
                )
            print(f'Loading iReflow model from {self.best_checkpoint_filepath} (train_mode_{resolved_mode})')
            self._load_best_model()
            
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
            self._base_run_save_dir = self.run_save_dir
            self._set_mode_specific_run_paths(1)
            
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

            self._run_print(f"run : nss{self.num_sampling_steps}_temp{self.temperature}_invtrans{self.invtrans_loss} in seed: {seed}")

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

                self.current_epoch = self.current_epoch + 1
                
                # 使用CRPS作为早停指标
                self.early_stopping(val_result['crps'], self.model)
                
                # 学习率调度
                old_lr = self.model_optim.param_groups[0]['lr']
                self.scheduler.step(val_result['crps'])
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
                    wandb.log({'learning_rate': current_lr}, step=self.current_epoch)

            self._load_best_model()
            test_result = self._test()
            if self._use_wandb():
                for k, v in test_result.items(): 
                    wandb.run.summary[f"test_{k}"] = v 
            
            if self._use_wandb():  
                wandb.finish()
            return test_result
        
        # 模式 2: 训练整个模型 (is_training=2 或默认)
        if self.is_training == 2:
            print('>>>>>>>training entire model (iTransformer + Uncertainty Estimator + Velocity Network) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        else:
            # 兼容旧代码：如果 is_training 不是 0, 1, 2，默认当作 2 处理
            print('>>>>>>>training entire model (default mode) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            self.is_training = 2
            self.model_configs.is_training = 2
        
        if self._use_wandb() and not self._init_wandb(self.project, seed): 
            return {}
        
        self._setup_run(seed)
        self._base_run_save_dir = self.run_save_dir
        self._set_mode_specific_run_paths(2)
        
        # 检查并恢复运行检查点（需要在模型初始化之后）
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : nss{self.num_sampling_steps}_temp{self.temperature}_invtrans{self.invtrans_loss} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

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

            self.current_epoch = self.current_epoch + 1
            
            # 使用CRPS作为早停指标
            self.early_stopping(val_result['crps'], self.model)
            
            # 学习率调度
            old_lr = self.model_optim.param_groups[0]['lr']
            self.scheduler.step(val_result['crps'])
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
                wandb.log({'learning_rate': current_lr}, step=self.current_epoch)

        self._load_best_model()
        test_result = self._test()
        if self._use_wandb():
            for k, v in test_result.items(): 
                wandb.run.summary[f"test_{k}"] = v 
        
        if self._use_wandb():  
            wandb.finish()
        return test_result
    
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
            self.scheduler.step(val_result['crps'])
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


import matplotlib.pyplot as plt

def prob_visual(data, true, pred=None, random_int=None, name='./pic/test.pdf'):
    plt.figure(figsize=(12,7))
    plt.grid(True) 
    pred = np.mean(data, axis=0)
    plt.plot(np.arange(192)+96, pred, label='Mean Prediction', linewidth=1.5, color='darkblue')
    # plt.plot(data_mean, label='DiffMean', linewidth=1.5, color='green')
    plt.plot(true, label='GroundTruth', linewidth=2., color='#9D2121')    
    # plt.fill_between(np.arange(len(up))+96, down, up, color="green", alpha=0.2, label="Uncertainty Range")
    percentiles = np.percentile(data, q=[2.5, 25, 75, 97.5], axis=0)
    plt.fill_between(np.arange(192)+96, percentiles[0], percentiles[3], color="#1f77b4", alpha=0.2, label="95% range")
    plt.fill_between(np.arange(192)+96, percentiles[1], percentiles[2], color="#0F4A74", alpha=0.2, label="50% Range")
    # for i in range(len(random_int)):
    #     plt.axvline(x = random_int[i] + 96, color="grey", linestyle="--", linewidth=1)
    # plt.ylim(-2.2, 0.5) # ETTh1
    # plt.ylim(-1.8, 0.) # ETTm1
    # plt.ylim(-4, 7.0) #traffic
    # plt.ylim(-8.0, 7.5)
    
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def std_visual(batch_x, batch_y, pred_std, name='./pic/test.pdf'):
    plt.figure(figsize=(12,7))
    # plt.grid(True) 
    x_std = np.std(batch_x)                  # 标量
    x_std = np.full(192, x_std, dtype=float)
    _,_,y_std = DDN(batch_y, 25)
    plt.subplot(2,1,1)
    plt.plot(batch_y, label='GroundTruth', linewidth=2., color='#9D2121')    
    plt.subplot(2,1,2)
    # print(pred_std.shape)
    # print(x_std.shape)
    # print(y_std.shape)
    # assert 0
    plt.plot(pred_std, linewidth=1.5, color='darkblue', label='pred')
    plt.plot(x_std, linewidth=1.5, color='darkgreen', label='revin')
    plt.plot(y_std, linewidth=1.5, color='yellow', label='sliding')

    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def zx_visual(z, x, name='./pic/test.pdf'):
    """
    z: shape [N, T] 或 [N, D]
    x: shape [N, T] 或 [N, D]
    沿 axis=0 计算 mean/std，并画出 mean 及 mean±std
    """
    z_std = np.std(z, axis=0)
    z_mean = np.mean(z, axis=0)

    x_std = np.std(x, axis=0)
    x_mean = np.mean(x, axis=0)

    t_z = np.arange(len(z_mean))
    t_x = np.arange(len(x_mean))

    plt.figure(figsize=(10, 7))

    # ===== z =====
    plt.subplot(2, 1, 1)
    plt.plot(t_x, x_mean, label='x mean', linewidth=2)
    plt.plot(t_x, x_mean + x_std, label='x mean + std', linestyle='--', linewidth=1.5)
    plt.plot(t_x, x_mean - x_std, label='x mean - std', linestyle='--', linewidth=1.5)
    plt.fill_between(t_x, x_mean - x_std, x_mean + x_std, alpha=0.2)
    plt.title('X statistics')
    plt.legend()
    plt.grid(True)

    # ===== x =====
    plt.subplot(2, 1, 2)    
    plt.plot(t_z, z_mean, label='z mean', linewidth=2)
    plt.plot(t_z, z_mean + z_std, label='z mean + std', linestyle='--', linewidth=1.5)
    plt.plot(t_z, z_mean - z_std, label='z mean - std', linestyle='--', linewidth=1.5)
    plt.fill_between(t_z, z_mean - z_std, z_mean + z_std, alpha=0.2)
    plt.title('Z statistics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()




# import numpy as np
# import matplotlib.pyplot as plt

# def std_visual(batch_x, batch_y, pred_std, name='./pic/test.pdf'):
#     fig, axes = plt.subplots(2, 1, figsize=(12, 7))

#     # 计算输入序列整体标准差，并扩展到与 batch_y 同长度
#     x_std = np.std(batch_x)
#     x_std = np.full_like(batch_y, x_std, dtype=float)

#     y_std = DDN(batch_y, 7)

#     # 上图
#     axes[0].plot(batch_y, label='GroundTruth', linewidth=2.0, color='#9D2121')
#     axes[0].grid(True)
#     axes[0].legend()

#     # 下图
#     axes[1].plot(pred_std, label='Pred Std', linewidth=1.5, color='darkblue')
#     axes[1].plot(x_std, label='Input Std', linewidth=1.5, color='darkgreen')
#     axes[1].plot(y_std, label='Target Std', linewidth=1.5, color='yellow')
#     axes[1].grid(True)
#     axes[1].legend()

#     plt.tight_layout()
#     plt.savefig(name, bbox_inches='tight')
#     plt.close()

def DDN(data, kernel):
    x = torch.tensor(data)
    x_window = x.unfold(-1, kernel, 1)
    m, s = x_window.mean(dim=-1).numpy(), x_window.std(dim=-1).numpy()
    m, s = np.pad(m, (kernel//2,kernel//2), mode='edge'), np.pad(s, (kernel//2,kernel//2), mode='edge')
    data = (data - m) / (s + 1e-5)
    return data, m, s
    

if __name__ == '__main__':
    setproctitle.setproctitle('iReflow_main')

    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(iReflowExp)
