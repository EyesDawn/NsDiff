"""
iReflow实验脚本
用于训练和评估iReflow模型
"""
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
import argparse
from src.models.iReflow import iReflow
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.early_stop import EarlyStopping
import yaml
import numpy as np
import setproctitle
try:
    import wandb
except:
    print("Warning: wandb is not installed, some functionality may not work.")


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
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
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
    
    # 训练配置
    is_training: int = 1
    lr: float = 0.0001
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    
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
    
    def _init_model(self):
        """初始化模型"""
        self.model = iReflow(self.model_configs).to(self.device)
        
        # 根据 is_training 参数决定训练哪些部分
        # 注意：对于 is_training=1，冻结操作应该在加载预训练权重之后进行
        # 因此这里先不冻结，冻结操作将在 _freeze_itransformer() 中进行
        if self.is_training == 1:
            # 只优化 velocity_net 和 uncertainty_estimator 的参数
            # iTransformer 的冻结将在加载权重后进行
            trainable_params = list(self.model.velocity_net.parameters()) + \
                             list(self.model.uncertainty_estimator.parameters())
            self.model_optim = torch.optim.Adam(
                trainable_params, lr=self.lr
            )
            print("Initialized model: will freeze iTransformer after loading weights, only training Velocity Network and Uncertainty Estimator")
        else:
            # 训练整个模型（is_training=2 或默认情况）
            self.model_optim = torch.optim.Adam(
                self.model.parameters(), lr=self.lr
            )
            if self.is_training == 2:
                print("Initialized model: training entire model (iTransformer + Velocity Network)")
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optim, mode='min', factor=0.5, patience=2
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
                'mae_point': []
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
    
    def _val(self):
        """验证：使用较少的样本数以加快验证速度"""
        # 设置验证时使用的样本数
        self._num_samples_for_eval = min(self.num_samples, 20)
        
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
    
    def _load_itransformer_only(self, setting):
        """
        只加载 iTransformer 的权重（用于 is_training=1 模式）
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
        checkpoint_keys = list(checkpoint.keys())
        print(f'Checkpoint keys (all {len(checkpoint_keys)} keys): {checkpoint_keys}')
        
        # 检查是否是 iTransformer checkpoint（键名包含 enc_embedding, encoder, projector）
        # 还是 iReflow checkpoint（包含 itransformer, velocity_net 等）
        checkpoint_keys_str = ' '.join(str(k) for k in checkpoint.keys())
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
        - is_training=1: 只训练Velocity Network，加载iTransformer模型的权重，最后进行测试评估
        - is_training=2: 训练整个模型(iTransformer+VelocityNetwork)
        """
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
            
            # 初始化数据加载器
            self._init_data_loader()
            
            # 初始化模型
            self._init_model()
            
            # 初始化指标
            self._init_metrics()
            
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
        
        # 模式 1: 只训练 Velocity Network（加载 iTransformer 权重）
        if self.is_training == 1:
            print('>>>>>>>training Velocity Network only (iTransformer frozen) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            if self._use_wandb() and not self._init_wandb(self.project, seed): 
                return {}
            
            self._setup_run(seed)
            
            # 初始化数据加载器
            self._init_data_loader()
            
            # 初始化模型（先不冻结，等加载权重后再冻结）
            self._init_model()
            
            # 加载 iTransformer 权重
            self._load_itransformer_only(setting)
            
            # 加载权重后再冻结 iTransformer（确保冻结的是预训练权重，而不是随机初始化）
            self._freeze_itransformer()
            
            # 初始化指标
            self._init_metrics()
            
            # 设置早停和检查点路径
            self._setup_early_stopper()
            
            # 检查并恢复运行检查点（需要在模型初始化之后）
            if self._check_run_exist(seed):
                self._resume_run(seed)

            self._run_print(f"run : nss{self.num_sampling_steps}_temp{self.temperature} in seed: {seed}")

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
                        f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
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
                self.scheduler.step(val_result['loss'])

                self._save_run_check_point(seed)

                if self._use_wandb():
                    # 记录训练损失和详细指标
                    wandb.log({'training_loss': train_loss}, step=self.current_epoch)
                    for key, value in train_metrics.items():
                        wandb.log({f"train_{key}": value}, step=self.current_epoch)
                    wandb.log({f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                    wandb.log({f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

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
        
        # 初始化数据加载器
        self._init_data_loader()
        
        # 初始化模型（训练整个模型）
        self._init_model()
        
        # 初始化指标
        self._init_metrics()
        
        # 设置早停和检查点路径
        self._setup_early_stopper()
        
        # 检查并恢复运行检查点（需要在模型初始化之后）
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : nss{self.num_sampling_steps}_temp{self.temperature} in seed: {seed}")

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
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
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
            self.scheduler.step(val_result['loss'])

            self._save_run_check_point(seed)

            if self._use_wandb():
                # 记录训练损失和详细指标
                wandb.log({'training_loss': train_loss}, step=self.current_epoch)
                for key, value in train_metrics.items():
                    wandb.log({f"train_{key}": value}, step=self.current_epoch)
                wandb.log({f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                wandb.log({f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

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
        self._init_data_loader()
        
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
            self.scheduler.step(val_result['loss'])
            
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

