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
    # 模型配置
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2  # iTransformer encoder层数
    flow_layers: int = 3  # Velocity Network层数
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
    learning_rate: float = 0.0001
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
        
        # 初始化检查点路径（将在_init_model中更新为完整路径）
        self.checkpoint_path = None
    
    def _init_model(self):
        """初始化模型"""
        self.model = iReflow(self.model_configs).to(self.device)
        self.model_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optim, mode='min', factor=0.5, patience=5
        )
        
        # 设置检查点路径（使用基类的run_save_dir，如果已设置）
        if self.checkpoint_path is None:
            if hasattr(self, 'run_save_dir') and self.run_save_dir:
                self.checkpoint_path = os.path.join(self.run_save_dir, "best_model.pth")
            else:
                # 如果run_save_dir还未设置，使用临时路径（将在train中更新）
                os.makedirs("./checkpoints/iReflow/", exist_ok=True)
                self.checkpoint_path = "./checkpoints/iReflow/best_model.pth"
        
        # Early Stopping
        self.early_stopping = iReflowEarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )
        
        # 打印模型参数
        num_params = count_parameters(self.model)
        print(f"Model initialized with {num_params} parameters")
    
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
        
        return y_hat, batch_y, loss
    
    def _train(self):
        """训练一个epoch"""
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_losses = []
            
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
                pred, true, loss = self._process_train_batch(
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
                
                # 更新进度条
                progress_bar.set_postfix(
                    loss=loss.item(),
                    avg_loss=np.mean(train_losses)
                )
                progress_bar.update(batch_x.shape[0])
        
        avg_train_loss = np.mean(train_losses)
        return avg_train_loss
    
    def _val(self):
        """验证"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.val_loader):
                # 转换到设备
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                
                # 前向传播（训练模式以计算损失）
                loss, loss_dict, y_hat = self.model(
                    x_enc=batch_x,
                    x_mark_enc=batch_x_date_enc,
                    x_dec=None,
                    x_mark_dec=batch_y_date_enc,
                    y_gt=batch_y,
                    mode='train'
                )
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        return avg_val_loss
    
    def _test(self):
        """
        测试：生成多个样本并计算概率预测指标
        """
        self.model.eval()
        
        # 重置指标
        self.metrics.reset()
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(tqdm(self.test_loader, desc="Testing")):
                # 转换到设备
                batch_x = batch_x.to(self.device).float()
                origin_y = origin_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                
                # 生成多个样本
                samples, y_hat, sigma = self.model.forecast(
                    x_enc=batch_x,
                    x_mark_enc=batch_x_date_enc,
                    num_samples=self.num_samples,
                    temperature=self.temperature
                )
                
                # samples: [B, num_samples, P, D]
                # 反归一化
                if self.invtrans_loss:
                    B, N, P, D = samples.shape
                    samples_flat = samples.reshape(B * N, P, D)
                    samples_flat = self.scaler.inverse_transform(samples_flat)
                    samples = samples_flat.reshape(B, N, P, D)
                
                # 转换维度以匹配指标期望的格式
                # 从 [B, num_samples, P, D] 转换为 [B, P, D, num_samples]
                samples = samples.permute(0, 2, 3, 1)  # [B, P, D, num_samples]
                
                # 转移到CPU以计算指标
                preds = samples.cpu()  # [B, P, D, num_samples]
                truths = origin_y.cpu()  # [B, P, D]
                
                all_preds.append(preds)
                all_trues.append(truths)
        
        # 拼接所有批次
        all_preds = torch.cat(all_preds, dim=0)  # [N_total, P, D, num_samples]
        all_trues = torch.cat(all_trues, dim=0)  # [N_total, P, D]
        
        # 计算指标
        self.metrics.update(all_preds, all_trues)
        results = self.metrics.compute()
        
        # 转换为字典
        results_dict = {k: v.item() for k, v in results.items()}
        
        return results_dict
    
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
        
        # 确保检查点路径使用run_save_dir（如果已设置）
        if hasattr(self, 'run_save_dir') and self.run_save_dir:
            self.checkpoint_path = os.path.join(self.run_save_dir, "best_model.pth")
            # 更新early_stopping的路径
            self.early_stopping.path = self.checkpoint_path
        
        print("\nStarting training...")
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # 训练
            train_loss = self._train()
            print(f"Train Loss: {train_loss:.6f}")
            
            # 验证
            val_loss = self._val()
            print(f"Val Loss: {val_loss:.6f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # Early Stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        
        # 测试
        print("\n" + "=" * 50)
        print("Testing on best model")
        print("=" * 50)
        test_results = self._test()
        
        print("\nTest Results:")
        for k, v in test_results.items():
            print(f"{k}: {v:.6f}")
        
        return test_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='iReflow Experiment')
    
    # 数据集参数
    parser.add_argument('--dataset_type', type=str, default='ETTh1', help='数据集类型')
    parser.add_argument('--data_path', type=str, default='./data/', help='数据路径')
    parser.add_argument('--windows', type=int, default=168, help='历史窗口长度')
    parser.add_argument('--pred_len', type=int, default=192, help='预测长度')
    parser.add_argument('--horizon', type=int, default=1, help='预测步长')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--flow_layers', type=int, default=3, help='Flow层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='FFN维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--use_norm', type=bool, default=True, help='使用归一化')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    # Flow参数
    parser.add_argument('--num_sampling_steps', type=int, default=1, help='ODE求解步数')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--num_samples', type=int, default=100, help='测试样本数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=2021, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    reproducible(args.seed)
    
    # 创建实验
    exp = iReflowExp(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        windows=args.windows,
        pred_len=args.pred_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        flow_layers=args.flow_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_norm=args.use_norm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        num_sampling_steps=args.num_sampling_steps,
        temperature=args.temperature,
        num_samples=args.num_samples,
        device=args.device
    )
    
    # 训练和测试
    results = exp.train()
    
    return results


if __name__ == '__main__':
    main()

