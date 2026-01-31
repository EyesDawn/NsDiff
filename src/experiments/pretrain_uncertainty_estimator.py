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
        enc_features, y_hat, sigma = self.model.get_encoder_features(
            batch_x, batch_x_date_enc
        )
        
        # 只计算 NLL Loss（Gaussian Negative Log-Likelihood）
        # NLL = 0.5 * log(sigma^2) + 0.5 * (y_gt - y_hat)^2 / sigma^2
        var = sigma ** 2
        nll_loss = 0.5 * torch.log(var + 1e-6) + 0.5 * (batch_y - y_hat)**2 / (var + 1e-6)
        nll_loss = nll_loss.mean()
        
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
                enc_features, y_hat, sigma = self.model.get_encoder_features(
                    batch_x, batch_x_date_enc
                )
                
                # 计算 NLL Loss
                var = sigma ** 2
                nll_loss = 0.5 * torch.log(var + 1e-6) + 0.5 * (batch_y - y_hat)**2 / (var + 1e-6)
                nll_loss = nll_loss.mean()
                
                val_losses.append(nll_loss.item())
                val_metrics['nll_loss'].append(nll_loss.item())
                val_metrics['mean_sigma'].append(sigma.mean().item())
                val_metrics['mae_point'].append(F.l1_loss(y_hat, batch_y).item())
        
        # 调用基类的 _evaluate 方法获取概率预测指标
        from src.experiments.prob_forecast import ProbForecastExp
        result = ProbForecastExp._evaluate(self, self.val_loader)
        
        # 添加 NLL Loss 和 sigma 统计
        result['loss'] = np.mean(val_losses)
        result['nll_loss'] = np.mean(val_metrics['nll_loss'])
        result['mean_sigma'] = np.mean(val_metrics['mean_sigma'])
        result['mae_point'] = np.mean(val_metrics['mae_point'])
        
        # 清理标志
        delattr(self, '_num_samples_for_eval')
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


if __name__ == '__main__':
    setproctitle.setproctitle('UncertaintyEstimator_Pretrain')
    
    import fire
    fire.Fire(UncertaintyEstimatorPretrainExp)

