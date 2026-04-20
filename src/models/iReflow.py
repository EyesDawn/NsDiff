import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.iTransformer import Model as iTransformer
from src.nn.velocity_network import VelocityNetwork


class iReflow(nn.Module):
    """
    iReflow主模型
    
    包含两个阶段:
    1. Encoder Stage (Conditioner): iTransformer提供点预测、变量特征和置信度
    2. Flow Stage (Generator): 学习从预测分布到真实分布的速度场
    """
    
    def __init__(self, configs):
        super(iReflow, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.is_training = getattr(configs, 'is_training', 2)
        
        # Stage 1: iTransformer as Conditioner
        self.itransformer = iTransformer(configs)
        
        # 不确定性估计器（Aleatoric Uncertainty）
        # 直接输出 sigma（标准差）
        # 输入：encoder特征 [B, D, d_model]
        # 输出：sigma [B, D, P] -> [B, P, D]
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.LayerNorm(configs.d_model // 2),
            # nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.1),
            nn.Linear(configs.d_model // 2, configs.pred_len),
            nn.Softplus()
        )
        
        # 初始化：让sigma的初始值更合理（基于预测长度的经验值）
        # 使用较小的初始值，避免sigma过大导致训练不稳定
        init_sigma = max(float(getattr(configs, 'sigma_init', 0.1)), 1e-6)
        for m in self.uncertainty_estimator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # 较小的gain，让sigma初始值较小
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # 仅调整最后一层 bias，使 Softplus(bias) ≈ init_sigma
        last_linear = self.uncertainty_estimator[-2]
        if isinstance(last_linear, nn.Linear) and last_linear.bias is not None:
            sigma_bias = math.log(math.expm1(init_sigma))
            nn.init.constant_(last_linear.bias, sigma_bias)
        
        # Stage 2: Velocity Network as Generator
        self.velocity_net = VelocityNetwork(
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.flow_layers if hasattr(configs, 'flow_layers') else 3,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            use_relative_space=configs.use_relative_space
        )
        
        # 采样步数
        self.num_sampling_steps = configs.num_sampling_steps if hasattr(configs, 'num_sampling_steps') else 1
        self.x0_dist = getattr(configs, 'x0_dist', 'pred_gaussian')
        if self.x0_dist not in {'pred_gaussian', 'standard_normal'}:
            raise ValueError(
                f"Unsupported x0_dist: {self.x0_dist}. "
                "Expected one of {'pred_gaussian', 'standard_normal'}."
            )

        # Loss 权重（默认不改变现有行为）
        self.point_loss_weight = getattr(configs, 'point_loss_weight', 1.0)
        self.nll_loss_weight = getattr(configs, 'nll_loss_weight', 1.0)
        self.velocity_loss_weight = getattr(configs, 'velocity_loss_weight', 1.0)
        self.gaussian_nll_loss = nn.GaussianNLLLoss()

    def _build_x0(self, y_hat, sigma, epsilon, temperature=1.0):
        """根据配置构建 Source State X_0。"""
        if self.x0_dist == 'pred_gaussian':
            return y_hat + temperature * epsilon * sigma
        # standard_normal: X_0 ~ N(0, I)
        return temperature * epsilon

        
    def _get_encoder_outputs(self, x_enc, x_mark_enc):
        """
        获取编码器输出以及 uncertainty_estimator 的分布参数。
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
        Returns:
            enc_features: [B, D, d_model] 变量特征（只包含原始变量，不含时间特征）
            y_hat: [B, P, D] 点预测
            s: [B, P, D] 对数方差 log(sigma^2)（由 sigma 派生，兼容旧接口）
            var: [B, P, D] 方差
            sigma: [B, P, D] 标准差
        """
        # 记录原始变量数量（用于过滤协变量）
        B, L, N = x_enc.shape  # N是原始变量数
        
        # 获取embedding和encoder输出
        if self.itransformer.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc_norm = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_norm = x_enc_norm / stdev
            stdev = stdev.detach()
        else:
            x_enc_norm = x_enc
            means = None
            stdev = None
        
        # Embedding: [B, L, D] -> [B, D+T, d_model] (如果x_mark不为None，会拼接时间特征)
        enc_out = self.itransformer.enc_embedding(x_enc_norm, x_mark_enc)
        
        # Encoder: [B, D+T, d_model] -> [B, D+T, d_model]
        enc_features_full, _ = self.itransformer.encoder(enc_out, attn_mask=None)
        
        # 过滤协变量：只保留前N个变量（原始变量），去掉时间特征
        # [B, D+T, d_model] -> [B, D, d_model]
        enc_features = enc_features_full[:, :N, :]
        
        # 点预测: [B, D, d_model] -> [B, D, P] -> [B, P, D]
        # 注意：projector在enc_features_full上操作，然后过滤
        y_hat_full = self.itransformer.projector(enc_features_full).permute(0, 2, 1)  # [B, P, D+T]
        y_hat = y_hat_full[:, :, :N]  # [B, P, D] 过滤协变量
        
        # 反归一化
        if self.itransformer.use_norm:
            y_hat = y_hat * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y_hat = y_hat + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        # 估计不确定性: [B, D, d_model] -> [B, D, P] -> [B, P, D]
        sigma = self.uncertainty_estimator(enc_features).permute(0, 2, 1)

        # 如果使用了归一化，在 sigma 空间完成尺度变换
        if self.itransformer.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
            sigma = sigma * scale
            min_sigma = torch.full_like(sigma, 0.001)
            sigma = torch.clamp(sigma, min=min_sigma, max=2.0 * scale)
        else:
            sigma = torch.clamp(sigma, min=0.001)
        var = sigma.pow(2)
        s = torch.log(var)

        return enc_features, y_hat, s, var, sigma

    def get_encoder_features(self, x_enc, x_mark_enc):
        """
        获取iTransformer编码器的变量特征H
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
        Returns:
            enc_features: [B, D, d_model] 变量特征（只包含原始变量，不含时间特征）
            y_hat: [B, P, D] 点预测
            sigma: [B, P, D] 预测标准差
        """
        enc_features, y_hat, _, _, sigma = self._get_encoder_outputs(x_enc, x_mark_enc)
        return enc_features, y_hat, sigma
    
    def compute_loss(self, x_enc, x_mark_enc, y_gt):
        """
        计算训练损失
        
        Flow Construction:
        - X_0 = y_hat + epsilon * sigma (Source State)
        - X_1 = y_gt (Target State)
        - X_tau = tau * X_1 + (1 - tau) * X_0 (Linear Interpolation)
        - v_target = X_1 - X_0 (Ground Truth Velocity)
        
        Loss: E[||v_theta(X_tau, tau | H, sigma) - (X_1 - X_0)||^2]
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            y_gt: [B, P, D] 真实未来序列
        Returns:
            loss: scalar
            loss_dict: 包含各项损失的字典
        """
        B, P, D = y_gt.shape
        device = y_gt.device
        
        # Stage 1: 获取条件信息，计算负对数似然损失
        enc_features, y_hat, sigma = self.get_encoder_features(x_enc, x_mark_enc)

        point_loss = F.mse_loss(y_hat, y_gt)
        nll_loss = None
        if self.is_training == 2:
            nll_loss = self.gaussian_nll_loss(y_hat, y_gt, sigma.pow(2))

        # Stage 2: 构建Rectified Flow
        
        # 采样噪声 epsilon ~ N(0, I)
        epsilon = torch.randn_like(y_gt)
        
        # Source State: X_0 ~ N(y_hat, sigma^2)
        # Stage 3 只训练 velocity_net，因此保持 detach；
        # 端到端模式则移除 detach，使点预测、sigma 和速度场联合优化。
        if self.is_training == 2:
            y_hat_flow = y_hat
            sigma_flow = sigma
        else:
            y_hat_flow = y_hat.detach()
            sigma_flow = sigma.detach()
        X_0 = self._build_x0(y_hat_flow, sigma_flow, epsilon, temperature=1.0)
        
        # Target State: X_1 = y_gt
        X_1 = y_gt
        
        # 采样时间步 tau ~ U[0, 1]
        # 使用 Broadcasting 技巧避免 reshape
        tau = torch.rand(B, 1, 1, device=device)
        
        # Linear Interpolation: X_tau
        X_tau = tau * X_1 + (1 - tau) * X_0
        
        # Ground Truth Velocity: v_target = X_1 - X_0
        v_target = X_1 - X_0
        
        # Stage 3: 预测速度场
        v_pred = self.velocity_net(X_tau, tau.squeeze(), enc_features, y_hat_flow, sigma_flow)
        
        # Stage 4: 计算损失
        # MSE Loss on velocity
        velocity_loss = F.mse_loss(v_pred, v_target)
        
        total_loss = self.velocity_loss_weight * velocity_loss
        if self.is_training == 2:
            total_loss = total_loss + self.point_loss_weight * point_loss
            total_loss = total_loss + self.nll_loss_weight * nll_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'point_loss': point_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'mean_sigma': sigma.mean().item(),
            'min_sigma': sigma.min().item(),
            'max_sigma': sigma.max().item(),
            'mae_point': F.l1_loss(y_hat, y_gt).item(),
            'mse_point': point_loss.item(),
        }
        if nll_loss is not None:
            loss_dict['nll_loss'] = nll_loss.item()
        
        return total_loss, loss_dict
    
    @torch.no_grad()
    def sample(self, x_enc, x_mark_enc, num_samples=1, temperature=1.0, return_trajs=False):
        """
        采样/推理过程
        
        使用ODE求解器从X_0流向X_1
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            num_samples: 生成的样本数
            temperature: 温度系数，控制采样多样性
        Returns:
            samples: [B, num_samples, P, D] 预测样本
            y_hat: [B, P, D] 点预测
            sigma: [B, P, D] 不确定性
        """
        B, L, D = x_enc.shape
        device = x_enc.device
        
        # Stage 1: 获取条件信息
        enc_features, y_hat, sigma = self.get_encoder_features(x_enc, x_mark_enc)
        
        # Stage 2: 采样初始化
        samples = []
        x_samples = [] if return_trajs else None
        z_samples = [] if return_trajs else None
        
        for _ in range(num_samples):
            # 采样随机噪声
            epsilon = torch.randn(B, self.pred_len, D, device=device)
            
            # 初始状态 X_0
            X_tau = self._build_x0(y_hat, sigma, epsilon, temperature=temperature)

            if return_trajs:
                cur_x_traj = []
                cur_z_traj = []

            # Stage 3: ODE求解
            if self.num_sampling_steps == 1:
                # One-step generation (极快速)
                tau = torch.zeros(B, device=device)
                if return_trajs:
                    z_tau = (X_tau - y_hat) / sigma
                    cur_z_traj.append(z_tau)
                    cur_x_traj.append(X_tau)
                v = self.velocity_net(X_tau, tau, enc_features, y_hat, sigma)
                X_pred = X_tau + v
            else:
                # Multi-step Euler method
                # 改进：使用更稳定的ODE求解方法
                dt = 1.0 / self.num_sampling_steps
                for i in range(self.num_sampling_steps):
                    tau_val = i * dt
                    tau = torch.ones(B, device=device) * tau_val
                    v = self.velocity_net(X_tau, tau, enc_features, y_hat, sigma)

                    if return_trajs:
                        z_tau = (X_tau - y_hat) / sigma
                        cur_z_traj.append(z_tau)
                        cur_x_traj.append(X_tau)

                    # Euler step
                    X_tau = X_tau + v * dt
                X_pred = X_tau
            
            samples.append(X_pred)

            if return_trajs:
                cur_z_traj = torch.stack(cur_z_traj, dim=0) # [num_sampling_steps, B, P, D]
                cur_x_traj = torch.stack(cur_x_traj, dim=0) # [num_sampling_steps, B, P, D]
                z_samples.append(cur_z_traj)
                x_samples.append(cur_x_traj)
        
        # [num_samples, B, P, D] -> [B, num_samples, P, D]
        samples = torch.stack(samples, dim=1)

        if return_trajs:
            z_samples = torch.stack(z_samples, dim=2) # [num_sampling_steps, B, num_samples, P, D]
            x_samples = torch.stack(x_samples, dim=2) # [num_sampling_steps, B, num_samples, P, D]
        
        return samples, y_hat, sigma, z_samples, x_samples
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, y_gt=None, mode='train'):
        """
        前向传播
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            x_dec: [B, P, D] 解码器输入（未使用）
            x_mark_dec: [B, P, T] 解码器时间标记（未使用）
            y_gt: [B, P, D] 真实未来序列（仅训练时使用）
            mode: 'train' or 'sample'
        Returns:
            训练模式: (loss, loss_dict, y_hat)
            采样模式: (samples, y_hat, sigma)
        """
        if mode == 'train':
            assert y_gt is not None, "y_gt is required in training mode"
            loss, loss_dict = self.compute_loss(x_enc, x_mark_enc, y_gt)
            _, y_hat, _ = self.get_encoder_features(x_enc, x_mark_enc)
            return loss, loss_dict, y_hat
        else:
            # 采样模式
            samples, y_hat, sigma, _, _ = self.sample(x_enc, x_mark_enc, num_samples=1)
            return samples[:, 0, :, :], y_hat, sigma  # 返回第一个样本
    
    def forecast(self, x_enc, x_mark_enc, num_samples=100, temperature=1.0, return_trajs=False):
        """
        概率预测接口
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            num_samples: 生成的样本数
            temperature: 温度系数
        Returns:
            samples: [B, num_samples, P, D] 预测样本
            y_hat: [B, P, D] 点预测
            sigma: [B, P, D] 不确定性
        """
        return self.sample(x_enc, x_mark_enc, num_samples, temperature, return_trajs=return_trajs)
