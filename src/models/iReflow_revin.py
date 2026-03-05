import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.iTransformer import Model as iTransformer
from src.nn.velocity_network_revin import VelocityNetwork
from src.utils.revin import RevIN


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
        
        # Stage 1: iTransformer as Conditioner
        self.itransformer = iTransformer(configs)
        
        # 不确定性估计器（Aleatoric Uncertainty）
        # 改进：使用encoder特征和预测残差来估计sigma
        # 输入：encoder特征 [B, D, d_model]
        # 输出：sigma [B, D, P] -> [B, P, D]
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            # nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.1),
            nn.Linear(configs.d_model // 2, configs.pred_len),
            nn.Softplus()  # 确保sigma > 0
        )
        
        # 初始化：让sigma的初始值更合理（基于预测长度的经验值）
        # 使用较小的初始值，避免sigma过大导致训练不稳定
        for m in self.uncertainty_estimator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # 较小的gain，让sigma初始值较小
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # RevIN：用于统计历史序列的 μ_X, σ_X（仅做统计，不直接改动模型输入）
        # num_features 对于 affine=False 仅用于参数形状，这里使用 enc_in 保持语义一致
        self.revin = RevIN(num_features=getattr(configs, "enc_in", 1), affine=False)

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

        # Loss 权重（默认不改变现有行为）
        self.nll_loss_weight = getattr(configs, 'nll_loss_weight', 1.0)
        self.velocity_loss_weight = getattr(configs, 'velocity_loss_weight', 1.0)

        
    def get_encoder_features(self, x_enc, x_mark_enc):
        """
        获取iTransformer编码器的变量特征H
        
        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
        Returns:
            enc_features: [B, D, d_model] 变量特征（只包含原始变量，不含时间特征）
            y_hat: [B, P, D] 点预测
            sigma: [B, P, D] 预测不确定性
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
        
        # 如果使用了归一化，sigma也需要相应缩放
        # 改进：使用相对标准差，避免sigma过大
        if self.itransformer.use_norm:
            # 使用相对标准差（相对于均值），让sigma更合理
            # sigma = sigma * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            # 改进：使用较小的缩放因子，避免sigma过大
            relative_std = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            # 限制sigma的最大值，避免过大
            # 确保min和max都是Tensor，形状匹配
            min_sigma = torch.full_like(sigma, 1e-6)
            max_sigma = relative_std * 2.0
            sigma = torch.clamp(sigma * relative_std, min=min_sigma, max=max_sigma)
        else:
            # 即使没有归一化，也限制sigma的范围
            sigma = torch.clamp(sigma, min=1e-6, max=1.0)
        
        return enc_features, y_hat, sigma
    
    def compute_loss(self, x_enc, x_mark_enc, y_gt):
        """
        计算训练损失

        Flow Construction（全程使用 RevIN 统计量，不依赖 y_hat / sigma）:
        - mu_X, sigma_X = RevIN(x_enc)          # 历史序列的均值 / 标准差
        - X_0 = mu_X + epsilon * sigma_X        # Source State
        - X_1 = y_gt                            # Target State
        - X_tau = tau * X_1 + (1 - tau) * X_0  # Linear Interpolation
        - v_target = X_1 - X_0                 # Ground Truth Velocity

        Loss: E[||v_theta(X_tau, tau | H, mu_X, sigma_X) - v_target||^2]

        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            y_gt: [B, P, D] 真实未来序列
        Returns:
            velocity_loss: scalar
            loss_dict: 包含各项损失的字典
        """
        B, P, D = y_gt.shape
        device = y_gt.device

        # Stage 1: 获取 iTransformer 编码器特征（仅用于 Cross-Attention 条件）
        enc_features = self._get_enc_features(x_enc, x_mark_enc)

        # Stage 1b: RevIN 统计历史序列的 μ_X, σ_X（形状 [B, 1, D]）
        _ = self.revin(x_enc, mode="norm")
        mu_X = self.revin.mean     # [B, 1, D]
        sigma_X = self.revin.stdev # [B, 1, D]

        # Stage 2: 构建 Rectified Flow
        # 采样噪声 epsilon ~ N(0, I)
        epsilon = torch.randn_like(y_gt)

        # Source State: X_0 ~ N(mu_X, sigma_X^2)
        # mu_X / sigma_X 形状 [B, 1, D]，自动广播到 [B, P, D]
        X_0 = mu_X + epsilon * sigma_X

        # Target State: X_1 = y_gt
        X_1 = y_gt

        # 采样时间步 tau ~ U[0, 1]，Broadcasting 技巧
        tau = torch.rand(B, 1, 1, device=device)

        # Linear Interpolation: X_tau
        X_tau = tau * X_1 + (1 - tau) * X_0

        # Ground Truth Velocity: v_target = X_1 - X_0
        v_target = X_1 - X_0

        # Stage 3: 预测速度场（VelocityNetwork 使用 mu_X, sigma_X 进行相对坐标变换）
        v_pred = self.velocity_net(X_tau, tau.squeeze(), enc_features, mu_X, sigma_X)

        # Stage 4: 计算损失
        velocity_loss = F.mse_loss(v_pred, v_target)

        loss_dict = {
            'velocity_loss': velocity_loss.item(),
            'mean_sigma_X': sigma_X.mean().item(),
            'min_sigma_X': sigma_X.min().item(),
            'max_sigma_X': sigma_X.max().item(),
        }

        return velocity_loss, loss_dict
    
    def _get_enc_features(self, x_enc, x_mark_enc):
        """
        只获取 iTransformer 编码器输出的变量特征（不计算 y_hat / sigma）。
        用于 flow 阶段的 Cross-Attention 条件。

        Returns:
            enc_features: [B, D, d_model]
        """
        B, L, N = x_enc.shape
        if self.itransformer.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc_norm = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_norm = x_enc_norm / stdev
        else:
            x_enc_norm = x_enc
        enc_out = self.itransformer.enc_embedding(x_enc_norm, x_mark_enc)
        enc_features_full, _ = self.itransformer.encoder(enc_out, attn_mask=None)
        return enc_features_full[:, :N, :]  # [B, D, d_model]

    @torch.no_grad()
    def sample(self, x_enc, x_mark_enc, num_samples=1, temperature=1.0):
        """
        采样/推理过程（全程使用 RevIN 统计量，不依赖 y_hat / sigma）

        使用ODE求解器从X_0流向X_1

        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            num_samples: 生成的样本数
            temperature: 温度系数，控制采样多样性
        Returns:
            samples: [B, num_samples, P, D] 预测样本
            mu_X:    [B, 1, D] RevIN 历史均值（供外部评估使用）
            sigma_X: [B, 1, D] RevIN 历史标准差（供外部评估使用）
        """
        B, L, D = x_enc.shape
        device = x_enc.device

        # Stage 1: 获取编码器特征（仅 enc_features，不计算 y_hat / sigma）
        enc_features = self._get_enc_features(x_enc, x_mark_enc)

        # Stage 1b: RevIN 统计历史序列的 μ_X, σ_X（形状 [B, 1, D]）
        _ = self.revin(x_enc, mode="norm")
        mu_X = self.revin.mean     # [B, 1, D]
        sigma_X = self.revin.stdev # [B, 1, D]

        # Stage 2: 采样初始化
        samples = []

        for _ in range(num_samples):
            # 采样随机噪声
            epsilon = torch.randn(B, self.pred_len, D, device=device)

            # 初始状态: X_0 ~ N(mu_X, (temperature * sigma_X)^2)
            # mu_X / sigma_X 形状 [B, 1, D]，自动广播到 [B, P, D]
            X_tau = mu_X + temperature * epsilon * sigma_X

            # Stage 3: ODE求解
            if self.num_sampling_steps == 1:
                # One-step generation
                tau = torch.zeros(B, device=device)
                v = self.velocity_net(X_tau, tau, enc_features, mu_X, sigma_X)
                X_pred = X_tau + v
            else:
                # Multi-step Euler method
                dt = 1.0 / self.num_sampling_steps
                for i in range(self.num_sampling_steps):
                    tau_val = i * dt
                    tau = torch.ones(B, device=device) * tau_val
                    v = self.velocity_net(X_tau, tau, enc_features, mu_X, sigma_X)
                    X_tau = X_tau + v * dt
                X_pred = X_tau

            samples.append(X_pred)

        # [num_samples, B, P, D] -> [B, num_samples, P, D]
        samples = torch.stack(samples, dim=1)

        return samples, mu_X, sigma_X
    
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
            训练模式: (velocity_loss, loss_dict, mu_X)
                mu_X: [B, 1, D] RevIN 历史均值，作为点预测的替代返回给上层
            采样模式: (X_pred, mu_X, sigma_X)
                X_pred:  [B, P, D] 单次采样预测
                mu_X:    [B, 1, D] RevIN 历史均值
                sigma_X: [B, 1, D] RevIN 历史标准差
        """
        if mode == 'train':
            assert y_gt is not None, "y_gt is required in training mode"
            velocity_loss, loss_dict = self.compute_loss(x_enc, x_mark_enc, y_gt)
            # mu_X 已在 compute_loss 内部计算并存入 revin，直接复用
            mu_X = self.revin.mean  # [B, 1, D]
            return velocity_loss, loss_dict, mu_X
        else:
            # 采样模式
            samples, mu_X, sigma_X = self.sample(x_enc, x_mark_enc, num_samples=1)
            return samples[:, 0, :, :], mu_X, sigma_X
    
    def forecast(self, x_enc, x_mark_enc, num_samples=100, temperature=1.0):
        """
        概率预测接口

        Args:
            x_enc: [B, L, D] 历史序列
            x_mark_enc: [B, L, T] 时间标记
            num_samples: 生成的样本数
            temperature: 温度系数
        Returns:
            samples: [B, num_samples, P, D] 预测样本
            mu_X:    [B, 1, D] RevIN 历史均值（点预测替代）
            sigma_X: [B, 1, D] RevIN 历史标准差（不确定性替代）
        """
        return self.sample(x_enc, x_mark_enc, num_samples, temperature)

