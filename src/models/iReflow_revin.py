import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.iTransformer import Model as iTransformer
from src.nn.velocity_network_revin import VelocityNetwork
from src.utils.revin import RevIN


class iReflow(nn.Module):
    """
    iReflow (RevIN 版本)

    架构：
    1. 条件编码器（Conditioner）：
       - use_itransformer_enc=True ：iTransformer 编码器输出变量特征 H（含变量间自注意力）
       - use_itransformer_enc=False：RevIN 标准化历史序列倒置嵌入（轻量替代）
    2. Flow 生成器（Generator）：
       - 以 RevIN 统计量 (μ_X, σ_X) 定义源状态和相对坐标系
       - VelocityNetwork 学习从 N(μ_X, σ_X²) 到 y_gt 的速度场
    """

    def __init__(self, configs):
        super(iReflow, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # ── 是否使用 iTransformer 编码器作为条件特征 ─────────────────────────
        # True ：使用 iTransformer encoder 输出的变量 token [B, D, d_model]
        # False：使用 RevIN 标准化历史序列的倒置嵌入（更轻量，无跨变量注意力）
        self.use_itransformer_enc = getattr(configs, 'use_itransformer_enc', True)

        if self.use_itransformer_enc:
            # 路径 A：完整 iTransformer 编码器
            self.itransformer = iTransformer(configs)
        else:
            # 路径 B：轻量替代编码器 [B, D, seq_len] -> [B, D, d_model]
            self.history_embedding = nn.Linear(configs.seq_len, configs.d_model)

        # ── RevIN：统计历史序列的 μ_X, σ_X ──────────────────────────────────
        # affine=False：仅做统计（均值 / 标准差），无可学习参数
        self.revin = RevIN(num_features=getattr(configs, "enc_in", 1), affine=False)

        # ── Velocity Network ─────────────────────────────────────────────────
        self.velocity_net = VelocityNetwork(
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.flow_layers if hasattr(configs, 'flow_layers') else 3,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            use_relative_space=configs.use_relative_space
        )

        # ODE 求解步数（1 = one-step generation）
        self.num_sampling_steps = getattr(configs, 'num_sampling_steps', 1)
    
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

        # Stage 1a: RevIN 统计历史序列的 μ_X, σ_X（形状 [B, 1, D]）
        _ = self.revin(x_enc, mode="norm")
        mu_X = self.revin.mean     # [B, 1, D]
        sigma_X = self.revin.stdev # [B, 1, D]

        # Stage 1b: 获取条件编码器特征（Cross-Attention Key/Value）
        enc_features = self._get_enc_features(x_enc, x_mark_enc, mu_X, sigma_X)

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
    
    def _get_enc_features(self, x_enc, x_mark_enc, mu_X, sigma_X):
        """
        获取 Cross-Attention 条件特征，形状 [B, D, d_model]。

        Args:
            x_enc:    [B, L, D] 历史序列
            x_mark_enc: 时间标记
            mu_X:     [B, 1, D] RevIN 均值（已在外部计算好）
            sigma_X:  [B, 1, D] RevIN 标准差（已在外部计算好）

        路径 A（use_itransformer_enc=True）：
            iTransformer 编码器输出的变量 token（含变量间自注意力）。

        路径 B（use_itransformer_enc=False）：
            RevIN 标准化历史序列倒置嵌入（轻量替代）：
                z_enc = (x_enc - mu_X) / sigma_X   [B, L, D]
                → 转置 [B, D, L]
                → Linear(seq_len, d_model) → [B, D, d_model]
            保留变量级历史模式，通过 RevIN 消除均值/方差影响。

        Returns:
            enc_features: [B, D, d_model]
        """
        B, L, N = x_enc.shape

        if self.use_itransformer_enc:
            # ── 路径 A：完整 iTransformer 编码器 ──────────────────────────────
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

        else:
            # ── 路径 B：RevIN 倒置嵌入（轻量替代） ────────────────────────────
            # mu_X / sigma_X 由外部传入，无需在此重复计算 RevIN
            z_enc = (x_enc - mu_X) / sigma_X          # [B, L, D]
            return self.history_embedding(z_enc.permute(0, 2, 1))  # [B, D, d_model]

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

        # Stage 1a: RevIN 统计历史序列的 μ_X, σ_X（形状 [B, 1, D]）
        _ = self.revin(x_enc, mode="norm")
        mu_X = self.revin.mean     # [B, 1, D]
        sigma_X = self.revin.stdev # [B, 1, D]

        # Stage 1b: 获取条件编码器特征（Cross-Attention Key/Value）
        enc_features = self._get_enc_features(x_enc, x_mark_enc, mu_X, sigma_X)

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

