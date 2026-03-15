"""
Variate-Aware Velocity Network (MLP variant) for iReflow
velocity_network.py 的通道独立MLP变种：
将 VariateCrossAttentionLayer (Self-Attn + Cross-Attn + FFN) 替换为
通道独立的 VariateMLPLayer，每个变量仅依赖自身特征，不进行跨变量交互。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """
    时间步嵌入层，将flow time τ映射到高维空间
    使用正弦位置编码
    """
    def __init__(self, d_model, max_period=10000):
        super(TimestepEmbedding, self).__init__()
        self.d_model = d_model
        self.max_period = max_period

        # MLP投影
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] or [B, 1], flow time τ ∈ [0, 1]
        Returns:
            [B, d_model]
        """
        if len(timesteps.shape) == 1:
            timesteps = timesteps.unsqueeze(-1)  # [B, 1]

        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )

        args = timesteps * freqs[None]  # [B, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d_model]

        if self.d_model % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return self.mlp(embedding)


class AdaptiveLayerNorm(nn.Module):
    """
    自适应层归一化，用于将时间信息注入到特征中
    """
    def __init__(self, d_model):
        super(AdaptiveLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale = nn.Linear(d_model, d_model)
        self.shift = nn.Linear(d_model, d_model)

    def forward(self, x, time_emb):
        """
        Args:
            x: [B, D, d_model] 变量tokens
            time_emb: [B, d_model] 时间嵌入
        Returns:
            [B, D, d_model]
        """
        x_norm = self.norm(x)

        scale = self.scale(time_emb).unsqueeze(1)  # [B, 1, d_model]
        shift = self.shift(time_emb).unsqueeze(1)  # [B, 1, d_model]

        return x_norm * (1 + scale) + shift


class VariateMLPLayer(nn.Module):
    """
    通道独立MLP层，替代 VariateCrossAttentionLayer。
    每个变量仅依赖自身特征，不进行任何跨变量交互。

    对应关系：
      Self-Attention  → 时间调制逐变量线性变换（ada_norm1 + self_linear）
      Cross-Attention → 逐变量门控编码器特征融合（norm1 + enc_proj + gate_proj）
      FFN             → 两层逐变量MLP（ada_norm2 + ff1 + ff2）

    "通道独立"保证：所有 Linear / LayerNorm 均作用于最后一维 d_model，
    D 个变量共享权重但彼此之间无信息交换。
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(VariateMLPLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # ---------- 替代 Self-Attention ----------
        # 逐变量线性变换：每个变量独立地对自身 d_model 维特征做映射
        self.self_linear = nn.Linear(d_model, d_model)

        # ---------- 替代 Cross-Attention ----------
        # 逐变量门控融合：variable i 只融合对应的 enc_features[i]，不跨变量
        #   enc_proj : 对编码器特征做线性投影
        #   gate_proj: 根据 [x_norm; enc_out] 计算融合权重
        self.enc_proj  = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # ---------- FFN (channel-independent) ----------
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # ---------- Layer Norms ----------
        self.norm1 = nn.LayerNorm(d_model)   # Cross 步骤前归一化
        self.norm3 = nn.LayerNorm(d_model)   # 输出归一化（对应原 norm3）

        self.dropout   = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

        # ---------- Adaptive LayerNorm for time injection ----------
        self.ada_norm1 = AdaptiveLayerNorm(d_model)   # Self-MLP 前
        self.ada_norm2 = AdaptiveLayerNorm(d_model)   # FFN 前

    def forward(self, x, enc_features, time_emb):
        """
        Args:
            x:            [B, D, d_model] 当前流状态的变量 tokens
            enc_features: [B, D, d_model] iTransformer 编码器的变量特征
            time_emb:     [B, d_model]    时间嵌入
        Returns:
            [B, D, d_model]
        """
        # ── Step 1: 替代 Self-Attention ──────────────────────────────────
        # 时间调制 → 逐变量线性变换 → 残差
        residual = x
        x = self.ada_norm1(x, time_emb)               # [B, D, d_model]
        x = residual + self.dropout(self.self_linear(x))

        # ── Step 2: 替代 Cross-Attention ─────────────────────────────────
        # 逐变量门控编码器特征融合，variable i 仅使用 enc_features[:, i, :]
        residual = x
        x_norm  = self.norm1(x)                        # [B, D, d_model]
        enc_out = self.enc_proj(enc_features)           # [B, D, d_model]
        gate    = torch.sigmoid(
            self.gate_proj(torch.cat([x_norm, enc_out], dim=-1))
        )                                               # [B, D, d_model]
        x = residual + self.dropout(gate * enc_out)

        # ── Step 3: FFN + 时间调制 ────────────────────────────────────────
        residual = x
        y = self.ada_norm2(x, time_emb)                # [B, D, d_model]
        y = self.dropout(self.activation(self.ff1(y)))  # [B, D, d_ff]
        y = self.dropout(self.ff2(y))                   # [B, D, d_model]
        x = residual + y

        return self.norm3(x)


class VelocityNetworkMLP(nn.Module):
    """
    速度场网络 v_θ（MLP通道独立变种）
    将 VelocityNetwork 中的 VariateCrossAttentionLayer 替换为 VariateMLPLayer。
    输入：噪声状态 X_τ、时间 τ、条件 (H, y_hat, sigma)
    输出：速度场 v
    """
    def __init__(self, pred_len, d_model=512, n_heads=None, e_layers=3, d_ff=2048,
                 dropout=0.1, use_relative_space=True):
        super(VelocityNetworkMLP, self).__init__()
        self.pred_len = pred_len
        self.d_model  = d_model
        self.use_relative_space = use_relative_space

        # Step 1: Inverted Embedding - 将时间序列嵌入为变量 tokens
        # X_τ: [B, P, D] -> [B, D, P] -> [B, D, d_model]
        self.value_embedding = nn.Linear(pred_len, d_model)
        self.dropout_emb     = nn.Dropout(dropout)

        # Step 2: Time Injection
        self.time_embedding = TimestepEmbedding(d_model)

        # Step 3: 通道独立 MLP Layers（替代 VariateCrossAttentionLayer）
        self.layers = nn.ModuleList([
            VariateMLPLayer(d_model, d_ff, dropout)
            for _ in range(e_layers)
        ])

        # Step 4: Final Projection
        # [B, D, d_model] -> [B, D, P] -> [B, P, D]
        self.projector = nn.Linear(d_model, pred_len)

        # Layer Norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_tau, tau, enc_features, y_hat, sigma):
        """
        Args:
            x_tau:        [B, P, D] 当前流状态
            tau:          [B] or [B, 1] flow time ∈ [0, 1]
            enc_features: [B, D, d_model] iTransformer 编码器输出的变量特征 H
            y_hat:        [B, P, D] iTransformer 的点预测
            sigma:        [B, P, D] 预测不确定性
        Returns:
            v: [B, P, D] 速度场
        """
        # Step 0: 坐标变换到相对空间（可选，通过配置控制）
        if self.use_relative_space:
            z_tau = (x_tau - y_hat) / sigma
        else:
            z_tau = x_tau

        # Step 1: Inverted Embedding
        # [B, P, D] -> [B, D, P] -> [B, D, d_model]
        x = z_tau.permute(0, 2, 1)
        x = self.value_embedding(x)
        x = self.dropout_emb(x)

        # Step 2: Time Embedding
        time_emb = self.time_embedding(tau)  # [B, d_model]

        # Step 3: 通道独立 MLP Layers with time injection
        for layer in self.layers:
            x = layer(x, enc_features, time_emb)

        x = self.norm(x)

        # Step 4: Final Projection
        # [B, D, d_model] -> [B, D, P]
        v = self.projector(x)
        # [B, D, P] -> [B, P, D]
        u = v.permute(0, 2, 1)

        # Step 5: 逆变换回绝对空间
        if self.use_relative_space:
            v = sigma * u
        else:
            v = u

        return v

