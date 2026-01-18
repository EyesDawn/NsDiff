"""
Variate-Aware Velocity Network for iReflow
实现了基于iTransformer架构的速度场网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.nn.iTransformer_SelfAttention_Family import FullAttention, AttentionLayer


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
        # 归一化
        x_norm = self.norm(x)
        
        # 时间调制
        scale = self.scale(time_emb).unsqueeze(1)  # [B, 1, d_model]
        shift = self.shift(time_emb).unsqueeze(1)  # [B, 1, d_model]
        
        return x_norm * (1 + scale) + shift


class VariateCrossAttentionLayer(nn.Module):
    """
    变量交叉注意力层
    实现Self-Attention + Cross-Attention + FFN的组合
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu"):
        super(VariateCrossAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Self-Attention: 处理当前噪声状态的变量关系
        self.self_attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        
        # Cross-Attention: 与iTransformer的变量特征对齐
        self.cross_attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        
        # Feed-Forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu
        
        # Adaptive LayerNorm for time injection
        self.ada_norm1 = AdaptiveLayerNorm(d_model)
        self.ada_norm2 = AdaptiveLayerNorm(d_model)
    
    def forward(self, x, enc_features, time_emb):
        """
        Args:
            x: [B, D, d_model] 当前流状态的变量tokens
            enc_features: [B, D, d_model] iTransformer编码器的变量特征
            time_emb: [B, d_model] 时间嵌入
        Returns:
            [B, D, d_model]
        """
        # Self-Attention with time-adaptive norm
        residual = x
        x = self.ada_norm1(x, time_emb)
        new_x, _ = self.self_attention(x, x, x, attn_mask=None)
        x = residual + self.dropout(new_x)
        
        # Cross-Attention
        residual = x
        x = self.norm1(x)
        new_x, _ = self.cross_attention(x, enc_features, enc_features, attn_mask=None)
        x = residual + self.dropout(new_x)
        
        # Feed-Forward with time-adaptive norm
        residual = x
        y = self.ada_norm2(x, time_emb)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = residual + y
        
        return self.norm3(x)


class ConfidenceGating(nn.Module):
    """
    置信度门控层
    根据iTransformer的预测不确定性调节速度场的幅度
    
    改进：使用更合理的门控机制
    - 高不确定性（大sigma）→ 大门控系数 → 允许更多修正
    - 低不确定性（小sigma）→ 小门控系数 → 减少修正
    - 但避免完全抑制修正（使用偏移量）
    """
    def __init__(self, pred_len, d_model):
        super(ConfidenceGating, self).__init__()
        # 将sigma映射到gate系数
        self.sigma_proj = nn.Sequential(
            nn.Linear(pred_len, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        # bias 初始值为0，使得初始门控系数为0.5
        self.gate_bias = nn.Parameter(torch.zeros(1, 1, d_model))  # 最小门控系数
    
    def forward(self, x, sigma):
        """
        Args:
            x: [B, D, d_model] 变量tokens
            sigma: [B, P, D] 预测不确定性
        Returns:
            [B, D, d_model]
        """
        # sigma: [B, P, D] -> [B, D, P]
        sigma = sigma.permute(0, 2, 1)
        
        # 使用 log 变换处理 sigma 的量级差异
        sigma_log = torch.log(sigma + 1e-6)
        
        # 计算门控系数
        gate = self.sigma_proj(sigma_log)  # [B, D, d_model]
        
        # 改进：添加偏移量，确保最小门控系数
        gate = gate + self.gate_bias.to(x.device)

        # 限制在[0.01, 1.0]范围内，避免完全抑制修正
        gate = torch.clamp(gate, min=0.01, max=1.0)
        
        return x * gate


class VelocityNetwork(nn.Module):
    """
    速度场网络 v_θ
    输入：噪声状态X_τ、时间τ、条件(H, y_hat, sigma)
    输出：速度场 v
    """
    def __init__(self, pred_len, d_model=512, n_heads=8, e_layers=3, d_ff=2048, dropout=0.1):
        super(VelocityNetwork, self).__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Step 1: Inverted Embedding - 将时间序列嵌入为变量tokens
        # X_τ: [B, P, D] -> [B, D, P] -> [B, D, d_model]
        self.value_embedding = nn.Linear(pred_len, d_model)
        self.dropout_emb = nn.Dropout(dropout)
        
        # Step 2: Time Injection
        self.time_embedding = TimestepEmbedding(d_model)
        
        # Step 3: Variate-Cross-Attention Layers
        self.layers = nn.ModuleList([
            VariateCrossAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        
        # Step 4: Confidence Gating
        self.confidence_gate = ConfidenceGating(pred_len, d_model)
        
        # Step 5: Final Projection
        # [B, D, d_model] -> [B, D, P] -> [B, P, D]
        self.projector = nn.Linear(d_model, pred_len)
        
        # Layer Norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x_tau, tau, enc_features, y_hat, sigma):
        """
        Args:
            x_tau: [B, P, D] 当前流状态
            tau: [B] or [B, 1] flow time ∈ [0, 1]
            enc_features: [B, D, d_model] iTransformer编码器输出的变量特征H
            y_hat: [B, P, D] iTransformer的点预测
            sigma: [B, P, D] 预测不确定性
        Returns:
            v: [B, P, D] 速度场
        """
        
        # Step 1: Inverted Embedding
        # [B, P, D] -> [B, D, P]
        x = x_tau.permute(0, 2, 1)
        y_anchor = y_hat.permute(0, 2, 1)
        
        # [B, D, P] -> [B, D, d_model]
        x_emb = self.value_embedding(x)
        y_emb = self.value_embedding(y_anchor)

        # 显式注入 y_hat 信息，帮助模型理解预测的相对位置
        x = x_emb + y_emb
        x = self.dropout_emb(x)
        
        # Step 2: Time Embedding
        time_emb = self.time_embedding(tau)  # [B, d_model]
        
        # Step 3: Variate-Cross-Attention with time injection
        for layer in self.layers:
            x = layer(x, enc_features, time_emb)
        
        x = self.norm(x)
        
        # Step 4: Confidence Gating
        x = self.confidence_gate(x, sigma)
        
        # Step 5: Final Projection
        # [B, D, d_model] -> [B, D, P]
        v = self.projector(x)
        # [B, D, P] -> [B, P, D]
        v = v.permute(0, 2, 1)
        
        return v

