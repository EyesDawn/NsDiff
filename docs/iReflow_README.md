# iReflow: Rectified Flow with Inverted Variate-Awareness

## 概述

iReflow是一个创新的概率时间序列预测框架，结合了以下两大核心技术：

1. **iTransformer**: 作为Conditioner，提供点预测、变量特征和不确定性估计
2. **Rectified Flow**: 作为Generator，学习从预测分布到真实分布的速度场

与传统扩散模型不同，iReflow使用**Residual-Centric Flow**，起点是有噪声的点预测而非标准高斯分布，这使得流的传输更加高效。

## 架构设计

### 整体流程

```
输入: X_hist [B, L, D]
   ↓
┌─────────────────────────────────────────────┐
│  Stage 1: iTransformer (Conditioner)        │
│  ├─ Inverted Embedding                      │
│  ├─ Encoder (变量级Transformer)             │
│  └─ Output:                                 │
│      • y_hat: 点预测 [B, P, D]              │
│      • H: 变量特征 [B, D, d_model]          │
│      • σ: 不确定性 [B, P, D]                │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│  Stage 2: Rectified Flow                   │
│  ┌───────────────────────────────────────┐ │
│  │ 训练时:                                │ │
│  │ X_0 = y_hat + ε·σ  (Source)          │ │
│  │ X_1 = y_gt         (Target)          │ │
│  │ X_τ = τ·X_1 + (1-τ)·X_0              │ │
│  │ v* = X_1 - X_0     (True Velocity)   │ │
│  │ Loss = ||v_θ - v*||²                 │ │
│  └───────────────────────────────────────┘ │
│  ┌───────────────────────────────────────┐ │
│  │ 推理时:                                │ │
│  │ X_0 = y_hat + ε·σ·T                  │ │
│  │ v = v_θ(X_0, τ=0 | H, σ)             │ │
│  │ X_pred = X_0 + v  (One-step!)        │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
   ↓
输出: 预测样本 [B, num_samples, P, D]
```

### Velocity Network架构

```
输入: X_τ [B, P, D], τ, H [B, D, d_model], y_hat, σ
   ↓
┌────────────────────────────────────────────┐
│ Step 1: Inverted Embedding                │
│ [B, P, D] → [B, D, P] → [B, D, d_model]  │
└────────────────────────────────────────────┘
   ↓
┌────────────────────────────────────────────┐
│ Step 2: Time Injection                    │
│ τ → Sinusoidal Embedding → [B, d_model]  │
└────────────────────────────────────────────┘
   ↓
┌────────────────────────────────────────────┐
│ Step 3: Variate-Cross-Attention (×L)      │
│ ┌──────────────────────────────────────┐  │
│ │ • AdaLN(time)                        │  │
│ │ • Self-Attention(变量内关系)          │  │
│ │ • Cross-Attention(与H对齐)           │  │
│ │ • Feed-Forward                       │  │
│ └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
   ↓
┌────────────────────────────────────────────┐
│ Step 4: Confidence Gating                 │
│ gate = Sigmoid(MLP(σ))                    │
│ x = x ⊙ gate                              │
└────────────────────────────────────────────┘
   ↓
┌────────────────────────────────────────────┐
│ Step 5: Projection                        │
│ [B, D, d_model] → [B, D, P] → [B, P, D]  │
└────────────────────────────────────────────┘
   ↓
输出: v [B, P, D]
```

## 文件结构

```
NsDiff/
├── src/
│   ├── models/
│   │   ├── iTransformer.py          # iTransformer基础模型
│   │   └── iReflow.py                # iReflow主模型 ⭐
│   ├── nn/
│   │   ├── velocity_network.py       # 速度场网络 ⭐
│   │   ├── iTransformer_EncDec.py   # Encoder/Decoder
│   │   ├── iTransformer_Embed.py    # 嵌入层
│   │   └── iTransformer_SelfAttention_Family.py  # 注意力机制
│   └── experiments/
│       └── iReflow.py                # 实验脚本 ⭐
├── configs/
│   └── ireflow.yaml                  # 配置文件 ⭐
├── scripts/
│   └── iReflow/
│       └── ETTh1.sh                  # 运行脚本 ⭐
└── docs/
    └── iReflow_README.md             # 本文档 ⭐
```

## 核心组件

### 1. VelocityNetwork (`src/nn/velocity_network.py`)

速度场网络，预测流的速度向量。

**关键模块：**
- `TimestepEmbedding`: 时间步嵌入
- `AdaptiveLayerNorm`: 自适应层归一化（注入时间信息）
- `VariateCrossAttentionLayer`: 变量交叉注意力层

### 2. iReflow模型 (`src/models/iReflow.py`)

主模型，整合iTransformer和Velocity Network。

**关键方法：**
- `get_encoder_features()`: 提取变量特征H和不确定性σ
- `compute_loss()`: 计算训练损失
- `sample()`: ODE求解，生成预测样本
- `forecast()`: 概率预测接口

### 3. 实验脚本 (`src/experiments/iReflow.py`)

训练和评估框架。

**关键功能：**
- 数据加载和预处理
- 训练循环
- 验证和早停
- 概率预测指标计算（CRPS, PICP等）

## 使用方法

### 1. 基本训练

```bash
# 运行ETTh1数据集
bash ./scripts/iReflow/ETTh1.sh
```

### 2. 自定义训练

```bash
export PYTHONPATH=./

python3 ./src/experiments/iReflow.py \
    --dataset_type="ETTh1" \
    --windows=168 \
    --pred_len=192 \
    --d_model=512 \
    --n_heads=8 \
    --e_layers=2 \
    --flow_layers=3 \
    --batch_size=32 \
    --learning_rate=0.0001 \
    --epochs=100 \
    --num_sampling_steps=1 \
    --temperature=1.0 \
    --num_samples=100 \
    --device="cuda:0"
```

### 3. 使用配置文件

```python
import yaml
from src.experiments.iReflow import iReflowExp

# 加载配置
with open('configs/ireflow.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建实验
exp = iReflowExp(**config)

# 训练
results = exp.train()
```

### 4. 推理示例

```python
import torch
from src.models.iReflow import iReflow

# 加载模型
model = iReflow(configs).to('cuda')
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# 预测
with torch.no_grad():
    samples, y_hat, sigma = model.forecast(
        x_enc=x_hist,              # [B, L, D]
        x_mark_enc=x_mark,         # [B, L, T]
        num_samples=100,           # 生成100个样本
        temperature=1.0            # 温度系数
    )
    
# samples: [B, 100, P, D]
# 可以计算均值、分位数等统计量
mean_pred = samples.mean(dim=1)        # [B, P, D]
quantile_90 = samples.quantile(0.9, dim=1)  # [B, P, D]
```

## 关键参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 512 | 模型维度 |
| `n_heads` | 8 | 注意力头数 |
| `e_layers` | 2 | iTransformer编码器层数 |
| `flow_layers` | 3 | Velocity Network层数 |
| `d_ff` | 2048 | Feed-Forward维度 |
| `dropout` | 0.1 | Dropout率 |
| `use_norm` | True | 使用Non-stationary归一化 |

### Flow参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_sampling_steps` | 1 | ODE求解步数（1=极速单步生成） |
| `temperature` | 1.0 | 采样温度（控制多样性） |
| `num_samples` | 100 | 测试时生成的样本数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批大小 |
| `learning_rate` | 0.0001 | 学习率 |
| `epochs` | 100 | 训练轮数 |
| `patience` | 10 | 早停耐心值 |

## 设计亮点

### 1. Residual-Centric Flow

传统扩散模型从标准高斯 $\mathcal{N}(0, I)$ 开始，而iReflow从 $X_0 = \hat{y} + \epsilon \cdot \hat{\sigma}$ 开始：

- **优势1**: 起点更接近终点，流的传输代价更小
- **优势2**: 自适应起点（预测越准，噪声越小）
- **优势3**: 保留了点预测的先验知识

### 2. Variate-Aligned Architecture

Velocity Network与iTransformer保持架构一致：

- **一致的变量视角**: 都将每个变量的时间序列作为Token
- **特征复用**: 通过Cross-Attention复用iTransformer学到的变量关系
- **高效对齐**: 避免重新学习变量间的依赖关系

### 3. Confidence-Aware Generation

利用不确定性 $\hat{\sigma}$ 进行门控：

- 高置信度区域 → 小门控系数 → 抑制修正
- 低置信度区域 → 大门控系数 → 允许修正
- 防止过度修正已经准确的预测

### 4. One-Step Generation

得益于Rectified Flow的直线路径特性：

- 单步ODE求解即可生成高质量样本
- 推理速度极快（相比多步扩散模型）
- 适合实时预测场景

## 性能优势

与baseline方法相比，iReflow具有以下优势：

1. **更准确的不确定性估计**: 利用iTransformer的Aleatoric Uncertainty
2. **更快的采样速度**: One-step generation
3. **更好的变量关系建模**: Variate-Cross-Attention
4. **自适应流构建**: Residual-Centric起点

## 评估指标

iReflow支持以下概率预测指标：

- **CRPS** (Continuous Ranked Probability Score): 连续排名概率得分
- **PICP** (Prediction Interval Coverage Probability): 预测区间覆盖率
- **QICE** (Quantile Interval Coverage Error): 分位数区间覆盖误差
- **Prob-MAE/MSE/RMSE**: 概率预测的误差指标

## 故障排除

### 1. 内存不足

- 减小 `batch_size`
- 减小 `num_samples`（测试时）
- 减小 `d_model` 或 `flow_layers`

### 2. 训练不稳定

- 降低 `learning_rate`
- 增加梯度裁剪（默认1.0）
- 检查数据归一化

### 3. 采样质量差

- 增加 `num_sampling_steps`（从1增加到5-10）
- 调整 `temperature`（增加多样性）
- 增加训练轮数

## 扩展方向

1. **多步精炼**: 使用Reflow技术进一步精炼流路径
2. **条件生成**: 加入外部协变量（如节假日、天气）
3. **长序列预测**: 扩展到更长的预测范围
4. **多模态**: 结合文本、图像等多模态信息

## 引用

如果您使用iReflow，请引用：

```bibtex
@article{ireflow2025,
  title={iReflow: Rectified Flow with Inverted Variate-Awareness for Probabilistic Time Series Forecasting},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues
- Email: your.email@example.com

---

**Happy Forecasting! 🚀**

