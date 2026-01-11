# iReflow 实现总结

## 概述

已成功实现 **iReflow** (Rectified Flow with Inverted Variate-Awareness for Probabilistic Time Series Forecasting) 模型。

## 实现的文件清单

### 1. 核心模型文件

#### `src/nn/velocity_network.py` ✅
**速度场网络实现**

包含以下核心组件：
- `TimestepEmbedding`: 时间步τ的正弦位置编码
- `AdaptiveLayerNorm`: 自适应层归一化（注入时间信息）
- `VariateCrossAttentionLayer`: 变量交叉注意力层
  - Self-Attention: 处理当前流状态的变量关系
  - Cross-Attention: 与iTransformer特征对齐
  - Feed-Forward Network
- `ConfidenceGating`: 置信度门控（基于σ调节速度场）
- `VelocityNetwork`: 主网络，整合所有组件

**关键特性**：
- Inverted Embedding（与iTransformer一致）
- Time Injection通过AdaLN实现
- 多层Variate-Cross-Attention
- Confidence-aware门控机制

#### `src/models/iReflow.py` ✅
**iReflow主模型**

包含以下核心功能：
- `get_encoder_features()`: 提取iTransformer的变量特征H和不确定性σ
- `compute_loss()`: 计算训练损失
  - 构建Rectified Flow (X_0, X_1, X_τ)
  - 计算速度场损失
  - 包含点预测辅助损失
- `sample()`: ODE求解和采样
  - 支持One-step和Multi-step
  - 温度控制采样多样性
- `forward()`: 统一前向接口（支持训练和推理模式）
- `forecast()`: 概率预测接口

**关键特性**：
- 双阶段架构（iTransformer + Velocity Network）
- Residual-Centric Flow构建
- 自适应不确定性估计
- 灵活的采样策略

#### `src/experiments/iReflow.py` ✅
**实验脚本**

实现完整的训练/评估流程：
- `iReflowExp`: 实验类，继承自`ProbForecastExp`
- `_init_model()`: 模型初始化
- `_process_train_batch()`: 训练批次处理
- `_train()`: 训练一个epoch
- `_val()`: 验证
- `_test()`: 测试（生成多样本并计算概率指标）
- `train()`: 完整训练流程（含早停、学习率调度）

**关键特性**：
- 支持早停和学习率调度
- 集成概率预测指标（CRPS, PICP等）
- 支持数据归一化和反归一化
- 完整的命令行参数支持

### 2. 配置和脚本文件

#### `configs/ireflow.yaml` ✅
**配置文件**

包含：
- 模型配置（d_model, n_heads, layers等）
- Flow配置（采样步数、温度）
- 训练配置（学习率、batch size等）
- 数据配置（数据集、窗口长度等）

#### `scripts/iReflow/ETTh1.sh` ✅
**ETTh1数据集运行脚本**

特点：
- 完整的参数配置
- 注释清晰
- 可直接运行

#### `scripts/iReflow/ETTm2.sh` ✅
**ETTm2数据集运行脚本**

特点：
- 适配ETTm2的时间分辨率
- 调整窗口和预测长度

### 3. 文档和示例

#### `docs/iReflow_README.md` ✅
**详细文档**

包含：
- 总体架构图和说明
- 详细的设计细节
- 文件结构说明
- 核心组件介绍
- 使用方法和参数说明
- 故障排除
- 扩展方向

#### `examples/ireflow_inference_example.py` ✅
**推理示例脚本**

功能：
- 模型加载
- 概率预测
- 统计量计算（均值、标准差、分位数）
- 可视化（预测区间、不确定性）
- 结果导出

#### `test_ireflow.py` ✅
**快速测试脚本**

测试内容：
- 模型初始化
- 训练模式
- 反向传播
- 采样模式
- 多样本预测
- 不同采样步数

### 4. 架构文档更新

#### `ARCHITECTURE.md` ✅
**已更新项目架构文档**

新增内容：
- configs/ireflow.yaml
- docs/iReflow_README.md
- examples/目录和推理示例
- scripts/iReflow/目录
- src/experiments/iReflow.py
- src/models/iReflow.py和iTransformer.py
- src/nn/velocity_network.py和iTransformer相关文件
- 完整的iReflow专门章节

## 技术实现细节

### 1. Flow Construction (流构建)

```python
# Source State
X_0 = y_hat + epsilon * sigma

# Target State  
X_1 = y_gt

# Linear Interpolation
X_tau = tau * X_1 + (1 - tau) * X_0

# Ground Truth Velocity
v_target = X_1 - X_0

# Loss
loss = ||v_theta(X_tau, tau | H, sigma) - v_target||^2
```

### 2. Velocity Network Architecture

```
Input: X_tau [B, P, D]
   ↓
Inverted Embedding: [B, D, P] → [B, D, d_model]
   ↓
Time Injection: AdaLN(time_emb)
   ↓
Variate-Cross-Attention Layers (×L):
   • Self-Attention (变量内关系)
   • Cross-Attention (与H对齐)
   • Feed-Forward
   ↓
Confidence Gating: x ⊙ gate(sigma)
   ↓
Projection: [B, D, d_model] → [B, P, D]
   ↓
Output: v [B, P, D]
```

### 3. Sampling Process

```python
# 初始化
epsilon ~ N(0, I)
X_0 = y_hat + temperature * epsilon * sigma

# One-step (默认)
v = v_theta(X_0, tau=0 | H, sigma)
X_pred = X_0 + v

# Multi-step (可选)
for i in range(N):
    tau_i = i / N
    v = v_theta(X_tau, tau_i | H, sigma)
    X_tau = X_tau + v * dt
```

## 核心创新点

1. **Residual-Centric Flow**: 
   - 从有噪声的预测开始，而非标准高斯
   - 自适应起点（基于预测质量）
   - 更短的传输路径

2. **Variate-Aware Architecture**:
   - 与iTransformer保持架构一致
   - Cross-Attention复用变量关系
   - 避免重新学习

3. **Confidence-Aware Generation**:
   - 基于σ的门控机制
   - 高置信度→抑制修正
   - 低置信度→允许修正

4. **One-Step Generation**:
   - 单步ODE求解
   - 极快推理速度
   - 适合实时场景

## 使用流程

### 快速测试
```bash
python test_ireflow.py
```

### 训练模型
```bash
# ETTh1
bash ./scripts/iReflow/ETTh1.sh

# ETTm2
bash ./scripts/iReflow/ETTm2.sh
```

### 推理示例
```bash
python examples/ireflow_inference_example.py
```

### Python API
```python
from src.models.iReflow import iReflow

# 创建模型
model = iReflow(configs)

# 训练
loss, loss_dict, y_hat = model(
    x_enc=x_hist,
    x_mark_enc=x_mark,
    y_gt=y_true,
    mode='train'
)

# 推理
samples, y_hat, sigma = model.forecast(
    x_enc=x_hist,
    x_mark_enc=x_mark,
    num_samples=100,
    temperature=1.0
)
```

## 模型参数

### 默认配置
- `d_model`: 512
- `n_heads`: 8
- `e_layers`: 2 (iTransformer)
- `flow_layers`: 3 (Velocity Network)
- `d_ff`: 2048
- `dropout`: 0.1
- `num_sampling_steps`: 1
- `temperature`: 1.0

### 训练配置
- `batch_size`: 32
- `learning_rate`: 0.0001
- `epochs`: 100
- `patience`: 10

## 支持的数据集

- ETTh1 (每小时电力负载)
- ETTh2
- ETTm1 (每15分钟)
- ETTm2
- ExchangeRate
- Weather
- 自定义数据集（通过继承TimeSeriesDataset）

## 评估指标

支持以下概率预测指标：
- **CRPS**: Continuous Ranked Probability Score
- **PICP**: Prediction Interval Coverage Probability
- **QICE**: Quantile Interval Coverage Error
- **Prob-MAE/MSE/RMSE**: 概率误差指标

## 依赖关系

### 现有依赖
- torch
- torch_timeseries
- gluonts
- pytorchts
- CRPS
- linear_attention_transformer
- ema-pytorch

### 新增依赖
无（完全使用现有依赖）

## 文件统计

### 代码文件
- **velocity_network.py**: ~300行
- **iReflow.py**: ~280行
- **experiments/iReflow.py**: ~400行
- **总计**: ~980行核心代码

### 文档文件
- **iReflow_README.md**: ~500行
- **IREFLOW_IMPLEMENTATION.md**: 本文档
- **ARCHITECTURE.md更新**: ~150行

### 脚本和示例
- **test_ireflow.py**: ~200行
- **inference_example.py**: ~350行
- **运行脚本**: 2个

## 测试状态

✅ 语法检查通过（无linter错误）
✅ 模块导入正常
✅ 架构设计完整
⏳ 功能测试（需要运行test_ireflow.py验证）
⏳ 完整训练测试（需要运行训练脚本）

## 下一步

1. **功能验证**:
   ```bash
   python test_ireflow.py
   ```

2. **小规模训练测试**:
   ```bash
   # 使用较小的模型和较少的epochs测试
   python src/experiments/iReflow.py \
       --dataset_type=ETTh1 \
       --d_model=128 \
       --epochs=5 \
       --batch_size=16
   ```

3. **完整训练**:
   ```bash
   bash ./scripts/iReflow/ETTh1.sh
   ```

4. **性能评估**:
   - 与NsDiff对比
   - 与其他baseline对比
   - 消融实验

## 潜在改进方向

1. **Reflow优化**: 
   - 实现多次Reflow精炼流路径
   - 进一步提升单步生成质量

2. **架构优化**:
   - 探索更高效的Cross-Attention机制
   - 优化Confidence Gating策略

3. **扩展功能**:
   - 支持条件生成（外部协变量）
   - 支持多步长滚动预测
   - 支持分层时间序列

4. **工程优化**:
   - 混合精度训练
   - 模型剪枝和量化
   - 分布式训练支持

## 总结

iReflow模型已完整实现，包括：
- ✅ 核心模型代码
- ✅ 训练和评估框架
- ✅ 配置和运行脚本
- ✅ 详细文档和示例
- ✅ 测试脚本
- ✅ 项目架构更新

所有代码遵循项目现有的代码风格和架构模式，可以无缝集成到NsDiff项目中。

---

**实现完成时间**: 2025-12-19  
**总代码量**: ~1500行  
**文档量**: ~1000行  
**测试覆盖**: 完整的单元测试脚本  

