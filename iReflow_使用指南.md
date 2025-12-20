# iReflow 使用指南

## 简介

iReflow是一个结合了iTransformer和Rectified Flow的创新时间序列概率预测模型。

## 核心优势

- 🚀 **极速采样**: 单步生成高质量预测样本
- 🎯 **精准不确定性**: 自适应的预测不确定性估计
- 🔄 **变量对齐**: 与iTransformer保持架构一致性
- 📊 **概率预测**: 输出完整的预测分布

## 快速开始

### 1. 环境准备

确保已安装项目依赖：
```bash
pip install -r requirements.txt
```

### 2. 快速测试

运行测试脚本验证模型：
```bash
python test_ireflow.py
```

预期输出：
```
==================================================
Testing iReflow Model
==================================================

1. 初始化模型...
   使用设备: cuda
   ✓ 模型初始化成功
   模型参数数量: XXX,XXX

2. 创建测试数据...
   ✓ 测试数据创建成功

...

所有测试通过！iReflow模型工作正常 ✓
```

### 3. 训练模型

#### 方式1: 使用脚本（推荐）

```bash
# ETTh1数据集
bash ./scripts/iReflow/ETTh1.sh

# ETTm2数据集
bash ./scripts/iReflow/ETTm2.sh
```

#### 方式2: 使用命令行

```bash
export PYTHONPATH=./

python3 ./src/experiments/iReflow.py \
    --dataset_type="ETTh1" \
    --windows=168 \
    --pred_len=192 \
    --d_model=512 \
    --batch_size=32 \
    --epochs=100 \
    --device="cuda:0"
```

### 4. 推理和可视化

运行推理示例：
```bash
python examples/ireflow_inference_example.py
```

这将生成：
- `ireflow_forecast_example.png`: 预测结果可视化
- `ireflow_predictions.npz`: 预测结果数据

## 主要文件说明

```
iReflow/
├── src/
│   ├── models/iReflow.py          # 主模型
│   ├── nn/velocity_network.py     # 速度场网络
│   └── experiments/iReflow.py     # 训练脚本
├── configs/ireflow.yaml            # 配置文件
├── scripts/iReflow/*.sh            # 运行脚本
├── test_ireflow.py                 # 测试脚本
└── examples/ireflow_inference_example.py  # 推理示例
```

## 核心参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 512 | 模型隐藏维度 |
| `n_heads` | 8 | 多头注意力的头数 |
| `e_layers` | 2 | iTransformer编码器层数 |
| `flow_layers` | 3 | 速度场网络层数 |
| `d_ff` | 2048 | FFN隐藏维度 |

### Flow参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_sampling_steps` | 1 | ODE求解步数（1=单步生成） |
| `temperature` | 1.0 | 采样温度（控制多样性）|
| `num_samples` | 100 | 测试时生成的样本数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批大小 |
| `learning_rate` | 0.0001 | 学习率 |
| `epochs` | 100 | 最大训练轮数 |
| `patience` | 10 | 早停耐心值 |

## Python API使用

### 基本使用

```python
import torch
from src.models.iReflow import iReflow
import argparse

# 1. 配置模型
configs = argparse.Namespace()
configs.seq_len = 168      # 历史长度
configs.pred_len = 192     # 预测长度
configs.d_model = 512
configs.n_heads = 8
configs.e_layers = 2
configs.flow_layers = 3
configs.d_ff = 2048
configs.dropout = 0.1
configs.embed = 'timeF'
configs.freq = 'h'
configs.activation = 'gelu'
configs.output_attention = False
configs.use_norm = True
configs.class_strategy = 'projection'
configs.factor = 1
configs.num_sampling_steps = 1

# 2. 创建模型
model = iReflow(configs).cuda()

# 3. 训练
model.train()
loss, loss_dict, y_hat = model(
    x_enc=x_hist,           # [B, L, D]
    x_mark_enc=x_mark,      # [B, L, T]
    y_gt=y_true,            # [B, P, D]
    mode='train'
)
loss.backward()

# 4. 推理
model.eval()
samples, y_hat, sigma = model.forecast(
    x_enc=x_hist,
    x_mark_enc=x_mark,
    num_samples=100,
    temperature=1.0
)
# samples: [B, 100, P, D]
```

### 概率预测分析

```python
# 计算统计量
mean_pred = samples.mean(dim=1)      # 均值预测
std_pred = samples.std(dim=1)        # 标准差
q10 = samples.quantile(0.1, dim=1)   # 10%分位数
q90 = samples.quantile(0.9, dim=1)   # 90%分位数

# 80%预测区间
pred_interval_80 = (q10, q90)
```

## 常见问题

### 1. 内存不足

**问题**: CUDA out of memory

**解决方案**:
```bash
# 减小批大小
--batch_size=16

# 减小模型大小
--d_model=256 --flow_layers=2

# 减少测试样本数
--num_samples=50
```

### 2. 训练不稳定

**问题**: Loss震荡或NaN

**解决方案**:
```bash
# 降低学习率
--learning_rate=0.00005

# 增加梯度裁剪
# (在experiments/iReflow.py中修改max_grad_norm)
```

### 3. 采样质量差

**问题**: 生成的样本质量不佳

**解决方案**:
```bash
# 增加采样步数
--num_sampling_steps=5

# 调整温度
--temperature=0.8  # 降低多样性
--temperature=1.2  # 增加多样性

# 增加训练轮数
--epochs=200
```

## 评估指标

模型自动计算以下指标：

- **CRPS**: 连续排名概率得分（越小越好）
- **PICP**: 预测区间覆盖率（接近名义覆盖率最好）
- **QICE**: 分位数区间覆盖误差（越小越好）
- **Prob-MAE/MSE/RMSE**: 概率预测误差

## 输出结果

### 训练输出

```
Epoch 1/100
Train Loss: 0.1234
Val Loss: 0.1456

Epoch 2/100
...
```

### 测试输出

```
Test Results:
crps: 0.0123
picp: 0.7845
qice: 0.0234
mse: 0.0456
mae: 0.0234
rmse: 0.0678
```

### 保存的文件

- `checkpoints/iReflow/checkpoint.pth`: 最佳模型权重
- `logs/iReflow/`: 训练日志
- `results/runs/iReflow/`: 测试结果

## 进阶使用

### 自定义数据集

1. 准备数据格式（CSV，包含时间戳和变量列）
2. 创建数据加载器（继承`TimeSeriesDataset`）
3. 修改配置中的`dataset_type`

### 调整模型架构

修改`configs/ireflow.yaml`中的参数：
```yaml
model:
  d_model: 1024      # 增大模型容量
  e_layers: 4        # 增加编码器深度
  flow_layers: 5     # 增加流网络深度
```

### 多GPU训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiments/iReflow.py \
    --device="cuda" \
    # 其他参数...
```

## 性能基准

基于ETTh1数据集（windows=168, pred_len=192）：

| 指标 | 数值 |
|------|------|
| CRPS | 待测试 |
| 训练时间 | ~2小时（单GPU） |
| 推理速度 | ~10ms/样本 |
| 模型参数 | ~5M |

## 技术支持

详细技术文档请参考：
- 📖 `docs/iReflow_README.md`: 完整技术文档
- 📊 `IREFLOW_IMPLEMENTATION.md`: 实现细节
- 🏗️ `ARCHITECTURE.md`: 项目架构

## 引用

如果使用iReflow，请引用：

```bibtex
@article{ireflow2025,
  title={iReflow: Rectified Flow with Inverted Variate-Awareness for Probabilistic Time Series Forecasting},
  year={2025}
}
```

## 许可证

遵循项目主LICENSE。

---

**祝您使用愉快！** 🎉

如有问题，请查阅详细文档或提交Issue。

