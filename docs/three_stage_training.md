# iReflow 三阶段训练流程

## 概述

iReflow 模型采用三阶段训练策略，每个阶段专注于学习不同的能力：

```
Stage 1: 点预测 (Point Prediction)
   ↓
Stage 2: 不确定性校准 (Uncertainty Calibration)
   ↓
Stage 3: 流匹配 (Flow Matching)
```

## 为什么需要三阶段训练？

### 问题背景

在之前的实现中，`is_training=1` 模式同时训练 `uncertainty_estimator` 和 `velocity_net`，这导致：

1. **训练不稳定**：两个网络同时学习，相互影响
2. **收敛困难**：uncertainty_estimator 需要先学会估计误差，velocity_net 才能基于校准后的分布进行训练
3. **性能欠佳**：没有充分利用每个组件的专业能力

### 解决方案

将训练拆分为三个独立的阶段，每个阶段专注于一个目标：

1. **Stage 1**: 训练 iTransformer 获得良好的点预测
2. **Stage 2**: 冻结 iTransformer，训练 Uncertainty Estimator 学习误差分布
3. **Stage 3**: 冻结前两者，训练 Velocity Network 学习流变换

## 三个阶段详解

### Stage 1: 预训练点预测器 (iTransformer)

**目标**: 获得一个还不错的点预测器

**训练内容**:
- 只训练 `iTransformer`
- Loss: `MSE(y_hat, y_gt)`

**实现**:
- 使用 `iTransformer.py` 进行训练
- 或使用 `iReflow.py` 的 `is_training=2` 模式进行联合训练

**保存位置**:
```
./results/runs/iTransformer/{dataset}/{setting}/checkpoint.pth
```

### Stage 2: 预训练不确定性估计器

**目标**: 让 sigma 学会预测 Stage 1 模型的误差分布

**训练内容**:
- 冻结 `iTransformer`（加载 Stage 1 权重）
- 只训练 `uncertainty_estimator`
- Loss: `NLL_Loss = 0.5 * log(sigma²) + 0.5 * (y_gt - y_hat)² / sigma²`

**实现**:
- 使用 `pretrain_uncertainty_estimator.py` 进行训练

**命令示例**:
```bash
python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
    --wandb_project iReflow-Stage2 \
    --is_training 1 \
    --data ETTm2 \
    --seq_len 96 \
    --pred_len 192 \
    --checkpoints ./results/runs/iTransformer/ \
    runs --seeds="[2223]"
```

**保存位置**:
```
./results/runs/estimator/{dataset}/{setting}/seed_{seed}/best_model.pth
```

### Stage 3: 训练速度网络

**目标**: 学习从校准后的分布流向真实值

**训练内容**:
- 冻结 `iTransformer` 和 `uncertainty_estimator`（加载前两个阶段的权重）
- 只训练 `velocity_net`
- Loss: `Velocity_MSE = ||v_pred - (y_gt - y_0)||²`

**实现**:
- 使用 `iReflow.py` 的 `is_training=1` 模式

**命令示例**:
```bash
python3 -u ./src/experiments/iReflow.py \
    --wandb_project iReflow-Stage3 \
    --is_training 1 \
    --data ETTm2 \
    --seq_len 96 \
    --pred_len 192 \
    --checkpoints ./results/runs/iTransformer/ \
    --num_sampling_steps 5 \
    runs --seeds="[2223]"
```

**保存位置**:
```
./results/runs/iReflow/{dataset}/{hash}/train_mode_1/best_model.pth
```

## 训练模式说明

修改后的 `iReflow.py` 支持三种训练模式：

### is_training=0: 仅测试
```python
--is_training 0
```
- 不训练任何模型
- 只加载已保存的完整权重并进行测试评估
- 若 `train_mode_1` 和 `train_mode_2` 同时存在，需额外指定 `--checkpoint_mode_for_test 1` 或 `--checkpoint_mode_for_test 2`

### is_training=1: Stage 3 训练
```python
--is_training 1
```
- 只训练 `velocity_net`
- 需要加载：
  - Stage 1 预训练的 `iTransformer`
  - Stage 2 预训练的 `uncertainty_estimator`
- 最后进行测试评估

### is_training=2: 端到端训练
```python
--is_training 2
```
- 训练整个模型：`iTransformer + uncertainty_estimator + velocity_net`
- Loss: `point_loss_weight * MSE(y_hat, y_gt) + nll_loss_weight * GaussianNLL + velocity_loss_weight * Velocity_MSE`
- 端到端模式下，Flow Loss 不再对 `y_hat/sigma` 做 `detach()`，三部分参数会联合优化
- 适合快速实验或对比基线

## 完整训练示例

### 方法 1: 使用三阶段训练脚本

```bash
cd /workspace/NsDiff
chmod +x scripts/iReflow/three_stage_training_example.sh
./scripts/iReflow/three_stage_training_example.sh
```

### 方法 2: 手动执行每个阶段

#### Step 1: 训练 iTransformer
```bash
# 使用 iTransformer 脚本
python3 -u ./src/experiments/iTransformer.py \
    --wandb_project iReflow-Stage1 \
    --is_training 1 \
    --data ETTm2 \
    --seq_len 96 \
    --pred_len 192 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 2048 \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 20 \
    --patience 6 \
    --checkpoints ./results/runs/iTransformer/ \
    runs --seeds="[2223]"
```

#### Step 2: 预训练 Uncertainty Estimator
```bash
python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
    --wandb_project iReflow-Stage2 \
    --is_training 1 \
    --data ETTm2 \
    --seq_len 96 \
    --pred_len 192 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --flow_layers 3 \
    --d_ff 2048 \
    --batch_size 32 \
    --lr 0.0005 \
    --epochs 20 \
    --patience 6 \
    --checkpoints ./results/runs/iTransformer/ \
    runs --seeds="[2223]"
```

#### Step 3: 训练 Velocity Network
```bash
python3 -u ./src/experiments/iReflow.py \
    --wandb_project iReflow-Stage3 \
    --is_training 1 \
    --data ETTm2 \
    --seq_len 96 \
    --pred_len 192 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --flow_layers 3 \
    --d_ff 2048 \
    --batch_size 32 \
    --lr 0.0005 \
    --epochs 20 \
    --patience 6 \
    --num_sampling_steps 5 \
    --temperature 1.0 \
    --num_samples 100 \
    --checkpoints ./results/runs/iTransformer/ \
    runs --seeds="[2223]"
```

## 权重加载逻辑

### Stage 2 加载路径

`pretrain_uncertainty_estimator.py` 会加载：
- **iTransformer**: `{checkpoints}/{setting}/checkpoint.pth`

### Stage 3 加载路径

`iReflow.py` (is_training=1) 会加载：
- **iTransformer**: `{checkpoints}/{setting}/checkpoint.pth`
- **Uncertainty Estimator**: `./results/runs/estimator/{dataset}/{setting}/seed_{seed}/best_model.pth`

`iReflow.py` 的训练结果目录按模式隔离：
```
{save_dir}/runs/{model_type}/{dataset}/w{windows}h{horizon}s{pred_len}/{hash}/train_mode_1/
{save_dir}/runs/{model_type}/{dataset}/w{windows}h{horizon}s{pred_len}/{hash}/train_mode_2/
```

## 验证训练效果

### Stage 1 验证
- 检查点预测 MAE/MSE 是否合理
- 观察训练/验证损失曲线

### Stage 2 验证
- 检查 NLL Loss 是否收敛
- 观察 mean_sigma 是否在合理范围
- 查看 CRPS 指标（应该比 Stage 1 有所改善）

### Stage 3 验证
- 检查 Velocity Loss 是否收敛
- 观察生成样本的质量（CRPS, Coverage, Sharpness）
- 对比 is_training=2 端到端训练的结果

### is_training=2 验证
- 检查 `point_loss`、`nll_loss` 和 `velocity_loss` 是否同时收敛
- 检查点预测 MAE/MSE 是否没有因 Flow Loss 明显退化
- 对比 `train_mode_1` 与 `train_mode_2` 的 CRPS/coverage/sharpness

## 常见问题

### Q1: 为什么 Stage 2 找不到 iTransformer 权重？
**A**: 确保 Stage 1 已经完成训练，并且权重保存在正确的路径：
```
./results/runs/iTransformer/{dataset}/{setting}/checkpoint.pth
```

### Q2: 为什么 Stage 3 找不到 Uncertainty Estimator 权重？
**A**: 确保 Stage 2 已经完成训练，并且权重保存在：
```
./results/runs/estimator/{dataset}/{setting}/seed_{seed}/best_model.pth
```

### Q3: 为什么 is_training=0 测试时报 checkpoint 歧义？
**A**: `is_training=1` 和 `is_training=2` 现在分别保存在 `train_mode_1/` 与 `train_mode_2/`。如果两个目录都存在，请显式指定：
```bash
--checkpoint_mode_for_test 1
```
或
```bash
--checkpoint_mode_for_test 2
```

### Q4: 三阶段训练和端到端训练（is_training=2）哪个更好？
**A**: 
- **三阶段训练**: 更稳定，每个组件训练得更充分，通常性能更好
- **端到端训练**: 更快速，且现在会联合优化点预测、sigma 和 flow，适合快速实验和消融研究

### Q5: 可以跳过某个阶段吗？
**A**: 不建议。每个阶段都依赖前一阶段的输出：
- Stage 2 需要 Stage 1 的 iTransformer
- Stage 3 需要 Stage 1 的 iTransformer 和 Stage 2 的 Uncertainty Estimator

### Q6: 如何调整各阶段的学习率？
**A**: 建议：
- Stage 1 (iTransformer): `lr=0.0001`
- Stage 2 (Uncertainty): `lr=0.0005`
- Stage 3 (Velocity): `lr=0.0005`

## 文件说明

### 新增文件
- `src/experiments/pretrain_uncertainty_estimator.py`: Stage 2 训练脚本
- `scripts/iReflow/three_stage_training_example.sh`: 三阶段训练示例脚本
- `docs/three_stage_training.md`: 本文档

### 修改文件
- `src/experiments/iReflow.py`: 
  - 修改 `_init_optimizer()`: is_training=1 只优化 velocity_net
  - 新增 `_freeze_uncertainty_estimator()`: 冻结 uncertainty_estimator
  - 新增 `_load_uncertainty_estimator()`: 加载预训练的 uncertainty_estimator
  - 修改 `run()`: 训练结果按 `train_mode_1` / `train_mode_2` 分目录保存，测试时支持 `checkpoint_mode_for_test`
  - 新增 `point_loss_weight / nll_loss_weight / velocity_loss_weight`
- `src/models/iReflow.py`:
  - 修改 `compute_loss()`: is_training=2 联合优化 `MSE + NLL + Velocity_MSE`
  - 修改 `sample()`: 修复 `num_sampling_steps=1` 时 one-step 采样崩溃

## 总结

三阶段训练流程提供了更清晰、更稳定的训练策略：

1. ✅ **解耦训练目标**: 每个阶段专注于一个任务
2. ✅ **提高稳定性**: 避免多个组件同时训练的相互干扰
3. ✅ **更好的性能**: 每个组件都能得到充分训练
4. ✅ **灵活性**: 可以独立调整每个阶段的超参数

**建议的训练顺序**: Stage 1 → Stage 2 → Stage 3 → 测试评估
