# iReflow 模型改进说明

## 问题分析

根据实验结果，iReflow模型在加入Flow Stage后，MSE和MAE指标反而变差，且PICP（预测区间覆盖率）接近0。这表明存在以下问题：

1. **PICP接近0**：预测区间几乎没有覆盖真实值，说明不确定性估计不准确
2. **MSE/MAE变差**：Velocity Network的修正反而让预测变差，说明学习方向可能有问题

## 根本原因

经过仔细分析代码，发现以下问题：

### 1. 不确定性估计器设计问题
- **问题**：`uncertainty_estimator`只从encoder特征估计sigma，没有考虑预测误差
- **影响**：sigma可能估计过小或过大，导致：
  - 如果sigma过小：X_0 ≈ y_hat，Velocity Network学习的速度场v ≈ y_gt - y_hat，但实际推理时如果sigma不匹配，预测会偏差
  - 如果sigma过大：X_0的噪声太大，Velocity Network难以学习正确的修正方向

### 2. 损失函数设计问题
- **问题**：只优化velocity loss，没有直接优化最终预测的MSE
- **影响**：Velocity Network可能学习到正确的速度场，但最终预测质量不一定好
- **原因**：训练和推理的分布可能不匹配（训练时使用epsilon，推理时使用temperature * epsilon）

### 3. 置信度门控机制问题
- **问题**：ConfidenceGating可能过度抑制修正
- **影响**：即使Velocity Network学习到正确的修正，也可能被门控机制抑制

### 4. 单步生成可能不够准确
- **问题**：虽然Rectified Flow理论上可以单步生成，但如果速度场学习不准确，单步可能不够
- **影响**：预测质量下降

## 改进方案

### 1. 改进不确定性估计器 ✅

**改进内容**：
- 添加Dropout层，提高泛化能力
- 使用更合理的初始化（较小的gain），让sigma初始值更合理
- 限制sigma的范围，避免过大或过小
- 使用相对标准差，让sigma的尺度更合理

**代码位置**：`src/models/iReflow.py` 第29-48行

**关键改进**：
```python
# 初始化：使用较小的gain
nn.init.xavier_uniform_(m.weight, gain=0.1)

# 限制sigma范围
sigma = torch.clamp(sigma * relative_std, min=1e-6, max=relative_std * 2.0)
```

### 2. 改进损失函数 ✅

**改进内容**：
- 加入最终预测损失（prediction_loss），直接优化最终预测的MSE
- 平衡各项损失的权重：
  - `velocity_loss`: 确保速度场学习正确
  - `prediction_loss`: 确保最终预测质量（权重0.5）
  - `point_loss`: 辅助损失，保持点预测质量（权重0.1）

**代码位置**：`src/models/iReflow.py` 第174-195行

**关键改进**：
```python
# 在训练时也进行采样，计算最终预测损失
X_0_train = y_hat + epsilon * sigma
tau_train = torch.zeros(B, device=device)
v_train = self.velocity_net(X_0_train, tau_train, enc_features, y_hat, sigma)
y_pred_train = X_0_train + v_train
prediction_loss = F.mse_loss(y_pred_train, y_gt)

# 平衡各项损失
total_loss = velocity_loss + 0.5 * prediction_loss + 0.1 * point_loss
```

**为什么有效**：
- 直接优化最终预测质量，确保Velocity Network学习的方向是正确的
- 训练和推理使用相同的采样方式（temperature=1.0），减少分布不匹配问题

### 3. 改进置信度门控机制 ✅

**改进内容**：
- 使用归一化的sigma来生成门控系数，避免sigma尺度问题
- 添加可学习的偏移量（gate_bias），确保即使sigma很小时也有一定的修正
- 限制门控系数在[0.1, 1.0]范围内，避免完全抑制修正

**代码位置**：`src/nn/velocity_network.py` 第149-210行

**关键改进**：
```python
# 归一化sigma
sigma_mean = sigma.mean(dim=-1, keepdim=True)
sigma_std = sigma.std(dim=-1, keepdim=True) + 1e-6
sigma_norm = (sigma - sigma_mean) / sigma_std
sigma_norm = torch.sigmoid(sigma_norm)

# 添加偏移量
gate = gate + self.gate_bias.to(x.device)
gate = torch.clamp(gate, min=0.1, max=1.0)
```

**为什么有效**：
- 归一化sigma避免尺度问题
- 最小门控系数0.1确保不会完全抑制修正
- 可学习的偏移量让模型自适应调整门控强度

### 4. 改进采样过程 ✅

**改进内容**：
- 保持多步ODE求解的支持
- 改进Euler方法的实现（虽然当前使用单步，但多步实现已优化）

**代码位置**：`src/models/iReflow.py` 第207-219行

## 预期效果

这些改进应该能够：

1. **提高PICP**：通过改进不确定性估计和限制sigma范围，预测区间应该更合理
2. **降低MSE/MAE**：通过加入prediction_loss直接优化最终预测质量
3. **提高训练稳定性**：通过改进损失函数和门控机制，训练过程应该更稳定

## 使用建议

### 1. 调整损失权重（如果需要）

如果发现prediction_loss过大或过小，可以调整权重：

```python
# 在 compute_loss 中调整
total_loss = velocity_loss + alpha * prediction_loss + beta * point_loss
# 默认：alpha=0.5, beta=0.1
```

### 2. 调整采样步数

如果单步生成质量不够，可以增加采样步数：

```bash
# 在脚本中设置
NUM_SAMPLING_STEPS=5  # 从1增加到5
```

### 3. 监控训练过程

关注以下指标：
- `velocity_loss`: 应该逐渐下降
- `prediction_loss`: 应该逐渐下降，且应该接近或低于point_loss
- `mean_sigma`: 应该在一个合理的范围内（不会过大或过小）
- `max_sigma` / `min_sigma`: 检查sigma的范围是否合理

## 下一步优化方向

如果这些改进还不够，可以考虑：

1. **自适应sigma估计**：根据历史预测误差动态调整sigma
2. **多尺度损失**：在不同时间步上计算损失，提高长程预测能力
3. **残差连接**：在Velocity Network中加入残差连接，提高训练稳定性
4. **温度调度**：在训练过程中逐渐降低temperature，提高生成质量

## 总结

这些改进主要解决了：
1. ✅ 不确定性估计不准确的问题
2. ✅ 损失函数没有直接优化最终预测的问题
3. ✅ 置信度门控过度抑制修正的问题
4. ✅ 训练和推理分布不匹配的问题

通过这些改进，iReflow模型应该能够：
- 生成更合理的预测区间（提高PICP）
- 提高预测质量（降低MSE/MAE）
- 更稳定地训练

