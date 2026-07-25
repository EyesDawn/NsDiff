# 三阶条件偏度分解（L3）Pilot

## 概要

在八个数据集（ETTh1、ETTh2、ETTm1、ETTm2、Electricity、SolarEnergy、Traffic、Weather）上新增 L3 条件偏度分解实验。当前 pilot 每个数据集固定一个 L2 checkpoint，并训练一个与其绑定的 L3 run；L2 不重新训练或重选。

L3 在冻结的 L2 位置--尺度分解上增加条件偏度参数 $\alpha(X)$，并在新的三阶残差坐标中训练独立生成器。该实验只检验条件偏度分解的可行性，不宣称解决重尾问题或普遍优于 L2。

## 模型与训练

### 冻结的 L2 模块

- 从 manifest 指定的 L2 checkpoint 加载 encoder、$\mu$、$\sigma$ 和原始 L2 generator。
- 所有 L2 参数被冻结，且 L2 保持在 inference mode。
- L3 不改变 $\mu$、$\sigma$ 或 encoder 的输出，不扩大原 L2 generator 的容量。

### 三阶变换

定义二阶标准化残差：

$$
z_2 = \frac{Y - \mu(X)}{\sigma(X)}.
$$

新增逐条件、逐元素的偏度变换：

$$
z_3 = T_\alpha(z_2) = z_2 + \alpha(X) \operatorname{softplus}(z_2).
$$

其中 $\alpha(X) > -1 + 10^{-6}$，由冻结 encoder 输出的变量 token 经线性头预测。$\alpha=0$ 时，$T_\alpha$ 严格为恒等映射，L3 退化为 L2。

变换的 Jacobian 为：

$$
\frac{\partial z_3}{\partial z_2} = 1 + \alpha(X)\operatorname{sigmoid}(z_2) > 0.
$$

因此变换可逆。采样阶段使用有界二分法计算 $T_\alpha^{-1}$，并在训练中检查 Jacobian 为正。

### 条件似然与生成器

使用 $z_3 \sim \mathcal{N}(0, I)$ 的条件负对数似然校准 $\alpha$，其中包括 Jacobian 项：

$$
\mathcal{L}_{\mathrm{NLL}}
= \mathbb{E}\left[
\frac{1}{2}z_3^2 + \frac{1}{2}\log(2\pi)
- \log\left|\frac{\partial z_3}{\partial z_2}\right|
\right].
$$

L3 新增一个与 L2 generator 完全同容量的 `VelocityNetwork`，但直接在 $z_3$ 坐标中训练。其 source 为标准高斯噪声 $\epsilon$，rectified-flow 目标为 $z_3$：

$$
z_\tau = \tau z_3 + (1-\tau)\epsilon,
\qquad
v^* = z_3 - \epsilon.
$$

训练目标为：

$$
\mathcal{L}_{\mathrm{L3}}
= \mathcal{L}_{\mathrm{CFM}} + \lambda_{\mathrm{NLL}}\mathcal{L}_{\mathrm{NLL}},
\qquad
\lambda_{\mathrm{NLL}}=1.
$$

验证集使用同一总损失早停和选择 L3 checkpoint；测试集不参与 checkpoint、seed 或配置选择。

### 采样

1. 在 $z_3$ 空间从标准高斯 source 经 L3 generator 生成样本。
2. 通过 $z_2=T_\alpha^{-1}(z_3)$ 回到二阶残差坐标。
3. 使用冻结的位置--尺度映射恢复预测：

$$
Y = \mu(X) + \sigma(X)z_2.
$$

## 实验配置与可复现性

- 基线 checkpoint 由 `configs/third_order_shape_manifest.yaml` 唯一指定，并在运行前校验 SHA-256。
- 当前 pilot 每个数据集使用 `l2_1` 和一个绑定的 L3 seed；完整五 seed 设计可通过补充 manifest 条目并设置 `--runs_per_dataset 5` 恢复。
- L3 复用对应 L2 的 submitted split、context length、forecast horizon、全局标准化、ODE 步数、预测样本数和评测尺度。
- L3 run 的新增可训练参数仅包括线性 $\alpha$ 头和同容量三阶 generator。
- `metadata.json` 记录 L2 checkpoint、哈希、配置、L3 seed、参数量、验证诊断与代码 commit。

## 评测与报告

对 L2 和 L3 分别报告：

- marginal CRPS、CRPS-sum、同一数据集内的 Energy Score（ES）、ensemble-mean MSE 和 MAE；
- 90%、95%、99% 中心预测区间的 coverage、绝对 coverage error 和平均宽度；
- 测试 target 的残差偏度与 excess kurtosis。

L2 的形状统计使用 $z_2$ 作为变换前后残差；L3 使用 $z_2$ 作为变换前残差、$z_3$ 作为变换后残差。验证集额外记录 $\alpha$ 的 NLL 与平均绝对值，作为条件形状诊断。

逐 run 结果写入：

- `evidence/shape_decomposition_per_seed.csv`
- `evidence/shape_decomposition.md`

CSV 至少包含：

```text
dataset, seed, design, CRPS, CRPS_sum, ES, MSE, MAE,
coverage_90, coverage_95, coverage_99,
coverage_error_90, coverage_error_95, coverage_error_99,
interval_width_90, interval_width_95, interval_width_99,
skewness_before, skewness_after,
excess_kurtosis_before, excess_kurtosis_after,
checkpoint, commit, config_path
```

并额外记录 `run_id`、`l3_seed`、`l3_checkpoint`、新增参数量、验证诊断、状态和失败原因。

## 多 GPU 执行

使用 `NsDiff` conda 环境，在三张 GPU 上按数据集并行运行：

```bash
GPU_IDS="0 1 2" bash scripts/iReflow/run_third_order_shape_pilot_multi_gpu.sh
```

每个进程写入独立 evidence shard；全部数据集完成后，合并脚本才会生成最终的 `evidence/shape_decomposition.md` 与 `evidence/shape_decomposition_per_seed.csv`，避免并发写入冲突。

## 验证与失败处理

- 单元检查覆盖 $\alpha=0$ 的恒等性、Jacobian 正性、正反变换误差与接近下界时的数值稳定性。
- 运行时断言 L2 参数无梯度、L3 generator 参数量等于 L2 generator 参数量。
- 若 L3 失败，保留已完成的 L2 评测结果，并在 CSV 与 Markdown 中记录 L3 失败原因。
- 若 L2 checkpoint、配置或环境导致运行无法开始，则分别写入 L2 与 L3 的失败记录。

## 结论边界

若 L3 通过可逆性和 Jacobian 检查、验证集显示条件形状信号，且测试残差偏度下降，可描述为第三阶条件偏度分解的可行性证据。若 CRPS、ES 或 calibration 也稳定改善，可作为更强但仍受限的实证支持。

负面或混合结果必须完整报告，只能表述为该受限 pilot 未显示稳定预测收益；不得将其外推为重尾建模能力、三阶层级普遍有效，或 LS-Flow 必然优于 L2。
