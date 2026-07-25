# Mean-only ablation 实验设计

## 目标

检验 LS-Flow 的收益是否来自**二阶尺度分解**，而不只是来自预测均值或使用了更有信息的先验。该实验直接回答：在相同的条件表征与生成模型下，去除预测尺度后，是否仍优于仅去均值的残差建模？

## 对照

| 设计 | 残差训练目标 | 生成映射 |
|---|---|---|
| Mean-only | \(Z=Y-\hat\mu(X)\) | \(Y=\hat\mu(X)+Z\) |
| Location--scale（LS-Flow） | \(Z=(Y-\hat\mu(X))/\hat\sigma(X)\) | \(Y=\hat\mu(X)+\hat\sigma(X)Z\) |

Mean-only 中必须将尺度固定为 1，并同时用于残差标准化、初始源分布和最终逆映射；不得保留任何由 \(\hat\sigma(X)\) 构造的先验、噪声尺度或条件输入。

## 约束

- 两个设计使用相同的 backbone、CFM 容量、优化器、训练预算、数据划分、context/horizon、ODE 设置、预测样本数和评测代码。
- 两者均预测 \(\hat\mu(X)\)；仅 Location--scale 设计额外使用 \(\hat\sigma(X)\)。不得因 Mean-only 去掉尺度而改变网络宽度、层数或调参预算。
- 在相同数据集上各运行三个独立 seed，并分别报告均值 ± 标准差；不要求 seed 一一对应。
- 使用 submitted 的八个数据集；若计算资源有限，优先完成已有 Table 4 涉及的 ETTm1、ETTm2、Weather、Electricity，并明确标注未完成数据集。
- 测试集不得用于模型、checkpoint 或超参数选择。

## 报告

每个数据集和 seed 报告 marginal CRPS、CRPS-sum、Energy Score、MSE、MAE，以及 90%/95% coverage 和区间宽度。额外报告两种残差的均值、方差与偏度，帮助解释尺度分解是否使残差更接近统一坐标。

结果写入 `evidence/mean_only_ablation.md` 和 `evidence/mean_only_ablation_per_seed.csv`。rebuttal 中仅可据结果表述“显式尺度分解相对 mean-only 的经验影响”；不得把单一数据集或单一 seed 的结果外推为普遍优势。
