# Third-order shape pilot：条件偏度分解实验设计

## 目标与边界

本 pilot 检验 LS-Flow 的层级分解能否从已提交的二阶位置--尺度形式，受控地扩展为条件三阶（偏度）分解。它服务于 Meta/R1/R3 的“框架可扩展性”关切，而不是提出新主方法或追求全面性能刷新。

偏度不是重尾。本实验不声称解决 heavy-tail 问题，也不引入四阶尾重参数；任何结果都不得外推为重尾建模能力。

## 对照与模型约束

| ID | 定义 | 训练要求 |
|---|---|---|
| L2 | 已提交的二阶 LS-Flow：位置--尺度残差分解 | 直接复用现有 checkpoint；不重新训练、不重新选择 checkpoint。 |
| L3 | 在 L2 的位置--尺度映射上增加条件偏度参数 `alpha(X)` 的三阶残差分解 | 对应 L2 checkpoint 的 encoder、`mu`、`sigma` 全部冻结；只训练新增的条件偏度部分及其三阶残差空间中的生成模块。 |

L2 checkpoint 覆盖五个已提交 seed：`2022--2026`。L2 与 L3 均须报告五个独立 seed 的汇总；不要求二者 seed 一一对应，也不得以单一有利 seed 代替五 seed 汇总。

L3 的形状变换须满足以下条件：

- 为逐条件、逐元素的可逆映射，且明确给出逆映射与 Jacobian；
- `alpha=0` 时严格退化为 L2 的二阶映射；
- `alpha` 通过包含 Jacobian 的条件似然进行校准，而不是作为任意辅助变量；
- 不改变冻结的 encoder、`mu`、`sigma` 输出，不扩大既有二阶模块或 CFM 的容量；
- 在三阶残差坐标中训练和生成；不得把二阶的生成映射直接套用到 L3。

## 数据集与执行

1. 在全部八个 submitted 数据集上开展该 pilot；不得依据验证集或测试集的偏度、性能或其他结果删减数据集。
2. 对每个数据集，评测五个既有 L2 checkpoint，并训练/评测五个独立 L3 run。L2 与 L3 必须使用相同的 submitted split、context length、forecast horizon、全局数据标准化、ODE 设置、预测样本数和评测代码。
3. 验证集仅可用于 L3 训练过程中的 checkpoint 选择和条件偏度诊断；诊断结果在全部八个数据集上保存，但不用于选择是否运行某个数据集。
4. 测试集不得用于选择 checkpoint、seed 或 L3 配置。即使某个数据集未显示明显偏度信号或 L3 训练失败，也须在最终报告中如实标注其完整结果或失败原因。

## 评测与报告

报告每个数据集、每个 seed 的 L2/L3：

- marginal CRPS、CRPS-sum、Energy Score（ES）、MSE、MAE；
- 90%、95%、99% 中心预测区间的 coverage、coverage error 和平均宽度；
- 三阶变换前后残差的偏度与 excess kurtosis，用于确认条件偏度是否确实被显式分解；
- 代码 commit、L2 checkpoint 标识、配置路径、L3 新增参数量及异常/失败 run。

分别汇总 L2 与 L3 五个 seed 的均值 ± 标准差。ES 仅在同一数据集内比较。不得只报告 CRPS、最佳 seed 或正向结果。

## 判定与 rebuttal 边界

若 L3 通过可逆性/Jacobian 检查、在验证集显示条件形状信号，并在测试中降低残差偏度，则可将其描述为“第三阶条件偏度分解的可行性证据”。若 CRPS/ES 或 calibration 也稳定改善，可作为更强但仍受限的实证支持。

若结果为负面或混合，完整报告并仅表述为“该受限 pilot 未显示稳定预测收益”。不得据此声称三阶层级普遍有效、LS-Flow 必然优于 L2，或偏度扩展已经处理重尾。

结果写入 `evidence/shape_decomposition.md` 和 `evidence/shape_decomposition_per_seed.csv`；后者至少包含：

`dataset, seed, design, CRPS, CRPS_sum, ES, MSE, MAE, coverage_90, coverage_95, coverage_99, coverage_error_90, coverage_error_95, coverage_error_99, interval_width_90, interval_width_95, interval_width_99, skewness_before, skewness_after, excess_kurtosis_before, excess_kurtosis_after, checkpoint, commit, config_path`。
