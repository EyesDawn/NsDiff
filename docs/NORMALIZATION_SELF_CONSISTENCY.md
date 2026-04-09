# iReflow 中两种 Normalization 的理论自洽性分析

## 问题

在当前 iReflow 实现中，`VelocityNetwork` 使用如下相对坐标：

\[
z_\tau = \frac{x_\tau - \hat{y}}{\sigma}
\]

对应代码见 [src/nn/velocity_network.py](/workspace/NsDiff/src/nn/velocity_network.py)。

现在希望从理论上比较以下两种 normalization，判断哪一种在当前 flow 定义下更自洽：

1. 使用条件分布参数 `y_hat, sigma`
   \[
   z_\tau^{(pred)} = \frac{x_\tau - \hat{y}}{\sigma}
   \]
2. 使用当前状态的统计量 `x_tau.mean, x_tau.std`
   \[
   z_\tau^{(state)} = \frac{x_\tau - \mu(x_\tau)}{s(x_\tau)}
   \]

这里“更自洽”不是指“某些数据集上可能更稳”，而是指：

1. 是否与当前的 `X_0 / X_\tau / v^*` 定义一致。
2. 是否与 velocity field 的参数化一致。
3. 是否与采样阶段的 ODE/Euler 更新一致。
4. 是否保留了条件概率分布的正确几何结构。

## 当前 iReflow 的 Flow 定义

代码中的训练构造见 [src/models/iReflow.py](/workspace/NsDiff/src/models/iReflow.py)：

\[
X_0 = \hat{y} + \epsilon \sigma,\quad \epsilon \sim \mathcal{N}(0, I)
\]

\[
X_1 = Y
\]

\[
X_\tau = \tau X_1 + (1-\tau) X_0
\]

\[
v^*(X_\tau, \tau) = X_1 - X_0
\]

其关键性质是：

1. `X_0` 来自条件高斯分布 `N(y_hat, sigma^2)`。
2. `X_\tau` 是 `X_0` 到 `Y` 的线性插值。
3. 目标速度 `v^*` 在这条直线路径上与 `tau` 无关，是常向量。

因此，这个模型本质上是在学习一个“以条件预测分布为起点”的 rectified flow。

## 判断自洽性的标准

设我们对 `x_\tau` 做一般的仿射变换：

\[
z_\tau = \frac{x_\tau - a}{b}
\]

若 `a, b` 是给定条件下的固定量，并且不依赖当前 `x_\tau`，则有

\[
x_\tau = a + b z_\tau
\]

对 `\tau` 求导：

\[
\frac{d x_\tau}{d\tau} = b \frac{d z_\tau}{d\tau}
\]

此时在 `z` 空间学习速度，再映回 `x` 空间，是严格成立的。因为坐标变换是固定的条件仿射变换。

但如果 `a, b` 依赖于 `x_\tau` 本身，例如 `a=\mu(x_\tau), b=s(x_\tau)`，则

\[
x_\tau = \mu(x_\tau) + s(x_\tau) z_\tau
\]

这不再是固定仿射坐标系。对 `\tau` 求导时会出现额外项：

\[
\frac{d x_\tau}{d\tau}=\frac{d \mu(x_\tau)}{d\tau}z_\tau \frac{d s(x_\tau)}{d\tau}s(x_\tau)\frac{d z_\tau}{d\tau}
\]

这意味着简单地“在标准化空间预测 `u`，再乘回一个尺度”已经不再对应真实的 `dx_\tau / d\tau`。

这就是两类 normalization 的根本差异。

## 方案一：使用 `y_hat, sigma` 的理论性质

### 1. 与源分布定义完全一致

由于

\[
X_0 = \hat{y} + \epsilon \sigma
\]

所以用

\[
z_\tau^{(pred)} = \frac{x_\tau - \hat{y}}{\sigma}
\]

时，在 `\tau=0` 有

\[
z_0 = \frac{X_0 - \hat{y}}{\sigma} = \epsilon \sim \mathcal{N}(0, I)
\]

这说明 source state 在该坐标系下被精确地标准化成标准高斯。这不是近似结论，而是由构造直接推出。

### 2. 插值路径在标准化空间中仍然线性

把 `X_\tau = \tau Y + (1-\tau)X_0` 代入：

\[
z_\tau=\frac{X_\tau - \hat{y}}{\sigma}=\tau \frac{Y-\hat{y}}{\sigma} (1-\tau)\frac{X_0-\hat{y}}{\sigma}
\]

因为

\[
\frac{X_0-\hat{y}}{\sigma} = \epsilon
\]

所以

\[
z_\tau=\tau \frac{Y-\hat{y}}{\sigma} (1-\tau)\epsilon
\]

即 `z` 空间里的轨迹仍然是从 `epsilon` 到标准化残差 `(Y-y_hat)/sigma` 的线性插值。

这与原始 rectified flow 的构造完全同构。

### 3. 速度变换是严格闭合的

因为

\[
x_\tau = \hat{y} + \sigma z_\tau
\]

且 `y_hat, sigma` 在一次采样轨迹内是条件固定量，不依赖当前 `x_\tau`，因此

\[
\frac{d x_\tau}{d\tau} = \sigma \frac{d z_\tau}{d\tau}
\]

所以若网络在 `z` 空间预测 `u_\theta`，则在 `x` 空间使用

\[
v_\theta = \sigma u_\theta
\]

是严格正确的。这也正是当前实现里 [src/nn/velocity_network.py](/workspace/NsDiff/src/nn/velocity_network.py) 的做法。

### 4. 条件概率几何被保留

`y_hat` 表示条件位置，`sigma` 表示条件尺度。使用它们做 normalization，本质上是让网络学习：

1. 当前状态距离条件均值有多远。
2. 这个偏差相对于条件不确定度有多大。

也就是说，网络处理的是“标准化残差”，而不是原始数值尺度。这与概率预测问题本身是一致的，因为最终样本质量取决于条件位置和条件尺度是否被正确建模。

## 方案二：使用 `x_tau.mean, x_tau.std` 的理论性质

### 1. 它与源分布构造不匹配

如果定义

\[
z_\tau^{(state)} = \frac{x_\tau - \mu(x_\tau)}{s(x_\tau)}
\]

则在 `\tau=0`：

\[
z_0 = \frac{X_0 - \mu(X_0)}{s(X_0)}
\]

这通常不等于 `\epsilon`，也不服从坐标独立的标准高斯。原因很简单：

1. `\mu(X_0)` 和 `s(X_0)` 是由 `X_0` 自己算出来的。
2. 它们会耦合不同预测步和不同变量。
3. 该变换把原本由 `y_hat, sigma` 定义的条件高斯结构抹成了“当前样本自己的白化表示”。

换言之，source distribution 在这个坐标系下失去了原始构造的可解释性。

### 2. 它不是固定条件坐标系，而是状态依赖坐标系

`x_tau.mean/std` 最大的问题不是“均值方差估计误差”，而是它们依赖当前状态本身。

于是坐标系随着轨迹移动而变化。此时：

\[
x_\tau = \mu(x_\tau) + s(x_\tau) z_\tau
\]

再对 `\tau` 求导，真实速度必须包含：

1. 均值漂移项 `d\mu/d\tau`
2. 尺度漂移项 `z_\tau ds/d\tau`
3. 标准化空间速度项 `s_\tau dz_\tau/d\tau`

而如果实现上仍然采用“预测 `u`，再乘 `std` 回去”的思路，那么实际上只保留了第 3 项，丢掉了前两项。

因此这种 normalization 与当前 velocity parameterization 并不闭合。

### 3. 它破坏了直线 flow 的简单结构

原问题里，`X_\tau` 是绝对空间中的线性插值，目标速度是常向量：

\[
v^* = Y - X_0
\]

但映到 `x_tau.mean/std` 坐标系后，由于坐标系本身在变，`z_\tau` 的轨迹一般不再是简单线性路径，目标速度也不再是简单常向量变换。

也就是说，原本简单的几何结构被重新扭曲了。

这类扭曲如果没有同时修正训练目标和逆变换公式，就属于理论上不自洽。

### 4. 它会弱化条件信息

`x_tau.mean/std` 使用的是当前状态本身的统计量，而不是条件预测分布的参数。这会带来一个重要后果：

1. `y_hat` 作为条件位置的作用被削弱。
2. `sigma` 作为条件不确定度的作用被削弱。
3. 网络更像是在做“当前轨迹自身白化后的动力学拟合”，而不是“从条件高斯先验流向目标分布”的动力学拟合。

从概率建模角度看，这偏离了 iReflow 当前的设计目标。

## 两种方案的直接对比

### 使用 `y_hat, sigma`

优点：

1. 与 `X_0 = y_hat + epsilon * sigma` 的定义完全对齐。
2. `tau=0` 时严格得到标准高斯坐标。
3. `z` 空间与 `x` 空间的速度关系是严格仿射变换。
4. 训练与采样阶段可使用同一套闭合公式。
5. 条件位置与条件尺度信息被完整保留。

代价：

1. 如果 `sigma` 本身估计很差，标准化后的残差会带噪。
2. 若 `sigma` 过小，数值稳定性需要额外处理，例如下界或 clamp。

### 使用 `x_tau.mean, x_tau.std`

优点：

1. 从表面上看，输入尺度可能更均匀。
2. 当 `sigma` 非常差时，训练早期可能看起来更稳定。

代价：

1. 与 source distribution 的定义不一致。
2. 坐标系依赖 `x_tau`，不是固定条件仿射变换。
3. 需要额外的导数项才能和真实速度严格对应。
4. 会扭曲原本线性的 rectified flow 几何结构。
5. 会弱化 `y_hat, sigma` 所表达的条件分布信息。

## 更严格的结论

在当前 iReflow 的数学设定下，`y_hat, sigma` normalization 是自洽的，`x_tau.mean, x_tau.std` normalization 不是。

更准确地说：

1. `y_hat, sigma` 属于条件固定的仿射坐标变换。
2. `x_tau.mean, x_tau.std` 属于状态依赖的自适应坐标变换。

对于前者，velocity field 的训练目标、推理更新、逆变换公式三者可以严格对齐。

对于后者，若仍沿用当前实现框架，就会出现理论缺口：你在 `z` 空间学到的速度，不能只靠乘回 `std` 就恢复成 `x` 空间的真实速度。

因此，如果问题是“哪一种更自洽”，答案很明确：

\[
\boxed{\text{使用 } (\hat{y}, \sigma) \text{ 的 normalization 更自洽}}
\]

## 这是否等价于“一定有更好的 CRPS”

不完全等价。

“更自洽”与“最终 CRPS 一定更优”之间仍有一步距离，因为实际效果还会受到以下因素影响：

1. `sigma` 的校准质量。
2. `sigma` 的数值稳定性。
3. `VelocityNetwork` 的容量。
4. 采样步数与 ODE 误差。
5. 数据集本身是否存在强烈的非高斯性或异方差偏差。

但是，在不改变当前 flow 定义和速度参数化的前提下，理论上更合理的基准选择仍然是 `y_hat, sigma`。

如果实验上它没有优于 `x_tau.mean/std`，首先应怀疑的是：

1. `sigma` 估计不准。
2. `sigma` 过小或过大导致训练不稳。
3. 采样器步数过少，使理论优势没有转化为实际分布质量。

而不是优先否定这种 normalization 的数学一致性。

## 对当前实现的建议

如果目标是在保持理论自洽的同时提高稳定性，优先建议以下方向，而不是改成 `x_tau.mean/std`：

1. 使用 `sigma_safe = clamp(sigma, min=eps)`。
2. 对 `sigma` 的上界和下界做更有针对性的约束。
3. 监控标准化残差 `(Y - y_hat) / sigma` 的分布是否接近稳定。
4. 检查 `sigma` 的 calibration 指标是否先于 flow 退化。
5. 若想引入统计量型 normalization，优先使用条件固定统计量，例如历史序列的 `mu_X, sigma_X`，而不是依赖 `x_tau` 的动态统计量。

最后一点很重要：固定条件统计量仍然属于“条件仿射变换”，理论上可以保持闭合；而 `x_tau.mean/std` 属于“状态依赖变换”，性质完全不同，不能混为一谈。

## 最终结论

在当前 iReflow 中：

1. `y_hat, sigma` normalization 与 source state、目标速度、ODE 采样和逆变换公式是统一的。
2. `x_tau.mean, x_tau.std` normalization 会把固定条件坐标系改成状态依赖坐标系，从而破坏这一统一性。

因此，从理论自洽性出发，应该优先选择：

\[
\boxed{
z_\tau = \frac{x_\tau - \hat{y}}{\sigma}
}
\]

而不是

\[
\boxed{
z_\tau = \frac{x_\tau - \mu(x_\tau)}{s(x_\tau)}
}
\]

前者是与当前 iReflow 设计一致的 normalization；后者若要成立，必须连同 flow 定义、速度目标和逆变换一起重写，而不能只替换输入标准化公式。
