# 为什么对 `sigma` 做 Clamp 反而让 Stage 3 的 CRPS 更好

## 现象

在当前实现中，`sigma` 在 [src/models/iReflow.py](/workspace/NsDiff/src/models/iReflow.py) 的 [142-148 行](/workspace/NsDiff/src/models/iReflow.py#L142) 被约束：

```python
if self.itransformer.use_norm:
    scale = stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
    sigma = sigma * scale
    min_sigma = torch.full_like(sigma, 1e-3)
    sigma = torch.clamp(sigma, min=min_sigma, max=2.0 * scale)
else:
    sigma = torch.clamp(sigma, min=1e-3)
```

你的观测是：

1. 做 `clamp` 后，`train_max_sigma` 通常被限制在 `20` 以内，`train_mean_sigma` 在 `1.0` 以内。
2. 不做 `clamp` 时，`train_max_sigma` 经常到 `100-200`，`train_mean_sigma` 在 `1.5` 左右。
3. 最终 Stage 3 的评估结果显示，做 `clamp` 的 run 有更低的 CRPS，也就是概率预测更好。

这不是一个“数值更稳定，所以效果偶然更好”的小现象，而是和当前 iReflow 中 `sigma` 的角色有关。

## 先给结论

在当前 iReflow 里，`sigma` 不是一个只用于日志或只用于高斯 NLL 的辅助量。它同时控制了：

1. Source state `X_0` 的扩散半径。
2. Velocity network 输入的相对坐标缩放。
3. Velocity network 输出回到绝对空间时的速度缩放。

因此，过大的 `sigma` 会同时把 Stage 3 的三个环节推向不利方向：

1. 初始分布过宽，source state 离真实目标分布太远。
2. 标准化后有用的条件残差信号被压小。
3. 输出误差又会被 `sigma` 放大回绝对空间。

从这个角度看，`clamp` 的作用不是简单“防止爆炸”，而是在给 Stage 3 加一个关于 transport geometry 的先验约束：不要让 source distribution 比真实 transport 所需的尺度宽太多。

这能直接解释为什么 `clamp` 后 CRPS 反而更好。

## 从代码看，`sigma` 在 Stage 3 里影响了什么

### 1. `sigma` 决定 Source State 的扩散半径

在 [src/models/iReflow.py:81-85](/workspace/NsDiff/src/models/iReflow.py#L81)：

\[
X_0 = \hat{y} + \epsilon \sigma,\quad \epsilon \sim \mathcal{N}(0, I)
\]

也就是说，`sigma` 越大，采样起点 `X_0` 离 `y_hat` 越远，source distribution 越分散。

### 2. `sigma` 决定输入给 velocity net 的相对坐标

在 [src/nn/velocity_network.py:195-197](/workspace/NsDiff/src/nn/velocity_network.py#L195)：

\[
z_\tau = \frac{x_\tau - \hat{y}}{\sigma}
\]

这意味着 `sigma` 越大，绝对空间里同样大小的残差，在网络看到的相对空间里就越小。

### 3. `sigma` 决定输出速度映回绝对空间的尺度

在 [src/nn/velocity_network.py:225-226](/workspace/NsDiff/src/nn/velocity_network.py#L225)：

\[
v_\theta = \sigma u_\theta
\]

这意味着网络在相对空间里犯下的误差，回到绝对空间时会被 `sigma` 放大。

## 大 `sigma` 为什么会伤害 Stage 3

下面是最核心的部分。

### 1. 大 `sigma` 会把 `X_0` 推得过远，transport 任务更难

Stage 3 的目标速度在 [src/models/iReflow.py:221-225](/workspace/NsDiff/src/models/iReflow.py#L221) 是：

\[
v^* = Y - X_0 = (Y - \hat{y}) - \sigma \epsilon
\]

如果 `sigma` 很大，那么 `v^*` 中很大一部分幅度来自 `-\sigma \epsilon`，而不是来自真正有意义的条件残差 `(Y-\hat{y})`。

这会带来两个结果：

1. 训练标签的绝对尺度被人为拉大。
2. 模型要花更多能力去“把随机噪声拉回来”，而不是去拟合真实数据分布的细节。

换句话说，Stage 3 本来应该学习“从合理的条件分布微调到真实分布”，但当 `sigma` 过大时，它变成了“先从一个过于发散的起点做大幅度回拉，再顺便拟合真实分布”。

这对有限容量、有限采样步数的 flow 来说通常是不利的。

### 2. 大 `sigma` 会降低标准化空间中的信号强度

把目标速度除以 `sigma`，可以得到相对空间中的目标：

\[
u^* = \frac{v^*}{\sigma}
= \frac{Y - \hat{y}}{\sigma} - \epsilon
\]

这里有一个很重要的信噪比问题：

1. `-\epsilon` 是噪声主导项。
2. `\frac{Y-\hat{y}}{\sigma}` 是真正反映条件误差结构的有用项。

当 `sigma` 变大时：

\[
\frac{Y-\hat{y}}{\sigma} \to 0
\]

于是 `u^*` 越来越接近纯噪声驱动的 `-\epsilon`。

这意味着 velocity network 学到的东西会越来越像“把噪声往回收”而不是“根据条件信息把样本送到正确的未来分布”。

这是为什么过大的 `sigma` 会让最终样本分布质量下降，即便训练损失可能仍然在下降。

### 3. 大 `sigma` 会造成“输入被压小，输出被放大”的不对称

当前参数化里：

1. 输入是除以 `sigma` 的：
   \[
   z_\tau = \frac{x_\tau-\hat{y}}{\sigma}
   \]
2. 输出是乘回 `sigma` 的：
   \[
   v_\theta = \sigma u_\theta
   \]

这会带来一个典型的不对称现象：

1. 有用的绝对残差进入网络时被压小。
2. 网络在相对空间中的小误差，回到绝对空间时被放大。

如果某些变量或时间步的 `sigma` 特别大，那么这些位置的优化会变差，因为它们既削弱了输入信号，又放大了输出误差。

`clamp` 的作用就是防止这种不对称被极端值主导。

### 4. 大 `sigma` 让少数极端位置主导训练

从训练统计上看，`train_max_sigma` 到 `100-200`，而平均值只有 `1.5` 左右，这意味着问题不是“整体 sigma 略大”，而是“存在少量极端大 sigma”。

这种长尾会带来两个问题：

1. 少数位置生成极远的 `X_0`，打乱 batch 内的几何结构。
2. 少数位置会贡献特别大的速度标签和梯度，扭曲整体优化方向。

所以 `clamp` 实际上是在去掉这些极端点对 flow 训练的劫持。

## 为什么这件事会反映在 CRPS 上

CRPS 是对整条预测分布的打分，不只是看点预测。

如果不做 `clamp`，那么大的 `sigma` 会让 Stage 3 的样本生成出现下面的问题：

1. 初始样本过于分散。
2. ODE 采样步数有限，无法把这些过于分散的样本充分 transport 回正确区域。
3. 网络更偏向于做大尺度 denoising，而不是做细粒度分布校准。

结果通常就是：

1. 样本分布的尾部更重或位置更散。
2. 样本之间的相对结构不够贴近真实条件分布。
3. 虽然多样性可能看起来更大，但这种多样性不一定是“有用的不确定性”。

CRPS 会惩罚这种“分布虽然宽，但不够准”的情况。

相反，做 `clamp` 后：

1. `X_0` 更靠近 `y_hat`。
2. Stage 3 的任务更像是局部 transport，而不是大幅回拉。
3. 有限容量的 velocity net 更容易把样本精细地送到真实分布附近。

因此最终 CRPS 反而更低。

## 为什么这并不和“高斯 sigma 指标更好”矛盾

这个问题尤其值得单独说清楚。

很多时候，不做 `clamp` 的 `sigma` 可能在某些高斯校准指标上更好，或者至少看起来更“保守”。但 Stage 3 的最终目标不是直接输出高斯分布 `N(\hat{y}, \sigma^2)`，而是：

1. 用这个分布生成 `X_0`。
2. 再通过 learned flow 把它 transport 成最终样本分布。

也就是说，Stage 3 需要的不是“单独作为高斯预测最优的 sigma”，而是“作为 source prior 最适合 transport 的 sigma”。

这两者并不等价。

一个过宽、过保守的 `sigma` 可能作为静态高斯预测并不差，但作为 flow 的 source prior 反而太难 transport。

从 flow 的角度看，source prior 需要满足的是：

1. 足够覆盖真实未来。
2. 但不要宽到让 transport 变成大范围的去噪问题。

`clamp` 恰好是在把 `sigma` 推向这个更适合 Stage 3 的区间。

## 结合 2026-04-09 的运行记录看这个现象

下面只看同一数据集 `Weather`、同一 Stage 3 设置下的 run。

### 未 Clamp 的 Stage 3 run

- `wandb/run-20260409_073952-cqrxzci6`
  - `train_max_sigma = 125.4468`
  - `train_mean_sigma = 1.9766`
  - `best_test_crps = 0.21896`
  - `best_test_picp = 0.44920`
  - `best_test_qice = 0.08296`

- `wandb/run-20260409_074628-8axt11va`
  - `train_max_sigma = 125.4126`
  - `train_mean_sigma = 1.9468`
  - `best_test_crps = 0.21774`
  - `best_test_picp = 0.46561`
  - `best_test_qice = 0.08022`

### Clamp 后的 Stage 3 run

- `wandb/run-20260409_080231-cperbbmb`
  - `train_max_sigma = 11.9257`
  - `train_mean_sigma = 0.6007`
  - `best_test_crps = 0.20545`
  - `best_test_picp = 0.60379`
  - `best_test_qice = 0.05493`

- `wandb/run-20260409_083725-z59da2vo`
  - `train_max_sigma = 17.8811`
  - `train_mean_sigma = 0.7466`
  - `best_test_crps = 0.20503`
  - `best_test_picp = 0.59894`
  - `best_test_qice = 0.05503`

这一组结果说明了两件事：

1. `clamp` 并不是只让训练数值“看起来正常”，它实实在在改善了最终的 sample-based 指标。
2. 改善不只体现在 CRPS，`PICP` 和 `QICE` 也同步改善，说明这不是纯粹的“把分布变尖锐”造成的单指标假象。

这更支持“`clamp` 改善了 flow 的 transport 条件和样本几何结构”这一解释。

## 对这个现象的更准确理解

更准确地说，`clamp` 在当前框架里发挥了四层作用。

### 1. 它是一个 source prior regularizer

`sigma` 决定 `X_0` 的分布宽度。对 `sigma` 做 `clamp`，本质上是在给 source prior 加先验：

1. 允许不确定性存在。
2. 但不允许它无限制扩散。

### 2. 它改善了 Stage 3 的信噪比

在相对空间里，训练目标是

\[
u^* = \frac{Y-\hat{y}}{\sigma} - \epsilon
\]

`sigma` 过大时，有用信号被稀释。`clamp` 提高了有用项在目标中的相对权重。

### 3. 它降低了优化中的梯度长尾

`train_max_sigma` 从 `100+` 压到 `20` 以内，说明大量极端值被处理掉了。这样训练不再被少数异常位置主导。

### 4. 它让有限步数 ODE 更容易工作

当前 Stage 3 用的是有限步 Euler 采样，见 [src/models/iReflow.py:286-306](/workspace/NsDiff/src/models/iReflow.py#L286)。

如果 source distribution 太宽，有限步数的 transport 很难把样本精细地拉回目标区域。`clamp` 把问题变成更局部的 transport，因此更适合当前采样器和网络容量。

## 这说明了什么

这个现象说明：在当前 iReflow 中，`sigma` 的最优取值不应当只按“高斯校准误差”来理解，而应当按“是否有利于后续 flow transport”来理解。

也就是说，Stage 2 输出的 `sigma` 在进入 Stage 3 时，身份已经发生了变化：

1. 它仍然是 uncertainty estimate。
2. 但它同时也是 flow 的几何尺度参数。

一旦 `sigma` 同时承担这两个角色，就不能简单追求“尽量大、尽量保守”。

在当前框架下，一个稍微保守过头的 `sigma`，很可能会恶化 Stage 3；而一个被约束过的、更局部的 `sigma`，反而会让最终 sample distribution 更好。

## 实践上的含义

基于这个现象，后续不应把 `clamp` 理解为临时补丁，而应把它看作一个正式的建模组件。

更具体地说：

1. `sigma` 的上界应该作为超参数认真调，而不是默认放开。
2. 需要同时看三类指标：
   - `sigma_*` 指标
   - Stage 3 的 `train_max_sigma / train_mean_sigma`
   - 最终 sample-based `CRPS / PICP / QICE`
3. 如果 `sigma_gauss_nll` 变好但 Stage 3 `CRPS` 变差，不要惊讶，这说明“高斯先验更保守”不等于“flow 样本更好”。

## 建议的后续验证

如果要把这个解释做得更扎实，建议继续记录以下统计量：

1. `|(Y-\hat{y})/\sigma|` 的均值和分位数。
   - 这能直接看出有用残差项是否被 `sigma` 压得太小。
2. `|v_target|` 的均值和分位数。
   - 这能直接看出大 `sigma` 是否在抬高 transport 难度。
3. `X_0` 到 `Y` 的距离分布。
   - 这能验证 `clamp` 是否真的缩短了 transport 路径。
4. `u_theta` 的误差分布和 `v_theta` 的误差分布。
   - 这能验证“输入压缩、输出放大”的不对称是否在影响训练。

## 最终结论

对 `sigma` 做 `clamp` 后 Stage 3 的 CRPS 更好，最合理的解释不是“模型更保守了”，而是：

1. `clamp` 把 source distribution 从“过于发散”拉回到“便于 transport”的尺度。
2. 它提高了相对空间里的有效信号强度。
3. 它降低了极端大 `sigma` 对训练的劫持。
4. 它让有限容量、有限步数的 flow 更容易把样本送到正确分布。

因此，在当前 iReflow 中，`sigma clamp` 更像是一个改善 transport geometry 的正则化，而不只是一个数值稳定性技巧。
