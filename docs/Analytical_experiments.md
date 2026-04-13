# 分析性实验

很好，你现在这三个实验已经形成一条非常完整的机制链了：

- **实验1**：说明 **PDN 在预测目标上做了什么分解**
- **实验2**：说明 **这种分解如何改变 transport 的几何形状**
- **实验3**：说明 **这种改变在什么条件下最有价值**

这和你文中的方法叙事是严格对齐的：原空间里模型同时承担宏观 shift 和微观 stochastic transport 的双重负担；PDN 先用未来统计量构造 predictive prior，再把问题变到 stationary normalized space，让网络专注于 micro-level dynamics。

下面我直接按“**论文里每张图怎么画**”来细化。

------

# 总体建议：三张图的角色分工

我建议主文中三张图分别承担不同任务，不要互相重复：

**Figure 1. Decoupling case study**
让审稿人一眼看到：**raw future 的差异主要来自 macro statistics，而不是 residual dynamics 本身。**

**Figure 2. PCA transport path**
让审稿人一眼看到：**PDN 让同一个 transport 问题从分散、异质的物理空间，变成更集中、更统一的 canonical space。**

**Figure 3. Synthetic drift stress test**
让审稿人一眼看到：**PDN 的优势不是普遍“玄学增益”，而是在 non-stationarity 强时明显放大。**

------

# 实验1：Decoupling case study

## 先回答你的关键问题：单数据集还是跨数据集？

**主文建议只用单个数据集，不要跨数据集。**

原因很简单：这个图的目标是证明 **decouple**，不是证明 **generalization across datasets**。
如果跨数据集，视觉冲击虽然会更大，但很容易混入无关因素：

- 单位不同
- 量纲不同
- seasonality 不同
- horizon 上的典型形状不同

这样审稿人看到 raw 轨迹差很多，未必会把它理解成“macro shift 被 decouple”，而可能只会觉得“这些本来就不是同一种数据”。

**所以最强的主图应该是：同一数据集、同一变量、不同测试窗口。**

这样你图里的差异只能来自：

- 不同 regime
- 不同时间段的 mean/scale drift

而不是来自 dataset semantics。

## 选哪个数据集最适合？

我建议：

- **主图用 Traffic**
- **附录复现用 ECL**
- Weather 可以作为备选

为什么是 Traffic：

- 宏观 level/scale 变化通常更明显，视觉冲击强；
- 同时局部 pattern 往往还有可识别的相似性，适合展示“macro 不同，但 residual dynamics 相似”；
- 比 EXG 更有结构感，比 ETT 更有 regime contrast。

**不建议主图用跨数据集拼盘。**
如果你非常想体现“不是个例”，可以在附录放一个 2×2 montage：

- Traffic
- ECL
- Weather
- ETTm2
  但主文只放一个最有冲击力的。

------

## 样本怎么选才最强？

不要随机选。
也不要纯人工挑。
最稳妥的方法是 **“规则化筛选 + 视觉代表性”**：

### 第一步：先筛高 drift 窗口

对测试集每个窗口，计算一个 drift score：
$$
d_i = \frac{|\mu^{f}_i - \mu^{p}_i|}{\sigma^{p}_i + \epsilon}

+ \left| \log \sigma^{f}_i - \log \sigma^{p}_i \right|
$$
其中：

- $\mu^p_i, \sigma^p_i$：过去窗口统计量
- $\mu^f_i, \sigma^f_i$：未来窗口真实统计量

先取 **top 10% 或 top 20% drift windows**。

### 第二步：在高 drift 里选“residual shape 相近”的样本

因为你想展示 decouple，所以要选那种：

- raw future 看起来差别很大
- 但做标准化后 shape 更相近

一个简单做法是：

- 先对 future 做 oracle standardization：$(X^f-\mu^f)/\sigma^f$
- 在高 drift 候选集中，找 pairwise correlation 较高的一组窗口
- 选 6 个样本

这样做的好处是：

- raw 图会非常“散”
- normalized residual 图会明显“收拢”
- 又不是主观 cherry-pick，因为你有明确规则

------

## Figure 1 的具体 layout

我建议做成 **3-panel 横排图**，不要做太多小图。

### Panel (a): Raw future trajectories

画 6 个选中的 future windows 叠加图。

- **横轴**：forecast horizon step
- **纵轴**：raw target value
- 每条轨迹一个固定颜色
- 不画 past，只画 future，避免视觉过载

你希望这张图一眼看上去：

- level 差异大
- amplitude 差异大
- 看起来像“不是一个共同问题”

### Panel (b): Predicted macro components

对同样 6 个样本，画：

- $\hat{\mu}$ 为主线
- $\hat{\mu}\pm\hat{\sigma}$ 为浅色 band
- **横轴**：forecast horizon step
- **纵轴**：raw target value
- 颜色与 Panel (a) 保持一致

这张图不是为了说预测多准，而是为了让审稿人看到：
**每个实例先被赋予了自己的 macro coordinate system。**

### Panel (c): PDN-normalized residual futures

画同样 6 个样本的：
$$
Z_1 = (X_1-\hat{\mu}) \oslash \hat{\sigma}
$$

- **横轴**：forecast horizon step
- **纵轴**：normalized value
- 仍用同一组颜色

你希望它看起来：

- 明显比 Panel (a) 更对齐
- level 差异消失
- amplitude 差异显著减弱
- 只剩下局部 fluctuation pattern 的差别

------

## Figure 1 的标题建议

**Macro–Micro Decoupling on Real Forecast Windows**

或者更直接一点：

**PDN Separates Macro Statistics from Residual Forecast Dynamics**

------

## Figure 1 的 caption 草稿

> **Figure X: PDN decouples macro shifts from residual dynamics on real forecasting windows.** We visualize six high-drift test windows from the Traffic dataset, selected by a deterministic drift criterion and then matched for residual-shape similarity. In raw space (left), the future trajectories exhibit large instance-specific shifts in level and scale. The predicted macro components $\hat{\mu}$ and $\hat{\mu}\pm\hat{\sigma}$ (middle) capture these shifts explicitly. After predictive distribution normalization (right), the same futures collapse into a substantially more aligned canonical form, indicating that the remaining modeling burden lies primarily in residual stochastic dynamics rather than in instance-specific mean/variance drift.

------

# 实验2：PCA transport path

## 数据集和样本怎么选？

**建议直接和实验1用同一个主数据集：Traffic。**

这样三张图叙事会非常连贯：

- Figure 1：你看到 raw future 被 decouple
- Figure 2：你看到 transport geometry 因此改变
- Figure 3：你看到在 drift 强时收益放大

如果 Figure 2 突然换到别的数据集，故事会断。

## 样本选法

这里不要再手挑 6 个样本。
你需要一个 **群体几何图**，所以应该用更多窗口：

- 从 Traffic 测试集选 **200–400 个窗口**
- 仍然用 drift score 先筛 top 20%
- 然后从中随机采样固定数量，例如 256 个窗口

这样：

- 有足够路径形成 bundle
- 同时是“高 drift 子集”，视觉对比会更明显
- 又不会被质疑只挑了少数极端 case

## 这张图里用什么 path？

用你文中定义的同一个 predictive prior 和同一个 OT path：

[
X_0 \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2 I), \quad
X_\tau = \tau X_1 + (1-\tau)X_0
]

然后做 PDN 变换得到：

[
Z_\tau = (X_\tau-\hat{\mu}) \oslash \hat{\sigma}
]

也就是：

- **左图 raw path**：$X_\tau$
- **右图 PDN path**：$Z_\tau$

这两个 panel 使用：

- 同一批 test windows
- 同一个 sampled noise $\epsilon$
- 同一组 $\tau \in {0,0.25,0.5,0.75,1}$

这样图里差异就可以完全归因于 **coordinate reparameterization / decoupling**，而不是别的因素。

------

## Figure 2 的具体 layout

我建议做成 **2 个主 panel + 1 个小辅助 panel**。

### Panel (a): Raw-space transport paths

- 对每个窗口、每个 $\tau$，把 $X_\tau$ flatten 成一个向量
- 在所有 $X_\tau$ 上做 PCA 到 2D
- 画 path bundle

具体画法：

- 每条路径是一个窗口
- 起点 $\tau=0$ 用圆点
- 终点 $\tau=1$ 用三角形或星形
- 中间状态用线连接
- 路径颜色从浅到深表示 $\tau$

**横轴**：PC1
**纵轴**：PC2

你希望视觉效果是：

- 路径方向很乱
- 扇形发散
- 不同实例长度差异明显

### Panel (b): PDN-space transport paths

- 对所有 $Z_\tau$ 重复同样 PCA 可视化

**横轴**：PC1
**纵轴**：PC2

你希望视觉效果是：

- 路径更集中
- 走向更统一
- 起点更集中在标准噪声附近
- 整体更像一个 canonical transport family

### Panel (c): Dispersion along flow time

这个 panel 很小，但我建议加。
因为它能把“视觉印象”变成一个非常轻量的量化支持，而不显得像表格。

定义例如：

[
D(\tau)=\frac{1}{N(N-1)}\sum_{i\neq j}|S_\tau^{(i)}-S_\tau^{(j)}|_2
]

其中 $S_\tau$ 可以是 $X_\tau$ 或 $Z_\tau$。

- **横轴**：flow time $\tau$
- **纵轴**：average pairwise distance
- 两条线：raw space / PDN space

你希望看到：

- raw line 全程更高
- PDN line 更低、更平滑

这个 panel 会非常有用，因为它一句话就把图的观感锁定成“path family 变得更统一”。

------

## PCA 图的一个关键实现细节

**不要把 raw 和 PDN 放在同一个 PCA 投影里。**
分别做 PCA 更合理。

因为：

- 原始空间和归一化空间的几何尺度本来就不同
- 共用一个 PCA 基底反而容易扭曲视觉

caption 里写清楚：

- each panel uses an independent PCA fit for visualization
- both panels use the same windows and the same sampled priors

就可以了。

------

## Figure 2 的标题建议

**PDN Reparameterizes the Same Transport into a Canonical Space**

------

## Figure 2 的 caption 草稿

> **Figure Y: PDN reparameterizes the same transport problem into a more canonical geometry.** We sample predictive priors $X_0 \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2 I)$ and construct the same OT paths for the same high-drift test windows from Traffic. Left: paths visualized in the raw physical space $X_\tau$. Right: the same paths after PDN, i.e., $Z_\tau=(X_\tau-\hat{\mu})/\hat{\sigma}$. While raw-space paths are widely dispersed and directionally heterogeneous due to instance-specific mean/scale drift, PDN-space paths become markedly more concentrated and regular. The small panel quantifies this effect by showing reduced inter-instance dispersion along flow time, consistent with the claim that PDN converts non-stationary transport into a more standard OT problem in a stationary normalized space.

------

# 实验3：Synthetic drift stress test

## 先回答你的问题：sample ribbon 还是 sample trajectories？

如果你只能二选一，**选 sample trajectories**。
因为这个实验左图的任务不是“精确展示统计区间”，而是 **让审稿人感受到 drift 越来越强**。

- ribbon 更整洁
- trajectories 更有“运动感”和“冲击力”

**最好的方案其实是 hybrid**：

- 画 15–20 条透明 sample trajectories
- 再叠加一条较粗的均值线
  不要只画 ribbon。

这样既有视觉冲击，又不会太乱。

------

## synthetic 数据怎么设计最合适？

这个实验最关键的是：
**局部随机动态固定，只改宏观 drift 强度。**

我建议做 3 种 stress：

1. **Mean drift only**
2. **Scale drift only**
3. **Joint mean + scale drift**

你方法同时预测 $\hat{\mu}$ 和 $\hat{\sigma}$，所以这三种都应该测。

------

## Figure 3 的具体 layout

我建议做成 **左边示意 + 右边双曲线图** 的结构。

### 左侧 block：ground-truth future illustrations

做成 4 个小面板横排，表示 drift strength 递增：

- $\delta = 0$
- $\delta = 1$
- $\delta = 2$
- $\delta = 3$

每个 panel 里画：

- 15 条透明 future trajectories
- 1 条粗均值线
- 可以把 past context 的最后一小段用灰色接在左侧，帮助审稿人看到“future 开始发生 drift”

**横轴**：forecast horizon step
**纵轴**：synthetic value

如果版面足够，左侧 block 只画 **joint drift** 就够了，因为视觉最强。
右侧性能图再分 mean / scale / joint 三种情况。

### 右侧 block：performance vs drift strength

我建议分成两个小 panel，避免一张图过于拥挤：

#### Panel (e): Mean-drift stress

- **横轴**：drift strength
- **纵轴**：CRPS 或 NLL
- 三条线：Raw Flow / Historical Norm / PDN

#### Panel (f): Scale-drift stress

同样画法。

如果你还想更完整，可以在 appendix 再给 **joint-drift** 曲线。
如果主文空间足够，右侧就直接放三张小曲线：

- mean drift
- scale drift
- joint drift

------

## Figure 3 的核心实验设置建议

为了避免审稿人质疑 synthetic 太随意，caption 或正文里要明确写：

- local stochastic mechanism fixed
- only macro drift strength varies
- all methods use the same backbone capacity and training budget

这样读者会很自然地接受：
曲线差异就是由 non-stationarity handling 导致的。

------

## Figure 3 的标题建议

**PDN Helps Most Under Strong Non-Stationarity**

------

## Figure 3 的 caption 草稿

> **Figure Z: PDN is most beneficial when non-stationarity becomes severe.** Left: synthetic future trajectories under increasing joint drift strength, where the local stochastic mechanism is fixed but the magnitude of mean/scale drift increases. Right: forecasting performance as a function of drift strength. Raw generative modeling degrades rapidly as macro drift intensifies, historical normalization offers only partial relief, and PDN remains substantially more stable across both mean-drift and scale-drift stress tests. This confirms that PDN’s gains come specifically from decoupling macro distribution shifts from residual stochastic transport, rather than from a generic modeling advantage.

------

# 三张图在正文里的顺序和过渡句

这个也很重要。
建议按下面顺序写，逻辑非常顺：

### Figure 1 先出

“我们先看 PDN 对预测目标本身做了什么分解。”

### Figure 2 接着出

“这种分解不仅改变了目标表示，也改变了 transport problem 的几何结构。”

### Figure 3 最后出

“因此，当 non-stationarity 真正成为瓶颈时，这种重参数化的优势会系统性放大。”

这三句过渡会让审稿人感觉这是一个完整故事，而不是三张零散的分析图。

------

# 最后给你一个直接可执行的版本

## Figure 1

- **数据集**：Traffic
- **样本**：同一变量的 6 个高 drift 测试窗口
- **布局**：Raw futures | $\hat{\mu}\pm\hat{\sigma}$ | PDN-normalized futures

## Figure 2

- **数据集**：Traffic
- **样本**：top 20% drift windows 中随机采样 256 个
- **布局**：Raw-space PCA paths | PDN-space PCA paths | dispersion-vs-$\tau$ 小图

## Figure 3

- **数据**：synthetic
- **布局**：4 个 drift-level trajectory panels | 2 或 3 个 performance-vs-drift line charts
- **左图风格**：优先 trajectories，不用纯 ribbon

------

我最推荐你现在就把 **Figure 1 的样本筛选规则** 和 **Figure 2 的 path construction 流程** 先定死，因为这两张图最依赖实现细节。下一步我可以继续把这三张图整理成 **论文中的正式小节写法**，包括每个实验该怎么用一段话介绍、结果分析应该怎么写。