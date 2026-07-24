# Repository Notes

- 本仓库的核心模型是 **LS-Flow（iReflow）**，用于概率时间序列预测。
- 运行实验使用 conda 环境：`NsDiff`。
  - 示例：`conda run -n NsDiff python ...`
- LS-Flow/iReflow 当前有两种训练方式：
  1. **三阶段训练**：分别训练点预测/encoder、uncertainty estimator（sigma）和 flow generator；最后阶段冻结前两部分，只训练生成器。
  2. **联合训练（end-to-end）**：同时优化 encoder、mu、sigma 与 flow generator。
- 复现实验时，优先复用已指定的 checkpoint、数据划分、ODE 设置和评测采样数；不要用测试集选择 checkpoint 或 seed。
