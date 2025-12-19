# NsDiff 项目架构图

## 项目概述
NsDiff (Non-stationary Diffusion) 是一个基于扩散模型的概率时间序列预测框架，专门为非平稳场景设计。

## 目录结构架构图

```
NsDiff/
│
├── 📁 configs/                          # 配置文件目录
│   ├── csbi.yaml                        # CSBI模型配置
│   ├── nsdiff.yml                       # NsDiff主模型配置
│   └── tmdm.yml                         # TMDM模型配置
│
├── 📁 data/                             # 数据目录
│   └── ETTh1/                           # ETTh1数据集（自动下载）
│
├── 📁 docs/                             # 文档目录
│   ├── _static/                         # 静态资源
│   └── MODIFICATIONS.md                 # 修改记录
│
├── 📁 fig/                              # 图片资源目录
│   └── overview.jpg                     # 项目概览图
│
├── 📁 notebooks/                        # Jupyter笔记本目录
│   └── uncertainty_variation.ipynb      # 不确定性变化分析
│
├── 📁 results/                          # 实验结果目录
│   └── runs/                            # 运行结果
│       ├── F/                           # F模型结果
│       └── G/                           # G模型结果
│
├── 📁 scripts/                          # 脚本目录
│   ├── CSBI/                            # CSBI模型运行脚本
│   ├── CSDI/                            # CSDI模型运行脚本
│   ├── D3VAE/                           # D3VAE模型运行脚本
│   ├── DiffusionTS/                     # DiffusionTS模型运行脚本
│   ├── NSDiff/                          # NsDiff主模型运行脚本
│   │   ├── ETTh1.sh                     # ETTh1数据集脚本
│   │   ├── ETTh2.sh                     # ETTh2数据集脚本
│   │   ├── ETTm1.sh                     # ETTm1数据集脚本
│   │   ├── ETTm2.sh                     # ETTm2数据集脚本
│   │   └── ExchangeRate.sh              # ExchangeRate数据集脚本
│   ├── pretrain_F/                      # F模型预训练脚本
│   │   ├── ETTh1.sh
│   │   └── ETTm2.sh
│   ├── pretrain_G/                      # G模型预训练脚本
│   │   ├── ETTh1.sh
│   │   ├── ETTh2.sh
│   │   └── ETTm2.sh
│   ├── SSSD/                            # SSSD模型运行脚本
│   ├── TimeDiff/                        # TimeDiff模型运行脚本
│   ├── TimeGrad/                        # TimeGrad模型运行脚本
│   ├── run.sh                           # 通用运行脚本
│   └── run_wandb.sh                     # WandB集成运行脚本
│
├── 📁 src/                              # 源代码目录（核心）
│   ├── 📁 datasets/                     # 数据集模块
│   │   ├── __init__.py
│   │   ├── gaussian_ns.py               # 高斯非平稳数据集
│   │   ├── gaussian_ns1.py              # 高斯非平稳数据集变体1
│   │   ├── gaussian_ns2.py              # 高斯非平稳数据集变体2
│   │   └── gaussian_ns_test.py          # 高斯非平稳测试数据集
│   │
│   ├── 📁 experiments/                  # 实验模块（训练/评估入口）
│   │   ├── __init__.py
│   │   ├── NsDiff.py                    # NsDiff主实验脚本 ⭐
│   │   ├── NsDiff_PE.py                 # NsDiff位置编码版本
│   │   ├── pretrain_f.py                # F模型预训练脚本
│   │   ├── pretrain_g.py                # G模型预训练脚本
│   │   ├── prob_forecast.py             # 概率预测基类
│   │   ├── CSBI.py                      # CSBI实验
│   │   ├── CSDI.py                      # CSDI实验
│   │   ├── D3VAE.py                     # D3VAE实验
│   │   ├── DiffusionTS.py               # DiffusionTS实验
│   │   ├── DiffusionTS_nonoverlap.py    # DiffusionTS非重叠版本
│   │   ├── SSSD.py                      # SSSD实验
│   │   ├── TimeDiff.py                  # TimeDiff实验
│   │   ├── TimeGrad.py                  # TimeGrad实验
│   │   ├── TMDM.py                      # TMDM实验
│   │   └── TMDM1.py                     # TMDM变体实验
│   │
│   ├── 📁 layer/                        # 核心层模块
│   │   ├── denoise.py                   # 去噪层（条件引导模型）
│   │   ├── g_backbone.py                # G模型骨干网络
│   │   ├── mu_backbone.py               # μ模型骨干网络（Transformer）
│   │   └── nsdiff_utils.py              # NsDiff工具函数
│   │       ├── q_sample()               # 前向扩散采样
│   │       ├── p_sample_loop()          # 反向扩散采样循环
│   │       ├── cal_sigma12()            # 计算sigma1和sigma2
│   │       ├── cal_sigma_tilde()        # 计算tilde sigma
│   │       └── cal_forward_noise()      # 计算前向噪声
│   │
│   ├── 📁 metrics/                      # 评估指标模块
│   │   ├── __init__.py
│   │   ├── CRPS.py                      # 连续排名概率得分
│   │   ├── CRPSsum.py                   # CRPS求和版本
│   │   ├── PICP.py                      # 预测区间覆盖率
│   │   ├── ProbMAE.py                   # 概率平均绝对误差
│   │   ├── ProbMSE.py                   # 概率均方误差
│   │   ├── ProbRMSE.py                  # 概率均方根误差
│   │   └── QICE.py                      # 分位数区间覆盖误差
│   │
│   ├── 📁 models/                       # 模型定义模块
│   │   ├── __init__.py
│   │   ├── NsDiff.py                    # NsDiff主模型 ⭐
│   │   ├── CSBI.py                      # CSBI模型
│   │   ├── CSDI.py                      # CSDI模型
│   │   ├── D3VAE.py                     # D3VAE模型
│   │   ├── DiffusionTS.py               # DiffusionTS模型
│   │   ├── SSSD.py                      # SSSD模型
│   │   ├── TimeDiff.py                  # TimeDiff模型
│   │   ├── TimeGrad.py                  # TimeGrad模型
│   │   ├── TMDM.py                      # TMDM模型
│   │   └── TMDM1.py                     # TMDM变体模型
│   │
│   ├── 📁 nn/                           # 神经网络组件模块
│   │   ├── __init__.py
│   │   ├── csbi_*.py                    # CSBI相关组件（数据、扩散、损失、网络、策略、SDE、工具）
│   │   ├── d3vae_*.py                   # D3VAE相关组件（扩散、嵌入、编码器、操作、工具）
│   │   ├── diffusionts_*.py             # DiffusionTS相关组件（高斯扩散、模型工具、Transformer）
│   │   ├── tmdm_*.py                    # TMDM相关组件（扩散工具、模型、非平稳Transformer）
│   │   ├── DDPM_CNNNet.py               # DDPM CNN网络
│   │   ├── diffusion_worker.py         # 扩散工作器
│   │   ├── dpm_sampler.py               # DPM采样器
│   │   ├── dpm_solver.py                # DPM求解器
│   │   ├── resnet.py                    # ResNet组件
│   │   ├── s4model.py                   # S4模型
│   │   ├── SSSDS4Imputer.py             # SSSD S4填充器
│   │   └── SSSDSAImputer.py             # SSSD SA填充器
│   │
│   └── 📁 utils/                        # 工具函数模块
│       ├── __init__.py
│       ├── diffusion_output.py          # 扩散输出处理
│       ├── diffusion_utils.py           # 扩散工具函数
│       ├── gaussian_diffusion.py        # 高斯扩散实现
│       └── sigma.py                     # Sigma计算工具
│           ├── wv_sigma()               # 小波方差sigma计算
│           └── wv_sigma_trailing()      # 拖尾小波方差sigma计算
│
├── README.md                            # 项目说明文档
├── requirements.txt                     # Python依赖包列表
└── ARCHITECTURE.md                      # 本架构文档

```

## 核心模块说明

### 1. 实验流程模块 (src/experiments/)
- **NsDiff.py**: 主实验脚本，包含训练、验证、测试的完整流程
- **pretrain_f.py / pretrain_g.py**: F和G模型的预训练脚本
- **prob_forecast.py**: 概率预测实验的基类，定义了通用的实验框架

### 2. 模型定义模块 (src/models/)
- **NsDiff.py**: 核心模型实现，包含：
  - 前向扩散过程（q_sample）
  - 反向扩散过程（p_sample_loop）
  - 非平稳性处理（sigma计算）
  - F和G两个子模型的集成

### 3. 核心层模块 (src/layer/)
- **mu_backbone.py**: μ（均值）模型的Transformer骨干网络
- **g_backbone.py**: G模型的骨干网络
- **denoise.py**: 条件引导的去噪模型
- **nsdiff_utils.py**: NsDiff的核心工具函数，实现扩散过程的数学计算

### 4. 工具模块 (src/utils/)
- **sigma.py**: 非平稳性sigma的计算，使用小波方差方法
- **gaussian_diffusion.py**: 高斯扩散过程的基础实现
- **diffusion_utils.py**: 扩散模型的通用工具函数

### 5. 评估指标模块 (src/metrics/)
- 提供多种概率预测评估指标：
  - CRPS（连续排名概率得分）
  - PICP（预测区间覆盖率）
  - ProbMAE/MSE/RMSE（概率误差指标）
  - QICE（分位数区间覆盖误差）

### 6. 数据集模块 (src/datasets/)
- 提供高斯非平稳时间序列数据集的生成和加载

### 7. 脚本模块 (scripts/)
- **NSDiff/**: NsDiff模型在不同数据集上的运行脚本
- **pretrain_F/**: F模型预训练脚本
- **pretrain_G/**: G模型预训练脚本
- 其他目录包含各种baseline模型的运行脚本

## 工作流程

```
1. 数据准备
   └──> data/ETTh1/ (自动下载)

2. 预训练阶段（可选）
   ├──> scripts/pretrain_F/ETTh1.sh
   └──> scripts/pretrain_G/ETTh1.sh

3. 主训练/测试阶段
   └──> scripts/NSDiff/ETTh1.sh
       └──> src/experiments/NsDiff.py
           ├──> 加载模型: src/models/NsDiff.py
           ├──> 使用层: src/layer/*.py
           ├──> 计算指标: src/metrics/*.py
           └──> 保存结果: results/runs/

4. 结果分析
   └──> results/runs/
```

## 关键依赖关系

```
NsDiff实验流程
    │
    ├──> NsDiff模型 (src/models/NsDiff.py)
    │       ├──> μ骨干网络 (src/layer/mu_backbone.py)
    │       ├──> G骨干网络 (src/layer/g_backbone.py)
    │       ├──> 去噪模型 (src/layer/denoise.py)
    │       └──> 扩散工具 (src/layer/nsdiff_utils.py)
    │               └──> Sigma计算 (src/utils/sigma.py)
    │
    ├──> 数据集加载 (src/datasets/*.py)
    │
    ├──> 评估指标 (src/metrics/*.py)
    │
    └──> 配置文件 (configs/nsdiff.yml)
```

## 技术特点

1. **非平稳性处理**: 通过小波方差方法计算时变的sigma参数
2. **双模型架构**: F模型和G模型分别处理不同的扩散过程
3. **概率预测**: 输出概率分布而非点预测
4. **多数据集支持**: ETTh1/ETTh2/ETTm1/ETTm2/ExchangeRate等
5. **Baseline对比**: 包含CSBI、CSDI、D3VAE、DiffusionTS等多种baseline实现

## 运行方式

### 方式1: 带预训练
```bash
# 1. 预训练F模型
bash ./scripts/pretrain_F/ETTh1.sh

# 2. 预训练G模型
bash ./scripts/pretrain_G/ETTh1.sh

# 3. 运行主实验
bash ./scripts/NSDiff/ETTh1.sh
```

### 方式2: 无预训练
```bash
# 直接运行（不使用预训练）
bash ./scripts/NSDiff/ETTh1.sh
```

## 输出结果

实验结果保存在 `results/runs/` 目录下，按模型类型和数据集组织：
- `results/runs/F/ETTh1/` - F模型结果
- `results/runs/G/ETTh1/` - G模型结果
- `results/runs/NsDiff/ETTh1/` - NsDiff完整模型结果

