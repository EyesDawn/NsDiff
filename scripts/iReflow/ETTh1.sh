#!/bin/bash

# iReflow运行脚本 - ETTh1数据集
# 使用Rectified Flow进行概率时间序列预测

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 数据集配置
DATASET="ETTh1"
DATA_PATH="./data/"

# 模型配置
D_MODEL=512
N_HEADS=8
E_LAYERS=2        # iTransformer编码器层数
FLOW_LAYERS=3     # Velocity Network层数
D_FF=2048
DROPOUT=0.1

# 训练配置
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=100
PATIENCE=10

# Flow配置
NUM_SAMPLING_STEPS=1    # 1表示one-step generation（极快）
TEMPERATURE=1.0         # 采样温度
NUM_SAMPLES=100         # 测试时生成的样本数

# 预测配置
WINDOWS=168    # 历史窗口（1周）
PRED_LEN=192   # 预测长度（8天）
HORIZON=1

# 设备配置
DEVICE="cuda:6"

# 运行实验
python3 ./src/experiments/iReflow.py \
    --dataset_type=${DATASET} \
    --data_path=${DATA_PATH} \
    --windows=${WINDOWS} \
    --pred_len=${PRED_LEN} \
    --horizon=${HORIZON} \
    --d_model=${D_MODEL} \
    --n_heads=${N_HEADS} \
    --e_layers=${E_LAYERS} \
    --flow_layers=${FLOW_LAYERS} \
    --d_ff=${D_FF} \
    --dropout=${DROPOUT} \
    --batch_size=${BATCH_SIZE} \
    --learning_rate=${LEARNING_RATE} \
    --epochs=${EPOCHS} \
    --patience=${PATIENCE} \
    --num_sampling_steps=${NUM_SAMPLING_STEPS} \
    --temperature=${TEMPERATURE} \
    --num_samples=${NUM_SAMPLES} \
    --device=${DEVICE}

echo "iReflow实验完成！"

