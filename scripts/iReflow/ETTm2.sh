#!/bin/bash

# iReflow运行脚本 - ETTm2数据集
# 使用Rectified Flow进行概率时间序列预测

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 数据集配置
DATASET="ETTm2"
DATA_PATH="./data/"

# 模型配置
D_MODEL=512
N_HEADS=8
E_LAYERS=2
FLOW_LAYERS=3
D_FF=2048
DROPOUT=0.1

# 训练配置
BATCH_SIZE=64
LEARNING_RATE=0.0001
EPOCHS=100
PATIENCE=10

# Flow配置
NUM_SAMPLING_STEPS=1
TEMPERATURE=1.0
NUM_SAMPLES=100

# 预测配置 - ETTm2使用15分钟级数据
WINDOWS=672    # 历史窗口（1周）
PRED_LEN=768   # 预测长度（8天）
HORIZON=1

# 设备配置
DEVICE="cuda:6"

SEEDS='[22,33]'
WANDB_PROJECT="iReflow"

# 运行实验
python3 ./src/experiments/iReflow.py \
    config_wandb --project=${WANDB_PROJECT} \
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
    --device=${DEVICE} \
    runs --seeds="${SEEDS}"

echo "iReflow-ETTm2 experiment completed!"

