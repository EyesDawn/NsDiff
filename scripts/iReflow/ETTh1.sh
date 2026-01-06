#!/bin/bash

# iReflow运行脚本 - ETTh1数据集
# 使用Rectified Flow进行概率时间序列预测

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 数据集配置
DATASET="ETTh1"
DATA_PATH="./data/ETTh1/ETTh1.csv"

# 模型配置 - 参考 ETTm1/ETTm2 的配置（ETT 数据集通常参数相似）
D_FEATURES="M"
ENC_IN=7
DEC_IN=7
C_OUT=7
DES="Exp"
D_MODEL=128
N_HEADS=8
E_LAYERS=2
FLOW_LAYERS=3
D_FF=128
DROPOUT=0.1

# 训练配置
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=10
PATIENCE=3

# Flow配置
NUM_SAMPLING_STEPS=1
TEMPERATURE=1.0
NUM_SAMPLES=100

# 预测配置
WINDOWS=96
PRED_LEN=192
HORIZON=1

# 设备配置
DEVICE="cuda:6"

SEEDS='[22,2023]'
WANDB_PROJECT="iReflow"

# 运行实验
python3 ./src/experiments/iReflow.py \
    config_wandb --project=${WANDB_PROJECT} \
    --dataset_type=${DATASET} \
    --data_path=${DATA_PATH} \
    --windows=${WINDOWS} \
    --features=${D_FEATURES} \
    --enc_in=${ENC_IN} \
    --dec_in=${DEC_IN} \
    --c_out=${C_OUT} \
    --pred_len=${PRED_LEN} \
    --horizon=${HORIZON} \
    --d_model=${D_MODEL} \
    --n_heads=${N_HEADS} \
    --e_layers=${E_LAYERS} \
    --flow_layers=${FLOW_LAYERS} \
    --d_ff=${D_FF} \
    --des=${DES} \
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

echo "iReflow-ETTh1 experiment completed!"

