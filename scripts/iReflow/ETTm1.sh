#!/bin/bash

# iReflow运行脚本 - ETTm1数据集
# 使用Rectified Flow进行概率时间序列预测

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 数据集配置
DATASET="ETTm1"
DATA_PATH="ETTm1/ETTm1.csv"
ROOT_PATH="./data/"

# 模型配置
MODEL_ID="ETTm1_96_192"
MODEL_NAME="iReflow"
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
IS_TRAINING=1
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=10
PATIENCE=3

# Flow配置
NUM_SAMPLING_STEPS=1
TEMPERATURE=1.0
NUM_SAMPLES=100

# 预测配置
SEQ_LEN=96
PRED_LEN=192
HORIZON=1

# 设备配置
DEVICE="cuda:2"

SEEDS='[2222,2023]'
WANDB_PROJECT="iReflow-v1"
CHECKPOINTS="./results/runs/iTransformer/"
ITR=1

# 运行实验
python3 -u ./src/experiments/iReflow.py \
    --wandb_project ${WANDB_PROJECT} \
    --is_training ${IS_TRAINING} \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${MODEL_ID} \
    --model ${MODEL_NAME} \
    --data ${DATASET} \
    --features ${D_FEATURES} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --e_layers ${E_LAYERS} \
    --enc_in ${ENC_IN} \
    --dec_in ${DEC_IN} \
    --c_out ${C_OUT} \
    --des ${DES} \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --itr ${ITR} \
    --checkpoints ${CHECKPOINTS} \
    --flow_layers ${FLOW_LAYERS} \
    --n_heads ${N_HEADS} \
    --dropout ${DROPOUT} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --num_sampling_steps ${NUM_SAMPLING_STEPS} \
    --temperature ${TEMPERATURE} \
    --num_samples ${NUM_SAMPLES} \
    --device ${DEVICE} \
    runs --seeds="${SEEDS}"

echo "iReflow-ETTm1 experiment completed!"

