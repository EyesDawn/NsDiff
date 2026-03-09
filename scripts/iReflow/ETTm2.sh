#!/bin/bash

# iReflow运行脚本 - ETTm2数据集
# 使用Rectified Flow进行概率时间序列预测

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 数据集配置
DATASET="ETTm2"
DATA_PATH="ETTm2/ETTm2.csv"
ROOT_PATH="./data/"

# 模型配置
MODEL_ID="ETTm2_96_192"
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
USE_RELATIVE_SPACE=False
USE_ITRANSFORMER_ENC=True

# 训练配置
IS_TRAINING=1
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=20
PATIENCE=10
LR_PATIENCE=1

# Flow配置
NUM_SAMPLING_STEPS=1
TEMPERATURE=1.0
NUM_SAMPLES=100

# 预测配置
SEQ_LEN=96
PRED_LEN=192
HORIZON=1

# 设备配置
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}
DEVICE="cuda:0"

# 实验配置
SEEDS='[2222]'
CHECKPOINTS="./results/runs/iTransformer/"
ITR=1

# ============================================================================
# Stage 2: 预训练 Uncertainty Estimator（不确定性估计）
# ============================================================================
# echo "============================================================================"
# echo "Stage 2: Pretraining Uncertainty Estimator"
# echo "============================================================================"
# echo ""

# STAGE2_WANDB_PROJECT="iReflow-Stage2-Uncertainty"
# STAGE2_CHECKPOINTS="./results/runs/iTransformer/"
# STAGE2_LR=0.0001

# python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
#     --wandb_project ${STAGE2_WANDB_PROJECT} \
#     --is_training 1 \
#     --root_path ${ROOT_PATH} \
#     --data_path ${DATA_PATH} \
#     --model_id ${MODEL_ID} \
#     --model iReflow \
#     --data ${DATASET} \
#     --features ${D_FEATURES} \
#     --seq_len ${SEQ_LEN} \
#     --pred_len ${PRED_LEN} \
#     --e_layers ${E_LAYERS} \
#     --enc_in ${ENC_IN} \
#     --dec_in ${DEC_IN} \
#     --c_out ${C_OUT} \
#     --des ${DES} \
#     --d_model ${D_MODEL} \
#     --d_ff ${D_FF} \
#     --batch_size ${BATCH_SIZE} \
#     --lr ${STAGE2_LR} \
#     --itr ${ITR} \
#     --checkpoints ${STAGE2_CHECKPOINTS} \
#     --flow_layers ${FLOW_LAYERS} \
#     --n_heads ${N_HEADS} \
#     --dropout ${DROPOUT} \
#     --epochs 10 \
#     --patience 3 \
#     --lr_patience 1 \
#     --num_sampling_steps 1 \
#     --temperature 1.0 \
#     --num_samples 100 \
#     --device ${DEVICE} \
#     --use_relative_space ${USE_RELATIVE_SPACE} \
#     runs --seeds="${SEEDS}"

# echo ""
# echo "Stage 2 completed!"
# echo ""

# ============================================================================
# Stage 3: 训练 Velocity Network
# ============================================================================
echo "============================================================================"
echo "Stage 3: Training Velocity Network"
echo "============================================================================"
echo ""

STAGE3_WANDB_PROJECT="iReflow-Stage3-Velocity-revin"
STAGE3_CHECKPOINTS="./results/runs/iTransformer/"
STAGE3_LR=0.0001

python3 -u ./src/experiments/iReflow.py \
    --wandb_project ${STAGE3_WANDB_PROJECT} \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${MODEL_ID} \
    --model iReflow \
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
    --lr ${STAGE3_LR} \
    --itr ${ITR} \
    --checkpoints ${STAGE3_CHECKPOINTS} \
    --flow_layers ${FLOW_LAYERS} \
    --n_heads ${N_HEADS} \
    --dropout ${DROPOUT} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --lr_patience ${LR_PATIENCE} \
    --num_sampling_steps ${NUM_SAMPLING_STEPS} \
    --temperature ${TEMPERATURE} \
    --num_samples ${NUM_SAMPLES} \
    --device ${DEVICE} \
    --use_relative_space ${USE_RELATIVE_SPACE} \
    --use_itransformer_enc ${USE_ITRANSFORMER_ENC} \
    runs --seeds="${SEEDS}"

echo ""
echo "Stage 3 completed!"
echo ""