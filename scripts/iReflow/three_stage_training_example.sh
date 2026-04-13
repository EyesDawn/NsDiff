#!/bin/bash

# iReflow 三阶段训练示例脚本
# 演示如何按照正确的顺序训练 iReflow 模型
#
# 三个阶段：
# Stage 1: 预训练 iTransformer（点预测器）
# Stage 2: 预训练 Uncertainty Estimator（不确定性估计）
# Stage 3: 训练 Velocity Network（流匹配）

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============================================================================
# 通用配置
# ============================================================================

# 数据集配置
DATASET="SolarEnergy"
DATA_PATH="solar_AL/solar_AL.txt"
ROOT_PATH="./data/"

# 模型配置
MODEL_ID="solar_96_192"
D_FEATURES="M"
ENC_IN=137
DEC_IN=137
C_OUT=137
DES="Exp"
D_MODEL=512
N_HEADS=8
E_LAYERS=2
FLOW_LAYERS=3
D_FF=512
DROPOUT=0.1
USE_RELATIVE_SPACE=True

# 训练配置
BATCH_SIZE=32
EPOCHS=40
PATIENCE=8
LR_PATIENCE=3

# 预测配置
SEQ_LEN=96
PRED_LEN=192

# 设备配置
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}
DEVICE="cuda:0"

# 其他配置
SEEDS='[2223]'
ITR=1

# ============================================================================
# Stage 1: 预训练 iTransformer（点预测器）
# ============================================================================
echo "============================================================================"
echo "Stage 1: Pretraining iTransformer (Point Predictor)"
echo "============================================================================"
echo ""

STAGE1_WANDB_PROJECT="iReflow-Stage1-iTransformer"
STAGE1_CHECKPOINTS="./results/runs/iTransformer/"
STAGE1_LR=0.0001

echo "运行 iTransformer 预训练..."
echo "注意: 这里假设你已经有预训练的 iTransformer 模型"
echo "如果没有，请先运行 iTransformer 训练脚本"
echo ""
echo "示例命令（需要单独运行）："
echo "python3 -u ./src/experiments/iTransformer.py \\"
echo "    --wandb_project ${STAGE1_WANDB_PROJECT} \\"
echo "    --is_training 1 \\"
echo "    --root_path ${ROOT_PATH} \\"
echo "    --data_path ${DATA_PATH} \\"
echo "    --model_id ${MODEL_ID} \\"
echo "    --model iTransformer \\"
echo "    --data ${DATASET} \\"
echo "    --features ${D_FEATURES} \\"
echo "    --seq_len ${SEQ_LEN} \\"
echo "    --pred_len ${PRED_LEN} \\"
echo "    --e_layers ${E_LAYERS} \\"
echo "    --enc_in ${ENC_IN} \\"
echo "    --dec_in ${DEC_IN} \\"
echo "    --c_out ${C_OUT} \\"
echo "    --des ${DES} \\"
echo "    --d_model ${D_MODEL} \\"
echo "    --d_ff ${D_FF} \\"
echo "    --batch_size ${BATCH_SIZE} \\"
echo "    --lr ${STAGE1_LR} \\"
echo "    --itr ${ITR} \\"
echo "    --checkpoints ${STAGE1_CHECKPOINTS} \\"
echo "    --n_heads ${N_HEADS} \\"
echo "    --dropout ${DROPOUT} \\"
echo "    --epochs ${EPOCHS} \\"
echo "    --patience ${PATIENCE} \\"
echo "    --device ${DEVICE} \\"
echo "    runs --seeds=\"${SEEDS}\""
echo ""
echo "跳过 Stage 1（假设已完成）..."
echo ""

# ============================================================================
# Stage 2: 预训练 Uncertainty Estimator（不确定性估计）
# ============================================================================
echo "============================================================================"
echo "Stage 2: Pretraining Uncertainty Estimator"
echo "============================================================================"
echo ""

STAGE2_WANDB_PROJECT="iReflow-Stage2-Uncertainty"
STAGE2_CHECKPOINTS="./results/runs/iTransformer/"
STAGE2_LR=0.0005

python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
    --wandb_project ${STAGE2_WANDB_PROJECT} \
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
    --lr ${STAGE2_LR} \
    --itr ${ITR} \
    --checkpoints ${STAGE2_CHECKPOINTS} \
    --flow_layers ${FLOW_LAYERS} \
    --n_heads ${N_HEADS} \
    --dropout ${DROPOUT} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --lr_patience ${LR_PATIENCE} \
    --num_sampling_steps 1 \
    --temperature 1.0 \
    --num_samples 100 \
    --device ${DEVICE} \
    --use_relative_space ${USE_RELATIVE_SPACE} \
    runs --seeds="${SEEDS}"

echo ""
echo "Stage 2 完成!"
echo ""

# ============================================================================
# Stage 3: 训练 Velocity Network
# ============================================================================
echo "============================================================================"
echo "Stage 3: Training Velocity Network"
echo "============================================================================"
echo ""

STAGE3_WANDB_PROJECT="iReflow-Stage3-Velocity"
STAGE3_CHECKPOINTS="./results/runs/iTransformer/"
STAGE3_LR=0.0005
NUM_SAMPLING_STEPS=5
TEMPERATURE=1.0
NUM_SAMPLES=100

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
    runs --seeds="${SEEDS}"

echo ""
echo "Stage 3 完成!"
echo ""

# ============================================================================
# 训练完成
# ============================================================================
echo "============================================================================"
echo "三阶段训练完成!"
echo "============================================================================"
echo ""
echo "训练流程总结:"
echo "1. Stage 1: iTransformer 学习了基本的点预测能力"
echo "2. Stage 2: Uncertainty Estimator 学习了预测误差分布"
echo "3. Stage 3: Velocity Network 学习了从分布到真实值的流"
echo ""
echo "模型权重保存在: ${STAGE3_CHECKPOINTS}"
echo ""
echo "如需进行测试评估，请运行:"
echo "python3 -u ./src/experiments/iReflow.py --is_training 0 [其他参数]"
echo ""


