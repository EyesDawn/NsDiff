#!/bin/bash

export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Traffic residual extraction for analytical experiments.
# Important: this script calls `fire.Fire(UncertaintyEstimatorPretrainExp)`,
# so experiment-constructor args must appear BEFORE `extract_residuals_on_test`.

DATASET="Traffic"
ROOT_PATH="./data/"
DATA_PATH="traffic/traffic.csv"
MODEL_ID="traffic_96_192"
MODEL_NAME="iReflow"
FEATURES="M"

SEQ_LEN=96
PRED_LEN=192
ENC_IN=862
DEC_IN=862
C_OUT=862
E_LAYERS=4
FLOW_LAYERS=3
D_MODEL=512
D_FF=512
N_HEADS=8
DROPOUT=0.1
USE_RELATIVE_SPACE=True

BATCH_SIZE=16
LR=0.0001
EPOCHS=20
PATIENCE=6
LR_PATIENCE=1
NUM_SAMPLING_STEPS=1
TEMPERATURE=1.0
NUM_SAMPLES=100

DEVICE="cuda:0"
CHECKPOINTS="./results/runs/iTransformer/"
SEED=2022
SAVE_PATH="./results/analysis/Traffic/Traffic_residuals_origin.npz"

python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
    --is_training 0 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${MODEL_ID} \
    --model ${MODEL_NAME} \
    --data ${DATASET} \
    --features ${FEATURES} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --e_layers ${E_LAYERS} \
    --enc_in ${ENC_IN} \
    --dec_in ${DEC_IN} \
    --c_out ${C_OUT} \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --itr 1 \
    --checkpoints ${CHECKPOINTS} \
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
    extract_residuals_on_test \
    --seed ${SEED} \
    --save_path ${SAVE_PATH} \
    --use_origin_scale True
