#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer
WANDB_PROJECT="iTransformer"


python -u ./src/experiments/iTransformer.py \
  --wandb_project ${WANDB_PROJECT} \
  --is_training 1 \
  --root_path ./data/ \
  --data_path traffic/traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data Traffic \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --checkpoints ./results/runs/iTransformer/

echo "iTransformer-Traffic experiment completed!"
