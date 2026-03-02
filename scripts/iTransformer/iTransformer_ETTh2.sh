#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
WANDB_PROJECT="iTransformer"


python -u ./src/experiments/iTransformer.py \
  --wandb_project ${WANDB_PROJECT} \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTh2/ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --checkpoints ./results/runs/iTransformer/

echo "iTransformer-ETTh2 experiment completed!"
