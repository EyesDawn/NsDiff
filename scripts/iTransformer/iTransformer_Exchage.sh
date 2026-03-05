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
  --data_path ExchangeRate/exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data ExchangeRate \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --checkpoints ./results/runs/iTransformer/

echo "iTransformer-Exchage experiment completed!"
