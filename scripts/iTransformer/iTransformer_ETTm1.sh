#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer
WANDB_PROJECT="iTransformer"

# python -u ./src/experiments/iTransformer.py \
#   --wandb_project ${WANDB_PROJECT} \
#   --is_training 1 \
#   --root_path ./data/ETTm1/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_96 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --batch_size 128 \
#   --checkpoints ./results/runs/iTransformer/

python -u ./src/experiments/iTransformer.py \
  --wandb_project ${WANDB_PROJECT} \
  --is_training 1 \
  --root_path ./data/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
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
  --batch_size 128 \
  --checkpoints ./results/runs/iTransformer/

# python -u ./src/experiments/iTransformer.py \
#   --is_training 1 \
#   --root_path ./data/ETTm1/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_336 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --wandb_project ${WANDB_PROJECT}

# python -u ./src/experiments/iTransformer.py \
#   --is_training 1 \
#   --root_path ./data/ETTm1/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_720 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --wandb_project ${WANDB_PROJECT}

echo "iTransformer-ETTm1 experiment completed!"
