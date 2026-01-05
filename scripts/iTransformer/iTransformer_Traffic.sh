#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=5

model_name=iTransformer
WANDB_PROJECT="iTransformer"

# python -u ./src/experiments/iTransformer.py \
#   --wandb_project ${WANDB_PROJECT} \
#   --is_training 1 \
#   --root_path ./data/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_96 \
#   --model $model_name \
#   --data Traffic \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --batch_size 64 \
#   --checkpoints ./results/runs/iTransformer/

python -u ./src/experiments/iTransformer.py \
  --wandb_project ${WANDB_PROJECT} \
  --is_training 1 \
  --root_path ./data/ \
  --data_path traffic/traffic.txt \
  --model_id traffic_96_192 \
  --model $model_name \
  --data Traffic \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --batch_size 64 \
  --checkpoints ./results/runs/iTransformer/

# python -u ./src/experiments/iTransformer.py \
#   --is_training 1 \
#   --root_path ./data/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --data Traffic \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --wandb_project ${WANDB_PROJECT}

# python -u ./src/experiments/iTransformer.py \
#   --is_training 1 \
#   --root_path ./data/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --data Traffic \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --wandb_project ${WANDB_PROJECT}

echo "iTransformer-Traffic experiment completed!"
