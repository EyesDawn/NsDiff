#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
WANDB_PROJECT="iTransformer"


python -u ./src/experiments/iTransformer.py \
  --wandb_project ${WANDB_PROJECT} \
  --is_training 1 \
  --train_epochs 30 \
  --patience 10 \
  --root_path ./data/ \
  --data_path ETTh1/ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --dropout 0.2 \
  --lradj plateau \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 0.000001 \
  --itr 1 \
  --checkpoints ./results/runs/iTransformer/ | tee ./results/logs/iTransformer/iTransformer_ETTh1.log
