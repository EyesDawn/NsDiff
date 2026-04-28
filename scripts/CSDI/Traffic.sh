export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/CSDI.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="Traffic" \
   --device="cuda:5" \
   --batch_size=4 \
   --horizon=1 \
   --layers=1 \
   --pred_len=192 \
   --windows=96 \
   runs --seeds='[4]'
