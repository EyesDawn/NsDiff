export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeGrad.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="ETTm2 " \
   --residual_layers=4 \
   --device="cuda:3" \
   --batch_size=16 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   runs --seeds='[1]'


export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeGrad.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="ETTm2 " \
   --residual_layers=4 \
   --device="cuda:3" \
   --batch_size=16 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   runs --seeds='[1]'