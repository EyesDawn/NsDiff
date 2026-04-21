export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/pretrain_g.py \
   --dataset_type="SolarEnergy" \
   --device="cuda:2" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --epochs=20 \
   --patience=5 \
   runs --seeds='[1]'
