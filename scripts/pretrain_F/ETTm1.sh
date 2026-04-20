export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/pretrain_f.py \
   --dataset_type="ETTm1" \
   --device="cuda:2" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1]'