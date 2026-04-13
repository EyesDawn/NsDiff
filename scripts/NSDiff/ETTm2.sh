export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 ./src/experiments/pretrain_f.py \
   --dataset_type="ETTm2" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1]'

python3 ./src/experiments/pretrain_g.py \
   --dataset_type="ETTm2" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --epochs=20 \
   --patience=5 \
   runs --seeds='[1]'

python3 ./src/experiments/NsDiff.py \
   --dataset_type="ETTm2" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=96 \
   --rolling_length=48 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1]'
