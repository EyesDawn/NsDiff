export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
#     extract_residuals_on_test \
#     --wandb_project "" \
#     --is_training 0 \
#     --root_path ./data/ \
#     --data_path weather/weather.csv \
#     --model_id weather_96_192 \
#     --model iReflow \
#     --data Weather \
#     --features M \
#     --seq_len 96 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 32 \
#     --lr 0.0001 \
#     --itr 1 \
#     --checkpoints ./results/runs/iTransformer/ \
#     --flow_layers 3 \
#     --n_heads 8 \
#     --dropout 0.1 \
#     --epochs 20 \
#     --patience 6 \
#     --lr_patience 1 \
#     --num_sampling_steps 1 \
#     --temperature 1.0 \
#     --num_samples 100 \
#     --device cuda:0 \
#     --use_relative_space True \
#     --seed 2222 \
#     --save_path ./results/analysis/Weather/Weather_residuals_origin.npz \
#     --use_origin_scale True

# python3 -u ./src/analysis/residual_histogram.py \
#     --npz_path ./results/analysis/electricity_residuals.npz \
#     --output_path ./results/analysis/electricity_residual_hist.png \
#     --clip_min -6 --clip_max 6

# python3 -u ./src/analysis/residual_wasserstein.py \
#     --npz_path ./results/analysis/electricity_residuals_fast.npz \
#     --output_json ./results/analysis/electricity_residuals_wasserstein.json \
#     --clip_min -6 --clip_max 6

# ── PDN 归一化前后概率密度分布对比 ─────────────────────────────────────────────
# 窗口图（[12, P, d] 展平，12个连续样本分3段×4个，每段一个子图）
#   --window_start     : N 轴起始索引列表（默认 0）
#   --window_size      : 连续样本总数（默认 12）
#   --window_segment_size : 每段样本数（默认 4），n_segments = window_size // window_segment_size
#   --feature_dims     : 指定特征维度（省略则自动选最多 12 个均匀分布维度）
#   --n_cols           : grid 列数
#   --separate_window  : 合并所有 window_start，只输出 2 张图（Y / ZPDN）
# python3 -u ./src/analysis/pdn_density_plot.py \
#     --npz_path  ./results/analysis/ETTm2/ETTm2_residuals_origin.npz \
#     --output_dir ./results/analysis/ETTm2 \
#     --prefix    ETTm2_d0_window_12 \
#     --bins 60 \
#     --n_cols 3 \
#     --separate_window \
#     --feature_dims 0 \
#     --window_start 710 720 730 740 750 760 770 780 790 \
#     --window_size 12 \
#     --window_segment_size 4 \
#     --no_clip \
#     --skip_per_feature

# 若需指定特定特征维度（例如 0 1 2 10 50 100），可加 --feature_dims 参数：
python3 -u ./src/analysis/pdn_density_plot.py \
    --npz_path  ./results/analysis/Weather/Weather_residuals_origin.npz \
    --output_dir ./results/analysis/Weather \
    --prefix    Weather_Seed0_origin \
    --no_clip \
    --bins 120 \
    --separate_per_feature \
    --sample_n_overall 10 \
    --sample_n_per_feature 10 \
    --kde_max_points 200000 \
    --seed 0 \
    --skip_window