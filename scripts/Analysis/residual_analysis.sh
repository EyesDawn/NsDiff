export PYTHONPATH=./
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# python3 -u ./src/experiments/pretrain_uncertainty_estimator.py \
#     extract_residuals_on_test \
#     --wandb_project "" \
#     --is_training 0 \
#     --root_path ./data/ \
#     --data_path electricity/electricity.csv \
#     --model_id electricity_96_192 \
#     --model iReflow \
#     --data Electricity \
#     --features M \
#     --seq_len 96 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 32 \
#     --lr 0.0005 \
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
#     --save_path ./results/analysis/electricity_residuals_origin.npz \
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
# 整体图（[N, P, D] 全部展平）+ 特征维度 grid 图（自动选最多 12 个均匀分布维度）
# python3 -u ./src/analysis/pdn_density_plot.py \
#     --npz_path  ./results/analysis/electricity_residuals_fast.npz \
#     --output_dir ./results/analysis \
#     --prefix    electricity \
#     --zpdn_clip_min -6 --zpdn_clip_max 6 \
#     --bins 120 \
#     --n_cols 3 \
#     --skip_overall

# 若需指定特定特征维度（例如 0 1 2 10 50 100），可加 --feature_dims 参数：
python3 -u ./src/analysis/pdn_density_plot.py \
    --npz_path  ./results/analysis/electricity_residuals_origin.npz \
    --output_dir ./results/analysis \
    --prefix    electricity_S0_origin \
    --no_clip \
    --bins 120 \
    --separate_per_feature \
    --sample_n_overall 10 \
    --sample_n_per_feature 10 \
    --kde_max_points 200000 \
    --seed 0 \
    --skip_overall