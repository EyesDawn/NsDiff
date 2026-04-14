
# ── Experiment 1: Decoupling case study ─────────────────────────────────────
# 推荐输入：使用 `extract_residuals_on_test` 导出的 npz，其中至少包含
# Y / mu_X / sigma_X / mu_Y_hat / sigma_Y_hat / Z_PDN。
#
# 若不指定 --feature_dim，脚本会自动选择 drift 最强的变量维度。
# 输出：
#   1) NeurIPS 风格三联图
#   2) 可复现的 JSON 元数据（记录选中的窗口索引及相关统计）
#
python3 -u ./src/analysis/decoupling_case_study.py \
    --npz_path ./results/analysis/ETTm1/ETTm1_decoupling_case_study_data_seed_2029.npz \
    --output_path ./results/analysis/ETTm1/ETTm1_decoupling_case_study_seed_2029_feature_2.pdf \
    --metadata_path ./results/analysis/ETTm1/ETTm1_decoupling_case_study_seed_2029_feature_2.json \
    --dataset_name ETTm1 \
    --feature_dim 2 \
    --num_samples 6 \
    --high_drift_ratio 0.2 \
    --min_high_drift_windows 24 \
    --min_past_std 0.02
