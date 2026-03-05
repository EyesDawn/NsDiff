import argparse
import json
import os
from typing import Tuple, Dict

import numpy as np

try:
    from scipy.stats import wasserstein_distance
except ImportError:  # scipy 可选
    wasserstein_distance = None


def _flatten_and_clip(
    z: np.ndarray,
    clip_range: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, float]:
    """
    将 3D 残差张量展平成 1D，并在给定区间内进行截断。

    Args:
        z: 形状 [N, P, D] 的残差张量 (numpy)。
        clip_range: 截断区间 (min, max)。若为 None 则不截断。

    Returns:
        clipped: 展平后并截断的 1D 数组。
        clipped_ratio: 被截断样本占总样本的比例（0~1）。
    """
    z_flat = z.astype(np.float64).ravel()
    if clip_range is None:
        return z_flat, 0.0

    vmin, vmax = clip_range
    before = z_flat.copy()
    z_flat = np.clip(z_flat, vmin, vmax)
    clipped_ratio = float(np.mean((before < vmin) | (before > vmax)))
    return z_flat, clipped_ratio


def _wasserstein_empirical(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    1D Wasserstein-1 距离的经验近似实现。

    若安装了 scipy，则优先使用 scipy.stats.wasserstein_distance；
    否则使用排序 + 经验分位数的近似：
        W1 ≈ mean(|Q_a(u_i) - Q_b(u_i)|),  u_i 均匀覆盖 [0,1]。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.size == 0 or b.size == 0:
        return float("nan")

    if wasserstein_distance is not None:
        return float(wasserstein_distance(a, b))

    # 无 scipy 时的近似实现
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n = a_sorted.size
    m = b_sorted.size
    k = min(n, m)
    if k == 0:
        return float("nan")

    # 在两个经验分布上采样相同数量的经验分位数
    idx_a = (np.linspace(0, n - 1, k)).astype(int)
    idx_b = (np.linspace(0, m - 1, k)).astype(int)
    qa = a_sorted[idx_a]
    qb = b_sorted[idx_b]
    return float(np.mean(np.abs(qa - qb)))


def _kl_empirical_to_normal(
    z: np.ndarray,
    clip_range: Tuple[float, float] | None = None,
    num_bins: int = 200,
    eps: float = 1e-8,
) -> float:
    """
    使用直方图近似 KL(p || q)，其中：
        - p: 残差 Z 的经验分布
        - q: 标准正态 N(0, 1)

    具体做法：
        1) 在给定区间内对 Z 做直方图，得到经验概率 p_i
        2) 在同一分箱中心上评估 N(0,1) 的 pdf，并离散化为 q_i
        3) KL(p||q) ≈ sum_i p_i * log(p_i / q_i)
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return float("nan")

    if clip_range is not None:
        vmin, vmax = clip_range
    else:
        # 若未指定区间，则使用数据的 [1%, 99%] 分位作为稳定区间
        vmin, vmax = np.quantile(z, [0.01, 0.99])

    # 经验直方图（不使用 density，让 p_i 直接是概率）
    hist, bin_edges = np.histogram(z, bins=num_bins, range=(vmin, vmax), density=False)
    total = hist.sum()
    if total == 0:
        return float("nan")
    p = hist.astype(np.float64) / float(total)

    # 分箱中心
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 在中心点上评估标准正态 pdf，并离散化成近似概率 q_i
    # pdf(x) = exp(-0.5 x^2) / sqrt(2π)
    q_unnorm = np.exp(-0.5 * centers ** 2) / np.sqrt(2.0 * np.pi)
    q = q_unnorm / (q_unnorm.sum() + eps)

    # 避免 log(0)
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)

    kl = float(np.sum(p_safe * np.log(p_safe / q_safe)))
    return kl


def compute_wasserstein_metrics(
    npz_path: str,
    clip_range: Tuple[float, float] | None,
    normal_sample_size: int = 200_000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    计算 Z_RevIN, Z_PDN 与标准正态 N(0,1) 之间的 Wasserstein-1 距离。

    同时给出 Z_RevIN 与 Z_PDN 之间的 W1，方便横向对比。
    """
    data = np.load(npz_path)
    if "Z_RevIN" not in data or "Z_PDN" not in data:
        raise KeyError(
            f"{npz_path} 中未找到 'Z_RevIN' / 'Z_PDN'，"
            f"请确认已使用 `extract_residuals_on_test` 生成该文件。"
        )

    z_revin = data["Z_RevIN"]
    z_pdn = data["Z_PDN"]

    z_revin_flat, ratio_revin = _flatten_and_clip(z_revin, clip_range)
    z_pdn_flat, ratio_pdn = _flatten_and_clip(z_pdn, clip_range)

    rng = np.random.default_rng(seed)
    z_normal = rng.normal(loc=0.0, scale=1.0, size=normal_sample_size).astype(
        np.float64
    )

    w_revin_normal = _wasserstein_empirical(z_revin_flat, z_normal)
    w_pdn_normal = _wasserstein_empirical(z_pdn_flat, z_normal)
    w_revin_pdn = _wasserstein_empirical(z_revin_flat, z_pdn_flat)

    metrics: Dict[str, float] = {
        "w1_revin_vs_normal": w_revin_normal,
        "w1_pdn_vs_normal": w_pdn_normal,
        "w1_revin_vs_pdn": w_revin_pdn,
        "clip_ratio_revin": ratio_revin,
        "clip_ratio_pdn": ratio_pdn,
        "num_samples_revin": float(z_revin_flat.size),
        "num_samples_pdn": float(z_pdn_flat.size),
        "normal_sample_size": float(normal_sample_size),
    }
    return metrics


def compute_kl_metrics(
    npz_path: str,
    clip_range: Tuple[float, float] | None,
    num_bins: int = 200,
) -> Dict[str, float]:
    """
    计算 Z_RevIN, Z_PDN 相对于标准正态 N(0,1) 的 KL 散度：
        - KL(RevIN || N(0,1))
        - KL(PDN   || N(0,1))

    与 Wasserstein 指标类似，先展平并（可选）截断，再在给定区间上用直方图近似。
    """
    data = np.load(npz_path)
    if "Z_RevIN" not in data or "Z_PDN" not in data:
        raise KeyError(
            f"{npz_path} 中未找到 'Z_RevIN' / 'Z_PDN'，"
            f"请确认已使用 `extract_residuals_on_test` 生成该文件。"
        )

    z_revin = data["Z_RevIN"]
    z_pdn = data["Z_PDN"]

    z_revin_flat, ratio_revin = _flatten_and_clip(z_revin, clip_range)
    z_pdn_flat, ratio_pdn = _flatten_and_clip(z_pdn, clip_range)

    kl_revin = _kl_empirical_to_normal(z_revin_flat, clip_range=clip_range, num_bins=num_bins)
    kl_pdn = _kl_empirical_to_normal(z_pdn_flat, clip_range=clip_range, num_bins=num_bins)

    metrics: Dict[str, float] = {
        "kl_revin_vs_normal": kl_revin,
        "kl_pdn_vs_normal": kl_pdn,
        "clip_ratio_revin": ratio_revin,
        "clip_ratio_pdn": ratio_pdn,
        "num_samples_revin": float(z_revin_flat.size),
        "num_samples_pdn": float(z_pdn_flat.size),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Wasserstein-1 distances between residuals "
            "(RevIN / PDN) and standard normal N(0,1)."
        )
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="./results/analysis/electricity_residuals_fast.npz",
        help="Path to the residuals npz file generated by extract_residuals_on_test.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./results/analysis/electricity_residuals_wasserstein.json",
        help="Path to save Wasserstein metrics as JSON.",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=-6.0,
        help="Minimum residual value for clipping. Set together with --clip_max.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=6.0,
        help="Maximum residual value for clipping. Set together with --clip_min.",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable clipping (use full residual range).",
    )
    parser.add_argument(
        "--normal_sample_size",
        type=int,
        default=200_000,
        help="Number of samples drawn from N(0,1) for empirical W1 approximation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for normal sampling.",
    )

    args = parser.parse_args()

    if args.no_clip:
        clip_range = None
    else:
        clip_range = (args.clip_min, args.clip_max)

    # 1) Wasserstein-1 指标
    metrics_w = compute_wasserstein_metrics(
        npz_path=args.npz_path,
        clip_range=clip_range,
        normal_sample_size=args.normal_sample_size,
        seed=args.seed,
    )

    # 2) KL 散度指标（相对于标准正态 N(0,1)）
    metrics_kl = compute_kl_metrics(
        npz_path=args.npz_path,
        clip_range=clip_range,
        num_bins=200,
    )

    # 合并两类指标（若存在同名键，KL 指标会覆盖 Wasserstein 指标，但目前键名互不冲突）
    metrics = {**metrics_w, **metrics_kl}

    # 控制台打印，便于直接查看
    print("===== Wasserstein-1 distances (W1) & KL divergence =====")
    if clip_range is None:
        print("Clipping: disabled (using full residual range)")
    else:
        print(f"Clipping range: [{clip_range[0]}, {clip_range[1]}]")
    print(f"W1(RevIN,  N(0,1)) = {metrics['w1_revin_vs_normal']:.6f}")
    print(f"W1(PDN,    N(0,1)) = {metrics['w1_pdn_vs_normal']:.6f}")
    print(f"W1(RevIN, PDN   ) = {metrics['w1_revin_vs_pdn']:.6f}")
    print(f"KL(RevIN || N(0,1)) = {metrics['kl_revin_vs_normal']:.6f}")
    print(f"KL(PDN   || N(0,1)) = {metrics['kl_pdn_vs_normal']:.6f}")
    print(
        f"Clipped ratio – RevIN: {metrics['clip_ratio_revin']:.4f}, "
        f"PDN: {metrics['clip_ratio_pdn']:.4f}"
    )

    # 保存到 JSON，便于论文表格或后续脚本处理
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved Wasserstein metrics to: {args.output_json}")


if __name__ == "__main__":
    main()


