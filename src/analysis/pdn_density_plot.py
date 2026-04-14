"""
pdn_density_plot.py
===================
绘制 PDN 归一化前后的概率密度分布对比图。

  - "PDN 之前" : Y   —— dataloader 输出的真实未来序列，尚未经过 PDN 归一化
  - "PDN 之后" : Z_PDN = (Y - mu_Y_hat) / sigma_Y_hat

输出图表分两类：
  1. 整体图（overall）: 将 [N, P, D] 全部展平后，叠加绘制 Y 与 Z_PDN 的密度曲线，
     同时附上均值/方差/标准正态参考线。
  2. 特征维度图（per-feature）: 对指定的若干特征维度 d，将 [N, P] 展平后，
     绘制该维度下 Y[:, :, d] 与 Z_PDN[:, :, d] 的密度对比图，
     多个维度合并为一张 grid 图。
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    sns = None
    _HAS_SNS = False

try:
    from scipy.stats import gaussian_kde, norm as sp_norm
    _HAS_SCIPY = True
except ImportError:
    gaussian_kde = None
    sp_norm = None
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _normal_pdf(xs: np.ndarray) -> np.ndarray:
    """标准正态 pdf，不依赖 scipy。"""
    return np.exp(-0.5 * xs ** 2) / np.sqrt(2.0 * np.pi)


def _flatten(arr: np.ndarray, clip_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """展平为 1D float64，并可选地做截断。"""
    flat = arr.astype(np.float64).ravel()
    if clip_range is not None:
        flat = np.clip(flat, clip_range[0], clip_range[1])
    return flat


def _subsample_1d(
    x: np.ndarray,
    max_points: Optional[int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """对 1D 数组做无放回下采样（用于限制 KDE 计算量）。"""
    if max_points is None:
        return x
    if max_points <= 0:
        return x
    if x.size <= max_points:
        return x
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx]


def _maybe_subsample_N(
    Y: np.ndarray,
    Z_PDN: np.ndarray,
    sample_n: Optional[int],
    rng: np.random.Generator,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 N 维对 [N,P,D] 做无放回抽样，降低整体绘图规模。"""
    if sample_n is None:
        return Y, Z_PDN
    if sample_n <= 0:
        return Y, Z_PDN
    N = Y.shape[0]
    k = min(int(sample_n), N)
    if k == N:
        return Y, Z_PDN
    idx = rng.choice(N, size=k, replace=False)
    print(f"[Sampling:{tag}] N {N} -> {k}")
    return Y[idx], Z_PDN[idx]


def _plot_single_on_ax(
    ax: plt.Axes,
    data: np.ndarray,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    color: str,
    label: str,
    add_normal_ref: bool = True,
    title: str = "",
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    在给定 Axes 上绘制**单条**分布的密度图（直方图 + KDE），附上标准正态参考曲线（可选）。

    Args:
        ax            : 目标 Axes。
        data          : 待绘制数据（任意形状 numpy 数组，内部自动展平）。
        clip_range    : (xmin, xmax) 截断并作为坐标范围；None 则用数据实际范围。
        bins          : 直方图分桶数。
        color         : 直方图与 KDE 曲线的颜色。
        label         : 图例标签。
        add_normal_ref: 是否叠加 N(0,1) 参考曲线。
        title         : 子图标题。
        show_stats    : 是否在图内标注均值 ± 标准差统计量。
    """
    flat = _flatten(data, clip_range)

    if clip_range is not None:
        xmin, xmax = clip_range
    else:
        xmin = float(flat.min())
        xmax = float(flat.max())

    xs = np.linspace(xmin, xmax, 512)

    if _HAS_SNS:
        sns.histplot(flat, bins=bins, stat="density",
                     color=color, alpha=0.40, label=label,
                     edgecolor=None, ax=ax)
    else:
        ax.hist(flat, bins=bins, density=True,
                color=color, alpha=0.40, label=label)

    if _HAS_SCIPY and flat.size > 1:
        try:
            flat_kde = _subsample_1d(flat, kde_max_points, rng=rng)
            kde = gaussian_kde(flat_kde, bw_method="scott")
            ax.plot(xs, kde(xs), color=color, linewidth=1.8, linestyle="-")
        except Exception:
            pass

    if add_normal_ref:
        ys_norm = sp_norm.pdf(xs, 0.0, 1.0) if _HAS_SCIPY else _normal_pdf(xs)
        ax.plot(xs, ys_norm, "k--", linewidth=1.6, label=r"$\mathcal{N}(0,1)$")

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.30)
    if title:
        ax.set_title(title, fontsize=14)

    if show_stats:
        stats_text = f"μ={flat.mean():.3f}, σ={flat.std():.3f}"
        ax.text(
            0.02, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
        )


def _plot_density_on_ax(
    ax: plt.Axes,
    data_before: np.ndarray,
    data_after: np.ndarray,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    label_before: str = "Y (before PDN)",
    label_after: str = r"$Z_{\rm PDN}$ (after PDN)",
    add_normal_ref: bool = True,
    title: str = "",
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    在给定 Axes 上叠加绘制 data_before 与 data_after 的密度分布，
    附上标准正态参考曲线（可选）。

    Args:
        ax            : 目标 Axes。
        data_before   : Y 数据（任意形状 numpy 数组）。
        data_after    : Z_PDN 数据（任意形状 numpy 数组）。
        clip_range    : (xmin, xmax) 用于截断与坐标范围。
        bins          : 直方图分桶数。
        label_before  : 前者图例标签。
        label_after   : 后者图例标签。
        add_normal_ref: 是否叠加 N(0,1) 曲线。
        title         : 子图标题。
        show_stats    : 是否在图内标注 Y 与 Z_PDN 的均值 ± 标准差统计量。
    """
    # ---- 截断 ----
    b = _flatten(data_before, clip_range)
    a = _flatten(data_after, clip_range)

    # ---- x 范围 ----
    if clip_range is not None:
        xmin, xmax = clip_range
    else:
        xmin = float(min(b.min(), a.min()))
        xmax = float(max(b.max(), a.max()))

    xs = np.linspace(xmin, xmax, 512)

    # ---- 绘图 ----
    if _HAS_SNS:
        sns.histplot(b, bins=bins, stat="density",
                     color="tab:orange", alpha=0.40, label=label_before,
                     edgecolor=None, ax=ax)
        sns.histplot(a, bins=bins, stat="density",
                     color="tab:blue",   alpha=0.40, label=label_after,
                     edgecolor=None, ax=ax)
    else:
        ax.hist(b, bins=bins, density=True,
                color="tab:orange", alpha=0.40, label=label_before)
        ax.hist(a, bins=bins, density=True,
                color="tab:blue",   alpha=0.40, label=label_after)

    # KDE 曲线（若有 scipy）
    if _HAS_SCIPY and b.size > 1:
        try:
            b_kde = _subsample_1d(b, kde_max_points, rng=rng)
            kde_b = gaussian_kde(b_kde, bw_method="scott")
            ax.plot(xs, kde_b(xs), color="tab:orange", linewidth=1.8, linestyle="-")
        except Exception:
            pass
    if _HAS_SCIPY and a.size > 1:
        try:
            a_kde = _subsample_1d(a, kde_max_points, rng=rng)
            kde_a = gaussian_kde(a_kde, bw_method="scott")
            ax.plot(xs, kde_a(xs), color="tab:blue", linewidth=1.8, linestyle="-")
        except Exception:
            pass

    # 标准正态参考曲线
    if add_normal_ref:
        ys_norm = sp_norm.pdf(xs, 0.0, 1.0) if _HAS_SCIPY else _normal_pdf(xs)
        ax.plot(xs, ys_norm, "k--", linewidth=1.6,
                label=r"$\mathcal{N}(0,1)$")

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.30)
    if title:
        ax.set_title(title, fontsize=14)

    # 标注统计量（均值 ± 标准差）
    if show_stats:
        stats_text = (
            f"Y:     μ={b.mean():.3f}, σ={b.std():.3f}\n"
            f"Z_PDN: μ={a.mean():.3f}, σ={a.std():.3f}"
        )
        ax.text(
            0.02, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
        )


# ---------------------------------------------------------------------------
# 核心绘图函数
# ---------------------------------------------------------------------------

def plot_overall(
    Y: np.ndarray,
    Z_PDN: np.ndarray,
    output_path: str,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    add_normal_ref: bool = True,
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    绘制整体密度分布对比图（将 [N, P, D] 全部展平后对比）。

    Args:
        Y          : 真实未来序列，shape [N, P, D]。
        Z_PDN      : PDN 归一化后残差，shape [N, P, D]。
        output_path: 输出图片路径。
        clip_range : 截断区间，None 表示不截断。
        bins       : 直方图分桶数。
        add_normal_ref: 是否叠加标准正态参考曲线。
        show_stats    : 是否在图内标注统计量。
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=600)

    _plot_density_on_ax(
        ax=ax,
        data_before=Y,
        data_after=Z_PDN,
        clip_range=clip_range,
        bins=bins,
        label_before="Y  (before PDN)",
        label_after=r"$Z_{\rm PDN}$  (after PDN)",
        add_normal_ref=add_normal_ref,
        title=f"Overall Density: Y vs $Z_{{\\rm PDN}}$  [N={Y.shape[0]}, P={Y.shape[1]}, D={Y.shape[2]}]",
        show_stats=show_stats,
        kde_max_points=kde_max_points,
        rng=rng,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Overall] Saved to: {output_path}")


def plot_window(
    Y: np.ndarray,
    Z_PDN: np.ndarray,
    output_path: str,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    feature_dims: Optional[List[int]],
    n_cols: int,
    window_start: int = 0,
    window_size: int = 12,
    segment_size: int = 4,
    add_normal_ref: bool = True,
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    绘制窗口密度图：选定特征维度，对 N 轴上 window_size 个连续样本按 segment_size
    均分为若干段，每段（dim, segment）对绘制一个子图，合并为 n_cols×n_rows 的 grid 图。

    Args:
        Y             : 真实未来序列，shape [N, P, D]。
        Z_PDN         : PDN 归一化后残差，shape [N, P, D]。
        output_path   : 输出图片路径。
        clip_range    : 截断区间，None 表示不截断。
        bins          : 直方图分桶数。
        feature_dims  : 要展示的特征维度索引列表；None 则自动选取最多 12 个均匀分布的维度。
        n_cols        : grid 列数。
        window_start  : 12 个连续样本在 N 轴的起始索引。
        window_size   : 连续样本总数（默认 12）。
        segment_size  : 每段样本数（默认 4），n_segments = window_size // segment_size。
        add_normal_ref: 是否叠加标准正态参考曲线。
        show_stats    : 是否在每个子图内标注统计量。
    """
    N, P, D = Y.shape
    dims = _resolve_feature_dims(feature_dims, D)
    if dims is None:
        return

    n_segments = window_size // segment_size

    # 校验起始索引
    end_idx = window_start + window_size
    if end_idx > N:
        print(
            f"[Window] 警告: window_start={window_start} + window_size={window_size} = "
            f"{end_idx} 超出 N={N}，自动截断至 N。"
        )
        end_idx = N

    # 子图顺序：dim 为外层，segment 为内层（每行一个 dim）
    n_plots = len(dims) * n_segments
    fig, gs, n_rows, nc = _build_grid_fig(n_plots, n_cols)

    for i, d in enumerate(dims):
        for s in range(n_segments):
            si = window_start + s * segment_size
            ei = min(si + segment_size, N)
            Y_seg = Y[si:ei, :, d]      # [segment_size, P]
            Z_seg = Z_PDN[si:ei, :, d]
            idx = i * n_segments + s
            row, col = divmod(idx, nc)
            ax = fig.add_subplot(gs[row, col])
            _plot_density_on_ax(
                ax=ax,
                data_before=Y_seg,
                data_after=Z_seg,
                clip_range=clip_range,
                bins=bins,
                label_before="Y",
                label_after=r"$Z_{\rm PDN}$",
                add_normal_ref=add_normal_ref,
                title=f"dim={d}  seg{s + 1} [N={si}:{ei}]",
                show_stats=show_stats,
                kde_max_points=kde_max_points,
                rng=rng,
            )

    _hide_extra_subplots(fig, gs, n_plots, n_rows, nc)
    fig.suptitle(
        rf"Window Density: Y vs $Z_{{\rm PDN}}$  "
        rf"[start={window_start}, size={window_size}, seg={segment_size}]",
        fontsize=12,
        y=1.01,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Window] {len(dims)} dims × {n_segments} segs → Saved to: {output_path}")


def plot_window_separate(
    Y: np.ndarray,
    Z_PDN: np.ndarray,
    output_path_y: str,
    output_path_zpdn: str,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    feature_dims: Optional[List[int]],
    n_cols: int,
    window_start: Optional[List[int]] = None,
    window_size: int = 12,
    segment_size: int = 4,
    add_normal_ref: bool = False,
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    窗口密度图的分开模式：将 Y 与 Z_PDN **各自独立**绘制成一张 grid 图。

    每张图的子图排布与 plot_window() 相同（dim 为外层，segment 为内层），
    但每个子图只绘制单条分布（复用 _plot_single_on_ax）。

    生成两个文件：
      - output_path_y    : 每个子图仅显示 Y[si:ei, :, d] 的密度分布
      - output_path_zpdn : 每个子图仅显示 Z_PDN[si:ei, :, d] 的密度分布

    Args:
        Y               : 真实未来序列，shape [N, P, D]。
        Z_PDN           : PDN 归一化后残差，shape [N, P, D]。
        output_path_y   : Y grid 图输出路径。
        output_path_zpdn: Z_PDN grid 图输出路径。
        clip_range      : 截断区间，None 表示不截断。
        bins            : 直方图分桶数。
        feature_dims    : 要展示的特征维度索引列表；None 则自动选取最多 12 个均匀分布的维度。
        n_cols          : grid 列数。
        window_start    : N 轴起始索引列表，每个 start 都会作为独立窗口拼到同一张图里。
        window_size     : 连续样本总数（默认 12）。
        segment_size    : 每段样本数（默认 4）。
        add_normal_ref  : 是否叠加标准正态参考曲线。
        show_stats      : 是否在每个子图内标注统计量。
    """
    N, P, D = Y.shape
    dims = _resolve_feature_dims(feature_dims, D)
    if dims is None:
        return

    raw_starts = [0] if window_start is None else [int(ws) for ws in window_start]
    starts = [ws for ws in raw_starts if 0 <= ws < N]
    invalid_starts = [ws for ws in raw_starts if ws < 0 or ws >= N]
    if invalid_starts:
        print(f"[Window separate] 警告: 以下 window_start 越界，已忽略: {invalid_starts}")
    if len(starts) == 0:
        print("[Window separate] window_start 为空，跳过。")
        return

    n_segments = window_size // segment_size
    n_plots = len(starts) * len(dims) * n_segments

    for (arr, color, var_label, out_path, suptitle) in [
        (
            Y, "tab:orange", "Y", output_path_y,
            rf"Window Density: $Y$ (before PDN)  "
            rf"[starts={starts}, size={window_size}, seg={segment_size}]",
        ),
        (
            Z_PDN, "tab:blue", r"$Z_{\rm PDN}$", output_path_zpdn,
            rf"Window Density: $Z_{{\rm PDN}}$ (after PDN)  "
            rf"[starts={starts}, size={window_size}, seg={segment_size}]",
        ),
    ]:
        plt.close("all")
        fig, gs, n_rows, nc = _build_grid_fig(n_plots, n_cols)

        for w_idx, ws in enumerate(starts):
            for i, d in enumerate(dims):
                for s in range(n_segments):
                    si = ws + s * segment_size
                    ei = min(si + segment_size, N)
                    seg_data = arr[si:ei, :, d]     # [segment_size, P]
                    idx = (w_idx * len(dims) + i) * n_segments + s
                    row, col = divmod(idx, nc)
                    ax = fig.add_subplot(gs[row, col])
                    _plot_single_on_ax(
                        ax=ax,
                        data=seg_data,
                        clip_range=clip_range,
                        bins=bins,
                        color=color,
                        label=var_label,
                        add_normal_ref=add_normal_ref,
                        title=f"w={ws} dim={d} seg{s + 1} [N={si}:{ei}]",
                        show_stats=show_stats,
                        kde_max_points=kde_max_points,
                        rng=rng,
                    )

        _hide_extra_subplots(fig, gs, n_plots, n_rows, nc)
        fig.suptitle(suptitle, fontsize=12, y=1.01)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(
            f"[Window separate] {len(starts)} starts × {len(dims)} dims × {n_segments} segs"
            f" → Saved to: {out_path}"
        )


def _resolve_feature_dims(feature_dims: Optional[List[int]], D: int) -> Optional[List[int]]:
    """统一处理 feature_dims：None 时自动选取，否则过滤越界值。返回 None 表示可跳过。"""
    if feature_dims is None:
        n_show = min(12, D)
        return list(np.linspace(0, D - 1, n_show, dtype=int))
    valid = [d for d in feature_dims if 0 <= d < D]
    if not valid:
        print(f"[Per-feature] 指定的 feature_dims 全部超出范围 [0, {D-1}]，跳过。")
        return None
    return valid


def _build_grid_fig(
    n_plots: int,
    n_cols: int,
) -> Tuple["plt.Figure", "gridspec.GridSpec", int, int]:
    """创建 grid 画布，返回 (fig, gs, n_rows, n_cols_actual)。"""
    n_cols = min(n_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 3.8), dpi=600)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)
    return fig, gs, n_rows, n_cols


def _hide_extra_subplots(
    fig: "plt.Figure",
    gs: "gridspec.GridSpec",
    n_plots: int,
    n_rows: int,
    n_cols: int,
) -> None:
    """隐藏 grid 中多余的子图槽位。"""
    for j in range(n_plots, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)


def plot_per_feature(
    Y: np.ndarray,
    Z_PDN: np.ndarray,
    output_path: str,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    feature_dims: Optional[List[int]],
    n_cols: int,
    add_normal_ref: bool = True,
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    对指定的特征维度分别绘制 Y 与 Z_PDN **叠加**的密度对比图，合并为一张 grid 图。

    Args:
        Y            : 真实未来序列，shape [N, P, D]。
        Z_PDN        : PDN 归一化后残差，shape [N, P, D]。
        output_path  : 输出图片路径。
        clip_range   : 截断区间，None 表示不截断。
        bins         : 直方图分桶数。
        feature_dims : 要展示的特征维度索引列表；None 则自动选取最多 12 个均匀分布的维度。
        n_cols       : grid 列数。
        add_normal_ref: 是否叠加标准正态参考曲线。
        show_stats    : 是否在每个子图内标注统计量。
    """
    D = Y.shape[2]
    dims = _resolve_feature_dims(feature_dims, D)
    if dims is None:
        return

    n_plots = len(dims)
    fig, gs, n_rows, n_cols = _build_grid_fig(n_plots, n_cols)

    for i, d in enumerate(dims):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        _plot_density_on_ax(
            ax=ax,
            data_before=Y[:, :, d],
            data_after=Z_PDN[:, :, d],
            clip_range=clip_range,
            bins=bins,
            label_before="Y",
            label_after=r"$Z_{\rm PDN}$",
            add_normal_ref=add_normal_ref,
            title=f"Feature dim = {d}",
            show_stats=show_stats,
            kde_max_points=kde_max_points,
            rng=rng,
        )

    _hide_extra_subplots(fig, gs, n_plots, n_rows, n_cols)
    fig.suptitle(r"Per-Feature Density: Y vs $Z_{\rm PDN}$", fontsize=12, y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Per-feature combined] {n_plots} dims → Saved to: {output_path}")


def plot_per_feature_separate(
    Y: np.ndarray,
    Z_PDN: np.ndarray,
    output_path_y: str,
    output_path_zpdn: str,
    clip_range: Optional[Tuple[float, float]],
    bins: int,
    feature_dims: Optional[List[int]],
    n_cols: int,
    add_normal_ref: bool = True,
    show_stats: bool = True,
    kde_max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    对指定的特征维度，分别将 Y 与 Z_PDN **各自独立**绘制成一张 grid 图。

    生成两个文件：
      - output_path_y    : 每个子图仅显示 Y[:, :, d] 的密度分布
      - output_path_zpdn : 每个子图仅显示 Z_PDN[:, :, d] 的密度分布

    Args:
        Y               : 真实未来序列，shape [N, P, D]。
        Z_PDN           : PDN 归一化后残差，shape [N, P, D]。
        output_path_y   : Y grid 图输出路径。
        output_path_zpdn: Z_PDN grid 图输出路径。
        clip_range      : 截断区间，None 表示不截断。
        bins            : 直方图分桶数。
        feature_dims    : 要展示的特征维度索引列表；None 则自动选取最多 12 个均匀分布的维度。
        n_cols          : grid 列数。
        add_normal_ref  : 是否叠加标准正态参考曲线。
        show_stats      : 是否在每个子图内标注统计量。
    """
    D = Y.shape[2]
    dims = _resolve_feature_dims(feature_dims, D)
    if dims is None:
        return

    n_plots = len(dims)

    for (arr, color, var_label, out_path, suptitle) in [
        (Y,    "tab:orange", "Y",                   output_path_y,
         r"Per-Feature Density: $Y$ (before PDN)"),
        (Z_PDN, "tab:blue",  r"$Z_{\rm PDN}$",      output_path_zpdn,
         r"Per-Feature Density: $Z_{\rm PDN}$ (after PDN)"),
    ]:
        plt.close("all")
        fig, gs, n_rows, nc = _build_grid_fig(n_plots, n_cols)

        for i, d in enumerate(dims):
            row, col = divmod(i, nc)
            ax = fig.add_subplot(gs[row, col])
            _plot_single_on_ax(
                ax=ax,
                data=arr[:, :, d],
                clip_range=clip_range,
                bins=bins,
                color=color,
                label=var_label,
                add_normal_ref=add_normal_ref,
                title=f"Feature dim = {d}",
                show_stats=show_stats,
                kde_max_points=kde_max_points,
                rng=rng,
            )

        _hide_extra_subplots(fig, gs, n_plots, n_rows, nc)
        fig.suptitle(suptitle, fontsize=12, y=1.01)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Per-feature separate] {n_plots} dims → Saved to: {out_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot probability density distributions of Y (before PDN) "
            "and Z_PDN (after PDN) from a residuals .npz file."
        )
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="./results/analysis/electricity_residuals.npz",
        help="Path to .npz file generated by extract_residuals_on_test.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/analysis",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="electricity",
        help="Filename prefix for output images (e.g., 'electricity').",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=-8.0,
        help="Lower clipping bound for Y. Use --no_clip_y to disable Y clipping.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=8.0,
        help="Upper clipping bound for Y. Use --no_clip_y to disable Y clipping.",
    )
    parser.add_argument(
        "--zpdn_clip_min",
        type=float,
        default=-6.0,
        help="Lower clipping bound for Z_PDN (independent of Y clip).",
    )
    parser.add_argument(
        "--zpdn_clip_max",
        type=float,
        default=6.0,
        help="Upper clipping bound for Z_PDN (independent of Y clip).",
    )
    parser.add_argument(
        "--unified_clip_min",
        type=float,
        default=None,
        help=(
            "If set, override both --clip_min and --zpdn_clip_min with this value, "
            "forcing Y and Z_PDN to share the same x-axis range."
        ),
    )
    parser.add_argument(
        "--unified_clip_max",
        type=float,
        default=None,
        help="See --unified_clip_min.",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable all clipping (Y and Z_PDN shown on their natural range).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--sample_n_overall",
        type=int,
        default=None,
        help=(
            "If set, randomly subsample along N dimension before plotting the overall figure. "
            "Greatly reduces runtime for huge .npz."
        ),
    )
    parser.add_argument(
        "--sample_n_per_feature",
        type=int,
        default=None,
        help=(
            "If set, randomly subsample along N dimension before plotting per-feature figures."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling (reproducible plots).",
    )
    parser.add_argument(
        "--kde_max_points",
        type=int,
        default=200000,
        help=(
            "Max number of 1D points used for KDE. If flattened data exceeds this, KDE "
            "will be computed on a random subset to keep runtime manageable."
        ),
    )
    parser.add_argument(
        "--feature_dims",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Specific feature dimension indices to plot in the per-feature figure. "
            "If omitted, up to 12 evenly-spaced dimensions are selected automatically."
        ),
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=3,
        help="Number of columns in the per-feature / window grid figure.",
    )
    parser.add_argument(
        "--window_start",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "One or more starting indices along the N axis for the window plot "
            "(default: [0]). In default overlay mode, each value produces one file "
            "with _w{start}. In --separate_window mode, all starts are merged into "
            "a single Y figure and a single Z_PDN figure."
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=12,
        help="Total number of consecutive samples for the window plot (default: 12).",
    )
    parser.add_argument(
        "--window_segment_size",
        type=int,
        default=4,
        help=(
            "Number of consecutive samples per segment in the window plot (default: 4). "
            "n_segments = window_size // window_segment_size."
        ),
    )
    parser.add_argument(
        "--skip_window",
        action="store_true",
        help="Skip the window (all-dims-combined) figure.",
    )
    parser.add_argument(
        "--skip_per_feature",
        action="store_true",
        help="Skip the per-feature-dimension figure.",
    )
    parser.add_argument(
        "--separate_window",
        action="store_true",
        help=(
            "When set, plot Y and Z_PDN in two SEPARATE grid figures for the window plot "
            "(one for Y, one for Z_PDN) instead of overlaying them. "
            "All window_start values are merged into those two figures. "
            "Output files: {prefix}_pdn_density_window_Y.png and "
            "{prefix}_pdn_density_window_ZPDN.png."
        ),
    )
    parser.add_argument(
        "--separate_per_feature",
        action="store_true",
        help=(
            "When set, plot Y and Z_PDN in two SEPARATE grid figures "
            "(one for Y, one for Z_PDN) instead of overlaying them. "
            "Output files: {prefix}_pdn_density_per_feature_Y.png and "
            "{prefix}_pdn_density_per_feature_ZPDN.png."
        ),
    )
    parser.add_argument(
        "--no_stats",
        action="store_true",
        help="Suppress the in-figure statistics annotation (mean ± std).",
    )
    parser.add_argument(
        "--no_normal_ref",
        action="store_true",
        help="Do not draw the standard normal reference curve.",
    )

    args = parser.parse_args()

    show_stats: bool = not args.no_stats
    add_normal_ref: bool = not args.no_normal_ref
    rng = np.random.default_rng(args.seed)

    # ---- 加载数据 ----
    print(f"Loading: {args.npz_path}")
    data = np.load(args.npz_path)
    required = {"Y", "Z_PDN"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(
            f"{args.npz_path} 中缺少以下键: {missing}。\n"
            "请确认已通过 `extract_residuals_on_test` 生成该文件。"
        )

    Y = data["Y"]           # [N, P, D]
    Z_PDN = data["Z_PDN"]   # [N, P, D]
    assert Y.shape == Z_PDN.shape, (
        f"Y.shape={Y.shape} 与 Z_PDN.shape={Z_PDN.shape} 不一致！"
    )
    N, P, D = Y.shape
    print(f"Data shape: N={N}, P={P}, D={D}")
    if args.sample_n_overall is not None:
        print(f"sample_n_overall={args.sample_n_overall} (seed={args.seed})")
    if args.sample_n_per_feature is not None:
        print(f"sample_n_per_feature={args.sample_n_per_feature} (seed={args.seed})")
    if args.kde_max_points is not None:
        print(f"kde_max_points={args.kde_max_points}")

    # ---- 确定截断区间 ----
    if args.no_clip:
        clip_overall: Optional[Tuple[float, float]] = None
        clip_feat: Optional[Tuple[float, float]] = None
    elif args.unified_clip_min is not None and args.unified_clip_max is not None:
        clip_overall = (args.unified_clip_min, args.unified_clip_max)
        clip_feat = clip_overall
    else:
        # 整体图/特征图共用 Z_PDN 的 clip（Y 量级通常更大，以 Z_PDN 范围为坐标轴）
        clip_overall = (args.zpdn_clip_min, args.zpdn_clip_max)
        clip_feat = clip_overall

    # ---- 窗口图 ----
    if not args.skip_window:
        if args.separate_window:
            # 分开模式：所有 window_start 合并后，Y 与 Z_PDN 各自一张 grid 图
            plot_window_separate(
                Y=Y,
                Z_PDN=Z_PDN,
                output_path_y=os.path.join(
                    args.output_dir, f"{args.prefix}_pdn_density_window_Y.png"
                ),
                output_path_zpdn=os.path.join(
                    args.output_dir, f"{args.prefix}_pdn_density_window_ZPDN.png"
                ),
                clip_range=clip_overall,
                bins=args.bins,
                feature_dims=args.feature_dims,
                n_cols=args.n_cols,
                window_start=args.window_start,
                window_size=args.window_size,
                segment_size=args.window_segment_size,
                add_normal_ref=add_normal_ref,
                show_stats=show_stats,
                kde_max_points=args.kde_max_points,
                rng=rng,
            )
        else:
            # 默认叠加模式：每个 window_start 输出一张图
            for ws in args.window_start:
                plot_window(
                    Y=Y,
                    Z_PDN=Z_PDN,
                    output_path=os.path.join(
                        args.output_dir, f"{args.prefix}_pdn_density_window_w{ws}.png"
                    ),
                    clip_range=clip_overall,
                    bins=args.bins,
                    feature_dims=args.feature_dims,
                    n_cols=args.n_cols,
                    window_start=ws,
                    window_size=args.window_size,
                    segment_size=args.window_segment_size,
                    add_normal_ref=add_normal_ref,
                    show_stats=show_stats,
                    kde_max_points=args.kde_max_points,
                    rng=rng,
                )

    # ---- 特征维度图 ----
    if not args.skip_per_feature:
        Y_feat, Z_feat = _maybe_subsample_N(
            Y, Z_PDN, args.sample_n_per_feature, rng=rng, tag="per_feature"
        )
        if args.separate_per_feature:
            # 分开模式：Y 与 Z_PDN 各自一张 grid 图
            plot_per_feature_separate(
                Y=Y_feat,
                Z_PDN=Z_feat,
                output_path_y=os.path.join(
                    args.output_dir, f"{args.prefix}_pdn_density_per_feature_Y.png"
                ),
                output_path_zpdn=os.path.join(
                    args.output_dir, f"{args.prefix}_pdn_density_per_feature_ZPDN.png"
                ),
                clip_range=clip_feat,
                bins=args.bins,
                feature_dims=args.feature_dims,
                n_cols=args.n_cols,
                add_normal_ref=add_normal_ref,
                show_stats=show_stats,
                kde_max_points=args.kde_max_points,
                rng=rng,
            )
        else:
            # 默认叠加模式：Y 与 Z_PDN 画在同一张 grid 图
            plot_per_feature(
                Y=Y_feat,
                Z_PDN=Z_feat,
                output_path=os.path.join(
                    args.output_dir, f"{args.prefix}_pdn_density_per_feature.png"
                ),
                clip_range=clip_feat,
                bins=args.bins,
                feature_dims=args.feature_dims,
                n_cols=args.n_cols,
                add_normal_ref=add_normal_ref,
                show_stats=show_stats,
                kde_max_points=args.kde_max_points,
                rng=rng,
            )

    print("Done.")


if __name__ == "__main__":
    main()
