import math
from typing import Dict, Iterable, List, Optional

import torch


def _normal_pdf(z: torch.Tensor) -> torch.Tensor:
    # φ(z) = (1/sqrt(2π)) exp(-0.5 z^2)
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Full Gaussian NLL: 0.5*log(2πσ^2) + 0.5*(y-mu)^2/σ^2
    Returns elementwise tensor with the same shape as y.
    """
    var = torch.clamp(sigma, min=eps) ** 2
    return 0.5 * torch.log(2.0 * math.pi * var) + 0.5 * (y - mu) ** 2 / var


def gaussian_crps(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    CRPS for Gaussian N(mu, sigma^2).
    Formula: CRPS = σ * [ z(2Φ(z)-1) + 2φ(z) - 1/√π ], z=(y-μ)/σ
    Returns elementwise tensor with the same shape as y.
    """
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    phi = _normal_pdf(z)
    Phi = _normal_cdf(z)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def gaussian_interval_coverage(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    levels: Iterable[float],
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute empirical coverage for central prediction intervals of a Gaussian.
    levels: e.g. [0.5, 0.8, 0.9, 0.95] meaning 50%, 80%, ...
    Returns dict: {"cov_90": 0.89, "width_90": 1.23, ...}
    """
    sigma = torch.clamp(sigma, min=eps)
    y = y.detach()
    mu = mu.detach()
    sigma = sigma.detach()

    out: Dict[str, float] = {}
    for lv in levels:
        lv_f = float(lv)
        # z_{(1+lv)/2} for standard normal
        # Use erfinv: Φ^{-1}(p) = sqrt(2) * erfinv(2p-1)
        p = (1.0 + lv_f) / 2.0
        z = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0)))
        half_width = z * sigma
        lower = mu - half_width
        upper = mu + half_width
        inside = (y >= lower) & (y <= upper)
        out[f"cov_{int(round(lv_f*100))}"] = float(inside.float().mean().item())
        out[f"width_{int(round(lv_f*100))}"] = float((2.0 * half_width).mean().item())
    return out


def pit_statistics(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    bins: int = 20,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    PIT u = Φ((y-μ)/σ). For a calibrated distribution, u ~ Uniform(0,1).
    Returns a few scalars: mean/var, KS statistic, and histogram L1 distance to uniform.
    """
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    u = _normal_cdf(z).clamp(0.0, 1.0)

    u_flat = u.reshape(-1)
    # mean/var
    pit_mean = float(u_flat.mean().item())
    pit_var = float(u_flat.var(unbiased=False).item())

    # KS statistic vs U(0,1)
    u_sorted, _ = torch.sort(u_flat)
    n = u_sorted.numel()
    if n == 0:
        ks = float("nan")
    else:
        # empirical CDF at each point i: i/n (1-indexed)
        i = torch.arange(1, n + 1, device=u_sorted.device, dtype=u_sorted.dtype)
        ecdf = i / n
        ks = float(torch.max(torch.abs(ecdf - u_sorted)).item())

    # histogram L1 distance to uniform
    if bins <= 0:
        pit_l1 = float("nan")
    else:
        hist = torch.histc(u_flat, bins=bins, min=0.0, max=1.0)
        hist = hist / torch.clamp(hist.sum(), min=1.0)
        uniform = torch.full_like(hist, 1.0 / bins)
        pit_l1 = float(torch.abs(hist - uniform).sum().item())

    return {
        "pit_mean": pit_mean,
        "pit_var": pit_var,
        "pit_ks": ks,
        "pit_hist_l1": pit_l1,
    }


def compute_sigma_metrics(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    interval_levels: Optional[List[float]] = None,
    pit_bins: int = 20,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Convenience wrapper returning a flat dict of commonly used calibration/sharpness metrics.
    All metrics are computed in the provided scale (so make sure y/mu/sigma are consistent).
    """
    if interval_levels is None:
        interval_levels = [0.5, 0.8, 0.9, 0.95]

    y = y.detach()
    mu = mu.detach()
    sigma = sigma.detach()

    nll = gaussian_nll(y, mu, sigma, eps=eps).mean()
    crps = gaussian_crps(y, mu, sigma, eps=eps).mean()
    sharp = torch.clamp(sigma, min=eps).mean()

    out: Dict[str, float] = {
        "gauss_nll": float(nll.item()),
        "gauss_crps": float(crps.item()),
        "sharpness_sigma_mean": float(sharp.item()),
        "sigma_min": float(torch.clamp(sigma, min=eps).min().item()),
        "sigma_max": float(torch.clamp(sigma, min=eps).max().item()),
    }
    out.update(gaussian_interval_coverage(y, mu, sigma, interval_levels, eps=eps))
    out.update(pit_statistics(y, mu, sigma, bins=pit_bins, eps=eps))
    return out


