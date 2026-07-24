"""Conditional third-order (skewness) decomposition for LS-Flow.

The L3 model deliberately wraps a *frozen* L2 :class:`iReflow` checkpoint.
It never changes the L2 location/scale conditioner or its generator.  Instead
it learns a new, same-capacity generator in the transformed residual space.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftplusSkewTransform(nn.Module):
    """Elementwise monotone skew transform ``z3 = z2 + alpha * softplus(z2)``.

    ``alpha > -1`` is sufficient for invertibility because the Jacobian is
    ``1 + alpha * sigmoid(z2)``.  The transform is exactly the identity when
    ``alpha == 0``.  The inverse is evaluated by bracketed bisection; it is
    only used during sampling, where gradients are not required.
    """

    def __init__(self, alpha_floor: float = 1e-6, inverse_steps: int = 64):
        super().__init__()
        self.alpha_floor = float(alpha_floor)
        self.inverse_steps = int(inverse_steps)

    def forward(self, z2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return z2 + alpha * F.softplus(z2)

    def log_abs_det_jacobian(self, z2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        jacobian = 1.0 + alpha * torch.sigmoid(z2)
        if torch.any(jacobian <= 0):
            raise ValueError("Third-order transform has a non-positive Jacobian.")
        return torch.log(jacobian)

    @torch.no_grad()
    def inverse(self, z3: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Return ``z2`` such that ``forward(z2, alpha) == z3``.

        The lower derivative bound is ``alpha_floor``.  The resulting wide
        bracket remains valid even for alpha values close to -1.
        """
        original_dtype = z3.dtype
        # Near alpha=-1 the valid bracket can be large.  Evaluate bisection in
        # float64 so float32 spacing at that magnitude cannot limit accuracy.
        z3 = z3.double()
        alpha = alpha.double()
        min_slope = torch.clamp(1.0 + alpha, min=self.alpha_floor)
        radius = (z3.abs() + 30.0) / min_slope
        lo = -radius
        hi = radius
        for _ in range(self.inverse_steps):
            mid = (lo + hi) * 0.5
            value = self.forward(mid, alpha)
            lo = torch.where(value < z3, mid, lo)
            hi = torch.where(value >= z3, mid, hi)
        return ((lo + hi) * 0.5).to(dtype=original_dtype)


def constrained_alpha(raw_alpha: torch.Tensor, alpha_floor: float = 1e-6) -> torch.Tensor:
    """Map an unconstrained head output to ``(-1 + alpha_floor, inf)``."""
    return -1.0 + float(alpha_floor) + F.softplus(raw_alpha)


class ThirdOrderShapeModel(nn.Module):
    """Frozen L2 conditioner plus a trainable L3 alpha head and generator."""

    def __init__(self, l2_model: nn.Module, configs, alpha_floor: float = 1e-6):
        super().__init__()
        # Keep the pure transform usable in lightweight analysis/test
        # environments that do not install the optional Reformer dependency.
        from src.nn.velocity_network import VelocityNetwork

        self.l2 = l2_model
        self.pred_len = int(configs.pred_len)
        self.num_sampling_steps = int(configs.num_sampling_steps)
        self.alpha_floor = float(alpha_floor)
        self.transform = SoftplusSkewTransform(alpha_floor=alpha_floor)

        # The smallest practical conditional head: one horizon value per
        # encoder variable token.  Zero initialization makes the initial L3
        # mapping exactly L2.
        self.alpha_head = nn.Linear(configs.d_model, configs.pred_len)
        nn.init.zeros_(self.alpha_head.weight)
        alpha_zero_bias = math.log(math.expm1(1.0 - self.alpha_floor))
        nn.init.constant_(self.alpha_head.bias, alpha_zero_bias)

        # This is a new generator, but with exactly the submitted L2 capacity.
        # Direct coordinate mode means its state and velocity are both z3.
        self.generator = VelocityNetwork(
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.flow_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            use_relative_space=False,
        )
        self.freeze_l2()

    def freeze_l2(self) -> None:
        self.l2.eval()
        for parameter in self.l2.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep all submitted L2 modules in inference mode during L3 training."""
        super().train(mode)
        self.l2.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def generator_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.generator.parameters())

    def l2_generator_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.l2.velocity_net.parameters())

    def _condition(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calling the frozen L2 under no_grad protects the submitted modules
        # even if the parent model is in train() mode.
        with torch.no_grad():
            enc_features, mu, sigma = self.l2.get_encoder_features(x_enc, x_mark_enc)
        raw_alpha = self.alpha_head(enc_features).permute(0, 2, 1)
        alpha = constrained_alpha(raw_alpha, self.alpha_floor)
        return enc_features, mu, sigma, alpha

    def residuals(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_features, mu, sigma, alpha = self._condition(x_enc, x_mark_enc)
        z2 = (y - mu) / sigma
        z3 = self.transform(z2, alpha)
        return enc_features, mu, sigma, alpha, z2, z3

    def compute_loss(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, y: torch.Tensor, nll_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        enc_features, _, _, alpha, z2, z3 = self.residuals(x_enc, x_mark_enc, y)
        log_det = self.transform.log_abs_det_jacobian(z2, alpha)
        nll = (0.5 * z3.square() + 0.5 * math.log(2.0 * math.pi) - log_det).mean()

        epsilon = torch.randn_like(z3)
        tau = torch.rand(z3.shape[0], 1, 1, device=z3.device)
        z_tau = tau * z3 + (1.0 - tau) * epsilon
        target_velocity = z3 - epsilon
        zeros = torch.zeros_like(z3)
        ones = torch.ones_like(z3)
        predicted_velocity = self.generator(z_tau, tau.squeeze(-1).squeeze(-1), enc_features, zeros, ones)
        cfm_loss = F.mse_loss(predicted_velocity, target_velocity)
        total = cfm_loss + float(nll_weight) * nll
        return total, {
            "total_loss": float(total.detach()),
            "cfm_loss": float(cfm_loss.detach()),
            "alpha_nll": float(nll.detach()),
            "mean_abs_alpha": float(alpha.detach().abs().mean()),
            "min_jacobian": float((1.0 + alpha.detach() * torch.sigmoid(z2.detach())).min()),
        }

    @torch.no_grad()
    def forecast(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, num_samples: int = 100, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_features, mu, sigma, alpha = self._condition(x_enc, x_mark_enc)
        batch, _, dimensions = mu.shape
        generated = []
        zeros = torch.zeros_like(mu)
        ones = torch.ones_like(mu)
        steps = max(1, self.num_sampling_steps)
        step_size = 1.0 / steps

        for _ in range(num_samples):
            z3 = temperature * torch.randn(batch, self.pred_len, dimensions, device=mu.device)
            for step in range(steps):
                tau = torch.full((batch,), step * step_size, device=mu.device)
                velocity = self.generator(z3, tau, enc_features, zeros, ones)
                z3 = z3 + velocity * step_size
            z2 = self.transform.inverse(z3, alpha)
            generated.append(mu + sigma * z2)
        return torch.stack(generated, dim=1), mu, sigma, alpha
