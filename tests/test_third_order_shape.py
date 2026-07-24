import torch

from src.models.third_order_shape import SoftplusSkewTransform, constrained_alpha


def test_zero_alpha_is_exact_identity():
    transform = SoftplusSkewTransform()
    z2 = torch.linspace(-6, 6, 101)
    alpha = torch.zeros_like(z2)
    assert torch.equal(transform(z2, alpha), z2)
    assert torch.equal(transform.log_abs_det_jacobian(z2, alpha), torch.zeros_like(z2))


def test_inverse_and_jacobian_are_valid_near_boundary():
    transform = SoftplusSkewTransform(inverse_steps=96)
    z2 = torch.linspace(-7, 7, 257)
    raw = torch.full_like(z2, -8.0)
    alpha = constrained_alpha(raw)
    z3 = transform(z2, alpha)
    reconstructed = transform.inverse(z3, alpha)
    # At alpha close to -1, a float32 forward value represents a very small
    # slope; the remaining error is therefore dominated by input quantization.
    assert torch.allclose(reconstructed, z2, atol=5e-4, rtol=5e-5)
    assert torch.isfinite(transform.log_abs_det_jacobian(z2, alpha)).all()
