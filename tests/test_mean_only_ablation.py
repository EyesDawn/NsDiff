from types import SimpleNamespace

import torch

from src.metrics import EnergyScore
from src.models.iReflow import iReflow


def make_config(mode="mean_only"):
    return SimpleNamespace(
        seq_len=4,
        pred_len=3,
        d_model=8,
        n_heads=2,
        e_layers=1,
        flow_layers=1,
        d_ff=16,
        dropout=0.0,
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        use_norm=True,
        class_strategy="projection",
        factor=1,
        use_relative_space=True,
        num_sampling_steps=1,
        x0_dist="pred_gaussian",
        is_training=1,
        ablation_mode=mode,
    )


def test_mean_only_has_no_sigma_parameters_and_uses_unit_scale():
    model = iReflow(make_config())
    assert model.uncertainty_estimator is None
    assert not any(name.startswith("uncertainty_estimator") for name, _ in model.named_parameters())

    x = torch.randn(2, 4, 2)
    marks = torch.zeros(2, 4, 1)
    _, _, scale = model.get_encoder_features(x, marks)
    assert torch.equal(scale, torch.ones_like(scale))

    mean = torch.randn(2, 3, 2)
    noise = torch.randn_like(mean)
    assert torch.equal(model._build_x0(mean, torch.ones_like(mean), noise), mean + noise)


def test_energy_score_matches_scalar_ensemble_definition():
    pred = torch.tensor([[[[0.0, 2.0]]]])
    truth = torch.tensor([[[1.0]]])
    metric = EnergyScore()
    metric.update(pred, truth)
    # E|X-y| = 1 and (1 / 2S^2) sum_{i,j}|Xi-Xj| = 0.5.
    assert torch.isclose(metric.compute(), torch.tensor(0.5), atol=1e-6)


def test_mean_only_rejects_a_source_without_mu():
    config = make_config()
    config.x0_dist = "standard_normal"
    try:
        iReflow(config)
    except ValueError as exc:
        assert "Mean-only requires" in str(exc)
    else:
        raise AssertionError("Mean-only accepted a source distribution without mu(X).")
