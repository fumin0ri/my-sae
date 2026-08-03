import pytest
import torch

from shared_residual.rectified_lpjepa_sae import (
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    rectified_distribution_matching_loss,
    rectified_lpjepa_loss,
    rgg_mean_for_active_fraction,
    sample_rectified_generalized_gaussian,
    unit_variance_generalized_gaussian_sigma,
)


def make_model() -> RectifiedLpJEPASAE:
    model = RectifiedLpJEPASAE(
        RectifiedLpJEPAConfig(
            d_in=8,
            d_sae=20,
            low_k=4,
            max_span_length=6,
            high_fraction=0.2,
            target_active_fraction=0.1,
        )
    )
    model.initialize_from_statistics(torch.zeros(8), 1.0)
    return model


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_rgg_parameterization_controls_empirical_active_fraction(p: float) -> None:
    sigma = unit_variance_generalized_gaussian_sigma(p)
    mu = rgg_mean_for_active_fraction(p, 0.1, sigma)
    torch.manual_seed(1)
    samples = sample_rectified_generalized_gaussian(
        (200_000,),
        p=p,
        mu=mu,
        sigma=sigma,
        device=torch.device("cpu"),
    )
    assert abs(float((samples > 0).float().mean()) - 0.1) < 0.005


def test_high_is_shifted_relu_and_only_low_is_topk() -> None:
    model = make_model()
    assert model.cfg.d_high == 4
    assert model.cfg.d_low == 16
    with torch.no_grad():
        model.encoder.linear.weight.zero_()
        model.encoder.linear.bias.fill_(1.0)
    code = model.encode(torch.randn(3, 8))
    high, low = model.split_code(code)
    assert torch.all(high > 0)
    assert torch.all((low > 0).sum(dim=-1) == model.cfg.low_k)


def test_model_has_no_predictor_or_horizon_conditioning() -> None:
    model = make_model()
    outputs = model(torch.randn(2, 8), torch.randn(2, 8))
    assert "predicted_codes" not in outputs
    assert not hasattr(model, "transition_predictor")
    assert outputs["high_a"].shape == (2, model.cfg.d_high)
    expected = outputs["high_reconstruction_a"] + model.decode_low(
        outputs["low_a"], ema=False, add_bias=False
    )
    assert torch.allclose(outputs["full_reconstruction_a"], expected)


def test_identical_views_have_zero_invariance_error() -> None:
    model = make_model()
    x = torch.randn(4, 8)
    _, metrics = rectified_lpjepa_loss(
        model,
        x,
        x,
        invariance_weight=1.0,
        rdm_weight=1.0,
        rdm_projections=8,
        rdm_projection_chunk_size=4,
    )
    assert metrics["invariance_raw_mse"] == pytest.approx(0.0, abs=1e-8)


def test_loss_updates_online_sae_but_not_ema() -> None:
    model = make_model()
    loss, metrics = rectified_lpjepa_loss(
        model,
        torch.randn(6, 8),
        torch.randn(6, 8),
        invariance_weight=1.0,
        rdm_weight=2.0,
        rdm_projections=8,
        rdm_projection_chunk_size=4,
    )
    loss.backward()
    assert model.encoder.linear.weight.grad is not None
    assert model.decoder.grad is not None
    assert all(parameter.grad is None for parameter in model.ema_encoder.parameters())
    assert model.ema_decoder.grad is None
    assert metrics["low_l0"] <= model.cfg.low_k
    assert "rdm_loss" in metrics
    assert "high_positive_margin" in metrics


def test_rdm_loss_is_finite_and_differentiable() -> None:
    model = make_model()
    first = torch.rand(12, model.cfg.d_high, requires_grad=True)
    second = torch.rand(12, model.cfg.d_high, requires_grad=True)
    loss, metrics = rectified_distribution_matching_loss(
        (first, second), model.cfg, projections=12, projection_chunk_size=5
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["raw"])
    assert first.grad is not None and second.grad is not None


def test_initialization_and_ema_cover_full_high_low_sae() -> None:
    model = make_model()
    assert torch.allclose(
        model.encoder.linear.bias[: model.cfg.d_high],
        torch.full(
            (model.cfg.d_high,),
            model.cfg.target_mu,
            dtype=model.encoder.linear.bias.dtype,
        ),
    )
    before = model.ema_decoder.clone()
    with torch.no_grad():
        model.decoder.add_(0.1)
        model.encoder.linear.weight.add_(0.2)
    model.update_ema_sae(decay=0.5)
    assert not torch.equal(before, model.ema_decoder)
    assert torch.allclose(
        model.ema_decoder.norm(dim=1), torch.ones(model.cfg.d_sae), atol=1e-5
    )
