import pytest
import torch

from shared_residual.rectified_lpjepa_sae import (
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    axis_aligned_distribution_matching_loss,
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
            high_k=2,
            low_k=4,
            max_span_length=6,
            high_fraction=0.2,
            target_active_fraction=0.75,
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


def test_dense_high_is_relu_and_reconstruction_high_is_topk() -> None:
    model = make_model()
    assert model.cfg.d_high == 4
    assert model.cfg.d_low == 16
    with torch.no_grad():
        model.encoder.linear.weight.zero_()
        model.encoder.linear.bias.fill_(1.0)
    code, dense_high = model.encode_with_dense_high(torch.randn(3, 8))
    sparse_high, low = model.split_code(code)
    assert torch.all(dense_high > 0)
    assert torch.all((sparse_high > 0).sum(dim=-1) == model.cfg.high_k)
    assert torch.all((low > 0).sum(dim=-1) == model.cfg.low_k)


def test_model_has_no_predictor_or_horizon_conditioning() -> None:
    model = make_model()
    outputs = model(torch.randn(2, 8), torch.randn(2, 8))
    assert "predicted_codes" not in outputs
    assert not hasattr(model, "transition_predictor")
    assert outputs["dense_high_a"].shape == (2, model.cfg.d_high)
    assert outputs["sparse_high_a"].shape == (2, model.cfg.d_high)
    expected = outputs["high_reconstruction_a"] + model.decode_low(
        outputs["low_a"], add_bias=False
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


def test_loss_updates_the_single_sae() -> None:
    model = make_model()
    loss, metrics = rectified_lpjepa_loss(
        model,
        torch.randn(6, 8),
        torch.randn(6, 8),
        invariance_weight=1.0,
        rdm_weight=2.0,
        rdm_projections=8,
        rdm_projection_chunk_size=4,
        axis_rdm_features=3,
        axis_rdm_weight=1.0,
    )
    loss.backward()
    assert model.encoder.linear.weight.grad is not None
    assert model.decoder.grad is not None
    assert not hasattr(model, "ema_encoder")
    assert not hasattr(model, "ema_decoder")
    assert metrics["low_l0"] <= model.cfg.low_k
    assert "rdm_loss" in metrics
    assert "high_positive_margin" in metrics
    assert "dense_high_positive_margin" in metrics
    assert metrics["high_l0"] <= model.cfg.high_k
    assert metrics["dense_high_l0"] >= metrics["high_l0"]


def test_rdm_loss_is_finite_and_differentiable() -> None:
    model = make_model()
    first = torch.rand(12, model.cfg.d_high, requires_grad=True)
    second = torch.rand(12, model.cfg.d_high, requires_grad=True)
    loss, metrics = rectified_distribution_matching_loss(
        (first, second), model.cfg, projections=12, projection_chunk_size=5,
        axis_features=3, axis_weight=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["random_projection"])
    assert torch.isfinite(metrics["axis_aligned"])
    assert int(metrics["axis_sampled_features"]) == 3
    assert first.grad is not None and second.grad is not None


def test_initialization_covers_full_high_low_sae() -> None:
    model = make_model()
    assert torch.allclose(
        model.encoder.linear.bias[: model.cfg.d_high],
        torch.full(
            (model.cfg.d_high,),
            model.cfg.target_mu,
            dtype=model.encoder.linear.bias.dtype,
        ),
    )
    assert torch.allclose(
        model.decoder.norm(dim=1), torch.ones(model.cfg.d_sae), atol=1e-5
    )


def test_axis_rdm_is_zero_for_identical_coordinate_distributions() -> None:
    target = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
    permuted = target.index_select(0, torch.tensor([2, 0, 1])).requires_grad_()
    loss, sampled = axis_aligned_distribution_matching_loss(
        (permuted,), target, features=2
    )
    loss.backward()
    assert float(loss) == pytest.approx(0.0, abs=1e-8)
    assert int(sampled) == 2
    assert permuted.grad is not None


def test_axis_rdm_zero_features_is_an_exact_ablation() -> None:
    target = torch.rand(4, 3)
    view = torch.rand(4, 3, requires_grad=True)
    loss, sampled = axis_aligned_distribution_matching_loss(
        (view,), target, features=0
    )
    loss.backward()
    assert float(loss) == 0.0
    assert int(sampled) == 0
    assert view.grad is not None
