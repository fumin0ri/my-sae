import math

import torch
import pytest

from shared_residual.transition_jepa_sae import (
    TransitionJEPAConfig,
    TransitionJEPASAE,
    horizon_loss_weight_table,
    horizon_sampling_probabilities,
    predictor_output_bias_init,
    transition_jepa_loss,
)


def make_model(predictor_output: str = "softplus") -> TransitionJEPASAE:
    model = TransitionJEPASAE(
        TransitionJEPAConfig(
            d_in=8,
            d_sae=20,
            k=5,
            max_span_length=6,
            predictor_width=8,
            predictor_output=predictor_output,
            high_fraction=0.2,
        )
    )
    model.initialize_from_statistics(torch.zeros(8), 1.0)
    return model


def test_inverse_probability_weights_equalize_expected_horizon_mass() -> None:
    probabilities = horizon_sampling_probabilities(2, 8)
    weights = horizon_loss_weight_table(2, 8, "inverse_probability").double()
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    expected_mass = probabilities[1:] * weights[1:]
    assert torch.allclose(
        expected_mass,
        torch.full_like(expected_mass, 1.0 / 7.0),
    )
    assert abs(float(probabilities[1] / probabilities[7]) - 18.15) < 0.02
    assert torch.allclose(
        horizon_loss_weight_table(2, 8, "none")[1:],
        torch.ones(7),
    )


def test_only_high_low_partition_exists() -> None:
    model = make_model()
    assert model.cfg.d_high == 4
    assert model.cfg.d_low == 16
    assert model.cfg.k_high == 1
    assert model.cfg.k_low == 4
    code = model.encode(torch.randn(3, 6, 8))
    high, low = model.split_code(code)
    assert high.shape == (3, 6, 4)
    assert low.shape == (3, 6, 16)
    assert torch.all((high > 0).sum(dim=-1) <= 1)
    assert torch.all((low > 0).sum(dim=-1) <= 4)


def test_prediction_supervises_only_high_endpoint() -> None:
    model = make_model()
    outputs = model(
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.tensor([1, 5]),
    )
    assert outputs["predicted_codes"].shape == (2, 4)
    assert outputs["target_codes"].shape == (2, 4)
    assert outputs["target_low_code"].shape == (2, 16)
    expected = outputs["online_high_reconstruction"] + model.decode_low(
        outputs["online_target_low"], ema=False, add_bias=False
    )
    assert torch.allclose(outputs["online_target_reconstruction"], expected)


def test_horizon_only_control_is_independent_of_context_code() -> None:
    model = make_model()
    horizons = torch.arange(model.cfg.max_span_length - 1, 0, -1)
    first = torch.randn(2, model.cfg.max_span_length - 1, model.cfg.d_high)
    second = torch.randn_like(first)
    first_prediction = model.predict_from_code(
        first, horizons, use_context=False
    )
    second_prediction = model.predict_from_code(
        second, horizons, use_context=False
    )
    assert torch.allclose(first_prediction, second_prediction)


def test_pair_predictor_uses_explicit_per_sample_horizon() -> None:
    model = make_model()
    context = torch.randn(3, model.cfg.d_high)
    output = model.predict_from_code(context, torch.tensor([1, 3, 5]))
    assert output.shape == (3, model.cfg.d_high)
    with pytest.raises(ValueError):
        model.predict_from_code(context, torch.tensor([0, 3, 5]))


def test_relu_topk_predictor_uses_non_dead_scale_matched_initialization() -> None:
    model = make_model("relu_topk")
    expected_bias = math.log1p(math.exp(-4.0))
    assert predictor_output_bias_init("relu_topk") == pytest.approx(expected_bias)
    assert torch.allclose(
        model.transition_predictor.output.bias,
        torch.full_like(model.transition_predictor.output.bias, expected_bias),
    )
    with torch.no_grad():
        model.transition_predictor.output.weight.zero_()
    prediction = model.predict_from_code(
        torch.randn(3, model.cfg.d_high), torch.tensor([1, 3, 5])
    )
    assert torch.all((prediction > 0).sum(dim=-1) == model.cfg.k_high)
    assert torch.all(prediction[prediction > 0] > 0)


def test_softplus_predictor_remains_dense_baseline() -> None:
    model = make_model("softplus")
    prediction = model.predict_from_code(
        torch.randn(3, model.cfg.d_high), torch.tensor([1, 3, 5])
    )
    assert torch.all(prediction > 0)
    assert torch.all(model.transition_predictor.output.bias == -4.0)


def test_online_forecast_encoder_matches_online_high_partition() -> None:
    model = make_model()
    x = torch.randn(2, 8)
    expected, _ = model.split_code(model.encode(x))
    assert torch.allclose(model.encode_forecast_online(x), expected)


def test_loss_updates_online_sae_and_predictor_not_ema() -> None:
    model = make_model()
    loss, metrics = transition_jepa_loss(
        model,
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.tensor([1, 2, 3, 5]),
        prediction_weight=1.0,
    )
    loss.backward()
    assert model.encoder.linear.weight.grad is not None
    assert model.decoder.grad is not None
    assert model.transition_predictor.output.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.ema_encoder.parameters())
    assert model.ema_decoder.grad is None
    assert metrics["high_l0"] <= model.cfg.k_high
    assert metrics["low_l0"] <= model.cfg.k_low
    assert metrics["prediction_loss"] == pytest.approx(
        metrics["prediction_loss_unweighted"]
    )
    assert metrics["mean_horizon_loss_weight"] == pytest.approx(1.0)
    assert "residual_prediction_fvu" not in metrics


def test_initialization_and_ema_cover_full_high_low_sae() -> None:
    model = make_model()
    before = model.ema_decoder.clone()
    with torch.no_grad():
        model.decoder.add_(0.1)
        model.encoder.linear.weight.add_(0.2)
    model.update_ema_sae(decay=0.5)
    assert not torch.equal(before, model.ema_decoder)
    assert torch.allclose(
        model.ema_decoder.norm(dim=1), torch.ones(model.cfg.d_sae), atol=1e-5
    )
