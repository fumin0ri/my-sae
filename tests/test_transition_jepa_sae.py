import torch

from shared_residual.transition_jepa_sae import (
    TransitionJEPAConfig,
    TransitionJEPASAE,
    transition_jepa_loss,
)


def make_model() -> TransitionJEPASAE:
    model = TransitionJEPASAE(
        TransitionJEPAConfig(
            d_in=8,
            d_sae=20,
            k=5,
            window_size=6,
            predictor_width=8,
            high_fraction=0.2,
        )
    )
    model.initialize_from_statistics(torch.zeros(8), 1.0)
    return model


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
    outputs = model(torch.randn(2, 6, 8))
    assert outputs["predicted_codes"].shape == (2, 5, 4)
    assert outputs["target_codes"].shape == (2, 5, 4)
    assert outputs["target_low_code"].shape == (2, 16)
    expected = outputs["online_high_reconstruction"] + model.decode_low(
        outputs["online_target_low"], ema=False, add_bias=False
    )
    assert torch.allclose(outputs["online_target_reconstruction"], expected)


def test_position_only_control_is_independent_of_context_code() -> None:
    model = make_model()
    positions = torch.arange(model.cfg.window_size - 1)
    first = torch.randn(2, model.cfg.window_size - 1, model.cfg.d_high)
    second = torch.randn_like(first)
    first_prediction = model.predict_from_code(
        first, positions, use_context=False
    )
    second_prediction = model.predict_from_code(
        second, positions, use_context=False
    )
    assert torch.allclose(first_prediction, second_prediction)


def test_online_forecast_encoder_matches_online_high_partition() -> None:
    model = make_model()
    x = torch.randn(2, 8)
    expected, _ = model.split_code(model.encode(x))
    assert torch.allclose(model.encode_forecast_online(x), expected)


def test_loss_updates_online_sae_and_predictor_not_ema() -> None:
    model = make_model()
    loss, metrics = transition_jepa_loss(
        model,
        torch.randn(4, 6, 8),
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
