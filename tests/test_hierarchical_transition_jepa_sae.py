import torch

from shared_residual.hierarchical_transition_jepa_sae import (
    HierarchicalTransitionJEPAConfig,
    HierarchicalTransitionJEPASAE,
    hierarchical_transition_jepa_loss,
)
from shared_residual.standard_sae import (
    StandardSAEConfig,
    StandardSparseAutoencoder,
)


def make_model() -> HierarchicalTransitionJEPASAE:
    return HierarchicalTransitionJEPASAE(
        HierarchicalTransitionJEPAConfig(
            d_in=8,
            d_sae=20,
            k=5,
            window_size=6,
            predictor_width=8,
            high_fraction=0.2,
            high_reconstruction_weight=0.2,
        )
    )


def test_partition_preserves_total_capacity_and_l0_budget() -> None:
    model = make_model()
    assert model.cfg.d_high == 4
    assert model.cfg.d_low == 16
    assert model.cfg.k_high == 1
    assert model.cfg.k_low == 4
    assert model.cfg.d_high + model.cfg.d_low == model.cfg.d_sae
    assert model.cfg.k_high + model.cfg.k_low == model.cfg.k
    codes = model.encode(torch.randn(3, 6, 8))
    high, low = model.split_code(codes)
    assert high.shape == (3, 6, 4)
    assert low.shape == (3, 6, 16)
    assert torch.all((high > 0).sum(dim=-1) <= 1)
    assert torch.all((low > 0).sum(dim=-1) <= 4)


def test_forward_forecasts_only_high_and_reconstructs_with_both_groups() -> None:
    model = make_model()
    outputs = model(torch.randn(3, 6, 8))
    assert outputs["predicted_codes"].shape == (3, 5, 4)
    assert outputs["target_codes"].shape == (3, 5, 4)
    assert outputs["low_context_codes"].shape == (3, 5, 16)
    assert outputs["target_low_code"].shape == (3, 16)
    expected = outputs["online_high_reconstruction"] + model.decode_low(
        outputs["online_target_low_code"],
        ema=False,
        add_bias=False,
    )
    assert torch.allclose(outputs["online_target_reconstruction"], expected)


def test_hierarchical_loss_updates_online_groups_but_not_ema_teacher() -> None:
    model = make_model()
    loss, metrics = hierarchical_transition_jepa_loss(
        model,
        torch.randn(4, 6, 8),
        prediction_weight=1.0,
        residual_prediction_weight=0.1,
        use_context=True,
    )
    loss.backward()
    assert model.encoder.linear.weight.grad is not None
    assert model.decoder.grad is not None
    assert model.transition_predictor.output.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in model.ema_encoder.parameters()
    )
    assert model.ema_decoder.grad is None
    assert "online_high_reconstruction_fvu" in metrics
    assert metrics["high_l0"] <= model.cfg.k_high
    assert metrics["low_l0"] <= model.cfg.k_low


def test_high_only_forecast_decoder_cannot_use_low_rows() -> None:
    model = make_model()
    code = torch.randn(2, 3, model.cfg.d_high)
    before = model.decode_forecast_ema(code, add_bias=False)
    with torch.no_grad():
        model.ema_decoder[model.cfg.d_high :].add_(1000)
    after = model.decode_forecast_ema(code, add_bias=False)
    assert torch.allclose(before, after)


def test_hierarchical_model_loads_the_common_standard_sae_exactly() -> None:
    model = make_model()
    standard = StandardSparseAutoencoder(
        StandardSAEConfig(
            d_in=8,
            d_sae=20,
            k=5,
            window_size=6,
        )
    )
    checkpoint = {
        "config": {
            "d_in": 8,
            "d_sae": 20,
            "k": 5,
            "window_size": 6,
        },
        "state_dict": standard.state_dict(),
    }
    model.load_standard_sae(checkpoint)
    assert torch.equal(
        model.encoder.linear.weight,
        standard.encoder.linear.weight,
    )
    assert torch.equal(model.decoder, standard.decoder)
    assert torch.equal(
        model.ema_encoder.linear.weight,
        standard.encoder.linear.weight,
    )
