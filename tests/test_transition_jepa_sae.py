from dataclasses import asdict

import torch

from shared_residual.standard_sae import (
    StandardSAEConfig,
    StandardSparseAutoencoder,
)
from shared_residual.transition_jepa_sae import (
    TransitionJEPAConfig,
    TransitionJEPASAE,
    support_metrics,
    transition_jepa_loss,
)


def make_model(window_size: int = 6) -> tuple[TransitionJEPASAE, torch.Tensor]:
    x = torch.randn(8, window_size, 12)
    standard_cfg = StandardSAEConfig(
        d_in=12,
        d_sae=24,
        k=3,
        window_size=window_size,
    )
    standard = StandardSparseAutoencoder(standard_cfg)
    standard.initialize_from_data(x)
    checkpoint = {
        "config": asdict(standard_cfg),
        "state_dict": standard.state_dict(),
    }
    model = TransitionJEPASAE(
        TransitionJEPAConfig(
            d_in=12,
            d_sae=24,
            k=3,
            window_size=window_size,
            predictor_width=8,
        )
    )
    model.load_standard_sae(checkpoint)
    return model, x


def test_every_context_forecasts_the_same_fixed_endpoint() -> None:
    model, x = make_model()
    outputs = model(x)
    assert outputs["online_target_reconstruction"].shape == (8, 12)
    assert outputs["target_reconstruction"].shape == (8, 12)
    assert outputs["context_code"].shape == (8, 24)
    assert outputs["context_codes"].shape == (8, 5, 24)
    assert outputs["online_target_code"].shape == (8, 24)
    assert outputs["target_code"].shape == (8, 24)
    assert outputs["target_codes"].shape == (8, 5, 24)
    assert outputs["predicted_codes"].shape == (8, 5, 24)
    assert torch.allclose(
        outputs["target_codes"],
        outputs["target_code"][:, None, :].expand(-1, 5, -1),
    )
    assert torch.all(outputs["predicted_codes"] >= 0)
    assert torch.all(
        (outputs["sparse_predicted_codes"] > 0).sum(dim=-1) <= 3
    )


def test_k_only_prediction_is_independent_of_context_code() -> None:
    model, _ = make_model()
    left = torch.randn(5, 24)
    right = torch.randn(5, 24)
    left_prediction = model.predict_from_code(left, use_context=False)
    right_prediction = model.predict_from_code(right, use_context=False)
    assert torch.allclose(left_prediction, right_prediction)


def test_default_position_embeddings_are_zero_through_endpoint_minus_one() -> None:
    model, _ = make_model()
    contexts = torch.randn(3, model.cfg.window_size - 1, 24)
    default = model.predict_from_code(contexts)
    explicit = model.predict_from_code(
        contexts,
        context_positions=torch.arange(model.cfg.window_size - 1),
    )
    assert torch.allclose(default, explicit)


def test_joint_loss_backpropagates_online_but_not_ema_encoder() -> None:
    model, x = make_model()
    loss, metrics = transition_jepa_loss(
        model,
        x,
        prediction_weight=1.0,
        residual_prediction_weight=0.1,
        use_context=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        f"context_{context}_horizon_{model.cfg.window_size - 1 - context}"
        "_cosine" in metrics
        for context in range(model.cfg.window_size - 1)
    )
    assert model.encoder.linear.weight.grad is not None
    assert model.decoder.grad is not None
    assert model.transition_predictor.output.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in model.ema_encoder.parameters()
    )
    assert model.ema_decoder.grad is None


def test_ema_decoder_backpropagates_to_code_but_not_teacher() -> None:
    model, _ = make_model()
    model.zero_grad(set_to_none=True)
    target = torch.randn(4, 24, requires_grad=True)
    reconstruction = model.decode_ema(target)
    reconstruction.square().mean().backward()
    assert target.grad is not None
    assert model.ema_decoder.grad is None


def test_ema_update_tracks_full_sae_and_normalizes_decoder_rows() -> None:
    model, _ = make_model()
    before_bias = model.ema_pre_bias.clone()
    before_encoder = model.ema_encoder.linear.weight.clone()
    before_decoder = model.ema_decoder.clone()
    with torch.no_grad():
        model.pre_bias.add_(2.0)
        model.encoder.linear.weight.add_(1.0)
        model.decoder.add_(0.5)
    model.update_ema_sae(decay=0.5)
    assert torch.allclose(model.ema_pre_bias, before_bias + 1.0)
    assert not torch.allclose(model.ema_encoder.linear.weight, before_encoder)
    assert not torch.allclose(model.ema_decoder, before_decoder)
    assert torch.allclose(
        model.ema_decoder.norm(dim=1),
        torch.ones(model.cfg.d_sae),
        atol=1e-6,
    )


def test_evaluation_can_use_final_ema_context_encoder() -> None:
    model, x = make_model()
    with torch.no_grad():
        model.ema_encoder.linear.bias.add_(1.0)
    online = model(x, use_ema_context=False)
    ema = model(x, use_ema_context=True)
    assert not torch.allclose(online["context_codes"], ema["context_codes"])


def test_final_ema_sae_exports_as_standard_sae() -> None:
    model, x = make_model()
    exported = StandardSparseAutoencoder(
        StandardSAEConfig(
            d_in=model.cfg.d_in,
            d_sae=model.cfg.d_sae,
            k=model.cfg.k,
            window_size=model.cfg.window_size,
        )
    )
    exported.load_state_dict(model.final_ema_sae_state_dict())
    expected_code = model.encode_ema(x[:, -1])
    actual_code = exported.encode(x[:, -1])
    assert torch.allclose(actual_code, expected_code)
    assert torch.allclose(
        exported.decode(actual_code),
        model.decode_ema(expected_code),
    )


def test_support_metrics_recover_exact_topk_target() -> None:
    prediction = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]]
    )
    target = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]
    )
    precision, recall, jaccard = support_metrics(prediction, target, k=2)
    assert torch.allclose(precision, torch.ones_like(precision))
    assert torch.allclose(recall, torch.ones_like(recall))
    assert torch.allclose(jaccard, torch.ones_like(jaccard))
