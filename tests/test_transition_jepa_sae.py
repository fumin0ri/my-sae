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
    assert outputs["target_compatibility_reconstruction"].shape == (8, 12)
    assert outputs["context_code"].shape == (8, 24)
    assert outputs["context_codes"].shape == (8, 5, 24)
    assert outputs["online_target_code"].shape == (8, 24)
    assert outputs["target_code"].shape == (8, 24)
    assert outputs["target_codes"].shape == (8, 5, 24)
    assert outputs["predicted_codes"].shape == (8, 5, 24)
    assert outputs["context_state"].shape == (8, 5, 8)
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
    left_prediction, _ = model.predict_from_code(left, use_context=False)
    right_prediction, _ = model.predict_from_code(right, use_context=False)
    assert torch.allclose(left_prediction, right_prediction)


def test_default_position_embeddings_are_zero_through_endpoint_minus_one() -> None:
    model, _ = make_model()
    contexts = torch.randn(3, model.cfg.window_size - 1, 24)
    default, _ = model.predict_from_code(contexts)
    explicit, _ = model.predict_from_code(
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
        target_compatibility_weight=0.25,
        variance_weight=0.01,
        variance_target=1.0,
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
    assert model.transition_predictor.output.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )


def test_ema_target_compatibility_updates_only_the_shared_decoder() -> None:
    model, x = make_model()
    model.zero_grad(set_to_none=True)
    target = model.encode_target(x[:, -1])
    reconstruction = model.decode_target_compatibility(target)
    reconstruction.square().mean().backward()
    assert model.decoder.grad is not None
    assert model.pre_bias.grad is None
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )


def test_ema_update_includes_target_normalization_bias() -> None:
    model, _ = make_model()
    before = model.target_pre_bias.clone()
    with torch.no_grad():
        model.pre_bias.add_(2.0)
    model.update_target_encoder(decay=0.5)
    assert torch.allclose(model.target_pre_bias, before + 1.0)


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
