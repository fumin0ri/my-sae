import torch

from shared_residual.persistent_eval import different_group_permutation
from shared_residual.persistent_sae import (
    PersistentSAEConfig,
    PersistentSparseAutoencoder,
    group_contrastive_loss,
    persistent_loss,
)


def test_persistent_sae_is_predictor_free_and_has_individual_targets() -> None:
    cfg = PersistentSAEConfig(
        d_in=16,
        d_sae=32,
        k=4,
        window_size=10,
    )
    model = PersistentSparseAutoencoder(cfg)
    x = torch.randn(6, 10, 16)
    model.initialize_from_data(x)
    outputs = model(x)
    assert outputs["reconstruction"].shape == x.shape
    assert outputs["context_code"].shape == (6, 32)
    assert outputs["target_codes"].shape == (6, 9, 32)
    assert not hasattr(model, "predictor")
    assert not hasattr(model, "context_transformer")


def test_persistent_loss_reports_all_nine_offsets_and_stops_target_gradient() -> None:
    model = PersistentSparseAutoencoder(
        PersistentSAEConfig(d_in=12, d_sae=24, k=3, window_size=10)
    )
    x = torch.randn(8, 10, 12)
    model.initialize_from_data(x)
    group_ids = torch.arange(8)
    loss, metrics = persistent_loss(
        model,
        x,
        group_ids,
        persistence_weight=1.0,
        contrastive_weight=0.1,
        temperature=0.2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(f"offset_{offset}_cosine" in metrics for offset in range(1, 10))
    assert all(f"offset_{offset}_survival" in metrics for offset in range(1, 10))
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )
    assert model.encoder.linear.weight.grad is not None


def test_group_contrastive_prefers_correct_individual_pairs() -> None:
    context = torch.eye(4)
    targets = context[:, None, :].repeat(1, 9, 1)
    groups = torch.arange(4)
    correct = group_contrastive_loss(context, targets, groups, 0.1)
    shuffled = group_contrastive_loss(
        context,
        targets.roll(1, dims=0),
        groups,
        0.1,
    )
    assert correct < shuffled


def test_different_group_null_never_keeps_the_same_group() -> None:
    groups = torch.tensor([0, 0, 1, 1, 2, 2]).numpy()
    permutation = different_group_permutation(groups, seed=3)
    assert (groups[permutation] != groups).all()
