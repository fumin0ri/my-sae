import math

import torch

from shared_residual.predictive_sae import (
    PredictiveSAEConfig,
    PredictiveSparseAutoencoder,
    cosine_learning_rate,
    grouped_three_way_split,
    make_span_spec,
    predictive_loss,
)


def test_cosine_schedule_warms_up_and_decays() -> None:
    assert math.isclose(
        cosine_learning_rate(1, 100, 1e-3, 10, 0.1),
        1e-4,
    )
    assert math.isclose(
        cosine_learning_rate(10, 100, 1e-3, 10, 0.1),
        1e-3,
    )
    assert math.isclose(
        cosine_learning_rate(100, 100, 1e-3, 10, 0.1),
        1e-4,
    )


def test_causal_span_never_uses_right_context() -> None:
    span = make_span_spec(
        window_size=48,
        context_width=24,
        target_size=4,
        gap=4,
        context_mode="causal",
        target_start=44,
    )
    assert len(span.context_indices) == 24
    assert len(span.target_indices) == 4
    assert max(span.context_indices) == min(span.target_indices) - 5
    assert max(span.context_indices) < min(span.target_indices)
    assert not set(span.context_indices) & set(span.target_indices)


def test_retrospective_span_is_explicitly_bidirectional() -> None:
    span = make_span_spec(
        window_size=64,
        context_width=20,
        target_size=4,
        gap=3,
        context_mode="retrospective",
        target_start=28,
    )
    assert min(span.context_indices) < min(span.target_indices)
    assert max(span.context_indices) > max(span.target_indices)
    assert not set(span.context_indices) & set(span.target_indices)


def test_predictive_sae_shapes_and_loss() -> None:
    cfg = PredictiveSAEConfig(
        d_in=16,
        d_sae=32,
        k=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        context_width=8,
        max_window_size=20,
        max_target_size=3,
        max_gap=3,
    )
    model = PredictiveSparseAutoencoder(cfg)
    x = torch.randn(5, 20, 16)
    model.initialize_from_data(x)
    span = make_span_spec(20, 8, 3, 2, target_start=17)
    outputs = model(x, span)
    assert outputs["reconstruction"].shape == x.shape
    assert outputs["predicted_codes"].shape == (5, 3, 32)
    assert outputs["predictable"].shape == (5, 3, 16)
    assert outputs["innovation"].shape == (5, 3, 16)
    assert torch.all((outputs["predicted_codes"] > 0).sum(dim=-1) <= 4)
    loss, metrics = predictive_loss(model, x, span, 1.0, 0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["l0"] <= 4
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_three_way_split_keeps_groups_intact() -> None:
    metadata = [
        {"group_id": f"g{group}", "row": row}
        for group in range(20)
        for row in range(3)
    ]
    train, validation, test = grouped_three_way_split(
        metadata,
        validation_fraction=0.2,
        test_fraction=0.2,
        group_key="group_id",
        seed=7,
    )
    parts = [set(train), set(validation), set(test)]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]
    assert set.union(*parts) == set(range(len(metadata)))
    group_to_part = {}
    for part_id, indices in enumerate(parts):
        for index in indices:
            group = metadata[index]["group_id"]
            group_to_part.setdefault(group, part_id)
            assert group_to_part[group] == part_id
