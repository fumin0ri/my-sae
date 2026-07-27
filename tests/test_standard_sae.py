import math

import torch

from shared_residual.evaluation import different_group_permutation
from shared_residual.standard_sae import (
    StandardSAEConfig,
    StandardSparseAutoencoder,
    standard_sae_loss,
)
from shared_residual.training import (
    cosine_learning_rate,
    grouped_three_way_split,
)


def test_standard_sae_reconstructs_every_window_position() -> None:
    model = StandardSparseAutoencoder(
        StandardSAEConfig(
            d_in=12,
            d_sae=24,
            k=3,
            window_size=10,
        )
    )
    x = torch.randn(8, 10, 12)
    model.initialize_from_data(x)
    outputs = model(x)
    assert outputs["reconstruction"].shape == x.shape
    assert outputs["codes"].shape == (8, 10, 24)
    assert torch.all((outputs["codes"] > 0).sum(dim=-1) <= 3)
    loss, metrics = standard_sae_loss(model, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["sae_l0"] <= 3
    assert model.encoder.linear.weight.grad is not None


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


def test_different_group_null_changes_every_group() -> None:
    groups = torch.tensor([0, 0, 1, 1, 2, 2]).numpy()
    permutation = different_group_permutation(groups, seed=3)
    assert (groups[permutation] != groups).all()
