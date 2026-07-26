import numpy as np
import torch

from shared_residual.research import (
    group_labels,
    indices_for_groups,
    stratified_group_split,
)


def make_metadata() -> list[dict[str, str]]:
    rows = []
    for group_index in range(40):
        label = f"state-{group_index % 4}"
        for paraphrase in range(3):
            rows.append(
                {
                    "group_id": f"group-{group_index}",
                    "state": label,
                    "id": f"{group_index}-{paraphrase}",
                }
            )
    return rows


def test_group_split_has_no_paraphrase_leakage() -> None:
    metadata = make_metadata()
    groups, labels = group_labels(metadata, "group_id", "state")
    development, test = stratified_group_split(groups, labels, 0.2, 7)
    assert set(development).isdisjoint(set(test))
    development_idx = indices_for_groups(metadata, "group_id", development)
    test_idx = indices_for_groups(metadata, "group_id", test)
    assert set(development_idx.tolist()).isdisjoint(set(test_idx.tolist()))
    assert len(development_idx) + len(test_idx) == len(metadata)


def test_group_split_preserves_all_classes() -> None:
    metadata = make_metadata()
    groups, labels = group_labels(metadata, "group_id", "state")
    development, test = stratified_group_split(groups, labels, 0.2, 11)
    label_by_group = dict(zip(groups.tolist(), labels.tolist()))
    assert {label_by_group[group] for group in development} == set(labels)
    assert {label_by_group[group] for group in test} == set(labels)


def test_indices_for_groups_accepts_numpy_values() -> None:
    metadata = make_metadata()
    indices = indices_for_groups(
        metadata,
        "group_id",
        np.asarray(["group-1", "group-3"]),
    )
    assert torch.equal(indices, torch.tensor([3, 4, 5, 9, 10, 11]))
