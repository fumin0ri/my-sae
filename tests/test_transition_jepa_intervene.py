import pytest
import torch

from shared_residual.transition_jepa_intervene import (
    resolve_offsets,
    select_eligible_pairs,
)


def test_default_intervention_offsets_follow_checkpoint_window() -> None:
    assert resolve_offsets(None, 4) == (1, 2, 3)
    assert resolve_offsets(None, 16) == tuple(range(1, 16))


def test_explicit_intervention_offsets_are_validated() -> None:
    assert resolve_offsets((1, 4), 5) == (1, 4)
    with pytest.raises(ValueError):
        resolve_offsets((1, 5), 5)


def test_causal_pair_selection_skips_short_prefixes_before_limit() -> None:
    class LengthTokenizer:
        def __call__(self, text, **_kwargs):
            length = int(text)
            return {
                "input_ids": torch.arange(length, dtype=torch.long)[None, :]
            }

    rows = [
        {"source_text": "3", "target_text": "6"},
        {"source_text": "4", "target_text": "4"},
        {"source_text": "7", "target_text": "5"},
        {"source_text": "8", "target_text": "8"},
    ]
    selected, skipped, examined = select_eligible_pairs(
        rows,
        LengthTokenizer(),
        window_size=4,
        source_key="source_text",
        target_key="target_text",
        maximum=2,
    )
    assert [row_index for row_index, *_ in selected] == [1, 2]
    assert skipped == 1
    assert examined == 3
