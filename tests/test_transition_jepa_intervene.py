import pytest
import torch

from shared_residual.transition_jepa_intervene import (
    build_parser,
    resolve_horizon,
    select_eligible_pairs,
)


def test_default_intervention_horizon_uses_position_zero() -> None:
    assert resolve_horizon(None, 4) == 3
    assert resolve_horizon(None, 16) == 15


def test_explicit_intervention_horizon_is_validated() -> None:
    assert resolve_horizon(4, 5) == 4
    with pytest.raises(ValueError):
        resolve_horizon(5, 5)


def test_intervention_defaults_to_training_matched_online_context() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "model",
            "--pairs",
            "pairs.jsonl",
            "--checkpoint",
            "sae.pt",
            "--output",
            "result.jsonl",
            "--layer",
            "1",
        ]
    )
    assert args.context_encoder == "online"


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
