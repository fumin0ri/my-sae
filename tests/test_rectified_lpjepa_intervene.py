import torch

from shared_residual.rectified_lpjepa_intervene import build_parser, select_eligible_pairs


def test_intervention_has_only_one_sae_variant() -> None:
    args = build_parser().parse_args(
        [
            "--model", "model", "--pairs", "pairs.jsonl", "--checkpoint", "sae.pt",
            "--output", "result.jsonl", "--layer", "1",
        ]
    )
    assert not hasattr(args, "sae_variant")


def test_causal_pair_selection_skips_empty_prefixes_before_limit() -> None:
    class LengthTokenizer:
        def __call__(self, text, **_kwargs):
            length = int(text)
            return {"input_ids": torch.arange(length, dtype=torch.long)[None, :]}

    rows = [
        {"source_text": "0", "target_text": "6"},
        {"source_text": "4", "target_text": "4"},
        {"source_text": "7", "target_text": "5"},
    ]
    selected, skipped, examined = select_eligible_pairs(
        rows, LengthTokenizer(), 1, "source_text", "target_text", 2
    )
    assert [row_index for row_index, *_ in selected] == [1, 2]
    assert skipped == 1
    assert examined == 3
