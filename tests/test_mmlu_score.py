import pytest
import torch

from shared_residual.mmlu_score import accuracy_summary, eligible_indices


def test_accuracy_summary_reports_context_syntax_and_subject() -> None:
    metadata = [
        {
            "context_category": "STEM",
            "syntax_template": "template_0",
            "subject": "physics",
        },
        {
            "context_category": "STEM",
            "syntax_template": "template_1",
            "subject": "physics",
        },
        {
            "context_category": "humanities",
            "syntax_template": "template_0",
            "subject": "history",
        },
        {
            "context_category": "humanities",
            "syntax_template": "template_1",
            "subject": "history",
        },
    ]
    report = accuracy_summary(
        ["A", "B", "C", "D"],
        ["A", "C", "C", "A"],
        metadata,
    )
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["by_context"]["STEM"]["accuracy"] == pytest.approx(0.5)
    assert report["by_context"]["humanities"]["accuracy"] == pytest.approx(0.5)
    assert report["by_syntax"]["template_0"]["accuracy"] == pytest.approx(1.0)
    assert report["by_syntax"]["template_1"]["accuracy"] == pytest.approx(0.0)


def test_minimum_token_filter_matches_residual_window_eligibility() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    assert eligible_indices(attention_mask, minimum_tokens=4) == [0, 2]
