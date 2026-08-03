from pathlib import Path

import pytest
import torch

import shared_residual.rectified_lpjepa_eval as eval_module
from shared_residual.rectified_lpjepa_eval import (
    causal_lm_loss,
    evaluate_sae_quality,
    evaluate_view_invariance,
)
from shared_residual.rectified_lpjepa_sae import RectifiedLpJEPAConfig, RectifiedLpJEPASAE


def make_model() -> RectifiedLpJEPASAE:
    model = RectifiedLpJEPASAE(
        RectifiedLpJEPAConfig(
            d_in=8,
            d_sae=20,
            low_k=4,
            max_span_length=4,
            high_fraction=0.2,
        )
    )
    model.initialize_from_statistics(torch.zeros(8), 1.0)
    return model


def test_standard_sae_quality_compares_online_and_ema_on_same_rows(monkeypatch) -> None:
    model = make_model()
    batch = torch.randn(12, 8)
    monkeypatch.setattr(eval_module, "validation_batches", lambda *_args, **_kwargs: iter([batch]))
    result = evaluate_sae_quality(
        model, Path("unused"), {}, 3, 1, torch.device("cpu"), "none"
    )
    assert set(result) == {"online", "ema", "online_ema_alignment"}
    assert result["online"]["n_positions"] == 12
    assert "high_active_fraction" in result["ema"]


def test_view_invariance_reports_shuffle_null_and_distance_curve(monkeypatch) -> None:
    model = make_model()
    batch = {
        "view_a": torch.randn(6, 8),
        "view_b": torch.randn(6, 8),
        "distance": torch.tensor([1, 1, 2, 2, 3, 3]),
    }
    monkeypatch.setattr(
        eval_module,
        "validation_view_pair_batches",
        lambda *_args, **_kwargs: iter([batch]),
    )
    result = evaluate_view_invariance(
        model, Path("unused"), {}, 6, 1, torch.device("cpu"), "none", 0
    )
    assert [row["distance"] for row in result["distance_curve"]] == [1, 2, 3]
    assert "overall_ema_high_margin" in result
    assert "ema_swap_reconstruction_fvu" in result["distance_curve"][0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_view_invariance_swap_decode_accepts_bfloat16_codes(monkeypatch) -> None:
    model = make_model().cuda()
    batch = {
        "view_a": torch.randn(4, 8),
        "view_b": torch.randn(4, 8),
        "distance": torch.tensor([1, 1, 2, 2]),
    }
    monkeypatch.setattr(
        eval_module,
        "validation_view_pair_batches",
        lambda *_args, **_kwargs: iter([batch]),
    )
    result = evaluate_view_invariance(
        model,
        Path("unused"),
        {},
        4,
        1,
        torch.device("cuda"),
        "bfloat16",
        0,
    )
    assert torch.isfinite(
        torch.tensor(result["distance_curve"][0]["ema_swap_reconstruction_fvu"])
    )


def test_causal_lm_loss_ignores_padded_targets() -> None:
    logits = torch.zeros(1, 4, 5)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    loss = causal_lm_loss(logits, input_ids, attention_mask)
    assert torch.isfinite(loss)
    assert abs(float(loss) - float(torch.log(torch.tensor(5.0)))) < 1e-6
