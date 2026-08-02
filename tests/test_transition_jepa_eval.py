import numpy as np
import torch
from pathlib import Path

import shared_residual.transition_jepa_eval as eval_module
from shared_residual.transition_jepa_eval import (
    HORIZON_METRICS,
    build_horizon_curve,
    causal_lm_loss,
    evaluate_sae_quality,
)
from shared_residual.transition_jepa_sae import TransitionJEPAConfig, TransitionJEPASAE


def test_horizon_curve_reports_both_forecast_null_gains() -> None:
    rows = 12
    contexts = 3
    statistics = {
        name: torch.ones(rows, contexts) for name in HORIZON_METRICS
    }
    statistics["online_code_cosine"].fill_(0.8)
    statistics["online_shuffled_context_cosine"].fill_(0.2)
    statistics["ema_code_cosine"].fill_(0.6)
    statistics["ema_shuffled_context_cosine"].fill_(0.25)
    statistics["horizon_only_cosine"].fill_(0.3)
    statistics["online_context_target_cosine"].fill_(0.4)
    statistics["ema_context_target_cosine"].fill_(0.35)
    statistics["online_residual_error"].fill_(0.25)
    statistics["ema_residual_error"].fill_(0.4)
    statistics["residual_energy"].fill_(1.0)
    curve = build_horizon_curve(
        statistics,
        np.asarray([f"question-{index}" for index in range(rows)]),
        seed=0,
    )
    assert [row["horizon"] for row in curve] == [3, 2, 1]
    assert abs(curve[0]["online_gain_over_shuffled"]["mean"] - 0.6) < 1e-6
    assert abs(
        curve[0]["online_gain_over_horizon_only"]["mean"] - 0.5
    ) < 1e-6
    assert abs(curve[0]["ema_gain_over_shuffled"]["mean"] - 0.35) < 1e-6
    assert abs(curve[0]["online_minus_ema_cosine"]["mean"] - 0.2) < 1e-6
    assert curve[0]["online_residual_prediction_fvu"] == 0.25
    assert abs(curve[0]["ema_residual_prediction_fvu"] - 0.4) < 1e-6


def test_standard_sae_quality_compares_online_and_ema_on_same_rows(
    monkeypatch,
) -> None:
    model = TransitionJEPASAE(
        TransitionJEPAConfig(
            d_in=8,
            d_sae=20,
            k=5,
            max_span_length=4,
            predictor_width=8,
            high_fraction=0.2,
        )
    )
    model.initialize_from_statistics(torch.zeros(8), 1.0)
    batch = torch.randn(12, 8)
    monkeypatch.setattr(
        eval_module,
        "validation_batches",
        lambda *_args, **_kwargs: iter([batch]),
    )
    result = evaluate_sae_quality(
        model,
        Path("unused"),
        {},
        batch_size=3,
        maximum_batches=1,
        device=torch.device("cpu"),
        amp_dtype="float32",
    )
    assert set(result) == {"online", "ema", "online_ema_alignment"}
    assert result["online"]["n_positions"] == 12
    assert result["ema"]["n_positions"] == 12
    assert "fraction_variance_explained" in result["online"]
    assert "fraction_variance_explained" in result["ema"]


def test_causal_lm_loss_ignores_padded_targets() -> None:
    logits = torch.zeros(1, 4, 5)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    loss = causal_lm_loss(logits, input_ids, attention_mask)
    assert torch.isfinite(loss)
    assert abs(float(loss) - float(torch.log(torch.tensor(5.0)))) < 1e-6
