import numpy as np
import torch

from shared_residual.transition_jepa_eval import (
    HORIZON_METRICS,
    build_horizon_curve,
    causal_lm_loss,
)


def test_horizon_curve_reports_both_forecast_null_gains() -> None:
    rows = 12
    contexts = 3
    statistics = {
        name: torch.ones(rows, contexts) for name in HORIZON_METRICS
    }
    statistics["code_cosine"].fill_(0.8)
    statistics["shuffled_context_cosine"].fill_(0.2)
    statistics["position_only_cosine"].fill_(0.3)
    statistics["context_target_cosine"].fill_(0.4)
    statistics["residual_error"].fill_(0.25)
    statistics["residual_energy"].fill_(1.0)
    curve = build_horizon_curve(
        statistics,
        np.asarray([f"question-{index}" for index in range(rows)]),
        seed=0,
    )
    assert [row["horizon"] for row in curve] == [3, 2, 1]
    assert abs(curve[0]["context_gain_over_shuffled"]["mean"] - 0.6) < 1e-6
    assert abs(
        curve[0]["context_gain_over_position_only"]["mean"] - 0.5
    ) < 1e-6
    assert curve[0]["residual_prediction_fvu"] == 0.25


def test_causal_lm_loss_ignores_padded_targets() -> None:
    logits = torch.zeros(1, 4, 5)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    loss = causal_lm_loss(logits, input_ids, attention_mask)
    assert torch.isfinite(loss)
    assert abs(float(loss) - float(torch.log(torch.tensor(5.0)))) < 1e-6
