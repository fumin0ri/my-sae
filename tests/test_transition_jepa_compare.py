import copy

import pytest

from shared_residual.transition_jepa_compare import validate_comparison


def make_run(output: str, auxk_enabled: bool) -> dict:
    return {
        "evaluation": {
            "checkpoint": {
                "data_fingerprint": "shared-data",
                "config": {
                    "d_in": 8,
                    "d_sae": 20,
                    "k": 5,
                    "max_span_length": 4,
                    "predictor_width": 8,
                    "predictor_expansion": 2,
                    "predictor_output": output,
                    "ema_decay": 0.996,
                    "high_fraction": 0.2,
                    "high_reconstruction_weight": 0.2,
                },
            },
            "split": {"split_seed": 0, "group_key": "question_id"},
        },
        "training": {
            "architecture": {
                "predictor_auxk": {"enabled": auxk_enabled}
            },
            "horizon_balancing": {"mode": "inverse_probability"},
        },
    }


def test_comparison_requires_only_the_prespecified_factor_changes() -> None:
    runs = {
        "softplus": make_run("softplus", False),
        "relu_topk": make_run("relu_topk", False),
        "relu_topk_auxk": make_run("relu_topk", True),
    }
    validate_comparison(runs)
    mismatched = copy.deepcopy(runs)
    mismatched["relu_topk_auxk"]["evaluation"]["checkpoint"][
        "data_fingerprint"
    ] = "different-data"
    with pytest.raises(ValueError, match="activation fingerprint"):
        validate_comparison(mismatched)
