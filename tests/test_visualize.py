import pytest

from shared_residual.visualize_results import candidate_config_summary


def candidate(seed: int, icc: float, null: float) -> dict:
    probe = {
        "shared_subspace": {"accuracy": 0.8},
        "rank_matched_mean_pca": {"accuracy": 0.6},
        "rank_matched_last_token_pca": {"accuracy": 0.5},
    }
    return {
        "layer": 3,
        "window_size": 10,
        "rank": 4,
        "ridge": 0.001,
        "seed": seed,
        "validation": {
            "mean_icc": icc,
            "null_mean_icc": null,
            "probe": probe,
        },
    }


def test_candidate_summary_aggregates_seeds() -> None:
    summary = candidate_config_summary(
        [candidate(0, 0.4, 0.1), candidate(1, 0.6, 0.2)]
    )
    row = summary[(3, 10, 4, 0.001)]
    assert row["icc_mean"] == pytest.approx(0.5)
    assert row["null_mean"] == pytest.approx(0.15)
    assert row["null_gap"] == pytest.approx(0.35)
    assert row["shared_probe"] == pytest.approx(0.8)
