import numpy as np
import torch

from shared_residual.transition_jepa_eval import (
    OFFSET_STATISTIC_NAMES,
    batch_offset_statistics,
    collect_model_outputs,
)
from shared_residual.transition_jepa_sae import (
    TransitionJEPAConfig,
    TransitionJEPASAE,
)


def test_batch_offset_statistics_reduce_dense_codes_to_scalars() -> None:
    batch_size = 2
    n_offsets = 5
    d_sae = 16
    target = torch.arange(
        1,
        batch_size * n_offsets * d_sae + 1,
        dtype=torch.float32,
    ).reshape(batch_size, n_offsets, d_sae)
    target_residual = torch.ones(batch_size, n_offsets, 3)
    outputs = {
        "predicted_codes": target.clone(),
        "sparse_predicted_codes": target.clone(),
        "target_codes": target,
        "predictable_residual": target_residual.clone(),
        "target_residual": target_residual,
    }
    statistics = batch_offset_statistics(
        outputs,
        shuffled_prediction=-target,
        pre_bias=torch.zeros(3),
    )
    assert set(statistics) == set(OFFSET_STATISTIC_NAMES)
    assert all(
        value.shape == (batch_size, n_offsets)
        for value in statistics.values()
    )
    assert all(
        value.numel() == batch_size * n_offsets
        for value in statistics.values()
    )
    assert torch.allclose(
        statistics["code_cosine"],
        torch.ones(batch_size, n_offsets),
    )
    assert torch.allclose(
        statistics["shuffled_context_cosine"],
        -torch.ones(batch_size, n_offsets),
    )
    assert torch.count_nonzero(statistics["code_nrmse"]) == 0
    assert torch.all(statistics["support_precision"] == 1)
    assert torch.all(statistics["support_recall"] == 1)
    assert torch.all(statistics["support_jaccard"] == 1)
    assert torch.count_nonzero(statistics["residual_error"]) == 0
    assert torch.all(statistics["residual_energy"] == 1)


def test_window_32_collection_retains_only_final_dense_test_codes() -> None:
    model = TransitionJEPASAE(
        TransitionJEPAConfig(
            d_in=4,
            d_sae=8,
            k=2,
            window_size=32,
            predictor_width=4,
        )
    )
    x = torch.randn(6, 32, 4).to(torch.bfloat16)
    collected = collect_model_outputs(
        model,
        x,
        test_indices=[0, 2, 4],
        groups=np.asarray(["a", "b", "c"]),
        batch_size=2,
        device=torch.device("cpu"),
        amp_dtype="none",
        use_context=True,
        seed=3,
        label="test",
        retain_final_test_codes=True,
    )
    assert collected["context"].shape == (6, 8)
    assert collected["final_prediction"].shape == (6, 8)
    assert collected["window_code_cosine"].shape == (3,)
    assert collected["final_test_prediction"].shape == (3, 8)
    assert collected["final_test_target"].shape == (3, 8)
    assert all(
        value.shape == (3, 31)
        for value in collected["offset_statistics"].values()
    )
    assert "test_prediction" not in collected
    assert "test_target" not in collected
    assert "test_shuffled_prediction" not in collected
