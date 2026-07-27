import numpy as np

from shared_residual.evaluation import cluster_bootstrap_means


def test_vectorized_cluster_bootstrap_resamples_whole_groups() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    groups = np.asarray(["left", "left", "right", "right"])
    first = cluster_bootstrap_means(values, groups, seed=3, samples=200)
    second = cluster_bootstrap_means(values, groups, seed=3, samples=200)
    assert np.array_equal(first, second)
    assert set(np.unique(first)).issubset({0.0, 0.5, 1.0})
    assert abs(first.mean() - 0.5) < 0.1
