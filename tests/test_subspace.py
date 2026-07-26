import torch

from shared_residual.subspace import (
    evaluate_basis,
    fit_centering,
    fit_generalized_subspace,
    principal_angle_similarity,
)


def test_recovers_shared_subspace() -> None:
    torch.manual_seed(0)
    n, t, d, k = 800, 10, 48, 3
    true_basis, _ = torch.linalg.qr(torch.randn(d, k), mode="reduced")
    position = torch.randn(t, d) * 0.3
    position -= position.mean(dim=0)
    shared = torch.randn(n, k) @ true_basis.T * 2.0
    noise = torch.randn(n, t, d) * 0.7
    x = shared[:, None, :] + position[None, :, :] + noise
    centering = fit_centering(x)
    y = centering.transform(x)
    estimated, _, _ = fit_generalized_subspace(y, rank=k, ridge=1e-3)
    similarity = principal_angle_similarity(true_basis, estimated)
    assert similarity["mean_squared_cosine"] > 0.9
    assert evaluate_basis(y, estimated)["mean_icc"] > 0.7


def test_position_effect_is_removed() -> None:
    torch.manual_seed(1)
    x = torch.randn(100, 10, 20) + torch.randn(10, 20)[None, :, :]
    y = fit_centering(x).transform(x)
    assert torch.allclose(y.mean(dim=0), torch.zeros(10, 20), atol=1e-5)
