import torch

from shared_residual.group_sae import GroupSAEConfig, SharedPrivateSAE, topk_relu


def test_topk_relu() -> None:
    x = torch.tensor([[-1.0, 3.0, 2.0, 1.0]])
    y = topk_relu(x, 2)
    assert torch.equal(y, torch.tensor([[0.0, 3.0, 2.0, 0.0]]))


def test_group_sae_shapes() -> None:
    model = SharedPrivateSAE(GroupSAEConfig(16, 32, 48, 3, 4))
    reconstruction, shared, private, private_write = model(torch.randn(5, 10, 16))
    assert reconstruction.shape == (5, 10, 16)
    assert shared.shape == (5, 32)
    assert private.shape == (5, 10, 48)
    assert private_write.shape == (5, 10, 16)
