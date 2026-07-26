import pytest

from shared_residual.modeling import major_minor, require_safe_torch_load


def test_major_minor_accepts_release_and_cuda_suffixes() -> None:
    assert major_minor("2.6.0") == (2, 6)
    assert major_minor("2.7.1+cu128") == (2, 7)
    assert major_minor("2.8.0.dev20260101") == (2, 8)


def test_safe_torch_load_rejects_versions_before_2_6() -> None:
    with pytest.raises(RuntimeError, match=r"PyTorch >= 2\.6"):
        require_safe_torch_load(False, "2.5.1+cu121")


def test_safe_torch_load_accepts_2_6_or_newer() -> None:
    require_safe_torch_load(False, "2.6.0+cu124")
    require_safe_torch_load(False, "3.0.0")


def test_safetensors_accepts_cuda_12_1_torch() -> None:
    require_safe_torch_load(True, "2.5.1+cu121")
