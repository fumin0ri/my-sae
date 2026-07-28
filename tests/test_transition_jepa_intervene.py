import pytest

from shared_residual.transition_jepa_intervene import resolve_offsets


def test_default_intervention_offsets_follow_checkpoint_window() -> None:
    assert resolve_offsets(None, 4) == (1, 2, 3)
    assert resolve_offsets(None, 16) == tuple(range(1, 16))


def test_explicit_intervention_offsets_are_validated() -> None:
    assert resolve_offsets((1, 4), 5) == (1, 4)
    with pytest.raises(ValueError):
        resolve_offsets((1, 5), 5)
