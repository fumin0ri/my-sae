from __future__ import annotations

import argparse

import torch


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(
        dict.fromkeys(
            int(part.strip()) for part in value.split(",") if part.strip()
        )
    )
    if not offsets:
        raise argparse.ArgumentTypeError("expected comma-separated offsets")
    return offsets


def parse_feature_ids(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(
        dict.fromkeys(int(part.strip()) for part in value.split(","))
    )


def restrict_features(
    codes: torch.Tensor,
    feature_ids: tuple[int, ...],
) -> torch.Tensor:
    if not feature_ids:
        return codes
    if min(feature_ids) < 0 or max(feature_ids) >= codes.shape[-1]:
        raise ValueError("a requested feature id is outside the SAE dictionary")
    mask = torch.zeros(
        codes.shape[-1],
        device=codes.device,
        dtype=codes.dtype,
    )
    mask[list(feature_ids)] = 1
    return codes * mask
