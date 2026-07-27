from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Iterable

import torch


def configure_accelerator(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def cosine_learning_rate(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    multiplier = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return base_lr * multiplier


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.optim.AdamW, bool]:
    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        list(parameters),
        lr=lr,
        weight_decay=weight_decay,
        fused=fused,
    )
    return optimizer, fused


def grouped_three_way_split(
    metadata: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
    group_key: str,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        group = str(row.get(group_key, f"row-{index}"))
        groups.setdefault(group, []).append(index)
    if len(groups) < 10:
        raise ValueError("need at least ten independent groups for train/validation/test")
    ordered = sorted(groups)
    permutation = torch.randperm(
        len(ordered),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    n_test = max(1, round(len(ordered) * test_fraction))
    n_validation = max(1, round(len(ordered) * validation_fraction))
    if n_test + n_validation > len(ordered) - 4:
        raise ValueError("validation/test fractions leave too few training groups")
    test_groups = {ordered[index] for index in permutation[:n_test]}
    validation_groups = {
        ordered[index]
        for index in permutation[n_test : n_test + n_validation]
    }
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for group, indices in groups.items():
        if group in test_groups:
            test_indices.extend(indices)
        elif group in validation_groups:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return train_indices, validation_indices, test_indices
