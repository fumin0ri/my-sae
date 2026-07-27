from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from .io import torch_load


def load_activation_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "shared-residual-activation-shards-v1":
        raise ValueError(f"unsupported activation manifest: {manifest_path}")
    if manifest["window_size"] != 10:
        raise ValueError("transition JEPA requires ten-token activation windows")
    for split in ("train", "validation"):
        if not manifest[split]["shards"]:
            raise ValueError(f"activation manifest has no {split} shards")
    return manifest_path.parent, manifest


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    identity = {
        "format": manifest["format"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "resolved_model_revision": manifest.get("resolved_model_revision"),
        "layer": manifest["layer"],
        "layer_path": manifest["layer_path"],
        "hook_point": manifest["hook_point"],
        "window_size": manifest["window_size"],
        "sequence_length": manifest.get("sequence_length"),
        "d_in": manifest["d_in"],
        "normalization": manifest["normalization"],
        "train": {
            "windows": manifest["train"]["windows"],
            "shards": manifest["train"]["shards"],
            "domain_counts": manifest["train"]["domain_counts"],
        },
        "validation": {
            "windows": manifest["validation"]["windows"],
            "shards": manifest["validation"]["shards"],
            "domain_counts": manifest["validation"]["domain_counts"],
        },
        "seed": manifest["seed"],
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shard_paths(
    root: Path,
    manifest: dict[str, Any],
    split: str,
) -> list[Path]:
    return [root / relative for relative in manifest[split]["shards"]]


def load_shard(path: Path) -> torch.Tensor:
    value = torch_load(path)
    activations = value["activations"] if isinstance(value, dict) else value
    if activations.ndim != 3 or activations.shape[1] != 10:
        raise ValueError(f"invalid activation shard shape at {path}")
    return activations


class ShuffledShardBatches:
    """Infinite deterministic batches without loading the full corpus in RAM."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        split: str,
        batch_size: int,
        seed: int,
    ):
        self.paths = shard_paths(root, manifest, split)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.path_order: list[int] = []
        self.path_position = 0
        self.current: torch.Tensor | None = None
        self.row_order: torch.Tensor | None = None
        self.row_position = 0
        self.epoch = 0

    def __iter__(self) -> ShuffledShardBatches:
        return self

    def _next_shard(self) -> None:
        if self.path_position >= len(self.path_order):
            self.path_order = torch.randperm(
                len(self.paths),
                generator=self.generator,
            ).tolist()
            self.path_position = 0
            self.epoch += 1
        path = self.paths[self.path_order[self.path_position]]
        self.path_position += 1
        self.current = load_shard(path)
        self.row_order = torch.randperm(
            len(self.current),
            generator=self.generator,
        )
        self.row_position = 0

    def __next__(self) -> torch.Tensor:
        pieces = []
        needed = self.batch_size
        while needed:
            if (
                self.current is None
                or self.row_order is None
                or self.row_position >= len(self.current)
            ):
                self._next_shard()
            assert self.current is not None
            assert self.row_order is not None
            take = min(needed, len(self.current) - self.row_position)
            indices = self.row_order[
                self.row_position : self.row_position + take
            ]
            pieces.append(self.current.index_select(0, indices))
            self.row_position += take
            needed -= take
        return torch.cat(pieces) if len(pieces) > 1 else pieces[0]


def validation_batches(
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
) -> Iterator[torch.Tensor]:
    emitted = 0
    for path in shard_paths(root, manifest, "validation"):
        shard = load_shard(path)
        for start in range(0, len(shard), batch_size):
            yield shard[start : start + batch_size]
            emitted += 1
            if maximum_batches > 0 and emitted >= maximum_batches:
                return
