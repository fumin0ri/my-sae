from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from .io import torch_load


ACTIVATION_FORMAT = "shared-residual-sequence-shards-v2"


def load_activation_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ACTIVATION_FORMAT:
        raise ValueError(
            f"unsupported activation manifest: {manifest_path}; re-extract "
            "long residual sequences with the current sr-extract-pile"
        )
    if int(manifest["max_span_length"]) < 2:
        raise ValueError("max_span_length must be at least two")
    if not 2 <= int(manifest["min_span_length"]) <= int(
        manifest["max_span_length"]
    ):
        raise ValueError("min_span_length must lie in [2, max_span_length]")
    if int(manifest["max_horizon"]) != int(manifest["max_span_length"]) - 1:
        raise ValueError("max_horizon must equal max_span_length - 1")
    if int(manifest["burn_in_tokens"]) < 0:
        raise ValueError("burn_in_tokens cannot be negative")
    required_length = int(manifest["burn_in_tokens"]) + int(
        manifest["max_span_length"]
    )
    if int(manifest["minimum_valid_length"]) < required_length:
        raise ValueError("minimum_valid_length cannot expose a sequence boundary")
    if int(manifest["sequence_length"]) < int(manifest["minimum_valid_length"]):
        raise ValueError("sequence_length is shorter than minimum_valid_length")
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
        "max_span_length": manifest["max_span_length"],
        "min_span_length": manifest["min_span_length"],
        "max_horizon": manifest["max_horizon"],
        "sequence_length": manifest["sequence_length"],
        "burn_in_tokens": manifest["burn_in_tokens"],
        "minimum_valid_length": manifest["minimum_valid_length"],
        "pair_sampling": manifest.get("pair_sampling"),
        "d_in": manifest["d_in"],
        "normalization": manifest["normalization"],
        "train": {
            "sequences": manifest["train"]["sequences"],
            "positions": manifest["train"]["positions"],
            "shards": manifest["train"]["shards"],
            "domain_counts": manifest["train"]["domain_counts"],
        },
        "validation": {
            "sequences": manifest["validation"]["sequences"],
            "positions": manifest["validation"]["positions"],
            "shards": manifest["validation"]["shards"],
            "domain_counts": manifest["validation"]["domain_counts"],
        },
        "seed": manifest["seed"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def shard_paths(
    root: Path,
    manifest: dict[str, Any],
    split: str,
) -> list[Path]:
    return [root / relative for relative in manifest[split]["shards"]]


@dataclass
class SequenceShard:
    activations: torch.Tensor
    valid_lengths: torch.Tensor


def load_sequence_shard(path: Path, sequence_length: int) -> SequenceShard:
    value = torch_load(path)
    if not isinstance(value, dict) or "activations" not in value:
        raise ValueError(f"invalid sequence shard at {path}")
    activations = value["activations"]
    valid_lengths = value.get("valid_lengths")
    if activations.ndim != 3 or activations.shape[1] != sequence_length:
        raise ValueError(
            f"invalid activation shard shape at {path}; expected "
            f"[n, {sequence_length}, d_in]"
        )
    if valid_lengths is None or valid_lengths.shape != (len(activations),):
        raise ValueError(f"invalid valid_lengths at {path}")
    if torch.any(valid_lengths < 1) or torch.any(valid_lengths > sequence_length):
        raise ValueError(f"valid_lengths out of range at {path}")
    return SequenceShard(activations, valid_lengths.long())


def _sample_pairs(
    sequences: torch.Tensor,
    valid_lengths: torch.Tensor,
    span_lengths: torch.Tensor,
    horizons: torch.Tensor,
    burn_in_tokens: int,
    max_horizon: int,
    generator: torch.Generator,
    row_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    sample_count = len(horizons)
    if len(valid_lengths) != sample_count or len(span_lengths) != sample_count:
        raise ValueError("one span length and horizon are required per sequence")
    if torch.any(horizons < 1) or torch.any(horizons >= span_lengths):
        raise ValueError("each horizon must lie in [1, span_length-1]")
    # The endpoint support is identical for every horizon. This prevents h from
    # becoming a proxy for proximity to the extracted-sequence boundary.
    minimum_endpoints = torch.full_like(horizons, burn_in_tokens + max_horizon)
    if torch.any(minimum_endpoints >= valid_lengths):
        raise ValueError("sequence is too short for sampled horizon and burn-in")
    spans = valid_lengths - minimum_endpoints
    uniforms = torch.rand(sample_count, generator=generator)
    endpoints = minimum_endpoints + torch.floor(uniforms * spans).long()
    contexts = endpoints - horizons
    span_starts = endpoints - span_lengths + 1
    rows = (
        torch.arange(sample_count)
        if row_indices is None
        else row_indices
    )
    if rows.shape != (sample_count,):
        raise ValueError("row_indices must have one entry per pair")
    return {
        "context": sequences[rows, contexts],
        "target": sequences[rows, endpoints],
        "span_length": span_lengths,
        "horizon": horizons,
        "span_start_index": span_starts,
        "context_index": contexts,
        "endpoint_index": endpoints,
    }


class RandomPairShardBatches:
    """Infinite deterministic residual pairs from random spans and contexts."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        split: str,
        batch_size: int,
        seed: int,
    ):
        self.paths = shard_paths(root, manifest, split)
        self.sequence_length = int(manifest["sequence_length"])
        self.min_span_length = int(manifest["min_span_length"])
        self.max_span_length = int(manifest["max_span_length"])
        self.max_horizon = int(manifest["max_horizon"])
        self.burn_in_tokens = int(manifest["burn_in_tokens"])
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.path_order: list[int] = []
        self.path_position = 0
        self.current: SequenceShard | None = None
        self.row_order: torch.Tensor | None = None
        self.row_position = 0
        self.epoch = 0

    def __iter__(self) -> RandomPairShardBatches:
        return self

    def _next_shard(self) -> None:
        if self.path_position >= len(self.path_order):
            self.path_order = torch.randperm(
                len(self.paths), generator=self.generator
            ).tolist()
            self.path_position = 0
            self.epoch += 1
        path = self.paths[self.path_order[self.path_position]]
        self.path_position += 1
        self.current = load_sequence_shard(path, self.sequence_length)
        self.row_order = torch.randperm(
            len(self.current.activations), generator=self.generator
        )
        self.row_position = 0

    def __next__(self) -> dict[str, torch.Tensor]:
        span_lengths = torch.randint(
            self.min_span_length,
            self.max_span_length + 1,
            (self.batch_size,),
            generator=self.generator,
        )
        horizons = 1 + torch.floor(
            torch.rand(self.batch_size, generator=self.generator)
            * (span_lengths - 1)
        ).long()
        pieces: dict[str, list[torch.Tensor]] = {}
        needed = self.batch_size
        pair_start = 0
        while needed:
            if (
                self.current is None
                or self.row_order is None
                or self.row_position >= len(self.current.activations)
            ):
                self._next_shard()
            assert self.current is not None
            assert self.row_order is not None
            take = min(needed, len(self.current.activations) - self.row_position)
            indices = self.row_order[self.row_position : self.row_position + take]
            selected_lengths = self.current.valid_lengths.index_select(0, indices)
            pair_end = pair_start + take
            sampled = _sample_pairs(
                self.current.activations,
                selected_lengths,
                span_lengths[pair_start:pair_end],
                horizons[pair_start:pair_end],
                self.burn_in_tokens,
                self.max_horizon,
                self.generator,
                row_indices=indices,
            )
            for key, value in sampled.items():
                pieces.setdefault(key, []).append(value)
            self.row_position += take
            pair_start = pair_end
            needed -= take
        return {
            key: torch.cat(values) if len(values) > 1 else values[0]
            for key, values in pieces.items()
        }


def validation_pair_batches(
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    seed: int,
) -> Iterator[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    max_horizon = int(manifest["max_horizon"])
    min_span = int(manifest["min_span_length"])
    max_span = int(manifest["max_span_length"])
    burn_in = int(manifest["burn_in_tokens"])
    emitted = 0
    for path in shard_paths(root, manifest, "validation"):
        shard = load_sequence_shard(path, int(manifest["sequence_length"]))
        for start in range(0, len(shard.activations), batch_size):
            sequences = shard.activations[start : start + batch_size]
            lengths = shard.valid_lengths[start : start + batch_size]
            span_lengths = torch.randint(
                min_span,
                max_span + 1,
                (len(sequences),),
                generator=generator,
            )
            horizons = 1 + torch.floor(
                torch.rand(len(sequences), generator=generator)
                * (span_lengths - 1)
            ).long()
            yield _sample_pairs(
                sequences,
                lengths,
                span_lengths,
                horizons,
                burn_in,
                max_horizon,
                generator,
            )
            emitted += 1
            if maximum_batches > 0 and emitted >= maximum_batches:
                return


def validation_batches(
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
) -> Iterator[torch.Tensor]:
    """Yield only valid held-out residual positions for conventional SAE metrics."""
    emitted = 0
    burn_in = int(manifest["burn_in_tokens"])
    pending: list[torch.Tensor] = []
    pending_count = 0
    for path in shard_paths(root, manifest, "validation"):
        shard = load_sequence_shard(path, int(manifest["sequence_length"]))
        for sequence, valid_length in zip(shard.activations, shard.valid_lengths):
            value = sequence[burn_in : int(valid_length)]
            pending.append(value)
            pending_count += len(value)
            while pending_count >= batch_size:
                joined = torch.cat(pending)
                yield joined[:batch_size]
                emitted += 1
                remainder = joined[batch_size:]
                pending = [remainder] if len(remainder) else []
                pending_count = len(remainder)
                if maximum_batches > 0 and emitted >= maximum_batches:
                    return
    if pending_count and (maximum_batches <= 0 or emitted < maximum_batches):
        yield torch.cat(pending)
