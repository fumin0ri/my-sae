from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
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


def _sample_view_pairs(
    sequences: torch.Tensor,
    valid_lengths: torch.Tensor,
    span_lengths: torch.Tensor,
    burn_in_tokens: int,
    max_horizon: int,
    generator: torch.Generator,
    row_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample two exchangeable positions from each random span.

    The two offsets are an ordered sample without replacement.  Consequently
    view A and view B have identical marginal position distributions; neither
    is a privileged endpoint or prediction target.
    """
    sample_count = len(span_lengths)
    if len(valid_lengths) != sample_count:
        raise ValueError("one span length is required per sequence")
    if torch.any(span_lengths < 2) or torch.any(span_lengths > max_horizon + 1):
        raise ValueError("span lengths must lie in [2, max_horizon+1]")
    minimum_ends = torch.full_like(span_lengths, burn_in_tokens + max_horizon)
    if torch.any(minimum_ends >= valid_lengths):
        raise ValueError("sequence is too short for sampled horizon and burn-in")
    spans = valid_lengths - minimum_ends
    uniforms = torch.rand(sample_count, generator=generator)
    span_ends = minimum_ends + torch.floor(uniforms * spans).long()
    span_starts = span_ends - span_lengths + 1
    offsets_a = torch.floor(
        torch.rand(sample_count, generator=generator) * span_lengths
    ).long()
    offsets_b = torch.floor(
        torch.rand(sample_count, generator=generator) * (span_lengths - 1)
    ).long()
    offsets_b = offsets_b + (offsets_b >= offsets_a).long()
    positions_a = span_starts + offsets_a
    positions_b = span_starts + offsets_b
    rows = (
        torch.arange(sample_count)
        if row_indices is None
        else row_indices
    )
    if rows.shape != (sample_count,):
        raise ValueError("row_indices must have one entry per pair")
    return {
        "view_a": sequences[rows, positions_a],
        "view_b": sequences[rows, positions_b],
        "span_length": span_lengths,
        "distance": (positions_a - positions_b).abs(),
        "span_start_index": span_starts,
        "span_end_index": span_ends,
        "position_a": positions_a,
        "position_b": positions_b,
    }


class RandomViewPairShardBatches:
    """Infinite deterministic batches with shard-I/O-amortized view pairs.

    Several independently sampled pairs are materialized while a sequence shard
    is resident in host memory.  A bounded CPU buffer then mixes those pairs and
    preferentially places at most one pair from a sequence in a batch (falling
    back to ``max_pairs_per_sequence_per_batch`` only near buffer exhaustion).
    This preserves sequence diversity for the batch-distribution RDM objective
    without rereading a full residual trajectory for every single pair.
    """

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        split: str,
        batch_size: int,
        seed: int,
        pairs_per_sequence: int = 8,
        max_pairs_per_sequence_per_batch: int = 2,
        shuffle_buffer_pairs: int = 4096,
    ):
        self.paths = shard_paths(root, manifest, split)
        self.sequence_length = int(manifest["sequence_length"])
        self.min_span_length = int(manifest["min_span_length"])
        self.max_span_length = int(manifest["max_span_length"])
        self.max_horizon = int(manifest["max_horizon"])
        self.burn_in_tokens = int(manifest["burn_in_tokens"])
        self.batch_size = batch_size
        self.pairs_per_sequence = pairs_per_sequence
        self.max_pairs_per_sequence_per_batch = (
            max_pairs_per_sequence_per_batch
        )
        pairs_per_epoch = int(manifest[split]["sequences"]) * pairs_per_sequence
        self.shuffle_buffer_pairs = max(
            batch_size, min(shuffle_buffer_pairs, pairs_per_epoch)
        )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if pairs_per_sequence < 1:
            raise ValueError("pairs_per_sequence must be positive")
        if max_pairs_per_sequence_per_batch < 1:
            raise ValueError(
                "max_pairs_per_sequence_per_batch must be positive"
            )
        if shuffle_buffer_pairs < 1:
            raise ValueError("shuffle_buffer_pairs must be positive")
        minimum_sequences = math.ceil(
            batch_size / max_pairs_per_sequence_per_batch
        )
        if int(manifest[split]["sequences"]) < minimum_sequences:
            raise ValueError(
                f"{split} needs at least {minimum_sequences} sequences for "
                "the requested per-batch sequence cap"
            )
        self.generator = torch.Generator().manual_seed(seed)
        self.path_order: list[int] = []
        self.path_position = 0
        self.epoch = 0
        self.pending_chunks: list[dict[str, torch.Tensor]] = []
        self.pending_count = 0
        self.ready_batches: list[dict[str, torch.Tensor]] = []

    def __iter__(self) -> RandomViewPairShardBatches:
        return self

    def _next_path_index(self) -> int:
        if self.path_position >= len(self.path_order):
            self.path_order = torch.randperm(
                len(self.paths), generator=self.generator
            ).tolist()
            self.path_position = 0
            self.epoch += 1
        path_index = self.path_order[self.path_position]
        self.path_position += 1
        return path_index

    def _sample_next_shard(self) -> dict[str, torch.Tensor]:
        path_index = self._next_path_index()
        shard = load_sequence_shard(
            self.paths[path_index], self.sequence_length
        )
        sequence_count = len(shard.activations)
        row_indices = torch.arange(sequence_count).repeat_interleave(
            self.pairs_per_sequence
        )
        sample_count = len(row_indices)
        span_lengths = torch.randint(
            self.min_span_length,
            self.max_span_length + 1,
            (sample_count,),
            generator=self.generator,
        )
        sampled = _sample_view_pairs(
            shard.activations,
            shard.valid_lengths.index_select(0, row_indices),
            span_lengths,
            self.burn_in_tokens,
            self.max_horizon,
            self.generator,
            row_indices=row_indices,
        )
        # Shard index and row uniquely identify a physical training sequence.
        # Keeping this identifier in batches also makes sequence diversity
        # directly auditable in tests and training diagnostics.
        sampled["sequence_id"] = (
            path_index * (1 << 32) + row_indices.to(torch.int64)
        )
        permutation = torch.randperm(sample_count, generator=self.generator)
        return {
            key: value.index_select(0, permutation)
            for key, value in sampled.items()
        }

    def _append_until_buffer_full(self) -> None:
        while self.pending_count < self.shuffle_buffer_pairs:
            chunk = self._sample_next_shard()
            self.pending_chunks.append(chunk)
            self.pending_count += len(chunk["view_a"])

    def _build_ready_batches(self) -> None:
        self._append_until_buffer_full()
        pool = {
            key: torch.cat([chunk[key] for chunk in self.pending_chunks])
            for key in self.pending_chunks[0]
        }
        sequence_ids = pool["sequence_id"].tolist()
        grouped: dict[int, list[int]] = {}
        for index, sequence_id in enumerate(sequence_ids):
            grouped.setdefault(sequence_id, []).append(index)
        for indices in grouped.values():
            order = torch.randperm(len(indices), generator=self.generator).tolist()
            indices[:] = [indices[index] for index in order]

        ready_indices: list[list[int]] = []
        while sum(
            min(len(indices), self.max_pairs_per_sequence_per_batch)
            for indices in grouped.values()
        ) >= self.batch_size:
            selected: list[int] = []
            for _ in range(self.max_pairs_per_sequence_per_batch):
                available = [
                    sequence_id
                    for sequence_id, indices in grouped.items()
                    if indices
                ]
                if not available:
                    break
                order = torch.randperm(
                    len(available), generator=self.generator
                ).tolist()
                for offset in order:
                    selected.append(grouped[available[offset]].pop())
                    if len(selected) == self.batch_size:
                        break
                if len(selected) == self.batch_size:
                    break
            if len(selected) != self.batch_size:
                break
            ready_indices.append(selected)

        if not ready_indices:
            raise RuntimeError(
                "pair shuffle buffer could not form a sequence-diverse batch; "
                "increase shuffle_buffer_pairs or the per-batch sequence cap"
            )
        self.ready_batches = [
            {
                key: value.index_select(
                    0, torch.tensor(indices, dtype=torch.long)
                )
                for key, value in pool.items()
            }
            for indices in ready_indices
        ]
        # __next__ uses pop(), so reverse once to preserve construction order:
        # high-diversity batches are emitted before the buffer tail that may
        # require the configured second pair from a sequence.
        self.ready_batches.reverse()
        remaining = [
            index for indices in grouped.values() for index in indices
        ]
        if remaining:
            remaining_indices = torch.tensor(remaining, dtype=torch.long)
            self.pending_chunks = [
                {
                    key: value.index_select(0, remaining_indices)
                    for key, value in pool.items()
                }
            ]
            self.pending_count = len(remaining)
        else:
            self.pending_chunks = []
            self.pending_count = 0

    def __next__(self) -> dict[str, torch.Tensor]:
        if not self.ready_batches:
            self._build_ready_batches()
        return self.ready_batches.pop()


def validation_view_pair_batches(
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
            yield _sample_view_pairs(
                sequences,
                lengths,
                span_lengths,
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
