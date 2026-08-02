import json
from pathlib import Path

import pytest
import torch

from shared_residual.activation_store import (
    ACTIVATION_FORMAT,
    RandomPairShardBatches,
    load_activation_manifest,
    manifest_fingerprint,
    validation_batches,
    validation_pair_batches,
)
from shared_residual.pile_extract import (
    DEFAULT_SHARD_POSITIONS,
    DEFAULT_TRAIN_POSITIONS,
    DEFAULT_VALIDATION_POSITIONS,
    PILE_MIXTURE_WEIGHTS,
    ShardWriter,
    build_parser,
    document_split,
    estimate_storage_bytes,
    pile_set_name,
    resolve_sequence_count,
)


def make_manifest(tmp_path, sequence_length=12, max_span_length=5):
    for split, count in (("train", 7), ("validation", 5)):
        directory = tmp_path / split
        directory.mkdir()
        torch.save(
            {
                "activations": torch.arange(
                    count * sequence_length * 4, dtype=torch.float32
                ).reshape(count, sequence_length, 4),
                "token_ids": torch.zeros(count, sequence_length, dtype=torch.long),
                "source_ids": torch.zeros(count, dtype=torch.int16),
                "valid_lengths": torch.full(
                    (count,), sequence_length, dtype=torch.int32
                ),
            },
            directory / "shard-00000.pt",
        )
    manifest = {
        "format": ACTIVATION_FORMAT,
        "dataset": {
            "name": "EleutherAI/the_pile_deduplicated",
            "config": "default",
        },
        "model": "test",
        "layer": 1,
        "layer_path": "layers.1",
        "hook_point": "post",
        "max_span_length": max_span_length,
        "min_span_length": 2,
        "max_horizon": max_span_length - 1,
        "sequence_length": sequence_length,
        "burn_in_tokens": 2,
        "minimum_valid_length": 2 + max_span_length,
        "d_in": 4,
        "seed": 3,
        "normalization": {"mean": [0.0] * 4, "scalar_rms": 1.0},
        "train": {
            "sequences": 7,
            "positions": 7 * sequence_length,
            "shards": ["train/shard-00000.pt"],
            "domain_counts": {"Pile-CC": 7},
        },
        "validation": {
            "sequences": 5,
            "positions": 5 * sequence_length,
            "shards": ["validation/shard-00000.pt"],
            "domain_counts": {"Pile-CC": 5},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_random_pair_batches_follow_random_span_design_and_are_boundary_safe(
    tmp_path,
) -> None:
    expected = make_manifest(tmp_path)
    root, manifest = load_activation_manifest(tmp_path)
    assert manifest_fingerprint(manifest) == manifest_fingerprint(expected)
    iterator = iter(
        RandomPairShardBatches(root, manifest, "train", batch_size=8, seed=9)
    )
    batch = next(iterator)
    assert batch["context"].shape == (8, 4)
    assert batch["target"].shape == (8, 4)
    assert torch.equal(
        batch["endpoint_index"] - batch["context_index"], batch["horizon"]
    )
    assert torch.all(batch["span_length"] >= manifest["min_span_length"])
    assert torch.all(batch["span_length"] <= manifest["max_span_length"])
    assert torch.all(batch["horizon"] < batch["span_length"])
    assert torch.all(batch["context_index"] >= batch["span_start_index"])
    assert torch.all(batch["context_index"] < batch["endpoint_index"])
    assert int(batch["context_index"].min()) >= manifest["burn_in_tokens"]
    assert int(batch["endpoint_index"].min()) >= (
        manifest["burn_in_tokens"] + manifest["max_horizon"]
    )


def test_validation_pairs_are_deterministic_and_residuals_exclude_padding(
    tmp_path,
) -> None:
    make_manifest(tmp_path)
    root, manifest = load_activation_manifest(tmp_path)
    left = list(validation_pair_batches(root, manifest, 3, 2, seed=11))
    right = list(validation_pair_batches(root, manifest, 3, 2, seed=11))
    assert len(left) == len(right) == 2
    assert torch.equal(left[0]["endpoint_index"], right[0]["endpoint_index"])
    held_out = list(validation_batches(root, manifest, 20, 2))
    assert [len(batch) for batch in held_out] == [20, 20]


def test_pile_metadata_and_official_mixture() -> None:
    assert pile_set_name({"meta": {"pile_set_name": "ArXiv"}}) == "ArXiv"
    assert pile_set_name({"meta": '{"pile_set_name": "Pile-CC"}'}) == "Pile-CC"
    assert len(PILE_MIXTURE_WEIGHTS) == 22
    assert abs(sum(PILE_MIXTURE_WEIGHTS.values()) - 1.0) < 1e-9
    assert pile_set_name({"text": "metadata-free Parquet row"}) == "unknown"


def test_deduplicated_pile_uses_real_hugging_face_config() -> None:
    args = build_parser().parse_args(
        ["--model", "test", "--output-dir", "out", "--layer", "1"]
    )
    assert args.dataset == "EleutherAI/the_pile_deduplicated"
    assert args.dataset_config == "default"


def test_pile_parser_accepts_nondefault_max_span() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "test",
            "--output-dir",
            "out",
            "--layer",
            "1",
            "--max-span-length",
            "16",
            "--sequence-length",
            "320",
        ]
    )
    assert args.max_span_length == 16
    assert args.min_span_length == 2
    assert args.train_sequences is None
    assert args.validation_sequences is None
    assert args.shard_sequences is None


def test_position_budget_keeps_storage_bounded() -> None:
    train = resolve_sequence_count(None, DEFAULT_TRAIN_POSITIONS, 320)
    validation = resolve_sequence_count(None, DEFAULT_VALIDATION_POSITIONS, 320)
    shard = resolve_sequence_count(None, DEFAULT_SHARD_POSITIONS, 320)
    assert train == 16_384
    assert validation == 512
    assert shard == 128
    estimated = estimate_storage_bytes(train + validation, 320, 4096)
    assert 40 * 2**30 < estimated < 45 * 2**30


def test_explicit_sequence_count_overrides_position_budget() -> None:
    assert resolve_sequence_count(123, DEFAULT_TRAIN_POSITIONS, 320) == 123


def test_shard_write_failure_leaves_no_corrupt_final_file(
    tmp_path,
    monkeypatch,
) -> None:
    writer = ShardWriter(
        tmp_path,
        split="train",
        shard_sequences=1,
        target_sequences=1,
    )

    def fail_after_partial_write(_payload, path) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="disk space and user quota"):
        writer.append(
            torch.zeros(1, 8, 4),
            torch.zeros(1, 8, dtype=torch.long),
            torch.tensor([8]),
            torch.zeros(1, dtype=torch.long),
            ["test"],
        )
    assert not list((tmp_path / "train").iterdir())


def test_document_split_is_deterministic_and_nontrivial() -> None:
    left = [document_split(index, 7, 0.1) for index in range(1000)]
    right = [document_split(index, 7, 0.1) for index in range(1000)]
    assert left == right
    assert 60 < left.count("validation") < 140
