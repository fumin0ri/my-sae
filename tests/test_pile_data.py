import json

import torch

from shared_residual.activation_store import (
    ShuffledShardBatches,
    load_activation_manifest,
    manifest_fingerprint,
    validation_batches,
)
from shared_residual.pile_extract import (
    PILE_MIXTURE_WEIGHTS,
    build_parser,
    document_split,
    pile_set_name,
)


def make_manifest(tmp_path):
    for split, count in (("train", 7), ("validation", 5)):
        directory = tmp_path / split
        directory.mkdir()
        torch.save(
            {
                "activations": torch.arange(
                    count * 10 * 4,
                    dtype=torch.float32,
                ).reshape(count, 10, 4),
                "token_ids": torch.zeros(count, 10, dtype=torch.long),
                "source_ids": torch.zeros(count, dtype=torch.int16),
            },
            directory / "shard-00000.pt",
        )
    manifest = {
        "format": "shared-residual-activation-shards-v1",
        "dataset": {
            "name": "EleutherAI/the_pile_deduplicated",
            "config": "default",
        },
        "model": "test",
        "layer": 1,
        "layer_path": "layers.1",
        "hook_point": "post",
        "window_size": 10,
        "d_in": 4,
        "seed": 3,
        "normalization": {"mean": [0.0] * 4, "scalar_rms": 1.0},
        "train": {
            "windows": 7,
            "shards": ["train/shard-00000.pt"],
            "domain_counts": {"Pile-CC": 7},
        },
        "validation": {
            "windows": 5,
            "shards": ["validation/shard-00000.pt"],
            "domain_counts": {"Pile-CC": 5},
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest


def test_sharded_batches_cycle_without_loading_corpus_at_once(tmp_path) -> None:
    expected = make_manifest(tmp_path)
    root, manifest = load_activation_manifest(tmp_path)
    assert manifest_fingerprint(manifest) == manifest_fingerprint(expected)
    iterator = iter(
        ShuffledShardBatches(
            root,
            manifest,
            "train",
            batch_size=4,
            seed=9,
        )
    )
    assert next(iterator).shape == (4, 10, 4)
    assert next(iterator).shape == (4, 10, 4)
    held_out = list(validation_batches(root, manifest, 3, 2))
    assert [len(batch) for batch in held_out] == [3, 2]


def test_pile_metadata_and_official_mixture() -> None:
    assert pile_set_name({"meta": {"pile_set_name": "ArXiv"}}) == "ArXiv"
    assert (
        pile_set_name({"meta": '{"pile_set_name": "Pile-CC"}'})
        == "Pile-CC"
    )
    assert len(PILE_MIXTURE_WEIGHTS) == 22
    assert abs(sum(PILE_MIXTURE_WEIGHTS.values()) - 1.0) < 1e-9
    assert pile_set_name({"text": "metadata-free Parquet row"}) == "unknown"


def test_deduplicated_pile_uses_real_hugging_face_config() -> None:
    args = build_parser().parse_args(
        ["--model", "test", "--output-dir", "out", "--layer", "1"]
    )
    assert args.dataset == "EleutherAI/the_pile_deduplicated"
    assert args.dataset_config == "default"


def test_document_split_is_deterministic_and_nontrivial() -> None:
    left = [document_split(index, 7, 0.1) for index in range(1000)]
    right = [document_split(index, 7, 0.1) for index in range(1000)]
    assert left == right
    assert 60 < left.count("validation") < 140
