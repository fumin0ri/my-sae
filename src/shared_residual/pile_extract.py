from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from tqdm import tqdm

from .io import write_json
from .modeling import (
    forward_backbone,
    get_layer,
    input_device,
    load_hf_model,
    tensor_from_layer_output,
)
from .training import configure_accelerator
from .activation_store import ACTIVATION_FORMAT


# Official effective mixture after the component-specific epoch multipliers.
# Source: EleutherAI/the-pile README. Values sum to approximately one.
PILE_MIXTURE_WEIGHTS = {
    "Pile-CC": 0.1811,
    "PubMed Central": 0.1440,
    "Books3": 0.1207,
    "OpenWebText2": 0.1001,
    "ArXiv": 0.0896,
    "Github": 0.0759,
    "FreeLaw": 0.0612,
    "StackExchange": 0.0513,
    "USPTO Backgrounds": 0.0365,
    "PubMed Abstracts": 0.0307,
    "Gutenberg (PG-19)": 0.0217,
    "OpenSubtitles": 0.0155,
    "Wikipedia (en)": 0.0153,
    "DM Mathematics": 0.0124,
    "Ubuntu IRC": 0.0088,
    "BookCorpus2": 0.0075,
    "EuroParl": 0.0073,
    "HackerNews": 0.0062,
    "YoutubeSubtitles": 0.0060,
    "PhilPapers": 0.0038,
    "NIH ExPorter": 0.0030,
    "Enron Emails": 0.0014,
}

DEFAULT_TRAIN_POSITIONS = 5_242_880
DEFAULT_VALIDATION_POSITIONS = 163_840
DEFAULT_SHARD_POSITIONS = 40_960
STORAGE_SAFETY_FACTOR = 1.05


def sequences_from_position_budget(positions: int, sequence_length: int) -> int:
    if positions < 1:
        raise ValueError("position budgets must be positive")
    if sequence_length < 1:
        raise ValueError("sequence length must be positive")
    return max(math.ceil(positions / sequence_length), 1)


def resolve_sequence_count(
    explicit_sequences: int | None,
    position_budget: int,
    sequence_length: int,
) -> int:
    if explicit_sequences is not None:
        return explicit_sequences
    return sequences_from_position_budget(position_budget, sequence_length)


def estimate_storage_bytes(
    sequences: int,
    sequence_length: int,
    d_in: int,
) -> int:
    positions = sequences * sequence_length
    # BF16 residuals and int64 token IDs dominate. Source IDs are int16 per
    # sequence; valid lengths are int32. The multiplier covers metadata.
    raw_bytes = positions * (d_in * 2 + 8) + sequences * 6
    return int(raw_bytes * STORAGE_SAFETY_FACTOR)


def human_gib(size: int) -> str:
    return f"{size / 2**30:.1f} GiB"


def ensure_new_output(output_dir: Path) -> None:
    artifacts = []
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        artifacts.append(manifest)
    for split in ("train", "validation"):
        directory = output_dir / split
        if directory.exists():
            artifacts.extend(directory.glob("shard-*.pt"))
            artifacts.extend(directory.glob("shard-*.pt.partial"))
    if artifacts:
        sample = ", ".join(str(path) for path in artifacts[:3])
        raise FileExistsError(
            f"{output_dir} already contains extraction artifacts ({sample}). "
            "Use a new --output-dir, or remove the failed extraction directory "
            "after verifying it is no longer needed."
        )


def check_disk_capacity(
    output_dir: Path,
    estimated_bytes: int,
    reserve_gib: float,
) -> None:
    free_bytes = shutil.disk_usage(output_dir).free
    reserve_bytes = int(reserve_gib * 2**30)
    print(
        "activation storage preflight: "
        f"estimated={human_gib(estimated_bytes)}, "
        f"free={human_gib(free_bytes)}, reserve={reserve_gib:.1f} GiB"
    )
    if free_bytes < estimated_bytes + reserve_bytes:
        raise RuntimeError(
            "Insufficient filesystem space for the requested activation "
            f"shards: need approximately {human_gib(estimated_bytes)} plus "
            f"{reserve_gib:.1f} GiB reserve, but {human_gib(free_bytes)} is "
            "free. Reduce the position/sequence budget, choose another "
            "--output-dir, or pass --skip-disk-space-check only after "
            "checking filesystem and user quota."
        )


def pile_set_name(row: dict[str, Any]) -> str:
    meta = row.get("meta", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if isinstance(meta, dict):
        return str(meta.get("pile_set_name", "unknown"))
    return "unknown"


def document_split(index: int, seed: int, validation_fraction: float) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{index}".encode("utf-8"),
        digest_size=8,
    ).digest()
    uniform = int.from_bytes(digest, "little") / 2**64
    return "validation" if uniform < validation_fraction else "train"


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        shard_sequences: int,
        target_sequences: int,
    ):
        self.directory = output_dir / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_sequences = shard_sequences
        self.target_sequences = target_sequences
        self.buffer_x: list[torch.Tensor] = []
        self.buffer_tokens: list[torch.Tensor] = []
        self.buffer_sources: list[torch.Tensor] = []
        self.buffer_lengths: list[torch.Tensor] = []
        self.buffer_count = 0
        self.written = 0
        self.shards: list[str] = []
        self.domain_counts: Counter[str] = Counter()
        self.sum: torch.Tensor | None = None
        self.sum_squares = 0.0
        self.positions = 0

    @property
    def remaining(self) -> int:
        return self.target_sequences - self.written - self.buffer_count

    def append(
        self,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        valid_lengths: torch.Tensor,
        source_ids: torch.Tensor,
        source_names: list[str],
    ) -> int:
        take = min(len(activations), max(self.remaining, 0))
        if take <= 0:
            return 0
        selected = activations[:take].detach()
        lengths = valid_lengths[:take].detach().long().cpu()
        selected_float = selected.float()
        valid_mask = (
            torch.arange(selected.shape[1], device=selected.device)[None, :]
            < lengths.to(selected.device)[:, None]
        )
        valid_values = selected_float[valid_mask]
        batch_sum = valid_values.sum(dim=0).double().cpu()
        batch_sum_squares = float(valid_values.square().sum().item())
        values = selected.to(torch.bfloat16).cpu()
        tokens = token_ids[:take].detach().cpu()
        sources = source_ids[:take].detach().to(torch.int16).cpu()
        self.buffer_x.append(values)
        self.buffer_tokens.append(tokens)
        self.buffer_sources.append(sources)
        self.buffer_lengths.append(lengths.to(torch.int32))
        self.buffer_count += take
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.sum_squares += batch_sum_squares
        self.positions += int(lengths.sum())
        self.domain_counts.update(
            source_names[index] for index in sources.tolist()
        )
        while self.buffer_count >= self.shard_sequences:
            self._flush(self.shard_sequences)
        return take

    def _flush(self, count: int) -> None:
        activations = torch.cat(self.buffer_x)
        token_ids = torch.cat(self.buffer_tokens)
        source_ids = torch.cat(self.buffer_sources)
        valid_lengths = torch.cat(self.buffer_lengths)
        output_x, activations = activations[:count], activations[count:]
        output_tokens, token_ids = token_ids[:count], token_ids[count:]
        output_sources, source_ids = source_ids[:count], source_ids[count:]
        output_lengths, valid_lengths = valid_lengths[:count], valid_lengths[count:]
        shard_index = len(self.shards)
        relative = f"{self.split}/shard-{shard_index:05d}.pt"
        output_path = self.directory / f"shard-{shard_index:05d}.pt"
        partial_path = output_path.with_suffix(".pt.partial")
        try:
            torch.save(
                {
                    "activations": output_x.contiguous(),
                    "token_ids": output_tokens.contiguous(),
                    "source_ids": output_sources.contiguous(),
                    "valid_lengths": output_lengths.contiguous(),
                },
                partial_path,
            )
            partial_path.replace(output_path)
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            try:
                free = human_gib(shutil.disk_usage(self.directory).free)
            except OSError:
                free = "unknown"
            raise RuntimeError(
                f"Failed to atomically write {output_path}; filesystem free "
                f"space is {free}. Check both disk space and user quota."
            ) from error
        self.shards.append(relative)
        self.written += count
        self.buffer_x = [activations] if len(activations) else []
        self.buffer_tokens = [token_ids] if len(token_ids) else []
        self.buffer_sources = [source_ids] if len(source_ids) else []
        self.buffer_lengths = [valid_lengths] if len(valid_lengths) else []
        self.buffer_count = len(activations)

    def finish(self) -> None:
        if self.buffer_count:
            self._flush(self.buffer_count)
        if self.written != self.target_sequences:
            raise RuntimeError(
                f"{self.split}: expected {self.target_sequences} sequences, "
                f"wrote {self.written}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "sequences": self.written,
            "positions": self.positions,
            "shards": self.shards,
            "domain_counts": dict(sorted(self.domain_counts.items())),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream The Pile and write long frozen-LLM residual sequences"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-point", choices=["pre", "post"], default="post")
    parser.add_argument(
        "--dataset",
        default="EleutherAI/the_pile_deduplicated",
    )
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--dataset-trust-remote-code",
        action="store_true",
        help="Allow a legacy Hugging Face dataset loading script",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--train-sequences",
        type=int,
        help=(
            "Explicit train sequence count. By default this is derived from "
            "--train-positions."
        ),
    )
    parser.add_argument(
        "--validation-sequences",
        type=int,
        help=(
            "Explicit validation sequence count. By default this is derived "
            "from --validation-positions."
        ),
    )
    parser.add_argument(
        "--train-positions",
        type=int,
        default=DEFAULT_TRAIN_POSITIONS,
    )
    parser.add_argument(
        "--validation-positions",
        type=int,
        default=DEFAULT_VALIDATION_POSITIONS,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument(
        "--max-span-length",
        "--window-size",
        dest="max_span_length",
        type=int,
        default=10,
        help="Maximum sampled span length; the maximum horizon is one less.",
    )
    parser.add_argument(
        "--min-span-length",
        type=int,
        default=2,
        help="Minimum sampled span length (inclusive).",
    )
    parser.add_argument("--sequence-length", type=int, default=320)
    parser.add_argument(
        "--burn-in-tokens",
        type=int,
        help="Minimum context index inside each long sequence (default: max span).",
    )
    parser.add_argument("--max-document-tokens", type=int, default=16384)
    parser.add_argument(
        "--shard-sequences",
        type=int,
        help=(
            "Explicit sequences per shard. By default this is derived from "
            "--shard-positions to bound shard size and host RAM."
        ),
    )
    parser.add_argument(
        "--shard-positions",
        type=int,
        default=DEFAULT_SHARD_POSITIONS,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-all-domains",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require all 22 source labels. The default deduplicated Parquet "
            "release has no per-document source metadata, so this is intended "
            "for the legacy EleutherAI/pile all configuration."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disk-reserve-gib", type=float, default=5.0)
    parser.add_argument("--skip-disk-space-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_span_length < 2:
        raise ValueError("--max-span-length must be at least 2")
    if not 2 <= args.min_span_length <= args.max_span_length:
        raise ValueError(
            "--min-span-length must lie in [2, --max-span-length]"
        )
    if args.burn_in_tokens is None:
        args.burn_in_tokens = args.max_span_length
    if args.burn_in_tokens < 0:
        raise ValueError("--burn-in-tokens cannot be negative")
    minimum_valid_length = args.burn_in_tokens + args.max_span_length
    if args.sequence_length < minimum_valid_length:
        raise ValueError(
            "--sequence-length must be at least burn_in_tokens + max_span_length"
        )
    if min(
        args.train_positions,
        args.validation_positions,
        args.shard_positions,
    ) < 1:
        raise ValueError("position budgets must be positive")
    explicit_train_sequences = args.train_sequences
    explicit_validation_sequences = args.validation_sequences
    explicit_shard_sequences = args.shard_sequences
    args.train_sequences = resolve_sequence_count(
        args.train_sequences,
        args.train_positions,
        args.sequence_length,
    )
    args.validation_sequences = resolve_sequence_count(
        args.validation_sequences,
        args.validation_positions,
        args.sequence_length,
    )
    args.shard_sequences = resolve_sequence_count(
        args.shard_sequences,
        args.shard_positions,
        args.sequence_length,
    )
    if min(
        args.train_sequences,
        args.validation_sequences,
        args.shard_sequences,
        args.batch_size,
    ) < 1:
        raise ValueError("sequence counts, shard size, and batch size must be positive")
    if args.disk_reserve_gib < 0:
        raise ValueError("--disk-reserve-gib cannot be negative")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must lie in (0, 0.5)")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The Pile extractor needs `datasets`; reinstall with "
            "`python -m pip install --upgrade -e .`"
        ) from error

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_new_output(output_dir)
    stream = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
        revision=args.dataset_revision,
        trust_remote_code=args.dataset_trust_remote_code,
    )
    stream = stream.shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer,
    )
    model, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device_map,
        args.trust_remote_code,
        args.revision,
        use_safetensors=True if args.use_safetensors else None,
    )
    d_in = int(model.get_input_embeddings().weight.shape[-1])
    estimated_storage = estimate_storage_bytes(
        args.train_sequences + args.validation_sequences,
        args.sequence_length,
        d_in,
    )
    print(
        "resolved extraction budget: "
        f"sequence_length={args.sequence_length}, "
        f"span={args.min_span_length}..{args.max_span_length}, "
        f"burn_in={args.burn_in_tokens}, "
        f"train={args.train_sequences:,} sequences, "
        f"validation={args.validation_sequences:,} sequences, "
        f"shard={args.shard_sequences:,} sequences"
    )
    if not args.skip_disk_space_check:
        check_disk_capacity(
            output_dir,
            estimated_storage,
            args.disk_reserve_gib,
        )
    configure_accelerator(input_device(model))
    layer_path, layer = get_layer(model, args.layer)
    captured: dict[str, torch.Tensor] = {}
    if args.hook_point == "pre":
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            captured["x"] = inputs[0].detach()

        handle = layer.register_forward_pre_hook(hook)
    else:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            captured["x"] = tensor_from_layer_output(output).detach()

        handle = layer.register_forward_hook(hook)

    writers = {
        "train": ShardWriter(
            output_dir,
            "train",
            args.shard_sequences,
            args.train_sequences,
        ),
        "validation": ShardWriter(
            output_dir,
            "validation",
            args.shard_sequences,
            args.validation_sequences,
        ),
    }
    source_to_id: dict[str, int] = {}
    sequence_ids: list[torch.Tensor] = []
    sequence_splits: list[str] = []
    sequence_sources: list[int] = []
    sequence_valid_lengths: list[int] = []
    documents = 0
    discarded_short_documents = 0
    target_total = args.train_sequences + args.validation_sequences
    progress = tqdm(total=target_total, unit="sequence", desc="Pile residuals")

    def run_batch() -> None:
        if not sequence_ids:
            return
        model_device = input_device(model)
        valid_lengths = torch.tensor(
            sequence_valid_lengths,
            device=model_device,
        )
        encoded = {
            "input_ids": torch.stack(sequence_ids).to(model_device),
            "attention_mask": (
                torch.arange(
                    args.sequence_length,
                    device=model_device,
                )[None, :]
                < valid_lengths[:, None]
            ).long(),
        }
        captured.clear()
        with torch.inference_mode():
            forward_backbone(model, encoded)
        hidden = captured["x"]
        for index, split_name in enumerate(sequence_splits):
            writer = writers[split_name]
            sources = torch.full(
                (1,),
                sequence_sources[index],
                dtype=torch.long,
                device=hidden.device,
            )
            names = [
                name
                for name, _ in sorted(
                    source_to_id.items(),
                    key=lambda item: item[1],
                )
            ]
            accepted = writer.append(
                hidden[index : index + 1],
                encoded["input_ids"][index : index + 1],
                valid_lengths[index : index + 1],
                sources,
                names,
            )
            progress.update(accepted)
        sequence_ids.clear()
        sequence_splits.clear()
        sequence_sources.clear()
        sequence_valid_lengths.clear()

    try:
        for document_index, row in enumerate(stream):
            if all(writer.remaining <= 0 for writer in writers.values()):
                break
            documents += 1
            split_name = document_split(
                document_index,
                args.seed,
                args.validation_fraction,
            )
            if writers[split_name].remaining <= 0:
                continue
            text = str(row.get(args.text_key, ""))
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_document_tokens,
                return_attention_mask=False,
            )["input_ids"]
            if len(ids) < minimum_valid_length:
                discarded_short_documents += 1
                continue
            source = pile_set_name(row)
            if source not in source_to_id:
                source_to_id[source] = len(source_to_id)
            source_id = source_to_id[source]
            for start in range(0, len(ids), args.sequence_length):
                if writers[split_name].remaining <= 0:
                    break
                chunk = ids[
                    start : min(start + args.sequence_length, len(ids))
                ]
                if len(chunk) < minimum_valid_length:
                    continue
                padded = chunk + [tokenizer.pad_token_id] * (
                    args.sequence_length - len(chunk)
                )
                sequence_ids.append(
                    torch.tensor(
                        padded,
                        dtype=torch.long,
                    )
                )
                sequence_splits.append(split_name)
                sequence_sources.append(source_id)
                sequence_valid_lengths.append(len(chunk))
                if len(sequence_ids) == args.batch_size:
                    run_batch()
        run_batch()
    finally:
        handle.remove()
        progress.close()
    for writer in writers.values():
        writer.finish()

    train_writer = writers["train"]
    assert train_writer.sum is not None
    mean = train_writer.sum / train_writer.positions
    mean_square = train_writer.sum_squares / train_writer.positions
    variance_sum = mean_square - float(mean.square().sum().item())
    scale = max(variance_sum / len(mean), 1e-16) ** 0.5
    observed = set(train_writer.domain_counts)
    source_metadata_available = bool(observed - {"unknown"})
    missing = sorted(set(PILE_MIXTURE_WEIGHTS) - observed)
    if args.require_all_domains and missing:
        raise RuntimeError(
            "Pile sample did not cover all official subcorpora: "
            + ", ".join(missing)
        )
    manifest = {
        "format": ACTIVATION_FORMAT,
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "revision": args.dataset_revision,
            "split": args.split,
            "streaming": True,
            "shuffle_buffer": args.shuffle_buffer,
            "official_mixture_weights": PILE_MIXTURE_WEIGHTS,
            "mixture_policy": (
                "upstream preweighted Pile mixture plus streaming buffer shuffle"
            ),
            "source_metadata_available": source_metadata_available,
            "require_all_domains": args.require_all_domains,
        },
        "model": args.model,
        "requested_model_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "layer": args.layer,
        "layer_path": layer_path,
        "hook_point": args.hook_point,
        "max_span_length": args.max_span_length,
        "min_span_length": args.min_span_length,
        "max_horizon": args.max_span_length - 1,
        "sequence_length": args.sequence_length,
        "burn_in_tokens": args.burn_in_tokens,
        "minimum_valid_length": minimum_valid_length,
        "d_in": int(train_writer.sum.numel()),
        "storage_dtype": "bfloat16",
        "storage": {
            "estimated_bytes": estimated_storage,
            "safety_factor": STORAGE_SAFETY_FACTOR,
            "disk_reserve_gib": args.disk_reserve_gib,
        },
        "sampling_budget": {
            "train_positions": args.train_positions,
            "validation_positions": args.validation_positions,
            "shard_positions": args.shard_positions,
            "train_sequences_explicit": explicit_train_sequences,
            "validation_sequences_explicit": explicit_validation_sequences,
            "shard_sequences_explicit": explicit_shard_sequences,
        },
        "pair_sampling": {
            "performed_online_during_training": True,
            "span_length_distribution": "uniform over min_span_length..max_span_length",
            "context_distribution": "uniform over non-endpoint positions in the sampled span",
            "horizon_rule": "h=t-k, so h is uniform over 1..L-1 conditional on L",
            "endpoint_distribution": "uniform over one horizon-independent eligible range",
            "context_rule": "k=t-h",
            "boundary_rule": "k >= burn_in_tokens",
        },
        "normalization": {
            "mean": mean.float().tolist(),
            "scalar_rms": scale,
        },
        "source_names": [
            name
            for name, _ in sorted(source_to_id.items(), key=lambda item: item[1])
        ],
        "train": train_writer.summary(),
        "validation": writers["validation"].summary(),
        "seed": args.seed,
        "documents_consumed": documents,
        "discarded_short_documents": discarded_short_documents,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"wrote {args.train_sequences:,} train and "
        f"{args.validation_sequences:,} validation sequences to {output_dir}"
    )


if __name__ == "__main__":
    main()
