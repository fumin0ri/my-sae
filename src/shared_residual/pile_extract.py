from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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


def windows_from_position_budget(positions: int, window_size: int) -> int:
    if positions < 1:
        raise ValueError("position budgets must be positive")
    if window_size < 1:
        raise ValueError("window size must be positive")
    return max(positions // window_size, 1)


def resolve_window_count(
    explicit_windows: int | None,
    position_budget: int,
    window_size: int,
) -> int:
    if explicit_windows is not None:
        return explicit_windows
    return windows_from_position_budget(position_budget, window_size)


def estimate_storage_bytes(
    windows: int,
    window_size: int,
    d_in: int,
) -> int:
    positions = windows * window_size
    # BF16 residuals and int64 token IDs dominate. Source IDs are int16 per
    # window; the multiplier covers serialization metadata and alignment.
    raw_bytes = positions * (d_in * 2 + 8) + windows * 2
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
            "free. Reduce the position/window budget, choose another "
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
        shard_windows: int,
        target_windows: int,
    ):
        self.directory = output_dir / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_windows = shard_windows
        self.target_windows = target_windows
        self.buffer_x: list[torch.Tensor] = []
        self.buffer_tokens: list[torch.Tensor] = []
        self.buffer_sources: list[torch.Tensor] = []
        self.buffer_count = 0
        self.written = 0
        self.shards: list[str] = []
        self.domain_counts: Counter[str] = Counter()
        self.sum: torch.Tensor | None = None
        self.sum_squares = 0.0
        self.positions = 0

    @property
    def remaining(self) -> int:
        return self.target_windows - self.written - self.buffer_count

    def append(
        self,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        source_ids: torch.Tensor,
        source_names: list[str],
    ) -> int:
        take = min(len(activations), max(self.remaining, 0))
        if take <= 0:
            return 0
        selected = activations[:take].detach()
        selected_float = selected.float()
        batch_sum = selected_float.sum(dim=(0, 1)).double().cpu()
        batch_sum_squares = float(selected_float.square().sum().item())
        values = selected.to(torch.bfloat16).cpu()
        tokens = token_ids[:take].detach().cpu()
        sources = source_ids[:take].detach().to(torch.int16).cpu()
        self.buffer_x.append(values)
        self.buffer_tokens.append(tokens)
        self.buffer_sources.append(sources)
        self.buffer_count += take
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.sum_squares += batch_sum_squares
        self.positions += take * values.shape[1]
        self.domain_counts.update(
            source_names[index] for index in sources.tolist()
        )
        while self.buffer_count >= self.shard_windows:
            self._flush(self.shard_windows)
        return take

    def _flush(self, count: int) -> None:
        activations = torch.cat(self.buffer_x)
        token_ids = torch.cat(self.buffer_tokens)
        source_ids = torch.cat(self.buffer_sources)
        output_x, activations = activations[:count], activations[count:]
        output_tokens, token_ids = token_ids[:count], token_ids[count:]
        output_sources, source_ids = source_ids[:count], source_ids[count:]
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
        self.buffer_count = len(activations)

    def finish(self) -> None:
        if self.buffer_count:
            self._flush(self.buffer_count)
        if self.written != self.target_windows:
            raise RuntimeError(
                f"{self.split}: expected {self.target_windows} windows, "
                f"wrote {self.written}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "windows": self.written,
            "positions": self.positions,
            "shards": self.shards,
            "domain_counts": dict(sorted(self.domain_counts.items())),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream The Pile and write sharded frozen-LLM residual windows"
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
        "--train-windows",
        type=int,
        help=(
            "Explicit train window count. By default this is derived from "
            "--train-positions so storage does not grow with window size."
        ),
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        help=(
            "Explicit validation window count. By default this is derived "
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
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=320)
    parser.add_argument("--max-document-tokens", type=int, default=16384)
    parser.add_argument(
        "--shard-windows",
        type=int,
        help=(
            "Explicit windows per shard. By default this is derived from "
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
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if args.sequence_length % args.window_size:
        raise ValueError("--sequence-length must be divisible by --window-size")
    if min(
        args.train_positions,
        args.validation_positions,
        args.shard_positions,
    ) < 1:
        raise ValueError("position budgets must be positive")
    explicit_train_windows = args.train_windows
    explicit_validation_windows = args.validation_windows
    explicit_shard_windows = args.shard_windows
    args.train_windows = resolve_window_count(
        args.train_windows,
        args.train_positions,
        args.window_size,
    )
    args.validation_windows = resolve_window_count(
        args.validation_windows,
        args.validation_positions,
        args.window_size,
    )
    args.shard_windows = resolve_window_count(
        args.shard_windows,
        args.shard_positions,
        args.window_size,
    )
    if min(
        args.train_windows,
        args.validation_windows,
        args.shard_windows,
        args.batch_size,
    ) < 1:
        raise ValueError("window counts, shard size, and batch size must be positive")
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
        args.train_windows + args.validation_windows,
        args.window_size,
        d_in,
    )
    print(
        "resolved extraction budget: "
        f"window_size={args.window_size}, "
        f"train={args.train_windows:,} windows, "
        f"validation={args.validation_windows:,} windows, "
        f"shard={args.shard_windows:,} windows"
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
            args.shard_windows,
            args.train_windows,
        ),
        "validation": ShardWriter(
            output_dir,
            "validation",
            args.shard_windows,
            args.validation_windows,
        ),
    }
    source_to_id: dict[str, int] = {}
    sequence_ids: list[torch.Tensor] = []
    sequence_splits: list[str] = []
    sequence_sources: list[int] = []
    sequence_window_counts: list[int] = []
    documents = 0
    discarded_short_documents = 0
    target_total = args.train_windows + args.validation_windows
    progress = tqdm(total=target_total, unit="window", desc="Pile residuals")

    def run_batch() -> None:
        if not sequence_ids:
            return
        model_device = input_device(model)
        valid_lengths = torch.tensor(
            sequence_window_counts,
            device=model_device,
        ) * args.window_size
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
        windows_per_sequence = args.sequence_length // args.window_size
        hidden_windows = hidden.reshape(
            len(sequence_ids),
            windows_per_sequence,
            args.window_size,
            hidden.shape[-1],
        )
        token_windows = encoded["input_ids"].reshape(
            len(sequence_ids),
            windows_per_sequence,
            args.window_size,
        )
        for index, split_name in enumerate(sequence_splits):
            writer = writers[split_name]
            valid_windows = sequence_window_counts[index]
            sources = torch.full(
                (valid_windows,),
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
                hidden_windows[index, :valid_windows],
                token_windows[index, :valid_windows],
                sources,
                names,
            )
            progress.update(accepted)
        sequence_ids.clear()
        sequence_splits.clear()
        sequence_sources.clear()
        sequence_window_counts.clear()

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
            usable = len(ids) // args.window_size * args.window_size
            if not usable:
                discarded_short_documents += 1
                continue
            source = pile_set_name(row)
            if source not in source_to_id:
                source_to_id[source] = len(source_to_id)
            source_id = source_to_id[source]
            for start in range(0, usable, args.sequence_length):
                if writers[split_name].remaining <= 0:
                    break
                chunk = ids[
                    start : min(start + args.sequence_length, usable)
                ]
                valid_windows = len(chunk) // args.window_size
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
                sequence_window_counts.append(valid_windows)
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
        "format": "shared-residual-activation-shards-v1",
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
        "window_size": args.window_size,
        "sequence_length": args.sequence_length,
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
            "train_windows_explicit": explicit_train_windows,
            "validation_windows_explicit": explicit_validation_windows,
            "shard_windows_explicit": explicit_shard_windows,
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
        f"wrote {args.train_windows:,} train and "
        f"{args.validation_windows:,} validation windows to {output_dir}"
    )


if __name__ == "__main__":
    main()
