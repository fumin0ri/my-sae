from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .extract import _record_range, window_starts
from .io import read_jsonl
from .modeling import (
    get_layer,
    input_device,
    load_hf_model,
    tensor_from_layer_output,
)


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return list(dict.fromkeys(parsed))


def register_multi_hooks(
    model: Any,
    layers: list[int],
    hook_point: str,
) -> tuple[dict[int, str], dict[int, torch.Tensor], list[Any]]:
    paths: dict[int, str] = {}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layers:
        path, module = get_layer(model, layer_index)
        paths[layer_index] = path
        if hook_point == "pre":
            def pre_hook(
                _module: Any,
                args: tuple[Any, ...],
                index: int = layer_index,
            ) -> None:
                captured[index] = args[0].detach()

            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            def post_hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                captured[index] = tensor_from_layer_output(output).detach()

            handles.append(module.register_forward_hook(post_hook))
    return paths, captured, handles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the same token windows from many residual-stream layers"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=parse_int_list, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--hook-point", choices=["pre", "post"], default="post")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-mode", choices=["last", "sliding"], default="last")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--preview-chars", type=int, default=160)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("input JSONL is empty")
    for row_index, row in enumerate(rows):
        if args.text_key not in row:
            raise KeyError(f"row {row_index} has no {args.text_key!r}")

    model, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device_map,
        args.trust_remote_code,
        args.revision,
    )
    paths, captured, handles = register_multi_hooks(
        model,
        args.layers,
        args.hook_point,
    )
    activations: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in args.layers
    }
    token_ids_out: list[torch.Tensor] = []
    position_ids_out: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    try:
        for batch_start in tqdm(
            range(0, len(rows), args.batch_size),
            desc="multi-layer extract",
        ):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            encoded = tokenizer(
                [str(row[args.text_key]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(input_device(model))
                for key, value in encoded.items()
            }
            captured.clear()
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            missing_layers = set(args.layers) - set(captured)
            if missing_layers:
                raise RuntimeError(f"hooks did not capture layers {missing_layers}")

            for local_index, row in enumerate(batch_rows):
                valid_length = int(
                    encoded["attention_mask"][local_index].sum().item()
                )
                range_start, range_end = _record_range(row, valid_length)
                starts = window_starts(
                    range_end - range_start,
                    args.window_size,
                    args.window_mode,
                    args.stride,
                )
                for relative_start in starts:
                    start = range_start + relative_start
                    end = start + args.window_size
                    ids = (
                        encoded["input_ids"][local_index, start:end]
                        .detach()
                        .cpu()
                    )
                    for layer in args.layers:
                        activations[layer].append(
                            captured[layer][local_index, start:end]
                            .detach()
                            .to(torch.float32)
                            .cpu()
                        )
                    token_ids_out.append(ids)
                    position_ids_out.append(
                        torch.arange(start, end, dtype=torch.long)
                    )
                    kept_meta = {
                        key: value
                        for key, value in row.items()
                        if key != args.text_key
                    }
                    kept_meta.update(
                        {
                            "record_index": batch_start + local_index,
                            "window_start": start,
                            "window_end": end,
                            "tokens": tokenizer.convert_ids_to_tokens(ids.tolist()),
                            "text_preview": str(row[args.text_key])[
                                : args.preview_chars
                            ],
                        }
                    )
                    metadata.append(kept_meta)
    finally:
        for handle in handles:
            handle.remove()

    if not metadata:
        raise ValueError("No windows extracted")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "token_ids": torch.stack(token_ids_out),
        "position_ids": torch.stack(position_ids_out),
        "metadata": metadata,
    }
    for layer in args.layers:
        bundle = {
            **common,
            "activations": torch.stack(activations[layer]),
            "config": {
                "backend": "hf",
                "model": args.model,
                "layer": layer,
                "layer_path": paths[layer],
                "hook_point": args.hook_point,
                "window_size": args.window_size,
                "window_mode": args.window_mode,
                "stride": args.stride,
                "dtype": args.dtype,
                "requested_revision": args.revision,
                "resolved_model_revision": getattr(
                    model.config,
                    "_commit_hash",
                    None,
                ),
                "resolved_tokenizer_revision": getattr(
                    tokenizer,
                    "init_kwargs",
                    {},
                ).get("_commit_hash"),
                "multi_layer_extraction": True,
            },
        }
        output = output_dir / f"layer-{layer:03d}.pt"
        torch.save(bundle, output)
        print(f"saved {tuple(bundle['activations'].shape)} to {output}")


if __name__ == "__main__":
    main()
