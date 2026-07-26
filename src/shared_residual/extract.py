from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .io import read_jsonl
from .modeling import capture_residual, get_layer, input_device, load_hf_model


def window_starts(length: int, width: int, mode: str, stride: int) -> list[int]:
    if length < width:
        return []
    if mode == "last":
        return [length - width]
    if mode == "sliding":
        starts = list(range(0, length - width + 1, stride))
        if starts[-1] != length - width:
            starts.append(length - width)
        return starts
    raise ValueError(f"unknown window mode: {mode}")


def _record_range(record: dict[str, Any], valid_length: int) -> tuple[int, int]:
    start = int(record.get("start_token", 0))
    end = int(record.get("end_token", valid_length))
    if start < 0:
        start += valid_length
    if end < 0:
        end += valid_length
    start = max(0, min(valid_length, start))
    end = max(start, min(valid_length, end))
    return start, end


def extract_hf(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("input JSONL is empty")
    for i, row in enumerate(rows):
        if args.text_key not in row:
            raise KeyError(f"row {i} has no {args.text_key!r}")

    model, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device_map,
        args.trust_remote_code,
        args.revision,
    )
    layer_path, layer_module = get_layer(model, args.layer)
    activations: list[torch.Tensor] = []
    token_ids_out: list[torch.Tensor] = []
    position_ids_out: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for batch_start in tqdm(range(0, len(rows), args.batch_size), desc="extract"):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        encoded = tokenizer(
            [str(r[args.text_key]) for r in batch_rows],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(input_device(model)) for k, v in encoded.items()}
        with torch.inference_mode(), capture_residual(
            layer_module, args.hook_point
        ) as captured:
            model(**encoded, use_cache=False)
        hidden = captured["activation"]

        for local_i, row in enumerate(batch_rows):
            valid_length = int(encoded["attention_mask"][local_i].sum().item())
            range_start, range_end = _record_range(row, valid_length)
            span_length = range_end - range_start
            starts = window_starts(
                span_length, args.window_size, args.window_mode, args.stride
            )
            for relative_start in starts:
                start = range_start + relative_start
                end = start + args.window_size
                ids = encoded["input_ids"][local_i, start:end].detach().cpu()
                activations.append(
                    hidden[local_i, start:end].detach().to(torch.float32).cpu()
                )
                token_ids_out.append(ids)
                position_ids_out.append(torch.arange(start, end, dtype=torch.long))
                kept_meta = {k: v for k, v in row.items() if k != args.text_key}
                kept_meta.update(
                    {
                        "record_index": batch_start + local_i,
                        "window_start": start,
                        "window_end": end,
                        "tokens": tokenizer.convert_ids_to_tokens(ids.tolist()),
                        "text_preview": str(row[args.text_key])[: args.preview_chars],
                    }
                )
                metadata.append(kept_meta)

    if not activations:
        raise ValueError(
            "No windows extracted. Check window_size, max_length, and token ranges."
        )
    return {
        "activations": torch.stack(activations),
        "token_ids": torch.stack(token_ids_out),
        "position_ids": torch.stack(position_ids_out),
        "metadata": metadata,
        "config": {
            "backend": "hf",
            "model": args.model,
            "layer": args.layer,
            "layer_path": layer_path,
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
        },
    }


def extract_transformer_lens(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[sae]'`") from exc

    rows = read_jsonl(args.data)
    model = HookedTransformer.from_pretrained(
        args.model,
        device=args.tl_device,
        dtype=getattr(torch, args.dtype),
        default_prepend_bos=False,
    )
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    activations: list[torch.Tensor] = []
    token_ids_out: list[torch.Tensor] = []
    position_ids_out: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for batch_start in tqdm(range(0, len(rows), args.batch_size), desc="extract"):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        encoded = tokenizer(
            [str(r[args.text_key]) for r in batch_rows],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(args.tl_device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name == args.hook_name
            )
        if args.hook_name not in cache:
            raise KeyError(f"hook {args.hook_name!r} was not captured")
        hidden = cache[args.hook_name]
        for local_i, row in enumerate(batch_rows):
            valid_length = int(encoded["attention_mask"][local_i].sum().item())
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
                ids = tokens[local_i, start:end].detach().cpu()
                activations.append(
                    hidden[local_i, start:end].detach().to(torch.float32).cpu()
                )
                token_ids_out.append(ids)
                position_ids_out.append(torch.arange(start, end))
                kept_meta = {k: v for k, v in row.items() if k != args.text_key}
                kept_meta.update(
                    {
                        "record_index": batch_start + local_i,
                        "window_start": start,
                        "window_end": end,
                        "tokens": tokenizer.convert_ids_to_tokens(ids.tolist()),
                        "text_preview": str(row[args.text_key])[: args.preview_chars],
                    }
                )
                metadata.append(kept_meta)
    if not activations:
        raise ValueError("No windows extracted")
    return {
        "activations": torch.stack(activations),
        "token_ids": torch.stack(token_ids_out),
        "position_ids": torch.stack(position_ids_out),
        "metadata": metadata,
        "config": {
            "backend": "transformer_lens",
            "model": args.model,
            "hook_name": args.hook_name,
            "window_size": args.window_size,
            "window_mode": args.window_mode,
            "stride": args.stride,
            "dtype": args.dtype,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL with at least a text field")
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["hf", "transformer_lens"], default="hf")
    p.add_argument("--text-key", default="text")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--hook-point", choices=["pre", "post"], default="post")
    p.add_argument("--hook-name", help="TransformerLens hook, e.g. blocks.12.hook_resid_post")
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--window-mode", choices=["last", "sliding"], default="last")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--preview-chars", type=int, default=160)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--revision")
    p.add_argument("--tl-device", default="cuda")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "transformer_lens" and not args.hook_name:
        raise SystemExit("--hook-name is required for transformer_lens")
    result = extract_hf(args) if args.backend == "hf" else extract_transformer_lens(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"saved {result['activations'].shape} to {out}")


if __name__ == "__main__":
    main()
