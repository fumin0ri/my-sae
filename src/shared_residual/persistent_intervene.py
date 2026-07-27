from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .intervene import answer_logprob, capture, tokenize_text
from .io import read_jsonl, torch_load, write_jsonl
from .modeling import (
    edit_residual,
    get_layer,
    input_device,
    load_hf_model,
    parse_dtype,
)
from .persistent_sae import PersistentSAEConfig, PersistentSparseAutoencoder
from .predictive_intervene import parse_feature_ids, restrict_features


def load_psae(
    path: str,
) -> tuple[PersistentSparseAutoencoder, dict[str, Any]]:
    checkpoint = torch_load(path)
    model = PersistentSparseAutoencoder(
        PersistentSAEConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(
        dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip())
    )
    if not offsets:
        raise argparse.ArgumentTypeError("expected comma-separated offsets")
    return offsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch or ablate a z0 persistent code at future window positions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-point", choices=["pre", "post"], default="post")
    parser.add_argument(
        "--mode",
        choices=["patch", "ablate", "random_ablate"],
        default="patch",
    )
    parser.add_argument("--offsets", type=parse_offsets, default=tuple(range(1, 10)))
    parser.add_argument("--feature-ids", type=parse_feature_ids, default=())
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--source-key", default="source_text")
    parser.add_argument("--target-key", default="target_text")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--contrast-answer-key", default="source_answer")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--sae-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.pairs)
    psae, checkpoint = load_psae(args.checkpoint)
    width = psae.cfg.window_size
    if min(args.offsets) < 1 or max(args.offsets) >= width:
        raise ValueError(f"--offsets must lie in [1, {width - 1}]")
    source_cfg = checkpoint.get("source_config", {})
    mismatches = []
    for key, requested in {
        "model": args.model,
        "hook_point": args.hook_point,
        "layer": args.layer,
    }.items():
        fitted = source_cfg.get(key)
        if fitted is not None and fitted != requested:
            mismatches.append(f"{key}: fitted={fitted!r}, requested={requested!r}")
    if mismatches and not args.allow_mismatch:
        raise ValueError(
            "Checkpoint/intervention mismatch:\n  "
            + "\n  ".join(mismatches)
            + "\nPass --allow-mismatch only for a deliberate cross-model experiment."
        )

    llm, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device_map,
        args.trust_remote_code,
        args.revision,
        use_safetensors=True if args.use_safetensors else None,
    )
    fitted_revision = source_cfg.get("resolved_model_revision")
    loaded_revision = getattr(llm.config, "_commit_hash", None)
    if (
        fitted_revision is not None
        and loaded_revision is not None
        and fitted_revision != loaded_revision
        and not args.allow_mismatch
    ):
        raise ValueError(
            "Checkpoint/intervention model revision mismatch: "
            f"fitted={fitted_revision!r}, loaded={loaded_revision!r}"
        )
    layer_path, layer = get_layer(llm, args.layer)
    fitted_layer_path = source_cfg.get("layer_path")
    if fitted_layer_path is not None and fitted_layer_path != layer_path:
        raise ValueError(
            f"checkpoint layer path {fitted_layer_path!r} != loaded {layer_path!r}"
        )
    token_device = input_device(llm)
    sae_dtype = parse_dtype(args.sae_dtype)
    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(tqdm(rows, desc=f"persistent {args.mode}")):
        source_ids = tokenize_text(
            tokenizer,
            str(row[args.source_key]),
            add_special_tokens=True,
        ).to(token_device)
        target_prefix_ids = tokenize_text(
            tokenizer,
            str(row[args.target_key]),
            add_special_tokens=True,
        ).to(token_device)
        answer_ids = tokenize_text(
            tokenizer,
            str(row[args.answer_key]),
            add_special_tokens=False,
        ).to(token_device)
        if min(len(source_ids), len(target_prefix_ids)) < width:
            raise ValueError(
                f"row {row_index}: prefix is shorter than checkpoint window {width}"
            )
        target_ids = torch.cat([target_prefix_ids, answer_ids])
        baseline_output, target_hidden = capture(
            llm,
            layer,
            args.hook_point,
            target_ids,
        )
        _, source_hidden = capture(
            llm,
            layer,
            args.hook_point,
            source_ids,
        )
        hidden_device = target_hidden.device
        psae.to(device=hidden_device, dtype=sae_dtype)
        target_window_start = len(target_prefix_ids) - width
        target_window = target_hidden[
            0, target_window_start : len(target_prefix_ids)
        ].to(dtype=sae_dtype)
        source_window = source_hidden[0, -width:].to(dtype=sae_dtype)
        with torch.inference_mode():
            source_z0 = restrict_features(
                psae.encode(source_window[0:1]),
                args.feature_ids,
            )
            target_z0 = restrict_features(
                psae.encode(target_window[0:1]),
                args.feature_ids,
            )
            code_delta = (
                source_z0 - target_z0 if args.mode == "patch" else -target_z0
            )
            learned_vector = psae.decode(code_delta, add_bias=False)[0]
            learned_delta = learned_vector[None, :].expand(len(args.offsets), -1)
            delta = learned_delta
            if args.mode == "random_ablate":
                generator = torch.Generator(device="cpu").manual_seed(
                    args.seed + row_index
                )
                random = torch.randn(
                    learned_delta.shape,
                    generator=generator,
                    dtype=torch.float32,
                ).to(hidden_device)
                random = random / random.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                delta = random * learned_delta.float().norm(
                    dim=-1, keepdim=True
                )
            delta = (args.alpha * delta).to(target_hidden.dtype)
        edit_positions = [
            target_window_start + offset for offset in args.offsets
        ]

        def apply_edit(hidden: torch.Tensor) -> torch.Tensor:
            edited = hidden.clone()
            edited[:, edit_positions, :] += delta
            return edited

        with torch.inference_mode(), edit_residual(
            layer,
            args.hook_point,
            apply_edit,
        ):
            edited_output = llm(input_ids=target_ids[None, :], use_cache=False)
        baseline_lp = answer_logprob(
            baseline_output.logits,
            target_ids,
            len(target_prefix_ids),
        )
        edited_lp = answer_logprob(
            edited_output.logits,
            target_ids,
            len(target_prefix_ids),
        )
        contrast_metrics: dict[str, float] = {}
        if args.contrast_answer_key in row:
            contrast_answer_ids = tokenize_text(
                tokenizer,
                str(row[args.contrast_answer_key]),
                add_special_tokens=False,
            ).to(token_device)
            contrast_ids = torch.cat([target_prefix_ids, contrast_answer_ids])
            with torch.inference_mode():
                contrast_baseline = llm(
                    input_ids=contrast_ids[None, :],
                    use_cache=False,
                )
            with torch.inference_mode(), edit_residual(
                layer,
                args.hook_point,
                apply_edit,
            ):
                contrast_edited = llm(
                    input_ids=contrast_ids[None, :],
                    use_cache=False,
                )
            contrast_baseline_lp = answer_logprob(
                contrast_baseline.logits,
                contrast_ids,
                len(target_prefix_ids),
            )
            contrast_edited_lp = answer_logprob(
                contrast_edited.logits,
                contrast_ids,
                len(target_prefix_ids),
            )
            contrast_metrics = {
                "baseline_contrast_answer_logprob": contrast_baseline_lp,
                "edited_contrast_answer_logprob": contrast_edited_lp,
                "delta_contrast_answer_logprob": (
                    contrast_edited_lp - contrast_baseline_lp
                ),
                "delta_contrast_minus_target_logprob": (
                    contrast_edited_lp
                    - contrast_baseline_lp
                    - edited_lp
                    + baseline_lp
                ),
            }
        answer_position = len(target_prefix_ids) - 1
        baseline_distribution = F.log_softmax(
            baseline_output.logits[0, answer_position].float(),
            dim=-1,
        )
        edited_distribution = F.log_softmax(
            edited_output.logits[0, answer_position].float(),
            dim=-1,
        )
        kl = torch.sum(
            baseline_distribution.exp()
            * (baseline_distribution - edited_distribution)
        )
        results.append(
            {
                "row_index": row_index,
                "mode": args.mode,
                "alpha": args.alpha,
                "offsets": list(args.offsets),
                "feature_ids": list(args.feature_ids),
                "baseline_answer_logprob": baseline_lp,
                "edited_answer_logprob": edited_lp,
                "delta_answer_logprob": edited_lp - baseline_lp,
                "first_token_kl_baseline_to_edited": float(kl.item()),
                "intervention_l2_per_offset": [
                    float(value)
                    for value in delta.float().norm(dim=-1).tolist()
                ],
                **contrast_metrics,
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {args.source_key, args.target_key}
                },
            }
        )
    write_jsonl(args.output, results)
    mean_delta = sum(row["delta_answer_logprob"] for row in results) / max(
        len(results), 1
    )
    print(
        f"saved {len(results)} interventions; "
        f"mean delta answer logprob={mean_delta:.4f}"
    )


if __name__ == "__main__":
    main()
