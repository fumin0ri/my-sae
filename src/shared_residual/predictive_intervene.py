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
from .predictive_sae import (
    PredictiveSAEConfig,
    PredictiveSparseAutoencoder,
    make_span_spec,
)


def parse_feature_ids(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))


def load_psae(path: str) -> tuple[PredictiveSparseAutoencoder, dict[str, Any]]:
    checkpoint = torch_load(path)
    model = PredictiveSparseAutoencoder(
        PredictiveSAEConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def restrict_features(
    codes: torch.Tensor,
    feature_ids: tuple[int, ...],
) -> torch.Tensor:
    if not feature_ids:
        return codes
    if min(feature_ids) < 0 or max(feature_ids) >= codes.shape[-1]:
        raise ValueError("a requested feature id is outside the SAE dictionary")
    mask = torch.zeros(codes.shape[-1], device=codes.device, dtype=codes.dtype)
    mask[list(feature_ids)] = 1
    return codes * mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causally patch predictable sparse features in the frozen LLM"
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
    parser.add_argument("--target-size", type=int, default=4)
    parser.add_argument("--gap", type=int, default=4)
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
        help="Use BF16 to keep a large SAE resident beside the LLM on a 24GB GPU",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.pairs)
    psae, checkpoint = load_psae(args.checkpoint)
    cfg = psae.cfg
    if cfg.context_mode != "causal":
        raise ValueError(
            "causal intervention requires a checkpoint trained with causal context"
        )
    source_cfg = checkpoint.get("source_config", {})
    width = int(source_cfg.get("window_size", cfg.max_window_size))
    span = make_span_spec(
        window_size=width,
        context_width=cfg.context_width,
        target_size=args.target_size,
        gap=args.gap,
        context_mode="causal",
        target_start=width - args.target_size,
    )
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
    for row_index, row in enumerate(tqdm(rows, desc=f"predictive {args.mode}")):
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
            0,
            target_window_start : len(target_prefix_ids),
        ].to(dtype=sae_dtype)[None, :, :]
        source_window = source_hidden[0, -width:].to(
            dtype=sae_dtype
        )[None, :, :]
        with torch.inference_mode():
            source_codes, _ = psae.predict_codes(source_window, span)
            target_codes, _ = psae.predict_codes(target_window, span)
            source_codes = restrict_features(source_codes, args.feature_ids)
            target_codes = restrict_features(target_codes, args.feature_ids)
            if args.mode == "patch":
                code_delta = source_codes - target_codes
            else:
                code_delta = -target_codes
            learned_delta = psae.decode(code_delta, add_bias=False)[0]
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
                delta = random * learned_delta.norm(
                    dim=-1,
                    keepdim=True,
                )
            delta = args.alpha * delta.to(target_hidden.dtype)
        edit_start = target_window_start + span.target_indices[0]
        edit_end = target_window_start + span.target_indices[-1] + 1

        def apply_edit(hidden: torch.Tensor) -> torch.Tensor:
            edited = hidden.clone()
            edited[:, edit_start:edit_end, :] += delta
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
                "target_size": args.target_size,
                "gap": args.gap,
                "feature_ids": list(args.feature_ids),
                "baseline_answer_logprob": baseline_lp,
                "edited_answer_logprob": edited_lp,
                "delta_answer_logprob": edited_lp - baseline_lp,
                "first_token_kl_baseline_to_edited": float(kl.item()),
                "intervention_l2": float(delta.float().norm().item()),
                "learned_intervention_l2": float(
                    (args.alpha * learned_delta).float().norm().item()
                ),
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
        len(results),
        1,
    )
    print(
        f"saved {len(results)} interventions; "
        f"mean delta answer logprob={mean_delta:.4f}"
    )


if __name__ == "__main__":
    main()
