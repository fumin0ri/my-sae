from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .io import read_jsonl, torch_load, write_jsonl
from .modeling import (
    capture_residual,
    edit_residual,
    get_layer,
    input_device,
    load_hf_model,
)


def tokenize_text(tokenizer: Any, text: str, add_special_tokens: bool) -> torch.Tensor:
    ids = tokenizer(
        text, add_special_tokens=add_special_tokens, return_tensors="pt"
    )["input_ids"][0]
    if ids.numel() == 0:
        raise ValueError("text tokenized to an empty sequence")
    return ids


def answer_logprob(logits: torch.Tensor, ids: torch.Tensor, prefix_len: int) -> float:
    log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = ids[1:]
    positions = torch.arange(prefix_len - 1, len(ids) - 1, device=ids.device)
    return float(log_probs[positions, targets[positions]].sum().item())


def capture(model: Any, layer: Any, hook_point: str, ids: torch.Tensor) -> tuple[Any, torch.Tensor]:
    with torch.inference_mode(), capture_residual(layer, hook_point) as state:
        output = model(input_ids=ids[None, :], use_cache=False)
    return output, state["activation"].detach()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal patching of a shared residual subspace")
    p.add_argument("--model", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--subspace", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--hook-point", choices=["pre", "post"], default="post")
    p.add_argument("--mode", choices=["patch", "ablate", "random_ablate"], default="patch")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--source-key", default="source_text")
    p.add_argument("--target-key", default="target_text")
    p.add_argument("--answer-key", default="answer")
    p.add_argument(
        "--contrast-answer-key",
        default="source_answer",
        help="Optional alternative answer used to measure directional patching",
    )
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--revision")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow model/layer/hook metadata to differ from the fitted subspace",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.pairs)
    state = torch_load(args.subspace)
    basis_cpu = state["basis"].float()
    position_effect_cpu = state["relative_position_effect"].float()
    mean_cpu = state["mean"].float()
    width = position_effect_cpu.shape[0]
    source_cfg = state.get("source_config", {})
    mismatches: list[str] = []
    for key, requested in {
        "model": args.model,
        "hook_point": args.hook_point,
    }.items():
        fitted = source_cfg.get(key)
        if fitted is not None and fitted != requested:
            mismatches.append(f"{key}: fitted={fitted!r}, requested={requested!r}")
    model, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device_map,
        args.trust_remote_code,
        args.revision,
    )
    fitted_revision = source_cfg.get("resolved_model_revision")
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if (
        fitted_revision is not None
        and loaded_revision is not None
        and fitted_revision != loaded_revision
    ):
        mismatches.append(
            "model revision: "
            f"fitted={fitted_revision!r}, loaded={loaded_revision!r}"
        )
    layer_path, layer = get_layer(model, args.layer)
    fitted_layer_path = source_cfg.get("layer_path")
    if fitted_layer_path is not None:
        if fitted_layer_path != layer_path:
            mismatches.append(
                f"layer_path: fitted={fitted_layer_path!r}, requested={layer_path!r}"
            )
    elif source_cfg.get("layer") is not None and source_cfg["layer"] != args.layer:
        mismatches.append(
            f"layer: fitted={source_cfg['layer']!r}, requested={args.layer!r}"
        )
    if mismatches and not args.allow_mismatch:
        raise ValueError(
            "Subspace/intervention mismatch:\n  "
            + "\n  ".join(mismatches)
            + "\nPass --allow-mismatch only if this is intentional."
        )
    device = input_device(model)
    learned_basis_cpu = basis_cpu
    if args.mode == "random_ablate":
        generator = torch.Generator().manual_seed(args.seed)
        random = torch.randn(
            basis_cpu.shape, generator=generator, dtype=basis_cpu.dtype
        )
        random = random - basis_cpu @ (basis_cpu.T @ random)
        basis_cpu, _ = torch.linalg.qr(random, mode="reduced")

    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(tqdm(rows, desc="intervene")):
        source_ids = tokenize_text(
            tokenizer, str(row[args.source_key]), add_special_tokens=True
        ).to(device)
        target_prefix_ids = tokenize_text(
            tokenizer, str(row[args.target_key]), add_special_tokens=True
        ).to(device)
        answer_ids = tokenize_text(
            tokenizer, str(row[args.answer_key]), add_special_tokens=False
        ).to(device)
        if len(source_ids) < width or len(target_prefix_ids) < width:
            raise ValueError(f"row {row_index}: source/target shorter than window width {width}")
        target_ids = torch.cat([target_prefix_ids, answer_ids])
        baseline_output, target_hidden = capture(
            model, layer, args.hook_point, target_ids
        )
        _, source_hidden = capture(model, layer, args.hook_point, source_ids)
        basis = basis_cpu.to(target_hidden.device, target_hidden.dtype)
        mean = mean_cpu.to(target_hidden.device, target_hidden.dtype)
        position_effect = position_effect_cpu.to(
            target_hidden.device, target_hidden.dtype
        )
        source_window = source_hidden[0, -width:]
        start = len(target_prefix_ids) - width
        target_window = target_hidden[0, start : len(target_prefix_ids)]
        source_common = (
            source_window - mean - position_effect
        ).mean(dim=0)
        target_common = (
            target_window - mean - position_effect
        ).mean(dim=0)
        if args.mode == "patch":
            raw_delta = source_common - target_common
        else:
            raw_delta = -target_common
        delta = args.alpha * basis @ (basis.T @ raw_delta)
        if args.mode == "random_ablate":
            learned_basis = learned_basis_cpu.to(
                target_hidden.device, target_hidden.dtype
            )
            matched_norm = (
                args.alpha
                * (learned_basis @ (learned_basis.T @ raw_delta)).float().norm()
            )
            delta = delta * (
                matched_norm / delta.float().norm().clamp_min(1e-12)
            ).to(delta.dtype)

        def apply_edit(hidden: torch.Tensor) -> torch.Tensor:
            edited = hidden.clone()
            edited[:, start : len(target_prefix_ids), :] += delta
            return edited

        with torch.inference_mode(), edit_residual(
            layer, args.hook_point, apply_edit
        ):
            edited_output = model(input_ids=target_ids[None, :], use_cache=False)
        baseline_lp = answer_logprob(
            baseline_output.logits, target_ids, len(target_prefix_ids)
        )
        edited_lp = answer_logprob(
            edited_output.logits, target_ids, len(target_prefix_ids)
        )
        contrast_metrics: dict[str, float] = {}
        if args.contrast_answer_key in row:
            contrast_answer_ids = tokenize_text(
                tokenizer,
                str(row[args.contrast_answer_key]),
                add_special_tokens=False,
            ).to(device)
            contrast_ids = torch.cat(
                [target_prefix_ids, contrast_answer_ids]
            )
            with torch.inference_mode():
                contrast_baseline_output = model(
                    input_ids=contrast_ids[None, :],
                    use_cache=False,
                )
            with torch.inference_mode(), edit_residual(
                layer,
                args.hook_point,
                apply_edit,
            ):
                contrast_edited_output = model(
                    input_ids=contrast_ids[None, :],
                    use_cache=False,
                )
            contrast_baseline_lp = answer_logprob(
                contrast_baseline_output.logits,
                contrast_ids,
                len(target_prefix_ids),
            )
            contrast_edited_lp = answer_logprob(
                contrast_edited_output.logits,
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
                    (contrast_edited_lp - contrast_baseline_lp)
                    - (edited_lp - baseline_lp)
                ),
            }
        first_answer_position = len(target_prefix_ids) - 1
        baseline_dist = F.log_softmax(
            baseline_output.logits[0, first_answer_position].float(), dim=-1
        )
        edited_dist = F.log_softmax(
            edited_output.logits[0, first_answer_position].float(), dim=-1
        )
        kl = torch.sum(
            baseline_dist.exp() * (baseline_dist - edited_dist)
        )
        results.append(
            {
                "row_index": row_index,
                "mode": args.mode,
                "alpha": args.alpha,
                "baseline_answer_logprob": baseline_lp,
                "edited_answer_logprob": edited_lp,
                "delta_answer_logprob": edited_lp - baseline_lp,
                "first_token_kl_baseline_to_edited": float(kl.item()),
                "intervention_l2": float(delta.float().norm().item()),
                **contrast_metrics,
                "metadata": {
                    k: v
                    for k, v in row.items()
                    if k not in {args.source_key, args.target_key}
                },
            }
        )
    write_jsonl(args.output, results)
    mean_delta = sum(r["delta_answer_logprob"] for r in results) / max(1, len(results))
    print(f"saved {len(results)} rows; mean Δ answer logprob = {mean_delta:.4f}")


if __name__ == "__main__":
    main()
