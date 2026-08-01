from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .activation_store import (
    load_activation_manifest,
    manifest_fingerprint,
    validation_batches,
)
from .evaluation import (
    clustered_mean_ci,
    collapse_diagnostics,
    different_group_permutation,
    fit_probe,
    pca_embedding,
    select_probe_dimensions,
)
from .group_sae import topk_relu
from .io import torch_load, write_json
from .modeling import edit_residual, get_layer, input_device, load_hf_model
from .training import autocast_context, configure_accelerator, grouped_three_way_split
from .transition_jepa_sae import (
    ARCHITECTURE_ID,
    TransitionJEPAConfig,
    TransitionJEPASAE,
)


PROBE_LABELS = {
    "semantics": "semantic_answer",
    "context": "context_category",
    "syntax": "syntax_template",
}
HORIZON_METRICS = (
    "context_target_cosine",
    "code_cosine",
    "shuffled_context_cosine",
    "position_only_cosine",
    "code_nrmse",
    "support_precision",
    "support_recall",
    "support_jaccard",
    "residual_error",
    "residual_energy",
)


def load_model(
    path: str | Path, device: torch.device
) -> tuple[TransitionJEPASAE, dict[str, Any]]:
    checkpoint = torch_load(path)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(
            f"{path} is not a {ARCHITECTURE_ID} checkpoint; only the current "
            "high/low model is supported"
        )
    model = TransitionJEPASAE(TransitionJEPAConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_sae_quality(
    model: TransitionJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, float | int]:
    """Conventional reconstruction, sparsity, and dictionary-usage metrics."""
    totals: dict[str, float] = defaultdict(float)
    active_features = torch.zeros(model.cfg.d_sae, dtype=torch.float32)
    positions = 0
    windows = 0
    batches = 0
    for x in tqdm(
        validation_batches(root, manifest, batch_size, maximum_batches),
        desc="standard SAE metrics",
    ):
        x = x.to(device, dtype=model.pre_bias.dtype, non_blocking=True)
        with autocast_context(device, amp_dtype):
            code = model.encode_ema(x)
            high, low = model.split_code(code)
            reconstruction = model.decode_ema(code)
            high_reconstruction = model.decode_high(high, ema=True)
            low_reconstruction = model.ema_pre_bias + model.decode_low(
                low, ema=True, add_bias=False
            )
        x32 = x.float()
        reconstruction32 = reconstruction.float()
        high32 = high_reconstruction.float()
        low32 = low_reconstruction.float()
        centered_energy = (
            x32 - model.ema_pre_bias.float()
        ).square().sum()
        totals["centered_energy"] += float(centered_energy)
        totals["full_squared_error"] += float(
            (x32 - reconstruction32).square().sum()
        )
        totals["high_squared_error"] += float((x32 - high32).square().sum())
        totals["low_squared_error"] += float((x32 - low32).square().sum())
        totals["l2_loss_sum"] += float(
            torch.linalg.vector_norm(x32 - reconstruction32, dim=-1).sum()
        )
        totals["cosine_sum"] += float(
            F.cosine_similarity(x32, reconstruction32, dim=-1).sum()
        )
        totals["l1_sum"] += float(code.float().abs().sum())
        totals["l0_sum"] += float((code != 0).float().sum())
        totals["high_l0_sum"] += float((high != 0).float().sum())
        totals["low_l0_sum"] += float((low != 0).float().sum())
        active_features += code.reshape(-1, model.cfg.d_sae).float().sum(dim=0).cpu()
        positions += x.shape[0] * x.shape[1]
        windows += x.shape[0]
        batches += 1
    if positions == 0:
        raise ValueError("no Pile validation activations were evaluated")
    scale = max(totals["centered_energy"], 1e-12)
    full_fvu = totals["full_squared_error"] / scale
    high_fvu = totals["high_squared_error"] / scale
    low_fvu = totals["low_squared_error"] / scale
    alive = active_features != 0
    return {
        "l2_loss": totals["l2_loss_sum"] / positions,
        "l1": totals["l1_sum"] / positions,
        "l0": totals["l0_sum"] / positions,
        "high_l0": totals["high_l0_sum"] / positions,
        "low_l0": totals["low_l0_sum"] / positions,
        "reconstruction_cosine": totals["cosine_sum"] / positions,
        "reconstruction_fvu": full_fvu,
        "fraction_variance_explained": 1.0 - full_fvu,
        "high_only_fvu": high_fvu,
        "high_only_fraction_variance_explained": 1.0 - high_fvu,
        "low_only_fvu": low_fvu,
        "low_only_fraction_variance_explained": 1.0 - low_fvu,
        "alive_feature_fraction": float(alive.float().mean()),
        "dead_feature_fraction": float((~alive).float().mean()),
        "n_positions": positions,
        "n_windows": windows,
        "n_batches": batches,
    }


def causal_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = -100
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _text_rows(stream: Iterable[dict[str, Any]], key: str) -> Iterable[str]:
    for row in stream:
        text = str(row.get(key, ""))
        if text.strip():
            yield text


@torch.no_grad()
def evaluate_loss_recovered(
    sae: TransitionJEPASAE,
    source_config: dict[str, Any],
    model_name: str,
    revision: str | None,
    use_safetensors: bool,
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_key: str,
    n_inputs: int,
    context_length: int,
    hook_point: str,
    layer_index: int,
) -> dict[str, float | int]:
    from datasets import load_dataset

    dataset_args: dict[str, Any] = {"split": dataset_split, "streaming": True}
    stream = (
        load_dataset(dataset_name, dataset_config, **dataset_args)
        if dataset_config
        else load_dataset(dataset_name, **dataset_args)
    )
    texts = iter(_text_rows(stream, text_key))
    llm, tokenizer = load_hf_model(
        model_name,
        dtype,
        device_map,
        trust_remote_code,
        revision,
        use_safetensors=True if use_safetensors else None,
    )
    fitted_model = source_config.get("model")
    if fitted_model is not None and fitted_model != model_name:
        raise ValueError(
            f"checkpoint model {fitted_model!r} != evaluation model {model_name!r}"
        )
    layer_path, layer = get_layer(llm, layer_index)
    fitted_layer_path = source_config.get("layer_path")
    if fitted_layer_path is not None and fitted_layer_path != layer_path:
        raise ValueError(
            f"checkpoint layer {fitted_layer_path!r} != evaluation layer {layer_path!r}"
        )
    token_device = input_device(llm)
    sae.to(token_device, dtype=llm.dtype).eval()
    totals: dict[str, float] = defaultdict(float)
    used = 0
    attempts = 0
    for _ in tqdm(range(n_inputs), desc="loss recovered"):
        while True:
            text = next(texts)
            attempts += 1
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=context_length,
            )
            if encoded["input_ids"].shape[1] >= 2:
                break
        encoded = {key: value.to(token_device) for key, value in encoded.items()}
        attention_mask = encoded.get(
            "attention_mask", torch.ones_like(encoded["input_ids"])
        )
        original = llm(**encoded, use_cache=False).logits

        def reconstruct(hidden: torch.Tensor) -> torch.Tensor:
            return sae.decode_ema(
                sae.encode_ema(hidden.to(sae.pre_bias.dtype))
            ).to(hidden.dtype)

        with edit_residual(layer, hook_point, reconstruct):
            reconstructed = llm(**encoded, use_cache=False).logits
        with edit_residual(layer, hook_point, torch.zeros_like):
            zero = llm(**encoded, use_cache=False).logits
        original_loss = causal_lm_loss(
            original, encoded["input_ids"], attention_mask
        )
        reconstructed_loss = causal_lm_loss(
            reconstructed, encoded["input_ids"], attention_mask
        )
        zero_loss = causal_lm_loss(zero, encoded["input_ids"], attention_mask)
        denominator = original_loss - zero_loss
        recovered = (reconstructed_loss - zero_loss) / denominator.clamp_max(-1e-8)
        totals["loss_original"] += float(original_loss)
        totals["loss_reconstructed"] += float(reconstructed_loss)
        totals["loss_zero"] += float(zero_loss)
        totals["fraction_loss_recovered"] += float(recovered)
        used += 1
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        **{key: value / max(used, 1) for key, value in totals.items()},
        "n_inputs": used,
        "documents_examined": attempts,
        "context_length": context_length,
    }


def _allocate_representations(
    n: int, model: TransitionJEPASAE
) -> dict[str, torch.Tensor]:
    return {
        "context_high": torch.empty((n, model.cfg.d_high), dtype=torch.float16),
        "predicted_endpoint_high": torch.empty(
            (n, model.cfg.d_high), dtype=torch.float16
        ),
        "endpoint_high": torch.empty((n, model.cfg.d_high), dtype=torch.float16),
        "context_low": torch.empty((n, model.cfg.d_low), dtype=torch.float16),
        "endpoint_low": torch.empty((n, model.cfg.d_low), dtype=torch.float16),
        "endpoint_full": torch.empty((n, model.cfg.d_sae), dtype=torch.float16),
    }


@torch.no_grad()
def encode_mmlu_representations(
    model: TransitionJEPASAE,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    representations = _allocate_representations(len(x), model)
    position_zero = torch.zeros(1, dtype=torch.long, device=device)
    for start in tqdm(range(0, len(x), batch_size), desc="MMLU representations"):
        end = min(start + batch_size, len(x))
        batch = x[start:end].to(
            device=device, dtype=model.pre_bias.dtype, non_blocking=True
        )
        with autocast_context(device, amp_dtype):
            context_full = model.encode_ema(batch[:, 0])
            endpoint_full = model.encode_ema(batch[:, -1])
            context_high, context_low = model.split_code(context_full)
            endpoint_high, endpoint_low = model.split_code(endpoint_full)
            predicted = model.predict_from_code(
                context_high,
                context_positions=position_zero,
                use_context=True,
                sparse_output=True,
            )[:, 0]
        values = {
            "context_high": context_high,
            "predicted_endpoint_high": predicted,
            "endpoint_high": endpoint_high,
            "context_low": context_low,
            "endpoint_low": endpoint_low,
            "endpoint_full": endpoint_full,
        }
        for key, value in values.items():
            representations[key][start:end].copy_(value.float().cpu())
    return representations


@torch.no_grad()
def collect_horizon_statistics(
    model: TransitionJEPASAE,
    x: torch.Tensor,
    test_indices: list[int],
    groups: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    n_contexts = model.cfg.window_size - 1
    statistics = {
        name: torch.empty((len(test_indices), n_contexts), dtype=torch.float32)
        for name in HORIZON_METRICS
    }
    permutation = different_group_permutation(groups, seed)
    shuffled_indices = [test_indices[int(index)] for index in permutation]
    positions = torch.arange(n_contexts, dtype=torch.long, device=device)
    for start in tqdm(
        range(0, len(test_indices), batch_size), desc="locked forecast nulls"
    ):
        end = min(start + batch_size, len(test_indices))
        batch = x[test_indices[start:end]].to(
            device=device, dtype=model.pre_bias.dtype, non_blocking=True
        )
        shuffled_batch = x[shuffled_indices[start:end]].to(
            device=device, dtype=model.pre_bias.dtype, non_blocking=True
        )
        with autocast_context(device, amp_dtype):
            context_full = model.encode_ema(batch[:, :-1])
            context_high, _ = model.split_code(context_full)
            target_high, _ = model.split_code(model.encode_ema(batch[:, -1]))
            target = target_high[:, None].expand(-1, n_contexts, -1)
            prediction = model.predict_from_code(
                context_high, positions, use_context=True
            )
            sparse_prediction = topk_relu(prediction, model.cfg.k_high)
            position_only = model.predict_from_code(
                context_high, positions, use_context=False
            )
            shuffled_high, _ = model.split_code(
                model.encode_ema(shuffled_batch[:, :-1])
            )
            shuffled_prediction = model.predict_from_code(
                shuffled_high, positions, use_context=True
            )
            predicted_residual = model.decode_high(
                sparse_prediction, ema=True, add_bias=True
            )
        prediction32 = prediction.float()
        target32 = target.float()
        sparse32 = sparse_prediction.float()
        predicted_active = sparse32 > 0
        target_active = target32 > 0
        intersection = (predicted_active & target_active).sum(dim=-1).float()
        union = (predicted_active | target_active).sum(dim=-1).float().clamp_min(1)
        target_residual = batch[:, -1, None, :].float()
        values = {
            "context_target_cosine": F.cosine_similarity(
                context_high.float(), target32, dim=-1
            ),
            "code_cosine": F.cosine_similarity(prediction32, target32, dim=-1),
            "shuffled_context_cosine": F.cosine_similarity(
                shuffled_prediction.float(), target32, dim=-1
            ),
            "position_only_cosine": F.cosine_similarity(
                position_only.float(), target32, dim=-1
            ),
            "code_nrmse": (prediction32 - target32).square().mean(dim=-1)
            / target32.square().mean(dim=-1).clamp_min(1e-8),
            "support_precision": intersection
            / predicted_active.sum(dim=-1).float().clamp_min(1),
            "support_recall": intersection
            / target_active.sum(dim=-1).float().clamp_min(1),
            "support_jaccard": intersection / union,
            "residual_error": (
                predicted_residual.float() - target_residual
            ).square().mean(dim=-1),
            "residual_energy": (
                target_residual - model.ema_pre_bias.float()
            ).square().mean(dim=-1),
        }
        for key, value in values.items():
            statistics[key][start:end].copy_(value.cpu())
    return statistics


def build_horizon_curve(
    statistics: dict[str, torch.Tensor],
    groups: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    n_contexts = statistics["code_cosine"].shape[1]
    rows: list[dict[str, Any]] = []
    for context_position in range(n_contexts):
        learned = statistics["code_cosine"][:, context_position].numpy()
        shuffled = statistics["shuffled_context_cosine"][:, context_position].numpy()
        position_only = statistics["position_only_cosine"][:, context_position].numpy()
        residual_error = statistics["residual_error"][:, context_position]
        residual_energy = statistics["residual_energy"][:, context_position]
        rows.append(
            {
                "context_position": context_position,
                "horizon": n_contexts - context_position,
                "context_target_cosine": float(
                    statistics["context_target_cosine"][:, context_position].mean()
                ),
                "code_cosine": float(learned.mean()),
                "shuffled_context_cosine": float(shuffled.mean()),
                "position_only_cosine": float(position_only.mean()),
                "context_gain_over_shuffled": clustered_mean_ci(
                    learned - shuffled, groups, seed + 101 * context_position
                ),
                "context_gain_over_position_only": clustered_mean_ci(
                    learned - position_only,
                    groups,
                    seed + 10007 + 101 * context_position,
                ),
                "code_nrmse": float(
                    statistics["code_nrmse"][:, context_position].mean()
                ),
                "support_precision": float(
                    statistics["support_precision"][:, context_position].mean()
                ),
                "support_recall": float(
                    statistics["support_recall"][:, context_position].mean()
                ),
                "support_jaccard": float(
                    statistics["support_jaccard"][:, context_position].mean()
                ),
                "residual_prediction_fvu": float(
                    residual_error.mean() / residual_energy.mean().clamp_min(1e-8)
                ),
            }
        )
    return rows


def evaluate_probes(
    representations: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    development_indices: list[int],
    test_indices: list[int],
    group_key: str,
    probe_max_dim: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for axis_index, (axis, label_key) in enumerate(PROBE_LABELS.items()):
        axis_results: dict[str, Any] = {}
        for representation_index, (name, values) in enumerate(
            representations.items()
        ):
            selected = select_probe_dimensions(
                values, development_indices, probe_max_dim
            )
            result, _ = fit_probe(
                selected,
                metadata,
                development_indices,
                test_indices,
                label_key,
                group_key,
                seed + 1000 * axis_index + representation_index,
            )
            result["input_dimension"] = values.shape[1]
            result["probe_dimension"] = selected.shape[1]
            axis_results[name] = result
        results[axis] = axis_results
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate conventional SAE quality and whether high-code endpoint "
            "forecasting beats shuffled and position-only controls"
        )
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mmlu-model-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-key", default="question_id")
    parser.add_argument("--probe-max-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-validation-batches", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default="bfloat16")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--hook-point", choices=["pre", "post"])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--eval-dataset", default="monology/pile-uncopyrighted")
    parser.add_argument("--eval-dataset-config")
    parser.add_argument("--eval-split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--loss-recovered-inputs", type=int, default=32)
    parser.add_argument("--loss-recovered-context-length", type=int, default=2048)
    parser.add_argument("--skip-loss-recovered", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.probe_max_dim < 1:
        raise ValueError("batch size and probe dimension must be positive")
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    configure_accelerator(device)
    root, manifest = load_activation_manifest(args.activation_manifest)
    model, checkpoint = load_model(args.checkpoint, device)
    fingerprint = manifest_fingerprint(manifest)
    if checkpoint.get("data_fingerprint") != fingerprint:
        raise ValueError("checkpoint and Pile activation manifest fingerprints differ")
    if model.cfg.window_size != int(manifest["window_size"]):
        raise ValueError("checkpoint and Pile activation window sizes differ")

    amp_dtype = args.amp_dtype if device.type == "cuda" else "none"
    sae_quality = evaluate_sae_quality(
        model,
        root,
        manifest,
        args.batch_size,
        args.maximum_validation_batches,
        device,
        amp_dtype,
    )

    bundle = torch_load(args.activations)
    x = bundle["activations"]
    metadata = bundle["metadata"]
    if len(x) != len(metadata):
        raise ValueError("MMLU activation rows and metadata differ")
    if x.ndim != 3 or x.shape[1:] != (
        model.cfg.window_size,
        model.cfg.d_in,
    ):
        raise ValueError(
            "MMLU activations must match checkpoint window size and residual width"
        )
    extraction = bundle.get("config", {})
    source_config = checkpoint.get("source_config", {})
    for key, expected in {
        "model": source_config.get("model"),
        "layer": source_config.get("layer"),
        "hook_point": source_config.get("hook_point"),
    }.items():
        actual = extraction.get(key)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(
                f"MMLU activation {key}={actual!r} != checkpoint {expected!r}"
            )
    train_indices, validation_indices, test_indices = grouped_three_way_split(
        metadata,
        args.validation_fraction,
        args.test_fraction,
        args.group_key,
        args.split_seed,
    )
    development_indices = sorted(train_indices + validation_indices)
    test_groups = np.asarray(
        [str(metadata[index].get(args.group_key, index)) for index in test_indices]
    )
    representations = encode_mmlu_representations(
        model, x, args.batch_size, device, amp_dtype
    )
    horizon_statistics = collect_horizon_statistics(
        model,
        x,
        test_indices,
        test_groups,
        args.batch_size,
        device,
        amp_dtype,
        args.seed,
    )
    horizon_curve = build_horizon_curve(
        horizon_statistics, test_groups, args.seed
    )
    probes = evaluate_probes(
        representations,
        metadata,
        development_indices,
        test_indices,
        args.group_key,
        args.probe_max_dim,
        args.seed,
    )
    diagnostics = {
        name: collapse_diagnostics(values) for name, values in representations.items()
    }
    mmlu_model_results = json.loads(
        Path(args.mmlu_model_results).read_text(encoding="utf-8")
    )
    activation_ids = {str(row[args.group_key]) for row in metadata}
    scored_ids = {str(value) for value in mmlu_model_results.get("question_ids", [])}
    mmlu_alignment = {
        "activation_rows": len(metadata),
        "base_model_scored_rows": int(mmlu_model_results.get("n", 0)),
        "question_id_overlap": len(activation_ids & scored_ids),
        "activation_only": len(activation_ids - scored_ids),
        "base_score_only": len(scored_ids - activation_ids),
    }

    loss_recovered = None
    if not args.skip_loss_recovered:
        model_name = args.model or source_config.get("model")
        layer_index = args.layer if args.layer is not None else source_config.get("layer")
        hook_point = args.hook_point or source_config.get("hook_point", "post")
        if model_name is None or layer_index is None:
            raise ValueError("model and layer are required for loss recovered")
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        loss_recovered = evaluate_loss_recovered(
            model,
            source_config,
            str(model_name),
            args.revision or source_config.get("resolved_model_revision"),
            args.use_safetensors,
            args.dtype,
            args.device_map,
            args.trust_remote_code,
            args.eval_dataset,
            args.eval_dataset_config,
            args.eval_split,
            args.text_key,
            args.loss_recovered_inputs,
            args.loss_recovered_context_length,
            str(hook_point),
            int(layer_index),
        )

    longest = horizon_curve[0]
    report = {
        "evaluation_protocol": {
            "goal": "conventional SAE quality plus endpoint-forecast validity",
            "sae": "final full-EMA high/low encoder-decoder",
            "forecast_controls": [
                "different-question shuffled context",
                "position-only predictor",
                "raw context-to-endpoint cosine",
            ],
            "mmlu_split": "question-grouped development/locked test",
            "time_smoothness_metrics": "not used",
        },
        "standard_sae_quality": sae_quality,
        "loss_recovered": loss_recovered,
        "forecast_validity": {
            "horizon_curve": horizon_curve,
            "longest_horizon": longest,
            "positive_over_shuffled": (
                longest["context_gain_over_shuffled"]["ci95_low"] > 0
            ),
            "positive_over_position_only": (
                longest["context_gain_over_position_only"]["ci95_low"] > 0
            ),
        },
        "mmlu_probe_accuracy": probes,
        "base_model_mmlu_accuracy": mmlu_model_results,
        "mmlu_alignment": mmlu_alignment,
        "representation_diagnostics": diagnostics,
        "split": {
            "development_n": len(development_indices),
            "locked_test_n": len(test_indices),
            "split_seed": args.split_seed,
            "group_key": args.group_key,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "architecture_id": checkpoint["architecture_id"],
            "config": checkpoint["config"],
            "data_fingerprint": fingerprint,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "transition_jepa_report.json", report)
    with (output_dir / "transition_horizon_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "context_position",
            "horizon",
            "context_target_cosine",
            "code_cosine",
            "shuffled_context_cosine",
            "position_only_cosine",
            "gain_over_shuffled",
            "gain_over_shuffled_ci95_low",
            "gain_over_shuffled_ci95_high",
            "gain_over_position_only",
            "gain_over_position_only_ci95_low",
            "gain_over_position_only_ci95_high",
            "code_nrmse",
            "support_precision",
            "support_recall",
            "support_jaccard",
            "residual_prediction_fvu",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in horizon_curve:
            shuffled = row["context_gain_over_shuffled"]
            position = row["context_gain_over_position_only"]
            writer.writerow(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "context_gain_over_shuffled",
                            "context_gain_over_position_only",
                        }
                    },
                    "gain_over_shuffled": shuffled["mean"],
                    "gain_over_shuffled_ci95_low": shuffled["ci95_low"],
                    "gain_over_shuffled_ci95_high": shuffled["ci95_high"],
                    "gain_over_position_only": position["mean"],
                    "gain_over_position_only_ci95_low": position["ci95_low"],
                    "gain_over_position_only_ci95_high": position["ci95_high"],
                }
            )
    with (output_dir / "mmlu_probe_accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "axis",
            "representation",
            "accuracy",
            "balanced_accuracy",
            "chance_accuracy",
            "ci95_low",
            "ci95_high",
            "n_development",
            "n_locked_test",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for axis, axis_results in probes.items():
            for representation, result in axis_results.items():
                writer.writerow(
                    {
                        "axis": axis,
                        "representation": representation,
                        "accuracy": result["accuracy"],
                        "balanced_accuracy": result["balanced_accuracy"],
                        "chance_accuracy": result["chance_accuracy"],
                        "ci95_low": result["group_bootstrap"]["ci95_low"],
                        "ci95_high": result["group_bootstrap"]["ci95_high"],
                        "n_development": result["n_development"],
                        "n_locked_test": result["n_locked_test"],
                    }
                )
    test_tensor = torch.as_tensor(test_indices, dtype=torch.long)
    torch.save(
        {
            "predicted_endpoint_high": pca_embedding(
                representations["predicted_endpoint_high"].index_select(0, test_tensor)
            ),
            "endpoint_high": pca_embedding(
                representations["endpoint_high"].index_select(0, test_tensor)
            ),
            "semantic_labels": [
                str(metadata[index][PROBE_LABELS["semantics"]])
                for index in test_indices
            ],
            "context_labels": [
                str(metadata[index][PROBE_LABELS["context"]])
                for index in test_indices
            ],
            "syntax_labels": [
                str(metadata[index][PROBE_LABELS["syntax"]])
                for index in test_indices
            ],
        },
        output_dir / "evaluation_embeddings.pt",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
