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
    validation_view_pair_batches,
)
from .evaluation import collapse_diagnostics, fit_probe, pca_embedding
from .io import torch_load, write_json
from .modeling import edit_residual, get_layer, input_device, load_hf_model
from .training import autocast_context, configure_accelerator, grouped_three_way_split
from .rectified_lpjepa_sae import (
    ARCHITECTURE_ID,
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    evaluate_losses as evaluate_training_objective,
)


PROBE_LABELS = {
    "semantics": "semantic_answer",
    "context": "context_category",
    "syntax": "syntax_template",
}


def load_model(
    path: str | Path, device: torch.device
) -> tuple[RectifiedLpJEPASAE, dict[str, Any]]:
    checkpoint = torch_load(path)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(
            f"{path} is not a {ARCHITECTURE_ID} checkpoint; retrain with the "
            "predictor-free Rectified LpJEPA objective"
        )
    model = RectifiedLpJEPASAE(
        RectifiedLpJEPAConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_sae_quality(
    model: RectifiedLpJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    """Evaluate the single trained SAE on held-out residuals."""
    totals: dict[str, float] = defaultdict(float)
    active_features = torch.zeros(model.cfg.d_sae, dtype=torch.float32)
    positions = 0
    residual_batch_size = batch_size * model.cfg.max_span_length
    for x in tqdm(
        validation_batches(root, manifest, residual_batch_size, maximum_batches),
        desc="standard SAE metrics",
    ):
        x = x.to(device, dtype=model.pre_bias.dtype, non_blocking=True)
        with autocast_context(device, amp_dtype):
            code, dense_high = model.encode_with_dense_high(x)
            reconstruction = model.decode(code)
            high, low = model.split_code(code)
            high_reconstruction = model.decode_high(high)
            low_reconstruction = model.pre_bias + model.decode_low(
                low, add_bias=False
            )
        x32 = x.float()
        totals["centered_energy"] += float(
            (x32 - model.pre_bias.float()).square().sum()
        )
        totals["full_squared_error"] += float(
            (x32 - reconstruction.float()).square().sum()
        )
        totals["high_squared_error"] += float(
            (x32 - high_reconstruction.float()).square().sum()
        )
        totals["low_squared_error"] += float(
            (x32 - low_reconstruction.float()).square().sum()
        )
        totals["l2_loss_sum"] += float(
            torch.linalg.vector_norm(x32 - reconstruction.float(), dim=-1).sum()
        )
        totals["cosine_sum"] += float(
            F.cosine_similarity(x32, reconstruction.float(), dim=-1).sum()
        )
        totals["l1_sum"] += float(code.float().abs().sum())
        totals["l0_sum"] += float((code != 0).float().sum())
        totals["high_l0_sum"] += float((high != 0).float().sum())
        totals["dense_high_l0_sum"] += float((dense_high != 0).float().sum())
        totals["low_l0_sum"] += float((low != 0).float().sum())
        totals["high_topk_saturated_sum"] += float(
            ((high != 0).sum(dim=-1) == model.cfg.high_k).float().sum()
        )
        totals["dense_high_energy"] += float(dense_high.float().square().sum())
        totals["sparse_high_energy"] += float(high.float().square().sum())
        totals["dense_sparse_high_cosine_sum"] += float(
            F.cosine_similarity(dense_high.float(), high.float(), dim=-1).sum()
        )
        active_features += (code != 0).reshape(-1, model.cfg.d_sae).any(
            dim=0
        ).float().cpu()
        positions += x.numel() // model.cfg.d_in
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
        "dense_high_l0": totals["dense_high_l0_sum"] / positions,
        "low_l0": totals["low_l0_sum"] / positions,
        "high_active_fraction": totals["high_l0_sum"]
        / (positions * model.cfg.d_high),
        "dense_high_active_fraction": totals["dense_high_l0_sum"]
        / (positions * model.cfg.d_high),
        "high_topk_saturation_fraction": totals["high_topk_saturated_sum"]
        / positions,
        "dense_to_sparse_high_energy_retained": totals["sparse_high_energy"]
        / max(totals["dense_high_energy"], 1e-12),
        "dense_sparse_high_cosine": totals["dense_sparse_high_cosine_sum"]
        / positions,
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
    }


def _mean_ci(values: list[float], seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    if len(array) == 1:
        value = float(array[0])
        return {"mean": value, "ci95_low": value, "ci95_high": value, "n": 1}
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(1000, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n": len(array),
    }


@torch.no_grad()
def evaluate_view_invariance(
    model: RectifiedLpJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp_dtype: str,
    seed: int,
) -> dict[str, Any]:
    by_distance: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for batch in tqdm(
        validation_view_pair_batches(
            root, manifest, batch_size, maximum_batches, seed
        ),
        desc="view invariance controls",
    ):
        view_a = batch["view_a"].to(device, dtype=model.pre_bias.dtype)
        view_b = batch["view_b"].to(device, dtype=model.pre_bias.dtype)
        distance = batch["distance"].tolist()
        with autocast_context(device, amp_dtype):
            code_a, dense_high_a = model.encode_with_dense_high(view_a)
            code_b, dense_high_b = model.encode_with_dense_high(view_b)
        high_a, low_a = model.split_code(code_a)
        high_b, low_b = model.split_code(code_b)
        permutation = torch.roll(torch.arange(len(view_a), device=device), 1)
        high_positive = F.cosine_similarity(
            high_a.float(), high_b.float(), dim=-1
        )
        high_shuffled = F.cosine_similarity(
            high_a.float(),
            high_b.index_select(0, permutation).float(),
            dim=-1,
        )
        dense_high_positive = F.cosine_similarity(
            dense_high_a.float(), dense_high_b.float(), dim=-1
        )
        dense_high_shuffled = F.cosine_similarity(
            dense_high_a.float(),
            dense_high_b.index_select(0, permutation).float(),
            dim=-1,
        )
        low_positive = F.cosine_similarity(
            low_a.float(), low_b.float(), dim=-1
        )
        residual_energy = (view_a - model.pre_bias).float().square().sum(dim=-1)
        with autocast_context(device, amp_dtype):
            reconstruction = model.decode(code_a)
            swapped = model.decode_high(high_b) + model.decode_low(
                low_a, add_bias=False
            )
            shuffled_swap = model.decode_high(
                high_b.index_select(0, permutation)
            ) + model.decode_low(low_a, add_bias=False)
        reconstruction_squared_error = (
            reconstruction - view_a
        ).float().square().sum(dim=-1)
        swap_squared_error = (swapped - view_a).float().square().sum(dim=-1)
        shuffled_swap_squared_error = (
            shuffled_swap - view_a
        ).float().square().sum(dim=-1)
        high_nrmse = torch.linalg.vector_norm(
            high_a.float() - high_b.float(), dim=-1
        ) / torch.linalg.vector_norm(high_b.float(), dim=-1).clamp_min(1e-8)
        tensors = {
            "high_positive_cosine": high_positive,
            "high_shuffled_cosine": high_shuffled,
            "high_margin": high_positive - high_shuffled,
            "dense_high_positive_cosine": dense_high_positive,
            "dense_high_shuffled_cosine": dense_high_shuffled,
            "dense_high_margin": dense_high_positive - dense_high_shuffled,
            "low_positive_cosine": low_positive,
            "high_nrmse": high_nrmse,
            "residual_energy": residual_energy,
            "reconstruction_squared_error": reconstruction_squared_error,
            "swap_squared_error": swap_squared_error,
            "shuffled_swap_squared_error": shuffled_swap_squared_error,
        }
        for row, horizon in enumerate(distance):
            for name, values in tensors.items():
                by_distance[int(horizon)][name].append(float(values[row]))
    curve: list[dict[str, Any]] = []
    for distance, values in sorted(by_distance.items()):
        row: dict[str, Any] = {"distance": distance, "n": len(next(iter(values.values())))}
        reported = {
            name: samples
            for name, samples in values.items()
            if not name.endswith("_squared_error") and name != "residual_energy"
        }
        for metric_index, (name, samples) in enumerate(reported.items()):
            summary = _mean_ci(samples, seed + distance * 100 + metric_index)
            row[name] = summary["mean"]
            if name.endswith("margin"):
                row[f"{name}_ci95_low"] = summary["ci95_low"]
                row[f"{name}_ci95_high"] = summary["ci95_high"]
        energy = max(sum(values["residual_energy"]), 1e-12)
        row["reconstruction_fvu"] = sum(
            values["reconstruction_squared_error"]
        ) / energy
        row["swap_reconstruction_fvu"] = sum(
            values["swap_squared_error"]
        ) / energy
        row["shuffled_swap_fvu"] = sum(
            values["shuffled_swap_squared_error"]
        ) / energy
        row["swap_penalty_fvu"] = (
            row["swap_reconstruction_fvu"] - row["reconstruction_fvu"]
        )
        row["shuffled_swap_penalty_fvu"] = (
            row["shuffled_swap_fvu"] - row["reconstruction_fvu"]
        )
        curve.append(row)
    all_margin = [
        value for metrics in by_distance.values() for value in metrics["high_margin"]
    ]
    margin_summary = _mean_ci(all_margin, seed + 1)
    all_dense_margin = [
        value
        for metrics in by_distance.values()
        for value in metrics["dense_high_margin"]
    ]
    dense_margin_summary = _mean_ci(all_dense_margin, seed + 2)
    return {
        "distance_curve": curve,
        "overall_high_margin": margin_summary,
        "overall_dense_high_margin": dense_margin_summary,
        "positive_over_shuffled": margin_summary["ci95_low"] > 0,
        "swap_fvu_aggregation": "sum squared error divided by sum centered residual energy",
    }


def causal_lm_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
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
    sae: RectifiedLpJEPASAE,
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
                text, return_tensors="pt", truncation=True, max_length=context_length
            )
            if encoded["input_ids"].shape[1] >= 2:
                break
        encoded = {key: value.to(token_device) for key, value in encoded.items()}
        attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
        original = llm(**encoded, use_cache=False).logits

        def reconstruct(hidden: torch.Tensor) -> torch.Tensor:
            value = hidden.to(sae.pre_bias.dtype)
            return sae.decode(sae.encode(value)).to(hidden.dtype)

        with edit_residual(layer, hook_point, reconstruct):
            reconstructed = llm(**encoded, use_cache=False).logits
        with edit_residual(layer, hook_point, torch.zeros_like):
            zero = llm(**encoded, use_cache=False).logits
        losses = {
            "loss_original": causal_lm_loss(original, encoded["input_ids"], attention_mask),
            "loss_reconstructed": causal_lm_loss(reconstructed, encoded["input_ids"], attention_mask),
            "loss_zero": causal_lm_loss(zero, encoded["input_ids"], attention_mask),
        }
        denominator = losses["loss_original"] - losses["loss_zero"]
        totals["fraction_loss_recovered"] += float(
            (losses["loss_reconstructed"] - losses["loss_zero"])
            / denominator.clamp_max(-1e-8)
        )
        for key, value in losses.items():
            totals[key] += float(value)
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


def _representation_batch(
    model: RectifiedLpJEPASAE, batch: torch.Tensor
) -> dict[str, torch.Tensor]:
    batch_size, width, _ = batch.shape
    flat = batch.reshape(-1, model.cfg.d_in)
    code = model.encode(flat).reshape(batch_size, width, model.cfg.d_sae)
    high, low = model.split_code(code)
    return {
        "high_mean": high.mean(dim=1),
        "endpoint_high": high[:, -1],
        "low_mean": low.mean(dim=1),
        "endpoint_low": low[:, -1],
        "endpoint_full": code[:, -1],
    }


@torch.no_grad()
def encode_mmlu_representations(
    model: RectifiedLpJEPASAE,
    x: torch.Tensor,
    development_indices: list[int],
    probe_max_dim: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Two-pass encoding stores only high-variance probe dimensions on CPU."""
    development_mask = torch.zeros(len(x), dtype=torch.bool)
    development_mask[development_indices] = True
    sums: dict[str, torch.Tensor] = {}
    squares: dict[str, torch.Tensor] = {}
    count = 0
    for start in tqdm(range(0, len(x), batch_size), desc="MMLU dimension pass"):
        end = min(start + batch_size, len(x))
        mask = development_mask[start:end]
        if not mask.any():
            continue
        batch = x[start:end].to(device, dtype=model.pre_bias.dtype)
        with autocast_context(device, amp_dtype):
            values = _representation_batch(model, batch)
        local_indices = mask.nonzero(as_tuple=False).flatten().to(device)
        for name, value in values.items():
            selected = value.index_select(0, local_indices).float()
            sums[name] = sums.get(name, torch.zeros(value.shape[-1], device=device)) + selected.sum(dim=0)
            squares[name] = squares.get(name, torch.zeros(value.shape[-1], device=device)) + selected.square().sum(dim=0)
        count += int(mask.sum())
    if count < 2:
        raise ValueError("at least two development rows are required for probes")
    selected_dimensions: dict[str, torch.Tensor] = {}
    for name in sums:
        variance = squares[name] / count - (sums[name] / count).square()
        width = min(probe_max_dim, len(variance))
        selected_dimensions[name] = torch.topk(variance, width).indices.sort().values
    representations = {
        name: torch.empty((len(x), len(indices)), dtype=torch.float16)
        for name, indices in selected_dimensions.items()
    }
    for start in tqdm(range(0, len(x), batch_size), desc="MMLU representation pass"):
        end = min(start + batch_size, len(x))
        batch = x[start:end].to(device, dtype=model.pre_bias.dtype)
        with autocast_context(device, amp_dtype):
            values = _representation_batch(model, batch)
        for name, value in values.items():
            representations[name][start:end].copy_(
                value.index_select(-1, selected_dimensions[name]).float().cpu()
            )
    return representations, {
        name: indices.cpu().tolist() for name, indices in selected_dimensions.items()
    }


def evaluate_probes(
    representations: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    development_indices: list[int],
    test_indices: list[int],
    group_key: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for axis_index, (axis, label_key) in enumerate(PROBE_LABELS.items()):
        axis_results: dict[str, Any] = {}
        for representation_index, (name, values) in enumerate(representations.items()):
            result, _ = fit_probe(
                values,
                metadata,
                development_indices,
                test_indices,
                label_key,
                group_key,
                seed + 1000 * axis_index + representation_index,
            )
            result["probe_dimension"] = values.shape[1]
            axis_results[name] = result
        results[axis] = axis_results
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SAE quality, Rectified distribution matching, and view invariance"
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
    if args.batch_size < 2 or args.probe_max_dim < 1:
        raise ValueError("batch size must be at least two and probe dimension positive")
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
    if model.cfg.max_span_length != int(manifest["max_span_length"]):
        raise ValueError("checkpoint and Pile maximum span lengths differ")
    amp_dtype = args.amp_dtype if device.type == "cuda" else "none"
    sae_quality = evaluate_sae_quality(
        model, root, manifest, args.batch_size, args.maximum_validation_batches, device, amp_dtype
    )
    view_invariance = evaluate_view_invariance(
        model, root, manifest, args.batch_size, args.maximum_validation_batches, device, amp_dtype, args.seed + 17
    )
    train_args = checkpoint.get("train_args", {})
    objective_validation = evaluate_training_objective(
        model,
        root,
        manifest,
        args.batch_size,
        min(8, args.maximum_validation_batches) if args.maximum_validation_batches else 8,
        device,
        amp_dtype,
        args.seed + 29,
        float(train_args.get("invariance_weight", 1.0)),
        float(train_args.get("rdm_weight", 5.0)),
        min(256, int(train_args.get("rdm_projections", 1024))),
        min(128, int(train_args.get("rdm_projection_chunk_size", 128))),
        min(512, int(train_args.get("axis_rdm_features", 512))),
        float(train_args.get("axis_rdm_weight", 1.0)),
    )
    model.eval()

    bundle = torch_load(args.activations)
    x = bundle["activations"]
    metadata = bundle["metadata"]
    if len(x) != len(metadata):
        raise ValueError("MMLU activation rows and metadata differ")
    if x.ndim != 3 or x.shape[1:] != (model.cfg.max_span_length, model.cfg.d_in):
        raise ValueError("MMLU activation span must match checkpoint span and residual width")
    extraction = bundle.get("config", {})
    source_config = checkpoint.get("source_config", {})
    for key, expected in {
        "model": source_config.get("model"),
        "layer": source_config.get("layer"),
        "hook_point": source_config.get("hook_point"),
    }.items():
        actual = extraction.get(key)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(f"MMLU activation {key}={actual!r} != checkpoint {expected!r}")
    train_indices, validation_indices, test_indices = grouped_three_way_split(
        metadata,
        args.validation_fraction,
        args.test_fraction,
        args.group_key,
        args.split_seed,
    )
    development_indices = sorted(train_indices + validation_indices)
    representations, selected_dimensions = encode_mmlu_representations(
        model,
        x,
        development_indices,
        args.probe_max_dim,
        args.batch_size,
        device,
        amp_dtype,
    )
    probes = evaluate_probes(
        representations,
        metadata,
        development_indices,
        test_indices,
        args.group_key,
        args.seed,
    )
    diagnostics = {
        name: collapse_diagnostics(values) for name, values in representations.items()
    }
    mmlu_model_results = json.loads(Path(args.mmlu_model_results).read_text(encoding="utf-8"))
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

    report = {
        "evaluation_protocol": {
            "goal": "standard SAE quality plus predictor-free shared-view validity",
            "positive_pair": "two exchangeable positions from the same random span",
            "jepa_code": "dense ReLU high candidates",
            "sae_evaluation_code": "ReLU plus Top-K sparse high code",
            "null": "sparse high code from a different validation sequence in the batch",
            "causal_decomposition_test": "swap same-span sparse high code while retaining position-specific low code",
            "mmlu_split": "question-grouped development/locked test",
            "predictor": None,
        },
        "standard_sae_quality": sae_quality,
        "loss_recovered": loss_recovered,
        "view_invariance": view_invariance,
        "rdm_validation": objective_validation,
        "mmlu_probe_accuracy": probes,
        "base_model_mmlu_accuracy": mmlu_model_results,
        "mmlu_alignment": mmlu_alignment,
        "representation_diagnostics": diagnostics,
        "selected_probe_dimensions": selected_dimensions,
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
    write_json(output_dir / "rectified_lpjepa_report.json", report)
    curve = view_invariance["distance_curve"]
    with (output_dir / "distance_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in curve for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)
    with (output_dir / "mmlu_probe_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "axis", "representation", "accuracy", "balanced_accuracy", "chance_accuracy",
            "ci95_low", "ci95_high", "n_development", "n_locked_test",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for axis, axis_results in probes.items():
            for representation, result in axis_results.items():
                writer.writerow({
                    "axis": axis,
                    "representation": representation,
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "chance_accuracy": result["chance_accuracy"],
                    "ci95_low": result["group_bootstrap"]["ci95_low"],
                    "ci95_high": result["group_bootstrap"]["ci95_high"],
                    "n_development": result["n_development"],
                    "n_locked_test": result["n_locked_test"],
                })
    test_tensor = torch.as_tensor(test_indices, dtype=torch.long)
    torch.save(
        {
            "high_mean": pca_embedding(representations["high_mean"].index_select(0, test_tensor)),
            "endpoint_high": pca_embedding(representations["endpoint_high"].index_select(0, test_tensor)),
            "endpoint_low": pca_embedding(representations["endpoint_low"].index_select(0, test_tensor)),
            "semantic_labels": [str(metadata[index][PROBE_LABELS["semantics"]]) for index in test_indices],
            "context_labels": [str(metadata[index][PROBE_LABELS["context"]]) for index in test_indices],
            "syntax_labels": [str(metadata[index][PROBE_LABELS["syntax"]]) for index in test_indices],
        },
        output_dir / "evaluation_embeddings.pt",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
