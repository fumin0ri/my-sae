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
    "online_context_target_cosine",
    "online_code_cosine",
    "online_shuffled_context_cosine",
    "ema_context_target_cosine",
    "ema_code_cosine",
    "ema_shuffled_context_cosine",
    "horizon_only_cosine",
    "online_code_nrmse",
    "ema_code_nrmse",
    "online_support_precision",
    "online_support_recall",
    "online_support_jaccard",
    "ema_support_precision",
    "ema_support_recall",
    "ema_support_jaccard",
    "online_residual_error",
    "ema_residual_error",
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
) -> dict[str, Any]:
    """Compare online and EMA SAE quality on exactly the same residuals."""
    totals: dict[str, dict[str, float]] = {
        "online": defaultdict(float),
        "ema": defaultdict(float),
    }
    active_features = {
        "online": torch.zeros(model.cfg.d_sae, dtype=torch.float32),
        "ema": torch.zeros(model.cfg.d_sae, dtype=torch.float32),
    }
    alignment: dict[str, float] = defaultdict(float)
    positions = 0
    residuals = 0
    batches = 0
    residual_batch_size = batch_size * model.cfg.max_span_length
    for x in tqdm(
        validation_batches(root, manifest, residual_batch_size, maximum_batches),
        desc="standard SAE metrics",
    ):
        x = x.to(device, dtype=model.pre_bias.dtype, non_blocking=True)
        with autocast_context(device, amp_dtype):
            online_code = model.encode(x)
            ema_code = model.encode_ema(x)
            variants = {
                "online": (
                    online_code,
                    model.decode(online_code),
                    model.pre_bias,
                    False,
                ),
                "ema": (
                    ema_code,
                    model.decode_ema(ema_code),
                    model.ema_pre_bias,
                    True,
                ),
            }
        x32 = x.float()
        for name, (code, reconstruction, bias, use_ema) in variants.items():
            high, low = model.split_code(code)
            with autocast_context(device, amp_dtype):
                high_reconstruction = model.decode_high(high, ema=use_ema)
                low_reconstruction = bias + model.decode_low(
                    low, ema=use_ema, add_bias=False
                )
            reconstruction32 = reconstruction.float()
            local = totals[name]
            local["centered_energy"] += float(
                (x32 - bias.float()).square().sum()
            )
            local["full_squared_error"] += float(
                (x32 - reconstruction32).square().sum()
            )
            local["high_squared_error"] += float(
                (x32 - high_reconstruction.float()).square().sum()
            )
            local["low_squared_error"] += float(
                (x32 - low_reconstruction.float()).square().sum()
            )
            local["l2_loss_sum"] += float(
                torch.linalg.vector_norm(x32 - reconstruction32, dim=-1).sum()
            )
            local["cosine_sum"] += float(
                F.cosine_similarity(x32, reconstruction32, dim=-1).sum()
            )
            local["l1_sum"] += float(code.float().abs().sum())
            local["l0_sum"] += float((code != 0).float().sum())
            local["high_l0_sum"] += float((high != 0).float().sum())
            local["low_l0_sum"] += float((low != 0).float().sum())
            active_features[name] += (
                code.reshape(-1, model.cfg.d_sae).float().sum(dim=0).cpu()
            )
        online_active = online_code > 0
        ema_active = ema_code > 0
        intersection = (online_active & ema_active).sum(dim=-1).float()
        union = (online_active | ema_active).sum(dim=-1).float().clamp_min(1)
        alignment["code_cosine_sum"] += float(
            F.cosine_similarity(online_code.float(), ema_code.float(), dim=-1).sum()
        )
        alignment["support_jaccard_sum"] += float((intersection / union).sum())
        batch_positions = x.numel() // model.cfg.d_in
        positions += batch_positions
        residuals += batch_positions
        batches += 1
    if positions == 0:
        raise ValueError("no Pile validation activations were evaluated")
    def finalize(name: str) -> dict[str, float | int]:
        local = totals[name]
        scale = max(local["centered_energy"], 1e-12)
        full_fvu = local["full_squared_error"] / scale
        high_fvu = local["high_squared_error"] / scale
        low_fvu = local["low_squared_error"] / scale
        alive = active_features[name] != 0
        return {
            "l2_loss": local["l2_loss_sum"] / positions,
            "l1": local["l1_sum"] / positions,
            "l0": local["l0_sum"] / positions,
            "high_l0": local["high_l0_sum"] / positions,
            "low_l0": local["low_l0_sum"] / positions,
            "reconstruction_cosine": local["cosine_sum"] / positions,
            "reconstruction_fvu": full_fvu,
            "fraction_variance_explained": 1.0 - full_fvu,
            "high_only_fvu": high_fvu,
            "high_only_fraction_variance_explained": 1.0 - high_fvu,
            "low_only_fvu": low_fvu,
            "low_only_fraction_variance_explained": 1.0 - low_fvu,
            "alive_feature_fraction": float(alive.float().mean()),
            "dead_feature_fraction": float((~alive).float().mean()),
            "n_positions": positions,
            "n_residuals": residuals,
            "n_batches": batches,
        }

    online = finalize("online")
    ema = finalize("ema")
    return {
        "online": online,
        "ema": ema,
        "online_ema_alignment": {
            "code_cosine": alignment["code_cosine_sum"] / positions,
            "support_jaccard": alignment["support_jaccard_sum"] / positions,
            "fve_online_minus_ema": (
                online["fraction_variance_explained"]
                - ema["fraction_variance_explained"]
            ),
            "reconstruction_cosine_online_minus_ema": (
                online["reconstruction_cosine"] - ema["reconstruction_cosine"]
            ),
        },
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

        def reconstruct_online(hidden: torch.Tensor) -> torch.Tensor:
            value = hidden.to(sae.pre_bias.dtype)
            return sae.decode(sae.encode(value)).to(hidden.dtype)

        def reconstruct_ema(hidden: torch.Tensor) -> torch.Tensor:
            return sae.decode_ema(
                sae.encode_ema(hidden.to(sae.pre_bias.dtype))
            ).to(hidden.dtype)

        with edit_residual(layer, hook_point, reconstruct_online):
            reconstructed_online = llm(**encoded, use_cache=False).logits
        with edit_residual(layer, hook_point, reconstruct_ema):
            reconstructed_ema = llm(**encoded, use_cache=False).logits
        with edit_residual(layer, hook_point, torch.zeros_like):
            zero = llm(**encoded, use_cache=False).logits
        original_loss = causal_lm_loss(
            original, encoded["input_ids"], attention_mask
        )
        reconstructed_online_loss = causal_lm_loss(
            reconstructed_online, encoded["input_ids"], attention_mask
        )
        reconstructed_ema_loss = causal_lm_loss(
            reconstructed_ema, encoded["input_ids"], attention_mask
        )
        zero_loss = causal_lm_loss(zero, encoded["input_ids"], attention_mask)
        denominator = original_loss - zero_loss
        recovered_online = (
            reconstructed_online_loss - zero_loss
        ) / denominator.clamp_max(-1e-8)
        recovered_ema = (
            reconstructed_ema_loss - zero_loss
        ) / denominator.clamp_max(-1e-8)
        totals["loss_original"] += float(original_loss)
        totals["loss_reconstructed_online"] += float(reconstructed_online_loss)
        totals["loss_reconstructed_ema"] += float(reconstructed_ema_loss)
        totals["loss_zero"] += float(zero_loss)
        totals["fraction_loss_recovered_online"] += float(recovered_online)
        totals["fraction_loss_recovered_ema"] += float(recovered_ema)
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
        "context_high_online": torch.empty(
            (n, model.cfg.d_high), dtype=torch.float16
        ),
        "predicted_endpoint_high_online": torch.empty(
            (n, model.cfg.d_high), dtype=torch.float16
        ),
        "context_high_ema": torch.empty(
            (n, model.cfg.d_high), dtype=torch.float16
        ),
        "predicted_endpoint_high_ema": torch.empty(
            (n, model.cfg.d_high), dtype=torch.float16
        ),
        "endpoint_high_ema": torch.empty(
            (n, model.cfg.d_high), dtype=torch.float16
        ),
        "context_low_online": torch.empty(
            (n, model.cfg.d_low), dtype=torch.float16
        ),
        "endpoint_low_ema": torch.empty(
            (n, model.cfg.d_low), dtype=torch.float16
        ),
        "endpoint_full_ema": torch.empty(
            (n, model.cfg.d_sae), dtype=torch.float16
        ),
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
    longest_horizon = torch.full(
        (1,), model.cfg.max_span_length - 1, dtype=torch.long, device=device
    )
    for start in tqdm(range(0, len(x), batch_size), desc="MMLU representations"):
        end = min(start + batch_size, len(x))
        batch = x[start:end].to(
            device=device, dtype=model.pre_bias.dtype, non_blocking=True
        )
        with autocast_context(device, amp_dtype):
            context_online_full = model.encode(batch[:, 0])
            context_ema_full = model.encode_ema(batch[:, 0])
            endpoint_ema_full = model.encode_ema(batch[:, -1])
            context_online_high, context_online_low = model.split_code(
                context_online_full
            )
            context_ema_high, _ = model.split_code(context_ema_full)
            endpoint_ema_high, endpoint_ema_low = model.split_code(
                endpoint_ema_full
            )
            predicted_online = model.predict_from_code(
                context_online_high,
                horizons=longest_horizon.expand(len(batch)),
                use_context=True,
                sparse_output=True,
            )
            predicted_ema = model.predict_from_code(
                context_ema_high,
                horizons=longest_horizon.expand(len(batch)),
                use_context=True,
                sparse_output=True,
            )
        values = {
            "context_high_online": context_online_high,
            "predicted_endpoint_high_online": predicted_online,
            "context_high_ema": context_ema_high,
            "predicted_endpoint_high_ema": predicted_ema,
            "endpoint_high_ema": endpoint_ema_high,
            "context_low_online": context_online_low,
            "endpoint_low_ema": endpoint_ema_low,
            "endpoint_full_ema": endpoint_ema_full,
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
    n_contexts = model.cfg.max_span_length - 1
    statistics = {
        name: torch.empty((len(test_indices), n_contexts), dtype=torch.float32)
        for name in HORIZON_METRICS
    }
    permutation = different_group_permutation(groups, seed)
    shuffled_indices = [test_indices[int(index)] for index in permutation]
    horizons = torch.arange(
        n_contexts, 0, -1, dtype=torch.long, device=device
    )
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
            online_context_full = model.encode(batch[:, :-1])
            online_context_high, _ = model.split_code(online_context_full)
            ema_context_full = model.encode_ema(batch[:, :-1])
            ema_context_high, _ = model.split_code(ema_context_full)
            target_high, _ = model.split_code(model.encode_ema(batch[:, -1]))
            target = target_high[:, None].expand(-1, n_contexts, -1)
            online_prediction = model.predict_from_code(
                online_context_high, horizons, use_context=True
            )
            ema_prediction = model.predict_from_code(
                ema_context_high, horizons, use_context=True
            )
            sparse_online_prediction = topk_relu(
                online_prediction, model.cfg.k_high
            )
            sparse_ema_prediction = topk_relu(
                ema_prediction, model.cfg.k_high
            )
            horizon_only = model.predict_from_code(
                online_context_high, horizons, use_context=False
            )
            shuffled_online_high, _ = model.split_code(
                model.encode(shuffled_batch[:, :-1])
            )
            shuffled_ema_high, _ = model.split_code(
                model.encode_ema(shuffled_batch[:, :-1])
            )
            shuffled_online_prediction = model.predict_from_code(
                shuffled_online_high, horizons, use_context=True
            )
            shuffled_ema_prediction = model.predict_from_code(
                shuffled_ema_high, horizons, use_context=True
            )
            online_predicted_residual = model.decode_high(
                sparse_online_prediction, ema=True, add_bias=True
            )
            ema_predicted_residual = model.decode_high(
                sparse_ema_prediction, ema=True, add_bias=True
            )
        target32 = target.float()
        target_active = target32 > 0
        target_residual = batch[:, -1, None, :].float()
        support: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, sparse_prediction in {
            "online": sparse_online_prediction,
            "ema": sparse_ema_prediction,
        }.items():
            predicted_active = sparse_prediction.float() > 0
            intersection = (predicted_active & target_active).sum(dim=-1).float()
            support[name] = (
                intersection
                / predicted_active.sum(dim=-1).float().clamp_min(1),
                intersection / target_active.sum(dim=-1).float().clamp_min(1),
                intersection
                / (predicted_active | target_active)
                .sum(dim=-1)
                .float()
                .clamp_min(1),
            )
        values = {
            "online_context_target_cosine": F.cosine_similarity(
                online_context_high.float(), target32, dim=-1
            ),
            "online_code_cosine": F.cosine_similarity(
                online_prediction.float(), target32, dim=-1
            ),
            "online_shuffled_context_cosine": F.cosine_similarity(
                shuffled_online_prediction.float(), target32, dim=-1
            ),
            "ema_context_target_cosine": F.cosine_similarity(
                ema_context_high.float(), target32, dim=-1
            ),
            "ema_code_cosine": F.cosine_similarity(
                ema_prediction.float(), target32, dim=-1
            ),
            "ema_shuffled_context_cosine": F.cosine_similarity(
                shuffled_ema_prediction.float(), target32, dim=-1
            ),
            "horizon_only_cosine": F.cosine_similarity(
                horizon_only.float(), target32, dim=-1
            ),
            "online_code_nrmse": (
                online_prediction.float() - target32
            ).square().mean(dim=-1)
            / target32.square().mean(dim=-1).clamp_min(1e-8),
            "ema_code_nrmse": (
                ema_prediction.float() - target32
            ).square().mean(dim=-1)
            / target32.square().mean(dim=-1).clamp_min(1e-8),
            "online_support_precision": support["online"][0],
            "online_support_recall": support["online"][1],
            "online_support_jaccard": support["online"][2],
            "ema_support_precision": support["ema"][0],
            "ema_support_recall": support["ema"][1],
            "ema_support_jaccard": support["ema"][2],
            "online_residual_error": (
                online_predicted_residual.float() - target_residual
            ).square().mean(dim=-1),
            "ema_residual_error": (
                ema_predicted_residual.float() - target_residual
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
    n_contexts = statistics["online_code_cosine"].shape[1]
    rows: list[dict[str, Any]] = []
    for context_position in range(n_contexts):
        learned = statistics["online_code_cosine"][:, context_position].numpy()
        shuffled = statistics[
            "online_shuffled_context_cosine"
        ][:, context_position].numpy()
        ema_learned = statistics["ema_code_cosine"][:, context_position].numpy()
        ema_shuffled = statistics[
            "ema_shuffled_context_cosine"
        ][:, context_position].numpy()
        horizon_only = statistics["horizon_only_cosine"][:, context_position].numpy()
        online_residual_error = statistics[
            "online_residual_error"
        ][:, context_position]
        ema_residual_error = statistics["ema_residual_error"][:, context_position]
        residual_energy = statistics["residual_energy"][:, context_position]
        rows.append(
            {
                "horizon": n_contexts - context_position,
                "online_context_target_cosine": float(
                    statistics["online_context_target_cosine"][
                        :, context_position
                    ].mean()
                ),
                "online_code_cosine": float(learned.mean()),
                "online_shuffled_context_cosine": float(shuffled.mean()),
                "ema_context_target_cosine": float(
                    statistics["ema_context_target_cosine"][
                        :, context_position
                    ].mean()
                ),
                "ema_code_cosine": float(ema_learned.mean()),
                "ema_shuffled_context_cosine": float(ema_shuffled.mean()),
                "horizon_only_cosine": float(horizon_only.mean()),
                "online_gain_over_shuffled": clustered_mean_ci(
                    learned - shuffled, groups, seed + 101 * context_position
                ),
                "online_gain_over_horizon_only": clustered_mean_ci(
                    learned - horizon_only,
                    groups,
                    seed + 10007 + 101 * context_position,
                ),
                "ema_gain_over_shuffled": clustered_mean_ci(
                    ema_learned - ema_shuffled,
                    groups,
                    seed + 20011 + 101 * context_position,
                ),
                "ema_gain_over_horizon_only": clustered_mean_ci(
                    ema_learned - horizon_only,
                    groups,
                    seed + 30011 + 101 * context_position,
                ),
                "online_minus_ema_cosine": clustered_mean_ci(
                    learned - ema_learned,
                    groups,
                    seed + 40009 + 101 * context_position,
                ),
                "online_code_nrmse": float(
                    statistics["online_code_nrmse"][:, context_position].mean()
                ),
                "ema_code_nrmse": float(
                    statistics["ema_code_nrmse"][:, context_position].mean()
                ),
                "online_support_precision": float(
                    statistics["online_support_precision"][:, context_position].mean()
                ),
                "online_support_recall": float(
                    statistics["online_support_recall"][:, context_position].mean()
                ),
                "online_support_jaccard": float(
                    statistics["online_support_jaccard"][:, context_position].mean()
                ),
                "ema_support_precision": float(
                    statistics["ema_support_precision"][:, context_position].mean()
                ),
                "ema_support_recall": float(
                    statistics["ema_support_recall"][:, context_position].mean()
                ),
                "ema_support_jaccard": float(
                    statistics["ema_support_jaccard"][:, context_position].mean()
                ),
                "online_residual_prediction_fvu": float(
                    online_residual_error.mean()
                    / residual_energy.mean().clamp_min(1e-8)
                ),
                "ema_residual_prediction_fvu": float(
                    ema_residual_error.mean()
                    / residual_energy.mean().clamp_min(1e-8)
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
            "forecasting beats shuffled and horizon-only controls"
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
    if model.cfg.max_span_length != int(manifest["max_span_length"]):
        raise ValueError("checkpoint and Pile maximum span lengths differ")

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
        model.cfg.max_span_length,
        model.cfg.d_in,
    ):
        raise ValueError(
            "MMLU activation span must match checkpoint max span and residual width"
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
            "goal": "conventional SAE quality plus horizon-conditioned forecast validity",
            "sae_comparison": "online and EMA high/low encoder-decoder pairs",
            "primary_forecast": "P(E_online(x_{t-h}), h) -> E_EMA(x_t)",
            "secondary_forecast": "P(E_EMA(x_{t-h}), h) -> E_EMA(x_t)",
            "forecast_controls": [
                "different-question shuffled context",
                "horizon-only predictor",
                "raw context-to-endpoint cosine",
            ],
            "mmlu_split": "question-grouped development/locked test",
            "time_smoothness_metrics": "not used",
        },
        "standard_sae_quality": sae_quality,
        "loss_recovered": loss_recovered,
        "forecast_validity": {
            "primary_context_encoder": "online (training-matched)",
            "secondary_context_encoder": "EMA compatibility",
            "horizon_curve": horizon_curve,
            "longest_horizon": longest,
            "online_positive_over_shuffled": (
                longest["online_gain_over_shuffled"]["ci95_low"] > 0
            ),
            "online_positive_over_horizon_only": (
                longest["online_gain_over_horizon_only"]["ci95_low"] > 0
            ),
            "ema_positive_over_shuffled": (
                longest["ema_gain_over_shuffled"]["ci95_low"] > 0
            ),
            "ema_positive_over_horizon_only": (
                longest["ema_gain_over_horizon_only"]["ci95_low"] > 0
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
            "horizon",
            "online_context_target_cosine",
            "online_code_cosine",
            "online_shuffled_context_cosine",
            "ema_context_target_cosine",
            "ema_code_cosine",
            "ema_shuffled_context_cosine",
            "horizon_only_cosine",
            "online_gain_over_shuffled",
            "online_gain_over_shuffled_ci95_low",
            "online_gain_over_shuffled_ci95_high",
            "online_gain_over_horizon_only",
            "online_gain_over_horizon_only_ci95_low",
            "online_gain_over_horizon_only_ci95_high",
            "ema_gain_over_shuffled",
            "ema_gain_over_shuffled_ci95_low",
            "ema_gain_over_shuffled_ci95_high",
            "ema_gain_over_horizon_only",
            "ema_gain_over_horizon_only_ci95_low",
            "ema_gain_over_horizon_only_ci95_high",
            "online_minus_ema_cosine",
            "online_minus_ema_cosine_ci95_low",
            "online_minus_ema_cosine_ci95_high",
            "online_code_nrmse",
            "ema_code_nrmse",
            "online_support_precision",
            "online_support_recall",
            "online_support_jaccard",
            "ema_support_precision",
            "ema_support_recall",
            "ema_support_jaccard",
            "online_residual_prediction_fvu",
            "ema_residual_prediction_fvu",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in horizon_curve:
            online_shuffled = row["online_gain_over_shuffled"]
            online_horizon = row["online_gain_over_horizon_only"]
            ema_shuffled = row["ema_gain_over_shuffled"]
            ema_horizon = row["ema_gain_over_horizon_only"]
            online_minus_ema = row["online_minus_ema_cosine"]
            writer.writerow(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "online_gain_over_shuffled",
                            "online_gain_over_horizon_only",
                            "ema_gain_over_shuffled",
                            "ema_gain_over_horizon_only",
                            "online_minus_ema_cosine",
                        }
                    },
                    "online_gain_over_shuffled": online_shuffled["mean"],
                    "online_gain_over_shuffled_ci95_low": online_shuffled["ci95_low"],
                    "online_gain_over_shuffled_ci95_high": online_shuffled["ci95_high"],
                    "online_gain_over_horizon_only": online_horizon["mean"],
                    "online_gain_over_horizon_only_ci95_low": online_horizon["ci95_low"],
                    "online_gain_over_horizon_only_ci95_high": online_horizon["ci95_high"],
                    "ema_gain_over_shuffled": ema_shuffled["mean"],
                    "ema_gain_over_shuffled_ci95_low": ema_shuffled["ci95_low"],
                    "ema_gain_over_shuffled_ci95_high": ema_shuffled["ci95_high"],
                    "ema_gain_over_horizon_only": ema_horizon["mean"],
                    "ema_gain_over_horizon_only_ci95_low": ema_horizon["ci95_low"],
                    "ema_gain_over_horizon_only_ci95_high": ema_horizon["ci95_high"],
                    "online_minus_ema_cosine": online_minus_ema["mean"],
                    "online_minus_ema_cosine_ci95_low": online_minus_ema["ci95_low"],
                    "online_minus_ema_cosine_ci95_high": online_minus_ema["ci95_high"],
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
            "predicted_endpoint_high_online": pca_embedding(
                representations["predicted_endpoint_high_online"].index_select(
                    0, test_tensor
                )
            ),
            "predicted_endpoint_high_ema": pca_embedding(
                representations["predicted_endpoint_high_ema"].index_select(
                    0, test_tensor
                )
            ),
            "endpoint_high_ema": pca_embedding(
                representations["endpoint_high_ema"].index_select(0, test_tensor)
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
