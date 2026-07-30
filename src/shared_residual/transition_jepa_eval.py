from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .group_sae import topk_relu
from .evaluation import (
    clustered_mean_ci,
    collapse_diagnostics,
    different_group_permutation,
    fit_probe,
    pca_embedding,
    select_probe_dimensions,
)
from .io import torch_load, write_json
from .training import (
    autocast_context,
    configure_accelerator,
    grouped_three_way_split,
)
from .transition_jepa_sae import (
    ARCHITECTURE_ID,
    TransitionJEPAConfig,
    TransitionJEPASAE,
)
from .hierarchical_transition_jepa_sae import (
    HIERARCHICAL_ARCHITECTURE_ID,
    HierarchicalTransitionJEPAConfig,
    HierarchicalTransitionJEPASAE,
)

PROBE_LABELS = {
    "semantics": "semantic_answer",
    "context": "context_category",
    "syntax": "syntax_template",
}
HORIZON_STATISTIC_NAMES = (
    "context_target_cosine",
    "code_cosine",
    "shuffled_context_cosine",
    "code_nrmse",
    "support_precision",
    "support_recall",
    "support_jaccard",
    "residual_error",
    "residual_energy",
    "prediction_norm",
    "target_norm",
)


def load_model(
    path: str | Path,
    device: torch.device,
) -> tuple[
    TransitionJEPASAE | HierarchicalTransitionJEPASAE,
    dict[str, Any],
]:
    checkpoint = torch_load(path)
    architecture_id = checkpoint.get("architecture_id")
    if architecture_id == ARCHITECTURE_ID:
        model = TransitionJEPASAE(
            TransitionJEPAConfig(**checkpoint["config"])
        )
    elif architecture_id == HIERARCHICAL_ARCHITECTURE_ID:
        model = HierarchicalTransitionJEPASAE(
            HierarchicalTransitionJEPAConfig(**checkpoint["config"])
        )
    else:
        raise ValueError(
            f"{path} has unsupported architecture_id={architecture_id!r}; "
            f"expected {ARCHITECTURE_ID!r} or "
            f"{HIERARCHICAL_ARCHITECTURE_ID!r}."
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def batch_horizon_statistics(
    outputs: dict[str, torch.Tensor],
    shuffled_prediction: torch.Tensor,
    pre_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = outputs["predicted_codes"]
    sparse_prediction = outputs["sparse_predicted_codes"]
    target = outputs["target_codes"]
    context_codes = outputs["context_codes"]
    batch_size, n_contexts, _ = prediction.shape
    statistics = {
        name: torch.empty(
            (batch_size, n_contexts),
            device=prediction.device,
            dtype=torch.float32,
        )
        for name in HORIZON_STATISTIC_NAMES
    }
    for context_position in range(n_contexts):
        predicted = prediction[:, context_position].float()
        shuffled = shuffled_prediction[:, context_position].float()
        target_code = target[:, context_position].float()
        context_code = context_codes[:, context_position].float()
        cosine = F.cosine_similarity(predicted, target_code, dim=-1)
        shuffled_cosine = F.cosine_similarity(
            shuffled,
            target_code,
            dim=-1,
        )
        target_energy = target_code.square().mean(dim=-1).clamp_min(1e-8)
        predicted_active = sparse_prediction[:, context_position] > 0
        target_active = target[:, context_position] > 0
        intersection = (predicted_active & target_active).sum(dim=-1).float()
        predicted_count = predicted_active.sum(dim=-1).float().clamp_min(1)
        target_count = target_active.sum(dim=-1).float().clamp_min(1)
        union = (
            (predicted_active | target_active)
            .sum(dim=-1)
            .float()
            .clamp_min(1)
        )
        residual_error = (
            outputs["predictable_residual"][:, context_position].float()
            - outputs["target_residual"][:, context_position].float()
        ).square().mean(dim=-1)
        residual_energy = (
            outputs["target_residual"][:, context_position].float()
            - pre_bias.float()
        ).square().mean(dim=-1)
        statistics["context_target_cosine"][:, context_position] = (
            F.cosine_similarity(context_code, target_code, dim=-1)
        )
        statistics["code_cosine"][:, context_position] = cosine
        statistics["shuffled_context_cosine"][:, context_position] = (
            shuffled_cosine
        )
        statistics["code_nrmse"][:, context_position] = (
            (predicted - target_code).square().mean(dim=-1) / target_energy
        )
        statistics["support_precision"][:, context_position] = (
            intersection / predicted_count
        )
        statistics["support_recall"][:, context_position] = (
            intersection / target_count
        )
        statistics["support_jaccard"][:, context_position] = (
            intersection / union
        )
        statistics["residual_error"][:, context_position] = residual_error
        statistics["residual_energy"][:, context_position] = residual_energy
        statistics["prediction_norm"][:, context_position] = predicted.norm(
            dim=-1
        )
        statistics["target_norm"][:, context_position] = target_code.norm(
            dim=-1
        )
    return statistics


@torch.no_grad()
def collect_model_outputs(
    model: TransitionJEPASAE | HierarchicalTransitionJEPASAE,
    x: torch.Tensor,
    test_indices: list[int],
    groups: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
    use_context: bool,
    seed: int,
    label: str,
    retain_long_horizon_test_codes: bool,
) -> dict[str, Any]:
    forecast_dim = model.forecast_dim
    context = torch.empty(
        (len(x), forecast_dim),
        dtype=torch.float16,
    )
    long_horizon_prediction = torch.empty_like(context)
    online_endpoint = torch.empty_like(context)
    endpoint_target = torch.empty_like(context)
    low_context = (
        torch.empty((len(x), model.low_dim), dtype=torch.float16)
        if isinstance(model, HierarchicalTransitionJEPASAE)
        else None
    )
    low_endpoint_target = (
        torch.empty_like(low_context) if low_context is not None else None
    )
    online_reconstruction_error = 0.0
    ema_reconstruction_error = 0.0
    ema_high_reconstruction_error = 0.0
    online_reconstruction_scale = 0.0
    ema_reconstruction_scale = 0.0
    endpoint_alignment_sum = 0.0
    for start in tqdm(
        range(0, len(x), batch_size),
        desc=f"{label}: encode/reconstruct",
    ):
        end = min(start + batch_size, len(x))
        batch = x[start:end].to(
            device=device,
            dtype=model.pre_bias.dtype,
            non_blocking=True,
        )
        with autocast_context(device, amp_dtype):
            outputs = model(
                batch,
                use_context=use_context,
                use_ema_context=True,
            )
        context[start:end].copy_(
            outputs["context_code"].to(torch.float16).cpu()
        )
        long_horizon_prediction[start:end].copy_(
            outputs["predicted_codes"][:, 0].to(torch.float16).cpu()
        )
        online_endpoint[start:end].copy_(
            outputs["online_target_code"].to(torch.float16).cpu()
        )
        endpoint_target[start:end].copy_(
            outputs["target_code"].to(torch.float16).cpu()
        )
        if low_context is not None:
            assert low_endpoint_target is not None
            low_context[start:end].copy_(
                outputs["low_context_code"].to(torch.float16).cpu()
            )
            low_endpoint_target[start:end].copy_(
                outputs["target_low_code"].to(torch.float16).cpu()
            )
        endpoint_alignment_sum += float(
            F.cosine_similarity(
                outputs["online_target_code"].float(),
                outputs["target_code"].float(),
                dim=-1,
            ).sum().item()
        )
        online_reconstruction_error += float(
            (
                outputs["online_target_reconstruction"] - batch[:, -1]
            ).float().square().sum().item()
        )
        ema_reconstruction_error += float(
            (
                outputs["target_reconstruction"] - batch[:, -1]
            ).float().square().sum().item()
        )
        if "target_high_reconstruction" in outputs:
            ema_high_reconstruction_error += float(
                (
                    outputs["target_high_reconstruction"] - batch[:, -1]
                ).float().square().sum().item()
            )
        online_reconstruction_scale += float(
            (batch[:, -1] - model.pre_bias).float().square().sum().item()
        )
        ema_reconstruction_scale += float(
            (
                batch[:, -1] - model.ema_pre_bias
            ).float().square().sum().item()
        )

    n_test = len(test_indices)
    n_contexts = model.cfg.window_size - 1
    avoided_dense_gib = (
        3 * n_test * n_contexts * forecast_dim * 2 / 2**30
    )
    print(
        f"{label}: streaming horizon statistics; avoiding approximately "
        f"{avoided_dense_gib:.1f} GiB of retained dense test codes"
    )
    horizon_statistics = {
        name: torch.empty((n_test, n_contexts), dtype=torch.float32)
        for name in HORIZON_STATISTIC_NAMES
    }
    long_horizon_test_prediction = (
        torch.empty((n_test, forecast_dim), dtype=torch.float16)
        if retain_long_horizon_test_codes
        else None
    )
    endpoint_test_target = (
        torch.empty_like(long_horizon_test_prediction)
        if long_horizon_test_prediction is not None
        else None
    )
    permutation = torch.as_tensor(
        different_group_permutation(groups, seed),
        dtype=torch.long,
    )
    shuffled_test_indices = [
        test_indices[int(position)]
        for position in permutation.tolist()
    ]
    for start in tqdm(
        range(0, n_test, batch_size),
        desc=f"{label}: locked streaming metrics",
    ):
        end = min(start + batch_size, n_test)
        indices = test_indices[start:end]
        batch = x[indices].to(
            device=device,
            dtype=model.pre_bias.dtype,
            non_blocking=True,
        )
        with autocast_context(device, amp_dtype):
            outputs = model(
                batch,
                use_context=use_context,
                use_ema_context=True,
            )
            if use_context:
                shuffled_batch = x[
                    shuffled_test_indices[start:end]
                ].to(
                    device=device,
                    dtype=model.pre_bias.dtype,
                    non_blocking=True,
                )
                shuffled_context_codes = model.encode_forecast_ema(
                    shuffled_batch[:, :-1]
                )
                shuffled = model.predict_from_code(
                    shuffled_context_codes,
                    use_context=True,
                )
            else:
                shuffled = outputs["predicted_codes"]
        batch_statistics = batch_horizon_statistics(
            outputs,
            shuffled,
            model.ema_pre_bias,
        )
        stacked_statistics = torch.stack(
            [batch_statistics[name] for name in HORIZON_STATISTIC_NAMES]
        ).cpu()
        for statistic_index, name in enumerate(HORIZON_STATISTIC_NAMES):
            horizon_statistics[name][start:end].copy_(
                stacked_statistics[statistic_index]
            )
        if long_horizon_test_prediction is not None:
            assert endpoint_test_target is not None
            long_horizon_test_prediction[start:end].copy_(
                outputs["predicted_codes"][:, 0]
                .to(torch.float16)
                .cpu()
            )
            endpoint_test_target[start:end].copy_(
                outputs["target_code"]
                .to(torch.float16)
                .cpu()
            )
    return {
        "context": context,
        "long_horizon_prediction": long_horizon_prediction,
        "online_endpoint": online_endpoint,
        "endpoint_target": endpoint_target,
        "low_context": low_context,
        "low_endpoint_target": low_endpoint_target,
        "horizon_statistics": horizon_statistics,
        "window_code_cosine": horizon_statistics["code_cosine"].mean(dim=1),
        "long_horizon_test_prediction": long_horizon_test_prediction,
        "endpoint_test_target": endpoint_test_target,
        "online_reconstruction_fvu": online_reconstruction_error
        / max(online_reconstruction_scale, 1e-12),
        "ema_reconstruction_fvu": ema_reconstruction_error
        / max(ema_reconstruction_scale, 1e-12),
        "ema_high_reconstruction_fvu": (
            ema_high_reconstruction_error
            / max(ema_reconstruction_scale, 1e-12)
            if low_context is not None
            else None
        ),
        "online_ema_endpoint_cosine": max(
            -1.0,
            min(1.0, endpoint_alignment_sum / max(len(x), 1)),
        ),
    }


def horizon_curve(
    outputs: dict[str, Any],
    groups: np.ndarray,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    statistics = outputs["horizon_statistics"]
    assert isinstance(statistics, dict)
    n_contexts = statistics["code_cosine"].shape[1]
    target_position = n_contexts
    rows = []
    for context_position in tqdm(
        range(n_contexts),
        desc=f"{label}: horizon metrics",
    ):
        horizon = target_position - context_position
        cosine = statistics["code_cosine"][:, context_position]
        shuffled_cosine = statistics[
            "shuffled_context_cosine"
        ][:, context_position]
        residual_error = statistics["residual_error"][:, context_position]
        residual_energy = statistics["residual_energy"][:, context_position]
        context_gain = cosine.numpy() - shuffled_cosine.numpy()
        rows.append(
            {
                "context_position": context_position,
                "horizon": horizon,
                "context_target_cosine": float(
                    statistics["context_target_cosine"][
                        :, context_position
                    ].mean().item()
                ),
                "code_cosine": float(cosine.mean().item()),
                "shuffled_context_cosine": float(
                    shuffled_cosine.mean().item()
                ),
                "context_gain": clustered_mean_ci(
                    context_gain,
                    groups,
                    seed + 101 * context_position,
                ),
                "code_nrmse": float(
                    statistics["code_nrmse"][
                        :, context_position
                    ].mean().item()
                ),
                "support_precision": float(
                    statistics["support_precision"][
                        :, context_position
                    ].mean().item()
                ),
                "support_recall": float(
                    statistics["support_recall"][
                        :, context_position
                    ].mean().item()
                ),
                "support_jaccard": float(
                    statistics["support_jaccard"][
                        :, context_position
                    ].mean().item()
                ),
                "residual_prediction_fvu": float(
                    residual_error.mean().div(
                        residual_energy.mean().clamp_min(1e-8)
                    ).item()
                ),
                "innovation_energy_fraction": float(
                    residual_error
                    .div(residual_energy.clamp_min(1e-8))
                    .mean()
                    .item()
                ),
                "prediction_norm": float(
                    statistics["prediction_norm"][
                        :, context_position
                    ].mean().item()
                ),
                "target_norm": float(
                    statistics["target_norm"][
                        :, context_position
                    ].mean().item()
                ),
            }
        )
    return rows


def top_forecast_features(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metadata: list[dict[str, Any]],
    test_indices: list[int],
    k: int,
    count: int = 24,
    examples: int = 5,
) -> list[dict[str, Any]]:
    sparse_prediction = topk_relu(prediction.float(), k)
    predicted_active = sparse_prediction > 0
    target_active = target.float() > 0
    frequency = predicted_active.float().mean(dim=0)
    precision = (
        (predicted_active & target_active).sum(dim=0).float()
        / predicted_active.sum(dim=0).float().clamp_min(1)
    )
    magnitude = sparse_prediction.mean(dim=0)
    valid = predicted_active.sum(dim=0) >= 3
    score = frequency.sqrt() * precision * magnitude
    # Very small smoke runs can have fewer than three activations for every
    # feature. Keep the report/heatmap usable while retaining the >=3 rule for
    # the confirmatory run.
    if valid.any():
        score = score.masked_fill(~valid, -1)
        candidate_count = int(valid.sum().item())
    else:
        candidate_count = len(score)
    feature_ids = torch.topk(
        score,
        min(count, candidate_count),
    ).indices.tolist()
    rows = []
    for feature_id in feature_ids:
        activations = sparse_prediction[:, feature_id]
        positions = torch.topk(
            activations,
            min(examples, len(activations)),
        ).indices.tolist()
        rows.append(
            {
                "feature_id": feature_id,
                "prediction_frequency": float(frequency[feature_id].item()),
                "target_precision": float(precision[feature_id].item()),
                "mean_prediction": float(magnitude[feature_id].item()),
                "top_examples": [
                    {
                        "window_index": test_indices[position],
                        "prediction": float(activations[position].item()),
                        "target": float(target[position, feature_id].item()),
                        "metadata": metadata[test_indices[position]],
                    }
                    for position in positions
                    if activations[position] > 0
                ],
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked-test evaluation of fixed-endpoint JEPA-SAE"
    )
    parser.add_argument("--activations", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument(
        "--hierarchical-checkpoint",
        required=True,
        help="T-SAE-inspired high/low model trained on the same artifacts",
    )
    parser.add_argument("--fixed-checkpoint", required=True)
    parser.add_argument("--k-only-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mmlu-model-results", required=True)
    parser.add_argument("--group-key", default="question_id")
    parser.add_argument("--semantics-key", default=PROBE_LABELS["semantics"])
    parser.add_argument("--context-key", default=PROBE_LABELS["context"])
    parser.add_argument("--syntax-key", default=PROBE_LABELS["syntax"])
    parser.add_argument("--probe-max-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def validate_mmlu_alignment(
    metadata: list[dict[str, Any]],
    results: dict[str, Any],
) -> None:
    activation_ids = [str(row["question_id"]) for row in metadata]
    if len(set(activation_ids)) != len(activation_ids):
        raise ValueError("activation rows contain duplicate MMLU question IDs")
    result_n = int(results.get("n", -1))
    result_ids_raw = results.get("question_ids")
    if result_ids_raw is None:
        if result_n != len(activation_ids):
            raise ValueError(
                "base-model MMLU results and activation rows cover different "
                f"question sets (base n={result_n:,}, activations "
                f"n={len(activation_ids):,}). This occurs when base scoring "
                "includes prompts shorter than the residual window. Rerun "
                "stage 4 with --minimum-tokens equal to WINDOW_SIZE; the "
                "existing activations and trained checkpoints can be reused."
            )
        return
    result_ids = [str(value) for value in result_ids_raw]
    if result_n != len(result_ids):
        raise ValueError(
            "base-model MMLU report has inconsistent n and question_ids"
        )
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("base-model MMLU results contain duplicate question IDs")
    missing = sorted(set(activation_ids) - set(result_ids))
    extra = sorted(set(result_ids) - set(activation_ids))
    if missing or extra:
        raise ValueError(
            "base-model MMLU results and activation rows cover different "
            f"question IDs (missing={len(missing):,}, extra={len(extra):,}). "
            "Rerun stage 4 with --minimum-tokens equal to WINDOW_SIZE."
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.probe_max_dim < 1:
        raise ValueError("--probe-max-dim must be positive")
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    configure_accelerator(device)
    bundle = torch_load(args.activations)
    # Keep the stored BF16 activations in their compact form. Batches are
    # promoted only when transferred to the evaluation device.
    x = bundle["activations"]
    metadata = bundle["metadata"]
    del bundle
    probe_labels = {
        "semantics": args.semantics_key,
        "context": args.context_key,
        "syntax": args.syntax_key,
    }
    for axis, key in probe_labels.items():
        if any(key not in row for row in metadata):
            raise KeyError(f"MMLU {axis} label {key!r} is missing")
    mmlu_model_results = json.loads(
        Path(args.mmlu_model_results).read_text(encoding="utf-8")
    )
    validate_mmlu_alignment(metadata, mmlu_model_results)
    train_indices, validation_indices, test_indices = grouped_three_way_split(
        metadata,
        args.validation_fraction,
        args.test_fraction,
        args.group_key,
        args.split_seed,
    )
    development_indices = train_indices + validation_indices
    groups = np.asarray(
        [
            str(metadata[index].get(args.group_key, index))
            for index in test_indices
        ]
    )
    paths = {
        "joint": args.joint_checkpoint,
        "hierarchical": args.hierarchical_checkpoint,
        "fixed": args.fixed_checkpoint,
        "k_only": args.k_only_checkpoint,
    }
    checkpoints: dict[str, dict[str, Any]] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    online_reconstruction_fvu: dict[str, float] = {}
    ema_reconstruction_fvu: dict[str, float] = {}
    ema_high_reconstruction_fvu: dict[str, float | None] = {}
    online_ema_endpoint_cosine: dict[str, float] = {}
    probe_contexts: dict[str, torch.Tensor] = {}
    probe_predictions: dict[str, torch.Tensor] = {}
    probe_online_endpoints: dict[str, torch.Tensor] = {}
    probe_targets: dict[str, torch.Tensor] = {}
    probe_low_contexts: dict[str, torch.Tensor] = {}
    probe_low_targets: dict[str, torch.Tensor] = {}
    test_contexts: dict[str, torch.Tensor] = {}
    window_cosines: dict[str, torch.Tensor] = {}
    joint_long_prediction: torch.Tensor | None = None
    joint_online_endpoint: torch.Tensor | None = None
    joint_endpoint_target: torch.Tensor | None = None
    reference_fingerprint: str | None = None
    reference_cfg: TransitionJEPAConfig | None = None
    hierarchical_cfg: HierarchicalTransitionJEPAConfig | None = None
    hierarchical_long_prediction: torch.Tensor | None = None
    hierarchical_endpoint_target: torch.Tensor | None = None
    for method_index, (method, path) in enumerate(paths.items()):
        model, checkpoint = load_model(path, device)
        if x.shape[1] != model.cfg.window_size:
            raise ValueError(
                "evaluation activation window does not match checkpoint: "
                f"activations={x.shape[1]}, checkpoint={model.cfg.window_size}"
            )
        if reference_fingerprint is None:
            reference_fingerprint = checkpoint["data_fingerprint"]
            reference_cfg = model.cfg
        elif checkpoint["data_fingerprint"] != reference_fingerprint:
            raise ValueError("all checkpoints must use the same Pile training data")
        amp_dtype = (
            str(checkpoint["train_args"].get("amp_dtype", "bfloat16"))
            if device.type == "cuda"
            else "none"
        )
        collected = collect_model_outputs(
            model,
            x,
            test_indices,
            groups,
            args.batch_size,
            device,
            amp_dtype,
            use_context=method != "k_only",
            seed=args.seed,
            label=method,
            retain_long_horizon_test_codes=method in {
                "joint",
                "hierarchical",
            },
        )
        curves[method] = horizon_curve(
            collected,
            groups,
            args.seed + method_index,
            method,
        )
        online_reconstruction_fvu[method] = float(
            collected["online_reconstruction_fvu"]
        )
        ema_reconstruction_fvu[method] = float(
            collected["ema_reconstruction_fvu"]
        )
        ema_high_reconstruction_fvu[method] = collected[
            "ema_high_reconstruction_fvu"
        ]
        online_ema_endpoint_cosine[method] = float(
            collected["online_ema_endpoint_cosine"]
        )
        context = collected["context"]
        long_prediction = collected["long_horizon_prediction"]
        online_endpoint = collected["online_endpoint"]
        endpoint_target = collected["endpoint_target"]
        window_code_cosine = collected["window_code_cosine"]
        long_test_prediction = collected["long_horizon_test_prediction"]
        endpoint_test_target = collected["endpoint_test_target"]
        low_context = collected["low_context"]
        low_endpoint_target = collected["low_endpoint_target"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(long_prediction, torch.Tensor)
        assert isinstance(online_endpoint, torch.Tensor)
        assert isinstance(endpoint_target, torch.Tensor)
        assert isinstance(window_code_cosine, torch.Tensor)
        probe_contexts[method] = select_probe_dimensions(
            context,
            development_indices,
            args.probe_max_dim,
        ).to(torch.float16)
        probe_predictions[method] = select_probe_dimensions(
            long_prediction,
            development_indices,
            args.probe_max_dim,
        ).to(torch.float16)
        probe_online_endpoints[method] = select_probe_dimensions(
            online_endpoint,
            development_indices,
            args.probe_max_dim,
        ).to(torch.float16)
        probe_targets[method] = select_probe_dimensions(
            endpoint_target,
            development_indices,
            args.probe_max_dim,
        ).to(torch.float16)
        if isinstance(low_context, torch.Tensor):
            assert isinstance(low_endpoint_target, torch.Tensor)
            probe_low_contexts[method] = select_probe_dimensions(
                low_context,
                development_indices,
                args.probe_max_dim,
            ).to(torch.float16)
            probe_low_targets[method] = select_probe_dimensions(
                low_endpoint_target,
                development_indices,
                args.probe_max_dim,
            ).to(torch.float16)
        if method in {"joint", "fixed"}:
            test_contexts[method] = context[test_indices].clone()
        window_cosines[method] = window_code_cosine
        if method == "joint":
            assert isinstance(long_test_prediction, torch.Tensor)
            assert isinstance(endpoint_test_target, torch.Tensor)
            joint_long_prediction = long_test_prediction
            joint_online_endpoint = online_endpoint[test_indices].clone()
            joint_endpoint_target = endpoint_test_target
        if method == "hierarchical":
            assert isinstance(model.cfg, HierarchicalTransitionJEPAConfig)
            assert isinstance(long_test_prediction, torch.Tensor)
            assert isinstance(endpoint_test_target, torch.Tensor)
            hierarchical_cfg = model.cfg
            hierarchical_long_prediction = long_test_prediction
            hierarchical_endpoint_target = endpoint_test_target
        checkpoints[method] = {
            "train_args": checkpoint["train_args"],
            "config": checkpoint["config"],
        }
        model.to("cpu")
        del model, checkpoint, collected, context, long_prediction
        del online_endpoint, endpoint_target, long_test_prediction
        del endpoint_test_target, low_context, low_endpoint_target
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert reference_fingerprint is not None
    assert reference_cfg is not None
    representations: dict[str, torch.Tensor] = {
        "joint_ema_z0": probe_contexts["joint"].float(),
        "fixed_ema_z0": probe_contexts["fixed"].float(),
        "hierarchical_high_ema_z0": (
            probe_contexts["hierarchical"].float()
        ),
        "hierarchical_low_ema_z0": (
            probe_low_contexts["hierarchical"].float()
        ),
        "joint_predicted_endpoint_from_h0": (
            probe_predictions["joint"].float()
        ),
        "fixed_predicted_endpoint_from_h0": (
            probe_predictions["fixed"].float()
        ),
        "hierarchical_high_predicted_endpoint_from_h0": (
            probe_predictions["hierarchical"].float()
        ),
        "position_only_predicted_endpoint": (
            probe_predictions["k_only"].float()
        ),
        "joint_online_endpoint": probe_online_endpoints["joint"].float(),
        "joint_ema_endpoint": probe_targets["joint"].float(),
        "hierarchical_high_ema_endpoint": (
            probe_targets["hierarchical"].float()
        ),
        "hierarchical_low_ema_endpoint": (
            probe_low_targets["hierarchical"].float()
        ),
        "raw_h0": select_probe_dimensions(
            x[:, 0],
            development_indices,
            args.probe_max_dim,
        ),
    }
    probes: dict[str, dict[str, Any]] = {}
    for axis_index, (axis, label_key) in enumerate(probe_labels.items()):
        probes[axis] = {
            name: fit_probe(
                values,
                metadata,
                development_indices,
                test_indices,
                label_key,
                args.group_key,
                args.seed + 1000 * axis_index,
            )[0]
            for name, values in tqdm(
                representations.items(),
                desc=f"MMLU {axis} probes",
            )
        }
    assert joint_long_prediction is not None
    assert joint_online_endpoint is not None
    assert joint_endpoint_target is not None
    assert hierarchical_cfg is not None
    assert hierarchical_long_prediction is not None
    assert hierarchical_endpoint_target is not None
    comparison = clustered_mean_ci(
        (window_cosines["joint"] - window_cosines["fixed"]).numpy(),
        groups,
        args.seed + 909,
    )
    hierarchical_comparison = clustered_mean_ci(
        (
            window_cosines["hierarchical"] - window_cosines["joint"]
        ).numpy(),
        groups,
        args.seed + 910,
    )
    features = top_forecast_features(
        joint_long_prediction,
        joint_endpoint_target,
        metadata,
        test_indices,
        reference_cfg.k,
    )
    feature_ids = [row["feature_id"] for row in features]
    hierarchical_features = top_forecast_features(
        hierarchical_long_prediction,
        hierarchical_endpoint_target,
        metadata,
        test_indices,
        hierarchical_cfg.k_high,
    )
    report = {
        "claim": (
            "The hierarchical model allocates a dedicated sparse high-level "
            "dictionary to the component of one fixed endpoint code that is "
            "forecastable from each earlier context; a separate low-level "
            "dictionary adds non-forecast-supervised reconstruction detail."
        ),
        "interpretation_boundary": (
            "This is a conditional forecast under the data distribution, not "
            "a deterministic transition operator without intervening tokens."
        ),
        "architecture": {
            "window_size": reference_cfg.window_size,
            "target_position": reference_cfg.window_size - 1,
            "longest_horizon": reference_cfg.window_size - 1,
            "contexts": (
                "final EMA Top-K SAE codes at h0..."
                f"h{reference_cfg.window_size - 2}"
            ),
            "target": (
                "one stop-gradient EMA SAE code at "
                f"h{reference_cfg.window_size - 1}"
            ),
            "predictor": "context-position-conditioned MLP",
            "predictor_position_input": "context_position k",
            "reported_horizon": "target_position - context_position",
            "online_sae": "gradient-trained student encoder and decoder",
            "ema_sae": "teacher and final encoder-decoder pair",
            "predicted_residual_decoder": "frozen EMA decoder",
            "ema_decoder_row_normalization": "after every EMA update",
            "excluded_objectives": [
                "EMA compatibility loss",
                "variance regularization",
            ],
            "evaluation_aggregation": (
                "stream scalar horizon statistics; retain dense codes only "
                "for the longest-horizon feature analysis"
            ),
            "hierarchical_partition": {
                "inspiration": (
                    "T-SAE/Temporal Matryoshka SAE group-0 high-level and "
                    "group-1 low-level cumulative reconstruction"
                ),
                "d_high": hierarchical_cfg.d_high,
                "d_low": hierarchical_cfg.d_low,
                "k_high": hierarchical_cfg.k_high,
                "k_low": hierarchical_cfg.k_low,
                "high_reconstruction_weight": (
                    hierarchical_cfg.high_reconstruction_weight
                ),
                "full_reconstruction_weight": (
                    1.0 - hierarchical_cfg.high_reconstruction_weight
                ),
                "prediction_supervision": "high group only",
            },
        },
        "benchmark": {
            "name": "MMLU",
            "dataset": metadata[0].get("dataset"),
            "dataset_revision": metadata[0].get("dataset_revision"),
            "dataset_split": metadata[0].get("dataset_split"),
            "n_questions": len(metadata),
            "probe_labels": probe_labels,
            "definitions": {
                "semantics": "correct option A/B/C/D after balanced permutation",
                "context": "official MMLU broad subject category",
                "syntax": "balanced surface prompt template",
            },
            "base_model_accuracy": mmlu_model_results,
        },
        "split": {
            "group_key": args.group_key,
            "split_seed": args.split_seed,
            "n_development_windows": len(development_indices),
            "n_locked_test_windows": len(test_indices),
            "n_locked_test_groups": len(np.unique(groups)),
        },
        "locked_test_horizon_curve": curves,
        "primary_joint_minus_fixed_code_cosine": comparison,
        "primary_hierarchical_minus_joint_code_cosine": (
            hierarchical_comparison
        ),
        "online_reconstruction_fvu": online_reconstruction_fvu,
        "ema_reconstruction_fvu": ema_reconstruction_fvu,
        "ema_high_reconstruction_fvu": ema_high_reconstruction_fvu,
        "online_ema_endpoint_cosine": online_ema_endpoint_cosine,
        "locked_test_mmlu_probes": probes,
        "collapse_diagnostics": {
            "joint_ema_z0": collapse_diagnostics(
                test_contexts["joint"].float()
            ),
            "fixed_ema_z0": collapse_diagnostics(
                test_contexts["fixed"].float()
            ),
            "hierarchical_high_ema_z0": collapse_diagnostics(
                probe_contexts["hierarchical"][test_indices].float()
            ),
            "hierarchical_low_ema_z0": collapse_diagnostics(
                probe_low_contexts["hierarchical"][test_indices].float()
            ),
            "joint_predicted_endpoint_from_h0_topk": collapse_diagnostics(
                topk_relu(
                    joint_long_prediction.float(),
                    reference_cfg.k,
                )
            ),
            "joint_ema_endpoint": collapse_diagnostics(
                joint_endpoint_target.float()
            ),
            "joint_online_endpoint": collapse_diagnostics(
                joint_online_endpoint.float()
            ),
        },
        "top_forecast_features": {
            "context_position": 0,
            "horizon": reference_cfg.window_size - 1,
            "features": features,
        },
        "hierarchical_top_high_features": {
            "context_position": 0,
            "horizon": reference_cfg.window_size - 1,
            "features": hierarchical_features,
        },
        "checkpoint_settings": checkpoints,
        "pile_training_data_fingerprint": reference_fingerprint,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "transition_jepa_report.json", report)
    with (output_dir / "transition_horizon_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "method",
            "context_position",
            "horizon",
            "context_target_cosine",
            "code_cosine",
            "shuffled_context_cosine",
            "context_gain",
            "context_gain_ci95_low",
            "context_gain_ci95_high",
            "code_nrmse",
            "support_precision",
            "support_recall",
            "support_jaccard",
            "residual_prediction_fvu",
            "innovation_energy_fraction",
            "prediction_norm",
            "target_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in curves.items():
            for row in rows:
                gain = row["context_gain"]
                writer.writerow(
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key != "context_gain"
                        },
                        "method": method,
                        "context_gain": gain["mean"],
                        "context_gain_ci95_low": gain["ci95_low"],
                        "context_gain_ci95_high": gain["ci95_high"],
                    }
                )
    with (output_dir / "mmlu_probe_accuracy.csv").open(
        "w",
        encoding="utf-8",
        newline="",
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
        for axis, axis_probes in probes.items():
            for representation, result in axis_probes.items():
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
    sparse_endpoint_prediction = topk_relu(
        joint_long_prediction.float(),
        reference_cfg.k,
    )
    torch.save(
        {
            "window_size": reference_cfg.window_size,
            "target_position": reference_cfg.window_size - 1,
            "longest_horizon": reference_cfg.window_size - 1,
            "embedding": pca_embedding(
                test_contexts["joint"].float()
            ),
            "semantic_labels": [
                str(metadata[index][args.semantics_key])
                for index in test_indices
            ],
            "context_labels": [
                str(metadata[index][args.context_key])
                for index in test_indices
            ],
            "syntax_labels": [
                str(metadata[index][args.syntax_key]) for index in test_indices
            ],
            "feature_ids": feature_ids,
            "feature_activations": (
                sparse_endpoint_prediction[:, feature_ids]
            ),
            "metadata": [metadata[index] for index in test_indices],
        },
        output_dir / "transition_visualization.pt",
    )
    print(f"wrote locked-test report to {output_dir / 'transition_jepa_report.json'}")


if __name__ == "__main__":
    main()
