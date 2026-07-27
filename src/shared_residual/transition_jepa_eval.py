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
    TransitionJEPAConfig,
    TransitionJEPASAE,
    support_metrics,
)

PROBE_LABELS = {
    "semantics": "semantic_answer",
    "context": "context_category",
    "syntax": "syntax_template",
}


def load_model(
    path: str | Path,
    device: torch.device,
) -> tuple[TransitionJEPASAE, dict[str, Any]]:
    checkpoint = torch_load(path)
    model = TransitionJEPASAE(TransitionJEPAConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def collect_model_outputs(
    model: TransitionJEPASAE,
    x: torch.Tensor,
    test_indices: list[int],
    groups: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
    use_context: bool,
    seed: int,
    label: str,
) -> dict[str, torch.Tensor | float]:
    contexts: list[torch.Tensor] = []
    final_predictions: list[torch.Tensor] = []
    reconstruction_error = 0.0
    reconstruction_scale = 0.0
    final_offset = torch.tensor(
        [model.cfg.window_size - 1],
        device=device,
        dtype=torch.long,
    )
    for start in tqdm(
        range(0, len(x), batch_size),
        desc=f"{label}: encode/reconstruct",
    ):
        batch = x[start : start + batch_size].to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            codes = model.encode(batch)
            reconstruction = model.decode(codes)
            prediction, _ = model.predict_from_code(
                codes[:, 0],
                offsets=final_offset,
                use_context=use_context,
            )
        contexts.append(codes[:, 0].float().cpu())
        final_predictions.append(prediction[:, 0].float().cpu())
        reconstruction_error += float(
            (reconstruction - batch).float().square().sum().item()
        )
        reconstruction_scale += float(
            (batch - model.pre_bias).float().square().sum().item()
        )

    context = torch.cat(contexts)
    test_x = x[test_indices]
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    residual_error: list[torch.Tensor] = []
    residual_energy: list[torch.Tensor] = []
    for start in tqdm(
        range(0, len(test_x), batch_size),
        desc=f"{label}: locked predictions",
    ):
        batch = test_x[start : start + batch_size].to(
            device,
            non_blocking=True,
        )
        with autocast_context(device, amp_dtype):
            outputs = model(batch, use_context=use_context)
        targets.append(outputs["target_codes"].float().cpu())
        predictions.append(outputs["predicted_codes"].float().cpu())
        residual_error.append(
            (
                outputs["predictable_residual"]
                - outputs["target_residual"]
            )
            .float()
            .square()
            .mean(dim=-1)
            .cpu()
        )
        residual_energy.append(
            (
                outputs["target_residual"] - model.pre_bias
            )
            .float()
            .square()
            .mean(dim=-1)
            .cpu()
        )
    target = torch.cat(targets)
    prediction = torch.cat(predictions)

    permutation = torch.as_tensor(
        different_group_permutation(groups, seed),
        dtype=torch.long,
    )
    shuffled_context = context[test_indices].index_select(0, permutation)
    shuffled_predictions: list[torch.Tensor] = []
    offsets = torch.arange(
        1,
        model.cfg.window_size,
        device=device,
        dtype=torch.long,
    )
    for start in tqdm(
        range(0, len(shuffled_context), batch_size),
        desc=f"{label}: shuffled null",
    ):
        batch_context = shuffled_context[start : start + batch_size].to(
            device=device,
            dtype=model.pre_bias.dtype,
            non_blocking=True,
        )
        with autocast_context(device, amp_dtype):
            shuffled, _ = model.predict_from_code(
                batch_context,
                offsets=offsets,
                use_context=use_context,
            )
        shuffled_predictions.append(shuffled.float().cpu())
    return {
        "context": context.to(torch.float16),
        "final_prediction": torch.cat(final_predictions).to(torch.float16),
        "test_target": target.to(torch.float16),
        "test_prediction": prediction.to(torch.float16),
        "test_shuffled_prediction": torch.cat(
            shuffled_predictions
        ).to(torch.float16),
        "residual_error": torch.cat(residual_error),
        "residual_energy": torch.cat(residual_energy),
        "reconstruction_fvu": reconstruction_error
        / max(reconstruction_scale, 1e-12),
    }


def offset_curve(
    outputs: dict[str, torch.Tensor | float],
    groups: np.ndarray,
    k: int,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    prediction = outputs["test_prediction"]
    shuffled = outputs["test_shuffled_prediction"]
    target = outputs["test_target"]
    residual_error = outputs["residual_error"]
    residual_energy = outputs["residual_energy"]
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(shuffled, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(residual_error, torch.Tensor)
    assert isinstance(residual_energy, torch.Tensor)
    rows = []
    for offset in tqdm(
        range(target.shape[1]),
        desc=f"{label}: offset metrics",
    ):
        predicted = prediction[:, offset].float()
        shuffled_predicted = shuffled[:, offset].float()
        target_code = target[:, offset].float()
        cosine = F.cosine_similarity(predicted, target_code, dim=-1)
        shuffled_cosine = F.cosine_similarity(
            shuffled_predicted,
            target_code,
            dim=-1,
        )
        nrmse = (
            (predicted - target_code).square().mean(dim=-1)
            / target_code.square().mean(dim=-1).clamp_min(1e-8)
        )
        precision, recall, jaccard = support_metrics(
            predicted,
            target_code,
            k,
        )
        context_gain = cosine.numpy() - shuffled_cosine.numpy()
        rows.append(
            {
                "offset": offset + 1,
                "code_cosine": float(cosine.mean().item()),
                "shuffled_context_cosine": float(
                    shuffled_cosine.mean().item()
                ),
                "context_gain": clustered_mean_ci(
                    context_gain,
                    groups,
                    seed + 101 * offset,
                ),
                "code_nrmse": float(nrmse.mean().item()),
                "support_precision": float(precision.mean().item()),
                "support_recall": float(recall.mean().item()),
                "support_jaccard": float(jaccard.mean().item()),
                "residual_prediction_fvu": float(
                    residual_error[:, offset].mean().div(
                        residual_energy[:, offset].mean().clamp_min(1e-8)
                    ).item()
                ),
                "innovation_energy_fraction": float(
                    residual_error[:, offset]
                    .div(residual_energy[:, offset].clamp_min(1e-8))
                    .mean()
                    .item()
                ),
                "prediction_norm": float(
                    predicted.norm(dim=-1).mean().item()
                ),
                "target_norm": float(
                    target_code.norm(dim=-1).mean().item()
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
        description="Locked-test evaluation of offset-conditioned JEPA-SAE"
    )
    parser.add_argument("--activations", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
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
    x = bundle["activations"].float()
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
    if int(mmlu_model_results.get("n", -1)) != len(metadata):
        raise ValueError(
            "base-model MMLU results and activation rows have different sizes"
        )
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
        "fixed": args.fixed_checkpoint,
        "k_only": args.k_only_checkpoint,
    }
    checkpoints: dict[str, dict[str, Any]] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    reconstruction_fvu: dict[str, float] = {}
    probe_contexts: dict[str, torch.Tensor] = {}
    probe_predictions: dict[str, torch.Tensor] = {}
    test_contexts: dict[str, torch.Tensor] = {}
    window_cosines: dict[str, torch.Tensor] = {}
    joint_final_prediction: torch.Tensor | None = None
    joint_final_target: torch.Tensor | None = None
    reference_fingerprint: str | None = None
    reference_cfg: TransitionJEPAConfig | None = None
    for method_index, (method, path) in enumerate(paths.items()):
        model, checkpoint = load_model(path, device)
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
        )
        curves[method] = offset_curve(
            collected,
            groups,
            model.cfg.k,
            args.seed + method_index,
            method,
        )
        reconstruction_fvu[method] = float(collected["reconstruction_fvu"])
        context = collected["context"]
        final_prediction = collected["final_prediction"]
        test_prediction = collected["test_prediction"]
        test_target = collected["test_target"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(final_prediction, torch.Tensor)
        assert isinstance(test_prediction, torch.Tensor)
        assert isinstance(test_target, torch.Tensor)
        probe_contexts[method] = select_probe_dimensions(
            context.float(),
            development_indices,
            args.probe_max_dim,
        ).to(torch.float16)
        probe_predictions[method] = select_probe_dimensions(
            final_prediction.float(),
            development_indices,
            args.probe_max_dim,
        ).to(torch.float16)
        if method in {"joint", "fixed"}:
            test_contexts[method] = context[test_indices].clone()
        window_cosines[method] = F.cosine_similarity(
            test_prediction.float(),
            test_target.float(),
            dim=-1,
        ).mean(dim=1)
        if method == "joint":
            joint_final_prediction = test_prediction[:, -1].clone()
            joint_final_target = test_target[:, -1].clone()
        checkpoints[method] = {
            "train_args": checkpoint["train_args"],
            "config": checkpoint["config"],
        }
        model.to("cpu")
        del model, checkpoint, collected, context, final_prediction
        del test_prediction, test_target
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert reference_fingerprint is not None
    assert reference_cfg is not None
    representations: dict[str, torch.Tensor] = {
        "joint_z0": probe_contexts["joint"].float(),
        "standard_sae_z0": probe_contexts["fixed"].float(),
        "joint_predicted_z9": probe_predictions["joint"].float(),
        "fixed_predicted_z9": probe_predictions["fixed"].float(),
        "k_only_predicted_z9": probe_predictions["k_only"].float(),
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
    assert joint_final_prediction is not None
    assert joint_final_target is not None
    comparison = clustered_mean_ci(
        (window_cosines["joint"] - window_cosines["fixed"]).numpy(),
        groups,
        args.seed + 909,
    )
    features = top_forecast_features(
        joint_final_prediction,
        joint_final_target,
        metadata,
        test_indices,
        reference_cfg.k,
    )
    feature_ids = [row["feature_id"] for row in features]
    report = {
        "claim": (
            "P(z0, k) extracts the sparse component of future residual states "
            "that is forecastable before observing intervening tokens."
        ),
        "interpretation_boundary": (
            "This is a conditional forecast under the data distribution, not "
            "a deterministic transition operator without intervening tokens."
        ),
        "architecture": {
            "window_size": reference_cfg.window_size,
            "context": "online Top-K SAE code at h0",
            "targets": "stop-gradient EMA SAE codes at h1...h9",
            "predictor": "offset-conditioned MLP",
            "target_aggregation": "none",
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
        "locked_test_offset_curve": curves,
        "primary_joint_minus_fixed_code_cosine": comparison,
        "reconstruction_fvu": reconstruction_fvu,
        "locked_test_mmlu_probes": probes,
        "collapse_diagnostics": {
            "joint_z0": collapse_diagnostics(
                test_contexts["joint"].float()
            ),
            "standard_sae_z0": collapse_diagnostics(
                test_contexts["fixed"].float()
            ),
            "joint_predicted_z9_topk": collapse_diagnostics(
                topk_relu(
                    joint_final_prediction.float(),
                    reference_cfg.k,
                )
            ),
        },
        "top_forecast_features_at_offset_9": features,
        "checkpoint_settings": checkpoints,
        "pile_training_data_fingerprint": reference_fingerprint,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "transition_jepa_report.json", report)
    with (output_dir / "transition_offset_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "method",
            "offset",
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
    sparse_final = topk_relu(
        joint_final_prediction.float(),
        reference_cfg.k,
    )
    torch.save(
        {
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
            "feature_activations": sparse_final[:, feature_ids],
            "metadata": [metadata[index] for index in test_indices],
        },
        output_dir / "transition_visualization.pt",
    )
    print(f"wrote locked-test report to {output_dir / 'transition_jepa_report.json'}")


if __name__ == "__main__":
    main()
