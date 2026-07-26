from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .io import torch_load, write_json
from .predictive_sae import (
    PredictiveSAEConfig,
    PredictiveSparseAutoencoder,
    autocast_context,
    configure_accelerator,
    fixed_spans,
    predictive_loss,
)


@dataclass
class LoadedModel:
    model: PredictiveSparseAutoencoder
    checkpoint: dict[str, Any]


def load_model(path: str | Path, device: torch.device) -> LoadedModel:
    checkpoint = torch_load(path)
    cfg = PredictiveSAEConfig(**checkpoint["config"])
    model = PredictiveSparseAutoencoder(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return LoadedModel(model=model, checkpoint=checkpoint)


@torch.no_grad()
def collect_representations(
    model: PredictiveSparseAutoencoder,
    x: torch.Tensor,
    spans: list[Any],
    batch_size: int,
    device: torch.device,
    amp_dtype: str = "none",
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {
        "target_code": [],
        "predicted_code": [],
        "predictable_residual": [],
        "innovation": [],
        "target_residual": [],
        "context_state": [],
    }
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        per_span: dict[str, list[torch.Tensor]] = {
            key: [] for key in collected
        }
        for span in spans:
            with autocast_context(device, amp_dtype):
                outputs = model(batch, span)
            per_span["target_code"].append(outputs["target_codes"].mean(dim=1))
            per_span["predicted_code"].append(
                outputs["predicted_codes"].mean(dim=1)
            )
            per_span["predictable_residual"].append(
                outputs["predictable"].mean(dim=1)
            )
            per_span["innovation"].append(outputs["innovation"].mean(dim=1))
            per_span["target_residual"].append(outputs["target"].mean(dim=1))
            per_span["context_state"].append(outputs["context_state"])
        for key, pieces in per_span.items():
            collected[key].append(torch.stack(pieces, dim=0).mean(dim=0).cpu())
    return {key: torch.cat(values) for key, values in collected.items()}


def representation_diagnostics(codes: torch.Tensor) -> dict[str, float]:
    values = codes.float()
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    effective_rank = float(
        variance.sum().square().div(variance.square().sum().clamp_min(1e-12)).item()
    )
    active = (values.abs() > 1e-8).float()
    return {
        "effective_rank": effective_rank,
        "mean_l0": float(active.sum(dim=-1).mean().item()),
        "active_dimension_fraction": float(
            (active.mean(dim=0) > 0).float().mean().item()
        ),
        "dead_dimension_fraction": float(
            (active.mean(dim=0) == 0).float().mean().item()
        ),
        "mean_dimension_std": float(centered.std(dim=0, unbiased=False).mean().item()),
    }


def group_bootstrap_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    seed: int,
    samples: int = 2000,
) -> dict[str, float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        values.append(float(np.mean(truth[indices] == prediction[indices])))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": samples,
    }


def fit_probe(
    values: torch.Tensor,
    metadata: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    label_key: str,
    group_key: str,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray([str(row[label_key]) for row in metadata])
    encoder = LabelEncoder().fit(labels[train_indices])
    train_y = encoder.transform(labels[train_indices])
    if any(label not in set(encoder.classes_) for label in labels[test_indices]):
        raise ValueError("locked test contains a label absent from development data")
    test_y = encoder.transform(labels[test_indices])
    train_x = values[train_indices].float().numpy()
    test_x = values[test_indices].float().numpy()
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(test_x)
    groups = np.asarray(
        [str(metadata[index].get(group_key, index)) for index in test_indices]
    )
    result = {
        "accuracy": float(accuracy_score(test_y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "chance_accuracy": float(
            max(np.bincount(test_y)) / max(1, len(test_y))
        ),
        "n_development": len(train_indices),
        "n_locked_test": len(test_indices),
        "classes": encoder.classes_.tolist(),
        "group_bootstrap": group_bootstrap_accuracy(
            test_y,
            prediction,
            groups,
            seed + 404,
        ),
    }
    return result, prediction == test_y


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def invariance_analysis(
    values: torch.Tensor,
    metadata: list[dict[str, Any]],
    indices: list[int],
    group_key: str,
    label_key: str,
    seed: int,
) -> dict[str, Any]:
    local = values[indices]
    rows = [metadata[index] for index in indices]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row.get(group_key, index)), []).append(index)
    same_problem: list[float] = []
    for members in groups.values():
        for left, right in itertools.combinations(members, 2):
            same_problem.append(float(cosine_rows(local[left : left + 1], local[right : right + 1]).item()))

    rng = np.random.default_rng(seed)
    same_state: list[float] = []
    different_state: list[float] = []
    candidates = list(range(len(rows)))
    for left in candidates:
        different_group = [
            right
            for right in candidates
            if rows[right].get(group_key) != rows[left].get(group_key)
        ]
        same = [
            right
            for right in different_group
            if rows[right].get(label_key) == rows[left].get(label_key)
        ]
        different = [
            right
            for right in different_group
            if rows[right].get(label_key) != rows[left].get(label_key)
        ]
        if same:
            right = int(rng.choice(same))
            same_state.append(float(cosine_rows(local[left : left + 1], local[right : right + 1]).item()))
        if different:
            right = int(rng.choice(different))
            different_state.append(float(cosine_rows(local[left : left + 1], local[right : right + 1]).item()))

    def summarize(group: list[float]) -> dict[str, float]:
        array = np.asarray(group, dtype=float)
        return {
            "n": len(group),
            "mean": float(array.mean()) if len(array) else float("nan"),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }

    same_problem_summary = summarize(same_problem)
    same_state_summary = summarize(same_state)
    different_state_summary = summarize(different_state)
    return {
        "same_problem_paraphrase": same_problem_summary,
        "different_problem_same_state": same_state_summary,
        "different_state": different_state_summary,
        "paraphrase_margin_over_same_state": (
            same_problem_summary["mean"] - same_state_summary["mean"]
        ),
        "semantic_margin_same_over_different_state": (
            same_state_summary["mean"] - different_state_summary["mean"]
        ),
    }


@torch.no_grad()
def gap_curve(
    model: PredictiveSparseAutoencoder,
    x: torch.Tensor,
    indices: list[int],
    spans: list[Any],
    batch_size: int,
    device: torch.device,
    amp_dtype: str = "none",
) -> list[dict[str, float | int]]:
    loader = DataLoader(
        TensorDataset(x[indices]),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    for span in spans:
        sums: dict[str, float] = {}
        count = 0
        for (batch,) in loader:
            with autocast_context(device, amp_dtype):
                _, metrics = predictive_loss(
                    model,
                    batch.to(device, non_blocking=True),
                    span,
                    prediction_weight=1.0,
                    residual_prediction_weight=0.1,
                )
            count += len(batch)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + len(batch) * value
        rows.append(
            {
                "target_size": span.target_size,
                "gap": span.gap,
                "code_cosine": sums["code_cosine"] / count,
                "code_nrmse": sums["code_nrmse"] / count,
                "residual_prediction_fvu": sums["residual_prediction_fvu"] / count,
            }
        )
    return rows


@torch.no_grad()
def shuffled_context_curve(
    model: PredictiveSparseAutoencoder,
    x: torch.Tensor,
    indices: list[int],
    spans: list[Any],
    batch_size: int,
    device: torch.device,
    amp_dtype: str = "none",
) -> list[dict[str, float | int]]:
    loader = DataLoader(
        TensorDataset(x[indices]),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    for span in spans:
        cosine_sum = 0.0
        residual_error_sum = 0.0
        residual_scale_sum = 0.0
        count = 0
        target_indices = torch.as_tensor(
            span.target_indices,
            device=device,
            dtype=torch.long,
        )
        for (batch,) in loader:
            batch = batch.to(device)
            # Grouped rows are contiguous, so a half-batch rotation avoids the
            # common failure where "shuffling" merely swaps two paraphrases of
            # the same underlying problem.
            shuffled_context = batch.roll(
                shifts=max(1, len(batch) // 2),
                dims=0,
            )
            with autocast_context(device, amp_dtype):
                predicted_codes, _ = model.predict_codes(
                    shuffled_context,
                    span,
                )
                target = batch.index_select(1, target_indices)
                target_codes = model.encode_target(target)
                predictable = model.decode(predicted_codes)
            cosine_sum += float(
                F.cosine_similarity(
                    predicted_codes,
                    target_codes,
                    dim=-1,
                ).sum().item()
            )
            residual_error_sum += float(
                (predictable - target).square().sum().item()
            )
            residual_scale_sum += float(
                (
                    target - model.pre_bias
                ).square().sum().item()
            )
            count += len(batch) * span.target_size
        rows.append(
            {
                "target_size": span.target_size,
                "gap": span.gap,
                "code_cosine": cosine_sum / max(count, 1),
                "residual_prediction_fvu": (
                    residual_error_sum / max(residual_scale_sum, 1e-12)
                ),
            }
        )
    return rows


def top_features(
    codes: torch.Tensor,
    metadata: list[dict[str, Any]],
    indices: list[int],
    count: int = 24,
    examples: int = 6,
) -> list[dict[str, Any]]:
    subset = codes[indices].float()
    frequency = (subset > 0).float().mean(dim=0)
    means = subset.mean(dim=0)
    score = means * frequency.sqrt()
    feature_ids = torch.topk(score, min(count, score.numel())).indices.tolist()
    result = []
    for feature_id in feature_ids:
        example_positions = torch.topk(
            subset[:, feature_id],
            min(examples, len(subset)),
        ).indices.tolist()
        result.append(
            {
                "feature_id": feature_id,
                "mean_activation": float(means[feature_id].item()),
                "activation_frequency": float(frequency[feature_id].item()),
                "top_examples": [
                    {
                        "window_index": indices[position],
                        "activation": float(subset[position, feature_id].item()),
                        "metadata": metadata[indices[position]],
                    }
                    for position in example_positions
                    if subset[position, feature_id] > 0
                ],
            }
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked-test evaluation of predictable and innovation components"
    )
    parser.add_argument("--activations", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-key", default="state")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    configure_accelerator(device)
    bundle = torch_load(args.activations)
    x = bundle["activations"].float()
    metadata = bundle["metadata"]
    joint = load_model(args.joint_checkpoint, device)
    baseline = load_model(args.baseline_checkpoint, device)
    if joint.checkpoint["split"] != baseline.checkpoint["split"]:
        raise ValueError("joint and baseline checkpoints must use the same split")
    split = joint.checkpoint["split"]
    train_indices = list(split["train_indices"]) + list(split["validation_indices"])
    test_indices = list(split["test_indices"])
    train_args = joint.checkpoint["train_args"]
    amp_dtype = (
        str(train_args.get("amp_dtype", "bfloat16"))
        if device.type == "cuda"
        else "none"
    )
    spans = fixed_spans(
        x.shape[1],
        joint.model.cfg.context_width,
        train_args["target_sizes"],
        train_args["gaps"],
        joint.model.cfg.context_mode,
    )
    joint_representations = collect_representations(
        joint.model,
        x,
        spans,
        args.batch_size,
        device,
        amp_dtype,
    )
    baseline_representations = collect_representations(
        baseline.model,
        x,
        spans,
        args.batch_size,
        device,
        amp_dtype,
    )
    representations = {
        "joint_predictable_code": joint_representations["predicted_code"],
        "joint_target_sae_code": joint_representations["target_code"],
        "joint_innovation_residual": joint_representations["innovation"],
        "posthoc_predictable_code": baseline_representations["predicted_code"],
        "standard_sae_code": baseline_representations["target_code"],
        "raw_target_residual": joint_representations["target_residual"],
    }
    probes: dict[str, Any] = {}
    correct: dict[str, np.ndarray] = {}
    for name, values in representations.items():
        probes[name], correct[name] = fit_probe(
            values,
            metadata,
            train_indices,
            test_indices,
            args.label_key,
            args.group_key,
            args.seed,
        )
    test_groups = np.asarray(
        [str(metadata[index].get(args.group_key, index)) for index in test_indices]
    )
    paired = (
        correct["joint_predictable_code"].astype(float)
        - correct["posthoc_predictable_code"].astype(float)
    )
    rng = np.random.default_rng(args.seed + 808)
    unique_groups = np.unique(test_groups)
    differences = []
    for _ in range(2000):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(test_groups == group) for group in sampled]
        )
        differences.append(float(paired[indices].mean()))
    low, high = np.quantile(differences, [0.025, 0.975])

    invariance = {
        name: invariance_analysis(
            values,
            metadata,
            test_indices,
            args.group_key,
            args.label_key,
            args.seed + index,
        )
        for index, (name, values) in enumerate(representations.items())
    }
    diagnostics = {
        name: representation_diagnostics(values[test_indices])
        for name, values in representations.items()
    }
    nuisance_probes: dict[str, Any] = {}
    for nuisance_key in (
        "template_id",
        "task_family",
        "contains_explicit_negation",
    ):
        development_values = {
            str(metadata[index].get(nuisance_key))
            for index in train_indices
            if nuisance_key in metadata[index]
        }
        test_values = {
            str(metadata[index].get(nuisance_key))
            for index in test_indices
            if nuisance_key in metadata[index]
        }
        if (
            len(development_values) < 2
            or not test_values
            or not test_values.issubset(development_values)
        ):
            continue
        nuisance_probes[nuisance_key] = {}
        for representation_name in (
            "joint_predictable_code",
            "posthoc_predictable_code",
            "raw_target_residual",
        ):
            nuisance_probes[nuisance_key][representation_name], _ = fit_probe(
                representations[representation_name],
                metadata,
                train_indices,
                test_indices,
                nuisance_key,
                args.group_key,
                args.seed + 17,
            )
    centered_target = (
        joint_representations["target_residual"][test_indices]
        - joint.model.pre_bias.detach().cpu()
    )
    predictable_write = (
        joint_representations["predictable_residual"][test_indices]
        - joint.model.pre_bias.detach().cpu()
    )
    innovation = joint_representations["innovation"][test_indices]
    energy = {
        "target_centered": float(centered_target.square().sum(dim=-1).mean().item()),
        "predictable_write": float(predictable_write.square().sum(dim=-1).mean().item()),
        "innovation": float(innovation.square().sum(dim=-1).mean().item()),
        "prediction_fvu": float(
            innovation.square().mean().div(
                centered_target.square().mean().clamp_min(1e-8)
            ).item()
        ),
    }
    report = {
        "claim": (
            "A sparse residual-space feature is called shared only when it can be "
            "predicted across a masked future gap and survives held-out semantic "
            "and causal tests."
        ),
        "split": {
            "group_key": args.group_key,
            "n_development_windows": len(train_indices),
            "n_locked_test_windows": len(test_indices),
            "locked_test_groups": len(unique_groups),
        },
        "masking": {
            "context_mode": joint.model.cfg.context_mode,
            "context_width": joint.model.cfg.context_width,
            "target_sizes": list(train_args["target_sizes"]),
            "gaps": list(train_args["gaps"]),
            "right_context_used": joint.model.cfg.context_mode == "retrospective",
        },
        "locked_test_probes": probes,
        "joint_minus_posthoc_probe_accuracy": {
            "difference": float(paired.mean()),
            "group_bootstrap_ci95_low": float(low),
            "group_bootstrap_ci95_high": float(high),
        },
        "paraphrase_and_semantic_invariance": invariance,
        "collapse_and_rank_diagnostics": diagnostics,
        "nuisance_probes": nuisance_probes,
        "predictable_innovation_energy": energy,
        "gap_curve": {
            "joint": gap_curve(
                joint.model,
                x,
                test_indices,
                spans,
                args.batch_size,
                device,
                amp_dtype,
            ),
            "posthoc": gap_curve(
                baseline.model,
                x,
                test_indices,
                spans,
                args.batch_size,
                device,
                amp_dtype,
            ),
        },
        "shuffled_context_null": {
            "joint": shuffled_context_curve(
                joint.model,
                x,
                test_indices,
                spans,
                args.batch_size,
                device,
                amp_dtype,
            ),
            "posthoc": shuffled_context_curve(
                baseline.model,
                x,
                test_indices,
                spans,
                args.batch_size,
                device,
                amp_dtype,
            ),
        },
        "top_predictable_features": top_features(
            joint_representations["predicted_code"],
            metadata,
            test_indices,
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "predictive_report.json", report)
    torch.save(
        {
            "joint": joint_representations,
            "baseline": baseline_representations,
            "metadata": metadata,
            "split": split,
            "label_key": args.label_key,
            "group_key": args.group_key,
        },
        output_dir / "predictive_codes.pt",
    )
    print(f"wrote locked-test report to {output_dir / 'predictive_report.json'}")


if __name__ == "__main__":
    main()
