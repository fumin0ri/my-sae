from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .io import torch_load, write_json
from .persistent_sae import PersistentSAEConfig, PersistentSparseAutoencoder
from .predictive_eval import fit_probe, invariance_analysis
from .predictive_sae import autocast_context, configure_accelerator


@dataclass
class LoadedModel:
    model: PersistentSparseAutoencoder
    checkpoint: dict[str, Any]


def load_model(path: str | Path, device: torch.device) -> LoadedModel:
    checkpoint = torch_load(path)
    model = PersistentSparseAutoencoder(
        PersistentSAEConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return LoadedModel(model=model, checkpoint=checkpoint)


@torch.no_grad()
def collect_codes(
    model: PersistentSparseAutoencoder,
    x: torch.Tensor,
    test_indices: list[int],
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, torch.Tensor | float]:
    contexts: list[torch.Tensor] = []
    test_targets: list[torch.Tensor] = []
    reconstruction_error = 0.0
    reconstruction_scale = 0.0
    test_lookup = {index: position for position, index in enumerate(test_indices)}
    test_chunks: list[tuple[int, torch.Tensor]] = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            codes = model.encode(batch)
            reconstruction = model.decode(codes)
        contexts.append(codes[:, 0].float().cpu())
        reconstruction_error += float((reconstruction - batch).float().square().sum().item())
        reconstruction_scale += float(
            (batch - model.pre_bias).float().square().sum().item()
        )
        local_test = [
            (test_lookup[index], index - start)
            for index in range(start, min(start + len(batch), len(x)))
            if index in test_lookup
        ]
        if local_test:
            output_positions = [pair[0] for pair in local_test]
            batch_positions = torch.tensor(
                [pair[1] for pair in local_test],
                device=device,
                dtype=torch.long,
            )
            with autocast_context(device, amp_dtype):
                target = model.encode_target(
                    batch.index_select(0, batch_positions)[:, 1:]
                )
            test_chunks.append((min(output_positions), target.float().cpu()))
    test_chunks.sort(key=lambda item: item[0])
    test_targets.extend(chunk for _, chunk in test_chunks)
    return {
        "context": torch.cat(contexts),
        "test_targets": torch.cat(test_targets),
        "reconstruction_fvu": reconstruction_error / max(reconstruction_scale, 1e-12),
    }


def different_group_permutation(groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    for _ in range(max(10, len(groups) * 2)):
        bad = groups[order] == groups
        if not bad.any():
            return order
        order[bad] = np.roll(order[bad], 1)
    # Deterministic fallback for unusually imbalanced group sizes.
    result = np.empty(len(groups), dtype=int)
    for index, group in enumerate(groups):
        candidates = np.flatnonzero(groups != group)
        if not len(candidates):
            raise ValueError("shuffled null needs at least two independent groups")
        result[index] = candidates[index % len(candidates)]
    return result


def clustered_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        bootstrap.append(float(values[indices].mean()))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": samples,
    }


@torch.no_grad()
def offset_metrics(
    context: torch.Tensor,
    targets: torch.Tensor,
    groups: np.ndarray,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    shuffled = torch.as_tensor(
        different_group_permutation(groups, seed),
        dtype=torch.long,
    )
    context_active = context > 0
    context_l0 = context_active.sum(dim=-1).float().clamp_min(1)
    rows = []
    for offset in range(targets.shape[1]):
        target = targets[:, offset]
        positive = F.cosine_similarity(context.float(), target.float(), dim=-1)
        null = F.cosine_similarity(
            context.float(),
            target.index_select(0, shuffled).float(),
            dim=-1,
        )
        target_active = target > 0
        intersection = (context_active & target_active).sum(dim=-1).float()
        union = (context_active | target_active).sum(dim=-1).float().clamp_min(1)
        survival = intersection / context_l0
        jaccard = intersection / union
        nrmse = (
            (context.float() - target.float()).square().mean(dim=-1)
            / target.float().square().mean(dim=-1).clamp_min(1e-8)
        )

        query = F.normalize(context.float().to(device), dim=-1)
        key = F.normalize(target.float().to(device), dim=-1)
        similarity = query @ key.T
        nearest = similarity.argmax(dim=1).cpu().numpy()
        retrieval = (groups[nearest] == groups).astype(float)
        labels = np.concatenate(
            [np.ones(len(positive)), np.zeros(len(null))]
        )
        scores = np.concatenate([positive.numpy(), null.numpy()])
        margin = positive.numpy() - null.numpy()
        rows.append(
            {
                "offset": offset + 1,
                "positive_cosine": float(positive.mean().item()),
                "shuffled_cosine": float(null.mean().item()),
                "cosine_margin": clustered_mean_ci(
                    margin,
                    groups,
                    seed + 101 * offset,
                ),
                "same_vs_shuffled_auc": float(roc_auc_score(labels, scores)),
                "support_survival": float(survival.mean().item()),
                "support_jaccard": float(jaccard.mean().item()),
                "code_nrmse": float(nrmse.mean().item()),
                "same_group_retrieval_at_1": float(retrieval.mean()),
            }
        )
    return rows


def collapse_diagnostics(codes: torch.Tensor) -> dict[str, float]:
    values = codes.float()
    active = values > 0
    variance = values.var(dim=0, unbiased=False)
    return {
        "mean_l0": float(active.sum(dim=-1).float().mean().item()),
        "active_dimension_fraction": float(
            (active.any(dim=0)).float().mean().item()
        ),
        "dead_dimension_fraction": float(
            (~active.any(dim=0)).float().mean().item()
        ),
        "variance_participation_dimension": float(
            variance.sum().square().div(variance.square().sum().clamp_min(1e-12)).item()
        ),
        "mean_feature_std": float(variance.sqrt().mean().item()),
    }


def select_probe_dimensions(
    values: torch.Tensor,
    development_indices: list[int],
    maximum: int = 4096,
) -> torch.Tensor:
    if values.shape[1] <= maximum:
        return values
    variance = values[development_indices].float().var(dim=0, unbiased=False)
    keep = torch.topk(variance, maximum).indices
    return values.index_select(1, keep)


def top_persistent_features(
    context: torch.Tensor,
    targets: torch.Tensor,
    metadata: list[dict[str, Any]],
    test_indices: list[int],
    count: int = 24,
    examples: int = 5,
) -> list[dict[str, Any]]:
    context_active = context > 0
    target_active = targets > 0
    context_count = context_active.sum(dim=0).float()
    frequency = context_active.float().mean(dim=0)
    survival = (
        (context_active[:, None, :] & target_active).sum(dim=0).float()
        / context_count[None, :].clamp_min(1)
    )
    mean_survival = survival.mean(dim=0)
    score = frequency.sqrt() * mean_survival
    valid = context_count >= 3
    score = score.masked_fill(~valid, -1)
    feature_ids = torch.topk(score, min(count, int(valid.sum().item()))).indices.tolist()
    rows = []
    for feature_id in feature_ids:
        activations = context[:, feature_id].float()
        example_positions = torch.topk(
            activations,
            min(examples, len(activations)),
        ).indices.tolist()
        rows.append(
            {
                "feature_id": feature_id,
                "context_frequency": float(frequency[feature_id].item()),
                "mean_survival": float(mean_survival[feature_id].item()),
                "survival_by_offset": [
                    float(value) for value in survival[:, feature_id].tolist()
                ],
                "mean_context_activation": float(activations.mean().item()),
                "top_examples": [
                    {
                        "window_index": test_indices[position],
                        "activation": float(activations[position].item()),
                        "metadata": metadata[test_indices[position]],
                    }
                    for position in example_positions
                    if activations[position] > 0
                ],
            }
        )
    return rows


def pca_embedding(values: torch.Tensor) -> torch.Tensor:
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=2, center=False, niter=3)
    return centered @ components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked-test evaluation of direct z0-to-z1...z9 persistence"
    )
    parser.add_argument("--activations", required=True)
    parser.add_argument("--persistent-checkpoint", required=True)
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
    proposed = load_model(args.persistent_checkpoint, device)
    baseline = load_model(args.baseline_checkpoint, device)
    if proposed.checkpoint["split"] != baseline.checkpoint["split"]:
        raise ValueError("proposed and baseline checkpoints must use the same split")
    if x.shape[1] != proposed.model.cfg.window_size:
        raise ValueError("activation/checkpoint window-size mismatch")
    split = proposed.checkpoint["split"]
    development_indices = list(split["train_indices"]) + list(
        split["validation_indices"]
    )
    test_indices = list(split["test_indices"])
    groups = np.asarray(
        [str(metadata[index].get(args.group_key, index)) for index in test_indices]
    )
    amp_dtype = (
        str(proposed.checkpoint["train_args"].get("amp_dtype", "bfloat16"))
        if device.type == "cuda"
        else "none"
    )
    proposed_outputs = collect_codes(
        proposed.model,
        x,
        test_indices,
        args.batch_size,
        device,
        amp_dtype,
    )
    baseline_outputs = collect_codes(
        baseline.model,
        x,
        test_indices,
        args.batch_size,
        device,
        amp_dtype,
    )
    proposed_context = proposed_outputs["context"]
    baseline_context = baseline_outputs["context"]
    assert isinstance(proposed_context, torch.Tensor)
    assert isinstance(baseline_context, torch.Tensor)
    proposed_test_context = proposed_context[test_indices]
    baseline_test_context = baseline_context[test_indices]
    proposed_targets = proposed_outputs["test_targets"]
    baseline_targets = baseline_outputs["test_targets"]
    assert isinstance(proposed_targets, torch.Tensor)
    assert isinstance(baseline_targets, torch.Tensor)

    representations = {
        "persistent_z0": select_probe_dimensions(
            proposed_context, development_indices
        ),
        "standard_sae_z0": select_probe_dimensions(
            baseline_context, development_indices
        ),
        "raw_h0": select_probe_dimensions(x[:, 0], development_indices),
    }
    probes = {
        name: fit_probe(
            values,
            metadata,
            development_indices,
            test_indices,
            args.label_key,
            args.group_key,
            args.seed,
        )[0]
        for name, values in representations.items()
    }
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
    proposed_offsets = offset_metrics(
        proposed_test_context,
        proposed_targets,
        groups,
        device,
        args.seed,
    )
    baseline_offsets = offset_metrics(
        baseline_test_context,
        baseline_targets,
        groups,
        device,
        args.seed,
    )
    features = top_persistent_features(
        proposed_test_context,
        proposed_targets,
        metadata,
        test_indices,
    )
    feature_ids = [row["feature_id"] for row in features]
    report = {
        "claim": (
            "A sparse state is persistent when the code present at token 0 "
            "matches each token 1..9 separately, exceeds a different-window "
            "null, and remains useful on locked problem groups."
        ),
        "architecture": {
            "context": "z0 = online_SAE(h0)",
            "targets": "zj = stopgrad(EMA_SAE(hj)), j=1..9",
            "target_aggregation": "none",
            "predictor": None,
            "window_size": proposed.model.cfg.window_size,
        },
        "split": {
            "group_key": args.group_key,
            "n_development_windows": len(development_indices),
            "n_locked_test_windows": len(test_indices),
            "n_locked_test_groups": len(np.unique(groups)),
        },
        "locked_test_offset_curve": {
            "persistent": proposed_offsets,
            "standard_sae": baseline_offsets,
        },
        "reconstruction_fvu": {
            "persistent": proposed_outputs["reconstruction_fvu"],
            "standard_sae": baseline_outputs["reconstruction_fvu"],
        },
        "locked_test_state_probes": probes,
        "paraphrase_and_semantic_invariance": invariance,
        "collapse_diagnostics": {
            "persistent_z0": collapse_diagnostics(proposed_test_context),
            "standard_sae_z0": collapse_diagnostics(baseline_test_context),
        },
        "top_persistent_features": features,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "persistent_report.json", report)
    with (output_dir / "persistent_offset_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "method",
            "offset",
            "positive_cosine",
            "shuffled_cosine",
            "cosine_margin",
            "cosine_margin_ci95_low",
            "cosine_margin_ci95_high",
            "same_vs_shuffled_auc",
            "support_survival",
            "support_jaccard",
            "code_nrmse",
            "same_group_retrieval_at_1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in (
            ("persistent", proposed_offsets),
            ("standard_sae", baseline_offsets),
        ):
            for row in rows:
                margin = row["cosine_margin"]
                writer.writerow(
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key != "cosine_margin"
                        },
                        "method": method,
                        "cosine_margin": margin["mean"],
                        "cosine_margin_ci95_low": margin["ci95_low"],
                        "cosine_margin_ci95_high": margin["ci95_high"],
                    }
                )
    torch.save(
        {
            "embedding": pca_embedding(proposed_test_context),
            "labels": [str(metadata[index][args.label_key]) for index in test_indices],
            "feature_ids": feature_ids,
            "feature_activations": proposed_test_context[:, feature_ids],
            "metadata": [metadata[index] for index in test_indices],
        },
        output_dir / "persistent_visualization.pt",
    )
    print(f"wrote locked-test report to {output_dir / 'persistent_report.json'}")


if __name__ == "__main__":
    main()
