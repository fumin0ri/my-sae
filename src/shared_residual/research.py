from __future__ import annotations

import argparse
import importlib.metadata as package_metadata
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .io import torch_load, write_json, write_jsonl
from .subspace import (
    _probe,
    common_codes,
    evaluate_basis,
    fit_centering,
    fit_generalized_subspace,
    permutation_test,
    principal_angle_similarity,
    rank_matched_pca_features,
)


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "scikit-learn",
        "matplotlib",
    ):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return list(dict.fromkeys(parsed))


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return list(dict.fromkeys(parsed))


def group_labels(
    metadata: list[dict[str, Any]],
    group_key: str,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        if group_key not in row:
            raise KeyError(f"metadata has no group key {group_key!r}")
        if label_key not in row:
            raise KeyError(f"metadata has no label key {label_key!r}")
        labels_by_group[str(row[group_key])].add(str(row[label_key]))
    mixed = {group: labels for group, labels in labels_by_group.items() if len(labels) != 1}
    if mixed:
        examples = list(mixed.items())[:3]
        raise ValueError(
            f"each group must have one label for stratification; mixed examples: {examples}"
        )
    groups = np.asarray(sorted(labels_by_group))
    labels = np.asarray([next(iter(labels_by_group[group])) for group in groups])
    return groups, labels


def indices_for_groups(
    metadata: list[dict[str, Any]],
    group_key: str,
    groups: np.ndarray | list[str],
) -> torch.Tensor:
    selected = set(str(group) for group in groups)
    return torch.tensor(
        [
            index
            for index, row in enumerate(metadata)
            if str(row[group_key]) in selected
        ],
        dtype=torch.long,
    )


def stratified_group_split(
    groups: np.ndarray,
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            groups,
            test_size=test_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        return train_test_split(
            groups,
            test_size=test_fraction,
            random_state=seed,
            stratify=None,
        )


def probe_comparison(
    y: torch.Tensor,
    basis: torch.Tensor,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    labels: list[Any],
) -> dict[str, Any]:
    shared = common_codes(y, basis)
    raw_mean = y.mean(dim=1)
    last_token = y[:, -1]
    mean_pca = rank_matched_pca_features(raw_mean, train_idx, basis.shape[1])
    last_pca = rank_matched_pca_features(last_token, train_idx, basis.shape[1])
    train_labels = [labels[index] for index in train_idx.tolist()]
    eval_labels = [labels[index] for index in eval_idx.tolist()]
    report: dict[str, Any] = {}
    for name, features in {
        "shared_subspace": shared,
        "rank_matched_mean_pca": mean_pca,
        "rank_matched_last_token_pca": last_pca,
        "raw_window_mean": raw_mean,
        "last_token": last_token,
    }.items():
        report[name] = _probe(
            features[train_idx].numpy(),
            features[eval_idx].numpy(),
            train_labels,
            eval_labels,
        )
    return report


def evaluate_candidate(
    y: torch.Tensor,
    basis: torch.Tensor,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    labels: list[Any],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    evaluation = evaluate_basis(y[eval_idx], basis)
    null = permutation_test(
        y[eval_idx],
        basis,
        evaluation["mean_icc"],
        permutations,
        seed,
    )
    probes = probe_comparison(y, basis, train_idx, eval_idx, labels)
    return {
        "mean_icc": evaluation["mean_icc"],
        "per_component_icc": evaluation["per_component_icc"],
        "positive_shared_fraction": evaluation["positive_shared_fraction"],
        "null_mean_icc": null.get("null_mean"),
        "permutation_p_value": null.get("p_value"),
        "probe": probes,
    }


def bootstrap_group_icc(
    y: torch.Tensor,
    basis: torch.Tensor,
    metadata: list[dict[str, Any]],
    test_idx: torch.Tensor,
    group_key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in test_idx.tolist():
        by_group[str(metadata[index][group_key])].append(index)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = [
            index
            for group in sampled
            for index in by_group[str(group)]
        ]
        values.append(
            evaluate_basis(y[torch.tensor(indices)], basis)["mean_icc"]
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "samples": samples,
    }


def split_half_stability(
    y: torch.Tensor,
    metadata: list[dict[str, Any]],
    development_idx: torch.Tensor,
    group_key: str,
    rank: int,
    ridge: float,
    pre_rank: int | None,
    seed: int,
) -> dict[str, Any]:
    groups = sorted(
        {str(metadata[index][group_key]) for index in development_idx.tolist()}
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    midpoint = len(groups) // 2
    first = indices_for_groups(metadata, group_key, groups[:midpoint])
    second = indices_for_groups(metadata, group_key, groups[midpoint:])
    try:
        seed_torch(seed)
        first_basis, _, _ = fit_generalized_subspace(
            y[first.to(y.device)],
            rank,
            ridge,
            pre_rank,
        )
        seed_torch(seed + 1)
        second_basis, _, _ = fit_generalized_subspace(
            y[second.to(y.device)],
            rank,
            ridge,
            pre_rank,
        )
        kept = min(first_basis.shape[1], second_basis.shape[1])
        return principal_angle_similarity(
            first_basis[:, :kept],
            second_basis[:, :kept],
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        return {"error": str(exc)}


def summarize_locked(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_icc": record["test"]["mean_icc"],
        "icc_ci95": record["icc_bootstrap"],
        "null_mean_icc": record["test"]["null_mean_icc"],
        "permutation_p_value": record["test"]["permutation_p_value"],
        "shared_probe_accuracy": record["test"]["probe"]["shared_subspace"].get(
            "accuracy"
        ),
        "mean_pca_probe_accuracy": record["test"]["probe"][
            "rank_matched_mean_pca"
        ].get("accuracy"),
        "last_pca_probe_accuracy": record["test"]["probe"][
            "rank_matched_last_token_pca"
        ].get("accuracy"),
        "split_half_mean_squared_cosine": record["split_half_stability"].get(
            "mean_squared_cosine"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nested grouped selection across layer, window width, rank, and ridge; "
            "then one locked outer test"
        )
    )
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sizes", type=parse_int_list, default=[4, 8, 10, 16])
    parser.add_argument("--ranks", type=parse_int_list, default=[2, 4, 8, 16])
    parser.add_argument("--ridges", type=parse_float_list, default=[1e-3])
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--label-key", default="state")
    parser.add_argument("--outer-test-fraction", type=float, default=0.2)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.25)
    parser.add_argument("--outer-seed", type=int, default=1729)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--pre-rank",
        type=int,
        default=512,
        help="PCA width before covariance eigendecomposition; 0 disables",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    activation_files = sorted(Path(args.activations_dir).glob("layer-*.pt"))
    if not activation_files:
        raise FileNotFoundError(
            f"no layer-*.pt files in {args.activations_dir}"
        )
    first_bundle = torch_load(activation_files[0])
    metadata = first_bundle["metadata"]
    labels = [row[args.label_key] for row in metadata]
    groups, group_targets = group_labels(
        metadata,
        args.group_key,
        args.label_key,
    )
    development_groups, test_groups = stratified_group_split(
        groups,
        group_targets,
        args.outer_test_fraction,
        args.outer_seed,
    )
    development_idx = indices_for_groups(
        metadata,
        args.group_key,
        development_groups,
    )
    test_idx = indices_for_groups(metadata, args.group_key, test_groups)
    label_by_group = {
        group: label for group, label in zip(groups.tolist(), group_targets.tolist())
    }
    development_targets = np.asarray(
        [label_by_group[str(group)] for group in development_groups]
    )

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    candidates: list[dict[str, Any]] = []
    activation_configs: list[dict[str, Any]] = []
    layer_files: dict[int, Path] = {}
    max_requested_rank = max(args.ranks)
    for activation_file in activation_files:
        bundle = torch_load(activation_file)
        activation_configs.append(bundle.get("config", {}))
        if bundle["metadata"] != metadata:
            raise ValueError(f"metadata mismatch in {activation_file}")
        x_full = bundle["activations"].float()
        layer = int(bundle["config"]["layer"])
        layer_files[layer] = activation_file
        max_width = x_full.shape[1]
        for width in args.window_sizes:
            if width > max_width:
                continue
            x = x_full[:, -width:, :]
            for seed in args.seeds:
                train_groups, validation_groups = stratified_group_split(
                    development_groups,
                    development_targets,
                    args.inner_validation_fraction,
                    seed,
                )
                train_idx = indices_for_groups(
                    metadata,
                    args.group_key,
                    train_groups,
                )
                validation_idx = indices_for_groups(
                    metadata,
                    args.group_key,
                    validation_groups,
                )
                centering = fit_centering(x[train_idx])
                y_cpu = centering.transform(x)
                y = y_cpu.to(device)
                for ridge_index, ridge in enumerate(args.ridges):
                    try:
                        seed_torch(
                            args.outer_seed
                            + seed
                            + layer * 10_000
                            + width * 100
                            + ridge_index
                        )
                        max_basis, eigenvalues, fit_stats = (
                            fit_generalized_subspace(
                                y[train_idx.to(device)],
                                max_requested_rank,
                                ridge,
                                args.pre_rank,
                            )
                        )
                    except (RuntimeError, torch.linalg.LinAlgError):
                        continue
                    for rank in args.ranks:
                        if rank > max_basis.shape[1]:
                            continue
                        basis = max_basis[:, :rank]
                        metrics = evaluate_candidate(
                            y_cpu,
                            basis.cpu(),
                            train_idx,
                            validation_idx,
                            labels,
                            args.permutations,
                            seed + layer * 10_000 + width * 100,
                        )
                        candidates.append(
                            {
                                "layer": layer,
                                "window_size": width,
                                "rank": rank,
                                "ridge": ridge,
                                "seed": seed,
                                "train_groups": len(train_groups),
                                "validation_groups": len(validation_groups),
                                "validation": metrics,
                                "generalized_eigenvalues": eigenvalues[
                                    :rank
                                ].cpu().tolist(),
                                "fit": fit_stats,
                            }
                        )
    if not candidates:
        raise RuntimeError("no candidate subspace could be fitted")

    grouped_scores: dict[tuple[int, int, int, float], list[float]] = defaultdict(list)
    grouped_icc: dict[tuple[int, int, int, float], list[float]] = defaultdict(list)
    for row in candidates:
        key = (
            row["layer"],
            row["window_size"],
            row["rank"],
            row["ridge"],
        )
        observed = float(row["validation"]["mean_icc"])
        null = row["validation"]["null_mean_icc"]
        grouped_icc[key].append(observed)
        grouped_scores[key].append(
            observed - float(null) if null is not None else observed
        )
    selected_key = max(
        grouped_scores,
        key=lambda key: (
            float(np.mean(grouped_scores[key])),
            -key[2],
        ),
    )
    selected = {
        "layer": selected_key[0],
        "window_size": selected_key[1],
        "rank": selected_key[2],
        "ridge": selected_key[3],
        "mean_inner_validation_icc": float(np.mean(grouped_icc[selected_key])),
        "mean_inner_validation_null_gap": float(
            np.mean(grouped_scores[selected_key])
        ),
        "std_inner_validation_null_gap": float(
            np.std(grouped_scores[selected_key], ddof=1)
            if len(grouped_scores[selected_key]) > 1
            else 0.0
        ),
    }

    selected_file = layer_files[selected["layer"]]
    selected_bundle = torch_load(selected_file)
    x_selected = selected_bundle["activations"][
        :, -selected["window_size"] :, :
    ].float()
    development_centering = fit_centering(x_selected[development_idx])
    y_development_cpu = development_centering.transform(x_selected)
    seed_torch(args.outer_seed + 50_000)
    development_basis, development_eigenvalues, fit_stats = (
        fit_generalized_subspace(
            y_development_cpu[development_idx].to(device),
            selected["rank"],
            selected["ridge"],
            args.pre_rank,
        )
    )
    development_basis_cpu = development_basis.cpu()
    if development_basis_cpu.shape[1] < selected["rank"]:
        raise RuntimeError(
            "selected rank was not positive on the full development split"
        )
    locked_test = evaluate_candidate(
        y_development_cpu,
        development_basis_cpu,
        development_idx,
        test_idx,
        labels,
        max(args.permutations, 1000),
        args.outer_seed + 1,
    )
    bootstrap = bootstrap_group_icc(
        y_development_cpu,
        development_basis_cpu,
        metadata,
        test_idx,
        args.group_key,
        args.bootstrap_samples,
        args.outer_seed + 2,
    )
    stability = split_half_stability(
        y_development_cpu.to(device),
        metadata,
        development_idx.to(device="cpu"),
        args.group_key,
        selected["rank"],
        selected["ridge"],
        args.pre_rank,
        args.outer_seed + 3,
    )
    locked_record = {
        "selected": selected,
        "development_groups": len(development_groups),
        "locked_test_groups": len(test_groups),
        "test": locked_test,
        "icc_bootstrap": bootstrap,
        "split_half_stability": stability,
        "generalized_eigenvalues": development_eigenvalues.cpu().tolist(),
        "fit": fit_stats,
    }

    final_centering = fit_centering(x_selected)
    y_all = final_centering.transform(x_selected)
    seed_torch(args.outer_seed + 60_000)
    final_basis, final_eigenvalues, final_fit = fit_generalized_subspace(
        y_all.to(device),
        selected["rank"],
        selected["ridge"],
        args.pre_rank,
    )
    if final_basis.shape[1] < selected["rank"]:
        raise RuntimeError("selected rank was not positive in the final all-data fit")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_json(output_dir / "locked_test.json", locked_record)
    write_json(
        output_dir / "selection.json",
        {
            "selected": selected,
            "locked_test_summary": summarize_locked(locked_record),
            "outer_split": {
                "seed": args.outer_seed,
                "development_group_ids": development_groups.tolist(),
                "locked_test_group_ids": test_groups.tolist(),
            },
            "method": {
                "selection_data": "inner grouped validation only",
                "test_data": "single untouched outer grouped split",
                "selection_metric": "mean held-out ICC minus position-shuffled null ICC",
            },
        },
    )
    write_json(
        output_dir / "environment.json",
        {
            **environment_manifest(),
            "command_arguments": vars(args),
            "activation_sources": activation_configs,
        },
    )
    torch.save(
        {
            "basis": final_basis.cpu(),
            "generalized_eigenvalues": final_eigenvalues.cpu(),
            "mean": final_centering.mean,
            "relative_position_effect": final_centering.relative_position_effect,
            "source_config": selected_bundle["config"],
            "fit_config": {
                **selected,
                "group_key": args.group_key,
                "label_key": args.label_key,
                "fit_scope": "all data after locked evaluation",
                "fit_stats": final_fit,
            },
        },
        output_dir / "final_subspace.pt",
    )
    torch.save(
        {
            "codes": common_codes(y_all, final_basis.cpu()),
            "metadata": metadata,
            "selected": selected,
        },
        output_dir / "final_codes.pt",
    )
    print(f"selected {selected}")
    print(f"locked test: {summarize_locked(locked_record)}")
    print(f"saved research analysis to {output_dir}")


if __name__ == "__main__":
    main()
