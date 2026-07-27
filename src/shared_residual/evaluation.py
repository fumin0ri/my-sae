from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def different_group_permutation(groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    for _ in range(max(10, len(groups) * 2)):
        bad = groups[order] == groups
        if not bad.any():
            return order
        order[bad] = np.roll(order[bad], 1)
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
        indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in chosen]
        )
        bootstrap.append(float(values[indices].mean()))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": samples,
    }


def group_bootstrap_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled]
        )
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
    classes = set(encoder.classes_)
    if any(label not in classes for label in labels[test_indices]):
        raise ValueError("locked test contains a label absent from development data")
    test_y = encoder.transform(labels[test_indices])
    scaler = StandardScaler()
    train_x = scaler.fit_transform(values[train_indices].float().numpy())
    test_x = scaler.transform(values[test_indices].float().numpy())
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
        "balanced_accuracy": float(
            balanced_accuracy_score(test_y, prediction)
        ),
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
    same_problem = [
        float(
            F.cosine_similarity(
                local[left : left + 1].float(),
                local[right : right + 1].float(),
                dim=-1,
            ).item()
        )
        for members in groups.values()
        for left, right in itertools.combinations(members, 2)
    ]

    rng = np.random.default_rng(seed)
    same_state: list[float] = []
    different_state: list[float] = []
    for left in range(len(rows)):
        independent = [
            right
            for right in range(len(rows))
            if rows[right].get(group_key) != rows[left].get(group_key)
        ]
        for candidates, output in (
            (
                [
                    right
                    for right in independent
                    if rows[right].get(label_key) == rows[left].get(label_key)
                ],
                same_state,
            ),
            (
                [
                    right
                    for right in independent
                    if rows[right].get(label_key) != rows[left].get(label_key)
                ],
                different_state,
            ),
        ):
            if candidates:
                right = int(rng.choice(candidates))
                output.append(
                    float(
                        F.cosine_similarity(
                            local[left : left + 1].float(),
                            local[right : right + 1].float(),
                            dim=-1,
                        ).item()
                    )
                )

    def summarize(group: list[float]) -> dict[str, float | int]:
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


def collapse_diagnostics(codes: torch.Tensor) -> dict[str, float]:
    values = codes.float()
    active = values > 0
    variance = values.var(dim=0, unbiased=False)
    return {
        "mean_l0": float(active.sum(dim=-1).float().mean().item()),
        "active_dimension_fraction": float(
            active.any(dim=0).float().mean().item()
        ),
        "dead_dimension_fraction": float(
            (~active.any(dim=0)).float().mean().item()
        ),
        "variance_participation_dimension": float(
            variance.sum()
            .square()
            .div(variance.square().sum().clamp_min(1e-12))
            .item()
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


def pca_embedding(values: torch.Tensor) -> torch.Tensor:
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(
        centered,
        q=2,
        center=False,
        niter=3,
    )
    return centered @ components
