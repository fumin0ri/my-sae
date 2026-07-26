from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .io import torch_load, write_json


@dataclass
class Centering:
    mean: torch.Tensor
    relative_position_effect: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean[None, None, :] - self.relative_position_effect[None, :, :]


def fit_centering(x: torch.Tensor, remove_relative_position: bool = True) -> Centering:
    mean = x.mean(dim=(0, 1))
    if remove_relative_position:
        position_effect = x.mean(dim=0) - mean[None, :]
    else:
        position_effect = torch.zeros_like(x[0])
    return Centering(mean=mean, relative_position_effect=position_effect)


def random_effect_covariances(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate shared-state and token-specific covariance.

    Model: y[w,t] = z[w] + eps[w,t], with eps conditionally independent over t.
    The unbiased estimators are:
      Sigma_eps = pooled within-window covariance
      Sigma_z   = Cov(window mean) - Sigma_eps / T
    """
    n, t, _ = y.shape
    if n < 3 or t < 2:
        raise ValueError("need at least 3 windows and 2 token positions")
    window_mean = y.mean(dim=1)
    window_mean = window_mean - window_mean.mean(dim=0, keepdim=True)
    eps = y - y.mean(dim=1, keepdim=True)
    sigma_eps = torch.einsum("ntd,nte->de", eps, eps) / (n * (t - 1))
    sigma_mean = window_mean.T @ window_mean / (n - 1)
    sigma_z = sigma_mean - sigma_eps / t
    return (sigma_z + sigma_z.T) / 2, (sigma_eps + sigma_eps.T) / 2


def fit_generalized_subspace(
    y: torch.Tensor,
    rank: int,
    ridge: float = 1e-3,
    pre_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if pre_rank is not None and pre_rank > 0 and y.shape[-1] > pre_rank:
        flat = y.reshape(-1, y.shape[-1])
        q = min(pre_rank, flat.shape[0] - 1, flat.shape[1] - 1)
        if q < rank:
            raise ValueError(
                f"pre_rank={pre_rank} leaves q={q}, smaller than rank={rank}"
            )
        _, singular_values, projection = torch.pca_lowrank(
            flat,
            q=q,
            center=False,
            niter=4,
        )
        reduced_basis, eigenvalues, stats = fit_generalized_subspace(
            y @ projection,
            rank,
            ridge,
            pre_rank=None,
        )
        basis, _ = torch.linalg.qr(
            projection @ reduced_basis,
            mode="reduced",
        )
        stats.update(
            {
                "pre_rank": float(q),
                "preprojection_variance_fraction": float(
                    singular_values.square().sum().item()
                    / flat.square().sum().clamp_min(1e-12).item()
                ),
            }
        )
        return basis, eigenvalues, stats
    sigma_z, sigma_eps = random_effect_covariances(y)
    d = y.shape[-1]
    ridge_abs = ridge * float(torch.trace(sigma_eps).item()) / d
    noise = sigma_eps + ridge_abs * torch.eye(d, device=y.device, dtype=y.dtype)
    chol = torch.linalg.cholesky(noise)
    left = torch.linalg.solve_triangular(chol, sigma_z, upper=False)
    whitened = torch.linalg.solve_triangular(
        chol, left.T, upper=False
    ).T
    whitened = (whitened + whitened.T) / 2
    eigenvalues, whitened_vectors = torch.linalg.eigh(whitened)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    whitened_vectors = whitened_vectors[:, order]
    positive = int((eigenvalues > 0).sum().item())
    k = min(rank, positive)
    if k == 0:
        raise RuntimeError("no positive shared-state eigenvalues found")
    generalized = torch.linalg.solve_triangular(
        chol.T, whitened_vectors[:, :k], upper=True
    )
    # Causal projections require an ordinary Euclidean orthoprojector.
    basis, _ = torch.linalg.qr(generalized, mode="reduced")
    stats = {
        "ridge_absolute": ridge_abs,
        "positive_generalized_eigenvalues": positive,
        "largest_generalized_eigenvalue": float(eigenvalues[0].item()),
        "smallest_kept_generalized_eigenvalue": float(eigenvalues[k - 1].item()),
    }
    return basis, eigenvalues[:k], stats


def common_codes(y: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return y.mean(dim=1) @ basis


def evaluate_basis(y: torch.Tensor, basis: torch.Tensor) -> dict[str, Any]:
    projected = y @ basis
    n, t, _ = projected.shape
    means = projected.mean(dim=1)
    eps = projected - means[:, None, :]
    eps_var = eps.square().sum(dim=(0, 1)) / (n * (t - 1))
    mean_var = means.var(dim=0, unbiased=True)
    shared_var = mean_var - eps_var / t
    total_var = shared_var + eps_var
    icc = shared_var / total_var.clamp_min(1e-12)
    return {
        "per_component_icc": icc.cpu().tolist(),
        "mean_icc": float(icc.mean().item()),
        "positive_shared_fraction": float((shared_var > 0).float().mean().item()),
        "shared_variance": shared_var.cpu().tolist(),
        "token_specific_variance": eps_var.cpu().tolist(),
    }


def permutation_test(
    y: torch.Tensor,
    basis: torch.Tensor,
    observed_mean_icc: float,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    if permutations <= 0:
        return {}
    generator = torch.Generator(device=y.device).manual_seed(seed)
    null: list[float] = []
    for _ in range(permutations):
        shuffled = torch.empty_like(y)
        for token_position in range(y.shape[1]):
            perm = torch.randperm(y.shape[0], generator=generator, device=y.device)
            shuffled[:, token_position] = y[perm, token_position]
        null.append(evaluate_basis(shuffled, basis)["mean_icc"])
    p = (1 + sum(v >= observed_mean_icc for v in null)) / (permutations + 1)
    return {
        "p_value": p,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "permutations": permutations,
    }


def principal_angle_similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    singular = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(singular))
    return {
        "cosines": singular.cpu().tolist(),
        "angles_degrees": angles.cpu().tolist(),
        "mean_squared_cosine": float(singular.square().mean().item()),
    }


def _probe(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: list[Any],
    test_labels: list[Any],
) -> dict[str, Any]:
    encoder = LabelEncoder().fit(train_labels + test_labels)
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    scaler = StandardScaler().fit(train_features)
    x_train = scaler.transform(train_features)
    x_test = scaler.transform(test_features)
    if len(encoder.classes_) <= 20:
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=0,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return {
            "task": "classification",
            "classes": [str(x) for x in encoder.classes_],
            "accuracy": float(accuracy_score(y_test, pred)),
            "majority_baseline": float(
                np.max(np.bincount(y_train)) / max(1, len(y_train))
            ),
        }
    # Many unique numeric values are more plausibly a regression target.
    try:
        y_train_float = np.asarray(train_labels, dtype=float)
        y_test_float = np.asarray(test_labels, dtype=float)
    except (TypeError, ValueError):
        return {"task": "skipped", "reason": "more than 20 non-numeric classes"}
    model = Ridge(alpha=1.0).fit(x_train, y_train_float)
    return {
        "task": "regression",
        "r2": float(r2_score(y_test_float, model.predict(x_test))),
    }


def rank_matched_pca_features(
    feature: torch.Tensor,
    train_idx: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    train = feature[train_idx]
    center = train.mean(dim=0)
    _, _, vh = torch.linalg.svd(train - center, full_matrices=False)
    k = min(rank, vh.shape[0])
    return (feature - center) @ vh[:k].T


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit a token-shared residual subspace")
    p.add_argument("--activations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--label-key")
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep-relative-position", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    bundle = torch_load(args.activations)
    x = bundle["activations"].to(torch.float32)
    n = x.shape[0]
    if n < 10:
        raise SystemExit("Use at least 10 windows; hundreds or thousands are recommended.")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(n, generator=generator)
    n_train = max(3, min(n - 3, round(n * args.train_fraction)))
    train_idx, test_idx = order[:n_train], order[n_train:]
    centering = fit_centering(
        x[train_idx], remove_relative_position=not args.keep_relative_position
    )
    y_train_cpu = centering.transform(x[train_idx])
    y_test_cpu = centering.transform(x[test_idx])
    compute_device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    y_train = y_train_cpu.to(compute_device)
    basis, eigenvalues, fit_stats = fit_generalized_subspace(
        y_train, args.rank, args.ridge
    )
    train_eval = evaluate_basis(y_train, basis)
    test_eval = evaluate_basis(y_test_cpu.to(compute_device), basis)
    permutation = permutation_test(
        y_test_cpu.to(compute_device),
        basis,
        test_eval["mean_icc"],
        args.permutations,
        args.seed + 1,
    )

    # Independent-half stability: a genuine subspace should recur.
    half = len(train_idx) // 2
    stability: dict[str, Any] = {}
    if half >= 3 and len(train_idx) - half >= 3:
        try:
            b1, _, _ = fit_generalized_subspace(
                y_train[:half], basis.shape[1], args.ridge
            )
            b2, _, _ = fit_generalized_subspace(
                y_train[half:], basis.shape[1], args.ridge
            )
            k = min(b1.shape[1], b2.shape[1])
            stability = principal_angle_similarity(b1[:, :k], b2[:, :k])
        except (RuntimeError, torch.linalg.LinAlgError) as exc:
            stability = {"error": str(exc)}

    basis_cpu = basis.cpu()
    y_all = centering.transform(x)
    codes = common_codes(y_all, basis_cpu)
    raw_mean = y_all.mean(dim=1)
    last_token = y_all[:, -1]
    probe_report: dict[str, Any] = {}
    if args.label_key:
        metadata = bundle["metadata"]
        missing = [i for i, m in enumerate(metadata) if args.label_key not in m]
        if missing:
            raise KeyError(f"label key missing in metadata rows, first: {missing[:5]}")
        labels = [m[args.label_key] for m in metadata]
        tr, te = train_idx.numpy(), test_idx.numpy()
        mean_pca = rank_matched_pca_features(raw_mean, train_idx, basis.shape[1])
        last_pca = rank_matched_pca_features(last_token, train_idx, basis.shape[1])
        for name, feature in {
            "shared_subspace": codes,
            "rank_matched_mean_pca": mean_pca,
            "rank_matched_last_token_pca": last_pca,
            "raw_window_mean": raw_mean,
            "last_token": last_token,
        }.items():
            probe_report[name] = _probe(
                feature[tr].numpy(),
                feature[te].numpy(),
                [labels[i] for i in tr],
                [labels[i] for i in te],
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "basis": basis_cpu,
            "generalized_eigenvalues": eigenvalues.cpu(),
            "mean": centering.mean,
            "relative_position_effect": centering.relative_position_effect,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "source_config": bundle.get("config", {}),
            "fit_config": vars(args),
        },
        out_dir / "subspace.pt",
    )
    torch.save(
        {
            "codes": codes,
            "raw_window_mean": raw_mean,
            "last_token": last_token,
            "metadata": bundle["metadata"],
        },
        out_dir / "codes.pt",
    )
    report = {
        "shape": list(x.shape),
        "kept_rank": basis.shape[1],
        "fit": fit_stats,
        "train": train_eval,
        "test": test_eval,
        "permutation_control": permutation,
        "split_half_stability": stability,
        "probe": probe_report,
        "interpretation_gate": {
            "statistical_signal": (
                test_eval["mean_icc"] > permutation.get("null_mean", -math.inf)
            ),
            "causal_test_still_required": True,
        },
    }
    write_json(out_dir / "report.json", report)
    print(f"saved subspace, codes, and report to {out_dir}")


if __name__ == "__main__":
    main()
