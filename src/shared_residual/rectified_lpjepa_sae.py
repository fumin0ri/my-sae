from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from .activation_store import (
    RandomViewPairShardBatches,
    load_activation_manifest,
    manifest_fingerprint,
    validation_view_pair_batches,
)
from .group_sae import topk_relu
from .io import write_json
from .training import autocast_context, configure_accelerator, cosine_learning_rate


ARCHITECTURE_ID = "high_low_rectified_lpjepa_sae_v1"
SUPPORTED_RGG_P = (1.0, 2.0)


def unit_variance_generalized_gaussian_sigma(p: float) -> float:
    """Scale sigma for Var[GN_p(0, sigma)] = 1 in the paper's convention."""
    if p not in SUPPORTED_RGG_P:
        raise ValueError(f"p must be one of {SUPPORTED_RGG_P}")
    return math.sqrt(math.gamma(1.0 / p)) / (
        p ** (1.0 / p) * math.sqrt(math.gamma(3.0 / p))
    )


def rgg_mean_for_active_fraction(
    p: float,
    active_fraction: float,
    sigma: float,
) -> float:
    """Choose mu so P[GN_p(mu, sigma) > 0] equals active_fraction."""
    if not 0.0 < active_fraction < 1.0:
        raise ValueError("active_fraction must lie strictly between zero and one")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if p == 1.0:
        if active_fraction <= 0.5:
            return sigma * math.log(2.0 * active_fraction)
        return -sigma * math.log(2.0 * (1.0 - active_fraction))
    if p == 2.0:
        probability = torch.tensor(active_fraction, dtype=torch.float64)
        return float(
            sigma
            * torch.distributions.Normal(0.0, 1.0).icdf(probability).item()
        )
    raise ValueError(f"p must be one of {SUPPORTED_RGG_P}")


def sample_rectified_generalized_gaussian(
    shape: tuple[int, ...],
    *,
    p: float,
    mu: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ReLU(GN_p(mu, sigma)) for p=1 (Laplace) or p=2 (Gaussian)."""
    if p == 1.0:
        uniform = torch.rand(shape, device=device, dtype=torch.float32) - 0.5
        noise = -sigma * uniform.sign() * torch.log1p(-2.0 * uniform.abs())
    elif p == 2.0:
        noise = sigma * torch.randn(shape, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"p must be one of {SUPPORTED_RGG_P}")
    return torch.relu(noise.add(mu)).to(dtype=dtype)


@dataclass
class RectifiedLpJEPAConfig:
    d_in: int
    d_sae: int = 2048
    low_k: int = 32
    max_span_length: int = 10
    ema_decay: float = 0.996
    high_fraction: float = 0.2
    high_reconstruction_weight: float = 0.1
    rgg_p: float = 1.0
    target_active_fraction: float = 0.025
    target_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.d_in < 1 or self.d_sae < 2:
            raise ValueError("d_in must be positive and d_sae must be at least two")
        if self.max_span_length < 2:
            raise ValueError("max_span_length must be at least two")
        if not 0.0 < self.high_fraction < 1.0:
            raise ValueError("high_fraction must lie strictly between zero and one")
        if not 0.0 <= self.high_reconstruction_weight <= 1.0:
            raise ValueError("high_reconstruction_weight must lie in [0, 1]")
        if self.d_high < 1 or self.d_low < 1:
            raise ValueError("both high and low dictionaries must be non-empty")
        if not 1 <= self.low_k <= self.d_low:
            raise ValueError("low_k must lie in [1, d_low]")
        if self.rgg_p not in SUPPORTED_RGG_P:
            raise ValueError(f"rgg_p must be one of {SUPPORTED_RGG_P}")
        if not 0.0 < self.target_active_fraction < 1.0:
            raise ValueError("target_active_fraction must lie in (0, 1)")
        if self.target_sigma < 0:
            raise ValueError("target_sigma must be zero (automatic) or positive")

    @property
    def d_high(self) -> int:
        return max(1, min(self.d_sae - 1, round(self.d_sae * self.high_fraction)))

    @property
    def d_low(self) -> int:
        return self.d_sae - self.d_high

    @property
    def resolved_target_sigma(self) -> float:
        return self.target_sigma or unit_variance_generalized_gaussian_sigma(
            self.rgg_p
        )

    @property
    def target_mu(self) -> float:
        return rgg_mean_for_active_fraction(
            self.rgg_p,
            self.target_active_fraction,
            self.resolved_target_sigma,
        )

    @property
    def expected_high_l0(self) -> float:
        return self.d_high * self.target_active_fraction


class HierarchicalRectifiedEncoder(nn.Module):
    """Shifted-ReLU high codes and an independent Top-K low dictionary."""

    def __init__(self, cfg: RectifiedLpJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.d_in, cfg.d_sae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = self.linear(x)
        high = torch.relu(dense[..., : self.cfg.d_high])
        low = topk_relu(dense[..., self.cfg.d_high :], self.cfg.low_k)
        return torch.cat((high, low), dim=-1)


class RectifiedLpJEPASAE(nn.Module):
    """High/low SAE whose high group is view-invariant and RGG-distributed."""

    def __init__(self, cfg: RectifiedLpJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("pre_scale", torch.ones(()))
        self.encoder = HierarchicalRectifiedEncoder(cfg)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))
        self.ema_encoder = copy.deepcopy(self.encoder)
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad_(False)
        self.ema_decoder = nn.Parameter(
            self.decoder.detach().clone(), requires_grad=False
        )
        self.register_buffer("ema_pre_bias", self.pre_bias.detach().clone())

    @torch.no_grad()
    def initialize_from_statistics(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor | float,
    ) -> None:
        if mean.shape != self.pre_bias.shape:
            raise ValueError("normalization mean does not match residual width")
        self.pre_bias.copy_(mean.to(self.pre_bias))
        self.pre_scale.copy_(torch.as_tensor(scale).to(self.pre_scale))
        self.normalize_decoder()
        self.encoder.linear.weight.copy_(self.decoder)
        self.encoder.linear.bias.zero_()
        self.encoder.linear.bias[: self.cfg.d_high].fill_(self.cfg.target_mu)
        self.sync_ema_sae()

    @torch.no_grad()
    def sync_ema_sae(self) -> None:
        self.ema_encoder.load_state_dict(self.encoder.state_dict())
        self.ema_decoder.copy_(self.decoder)
        self.ema_pre_bias.copy_(self.pre_bias)
        self.normalize_ema_decoder()

    @torch.no_grad()
    def update_ema_sae(self, decay: float | None = None) -> None:
        rate = self.cfg.ema_decay if decay is None else decay
        for target, online in zip(
            self.ema_encoder.parameters(), self.encoder.parameters()
        ):
            target.mul_(rate).add_(online.detach(), alpha=1.0 - rate)
        self.ema_pre_bias.mul_(rate).add_(
            self.pre_bias.detach(), alpha=1.0 - rate
        )
        self.ema_decoder.mul_(rate).add_(
            self.decoder.detach(), alpha=1.0 - rate
        )
        self.normalize_ema_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.div_(
            self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )

    @torch.no_grad()
    def normalize_ema_decoder(self) -> None:
        self.ema_decoder.div_(
            self.ema_decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )

    def split_code(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return code[..., : self.cfg.d_high], code[..., self.cfg.d_high :]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.pre_bias) / self.pre_scale)

    @torch.no_grad()
    def encode_ema(self, x: torch.Tensor) -> torch.Tensor:
        return self.ema_encoder((x - self.ema_pre_bias) / self.pre_scale)

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.decoder)
        return value + self.pre_bias if add_bias else value

    @torch.no_grad()
    def decode_ema(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.ema_decoder)
        return value + self.ema_pre_bias if add_bias else value

    def decode_high(
        self, z: torch.Tensor, *, ema: bool, add_bias: bool = True
    ) -> torch.Tensor:
        decoder = self.ema_decoder if ema else self.decoder
        bias = self.ema_pre_bias if ema else self.pre_bias
        value = self.pre_scale * (z @ decoder[: self.cfg.d_high])
        return value + bias if add_bias else value

    def decode_low(
        self, z: torch.Tensor, *, ema: bool, add_bias: bool = False
    ) -> torch.Tensor:
        decoder = self.ema_decoder if ema else self.decoder
        bias = self.ema_pre_bias if ema else self.pre_bias
        value = self.pre_scale * (z @ decoder[self.cfg.d_high :])
        return value + bias if add_bias else value

    def forward(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if view_a.ndim != 2 or view_a.shape[-1] != self.cfg.d_in:
            raise ValueError("view_a must have shape [batch, d_in]")
        if view_b.shape != view_a.shape:
            raise ValueError("view_b must match view_a")
        code_a = self.encode(view_a)
        code_b = self.encode(view_b)
        high_a, low_a = self.split_code(code_a)
        high_b, low_b = self.split_code(code_b)
        high_reconstruction_a = self.decode_high(high_a, ema=False)
        high_reconstruction_b = self.decode_high(high_b, ema=False)
        full_reconstruction_a = high_reconstruction_a + self.decode_low(
            low_a, ema=False, add_bias=False
        )
        full_reconstruction_b = high_reconstruction_b + self.decode_low(
            low_b, ema=False, add_bias=False
        )
        return {
            "code_a": code_a,
            "code_b": code_b,
            "high_a": high_a,
            "high_b": high_b,
            "low_a": low_a,
            "low_b": low_b,
            "high_reconstruction_a": high_reconstruction_a,
            "high_reconstruction_b": high_reconstruction_b,
            "full_reconstruction_a": full_reconstruction_a,
            "full_reconstruction_b": full_reconstruction_b,
        }

    @torch.no_grad()
    def final_ema_sae_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pre_bias": self.ema_pre_bias.detach(),
            "pre_scale": self.pre_scale.detach(),
            "encoder.linear.weight": self.ema_encoder.linear.weight.detach(),
            "encoder.linear.bias": self.ema_encoder.linear.bias.detach(),
            "decoder": self.ema_decoder.detach(),
        }


def rectified_distribution_matching_loss(
    views: tuple[torch.Tensor, ...],
    cfg: RectifiedLpJEPAConfig,
    projections: int,
    projection_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sliced 2-Wasserstein matching to an i.i.d. RGG product target."""
    if not views or any(view.ndim != 2 for view in views):
        raise ValueError("views must be non-empty [batch, d_high] matrices")
    if any(view.shape != views[0].shape for view in views):
        raise ValueError("all RDM views must have the same shape")
    if views[0].shape[-1] != cfg.d_high:
        raise ValueError("RDM views must contain only the high code")
    if projections < 1 or projection_chunk_size < 1:
        raise ValueError("projection counts must be positive")
    batch, dimension = views[0].shape
    target = sample_rectified_generalized_gaussian(
        (batch, dimension),
        p=cfg.rgg_p,
        mu=cfg.target_mu,
        sigma=cfg.resolved_target_sigma,
        device=views[0].device,
        dtype=torch.float32,
    )
    total = views[0].float().sum() * 0.0
    raw_total = total
    completed = 0
    while completed < projections:
        width = min(projection_chunk_size, projections - completed)
        directions = F.normalize(
            torch.randn(
                dimension,
                width,
                device=views[0].device,
                dtype=torch.float32,
            ),
            dim=0,
        )
        target_projection = torch.sort(target @ directions, dim=0).values
        target_energy = target_projection.square().mean().clamp_min(1e-8)
        chunk_raw = sum(
            (
                torch.sort(view.float() @ directions, dim=0).values
                - target_projection
            )
            .square()
            .mean()
            for view in views
        ) / len(views)
        total = total + width * chunk_raw / target_energy
        raw_total = raw_total + width * chunk_raw
        completed += width
    return total / projections, {
        "raw": raw_total / projections,
        "target_active_fraction": (target > 0).float().mean(),
        "target_l0": (target > 0).sum(dim=-1).float().mean(),
        "target_second_moment": target.square().mean(),
    }


def _pair_reconstruction_losses(
    model: RectifiedLpJEPASAE,
    outputs: dict[str, torch.Tensor],
    view_a: torch.Tensor,
    view_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    energy = (
        torch.cat(
            (view_a - model.pre_bias, view_b - model.pre_bias), dim=0
        )
        .float()
        .square()
        .mean()
        .clamp_min(1e-8)
    )
    full_fvu = 0.5 * (
        (outputs["full_reconstruction_a"] - view_a).float().square().mean()
        + (outputs["full_reconstruction_b"] - view_b).float().square().mean()
    ) / energy
    high_fvu = 0.5 * (
        (outputs["high_reconstruction_a"] - view_a).float().square().mean()
        + (outputs["high_reconstruction_b"] - view_b).float().square().mean()
    ) / energy
    reconstruction = (
        (1.0 - model.cfg.high_reconstruction_weight) * full_fvu
        + model.cfg.high_reconstruction_weight * high_fvu
    )
    return reconstruction, full_fvu, high_fvu


def rectified_lpjepa_loss(
    model: RectifiedLpJEPASAE,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    *,
    invariance_weight: float,
    rdm_weight: float,
    rdm_projections: int,
    rdm_projection_chunk_size: int,
    distance: torch.Tensor | None = None,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(view_a, view_b)
    reconstruction, full_fvu, high_fvu = _pair_reconstruction_losses(
        model, outputs, view_a, view_b
    )
    invariance_raw = (
        outputs["high_a"].float() - outputs["high_b"].float()
    ).square().mean()
    if rdm_weight > 0 or collect_metrics:
        rdm, rdm_metrics = rectified_distribution_matching_loss(
            (outputs["high_a"], outputs["high_b"]),
            model.cfg,
            rdm_projections,
            rdm_projection_chunk_size,
        )
    else:
        target = sample_rectified_generalized_gaussian(
            outputs["high_a"].shape,
            p=model.cfg.rgg_p,
            mu=model.cfg.target_mu,
            sigma=model.cfg.resolved_target_sigma,
            device=outputs["high_a"].device,
        )
        rdm = outputs["high_a"].float().sum() * 0.0
        rdm_metrics = {
            "raw": rdm,
            "target_active_fraction": (target > 0).float().mean(),
            "target_l0": (target > 0).sum(dim=-1).float().mean(),
            "target_second_moment": target.square().mean(),
        }
    invariance = invariance_raw / rdm_metrics["target_second_moment"].clamp_min(
        1e-8
    )
    loss = reconstruction + invariance_weight * invariance + rdm_weight * rdm
    if not collect_metrics:
        return loss, {}

    with torch.no_grad():
        batch = len(view_a)
        permutation = torch.roll(
            torch.arange(batch, device=view_a.device), shifts=1
        )
        high_positive = F.cosine_similarity(
            outputs["high_a"].float(), outputs["high_b"].float(), dim=-1
        )
        high_shuffled = F.cosine_similarity(
            outputs["high_a"].float(),
            outputs["high_b"].index_select(0, permutation).float(),
            dim=-1,
        )
        low_positive = F.cosine_similarity(
            outputs["low_a"].float(), outputs["low_b"].float(), dim=-1
        )
        swap_a = model.decode_high(
            outputs["high_b"], ema=False
        ) + model.decode_low(outputs["low_a"], ema=False, add_bias=False)
        swap_b = model.decode_high(
            outputs["high_a"], ema=False
        ) + model.decode_low(outputs["low_b"], ema=False, add_bias=False)
        energy = (
            torch.cat(
                (view_a - model.pre_bias, view_b - model.pre_bias), dim=0
            )
            .float()
            .square()
            .mean()
            .clamp_min(1e-8)
        )
        swap_fvu = 0.5 * (
            (swap_a - view_a).float().square().mean()
            + (swap_b - view_b).float().square().mean()
        ) / energy
    return loss, {
        "loss": float(loss.detach()),
        "reconstruction_loss": float(reconstruction.detach()),
        "full_reconstruction_fvu": float(full_fvu.detach()),
        "high_reconstruction_fvu": float(high_fvu.detach()),
        "invariance_loss": float(invariance.detach()),
        "invariance_raw_mse": float(invariance_raw.detach()),
        "rdm_loss": float(rdm.detach()),
        "rdm_raw_swd": float(rdm_metrics["raw"].detach()),
        "high_positive_cosine": float(high_positive.mean()),
        "high_shuffled_cosine": float(high_shuffled.mean()),
        "high_positive_margin": float((high_positive - high_shuffled).mean()),
        "low_positive_cosine": float(low_positive.mean()),
        "swap_reconstruction_fvu": float(swap_fvu),
        "high_l0": float(
            0.5
            * (
                (outputs["high_a"] > 0).sum(dim=-1).float().mean()
                + (outputs["high_b"] > 0).sum(dim=-1).float().mean()
            )
        ),
        "low_l0": float(
            0.5
            * (
                (outputs["low_a"] > 0).sum(dim=-1).float().mean()
                + (outputs["low_b"] > 0).sum(dim=-1).float().mean()
            )
        ),
        "high_active_fraction": float(
            0.5
            * (
                (outputs["high_a"] > 0).float().mean()
                + (outputs["high_b"] > 0).float().mean()
            )
        ),
        "sampled_target_active_fraction": float(
            rdm_metrics["target_active_fraction"]
        ),
        "sampled_target_l0": float(rdm_metrics["target_l0"]),
        "mean_distance": float(distance.float().mean()) if distance is not None else 0.0,
    }


@torch.no_grad()
def _ema_metrics(
    model: RectifiedLpJEPASAE,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
) -> dict[str, float]:
    code_a = model.encode_ema(view_a)
    code_b = model.encode_ema(view_b)
    high_a, low_a = model.split_code(code_a)
    high_b, low_b = model.split_code(code_b)
    reconstruction_a = model.decode_ema(code_a)
    reconstruction_b = model.decode_ema(code_b)
    high_reconstruction_a = model.decode_high(high_a, ema=True)
    high_reconstruction_b = model.decode_high(high_b, ema=True)
    energy = (
        torch.cat(
            (view_a - model.ema_pre_bias, view_b - model.ema_pre_bias), dim=0
        )
        .float()
        .square()
        .mean()
        .clamp_min(1e-8)
    )
    return {
        "ema_reconstruction_fvu": float(
            0.5
            * (
                (reconstruction_a - view_a).float().square().mean()
                + (reconstruction_b - view_b).float().square().mean()
            )
            / energy
        ),
        "ema_high_reconstruction_fvu": float(
            0.5
            * (
                (high_reconstruction_a - view_a).float().square().mean()
                + (high_reconstruction_b - view_b).float().square().mean()
            )
            / energy
        ),
        "ema_high_positive_cosine": float(
            F.cosine_similarity(high_a.float(), high_b.float(), dim=-1).mean()
        ),
        "ema_high_l0": float(
            0.5
            * (
                (high_a > 0).sum(dim=-1).float().mean()
                + (high_b > 0).sum(dim=-1).float().mean()
            )
        ),
        "ema_low_l0": float(
            0.5
            * (
                (low_a > 0).sum(dim=-1).float().mean()
                + (low_b > 0).sum(dim=-1).float().mean()
            )
        ),
    }


@torch.no_grad()
def evaluate_losses(
    model: RectifiedLpJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp_dtype: str,
    seed: int,
    invariance_weight: float,
    rdm_weight: float,
    rdm_projections: int,
    rdm_projection_chunk_size: int,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in validation_view_pair_batches(
            root, manifest, batch_size, maximum_batches, seed
        ):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with autocast_context(device, amp_dtype):
                _, metrics = rectified_lpjepa_loss(
                    model,
                    batch["view_a"],
                    batch["view_b"],
                    invariance_weight=invariance_weight,
                    rdm_weight=rdm_weight,
                    rdm_projections=rdm_projections,
                    rdm_projection_chunk_size=rdm_projection_chunk_size,
                    distance=batch["distance"],
                )
                metrics.update(
                    _ema_metrics(model, batch["view_a"], batch["view_b"])
                )
            n = len(batch["view_a"])
            count += n
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + n * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a high/low Rectified LpJEPA-SAE without a predictor"
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--low-k", type=int, default=32)
    parser.add_argument("--high-fraction", type=float, default=0.2)
    parser.add_argument("--high-reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--rgg-p", type=float, choices=SUPPORTED_RGG_P, default=1.0)
    parser.add_argument("--target-active-fraction", type=float, default=0.025)
    parser.add_argument(
        "--target-sigma",
        type=float,
        default=0.0,
        help="Zero selects the unit-pre-rectification-variance sigma from the paper.",
    )
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--rdm-weight", type=float, default=5.0)
    parser.add_argument("--rdm-projections", type=int, default=1024)
    parser.add_argument("--rdm-projection-chunk-size", type=int, default=128)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--sae-warmup-steps", type=int, default=1000)
    parser.add_argument("--regularization-ramp-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--sae-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default="bfloat16")
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 2 or not 0 <= args.sae_warmup_steps < args.steps:
        raise ValueError("sae_warmup_steps must lie in [0, steps)")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two for two-sample RDMReg")
    if args.invariance_weight < 0 or args.rdm_weight < 0:
        raise ValueError("invariance and RDM weights must be non-negative")
    if args.rdm_projections < 1 or args.rdm_projection_chunk_size < 1:
        raise ValueError("RDM projection counts must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    configure_accelerator(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    root, manifest = load_activation_manifest(args.activation_manifest)
    fingerprint = manifest_fingerprint(manifest)
    cfg = RectifiedLpJEPAConfig(
        d_in=int(manifest["d_in"]),
        d_sae=args.d_sae,
        low_k=args.low_k,
        max_span_length=int(manifest["max_span_length"]),
        ema_decay=args.ema_decay,
        high_fraction=args.high_fraction,
        high_reconstruction_weight=args.high_reconstruction_weight,
        rgg_p=args.rgg_p,
        target_active_fraction=args.target_active_fraction,
        target_sigma=args.target_sigma,
    )
    model = RectifiedLpJEPASAE(cfg).to(device)
    normalization = manifest["normalization"]
    model.initialize_from_statistics(
        torch.tensor(normalization["mean"]),
        float(normalization["scalar_rms"]),
    )
    iterator = iter(
        RandomViewPairShardBatches(
            root, manifest, "train", args.batch_size, args.seed
        )
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(trainable_parameters(model)),
                "lr": args.sae_lr,
                "base_lr": args.sae_lr,
            }
        ],
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    history: list[dict[str, Any]] = []
    for step in trange(1, args.steps + 1, desc="Rectified LpJEPA-SAE"):
        optimizer.param_groups[0]["lr"] = cosine_learning_rate(
            step,
            args.steps,
            args.sae_lr,
            min(args.warmup_steps, max(1, args.steps // 10)),
            args.min_lr_ratio,
        )
        rdm_ramp = min(1.0, step / max(args.regularization_ramp_steps, 1))
        joint_step = max(0, step - args.sae_warmup_steps)
        invariance_ramp = min(
            1.0, joint_step / max(args.regularization_ramp_steps, 1)
        )
        active_rdm_weight = args.rdm_weight * rdm_ramp
        active_invariance_weight = args.invariance_weight * invariance_ramp
        phase = "distribution_warmup" if joint_step == 0 else "joint"
        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(args.gradient_accumulation_steps):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in next(iterator).items()
            }
            with autocast_context(device, args.amp_dtype):
                loss, metrics = rectified_lpjepa_loss(
                    model,
                    batch["view_a"],
                    batch["view_b"],
                    invariance_weight=active_invariance_weight,
                    rdm_weight=active_rdm_weight,
                    rdm_projections=args.rdm_projections,
                    rdm_projection_chunk_size=args.rdm_projection_chunk_size,
                    distance=batch["distance"],
                    collect_metrics=should_log,
                )
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
        metrics = {
            key: value / args.gradient_accumulation_steps
            for key, value in metric_sums.items()
        }
        if args.gradient_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model), args.gradient_clip
            )
            if should_log:
                metrics["gradient_norm"] = float(norm)
        optimizer.step()
        model.normalize_decoder()
        model.update_ema_sae()
        if should_log:
            metrics["active_invariance_weight"] = active_invariance_weight
            metrics["active_rdm_weight"] = active_rdm_weight
            validation = evaluate_losses(
                model,
                root,
                manifest,
                args.batch_size,
                args.validation_batches,
                device,
                args.amp_dtype if device.type == "cuda" else "none",
                args.seed + 1,
                args.invariance_weight,
                args.rdm_weight,
                args.rdm_projections,
                args.rdm_projection_chunk_size,
            )
            history.append(
                {
                    "step": step,
                    "phase": phase,
                    "train": metrics,
                    "validation": validation,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_config = {
        "model": manifest["model"],
        "resolved_model_revision": manifest.get("resolved_model_revision"),
        "layer": manifest["layer"],
        "layer_path": manifest["layer_path"],
        "hook_point": manifest["hook_point"],
    }
    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    checkpoint_name = "rectified_lpjepa_sae.pt"
    torch.save(
        {
            "architecture_id": ARCHITECTURE_ID,
            "state_dict": state_dict,
            "config": asdict(cfg),
            "train_args": vars(args),
            "source_config": source_config,
            "activation_manifest": str(Path(args.activation_manifest)),
            "data_fingerprint": fingerprint,
        },
        output_dir / checkpoint_name,
    )
    torch.save(
        {
            "architecture_id": ARCHITECTURE_ID,
            "state_dict": {
                key: value.cpu()
                for key, value in model.final_ema_sae_state_dict().items()
            },
            "config": asdict(cfg),
            "source_rectified_checkpoint": checkpoint_name,
            "data_fingerprint": fingerprint,
            "source_config": source_config,
        },
        output_dir / "ema_sae.pt",
    )
    write_json(
        output_dir / "training_report.json",
        {
            "method": "high/low Rectified LpJEPA-SAE without a predictor",
            "architecture": {
                "id": ARCHITECTURE_ID,
                "d_high": cfg.d_high,
                "d_low": cfg.d_low,
                "low_k": cfg.low_k,
                "max_span_length": cfg.max_span_length,
                "view_sampling": "two exchangeable positions from one random span",
                "high_activation": "shifted ReLU; no Top-K",
                "low_activation": "ReLU plus Top-K",
                "high_role": "high-only reconstruction, view invariance, and RDMReg",
                "low_role": "position-specific incremental reconstruction",
                "predictor": None,
                "teacher_in_loss": None,
                "final_sae": "full EMA high/low encoder-decoder pair",
            },
            "rgg_target": {
                "p": cfg.rgg_p,
                "mu": cfg.target_mu,
                "sigma": cfg.resolved_target_sigma,
                "active_fraction": cfg.target_active_fraction,
                "expected_high_l0": cfg.expected_high_l0,
                "distribution": "ReLU(GN_p(mu, sigma)) independently per coordinate",
            },
            "objective": {
                "full_reconstruction_weight": 1.0 - cfg.high_reconstruction_weight,
                "high_reconstruction_weight": cfg.high_reconstruction_weight,
                "invariance_weight": args.invariance_weight,
                "rdm_weight": args.rdm_weight,
                "rdm_projections": args.rdm_projections,
                "rdm_projection_chunk_size": args.rdm_projection_chunk_size,
                "invariance": "normalized squared L2 between online high codes",
                "rdm": "normalized sliced two-sample 2-Wasserstein",
            },
            "data": {
                "dataset": manifest["dataset"],
                "fingerprint": fingerprint,
                "sampling": "random span and two exchangeable positions without replacement",
                "min_span_length": manifest["min_span_length"],
                "max_span_length": manifest["max_span_length"],
                "sequence_length": manifest["sequence_length"],
                "burn_in_tokens": manifest["burn_in_tokens"],
                "n_train_sequences": manifest["train"]["sequences"],
                "n_validation_sequences": manifest["validation"]["sequences"],
                "n_train_positions": manifest["train"]["positions"],
                "n_validation_positions": manifest["validation"]["positions"],
            },
            "accelerator": {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else None
                ),
            },
            "history": history,
        },
    )


if __name__ == "__main__":
    main()
