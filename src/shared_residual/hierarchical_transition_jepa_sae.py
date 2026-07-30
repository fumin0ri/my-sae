from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .group_sae import topk_relu
from .standard_sae import SparseEncoder
from .transition_jepa_sae import (
    PositionConditionedPredictor,
    TransitionJEPAConfig,
    TransitionJEPASAE,
    support_metrics,
)

HIERARCHICAL_ARCHITECTURE_ID = (
    "hierarchical_high_low_fixed_endpoint_ema_sae_v1"
)


@dataclass
class HierarchicalTransitionJEPAConfig(TransitionJEPAConfig):
    """T-SAE-inspired two-group dictionary with a forecastable high group."""

    high_fraction: float = 0.2
    high_reconstruction_weight: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.high_fraction < 1.0:
            raise ValueError("high_fraction must lie strictly between zero and one")
        if not 0.0 <= self.high_reconstruction_weight <= 1.0:
            raise ValueError(
                "high_reconstruction_weight must lie in [0, 1]"
            )
        if self.d_high < 1 or self.d_low < 1:
            raise ValueError("both high and low dictionaries must be non-empty")
        if self.k_high < 1 or self.k_low < 1:
            raise ValueError("both high and low Top-K budgets must be positive")

    @property
    def d_high(self) -> int:
        return max(1, min(self.d_sae - 1, round(self.d_sae * self.high_fraction)))

    @property
    def d_low(self) -> int:
        return self.d_sae - self.d_high

    @property
    def k_high(self) -> int:
        proposed = round(self.k * self.high_fraction)
        return max(1, min(self.d_high, self.k - 1, proposed))

    @property
    def k_low(self) -> int:
        return self.k - self.k_high


class HierarchicalSparseEncoder(SparseEncoder):
    """Apply independent Top-K budgets to high and low dictionary blocks."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        d_high: int,
        k_high: int,
        k_low: int,
    ):
        # SparseEncoder owns the checkpoint-compatible ``linear`` module.
        super().__init__(d_in, d_sae, k_high + k_low)
        self.d_high = d_high
        self.k_high = k_high
        self.k_low = k_low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = self.linear(x)
        high = topk_relu(dense[..., : self.d_high], self.k_high)
        low = topk_relu(dense[..., self.d_high :], self.k_low)
        return torch.cat((high, low), dim=-1)


class HierarchicalTransitionJEPASAE(TransitionJEPASAE):
    """Matryoshka high/low SAE whose high block is the JEPA state.

    The high block must reconstruct the endpoint on its own and is forecast
    from each earlier position. The low block is never forecast-supervised; it
    adds endpoint detail in the full reconstruction.
    """

    architecture_id = HIERARCHICAL_ARCHITECTURE_ID

    def __init__(self, cfg: HierarchicalTransitionJEPAConfig):
        # Build the baseline shape first so initialization and checkpoint names
        # remain compatible with the common standard SAE.
        super().__init__(cfg)
        original_encoder = self.encoder
        self.encoder = HierarchicalSparseEncoder(
            cfg.d_in,
            cfg.d_sae,
            cfg.d_high,
            cfg.k_high,
            cfg.k_low,
        )
        self.encoder.linear.load_state_dict(original_encoder.linear.state_dict())
        self.ema_encoder = copy.deepcopy(self.encoder)
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad_(False)
        predictor_cfg = TransitionJEPAConfig(
            d_in=cfg.d_in,
            d_sae=cfg.d_high,
            k=cfg.k_high,
            window_size=cfg.window_size,
            predictor_width=cfg.predictor_width,
            predictor_expansion=cfg.predictor_expansion,
            ema_decay=cfg.ema_decay,
        )
        self.transition_predictor = PositionConditionedPredictor(predictor_cfg)
        self.cfg = cfg

    @property
    def forecast_dim(self) -> int:
        return self.cfg.d_high

    @property
    def forecast_k(self) -> int:
        return self.cfg.k_high

    @property
    def low_dim(self) -> int:
        return self.cfg.d_low

    def split_code(
        self,
        code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return code[..., : self.cfg.d_high], code[..., self.cfg.d_high :]

    @torch.no_grad()
    def encode_forecast_ema(self, x: torch.Tensor) -> torch.Tensor:
        high, _ = self.split_code(self.encode_ema(x))
        return high

    def decode_high(
        self,
        z_high: torch.Tensor,
        *,
        ema: bool,
        add_bias: bool = True,
    ) -> torch.Tensor:
        decoder = self.ema_decoder if ema else self.decoder
        bias = self.ema_pre_bias if ema else self.pre_bias
        decoded = self.pre_scale * (z_high @ decoder[: self.cfg.d_high])
        return decoded + bias if add_bias else decoded

    def decode_low(
        self,
        z_low: torch.Tensor,
        *,
        ema: bool,
        add_bias: bool = False,
    ) -> torch.Tensor:
        decoder = self.ema_decoder if ema else self.decoder
        bias = self.ema_pre_bias if ema else self.pre_bias
        decoded = self.pre_scale * (z_low @ decoder[self.cfg.d_high :])
        return decoded + bias if add_bias else decoded

    def decode_forecast_ema(
        self,
        z: torch.Tensor,
        add_bias: bool = True,
    ) -> torch.Tensor:
        return self.decode_high(z, ema=True, add_bias=add_bias)

    def predict_from_code(
        self,
        context_code: torch.Tensor,
        context_positions: torch.Tensor | None = None,
        use_context: bool = True,
        sparse_output: bool = False,
    ) -> torch.Tensor:
        # Accept a full code for convenience at intervention/evaluation time,
        # but never expose the low block to the predictor.
        if context_code.shape[-1] == self.cfg.d_sae:
            context_code = context_code[..., : self.cfg.d_high]
        if context_code.shape[-1] != self.cfg.d_high:
            raise ValueError(
                f"context code must have width {self.cfg.d_high} (high) or "
                f"{self.cfg.d_sae} (full)"
            )
        if context_code.ndim == 2:
            context_code = context_code[:, None, :]
        if context_positions is None:
            if context_code.shape[1] == self.cfg.window_size - 1:
                context_positions = torch.arange(
                    self.cfg.window_size - 1,
                    device=context_code.device,
                    dtype=torch.long,
                )
            elif context_code.shape[1] == 1:
                context_positions = torch.tensor(
                    [0],
                    device=context_code.device,
                    dtype=torch.long,
                )
            else:
                raise ValueError(
                    "explicit context_positions are required for this shape"
                )
        dense = self.transition_predictor(
            context_code,
            context_positions,
            use_context=use_context,
        )
        return (
            topk_relu(dense, self.cfg.k_high)
            if sparse_output
            else dense
        )

    def forward(
        self,
        x: torch.Tensor,
        use_context: bool = True,
        use_ema_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.cfg.window_size:
            raise ValueError(
                "x must have shape [batch, "
                f"{self.cfg.window_size}, {self.cfg.d_in}]"
            )
        codes = self.encode(x)
        online_high, online_low = self.split_code(codes)
        if use_ema_context:
            with torch.no_grad():
                ema_codes = self.encode_ema(x)
                context_high = ema_codes[:, :-1, : self.cfg.d_high]
                context_low = ema_codes[:, :-1, self.cfg.d_high :]
        else:
            context_high = online_high[:, :-1]
            context_low = online_low[:, :-1]
        online_target_high = online_high[:, -1]
        online_target_low = online_low[:, -1]
        online_high_reconstruction = self.decode_high(
            online_target_high,
            ema=False,
        )
        online_target_reconstruction = online_high_reconstruction + (
            self.decode_low(online_target_low, ema=False, add_bias=False)
        )
        with torch.no_grad():
            target_full = self.encode_ema(x[:, -1])
            target_high, target_low = self.split_code(target_full)
            target_high_reconstruction = self.decode_high(
                target_high,
                ema=True,
            )
            target_reconstruction = target_high_reconstruction + (
                self.decode_low(target_low, ema=True, add_bias=False)
            )
        prediction = self.predict_from_code(
            context_high,
            use_context=use_context,
        )
        sparse_prediction = topk_relu(prediction, self.cfg.k_high)
        predictable_residual = self.decode_forecast_ema(sparse_prediction)
        target_codes = target_high[:, None, :].expand_as(prediction)
        target_residual = x[:, -1][:, None, :].expand_as(
            predictable_residual
        )
        return {
            "codes": codes,
            "high_codes": online_high,
            "low_codes": online_low,
            "context_codes": context_high,
            "context_code": context_high[:, 0],
            "low_context_codes": context_low,
            "low_context_code": context_low[:, 0],
            "online_target_code": online_target_high,
            "online_target_low_code": online_target_low,
            "online_target_full_code": codes[:, -1],
            "online_high_reconstruction": online_high_reconstruction,
            "online_target_reconstruction": online_target_reconstruction,
            "target_high_reconstruction": target_high_reconstruction,
            "target_reconstruction": target_reconstruction,
            "target_code": target_high,
            "target_low_code": target_low,
            "target_full_code": target_full,
            "target_codes": target_codes,
            "predicted_codes": prediction,
            "sparse_predicted_codes": sparse_prediction,
            "target_residual": target_residual,
            "predictable_residual": predictable_residual,
            "innovation_residual": target_residual - predictable_residual,
        }

    @torch.no_grad()
    def final_ema_sae_state_dict(self) -> dict[str, torch.Tensor]:
        # Same tensor names as a standard SAE, with group metadata stored in
        # the outer artifact config.
        return super().final_ema_sae_state_dict()


def hierarchical_transition_jepa_loss(
    model: HierarchicalTransitionJEPASAE,
    x: torch.Tensor,
    prediction_weight: float,
    residual_prediction_weight: float,
    use_context: bool,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x, use_context=use_context)
    target_residual = x[:, -1]
    residual_scale = (
        target_residual - model.pre_bias.detach()
    ).float().square().mean().clamp_min(1e-8)
    high_reconstruction = (
        outputs["online_high_reconstruction"] - target_residual
    ).float().square().mean() / residual_scale
    full_reconstruction = (
        outputs["online_target_reconstruction"] - target_residual
    ).float().square().mean() / residual_scale
    reconstruction = (
        model.cfg.high_reconstruction_weight * high_reconstruction
        + (1.0 - model.cfg.high_reconstruction_weight) * full_reconstruction
    )
    ema_scale = (
        target_residual - model.ema_pre_bias.detach()
    ).float().square().mean().clamp_min(1e-8)
    ema_high_reconstruction = (
        outputs["target_high_reconstruction"] - target_residual
    ).float().square().mean() / ema_scale
    ema_full_reconstruction = (
        outputs["target_reconstruction"] - target_residual
    ).float().square().mean() / ema_scale

    prediction = outputs["predicted_codes"]
    target = outputs["target_codes"].detach()
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    target_energy = target.square().mean(dim=-1).clamp_min(1e-8)
    nrmse = (prediction - target).square().mean(dim=-1) / target_energy
    prediction_loss = (1.0 - cosine + 0.25 * nrmse).mean()
    residual_prediction = (
        outputs["predictable_residual"] - outputs["target_residual"]
    ).square().mean() / ema_scale
    loss = reconstruction + prediction_weight * (
        prediction_loss + residual_prediction_weight * residual_prediction
    )

    metrics: dict[str, float] = {}
    if collect_metrics:
        precision, recall, jaccard = support_metrics(
            prediction.detach(),
            target,
            model.cfg.k_high,
        )
        sparse_prediction = outputs["sparse_predicted_codes"]
        metrics = {
            "loss": float(loss.detach().item()),
            "online_reconstruction_fvu": float(full_reconstruction.detach()),
            "online_high_reconstruction_fvu": float(
                high_reconstruction.detach()
            ),
            "weighted_reconstruction_fvu": float(reconstruction.detach()),
            "ema_reconstruction_fvu": float(
                ema_full_reconstruction.detach()
            ),
            "ema_high_reconstruction_fvu": float(
                ema_high_reconstruction.detach()
            ),
            "prediction_loss": float(prediction_loss.detach()),
            "code_cosine": float(cosine.mean().detach()),
            "code_nrmse": float(nrmse.mean().detach()),
            "support_precision": float(precision.mean()),
            "support_recall": float(recall.mean()),
            "support_jaccard": float(jaccard.mean()),
            "residual_prediction_fvu": float(residual_prediction.detach()),
            "sae_l0": float(
                (outputs["codes"] > 0).sum(dim=-1).float().mean()
            ),
            "high_l0": float(
                (outputs["high_codes"] > 0).sum(dim=-1).float().mean()
            ),
            "low_l0": float(
                (outputs["low_codes"] > 0).sum(dim=-1).float().mean()
            ),
            "predictor_dense_norm": float(
                prediction.float().norm(dim=-1).mean().detach()
            ),
            "predictor_topk_norm": float(
                sparse_prediction.float().norm(dim=-1).mean().detach()
            ),
            "target_code_norm": float(
                outputs["target_code"].float().norm(dim=-1).mean()
            ),
            "low_target_code_norm": float(
                outputs["target_low_code"].float().norm(dim=-1).mean()
            ),
            "innovation_energy_fraction": float(
                outputs["innovation_residual"]
                .float()
                .square()
                .mean()
                .div(ema_scale)
                .detach()
            ),
        }
        target_position = model.cfg.window_size - 1
        for position in range(target.shape[1]):
            horizon = target_position - position
            prefix = f"context_{position}_horizon_{horizon}"
            metrics[f"{prefix}_cosine"] = float(
                cosine[:, position].mean().detach()
            )
            metrics[f"{prefix}_nrmse"] = float(
                nrmse[:, position].mean().detach()
            )
            metrics[f"{prefix}_support_recall"] = float(
                recall[:, position].mean()
            )
    return loss, metrics
