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
    RandomPairShardBatches,
    load_activation_manifest,
    manifest_fingerprint,
    validation_pair_batches,
)
from .group_sae import topk_relu
from .io import write_json
from .training import (
    autocast_context,
    configure_accelerator,
    cosine_learning_rate,
)

ARCHITECTURE_ID = "high_low_random_pair_horizon_ema_sae_v3"
HORIZON_WEIGHTING_MODES = ("none", "inverse_probability")
PREDICTOR_OUTPUT_MODES = ("softplus", "relu_topk")
SOFTPLUS_OUTPUT_BIAS_INIT = -4.0


def predictor_output_bias_init(mode: str) -> float:
    """Choose a non-dead, scale-matched output bias for each predictor head."""
    if mode == "softplus":
        return SOFTPLUS_OUTPUT_BIAS_INIT
    if mode == "relu_topk":
        # Reusing -4 would put every ReLU unit in its zero-gradient region.
        # Match ReLU's initial positive level to softplus(-4) instead.
        return math.log1p(math.exp(SOFTPLUS_OUTPUT_BIAS_INIT))
    raise ValueError(f"unsupported predictor output mode: {mode}")


def horizon_sampling_probabilities(
    min_span_length: int,
    max_span_length: int,
) -> torch.Tensor:
    """Exact P(h) induced by uniform L and uniform non-endpoint context."""
    if not 2 <= min_span_length <= max_span_length:
        raise ValueError("span bounds must satisfy 2 <= min <= max")
    probabilities = torch.zeros(max_span_length, dtype=torch.float64)
    span_count = max_span_length - min_span_length + 1
    for span_length in range(min_span_length, max_span_length + 1):
        probabilities[1:span_length] += 1.0 / (
            span_count * (span_length - 1)
        )
    return probabilities


def horizon_loss_weight_table(
    min_span_length: int,
    max_span_length: int,
    mode: str,
) -> torch.Tensor:
    """Weights whose expected contribution is equal for every horizon."""
    if mode not in HORIZON_WEIGHTING_MODES:
        raise ValueError(f"unsupported horizon weighting mode: {mode}")
    probabilities = horizon_sampling_probabilities(
        min_span_length, max_span_length
    )
    weights = torch.ones(max_span_length, dtype=torch.float64)
    if mode == "inverse_probability":
        max_horizon = max_span_length - 1
        weights[1:] = 1.0 / (max_horizon * probabilities[1:])
    weights[0] = 0.0
    return weights.float()


@dataclass
class TransitionJEPAConfig:
    d_in: int
    d_sae: int = 2048
    k: int = 32
    max_span_length: int = 10
    predictor_width: int = 256
    predictor_expansion: int = 2
    predictor_output: str = "softplus"
    ema_decay: float = 0.996
    high_fraction: float = 0.2
    high_reconstruction_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.max_span_length < 2:
            raise ValueError("max_span_length must be at least two")
        if self.k < 2 or self.k > self.d_sae:
            raise ValueError("k must lie in [2, d_sae]")
        if not 0.0 < self.high_fraction < 1.0:
            raise ValueError("high_fraction must lie strictly between zero and one")
        if not 0.0 <= self.high_reconstruction_weight <= 1.0:
            raise ValueError("high_reconstruction_weight must lie in [0, 1]")
        if self.d_high < 1 or self.d_low < 1:
            raise ValueError("both high and low dictionaries must be non-empty")
        if self.k_high < 1 or self.k_low < 1:
            raise ValueError("both high and low Top-K budgets must be positive")
        if self.predictor_output not in PREDICTOR_OUTPUT_MODES:
            raise ValueError(
                f"predictor_output must be one of {PREDICTOR_OUTPUT_MODES}"
            )

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


class HierarchicalSparseEncoder(nn.Module):
    """One linear dictionary with independent high/low Top-K budgets."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.d_in, cfg.d_sae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = self.linear(x)
        high = topk_relu(dense[..., : self.cfg.d_high], self.cfg.k_high)
        low = topk_relu(dense[..., self.cfg.d_high :], self.cfg.k_low)
        return torch.cat((high, low), dim=-1)


class HorizonConditionedPredictor(nn.Module):
    """Predict a future high code from context code and explicit token distance."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.predictor_width
        hidden = cfg.predictor_expansion * width
        self.context_projection = nn.Linear(cfg.d_high, width, bias=False)
        self.horizon_embedding = nn.Embedding(cfg.max_span_length, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, cfg.d_high)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(
            self.output.bias,
            predictor_output_bias_init(cfg.predictor_output),
        )

    def forward_with_logits(
        self,
        context_code: torch.Tensor,
        horizons: torch.Tensor,
        use_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze_context = context_code.ndim == 2
        if context_code.ndim == 2:
            context_code = context_code[:, None, :]
        if context_code.ndim != 3:
            raise ValueError("context_code must have shape [batch, contexts, d_high]")
        if torch.any(horizons < 1) or torch.any(horizons >= self.cfg.max_span_length):
            raise ValueError("horizons must lie in [1, max_span_length-1]")
        if squeeze_context and horizons.shape == (context_code.shape[0],):
            horizon_state = self.horizon_embedding(horizons)[:, None, :]
        elif (
            not squeeze_context
            and horizons.ndim == 1
            and horizons.shape == (context_code.shape[1],)
        ):
            horizon_state = self.horizon_embedding(horizons)[None, :, :]
        elif horizons.shape == context_code.shape[:2]:
            horizon_state = self.horizon_embedding(horizons)
        else:
            raise ValueError("horizons must match the batch or context axis")
        state = self.context_projection(context_code)
        if not use_context:
            state = torch.zeros_like(state)
        logits = self.output(self.mlp(state + horizon_state))
        if self.cfg.predictor_output == "softplus":
            output = F.softplus(logits)
        else:
            output = topk_relu(logits, self.cfg.k_high)
        if squeeze_context:
            return output[:, 0], logits[:, 0]
        return output, logits

    def forward(
        self,
        context_code: torch.Tensor,
        horizons: torch.Tensor,
        use_context: bool = True,
    ) -> torch.Tensor:
        output, _ = self.forward_with_logits(
            context_code, horizons, use_context=use_context
        )
        return output


class PredictorActivityTracker:
    """Track main-path inactivity without counting the SAE-only warm-up."""

    def __init__(
        self,
        features: int,
        dead_after_batches: int,
        device: torch.device,
    ) -> None:
        if features < 1:
            raise ValueError("features must be positive")
        if dead_after_batches < 1:
            raise ValueError("dead_after_batches must be positive")
        self.dead_after_batches = dead_after_batches
        self.inactive_batches = torch.zeros(
            features, dtype=torch.long, device=device
        )
        self.ever_active = torch.zeros(
            features, dtype=torch.bool, device=device
        )
        self.prediction_batches = 0
        self.total_reactivations = torch.zeros(
            (), dtype=torch.long, device=device
        )

    @property
    def dead_mask(self) -> torch.Tensor:
        return self.inactive_batches >= self.dead_after_batches

    @torch.no_grad()
    def update(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 2 or prediction.shape[-1] != len(
            self.inactive_batches
        ):
            raise ValueError("prediction must have shape [batch, features]")
        active = (prediction > 0).any(dim=0)
        previously_dead = self.dead_mask
        reactivated = (previously_dead & active).sum()
        self.inactive_batches.add_(1)
        self.inactive_batches[active] = 0
        self.ever_active.logical_or_(active)
        self.prediction_batches += 1
        self.total_reactivations.add_(reactivated)
        return reactivated

    def summary(self) -> dict[str, int | float]:
        dead = int(self.dead_mask.sum().item())
        ever_active = int(self.ever_active.sum().item())
        features = len(self.inactive_batches)
        return {
            "features": features,
            "prediction_batches": self.prediction_batches,
            "dead_features": dead,
            "dead_feature_fraction": dead / features,
            "ever_active_features": ever_active,
            "ever_active_fraction": ever_active / features,
            "total_reactivations": int(self.total_reactivations.item()),
        }


def predictor_auxk_per_sample_loss(
    logits: torch.Tensor,
    main_prediction: torch.Tensor,
    target: torch.Tensor,
    dead_mask: torch.Tensor,
    k_aux: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use inactive predictor coordinates to fit missing positive target code."""
    if logits.ndim != 2 or logits.shape != main_prediction.shape:
        raise ValueError("logits and main_prediction must match [batch, features]")
    if target.shape != logits.shape:
        raise ValueError("target must match predictor logits")
    if dead_mask.shape != (logits.shape[-1],) or dead_mask.dtype != torch.bool:
        raise ValueError("dead_mask must be a boolean feature mask")
    if k_aux < 1:
        raise ValueError("k_aux must be positive")
    dead_indices = dead_mask.nonzero(as_tuple=False).flatten()
    if dead_indices.numel() == 0:
        zeros = logits.sum(dim=-1) * 0.0
        return zeros, zeros
    dead_logits = logits.index_select(-1, dead_indices)
    aux_prediction = topk_relu(
        dead_logits, min(k_aux, dead_logits.shape[-1])
    )
    missing_target = torch.relu(
        target.detach() - main_prediction.detach()
    ).index_select(-1, dead_indices)
    target_energy = target.detach().float().square().sum(dim=-1).clamp_min(1e-8)
    per_sample_loss = (
        (aux_prediction.float() - missing_target.float())
        .square()
        .sum(dim=-1)
        / target_energy
    )
    aux_l0 = (aux_prediction > 0).sum(dim=-1).float()
    return per_sample_loss, aux_l0


class TransitionJEPASAE(nn.Module):
    """High/low endpoint SAE; only the high group receives JEPA supervision."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("pre_scale", torch.ones(()))
        self.encoder = HierarchicalSparseEncoder(cfg)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))
        self.ema_encoder = copy.deepcopy(self.encoder)
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad_(False)
        self.ema_decoder = nn.Parameter(self.decoder.detach().clone(), requires_grad=False)
        self.register_buffer("ema_pre_bias", self.pre_bias.detach().clone())
        self.transition_predictor = HorizonConditionedPredictor(cfg)

    @property
    def forecast_dim(self) -> int:
        return self.cfg.d_high

    @property
    def forecast_k(self) -> int:
        return self.cfg.k_high

    @property
    def low_dim(self) -> int:
        return self.cfg.d_low

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
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

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

    def encode_forecast_online(self, x: torch.Tensor) -> torch.Tensor:
        high, _ = self.split_code(self.encode(x))
        return high

    @torch.no_grad()
    def encode_forecast_ema(self, x: torch.Tensor) -> torch.Tensor:
        high, _ = self.split_code(self.encode_ema(x))
        return high

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.decoder)
        return value + self.pre_bias if add_bias else value

    def decode_ema(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.ema_decoder)
        return value + self.ema_pre_bias if add_bias else value

    def decode_forecast_ema(
        self, z: torch.Tensor, add_bias: bool = True
    ) -> torch.Tensor:
        return self.decode_high(z, ema=True, add_bias=add_bias)

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

    def predict_from_code(
        self,
        context_high: torch.Tensor,
        horizons: torch.Tensor,
        use_context: bool = True,
        sparse_output: bool = False,
    ) -> torch.Tensor:
        if context_high.shape[-1] == self.cfg.d_sae:
            context_high = context_high[..., : self.cfg.d_high]
        if context_high.shape[-1] != self.cfg.d_high:
            raise ValueError("predictor input must be a high-group code")
        dense = self.transition_predictor(
            context_high, horizons, use_context=use_context
        )
        return topk_relu(dense, self.cfg.k_high) if sparse_output else dense

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        horizon: torch.Tensor,
        use_context: bool = True,
        use_ema_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 2 or context.shape[-1] != self.cfg.d_in:
            raise ValueError("context must have shape [batch, d_in]")
        if target.shape != context.shape:
            raise ValueError("target must match context shape")
        if horizon.shape != (len(context),):
            raise ValueError("horizon must have shape [batch]")
        target_code = self.encode(target)
        online_target_high, online_target_low = self.split_code(target_code)
        if use_ema_context:
            context_code = self.encode_ema(context)
        else:
            context_code = self.encode(context)
        context_high, context_low = self.split_code(context_code)
        high_reconstruction = self.decode_high(online_target_high, ema=False)
        full_reconstruction = high_reconstruction + self.decode_low(
            online_target_low, ema=False, add_bias=False
        )
        with torch.no_grad():
            ema_target = self.encode_ema(target)
            ema_target_high, ema_target_low = self.split_code(ema_target)
            ema_high_reconstruction = self.decode_high(ema_target_high, ema=True)
            ema_full_reconstruction = ema_high_reconstruction + self.decode_low(
                ema_target_low, ema=True, add_bias=False
            )
        prediction, prediction_logits = self.transition_predictor.forward_with_logits(
            context_high, horizon, use_context=use_context
        )
        return {
            "target_codes_online": target_code,
            "target_high_codes_online": online_target_high,
            "target_low_codes_online": online_target_low,
            "context_code": context_high,
            "low_context_code": context_low,
            "online_target_high": online_target_high,
            "online_target_low": online_target_low,
            "online_target_code": online_target_high,
            "online_high_reconstruction": high_reconstruction,
            "online_target_reconstruction": full_reconstruction,
            "target_code": ema_target_high,
            "target_low_code": ema_target_low,
            "target_codes": ema_target_high,
            "target_high_reconstruction": ema_high_reconstruction,
            "target_reconstruction": ema_full_reconstruction,
            "predicted_codes": prediction,
            "prediction_logits": prediction_logits,
            "horizon": horizon,
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


def support_metrics(
    prediction: torch.Tensor, target: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_active = topk_relu(prediction, k) > 0
    target_active = target > 0
    intersection = (predicted_active & target_active).sum(dim=-1).float()
    precision = intersection / predicted_active.sum(dim=-1).float().clamp_min(1)
    recall = intersection / target_active.sum(dim=-1).float().clamp_min(1)
    union = (predicted_active | target_active).sum(dim=-1).float().clamp_min(1)
    return precision, recall, intersection / union


def transition_jepa_loss(
    model: TransitionJEPASAE,
    context: torch.Tensor,
    target: torch.Tensor,
    horizon: torch.Tensor,
    prediction_weight: float,
    span_length: torch.Tensor | None = None,
    horizon_weight_table: torch.Tensor | None = None,
    activity_tracker: PredictorActivityTracker | None = None,
    update_activity_tracker: bool = False,
    predictor_auxk_weight: float = 0.0,
    predictor_auxk: int = 512,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if predictor_auxk_weight < 0:
        raise ValueError("predictor_auxk_weight must be non-negative")
    if predictor_auxk_weight > 0 and activity_tracker is None:
        raise ValueError("AuxK requires an activity tracker")
    outputs = model(context, target, horizon)
    scale = (target - model.pre_bias.detach()).float().square().mean().clamp_min(1e-8)
    high_fvu = (
        (outputs["online_high_reconstruction"] - target).float().square().mean()
        / scale
    )
    full_fvu = (
        (outputs["online_target_reconstruction"] - target).float().square().mean()
        / scale
    )
    reconstruction = (
        model.cfg.high_reconstruction_weight * high_fvu
        + (1.0 - model.cfg.high_reconstruction_weight) * full_fvu
    )
    ema_scale = (
        (target - model.ema_pre_bias.detach()).float().square().mean().clamp_min(1e-8)
    )
    ema_high_fvu = (
        (outputs["target_high_reconstruction"] - target).float().square().mean()
        / ema_scale
    )
    ema_full_fvu = (
        (outputs["target_reconstruction"] - target).float().square().mean()
        / ema_scale
    )
    prediction = outputs["predicted_codes"]
    target_codes = outputs["target_codes"].detach()
    cosine = F.cosine_similarity(prediction, target_codes, dim=-1)
    target_energy = target_codes.square().mean(dim=-1).clamp_min(1e-8)
    nrmse = (prediction - target_codes).square().mean(dim=-1) / target_energy
    per_sample_prediction_loss = 1.0 - cosine + 0.25 * nrmse
    if horizon_weight_table is None:
        sample_weights = torch.ones_like(per_sample_prediction_loss)
    else:
        if horizon_weight_table.shape != (model.cfg.max_span_length,):
            raise ValueError(
                "horizon_weight_table must have shape [max_span_length]"
            )
        sample_weights = horizon_weight_table.to(
            device=horizon.device,
            dtype=per_sample_prediction_loss.dtype,
        ).index_select(0, horizon)
    prediction_loss_unweighted = per_sample_prediction_loss.mean()
    prediction_loss = (sample_weights * per_sample_prediction_loss).mean()
    reactivated = prediction.new_zeros(())
    if activity_tracker is not None and update_activity_tracker:
        reactivated = activity_tracker.update(prediction.detach())
    auxk_per_sample = torch.zeros_like(per_sample_prediction_loss)
    auxk_l0 = torch.zeros_like(per_sample_prediction_loss)
    if predictor_auxk_weight > 0 and activity_tracker is not None:
        auxk_per_sample, auxk_l0 = predictor_auxk_per_sample_loss(
            outputs["prediction_logits"],
            prediction,
            target_codes,
            activity_tracker.dead_mask,
            predictor_auxk,
        )
    auxk_loss_unweighted = auxk_per_sample.mean()
    auxk_loss = (sample_weights * auxk_per_sample).mean()
    prediction_objective = prediction_loss + predictor_auxk_weight * auxk_loss
    loss = reconstruction + prediction_weight * prediction_objective
    if not collect_metrics:
        return loss, {}
    precision, recall, jaccard = support_metrics(
        prediction.detach(), target_codes, model.cfg.k_high
    )
    metrics = {
        "loss": float(loss.detach()),
        "online_high_reconstruction_fvu": float(high_fvu.detach()),
        "online_reconstruction_fvu": float(full_fvu.detach()),
        "ema_high_reconstruction_fvu": float(ema_high_fvu.detach()),
        "ema_reconstruction_fvu": float(ema_full_fvu.detach()),
        "prediction_loss": float(prediction_loss.detach()),
        "prediction_loss_unweighted": float(
            prediction_loss_unweighted.detach()
        ),
        "prediction_objective": float(prediction_objective.detach()),
        "predictor_auxk_loss": float(auxk_loss.detach()),
        "predictor_auxk_loss_unweighted": float(
            auxk_loss_unweighted.detach()
        ),
        "predictor_auxk_l0": float(auxk_l0.mean().detach()),
        "predictor_auxk_weight": predictor_auxk_weight,
        "predictor_reactivated_features": float(reactivated.detach()),
        "mean_horizon_loss_weight": float(sample_weights.mean().detach()),
        "code_cosine": float(cosine.mean().detach()),
        "code_nrmse": float(nrmse.mean().detach()),
        "support_precision": float(precision.mean()),
        "support_recall": float(recall.mean()),
        "support_jaccard": float(jaccard.mean()),
        "predictor_output_l0": float(
            (prediction.detach() > 0).sum(dim=-1).float().mean()
        ),
        "predictor_zero_sample_fraction": float(
            ((prediction.detach() > 0).sum(dim=-1) == 0).float().mean()
        ),
        "predictor_batch_alive_fraction": float(
            (prediction.detach() > 0).any(dim=0).float().mean()
        ),
        "high_l0": float(
            (outputs["target_high_codes_online"] > 0).sum(dim=-1).float().mean()
        ),
        "low_l0": float(
            (outputs["target_low_codes_online"] > 0).sum(dim=-1).float().mean()
        ),
        "mean_horizon": float(horizon.float().mean()),
    }
    if span_length is not None:
        metrics["mean_span_length"] = float(span_length.float().mean())
    if activity_tracker is not None:
        tracker_summary = activity_tracker.summary()
        metrics["predictor_dead_features"] = float(
            tracker_summary["dead_features"]
        )
        metrics["predictor_dead_feature_fraction"] = float(
            tracker_summary["dead_feature_fraction"]
        )
        metrics["predictor_ever_active_fraction"] = float(
            tracker_summary["ever_active_fraction"]
        )
        metrics["predictor_total_reactivations"] = float(
            tracker_summary["total_reactivations"]
        )
    return loss, metrics


@torch.no_grad()
def evaluate_losses(
    model: TransitionJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    prediction_weight: float,
    amp_dtype: str,
    seed: int,
    horizon_weight_table: torch.Tensor,
    activity_tracker: PredictorActivityTracker | None = None,
    predictor_auxk_weight: float = 0.0,
    predictor_auxk: int = 512,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in validation_pair_batches(
        root, manifest, batch_size, maximum_batches, seed
    ):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with autocast_context(device, amp_dtype):
            _, metrics = transition_jepa_loss(
                model,
                batch["context"],
                batch["target"],
                batch["horizon"],
                prediction_weight,
                span_length=batch["span_length"],
                horizon_weight_table=horizon_weight_table,
                activity_tracker=activity_tracker,
                update_activity_tracker=False,
                predictor_auxk_weight=predictor_auxk_weight,
                predictor_auxk=predictor_auxk,
            )
        count += len(batch["target"])
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + len(batch["target"]) * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the high/low random-pair horizon-conditioned EMA JEPA-SAE"
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--high-fraction", type=float, default=0.2)
    parser.add_argument("--high-reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--predictor-width", type=int, default=256)
    parser.add_argument("--predictor-expansion", type=int, default=2)
    parser.add_argument(
        "--predictor-output",
        choices=PREDICTOR_OUTPUT_MODES,
        default="softplus",
        help=(
            "softplus reproduces the dense baseline; relu_topk applies ReLU "
            "and the high-group Top-K budget inside the predictor during training"
        ),
    )
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument(
        "--predictor-auxk-weight",
        type=float,
        default=0.0,
        help=(
            "Coefficient for code-space AuxK. Only valid with relu_topk; "
            "zero disables AuxK."
        ),
    )
    parser.add_argument(
        "--predictor-auxk",
        type=int,
        default=512,
        help="Number of currently dead predictor features used by AuxK",
    )
    parser.add_argument(
        "--predictor-dead-batches",
        type=int,
        default=500,
        help=(
            "Mark a predictor feature dead after this many joint-phase "
            "training microbatches without a main-path activation"
        ),
    )
    parser.add_argument(
        "--horizon-weighting",
        choices=HORIZON_WEIGHTING_MODES,
        default="inverse_probability",
        help=(
            "Correct the horizon imbalance induced by random span/context "
            "sampling. inverse_probability gives every horizon equal expected "
            "prediction-loss mass."
        ),
    )
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--sae-warmup-steps", type=int, default=4000)
    parser.add_argument("--prediction-ramp-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--predictor-lr", type=float, default=3e-4)
    parser.add_argument("--sae-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default="bfloat16")
    parser.add_argument("--validation-batches", type=int, default=64)
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
    if args.predictor_auxk_weight < 0:
        raise ValueError("predictor_auxk_weight must be non-negative")
    if args.predictor_auxk < 1 or args.predictor_dead_batches < 1:
        raise ValueError("AuxK size and dead-batch threshold must be positive")
    if args.predictor_auxk_weight > 0 and args.predictor_output != "relu_topk":
        raise ValueError("predictor AuxK is only valid with relu_topk output")
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
    if int(manifest["max_horizon"]) != int(manifest["max_span_length"]) - 1:
        raise ValueError("activation manifest has inconsistent horizon bounds")
    cfg = TransitionJEPAConfig(
        d_in=int(manifest["d_in"]),
        d_sae=args.d_sae,
        k=args.k,
        max_span_length=int(manifest["max_span_length"]),
        predictor_width=args.predictor_width,
        predictor_expansion=args.predictor_expansion,
        predictor_output=args.predictor_output,
        ema_decay=args.ema_decay,
        high_fraction=args.high_fraction,
        high_reconstruction_weight=args.high_reconstruction_weight,
    )
    model = TransitionJEPASAE(cfg).to(device)
    activity_tracker = (
        PredictorActivityTracker(
            cfg.d_high,
            args.predictor_dead_batches,
            device,
        )
        if cfg.predictor_output == "relu_topk"
        else None
    )
    horizon_probabilities = horizon_sampling_probabilities(
        int(manifest["min_span_length"]),
        int(manifest["max_span_length"]),
    )
    horizon_weight_table = horizon_loss_weight_table(
        int(manifest["min_span_length"]),
        int(manifest["max_span_length"]),
        args.horizon_weighting,
    ).to(device)
    normalization = manifest["normalization"]
    model.initialize_from_statistics(
        torch.tensor(normalization["mean"]),
        float(normalization["scalar_rms"]),
    )
    iterator = iter(
        RandomPairShardBatches(root, manifest, "train", args.batch_size, args.seed)
    )
    predictor_parameters = list(model.transition_predictor.parameters())
    predictor_ids = {id(parameter) for parameter in predictor_parameters}
    sae_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in predictor_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": predictor_parameters, "lr": args.predictor_lr, "base_lr": args.predictor_lr},
            {"params": sae_parameters, "lr": args.sae_lr, "base_lr": args.sae_lr},
        ],
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    history: list[dict[str, Any]] = []
    for step in trange(1, args.steps + 1, desc="high/low endpoint JEPA-SAE"):
        for group in optimizer.param_groups:
            group["lr"] = cosine_learning_rate(
                step,
                args.steps,
                float(group["base_lr"]),
                min(args.warmup_steps, max(1, args.steps // 10)),
                args.min_lr_ratio,
            )
        joint_step = max(0, step - args.sae_warmup_steps)
        active_prediction_weight = args.prediction_weight * min(
            1.0, joint_step / max(args.prediction_ramp_steps, 1)
        )
        phase = "sae_warmup" if joint_step == 0 else "joint"
        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(args.gradient_accumulation_steps):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in next(iterator).items()
            }
            with autocast_context(device, args.amp_dtype):
                loss, metrics = transition_jepa_loss(
                    model,
                    batch["context"],
                    batch["target"],
                    batch["horizon"],
                    active_prediction_weight,
                    span_length=batch["span_length"],
                    horizon_weight_table=horizon_weight_table,
                    activity_tracker=activity_tracker,
                    update_activity_tracker=joint_step > 0,
                    predictor_auxk_weight=args.predictor_auxk_weight,
                    predictor_auxk=args.predictor_auxk,
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
            metrics["prediction_weight"] = active_prediction_weight
            validation = evaluate_losses(
                model,
                root,
                manifest,
                args.batch_size,
                args.validation_batches,
                device,
                args.prediction_weight,
                args.amp_dtype if device.type == "cuda" else "none",
                args.seed + 1,
                horizon_weight_table,
                activity_tracker,
                args.predictor_auxk_weight,
                args.predictor_auxk,
            )
            history.append(
                {"step": step, "phase": phase, "train": metrics, "validation": validation}
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
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
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
        output_dir / "transition_jepa_sae.pt",
    )
    torch.save(
        {
            "architecture_id": ARCHITECTURE_ID,
            "state_dict": {
                key: value.cpu() for key, value in model.final_ema_sae_state_dict().items()
            },
            "config": asdict(cfg),
            "source_transition_checkpoint": "transition_jepa_sae.pt",
            "data_fingerprint": fingerprint,
            "source_config": source_config,
        },
        output_dir / "ema_sae.pt",
    )
    write_json(
        output_dir / "training_report.json",
        {
            "method": "high/low random-pair horizon-conditioned full-EMA JEPA-SAE",
            "architecture": {
                "id": ARCHITECTURE_ID,
                "d_high": cfg.d_high,
                "d_low": cfg.d_low,
                "k_high": cfg.k_high,
                "k_low": cfg.k_low,
                "max_span_length": cfg.max_span_length,
                "max_horizon": cfg.max_span_length - 1,
                "conditioning": "explicit token horizon h=t-k",
                "predictor_output": cfg.predictor_output,
                "predictor_output_bias_initialization": predictor_output_bias_init(
                    cfg.predictor_output
                ),
                "predictor_output_topk": (
                    cfg.k_high if cfg.predictor_output == "relu_topk" else None
                ),
                "predictor_auxk": {
                    "enabled": args.predictor_auxk_weight > 0,
                    "coefficient": args.predictor_auxk_weight,
                    "k_aux": args.predictor_auxk,
                    "dead_after_prediction_batches": args.predictor_dead_batches,
                    "dead_definition": (
                        "not selected by main ReLU+Top-K output during joint-phase "
                        "training microbatches"
                    ),
                    "target": "stopgrad(ReLU(z_target - z_main)) on dead coordinates",
                    "normalization": "squared code error / total target-code energy",
                    "final_activity": (
                        activity_tracker.summary()
                        if activity_tracker is not None
                        else None
                    ),
                },
                "high_role": "high-only reconstruction plus endpoint JEPA prediction",
                "low_role": "incremental full reconstruction only",
                "initialization": "from Pile activation mean/RMS; no unsplit SAE",
                "final_sae": "full EMA high/low encoder-decoder pair",
            },
            "data": {
                "dataset": manifest["dataset"],
                "fingerprint": fingerprint,
                "sampling": "random span length, endpoint, and non-endpoint context",
                "min_span_length": manifest["min_span_length"],
                "sequence_length": manifest["sequence_length"],
                "burn_in_tokens": manifest["burn_in_tokens"],
                "n_train_sequences": manifest["train"]["sequences"],
                "n_validation_sequences": manifest["validation"]["sequences"],
                "n_train_positions": manifest["train"]["positions"],
                "n_validation_positions": manifest["validation"]["positions"],
            },
            "horizon_balancing": {
                "mode": args.horizon_weighting,
                "sampling_probabilities": {
                    str(horizon): float(horizon_probabilities[horizon])
                    for horizon in range(1, cfg.max_span_length)
                },
                "prediction_loss_weights": {
                    str(horizon): float(horizon_weight_table[horizon].cpu())
                    for horizon in range(1, cfg.max_span_length)
                },
                "normalization": (
                    "E_sampling[weight(h)] = 1; each horizon has equal "
                    "expected prediction-loss mass"
                ),
                "reconstruction_loss_weighting": "unchanged",
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
