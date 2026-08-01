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
    ShuffledShardBatches,
    load_activation_manifest,
    manifest_fingerprint,
    validation_batches,
)
from .group_sae import topk_relu
from .io import write_json
from .training import (
    autocast_context,
    configure_accelerator,
    cosine_learning_rate,
)

ARCHITECTURE_ID = "high_low_fixed_endpoint_ema_sae_v2"


@dataclass
class TransitionJEPAConfig:
    d_in: int
    d_sae: int = 2048
    k: int = 32
    window_size: int = 10
    predictor_width: int = 256
    predictor_expansion: int = 2
    ema_decay: float = 0.996
    high_fraction: float = 0.2
    high_reconstruction_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least two")
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


class PositionConditionedPredictor(nn.Module):
    """Predict the fixed endpoint high code from each earlier high code."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        width = cfg.predictor_width
        hidden = cfg.predictor_expansion * width
        self.context_projection = nn.Linear(cfg.d_high, width, bias=False)
        self.position_embedding = nn.Embedding(cfg.window_size, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, cfg.d_high)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(
        self,
        context_code: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        if context_code.ndim == 2:
            context_code = context_code[:, None, :]
        if context_code.ndim != 3:
            raise ValueError("context_code must have shape [batch, contexts, d_high]")
        if context_positions.shape != (context_code.shape[1],):
            raise ValueError("context_positions must match the context axis")
        state = self.context_projection(context_code)
        queries = state + self.position_embedding(context_positions)[None]
        return F.softplus(self.output(self.mlp(queries)))


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
        self.transition_predictor = PositionConditionedPredictor(cfg)

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

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.decoder)
        return value + self.pre_bias if add_bias else value

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

    def predict_from_code(
        self,
        context_high: torch.Tensor,
        context_positions: torch.Tensor | None = None,
        sparse_output: bool = False,
    ) -> torch.Tensor:
        if context_high.shape[-1] == self.cfg.d_sae:
            context_high = context_high[..., : self.cfg.d_high]
        if context_high.ndim == 2:
            context_high = context_high[:, None, :]
        if context_high.shape[-1] != self.cfg.d_high:
            raise ValueError("predictor input must be a high-group code")
        if context_positions is None:
            context_positions = torch.arange(
                context_high.shape[1], device=context_high.device, dtype=torch.long
            )
        dense = self.transition_predictor(context_high, context_positions)
        return topk_relu(dense, self.cfg.k_high) if sparse_output else dense

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.cfg.window_size, self.cfg.d_in):
            raise ValueError(
                f"x must have shape [batch, {self.cfg.window_size}, {self.cfg.d_in}]"
            )
        codes = self.encode(x)
        high, low = self.split_code(codes)
        online_target_high = high[:, -1]
        online_target_low = low[:, -1]
        high_reconstruction = self.decode_high(online_target_high, ema=False)
        full_reconstruction = high_reconstruction + self.decode_low(
            online_target_low, ema=False, add_bias=False
        )
        with torch.no_grad():
            ema_target = self.encode_ema(x[:, -1])
            ema_target_high, ema_target_low = self.split_code(ema_target)
            ema_high_reconstruction = self.decode_high(ema_target_high, ema=True)
            ema_full_reconstruction = ema_high_reconstruction + self.decode_low(
                ema_target_low, ema=True, add_bias=False
            )
        prediction = self.predict_from_code(high[:, :-1])
        sparse_prediction = topk_relu(prediction, self.cfg.k_high)
        predictable_residual = self.decode_high(sparse_prediction, ema=True)
        target_codes = ema_target_high[:, None].expand_as(prediction)
        target_residual = x[:, -1, None, :].expand_as(predictable_residual)
        return {
            "codes": codes,
            "high_codes": high,
            "low_codes": low,
            "online_target_high": online_target_high,
            "online_target_low": online_target_low,
            "online_high_reconstruction": high_reconstruction,
            "online_target_reconstruction": full_reconstruction,
            "target_code": ema_target_high,
            "target_low_code": ema_target_low,
            "target_codes": target_codes,
            "target_high_reconstruction": ema_high_reconstruction,
            "target_reconstruction": ema_full_reconstruction,
            "predicted_codes": prediction,
            "sparse_predicted_codes": sparse_prediction,
            "predictable_residual": predictable_residual,
            "target_residual": target_residual,
            "innovation_residual": target_residual - predictable_residual,
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
    x: torch.Tensor,
    prediction_weight: float,
    residual_prediction_weight: float,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x)
    target = x[:, -1]
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
    prediction_loss = (1.0 - cosine + 0.25 * nrmse).mean()
    residual_prediction = (
        (outputs["predictable_residual"] - outputs["target_residual"])
        .float()
        .square()
        .mean()
        / ema_scale
    )
    loss = reconstruction + prediction_weight * (
        prediction_loss + residual_prediction_weight * residual_prediction
    )
    if not collect_metrics:
        return loss, {}
    precision, recall, jaccard = support_metrics(
        prediction.detach(), target_codes, model.cfg.k_high
    )
    return loss, {
        "loss": float(loss.detach()),
        "online_high_reconstruction_fvu": float(high_fvu.detach()),
        "online_reconstruction_fvu": float(full_fvu.detach()),
        "ema_high_reconstruction_fvu": float(ema_high_fvu.detach()),
        "ema_reconstruction_fvu": float(ema_full_fvu.detach()),
        "prediction_loss": float(prediction_loss.detach()),
        "code_cosine": float(cosine.mean().detach()),
        "code_nrmse": float(nrmse.mean().detach()),
        "support_precision": float(precision.mean()),
        "support_recall": float(recall.mean()),
        "support_jaccard": float(jaccard.mean()),
        "residual_prediction_fvu": float(residual_prediction.detach()),
        "high_l0": float((outputs["high_codes"] > 0).sum(dim=-1).float().mean()),
        "low_l0": float((outputs["low_codes"] > 0).sum(dim=-1).float().mean()),
    }


@torch.no_grad()
def evaluate_losses(
    model: TransitionJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    prediction_weight: float,
    residual_prediction_weight: float,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for x in validation_batches(root, manifest, batch_size, maximum_batches):
        x = x.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            _, metrics = transition_jepa_loss(
                model, x, prediction_weight, residual_prediction_weight
            )
        count += len(x)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + len(x) * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the high/low fixed-endpoint EMA JEPA-SAE"
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--high-fraction", type=float, default=0.2)
    parser.add_argument("--high-reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--predictor-width", type=int, default=256)
    parser.add_argument("--predictor-expansion", type=int, default=2)
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--residual-prediction-weight", type=float, default=0.1)
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
    cfg = TransitionJEPAConfig(
        d_in=int(manifest["d_in"]),
        d_sae=args.d_sae,
        k=args.k,
        window_size=int(manifest["window_size"]),
        predictor_width=args.predictor_width,
        predictor_expansion=args.predictor_expansion,
        ema_decay=args.ema_decay,
        high_fraction=args.high_fraction,
        high_reconstruction_weight=args.high_reconstruction_weight,
    )
    model = TransitionJEPASAE(cfg).to(device)
    normalization = manifest["normalization"]
    model.initialize_from_statistics(
        torch.tensor(normalization["mean"]),
        float(normalization["scalar_rms"]),
    )
    iterator = iter(
        ShuffledShardBatches(root, manifest, "train", args.batch_size, args.seed)
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
            batch = next(iterator).to(device, non_blocking=True)
            with autocast_context(device, args.amp_dtype):
                loss, metrics = transition_jepa_loss(
                    model,
                    batch,
                    active_prediction_weight,
                    args.residual_prediction_weight,
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
                args.residual_prediction_weight,
                args.amp_dtype if device.type == "cuda" else "none",
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
            "method": "high/low fixed-endpoint full-EMA JEPA-SAE",
            "architecture": {
                "id": ARCHITECTURE_ID,
                "d_high": cfg.d_high,
                "d_low": cfg.d_low,
                "k_high": cfg.k_high,
                "k_low": cfg.k_low,
                "target_position": cfg.window_size - 1,
                "high_role": "high-only reconstruction plus endpoint JEPA prediction",
                "low_role": "incremental full reconstruction only",
                "initialization": "from Pile activation mean/RMS; no unsplit SAE",
                "final_sae": "full EMA high/low encoder-decoder pair",
            },
            "data": {
                "dataset": manifest["dataset"],
                "fingerprint": fingerprint,
                "n_train_windows": manifest["train"]["windows"],
                "n_validation_windows": manifest["validation"]["windows"],
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
