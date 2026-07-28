from __future__ import annotations

import argparse
import copy
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
from .io import torch_load, write_json
from .standard_sae import (
    StandardSAEConfig,
    StandardSparseAutoencoder,
)
from .training import (
    autocast_context,
    configure_accelerator,
    cosine_learning_rate,
)


@dataclass
class TransitionJEPAConfig:
    d_in: int
    d_sae: int = 2048
    k: int = 32
    window_size: int = 10
    predictor_width: int = 256
    predictor_expansion: int = 2
    ema_decay: float = 0.996


class OffsetConditionedPredictor(nn.Module):
    """Predict a future sparse code from z0 and a discrete token offset."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        width = cfg.predictor_width
        hidden = cfg.predictor_expansion * width
        self.context_projection = nn.Linear(cfg.d_sae, width, bias=False)
        self.offset_embedding = nn.Embedding(cfg.window_size, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, cfg.d_sae)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(
        self,
        context_code: torch.Tensor,
        offsets: torch.Tensor,
        use_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_context:
            state = self.context_projection(context_code)
        else:
            state = torch.zeros(
                len(context_code),
                self.context_projection.out_features,
                device=context_code.device,
                dtype=context_code.dtype,
            )
        queries = state[:, None, :] + self.offset_embedding(offsets)[None, :, :]
        hidden = self.mlp(queries)
        # A dense, non-negative prediction keeps gradients smooth at support
        # boundaries. Top-K is applied only for support metrics and interventions.
        return F.softplus(self.output(hidden)), state


class TransitionJEPASAE(StandardSparseAutoencoder):
    """Sparse offset-conditioned forecasting over a frozen LLM trajectory."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__(
            StandardSAEConfig(
                d_in=cfg.d_in,
                d_sae=cfg.d_sae,
                k=cfg.k,
                window_size=cfg.window_size,
            )
        )
        self.cfg = cfg
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.transition_predictor = OffsetConditionedPredictor(cfg)

    @torch.no_grad()
    def load_standard_sae(self, checkpoint: dict[str, Any]) -> None:
        source_cfg = checkpoint["config"]
        required = {
            "d_in": self.cfg.d_in,
            "d_sae": self.cfg.d_sae,
            "k": self.cfg.k,
            "window_size": self.cfg.window_size,
        }
        mismatches = {
            key: (source_cfg.get(key), expected)
            for key, expected in required.items()
            if source_cfg.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"standard SAE checkpoint mismatch: {mismatches}")
        missing, unexpected = self.load_state_dict(
            checkpoint["state_dict"],
            strict=False,
        )
        allowed_missing = {
            name
            for name in self.state_dict()
            if name.startswith(("target_encoder.", "transition_predictor."))
        }
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "could not initialize from standard SAE: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def update_target_encoder(self, decay: float | None = None) -> None:
        rate = self.cfg.ema_decay if decay is None else decay
        for target, online in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters(),
        ):
            target.mul_(rate).add_(online.detach(), alpha=1.0 - rate)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(
            (x - self.pre_bias.detach()) / self.pre_scale
        )

    def set_sae_trainable(self, trainable: bool) -> None:
        self.pre_bias.requires_grad_(trainable)
        self.decoder.requires_grad_(trainable)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def predict_from_code(
        self,
        context_code: torch.Tensor,
        offsets: torch.Tensor | None = None,
        use_context: bool = True,
        sparse_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if offsets is None:
            offsets = torch.arange(
                1,
                self.cfg.window_size,
                device=context_code.device,
                dtype=torch.long,
            )
        dense, state = self.transition_predictor(
            context_code,
            offsets,
            use_context=use_context,
        )
        return (
            topk_relu(dense, self.cfg.k) if sparse_output else dense,
            state,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_context: bool = True,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.cfg.window_size:
            raise ValueError(
                "x must have shape [batch, "
                f"{self.cfg.window_size}, {self.cfg.d_in}]"
            )
        codes = self.encode(x)
        reconstruction = self.decode(codes)
        with torch.no_grad():
            target_codes = self.encode_target(x[:, 1:])
        predicted_codes, context_state = self.predict_from_code(
            codes[:, 0],
            use_context=use_context,
        )
        sparse_prediction = topk_relu(predicted_codes, self.cfg.k)
        predictable_residual = self.decode(sparse_prediction)
        return {
            "codes": codes,
            "reconstruction": reconstruction,
            "context_code": codes[:, 0],
            "target_codes": target_codes,
            "predicted_codes": predicted_codes,
            "sparse_predicted_codes": sparse_prediction,
            "target_residual": x[:, 1:],
            "predictable_residual": predictable_residual,
            "innovation_residual": x[:, 1:] - predictable_residual,
            "context_state": context_state,
        }


def support_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    k: int,
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
    variance_weight: float,
    variance_target: float,
    use_context: bool,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x, use_context=use_context)
    residual_scale = x.var(dim=(0, 1), unbiased=False).mean().clamp_min(1e-8)
    reconstruction = (
        outputs["reconstruction"] - x
    ).square().mean() / residual_scale

    prediction = outputs["predicted_codes"]
    target = outputs["target_codes"].detach()
    cosine_by_example = F.cosine_similarity(prediction, target, dim=-1)
    target_energy = target.square().mean(dim=-1).clamp_min(1e-8)
    nrmse_by_example = (
        prediction - target
    ).square().mean(dim=-1) / target_energy
    prediction_by_example = 1.0 - cosine_by_example + 0.25 * nrmse_by_example
    prediction_loss = prediction_by_example.mean()
    residual_prediction = (
        outputs["predictable_residual"] - outputs["target_residual"]
    ).square().mean() / residual_scale

    if use_context and variance_weight > 0 and len(x) > 1:
        state_std = outputs["context_state"].float().std(
            dim=0,
            unbiased=False,
        )
        variance_loss = F.relu(variance_target - state_std).mean()
    else:
        variance_loss = prediction.sum() * 0.0
    loss = (
        reconstruction
        + prediction_weight
        * (prediction_loss + residual_prediction_weight * residual_prediction)
        + variance_weight * variance_loss
    )

    metrics: dict[str, float] = {}
    if collect_metrics:
        precision, recall, jaccard = support_metrics(
            prediction.detach(),
            target,
            model.cfg.k,
        )
        sparse_prediction = outputs["sparse_predicted_codes"]
        metrics = {
            "loss": float(loss.detach().item()),
            "reconstruction_fvu": float(reconstruction.detach().item()),
            "prediction_loss": float(prediction_loss.detach().item()),
            "code_cosine": float(cosine_by_example.mean().detach().item()),
            "code_nrmse": float(nrmse_by_example.mean().detach().item()),
            "support_precision": float(precision.mean().item()),
            "support_recall": float(recall.mean().item()),
            "support_jaccard": float(jaccard.mean().item()),
            "residual_prediction_fvu": float(residual_prediction.detach().item()),
            "variance_loss": float(variance_loss.detach().item()),
            "sae_l0": float(
                (outputs["codes"] > 0).sum(dim=-1).float().mean().item()
            ),
            "predictor_dense_norm": float(
                prediction.float().norm(dim=-1).mean().detach().item()
            ),
            "predictor_topk_norm": float(
                sparse_prediction.float().norm(dim=-1).mean().detach().item()
            ),
            "target_code_norm": float(
                target.float().norm(dim=-1).mean().item()
            ),
            "innovation_energy_fraction": float(
                outputs["innovation_residual"]
                .float()
                .square()
                .sum(dim=-1)
                .mean()
                .div(
                    (outputs["target_residual"] - model.pre_bias)
                    .float()
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .clamp_min(1e-8)
                )
                .detach()
                .item()
            ),
        }
        for offset in range(target.shape[1]):
            metrics[f"offset_{offset + 1}_cosine"] = float(
                cosine_by_example[:, offset].mean().detach().item()
            )
            metrics[f"offset_{offset + 1}_nrmse"] = float(
                nrmse_by_example[:, offset].mean().detach().item()
            )
            metrics[f"offset_{offset + 1}_support_recall"] = float(
                recall[:, offset].mean().item()
            )
    return loss, metrics


def build_optimizer(
    model: TransitionJEPASAE,
    objective: str,
    predictor_lr: float,
    sae_lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.optim.AdamW, bool]:
    groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for parameter in model.transition_predictor.parameters()
                if parameter.requires_grad
            ],
            "lr": predictor_lr,
            "base_lr": predictor_lr,
        }
    ]
    sae_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("transition_predictor.")
        and not name.startswith("target_encoder.")
    ]
    if sae_parameters:
        groups.append(
            {
                "params": sae_parameters,
                "lr": sae_lr,
                "base_lr": sae_lr,
            }
        )
    fused = device.type == "cuda"
    return (
        torch.optim.AdamW(
            groups,
            weight_decay=weight_decay,
            fused=fused,
        ),
        fused,
    )


def add_sae_parameter_group(
    optimizer: torch.optim.AdamW,
    model: TransitionJEPASAE,
    sae_lr: float,
) -> None:
    sae_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("transition_predictor.")
        and not name.startswith("target_encoder.")
    ]
    if not sae_parameters:
        raise ValueError("no trainable SAE parameters to add")
    optimizer.add_param_group(
        {
            "params": sae_parameters,
            "lr": sae_lr,
            "base_lr": sae_lr,
        }
    )


def set_scheduled_learning_rates(
    optimizer: torch.optim.AdamW,
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> None:
    for group in optimizer.param_groups:
        base_lr = float(group["base_lr"])
        group["lr"] = cosine_learning_rate(
            step,
            total_steps,
            base_lr,
            warmup_steps,
            minimum_ratio,
        )


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
    variance_weight: float,
    variance_target: float,
    use_context: bool,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for x in validation_batches(
        root,
        manifest,
        batch_size,
        maximum_batches,
    ):
        x = x.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            _, metrics = transition_jepa_loss(
                model,
                x,
                prediction_weight,
                residual_prediction_weight,
                variance_weight,
                variance_target,
                use_context,
            )
        count += len(x)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + len(x) * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an offset-conditioned JEPA-SAE from a standard SAE checkpoint"
        )
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--objective",
        choices=["joint", "fixed", "k_only"],
        default="joint",
    )
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--predictor-width", type=int, default=256)
    parser.add_argument("--predictor-expansion", type=int, default=2)
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--residual-prediction-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--variance-target", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--predictor-warmup-steps", type=int, default=800)
    parser.add_argument("--prediction-ramp-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--predictor-lr", type=float, default=3e-4)
    parser.add_argument("--sae-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if not 0 <= args.predictor_warmup_steps < args.steps:
        raise ValueError("--predictor-warmup-steps must lie in [0, steps)")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
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
    standard_checkpoint = torch_load(args.init_checkpoint)
    if standard_checkpoint.get("data_fingerprint") != fingerprint:
        raise ValueError(
            "standard SAE checkpoint must use the same Pile activation manifest"
        )
    iterator = iter(
        ShuffledShardBatches(
            root,
            manifest,
            "train",
            args.batch_size,
            args.seed,
        )
    )
    cfg = TransitionJEPAConfig(
        d_in=int(manifest["d_in"]),
        d_sae=args.d_sae,
        k=args.k,
        window_size=int(manifest["window_size"]),
        predictor_width=args.predictor_width,
        predictor_expansion=args.predictor_expansion,
        ema_decay=args.ema_decay,
    )
    model = TransitionJEPASAE(cfg).to(device)
    model.load_standard_sae(standard_checkpoint)
    source_config = standard_checkpoint.get("source_config", {})
    del standard_checkpoint
    use_context = args.objective != "k_only"
    model.set_sae_trainable(False)
    optimizer, fused_optimizer = build_optimizer(
        model,
        args.objective,
        args.predictor_lr,
        args.sae_lr,
        args.weight_decay,
        device,
    )
    history: list[dict[str, Any]] = []
    phase = "predictor_warmup" if args.objective == "joint" else args.objective
    for step in trange(
        1,
        args.steps + 1,
        desc=f"transition JEPA ({args.objective})",
    ):
        if (
            args.objective == "joint"
            and step == args.predictor_warmup_steps + 1
        ):
            phase = "joint"
            model.set_sae_trainable(True)
            # Preserve predictor weights and optimizer moments learned during
            # warm-up while adding the newly unfrozen SAE parameters.
            add_sae_parameter_group(
                optimizer,
                model,
                args.sae_lr,
            )
        phase_step = (
            step - args.predictor_warmup_steps
            if phase == "joint"
            else step
        )
        phase_steps = (
            args.steps - args.predictor_warmup_steps
            if phase == "joint"
            else (
                args.predictor_warmup_steps
                if phase == "predictor_warmup"
                else args.steps
            )
        )
        set_scheduled_learning_rates(
            optimizer,
            phase_step,
            max(phase_steps, 1),
            min(args.warmup_steps, max(1, phase_steps // 10)),
            args.min_lr_ratio,
        )
        if phase == "joint":
            ramp_progress = min(
                1.0,
                phase_step / max(args.prediction_ramp_steps, 1),
            )
            active_prediction_weight = args.prediction_weight * ramp_progress
        else:
            active_prediction_weight = args.prediction_weight
        should_log = (
            step == 1
            or step % args.log_every == 0
            or step in {args.predictor_warmup_steps, args.steps}
        )
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(args.gradient_accumulation_steps):
            batch = next(iterator)
            batch = batch.to(device, non_blocking=True)
            with autocast_context(device, args.amp_dtype):
                loss, metrics = transition_jepa_loss(
                    model,
                    batch,
                    active_prediction_weight,
                    args.residual_prediction_weight,
                    args.variance_weight if use_context else 0.0,
                    args.variance_target,
                    use_context,
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
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model),
                args.gradient_clip,
            )
            if should_log:
                metrics["gradient_norm"] = float(gradient_norm.item())
        optimizer.step()
        if phase == "joint":
            model.normalize_decoder()
            model.update_target_encoder()
        if should_log:
            metrics["prediction_weight"] = active_prediction_weight
            metrics["predictor_learning_rate"] = float(
                optimizer.param_groups[0]["lr"]
            )
            validation = evaluate_losses(
                model,
                root,
                manifest,
                args.batch_size,
                args.validation_batches,
                device,
                args.prediction_weight,
                args.residual_prediction_weight,
                args.variance_weight if use_context else 0.0,
                args.variance_target,
                use_context,
                args.amp_dtype if device.type == "cuda" else "none",
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
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "config": asdict(cfg),
        "train_args": vars(args),
        "source_config": source_config,
        "activation_manifest": str(Path(args.activation_manifest)),
        "data_fingerprint": fingerprint,
        "standard_checkpoint": str(Path(args.init_checkpoint)),
    }
    torch.save(checkpoint, output_dir / "transition_jepa_sae.pt")
    write_json(
        output_dir / "training_report.json",
        {
            "method": {
                "joint": "joint offset-conditioned JEPA-SAE",
                "fixed": "fixed standard SAE plus offset-conditioned predictor",
                "k_only": "offset-only predictor with no z0 input",
            }[args.objective],
            "objective": args.objective,
            "scientific_interpretation": (
                "P(z0, k) estimates the forecastable component of z_k; it is "
                "not a deterministic transition without intervening tokens."
            ),
            "architecture": {
                "window_size": cfg.window_size,
                "final_offset": cfg.window_size - 1,
                "context": "Top-K online SAE code at h0",
                "targets": (
                    "stop-gradient EMA SAE codes at h1..."
                    f"h{cfg.window_size - 1}"
                ),
                "predictor": "offset-conditioned MLP with dense softplus output",
                "target_aggregation": "none",
                "sae_reconstruction_positions": (
                    f"all {cfg.window_size} positions"
                ),
                "normalization": "dataset mean and scalar RMS",
            },
            "optimizer_budget": {
                "total_steps": args.steps,
                "predictor_warmup_steps": (
                    args.predictor_warmup_steps
                    if args.objective == "joint"
                    else args.steps
                ),
                "joint_steps": (
                    args.steps - args.predictor_warmup_steps
                    if args.objective == "joint"
                    else 0
                ),
            },
            "data": {
                "dataset": manifest["dataset"],
                "fingerprint": fingerprint,
                "n_train_windows": manifest["train"]["windows"],
                "n_validation_windows": manifest["validation"]["windows"],
                "train_domain_counts": manifest["train"]["domain_counts"],
                "validation_domain_counts": manifest["validation"][
                    "domain_counts"
                ],
            },
            "accelerator": {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "amp_dtype": (
                    args.amp_dtype if device.type == "cuda" else "none"
                ),
                "tf32_enabled": device.type == "cuda",
                "fused_adamw": fused_optimizer,
                "micro_batch_size": args.batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "effective_batch_size": (
                    args.batch_size * args.gradient_accumulation_steps
                ),
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else None
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 2**30
                    if device.type == "cuda"
                    else None
                ),
            },
            "history": history,
            "final_validation": history[-1]["validation"],
        },
    )
    print(f"saved {args.objective} transition-JEPA checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
