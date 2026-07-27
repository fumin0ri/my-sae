from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
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
    build_adamw,
    configure_accelerator,
    cosine_learning_rate,
)


@dataclass
class StandardSAEConfig:
    d_in: int
    d_sae: int = 2048
    k: int = 32
    window_size: int = 10


class SparseEncoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_sae)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return topk_relu(self.linear(x), self.k)


class StandardSparseAutoencoder(nn.Module):
    """Top-K SAE trained to reconstruct every residual in a window."""

    def __init__(self, cfg: StandardSAEConfig):
        super().__init__()
        if cfg.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if cfg.k > cfg.d_sae:
            raise ValueError("k cannot exceed d_sae")
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("pre_scale", torch.ones(()))
        self.encoder = SparseEncoder(cfg.d_in, cfg.d_sae, cfg.k)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))

    @torch.no_grad()
    def initialize_from_data(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=(0, 1))
        scale = (x - mean).square().mean().sqrt().clamp_min(1e-8)
        self.initialize_from_statistics(mean, scale)

    @torch.no_grad()
    def initialize_from_statistics(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor | float,
    ) -> None:
        if mean.shape != self.pre_bias.shape:
            raise ValueError("normalization mean does not match residual width")
        self.pre_bias.copy_(mean.to(self.pre_bias))
        self.pre_scale.copy_(
            torch.as_tensor(scale).to(self.pre_scale)
        )
        self.normalize_decoder()
        self.encoder.linear.weight.copy_(self.decoder)
        self.encoder.linear.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.div_(
            self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.pre_bias) / self.pre_scale)

    def decode(
        self,
        z: torch.Tensor,
        add_bias: bool = True,
    ) -> torch.Tensor:
        decoded = self.pre_scale * (z @ self.decoder)
        return decoded + self.pre_bias if add_bias else decoded

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.cfg.window_size:
            raise ValueError(
                "x must have shape [batch, "
                f"{self.cfg.window_size}, {self.cfg.d_in}]"
            )
        codes = self.encode(x)
        return {
            "codes": codes,
            "reconstruction": self.decode(codes),
        }


def standard_sae_loss(
    model: StandardSparseAutoencoder,
    x: torch.Tensor,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x)
    scale = x.float().var(
        dim=(0, 1),
        unbiased=False,
    ).mean().clamp_min(1e-8)
    loss = (
        (outputs["reconstruction"] - x)
        .float()
        .square()
        .mean()
        .div(scale)
    )
    metrics = {}
    if collect_metrics:
        metrics = {
            "loss": float(loss.detach().item()),
            "reconstruction_fvu": float(loss.detach().item()),
            "sae_l0": float(
                (outputs["codes"] > 0).sum(dim=-1).float().mean().item()
            ),
            "code_norm": float(
                outputs["codes"]
                .float()
                .norm(dim=-1)
                .mean()
                .detach()
                .item()
            ),
        }
    return loss, metrics


@torch.no_grad()
def evaluate_losses(
    model: StandardSparseAutoencoder,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
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
            _, metrics = standard_sae_loss(model, x)
        count += len(x)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + len(x) * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def trainable_parameters(
    model: nn.Module,
) -> Iterable[nn.Parameter]:
    return (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the reconstruction-only SAE used to initialize JEPA"
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    if args.steps < 1:
        raise ValueError("--steps must be positive")
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
    normalization = manifest["normalization"]
    iterator = iter(
        ShuffledShardBatches(
            root,
            manifest,
            "train",
            args.batch_size,
            args.seed,
        )
    )
    cfg = StandardSAEConfig(
        d_in=int(manifest["d_in"]),
        d_sae=args.d_sae,
        k=args.k,
        window_size=int(manifest["window_size"]),
    )
    model = StandardSparseAutoencoder(cfg).to(device)
    model.initialize_from_statistics(
        torch.tensor(normalization["mean"]),
        torch.tensor(float(normalization["scalar_rms"])),
    )
    optimizer, fused_optimizer = build_adamw(
        trainable_parameters(model),
        args.lr,
        args.weight_decay,
        device,
    )
    history: list[dict[str, Any]] = []
    for step in trange(1, args.steps + 1, desc="standard SAE"):
        learning_rate = cosine_learning_rate(
            step,
            args.steps,
            args.lr,
            min(args.warmup_steps, max(1, args.steps // 10)),
            args.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        should_log = (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        )
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(args.gradient_accumulation_steps):
            batch = next(iterator)
            batch = batch.to(device, non_blocking=True)
            with autocast_context(device, args.amp_dtype):
                loss, metrics = standard_sae_loss(
                    model,
                    batch,
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
        model.normalize_decoder()
        if should_log:
            metrics["learning_rate"] = learning_rate
            history.append(
                {
                    "step": step,
                    "phase": "standard",
                    "train": metrics,
                    "validation": evaluate_losses(
                        model,
                        root,
                        manifest,
                        args.batch_size,
                        args.validation_batches,
                        device,
                        (
                            args.amp_dtype
                            if device.type == "cuda"
                            else "none"
                        ),
                    ),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "config": asdict(cfg),
            "train_args": vars(args),
            "source_config": {
                "model": manifest["model"],
                "resolved_model_revision": manifest.get(
                    "resolved_model_revision"
                ),
                "layer": manifest["layer"],
                "layer_path": manifest["layer_path"],
                "hook_point": manifest["hook_point"],
            },
            "activation_manifest": str(Path(args.activation_manifest)),
            "data_fingerprint": fingerprint,
        },
        output_dir / "standard_sae.pt",
    )
    write_json(
        output_dir / "training_report.json",
        {
            "method": "standard reconstruction-only Top-K SAE",
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
    print(f"saved standard SAE checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
