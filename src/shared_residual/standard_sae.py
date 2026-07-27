from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .group_sae import topk_relu
from .io import torch_load, write_json
from .training import (
    autocast_context,
    build_adamw,
    configure_accelerator,
    cosine_learning_rate,
    grouped_three_way_split,
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
        self.pre_bias.copy_(mean.to(self.pre_bias))
        scale = (x - mean).square().mean().sqrt().clamp_min(1e-8)
        self.pre_scale.copy_(scale.to(self.pre_scale))
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
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for (x,) in loader:
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
    parser.add_argument("--activations", required=True)
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5")
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    configure_accelerator(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bundle = torch_load(args.activations)
    x = bundle["activations"].float()
    metadata = bundle["metadata"]
    source_config = bundle.get("config", {})
    del bundle
    if x.ndim != 3 or x.shape[1] != 10:
        raise ValueError("standard SAE requires [windows, 10, d_model]")
    train_indices, validation_indices, test_indices = grouped_three_way_split(
        metadata,
        args.validation_fraction,
        args.test_fraction,
        args.group_key,
        args.split_seed,
    )
    loader_options: dict[str, Any] = {
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        TensorDataset(x[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        TensorDataset(x[validation_indices]),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    cfg = StandardSAEConfig(
        d_in=x.shape[-1],
        d_sae=args.d_sae,
        k=args.k,
        window_size=x.shape[1],
    )
    model = StandardSparseAutoencoder(cfg).to(device)
    model.initialize_from_data(x[train_indices])
    optimizer, fused_optimizer = build_adamw(
        trainable_parameters(model),
        args.lr,
        args.weight_decay,
        device,
    )
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
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
            try:
                (batch,) = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                (batch,) = next(iterator)
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
                        validation_loader,
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
    split = {
        "group_key": args.group_key,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
    }
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "config": asdict(cfg),
            "train_args": vars(args),
            "source_config": source_config,
            "split": split,
        },
        output_dir / "standard_sae.pt",
    )
    write_json(
        output_dir / "training_report.json",
        {
            "method": "standard reconstruction-only Top-K SAE",
            "n_train_windows": len(train_indices),
            "n_validation_windows": len(validation_indices),
            "n_locked_test_windows": len(test_indices),
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
