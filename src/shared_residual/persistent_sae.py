from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .group_sae import topk_relu
from .io import torch_load, write_json
from .predictive_sae import (
    autocast_context,
    build_adamw,
    configure_accelerator,
    cosine_learning_rate,
    grouped_three_way_split,
)


@dataclass
class PersistentSAEConfig:
    d_in: int
    d_sae: int = 2048
    k: int = 32
    window_size: int = 10
    ema_decay: float = 0.996


class SparseEncoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_sae)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return topk_relu(self.linear(x), self.k)


class PersistentSparseAutoencoder(nn.Module):
    """A predictor-free SAE for sparse state persistence over ten tokens.

    The first residual in each window is the context. Its online SAE code z0 is
    matched directly and separately to each EMA target code z1, ..., z9. There
    is intentionally no target averaging, position embedding, or Transformer
    predictor in this model.
    """

    def __init__(self, cfg: PersistentSAEConfig):
        super().__init__()
        if cfg.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if cfg.k > cfg.d_sae:
            raise ValueError("k cannot exceed d_sae")
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("pre_scale", torch.ones(()))
        self.encoder = SparseEncoder(cfg.d_in, cfg.d_sae, cfg.k)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def initialize_from_data(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=(0, 1))
        self.pre_bias.copy_(mean.to(self.pre_bias))
        scale = (x - mean).square().mean().sqrt().clamp_min(1e-8)
        self.pre_scale.copy_(scale.to(self.pre_scale))
        self.normalize_decoder()
        self.encoder.linear.weight.copy_(self.decoder)
        self.encoder.linear.bias.zero_()
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def update_target_encoder(self, decay: float | None = None) -> None:
        rate = self.cfg.ema_decay if decay is None else decay
        for target, online in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters(),
        ):
            target.mul_(rate).add_(online.detach(), alpha=1.0 - rate)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.pre_bias) / self.pre_scale)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(
            (x - self.pre_bias.detach()) / self.pre_scale
        )

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        decoded = self.pre_scale * (z @ self.decoder)
        return decoded + self.pre_bias if add_bias else decoded

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.cfg.window_size:
            raise ValueError(
                "x must have shape [batch, "
                f"{self.cfg.window_size}, {self.cfg.d_in}]"
            )
        codes = self.encode(x)
        with torch.no_grad():
            target_codes = self.encode_target(x[:, 1:])
        return {
            "codes": codes,
            "reconstruction": self.decode(codes),
            "context_code": codes[:, 0],
            "target_codes": target_codes,
        }


def group_contrastive_loss(
    context: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Match z0 to every z_j while treating same-group rows as positives.

    This is averaged over offsets, never over target representations. Rows from
    paraphrases of the same underlying problem are not used as false negatives.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(context) < 2:
        return context.sum() * 0.0
    query = F.normalize(context.float(), dim=-1)
    key = F.normalize(targets.float(), dim=-1)
    positive_mask = group_ids[:, None].eq(group_ids[None, :])
    losses = []
    for offset in range(targets.shape[1]):
        logits = query @ key[:, offset].T / temperature
        numerator = torch.logsumexp(
            logits.masked_fill(~positive_mask, -torch.inf),
            dim=1,
        )
        denominator = torch.logsumexp(logits, dim=1)
        losses.append(-(numerator - denominator).mean())
    return torch.stack(losses).mean()


def persistent_loss(
    model: PersistentSparseAutoencoder,
    x: torch.Tensor,
    group_ids: torch.Tensor,
    persistence_weight: float,
    contrastive_weight: float,
    temperature: float,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x)
    scale = x.var(dim=(0, 1), unbiased=False).mean().clamp_min(1e-8)
    reconstruction = (outputs["reconstruction"] - x).square().mean() / scale
    context = outputs["context_code"][:, None, :]
    targets = outputs["target_codes"].detach()

    # Each z0-to-zj comparison is computed before reduction. Replacing targets
    # by their mean here would define a materially different experiment.
    cosine_by_example = F.cosine_similarity(context, targets, dim=-1)
    target_energy = targets.square().mean(dim=-1).clamp_min(1e-8)
    nrmse_by_example = (context - targets).square().mean(dim=-1) / target_energy
    direct_by_example = 1.0 - cosine_by_example + 0.25 * nrmse_by_example
    direct_persistence = direct_by_example.mean(dim=0).mean()
    contrastive = group_contrastive_loss(
        outputs["context_code"],
        targets,
        group_ids,
        temperature,
    )
    loss = (
        reconstruction
        + persistence_weight * direct_persistence
        + contrastive_weight * contrastive
    )

    metrics: dict[str, float] = {}
    if collect_metrics:
        context_active = context > 0
        target_active = targets > 0
        intersection = (context_active & target_active).sum(dim=-1).float()
        union = (context_active | target_active).sum(dim=-1).float()
        metrics = {
            "loss": float(loss.detach().item()),
            "reconstruction_fvu": float(reconstruction.detach().item()),
            "direct_persistence_loss": float(direct_persistence.detach().item()),
            "contrastive_loss": float(contrastive.detach().item()),
            "code_cosine": float(cosine_by_example.mean().detach().item()),
            "code_nrmse": float(nrmse_by_example.mean().detach().item()),
            "support_survival": float(
                intersection.div(
                    context_active.sum(dim=-1).float().clamp_min(1)
                ).mean().item()
            ),
            "support_jaccard": float(
                intersection.div(union.clamp_min(1)).mean().item()
            ),
            "l0": float(
                (outputs["codes"] > 0).float().sum(dim=-1).mean().item()
            ),
        }
        for offset in range(targets.shape[1]):
            metrics[f"offset_{offset + 1}_cosine"] = float(
                cosine_by_example[:, offset].mean().detach().item()
            )
            metrics[f"offset_{offset + 1}_survival"] = float(
                intersection[:, offset]
                .div(
                    context_active[:, 0].sum(dim=-1).float().clamp_min(1)
                )
                .mean()
                .item()
            )
    return loss, metrics


@torch.no_grad()
def evaluate_losses(
    model: PersistentSparseAutoencoder,
    loader: DataLoader,
    device: torch.device,
    persistence_weight: float,
    contrastive_weight: float,
    temperature: float,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for x, group_ids in loader:
        x = x.to(device, non_blocking=True)
        group_ids = group_ids.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            _, metrics = persistent_loss(
                model,
                x,
                group_ids,
                persistence_weight,
                contrastive_weight,
                temperature,
            )
        count += len(x)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + len(x) * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def encode_groups(
    metadata: list[dict[str, Any]],
    group_key: str,
) -> torch.Tensor:
    labels = [str(row.get(group_key, f"row-{index}")) for index, row in enumerate(metadata)]
    lookup = {label: index for index, label in enumerate(sorted(set(labels)))}
    return torch.tensor([lookup[label] for label in labels], dtype=torch.long)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a predictor-free z0-to-each-z1...z9 persistent SAE"
    )
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--objective",
        choices=["persistent", "standard"],
        default="persistent",
        help="persistent is proposed; standard is reconstruction-only control",
    )
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--persistence-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
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
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
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
    if x.ndim != 3:
        raise ValueError("activations must have shape [windows, tokens, d_model]")
    if x.shape[1] != 10:
        raise ValueError(
            "the prespecified persistent-SAE experiment requires exactly "
            f"10-token windows, but received {x.shape[1]}"
        )
    group_ids = encode_groups(metadata, args.group_key)
    train_indices, validation_indices, test_indices = grouped_three_way_split(
        metadata,
        args.validation_fraction,
        args.test_fraction,
        args.group_key,
        args.split_seed,
    )
    loader_options = {
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        TensorDataset(x[train_indices], group_ids[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        TensorDataset(x[validation_indices], group_ids[validation_indices]),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    cfg = PersistentSAEConfig(
        d_in=x.shape[-1],
        d_sae=args.d_sae,
        k=args.k,
        window_size=x.shape[1],
        ema_decay=args.ema_decay,
    )
    model = PersistentSparseAutoencoder(cfg).to(device)
    model.initialize_from_data(x[train_indices])
    optimizer, fused_optimizer = build_adamw(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        args.lr,
        args.weight_decay,
        device,
    )
    persistence_weight = (
        args.persistence_weight if args.objective == "persistent" else 0.0
    )
    contrastive_weight = (
        args.contrastive_weight if args.objective == "persistent" else 0.0
    )
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    for step in trange(1, args.steps + 1, desc=f"persistent SAE ({args.objective})"):
        learning_rate = cosine_learning_rate(
            step,
            args.steps,
            args.lr,
            min(args.warmup_steps, max(1, args.steps // 10)),
            args.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch, batch_groups = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch, batch_groups = next(iterator)
            batch = batch.to(device, non_blocking=True)
            batch_groups = batch_groups.to(device, non_blocking=True)
            with autocast_context(device, args.amp_dtype):
                loss, metrics = persistent_loss(
                    model,
                    batch,
                    batch_groups,
                    persistence_weight,
                    contrastive_weight,
                    args.temperature,
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
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.gradient_clip,
            )
            if should_log:
                metrics["gradient_norm"] = float(gradient_norm.item())
        optimizer.step()
        model.normalize_decoder()
        model.update_target_encoder()
        if should_log:
            metrics["learning_rate"] = learning_rate
            validation = evaluate_losses(
                model,
                validation_loader,
                device,
                persistence_weight,
                contrastive_weight,
                args.temperature,
                args.amp_dtype if device.type == "cuda" else "none",
            )
            history.append(
                {
                    "step": step,
                    "phase": args.objective,
                    "train": metrics,
                    "validation": validation,
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
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "config": asdict(cfg),
            "train_args": vars(args),
            "source_config": bundle.get("config", {}),
            "split": split,
        },
        output_dir / "persistent_sae.pt",
    )
    write_json(
        output_dir / "training_report.json",
        {
            "method": (
                "direct individual z0-to-z1...z9 persistent SAE"
                if args.objective == "persistent"
                else "standard reconstruction-only SAE"
            ),
            "objective": args.objective,
            "target_aggregation": "none; nine losses are computed independently",
            "architecture": {
                "context": "online SAE code at token 0",
                "targets": "stop-gradient EMA SAE codes at tokens 1..9",
                "normalization": "dataset mean and scalar RMS",
                "predictor": None,
                "position_embeddings": False,
            },
            "loss_weights": {
                "persistence": persistence_weight,
                "group_contrastive": contrastive_weight,
                "reconstruction": 1.0,
            },
            "n_train_windows": len(train_indices),
            "n_validation_windows": len(validation_indices),
            "n_locked_test_windows": len(test_indices),
            "accelerator": {
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "amp_dtype": args.amp_dtype if device.type == "cuda" else "none",
                "tf32_enabled": device.type == "cuda",
                "fused_adamw": fused_optimizer,
                "micro_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
    print(f"saved {args.objective} checkpoint and report to {output_dir}")


if __name__ == "__main__":
    main()
