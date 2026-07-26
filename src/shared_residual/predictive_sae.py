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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .group_sae import topk_relu
from .io import torch_load, write_json


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


@dataclass(frozen=True)
class SpanSpec:
    context_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    target_offsets: tuple[int, ...]
    target_size: int
    gap: int
    context_mode: str


def make_span_spec(
    window_size: int,
    context_width: int,
    target_size: int,
    gap: int,
    context_mode: str = "causal",
    target_start: int | None = None,
    generator: torch.Generator | None = None,
) -> SpanSpec:
    """Construct a leakage-aware context/target mask.

    ``causal`` uses only positions before the target. In a decoder-only model,
    residuals after the target have attended to the target tokens and are
    therefore an information leak. ``retrospective`` is retained as an explicit
    ablation and uses context on both sides.
    """
    if context_mode not in {"causal", "retrospective"}:
        raise ValueError("context_mode must be 'causal' or 'retrospective'")
    if min(window_size, context_width, target_size, gap) < 1:
        raise ValueError("window, context, target, and gap sizes must be positive")

    if context_mode == "causal":
        minimum_start = context_width + gap
        maximum_start = window_size - target_size
        if maximum_start < minimum_start:
            raise ValueError(
                f"window {window_size} is too short for context={context_width}, "
                f"gap={gap}, target={target_size}"
            )
        if target_start is None:
            target_start = int(
                torch.randint(
                    minimum_start,
                    maximum_start + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
        if not minimum_start <= target_start <= maximum_start:
            raise ValueError("target_start does not fit the requested causal mask")
        context_end = target_start - gap
        context = tuple(range(context_end - context_width, context_end))
    else:
        left_width = context_width // 2
        right_width = context_width - left_width
        minimum_start = left_width + gap
        maximum_start = window_size - target_size - gap - right_width
        if maximum_start < minimum_start:
            raise ValueError(
                f"window {window_size} is too short for retrospective "
                f"context={context_width}, gap={gap}, target={target_size}"
            )
        if target_start is None:
            target_start = int(
                torch.randint(
                    minimum_start,
                    maximum_start + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
        if not minimum_start <= target_start <= maximum_start:
            raise ValueError("target_start does not fit the requested retrospective mask")
        left = range(target_start - gap - left_width, target_start - gap)
        target_end = target_start + target_size
        right = range(target_end + gap, target_end + gap + right_width)
        context = tuple(left) + tuple(right)

    target = tuple(range(target_start, target_start + target_size))
    offsets = tuple(gap + 1 + index for index in range(target_size))
    if set(context) & set(target):
        raise AssertionError("context and target masks overlap")
    return SpanSpec(
        context_indices=context,
        target_indices=target,
        target_offsets=offsets,
        target_size=target_size,
        gap=gap,
        context_mode=context_mode,
    )


@dataclass
class PredictiveSAEConfig:
    d_in: int
    d_sae: int = 2048
    k: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    context_width: int = 24
    max_window_size: int = 64
    max_target_size: int = 8
    max_gap: int = 8
    context_mode: str = "causal"
    ema_decay: float = 0.996


class SparseEncoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_sae)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return topk_relu(self.linear(x), self.k)


class PredictiveSparseAutoencoder(nn.Module):
    """Top-K SAE whose residual-space features are trained to be predictable.

    The online SAE reconstructs the original LLM residual stream. A small JEPA
    predictor reads sparse codes from a masked context and predicts sparse codes
    produced by an EMA target encoder at a future target span. Decoder rows stay
    in the original residual space, so predicted features can be patched or
    ablated directly in the frozen language model.
    """

    def __init__(self, cfg: PredictiveSAEConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.k > cfg.d_sae:
            raise ValueError("k cannot exceed d_sae")
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.encoder = SparseEncoder(cfg.d_in, cfg.d_sae, cfg.k)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))

        self.context_projection = nn.Linear(cfg.d_sae, cfg.d_model, bias=False)
        self.context_positions = nn.Embedding(cfg.max_window_size, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(
            layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )
        self.target_offsets = nn.Embedding(
            cfg.max_gap + cfg.max_target_size + 2,
            cfg.d_model,
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.d_sae),
        )
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def initialize_from_data(self, x: torch.Tensor) -> None:
        self.pre_bias.copy_(x.mean(dim=(0, 1)).to(self.pre_bias))
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
        return self.encoder(x - self.pre_bias)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x - self.pre_bias.detach())

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        decoded = z @ self.decoder
        return decoded + self.pre_bias if add_bias else decoded

    def predict_codes(
        self,
        x: torch.Tensor,
        span: SpanSpec,
        sparse_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_indices = torch.as_tensor(
            span.context_indices,
            device=x.device,
            dtype=torch.long,
        )
        context = x.index_select(1, context_indices)
        context_codes = self.encode(context)
        hidden = self.context_projection(context_codes)
        position_ids = torch.arange(
            hidden.shape[1],
            device=x.device,
            dtype=torch.long,
        )
        hidden = hidden + self.context_positions(position_ids)[None, :, :]
        hidden = self.context_transformer(hidden)
        # The final observed position is the natural summary for the causal
        # model. Mean pooling is used only for the leakage-labelled ablation.
        state = (
            hidden[:, -1]
            if span.context_mode == "causal"
            else hidden.mean(dim=1)
        )
        offsets = torch.as_tensor(
            span.target_offsets,
            device=x.device,
            dtype=torch.long,
        )
        queries = state[:, None, :] + self.target_offsets(offsets)[None, :, :]
        dense = self.predictor(queries)
        return (topk_relu(dense, self.cfg.k) if sparse_output else dense), state

    def forward(self, x: torch.Tensor, span: SpanSpec) -> dict[str, torch.Tensor]:
        codes = self.encode(x)
        reconstruction = self.decode(codes)
        target_indices = torch.as_tensor(
            span.target_indices,
            device=x.device,
            dtype=torch.long,
        )
        target = x.index_select(1, target_indices)
        with torch.no_grad():
            target_codes = self.encode_target(target)
        predicted_codes, context_state = self.predict_codes(x, span)
        predictable = self.decode(predicted_codes)
        innovation = target - predictable
        return {
            "codes": codes,
            "reconstruction": reconstruction,
            "target": target,
            "target_codes": target_codes,
            "predicted_codes": predicted_codes,
            "predictable": predictable,
            "innovation": innovation,
            "context_state": context_state,
        }


def predictive_loss(
    model: PredictiveSparseAutoencoder,
    x: torch.Tensor,
    span: SpanSpec,
    prediction_weight: float,
    residual_prediction_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x, span)
    scale = x.var(dim=(0, 1), unbiased=False).mean().clamp_min(1e-8)
    reconstruction = (outputs["reconstruction"] - x).square().mean() / scale

    prediction = outputs["predicted_codes"]
    target = outputs["target_codes"].detach()
    active_target = target.square().mean().clamp_min(1e-8)
    code_nrmse = (prediction - target).square().mean() / active_target
    cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
    code_prediction = 1.0 - cosine + 0.25 * code_nrmse
    residual_prediction = (
        outputs["predictable"] - outputs["target"]
    ).square().mean() / scale
    loss = reconstruction + prediction_weight * (
        code_prediction + residual_prediction_weight * residual_prediction
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "reconstruction_fvu": float(reconstruction.detach().item()),
        "code_prediction_loss": float(code_prediction.detach().item()),
        "code_cosine": float(cosine.detach().item()),
        "code_nrmse": float(code_nrmse.detach().item()),
        "residual_prediction_fvu": float(residual_prediction.detach().item()),
        "l0": float((outputs["codes"] > 0).float().sum(dim=-1).mean().item()),
        "predictable_energy_fraction": float(
            (
                outputs["predictable"] - model.pre_bias
            ).square().sum(dim=-1).mean().div(
                (outputs["target"] - model.pre_bias)
                .square()
                .sum(dim=-1)
                .mean()
                .clamp_min(1e-8)
            ).detach().item()
        ),
    }
    return loss, metrics


def grouped_split(
    metadata: list[dict[str, Any]],
    validation_fraction: float,
    group_key: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        group = str(row.get(group_key, f"row-{index}"))
        groups.setdefault(group, []).append(index)
    if len(groups) < 4:
        raise ValueError("need at least four independent groups")
    ordered = sorted(groups)
    permutation = torch.randperm(
        len(ordered),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    n_validation = max(1, round(len(ordered) * validation_fraction))
    n_validation = min(n_validation, len(ordered) - 2)
    validation_groups = {ordered[index] for index in permutation[:n_validation]}
    train_indices = [
        index
        for group, indices in groups.items()
        if group not in validation_groups
        for index in indices
    ]
    validation_indices = [
        index
        for group, indices in groups.items()
        if group in validation_groups
        for index in indices
    ]
    return train_indices, validation_indices


def grouped_three_way_split(
    metadata: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
    group_key: str,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        group = str(row.get(group_key, f"row-{index}"))
        groups.setdefault(group, []).append(index)
    if len(groups) < 10:
        raise ValueError("need at least ten independent groups for train/validation/test")
    ordered = sorted(groups)
    permutation = torch.randperm(
        len(ordered),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    n_test = max(1, round(len(ordered) * test_fraction))
    n_validation = max(1, round(len(ordered) * validation_fraction))
    if n_test + n_validation > len(ordered) - 4:
        raise ValueError("validation/test fractions leave too few training groups")
    test_groups = {ordered[index] for index in permutation[:n_test]}
    validation_groups = {
        ordered[index]
        for index in permutation[n_test : n_test + n_validation]
    }
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for group, indices in groups.items():
        if group in test_groups:
            test_indices.extend(indices)
        elif group in validation_groups:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return train_indices, validation_indices, test_indices


def fixed_spans(
    window_size: int,
    context_width: int,
    target_sizes: Iterable[int],
    gaps: Iterable[int],
    context_mode: str,
) -> list[SpanSpec]:
    spans = []
    for target_size in target_sizes:
        for gap in gaps:
            spans.append(
                make_span_spec(
                    window_size=window_size,
                    context_width=context_width,
                    target_size=target_size,
                    gap=gap,
                    context_mode=context_mode,
                    target_start=None,
                    generator=torch.Generator().manual_seed(
                        10_000 + 101 * target_size + gap
                    ),
                )
            )
    return spans


@torch.no_grad()
def evaluate_losses(
    model: PredictiveSparseAutoencoder,
    loader: DataLoader,
    spans: list[SpanSpec],
    device: torch.device,
    prediction_weight: float,
    residual_prediction_weight: float,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for (x,) in loader:
        x = x.to(device)
        for span in spans:
            _, metrics = predictive_loss(
                model,
                x,
                span,
                prediction_weight,
                residual_prediction_weight,
            )
            batch_count = len(x)
            count += batch_count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + batch_count * value
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a JEPA-regularized predictive sparse autoencoder"
    )
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--objective",
        choices=["joint", "posthoc"],
        default="joint",
        help="joint is the proposed model; posthoc is the standard-SAE control",
    )
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--context-width", type=int, default=24)
    parser.add_argument("--target-sizes", type=parse_int_tuple, default=(2, 4, 8))
    parser.add_argument("--gaps", type=parse_int_tuple, default=(2, 4, 8))
    parser.add_argument(
        "--context-mode",
        choices=["causal", "retrospective"],
        default="causal",
    )
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--residual-prediction-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument(
        "--posthoc-fraction",
        type=float,
        default=0.35,
        help="fraction of posthoc steps reserved for its frozen-SAE predictor",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="fixed independently of the feature-learning seed",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5")
    if not 0.0 < args.posthoc_fraction < 1.0:
        raise ValueError("--posthoc-fraction must be between 0 and 1")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    bundle = torch_load(args.activations)
    x = bundle["activations"].float()
    metadata = bundle["metadata"]
    if x.ndim != 3:
        raise ValueError("activations must have shape [windows, tokens, d_model]")
    window_size = x.shape[1]
    largest_required = args.context_width + max(args.gaps) + max(args.target_sizes)
    if args.context_mode == "retrospective":
        largest_required += max(args.gaps)
    if window_size < largest_required:
        raise ValueError(
            f"activation window {window_size} is too short; need at least "
            f"{largest_required} for the requested mask grid"
        )

    train_indices, validation_indices, test_indices = grouped_three_way_split(
        metadata,
        args.validation_fraction,
        args.test_fraction,
        args.group_key,
        args.split_seed,
    )
    train_loader = DataLoader(
        TensorDataset(x[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(x[validation_indices]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    cfg = PredictiveSAEConfig(
        d_in=x.shape[-1],
        d_sae=args.d_sae,
        k=args.k,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_width=args.context_width,
        max_window_size=window_size,
        max_target_size=max(args.target_sizes),
        max_gap=max(args.gaps),
        context_mode=args.context_mode,
        ema_decay=args.ema_decay,
    )
    model = PredictiveSparseAutoencoder(cfg).to(device)
    model.initialize_from_data(x[train_indices])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    mask_generator = torch.Generator().manual_seed(args.seed + 919)
    target_sizes = tuple(args.target_sizes)
    gaps = tuple(args.gaps)
    validation_spans = fixed_spans(
        window_size,
        args.context_width,
        target_sizes,
        gaps,
        args.context_mode,
    )
    posthoc_start = (
        round(args.steps * (1.0 - args.posthoc_fraction))
        if args.objective == "posthoc"
        else args.steps + 1
    )
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    phase = "joint" if args.objective == "joint" else "sae"
    for step in trange(1, args.steps + 1, desc=f"predictive SAE ({args.objective})"):
        if args.objective == "posthoc" and step == posthoc_start:
            phase = "posthoc_predictor"
            for parameter in (
                [model.pre_bias, model.decoder]
                + list(model.encoder.parameters())
            ):
                parameter.requires_grad_(False)
            model.target_encoder.load_state_dict(model.encoder.state_dict())
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            (batch,) = next(iterator)
        target_size = target_sizes[
            int(torch.randint(len(target_sizes), (1,), generator=mask_generator).item())
        ]
        gap = gaps[int(torch.randint(len(gaps), (1,), generator=mask_generator).item())]
        span = make_span_spec(
            window_size,
            args.context_width,
            target_size,
            gap,
            args.context_mode,
            generator=mask_generator,
        )
        active_prediction_weight = (
            0.0 if args.objective == "posthoc" and phase == "sae" else args.prediction_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = predictive_loss(
            model,
            batch.to(device),
            span,
            active_prediction_weight,
            args.residual_prediction_weight,
        )
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
        if phase == "joint":
            model.update_target_encoder()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            validation = evaluate_losses(
                model,
                validation_loader,
                validation_spans,
                device,
                args.prediction_weight,
                args.residual_prediction_weight,
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
        "source_config": bundle.get("config", {}),
        "split": {
            "group_key": args.group_key,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
        },
    }
    torch.save(checkpoint, output_dir / "predictive_sae.pt")
    write_json(
        output_dir / "training_report.json",
        {
            "method": (
                "JEPA-regularized SAE"
                if args.objective == "joint"
                else "standard SAE + frozen post-hoc predictor"
            ),
            "objective": args.objective,
            "context_mode": args.context_mode,
            "leakage_warning": (
                None
                if args.context_mode == "causal"
                else "Right-context residuals have already attended to target tokens."
            ),
            "n_train_windows": len(train_indices),
            "n_validation_windows": len(validation_indices),
            "n_locked_test_windows": len(test_indices),
            "mask_grid": {
                "context_width": args.context_width,
                "target_sizes": list(target_sizes),
                "gaps": list(gaps),
            },
            "history": history,
            "final_validation": history[-1]["validation"],
        },
    )
    print(f"saved {args.objective} checkpoint and report to {output_dir}")


if __name__ == "__main__":
    main()
