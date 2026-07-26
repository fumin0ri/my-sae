from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

from .io import torch_load, write_json


def topk_relu(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.relu(x)
    if k >= x.shape[-1]:
        return x
    values, indices = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    return out.scatter(-1, indices, values)


@dataclass
class GroupSAEConfig:
    d_in: int
    d_shared: int
    d_private: int
    k_shared: int
    k_private: int


class SharedPrivateSAE(nn.Module):
    """Window SAE with an additive token-shared and token-private dictionary.

    x[w,t] ≈ bias + D_shared z_shared[w] + D_private z_private[w,t]
    """

    def __init__(self, cfg: GroupSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.shared_encoder = nn.Linear(cfg.d_in, cfg.d_shared)
        self.private_encoder = nn.Linear(cfg.d_in, cfg.d_private)
        self.shared_decoder = nn.Parameter(torch.empty(cfg.d_shared, cfg.d_in))
        self.private_decoder = nn.Parameter(torch.empty(cfg.d_private, cfg.d_in))
        nn.init.kaiming_uniform_(self.shared_decoder, a=5**0.5)
        nn.init.kaiming_uniform_(self.private_decoder, a=5**0.5)

    @torch.no_grad()
    def normalize_decoders(self) -> None:
        self.shared_decoder.div_(
            self.shared_decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )
        self.private_decoder.div_(
            self.private_decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = x - self.pre_bias
        # The shared encoder only sees a permutation-invariant window summary.
        z_shared = topk_relu(
            self.shared_encoder(centered.mean(dim=1)), self.cfg.k_shared
        )
        z_private = topk_relu(
            self.private_encoder(centered), self.cfg.k_private
        )
        shared_write = z_shared @ self.shared_decoder
        private_write = z_private @ self.private_decoder
        reconstruction = (
            self.pre_bias
            + shared_write[:, None, :]
            + private_write
        )
        return reconstruction, z_shared, z_private, private_write


def loss_terms(
    model: SharedPrivateSAE,
    x: torch.Tensor,
    private_mean_penalty: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction, z_shared, z_private, private_write = model(x)
    scale = x.var(dim=(0, 1), unbiased=False).mean().clamp_min(1e-8)
    reconstruction_loss = (reconstruction - x).square().mean() / scale
    # Prevent all common information being redundantly implemented by every
    # private token code. This is an identifiability regularizer, not a theorem.
    private_common = private_write.mean(dim=1).square().mean() / scale
    loss = reconstruction_loss + private_mean_penalty * private_common
    metrics = {
        "loss": float(loss.detach().item()),
        "reconstruction": float(reconstruction_loss.detach().item()),
        "private_common": float(private_common.detach().item()),
        "shared_l0": float((z_shared > 0).float().sum(dim=-1).mean().item()),
        "private_l0": float((z_private > 0).float().sum(dim=-1).mean().item()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: SharedPrivateSAE,
    loader: DataLoader,
    device: str,
    private_mean_penalty: float,
) -> dict[str, float]:
    sums: dict[str, float] = {}
    count = 0
    for (x,) in loader:
        _, metrics = loss_terms(model, x.to(device), private_mean_penalty)
        batch_n = len(x)
        count += batch_n
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + batch_n * value
    return {key: value / max(count, 1) for key, value in sums.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a joint shared/private Top-K SAE")
    p.add_argument("--activations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--d-shared", type=int, default=8192)
    p.add_argument("--d-private", type=int, default=8192)
    p.add_argument("--k-shared", type=int, default=32)
    p.add_argument("--k-private", type=int, default=32)
    p.add_argument("--private-mean-penalty", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=100)
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    bundle = torch_load(args.activations)
    x = bundle["activations"].float()
    cfg = GroupSAEConfig(
        d_in=x.shape[-1],
        d_shared=args.d_shared,
        d_private=args.d_private,
        k_shared=args.k_shared,
        k_private=args.k_private,
    )
    model = SharedPrivateSAE(cfg).to(args.device)
    with torch.no_grad():
        model.pre_bias.copy_(x.mean(dim=(0, 1)).to(args.device))
        model.normalize_decoders()
    n_val = max(1, round(len(x) * args.validation_fraction))
    n_train = len(x) - n_val
    if n_train < 2:
        raise ValueError("not enough windows for train/validation split")
    train_set, val_set = random_split(
        TensorDataset(x),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    iterator = iter(train_loader)
    history: list[dict[str, float]] = []
    for step in trange(1, args.steps + 1, desc="group SAE"):
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            (batch,) = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_terms(
            model, batch.to(args.device), args.private_mean_penalty
        )
        loss.backward()
        optimizer.step()
        model.normalize_decoders()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics["step"] = float(step)
            history.append(metrics)

    validation = evaluate(
        model, val_loader, args.device, args.private_mean_penalty
    )
    shared_codes: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            _, z_shared, _, _ = model(x[start : start + args.batch_size].to(args.device))
            shared_codes.append(z_shared.cpu())
    all_shared_codes = torch.cat(shared_codes)
    feature_mean = all_shared_codes.mean(dim=0)
    feature_frequency = (all_shared_codes > 0).float().mean(dim=0)
    top_feature_ids = torch.topk(
        feature_mean, min(100, cfg.d_shared)
    ).indices.tolist()
    feature_summary = []
    for feature_id in top_feature_ids:
        example_ids = torch.topk(
            all_shared_codes[:, feature_id], min(10, len(x))
        ).indices.tolist()
        feature_summary.append(
            {
                "feature_id": feature_id,
                "mean_activation": float(feature_mean[feature_id].item()),
                "window_frequency": float(feature_frequency[feature_id].item()),
                "top_examples": [
                    {
                        "window_index": i,
                        "activation": float(all_shared_codes[i, feature_id].item()),
                        "metadata": bundle["metadata"][i],
                    }
                    for i in example_ids
                    if all_shared_codes[i, feature_id] > 0
                ],
            }
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(cfg),
            "train_args": vars(args),
            "source_config": bundle.get("config", {}),
        },
        out_dir / "group_sae.pt",
    )
    torch.save(
        {
            "shared_codes": all_shared_codes,
            "metadata": bundle["metadata"],
            "config": asdict(cfg),
        },
        out_dir / "group_sae_codes.pt",
    )
    write_json(
        out_dir / "group_sae_report.json",
        {
            "validation": validation,
            "history": history,
            "features": feature_summary,
        },
    )
    print(f"saved model and report to {out_dir}")


if __name__ == "__main__":
    main()
