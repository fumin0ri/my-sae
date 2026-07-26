from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import trange

from .io import torch_load, write_json


def robust_shared_features(
    features: torch.Tensor,
    min_active_fraction: float,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate SAE latents that recur across token positions in a window."""
    active_fraction = (features > 0).float().mean(dim=1)
    magnitude = torch.quantile(features.float(), quantile, dim=1)
    common = magnitude * (active_fraction >= min_active_fraction)
    return common, active_fraction


def _load_sae(release: str, sae_id: str, device: str) -> Any:
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[sae]'`") from exc
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def _feature_width(sae: Any) -> int:
    if hasattr(sae, "cfg") and hasattr(sae.cfg, "d_sae"):
        return int(sae.cfg.d_sae)
    if hasattr(sae, "W_dec"):
        return int(sae.W_dec.shape[0])
    raise AttributeError("cannot determine SAE feature width")


def _decoder_directions(sae: Any, d_sae: int, d_in: int) -> torch.Tensor:
    if not hasattr(sae, "W_dec"):
        raise AttributeError("SAE has no W_dec; cannot compute subspace overlap")
    decoder = sae.W_dec.detach().float()
    if decoder.shape == (d_sae, d_in):
        return decoder
    if decoder.shape == (d_in, d_sae):
        return decoder.T
    raise ValueError(
        f"unexpected W_dec shape {tuple(decoder.shape)}, expected "
        f"{(d_sae, d_in)} or {(d_in, d_sae)}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare token-first and mean-first SAE analysis")
    p.add_argument("--activations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--sae-id", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-windows", type=int, default=8)
    p.add_argument("--min-active-fraction", type=float, default=0.7)
    p.add_argument("--quantile", type=float, default=0.25)
    p.add_argument("--top-features", type=int, default=100)
    p.add_argument("--top-examples", type=int, default=10)
    p.add_argument(
        "--subspace",
        help="Optional subspace.pt; ranks recurrent features by decoder overlap too",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    bundle = torch_load(args.activations)
    x = bundle["activations"].to(torch.float32)
    n, t, d = x.shape
    sae = _load_sae(args.release, args.sae_id, args.device)
    d_sae = _feature_width(sae)
    common_chunks: list[torch.Tensor] = []
    frequency_chunks: list[torch.Tensor] = []
    pooled_chunks: list[torch.Tensor] = []
    token_sse = 0.0
    token_energy = 0.0
    pooled_sse = 0.0
    pooled_energy = 0.0

    for start in trange(0, n, args.batch_windows, desc="SAE"):
        batch = x[start : start + args.batch_windows].to(args.device)
        flat = batch.reshape(-1, d)
        with torch.inference_mode():
            token_features = sae.encode(flat).reshape(batch.shape[0], t, d_sae)
            token_recon = sae.decode(token_features).reshape_as(batch)
            pooled = batch.mean(dim=1)
            pooled_features = sae.encode(pooled)
            pooled_recon = sae.decode(pooled_features)
        common, frequency = robust_shared_features(
            token_features, args.min_active_fraction, args.quantile
        )
        common_chunks.append(common.cpu())
        frequency_chunks.append(frequency.cpu())
        pooled_chunks.append(pooled_features.cpu())
        token_sse += float((token_recon - batch).square().sum().item())
        token_energy += float(
            (batch - batch.mean(dim=(0, 1), keepdim=True)).square().sum().item()
        )
        pooled_sse += float((pooled_recon - pooled).square().sum().item())
        pooled_energy += float(
            (pooled - pooled.mean(dim=0, keepdim=True)).square().sum().item()
        )

    common_codes = torch.cat(common_chunks)
    active_fraction = torch.cat(frequency_chunks)
    pooled_codes = torch.cat(pooled_chunks)
    scores = common_codes.mean(dim=0)
    nonzero = (common_codes > 0).float().mean(dim=0)
    overlap = torch.ones(d_sae)
    if args.subspace:
        subspace = torch_load(args.subspace)
        basis = subspace["basis"].float()
        decoder = _decoder_directions(sae, d_sae, d).cpu()
        decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        overlap = (decoder @ basis).square().sum(dim=1).sqrt()
    # Activation alone favors globally frequent features; overlap alone favors
    # inactive directions. Their product requires both recurrence and geometry.
    ranking_score = scores * overlap
    top_ids = torch.topk(ranking_score, min(args.top_features, d_sae)).indices
    feature_summary: list[dict[str, Any]] = []
    metadata = bundle["metadata"]
    for feature_id in top_ids.tolist():
        example_ids = torch.topk(
            common_codes[:, feature_id], min(args.top_examples, n)
        ).indices.tolist()
        feature_summary.append(
            {
                "feature_id": feature_id,
                "mean_common_activation": float(scores[feature_id].item()),
                "window_frequency": float(nonzero[feature_id].item()),
                "decoder_subspace_overlap": float(overlap[feature_id].item()),
                "ranking_score": float(ranking_score[feature_id].item()),
                "top_examples": [
                    {
                        "window_index": i,
                        "activation": float(common_codes[i, feature_id].item()),
                        "metadata": metadata[i],
                    }
                    for i in example_ids
                    if common_codes[i, feature_id] > 0
                ],
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_first_common_codes": common_codes,
            "token_active_fraction": active_fraction,
            "mean_first_codes": pooled_codes,
            "metadata": metadata,
            "config": vars(args),
        },
        out_dir / "sae_codes.pt",
    )
    write_json(
        out_dir / "sae_report.json",
        {
            "shape": [n, t, d],
            "d_sae": d_sae,
            "token_reconstruction_fvu": token_sse / max(token_energy, 1e-12),
            "mean_vector_reconstruction_fvu": pooled_sse / max(pooled_energy, 1e-12),
            "warning": (
                "Mean-first vectors are off the SAE training distribution. Prefer "
                "token-first recurrent features unless its reconstruction and causal "
                "tests support mean-first encoding."
            ),
            "ranking": (
                "mean recurrent activation × decoder/subspace overlap"
                if args.subspace
                else "mean recurrent activation"
            ),
            "features": feature_summary,
        },
    )
    print(f"saved SAE analysis to {out_dir}")


if __name__ == "__main__":
    main()
