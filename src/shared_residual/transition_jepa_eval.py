from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .activation_store import (
    load_activation_manifest,
    manifest_fingerprint,
    validation_batches,
)
from .io import torch_load, write_json
from .modeling import (
    edit_residual,
    get_layer,
    input_device,
    load_hf_model,
)
from .training import autocast_context, configure_accelerator
from .transition_jepa_sae import (
    ARCHITECTURE_ID,
    TransitionJEPAConfig,
    TransitionJEPASAE,
)


def load_model(
    path: str | Path, device: torch.device
) -> tuple[TransitionJEPASAE, dict[str, Any]]:
    checkpoint = torch_load(path)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(
            f"{path} is not a {ARCHITECTURE_ID} checkpoint; old unsplit "
            "checkpoints are intentionally unsupported"
        )
    model = TransitionJEPASAE(TransitionJEPAConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _active_feature_mean(
    values: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Mean over active features, then over sequences."""
    weights = active.to(values.dtype)
    per_sequence = (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)
    return per_sequence.mean()


def smoothness_tv(code: torch.Tensor) -> torch.Tensor:
    """T-SAE smoothness_tv: summed adjacent L1 variation."""
    if code.shape[1] < 2:
        return torch.zeros((), device=code.device)
    return (code[:, 1:] - code[:, :-1]).abs().sum(dim=(-1, -2)).mean()


def lipschitz_continuity(x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """Mean per-feature maximum adjacent |df|/||dx||, matching T-SAE."""
    if code.shape[1] < 2:
        return torch.zeros((), device=code.device)
    active = code.sum(dim=1) != 0
    dx = torch.linalg.vector_norm(x[:, 1:] - x[:, :-1], dim=-1).clamp_min(1e-10)
    ratio = (code[:, 1:] - code[:, :-1]).abs() / dx[..., None]
    maximum = ratio.max(dim=1).values
    return _active_feature_mean(maximum, active)


def fft_smoothness(code: torch.Tensor, cutoff_ratio: float = 0.5) -> torch.Tensor:
    """High/low frequency power ratio from T-SAE evaluation.py."""
    active = code.sum(dim=1) != 0
    spectrum = torch.fft.rfft(code.float(), dim=1)
    power = spectrum.abs().square()
    cutoff = int(cutoff_ratio * power.shape[1])
    low = power[:, :cutoff].sum(dim=1)
    high = power[:, cutoff:].sum(dim=1)
    ratio = high / (low + 1e-10)
    return _active_feature_mean(ratio, active)


def wavelet_smoothness(code: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """Haar-like detail/approximation energy ratio used by T-SAE."""
    values = []
    for sequence in code.float():
        active = sequence.sum(dim=0) != 0
        signal = sequence[:, active]
        if signal.numel() == 0:
            values.append(torch.zeros((), device=code.device))
            continue
        detail_energy = torch.zeros((), device=code.device)
        for _ in range(levels):
            if signal.shape[0] < 2:
                break
            if signal.shape[0] % 2:
                signal = signal[:-1]
            even, odd = signal[::2], signal[1::2]
            detail = (even - odd) / 2
            detail_energy = detail_energy + detail.square().sum()
            signal = (even + odd) / 2
        values.append(detail_energy / (signal.square().sum() + 1e-10))
    return torch.stack(values).mean()


def multiscale_smoothness(
    code: torch.Tensor, scales: tuple[int, ...] = (1, 2, 4, 8)
) -> torch.Tensor:
    """Fine/coarse difference-variance ratio used by T-SAE."""
    valid = [scale for scale in scales if scale < code.shape[1]]
    if not valid:
        return torch.zeros((), device=code.device)
    measures: dict[int, torch.Tensor] = {}
    for scale in valid:
        differences = code[:, scale:].float() - code[:, :-scale].float()
        measures[scale] = differences.var(dim=1).mean(dim=-1)
    ratio = measures[min(valid)] / (measures[max(valid)] + 1e-10)
    return ratio.nan_to_num(0.0).mean()


@torch.no_grad()
def batch_temporal_metrics(
    model: TransitionJEPASAE, x: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor]:
    """Port of AI4LIFE temporal-saes/dictionary_learning/evaluation.py."""
    full_code = model.encode_ema(x)
    high_code, low_code = model.split_code(full_code)
    reconstruction = model.decode_ema(full_code)
    # Upstream recon_splits intentionally excludes the shared decoder bias.
    high_reconstruction = model.decode_high(high_code, ema=True, add_bias=False)
    low_reconstruction = model.decode_low(low_code, ema=True, add_bias=False)

    l2_loss = torch.linalg.vector_norm(x - reconstruction, dim=-1).mean()
    l1_loss = full_code.norm(p=1, dim=-1).mean()
    l0 = (full_code != 0).float().sum(dim=-1).mean()
    sequence_l0 = (full_code.sum(dim=1) != 0).float().sum(dim=-1).mean()
    cosine = F.cosine_similarity(x.float(), reconstruction.float(), dim=-1).mean()
    l2_ratio = (
        torch.linalg.vector_norm(reconstruction.float(), dim=-1)
        / torch.linalg.vector_norm(x.float(), dim=-1).clamp_min(1e-10)
    ).mean()
    total_variance = x.float().var(dim=1).sum(dim=-1).clamp_min(1e-10)

    def fve(value: torch.Tensor) -> torch.Tensor:
        residual = (x.float() - value.float()).var(dim=1).sum(dim=-1)
        return (1.0 - residual / total_variance).mean()

    reconstruction_norm_squared = torch.linalg.vector_norm(
        reconstruction.float(), dim=-1
    ).square()
    reconstruction_dot = (x.float() * reconstruction.float()).sum(dim=-1)
    relative_bias = reconstruction_norm_squared.mean() / reconstruction_dot.mean()
    metrics = {
        "l2_loss": float(l2_loss),
        "l1_loss": float(l1_loss),
        "l0": float(l0),
        "sequence_l0": float(sequence_l0),
        "smoothness_tv_h": float(smoothness_tv(high_code)),
        "smoothness_tv_l": float(smoothness_tv(low_code)),
        "lipschitz_cont_tot": float(lipschitz_continuity(x, full_code)),
        "lipschitz_cont_h": float(lipschitz_continuity(x, high_code)),
        "lipschitz_cont_l": float(lipschitz_continuity(x, low_code)),
        "fft_tot": float(fft_smoothness(full_code)),
        "fft_h": float(fft_smoothness(high_code)),
        "fft_l": float(fft_smoothness(low_code)),
        "wavelet_tot": float(wavelet_smoothness(full_code)),
        "wavelet_h": float(wavelet_smoothness(high_code)),
        "wavelet_l": float(wavelet_smoothness(low_code)),
        "multiscale_tot": float(multiscale_smoothness(full_code)),
        "multiscale_h": float(multiscale_smoothness(high_code)),
        "multiscale_l": float(multiscale_smoothness(low_code)),
        "frac_variance_explained": float(fve(reconstruction)),
        "frac_variance_explained_high": float(fve(high_reconstruction)),
        "frac_variance_explained_low": float(fve(low_reconstruction)),
        "cossim": float(cosine),
        "l2_ratio": float(l2_ratio),
        "relative_reconstruction_bias": float(relative_bias),
    }
    active = full_code.reshape(-1, full_code.shape[-1]).sum(dim=0).float().cpu()
    return metrics, active


@torch.no_grad()
def evaluate_activation_shards(
    model: TransitionJEPASAE,
    root: Path,
    manifest: dict[str, Any],
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    active_features = torch.zeros(model.cfg.d_sae, dtype=torch.float32)
    windows = 0
    batches = 0
    for batch in tqdm(
        validation_batches(root, manifest, batch_size, maximum_batches),
        desc="T-SAE temporal metrics",
    ):
        batch = batch.to(device, dtype=model.pre_bias.dtype, non_blocking=True)
        with autocast_context(device, amp_dtype):
            metrics, active = batch_temporal_metrics(model, batch)
        for key, value in metrics.items():
            totals[key] += len(batch) * value
        active_features += active
        windows += len(batch)
        batches += 1
    if windows == 0:
        raise ValueError("no validation activations were evaluated")
    result = {key: value / windows for key, value in totals.items()}
    result["frac_alive"] = float((active_features != 0).float().mean())
    result["n_windows"] = windows
    result["n_batches"] = batches
    return result


def causal_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = -100
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100 if pad_token_id is None else -100,
    )


def _text_rows(stream: Iterable[dict[str, Any]], key: str) -> Iterable[str]:
    for row in stream:
        text = str(row.get(key, ""))
        if text.strip():
            yield text


@torch.no_grad()
def evaluate_loss_recovered(
    sae: TransitionJEPASAE,
    source_config: dict[str, Any],
    model_name: str,
    revision: str | None,
    use_safetensors: bool,
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_key: str,
    n_inputs: int,
    context_length: int,
    hook_point: str,
    layer_index: int,
) -> dict[str, float]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": dataset_split,
        "streaming": True,
    }
    if dataset_config:
        stream = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        stream = load_dataset(dataset_name, **kwargs)
    texts = iter(_text_rows(stream, text_key))
    llm, tokenizer = load_hf_model(
        model_name,
        dtype,
        device_map,
        trust_remote_code,
        revision,
        use_safetensors=True if use_safetensors else None,
    )
    fitted_model = source_config.get("model")
    if fitted_model is not None and fitted_model != model_name:
        raise ValueError(f"checkpoint model {fitted_model!r} != evaluation model {model_name!r}")
    layer_path, layer = get_layer(llm, layer_index)
    fitted_layer_path = source_config.get("layer_path")
    if fitted_layer_path is not None and fitted_layer_path != layer_path:
        raise ValueError(
            f"checkpoint layer {fitted_layer_path!r} != evaluation layer {layer_path!r}"
        )
    token_device = input_device(llm)
    sae.to(token_device, dtype=llm.dtype).eval()
    totals = defaultdict(float)
    used = 0
    attempts = 0
    for _ in tqdm(range(n_inputs), desc="T-SAE loss recovered"):
        while True:
            text = next(texts)
            attempts += 1
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=context_length,
            )
            if encoded["input_ids"].shape[1] >= 2:
                break
        encoded = {key: value.to(token_device) for key, value in encoded.items()}
        original = llm(**encoded, use_cache=False).logits

        def reconstruct(hidden: torch.Tensor) -> torch.Tensor:
            code = sae.encode_ema(hidden.to(sae.pre_bias.dtype))
            return sae.decode_ema(code).to(hidden.dtype)

        with edit_residual(layer, hook_point, reconstruct):
            reconstructed = llm(**encoded, use_cache=False).logits
        with edit_residual(layer, hook_point, torch.zeros_like):
            zero = llm(**encoded, use_cache=False).logits
        original_loss = causal_lm_loss(
            original,
            encoded["input_ids"],
            encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
            tokenizer.pad_token_id,
        )
        reconstructed_loss = causal_lm_loss(
            reconstructed,
            encoded["input_ids"],
            encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
            tokenizer.pad_token_id,
        )
        zero_loss = causal_lm_loss(
            zero,
            encoded["input_ids"],
            encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
            tokenizer.pad_token_id,
        )
        recovered = (reconstructed_loss - zero_loss) / (
            original_loss - zero_loss
        )
        totals["loss_original"] += float(original_loss)
        totals["loss_reconstructed"] += float(reconstructed_loss)
        totals["loss_zero"] += float(zero_loss)
        totals["frac_recovered"] += float(recovered)
        used += 1
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        **{key: value / max(used, 1) for key, value in totals.items()},
        "n_inputs": used,
        "documents_examined": attempts,
        "context_length": context_length,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the final high/low EMA SAE with T-SAE metrics"
    )
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16"], default="bfloat16")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--hook-point", choices=["pre", "post"])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--eval-dataset", default="monology/pile-uncopyrighted")
    parser.add_argument("--eval-dataset-config")
    parser.add_argument("--eval-split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--loss-recovered-inputs", type=int, default=32)
    parser.add_argument("--loss-recovered-context-length", type=int, default=2048)
    parser.add_argument("--skip-loss-recovered", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.loss_recovered_inputs < 1:
        raise ValueError("batch sizes and input counts must be positive")
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    configure_accelerator(device)
    root, manifest = load_activation_manifest(args.activation_manifest)
    model, checkpoint = load_model(args.checkpoint, device)
    fingerprint = manifest_fingerprint(manifest)
    if checkpoint.get("data_fingerprint") != fingerprint:
        raise ValueError("checkpoint and activation manifest fingerprints differ")
    if model.cfg.window_size != int(manifest["window_size"]):
        raise ValueError("checkpoint and activation windows differ")
    activation_metrics = evaluate_activation_shards(
        model,
        root,
        manifest,
        args.batch_size,
        args.maximum_batches,
        device,
        args.amp_dtype if device.type == "cuda" else "none",
    )
    source_config = checkpoint.get("source_config", {})
    loss_recovered = None
    if not args.skip_loss_recovered:
        model_name = args.model or source_config.get("model")
        layer_index = args.layer if args.layer is not None else source_config.get("layer")
        hook_point = args.hook_point or source_config.get("hook_point", "post")
        if model_name is None or layer_index is None:
            raise ValueError("model and layer are required for loss recovered")
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        loss_recovered = evaluate_loss_recovered(
            model,
            source_config,
            str(model_name),
            args.revision or source_config.get("resolved_model_revision"),
            args.use_safetensors,
            args.dtype,
            args.device_map,
            args.trust_remote_code,
            args.eval_dataset,
            args.eval_dataset_config,
            args.eval_split,
            args.text_key,
            args.loss_recovered_inputs,
            args.loss_recovered_context_length,
            hook_point,
            int(layer_index),
        )
    report = {
        "method": "AI4LIFE temporal-saes evaluation port",
        "upstream_source": {
            "repository": "https://github.com/AI4LIFE-GROUP/temporal-saes",
            "file": "dictionary_learning/dictionary_learning/evaluation.py",
            "evaluation_blob_sha": "0f0deec54f828137d8f637ecc8f12ec9af3a84cc",
            "metrics": [
                "l2_loss", "l1_loss", "l0", "sequence_l0",
                "smoothness_tv_h/l", "lipschitz_cont_tot/h/l",
                "fft_tot/h/l", "wavelet_tot/h/l", "multiscale_tot/h/l",
                "frac_variance_explained[_high/_low]", "cossim", "l2_ratio",
                "relative_reconstruction_bias", "frac_alive", "frac_recovered",
            ],
        },
        "evaluation_protocol": {
            "sae": "final full-EMA high/low encoder-decoder",
            "activation_split": "document-disjoint Pile validation shards",
            "window_size": model.cfg.window_size,
            "high_features": model.cfg.d_high,
            "low_features": model.cfg.d_low,
            "high_topk": model.cfg.k_high,
            "low_topk": model.cfg.k_low,
            "individual_group_reconstruction_bias": "excluded, matching upstream recon_splits",
            "loss_recovered_dataset": None if args.skip_loss_recovered else args.eval_dataset,
        },
        "activation_metrics": activation_metrics,
        "loss_recovered": loss_recovered,
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "architecture_id": checkpoint["architecture_id"],
            "config": checkpoint["config"],
            "data_fingerprint": fingerprint,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "temporal_sae_eval.json", report)
    with (output_dir / "temporal_sae_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        handle.write("metric,value\n")
        for key, value in activation_metrics.items():
            handle.write(f"{key},{value}\n")
        if loss_recovered is not None:
            for key, value in loss_recovered.items():
                handle.write(f"{key},{value}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
