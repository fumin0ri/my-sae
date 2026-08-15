from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from .group_sae import topk_relu
from .io import torch_load
from .rectified_lpjepa_sae import (
    ARCHITECTURE_ID,
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
)


Component = Literal["full", "high", "low"]


@dataclass
class SAEBenchAdapterConfig:
    """Duck-typed subset of SAE Lens config required by SAEBench 0.6."""

    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    component: Component
    context_size: int = 128
    hook_head_index: int | None = None
    architecture: str = "rectified_lpjepa_topk"
    architecture_str: str = "rectified_lpjepa_topk"
    apply_b_dec_to_input: bool = False
    finetuning_scaling_factor: bool = False
    activation_fn_str: str = "topk"
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dtype: str = "bfloat16"
    device: str = "cuda"
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_path: str = "Skylion007/openwebtext"
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple[Any, ...] = (None,)
    training_tokens: int = -1
    sae_lens_training_version: str | None = None
    neuronpedia_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transformer_lens_model_name(model_name: str) -> str:
    if model_name.startswith("EleutherAI/"):
        return model_name.split("/", 1)[1]
    return model_name


def hook_name_for(layer: int, hook_point: str) -> str:
    suffix = {"pre": "hook_resid_pre", "post": "hook_resid_post"}.get(
        hook_point
    )
    if suffix is None:
        raise ValueError(f"unsupported residual hook point: {hook_point!r}")
    return f"blocks.{layer}.{suffix}"


class RectifiedLpJEPASAEBenchAdapter(nn.Module):
    """Exact unit-decoder reparameterization of a trained Rectified SAE.

    The native model encodes normalized residuals and multiplies its decoded
    result by a scalar residual RMS.  SAEBench expects unit-norm ``W_dec`` rows,
    so this adapter moves that scalar into the feature activations.  Therefore
    ``decode(encode(x))`` is numerically equivalent to the native model while
    the exposed decoder directions remain unit norm.
    """

    def __init__(
        self,
        *,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        cfg: SAEBenchAdapterConfig,
        high_width: int,
        high_k: int,
        low_k: int,
    ) -> None:
        super().__init__()
        self.W_enc = nn.Parameter(w_enc, requires_grad=False)
        self.W_dec = nn.Parameter(w_dec, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.cfg = cfg
        self.high_width = high_width
        self.high_k = high_k
        self.low_k = low_k
        self.device = self.W_dec.device
        self.dtype = self.W_dec.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        preactivations = x.to(self.dtype) @ self.W_enc + self.b_enc
        if self.cfg.component == "high":
            return topk_relu(preactivations, self.high_k)
        if self.cfg.component == "low":
            return topk_relu(preactivations, self.low_k)
        high = topk_relu(preactivations[..., : self.high_width], self.high_k)
        low = topk_relu(preactivations[..., self.high_width :], self.low_k)
        return torch.cat((high, low), dim=-1)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts.to(self.dtype) @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def to(self, *args: Any, **kwargs: Any) -> RectifiedLpJEPASAEBenchAdapter:
        super().to(*args, **kwargs)
        self.device = self.W_dec.device
        self.dtype = self.W_dec.dtype
        self.cfg.device = str(self.device)
        self.cfg.dtype = str(self.dtype).removeprefix("torch.")
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        tolerance = 1e-2 if self.dtype in (torch.bfloat16, torch.float16) else 1e-5
        norms = self.W_dec.float().norm(dim=-1)
        return bool(
            torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
        )


def load_saebench_adapter(
    checkpoint_path: str | Path,
    *,
    component: Component = "full",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    context_size: int = 128,
    model_revision: str | None = None,
) -> tuple[RectifiedLpJEPASAEBenchAdapter, dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(
            f"checkpoint is not a {ARCHITECTURE_ID} Rectified LpJEPA-SAE"
        )
    native = RectifiedLpJEPASAE(RectifiedLpJEPAConfig(**checkpoint["config"]))
    native.load_state_dict(checkpoint["state_dict"])
    native.eval()

    source = checkpoint.get("source_config", {})
    layer = int(source.get("layer", 0))
    hook_point = str(source.get("hook_point", "post"))
    source_model = str(source.get("model", "EleutherAI/pythia-6.9b-deduped"))
    revision = model_revision or source.get("resolved_model_revision")
    # The target environment intentionally stays on torch 2.5.1+cu121, so do
    # not allow Transformers to fall back to vulnerable pickle weight files.
    model_kwargs: dict[str, Any] = {"use_safetensors": True}
    if revision:
        model_kwargs["revision"] = revision
    raw_encoder = native.encoder.linear.weight.detach()
    scale = native.pre_scale.detach()
    effective_bias = (
        scale * native.encoder.linear.bias.detach()
        - native.pre_bias.detach() @ raw_encoder.T
    )
    slices = {
        "full": slice(None),
        "high": slice(0, native.cfg.d_high),
        "low": slice(native.cfg.d_high, native.cfg.d_sae),
    }
    selected = slices[component]
    w_enc = raw_encoder[selected].T.contiguous()
    w_dec = native.decoder.detach()[selected].contiguous()
    b_enc = effective_bias[selected].contiguous()
    b_dec = native.pre_bias.detach().contiguous()
    width = w_dec.shape[0]
    cfg = SAEBenchAdapterConfig(
        model_name=transformer_lens_model_name(source_model),
        d_in=native.cfg.d_in,
        d_sae=width,
        hook_layer=layer,
        hook_name=hook_name_for(layer, hook_point),
        component=component,
        context_size=context_size,
        activation_fn_kwargs={
            "high_k": native.cfg.high_k,
            "low_k": native.cfg.low_k,
            "high_width": native.cfg.d_high,
        },
        model_from_pretrained_kwargs=model_kwargs,
        training_tokens=int(
            checkpoint.get("train_args", {}).get("training_tokens", -1)
        ),
    )
    adapter = RectifiedLpJEPASAEBenchAdapter(
        w_enc=w_enc,
        w_dec=w_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        cfg=cfg,
        high_width=native.cfg.d_high if component == "full" else width,
        high_k=native.cfg.high_k,
        low_k=native.cfg.low_k,
    ).to(device=device, dtype=dtype)
    if not adapter.check_decoder_norms():
        raise ValueError("SAEBench adapter decoder directions are not unit norm")
    return adapter, checkpoint
