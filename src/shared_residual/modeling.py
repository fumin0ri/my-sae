from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LAYER_PATHS = (
    "model.layers",          # Llama, Mistral, Qwen, Gemma
    "transformer.h",         # GPT-2
    "gpt_neox.layers",       # Pythia
    "model.decoder.layers",  # OPT
)


def parse_dtype(name: str) -> torch.dtype:
    names = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in names:
        raise ValueError(f"unknown dtype {name!r}; choose one of {sorted(names)}")
    return names[name]


def load_hf_model(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    revision: str | None = None,
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=parse_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        revision=revision,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tokenizer


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def forward_backbone(model: Any, encoded: dict[str, torch.Tensor]) -> Any:
    """Run transformer blocks without the expensive vocabulary projection."""
    return model.base_model(**encoded, use_cache=False, return_dict=True)


def find_layer_stack(model: Any) -> tuple[str, Any]:
    for path in LAYER_PATHS:
        try:
            stack = model.get_submodule(path)
        except AttributeError:
            continue
        if hasattr(stack, "__len__"):
            return path, stack
    raise ValueError(
        "Could not find transformer blocks. Pass a supported decoder-only model "
        f"or add its block path to LAYER_PATHS={LAYER_PATHS}."
    )


def get_layer(model: Any, layer: int) -> tuple[str, Any]:
    path, stack = find_layer_stack(model)
    n_layers = len(stack)
    if layer < 0:
        layer += n_layers
    if not 0 <= layer < n_layers:
        raise IndexError(f"layer {layer} outside [0, {n_layers})")
    return f"{path}.{layer}", stack[layer]


def tensor_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported layer output type: {type(output)}")


def replace_layer_output(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return value
    if isinstance(output, tuple):
        return (value, *output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    raise TypeError(f"unsupported layer output type: {type(output)}")


@contextmanager
def capture_residual(
    layer_module: Any, hook_point: str
) -> Iterator[dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}

    if hook_point == "pre":
        def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
            captured["activation"] = args[0].detach()

        handle = layer_module.register_forward_pre_hook(pre_hook)
    elif hook_point == "post":
        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            captured["activation"] = tensor_from_layer_output(output).detach()

        handle = layer_module.register_forward_hook(post_hook)
    else:
        raise ValueError("hook_point must be 'pre' or 'post'")

    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def edit_residual(
    layer_module: Any,
    hook_point: str,
    edit: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    if hook_point == "pre":
        def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            return (edit(args[0]), *args[1:])

        handle = layer_module.register_forward_pre_hook(pre_hook)
    elif hook_point == "post":
        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            return replace_layer_output(output, edit(tensor_from_layer_output(output)))

        handle = layer_module.register_forward_hook(post_hook)
    else:
        raise ValueError("hook_point must be 'pre' or 'post'")
    try:
        yield
    finally:
        handle.remove()
