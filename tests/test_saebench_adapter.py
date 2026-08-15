from dataclasses import asdict

import pytest
import torch

from shared_residual.rectified_lpjepa_sae import (
    ARCHITECTURE_ID,
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
)
from shared_residual.saebench_adapter import (
    hook_name_for,
    load_saebench_adapter,
    transformer_lens_model_name,
)
from shared_residual.saebench_eval import (
    _density_only_misc_metrics,
    _isolated_eval_command,
    _load_official_core_outputs,
)


def write_checkpoint(tmp_path):
    cfg = RectifiedLpJEPAConfig(
        d_in=8,
        d_sae=20,
        high_k=2,
        low_k=4,
        max_span_length=4,
        high_fraction=0.2,
        target_active_fraction=0.75,
    )
    model = RectifiedLpJEPASAE(cfg)
    model.initialize_from_statistics(torch.linspace(-0.4, 0.4, 8), 2.5)
    checkpoint = {
        "architecture_id": ARCHITECTURE_ID,
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "source_config": {
            "model": "EleutherAI/pythia-6.9b-deduped",
            "resolved_model_revision": "test-revision",
            "layer": 16,
            "hook_point": "post",
        },
        "train_args": {},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    return path, model


def test_full_adapter_is_an_exact_unit_decoder_reparameterization(tmp_path) -> None:
    path, native = write_checkpoint(tmp_path)
    adapter, _ = load_saebench_adapter(path, dtype=torch.float32)
    x = torch.randn(7, native.cfg.d_in)
    expected = native.decode(native.encode(x))
    actual = adapter(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert adapter.W_enc.shape == (native.cfg.d_in, native.cfg.d_sae)
    assert adapter.W_dec.shape == (native.cfg.d_sae, native.cfg.d_in)
    assert adapter.cfg.model_from_pretrained_kwargs == {
        "revision": "test-revision",
        "use_safetensors": True,
    }
    assert torch.allclose(
        adapter.W_dec.norm(dim=-1), torch.ones(native.cfg.d_sae), atol=1e-5
    )
    code = adapter.encode(x)
    assert torch.all((code[:, : native.cfg.d_high] > 0).sum(-1) <= native.cfg.high_k)
    assert torch.all((code[:, native.cfg.d_high :] > 0).sum(-1) <= native.cfg.low_k)


@pytest.mark.parametrize("component", ["high", "low"])
def test_component_adapters_match_native_component_reconstruction(
    tmp_path, component
) -> None:
    path, native = write_checkpoint(tmp_path)
    adapter, _ = load_saebench_adapter(
        path, component=component, dtype=torch.float32
    )
    x = torch.randn(5, native.cfg.d_in)
    code = native.encode(x)
    high, low = native.split_code(code)
    expected = (
        native.decode_high(high)
        if component == "high"
        else native.decode_low(low, add_bias=True)
    )
    assert torch.allclose(adapter(x), expected, atol=1e-5, rtol=1e-5)
    expected_width = native.cfg.d_high if component == "high" else native.cfg.d_low
    assert adapter.cfg.d_sae == expected_width


def test_saebench_model_and_hook_names_are_transformer_lens_compatible() -> None:
    assert (
        transformer_lens_model_name("EleutherAI/pythia-6.9b-deduped")
        == "pythia-6.9b-deduped"
    )
    assert hook_name_for(16, "post") == "blocks.16.hook_resid_post"
    assert hook_name_for(16, "pre") == "blocks.16.hook_resid_pre"
    with pytest.raises(ValueError):
        hook_name_for(16, "middle")


def test_cached_official_core_output_is_loaded(tmp_path) -> None:
    core = tmp_path / "core"
    core.mkdir()
    path = core / "rectified-lpjepa-full_custom_sae_eval_results.json"
    path.write_text(
        '{"eval_result_metrics":{"sparsity":{"l0":6.0}}}', encoding="utf-8"
    )
    rows = _load_official_core_outputs(
        tmp_path, [("rectified-lpjepa-full", object())]
    )
    assert rows[0]["metrics"]["sparsity"]["l0"] == 6.0
    assert rows[0]["official_result"] == str(path)


def test_density_only_misc_metrics_marks_weight_metrics_unavailable() -> None:
    feature_metrics = {
        "feature_density": torch.tensor([0.0, 0.005, 0.02, 0.2])
    }
    metrics = _density_only_misc_metrics(feature_metrics)
    assert metrics["average_max_encoder_cosine_sim"] == -1.0
    assert metrics["average_max_decoder_cosine_sim"] == -1.0
    assert metrics["frac_alive"] == pytest.approx(0.75)
    assert metrics["freq_over_1_percent"] == pytest.approx(0.5)
    assert metrics["freq_over_10_percent"] == pytest.approx(0.25)
    assert metrics["normalized_freq_over_1_percent"] == pytest.approx(
        0.22 / 0.225
    )
    assert metrics["normalized_freq_over_10_percent"] == pytest.approx(
        0.2 / 0.225
    )
    for name in (
        "encoder_bias",
        "encoder_decoder_cosine_sim",
        "encoder_norm",
        "max_decoder_cosine_sim",
        "max_encoder_cosine_sim",
    ):
        assert feature_metrics[name] == [-1.0] * 4


@pytest.mark.parametrize(
    "argv",
    [
        ["--checkpoint", "model.pt", "--evals", "core,sparse_probing"],
        ["--checkpoint", "model.pt", "--evals=core,sparse_probing"],
    ],
)
def test_isolated_eval_command_replaces_only_eval_selection(argv) -> None:
    command = _isolated_eval_command("sparse_probing", argv)
    assert command[:3] == [
        command[0],
        "-m",
        "shared_residual.saebench_eval",
    ]
    assert "--checkpoint" in command
    assert "model.pt" in command
    assert not any("core,sparse_probing" in value for value in command)
    assert any("sparse_probing" in value for value in command)
