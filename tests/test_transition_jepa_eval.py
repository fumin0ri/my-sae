import torch

from shared_residual.transition_jepa_eval import (
    batch_temporal_metrics,
    fft_smoothness,
    lipschitz_continuity,
    multiscale_smoothness,
    smoothness_tv,
    wavelet_smoothness,
)
from shared_residual.transition_jepa_sae import (
    TransitionJEPAConfig,
    TransitionJEPASAE,
)


def make_model() -> TransitionJEPASAE:
    model = TransitionJEPASAE(
        TransitionJEPAConfig(
            d_in=4,
            d_sae=10,
            k=5,
            window_size=8,
            predictor_width=4,
            high_fraction=0.2,
        )
    )
    model.initialize_from_statistics(torch.zeros(4), 1.0)
    return model


def test_temporal_smoothness_metrics_are_zero_for_constant_code() -> None:
    code = torch.ones(2, 8, 5)
    x = torch.arange(8, dtype=torch.float32)[None, :, None].expand(2, 8, 3)
    assert smoothness_tv(code) == 0
    assert lipschitz_continuity(x, code) == 0
    assert fft_smoothness(code) == 0
    assert wavelet_smoothness(code) == 0
    assert multiscale_smoothness(code) == 0


def test_batch_metrics_match_upstream_metric_names() -> None:
    model = make_model()
    metrics, active = batch_temporal_metrics(model, torch.randn(3, 8, 4))
    required = {
        "l2_loss", "l1_loss", "l0", "sequence_l0",
        "smoothness_tv_h", "smoothness_tv_l",
        "lipschitz_cont_tot", "lipschitz_cont_h", "lipschitz_cont_l",
        "fft_tot", "fft_h", "fft_l",
        "wavelet_tot", "wavelet_h", "wavelet_l",
        "multiscale_tot", "multiscale_h", "multiscale_l",
        "frac_variance_explained", "frac_variance_explained_high",
        "frac_variance_explained_low", "cossim", "l2_ratio",
        "relative_reconstruction_bias",
    }
    assert set(metrics) == required
    assert active.shape == (10,)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_high_low_recon_splits_exclude_shared_bias_like_upstream() -> None:
    model = make_model()
    with torch.no_grad():
        model.ema_pre_bias.fill_(3.0)
    x = torch.randn(2, 8, 4)
    code = model.encode_ema(x)
    high, low = model.split_code(code)
    high_component = model.decode_high(high, ema=True, add_bias=False)
    low_component = model.decode_low(low, ema=True, add_bias=False)
    full = model.decode_ema(code)
    assert torch.allclose(
        full,
        high_component + low_component + model.ema_pre_bias,
        atol=1e-5,
    )
