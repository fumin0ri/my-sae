from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import write_json

HIGH = "#059669"
LOW = "#84cc16"
TOTAL = "#2563eb"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_plot(run_dir: Path, figures: Path) -> None:
    report = load_json(run_dir / "model" / "training_report.json")
    history = report["history"]
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for key, label, color, style in (
        ("ema_high_reconstruction_fvu", "EMA high only", HIGH, "--"),
        ("ema_reconstruction_fvu", "EMA high + low", TOTAL, "-"),
    ):
        axes[0].plot(
            steps,
            [row["validation"][key] for row in history],
            label=label,
            color=color,
            linestyle=style,
        )
    axes[1].plot(
        steps,
        [row["validation"]["code_cosine"] for row in history],
        color=HIGH,
    )
    axes[2].plot(
        steps,
        [row["validation"]["high_l0"] for row in history],
        color=HIGH,
        label="High L0",
    )
    axes[2].plot(
        steps,
        [row["validation"]["low_l0"] for row in history],
        color=LOW,
        label="Low L0",
    )
    axes[0].set_title("Cumulative endpoint reconstruction")
    axes[0].set_ylabel("Validation FVU")
    axes[1].set_title("High endpoint forecast")
    axes[1].set_ylabel("Code cosine")
    axes[2].set_title("Independent group sparsity")
    axes[2].set_ylabel("Active features / position")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "training")


def reconstruction_plot(report: dict[str, Any], figures: Path) -> None:
    metrics = report["activation_metrics"]
    labels = ["High", "Low", "High + low"]
    fve = [
        metrics["frac_variance_explained_high"],
        metrics["frac_variance_explained_low"],
        metrics["frac_variance_explained"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    axes[0].bar(labels, fve, color=[HIGH, LOW, TOTAL])
    axes[0].axhline(0, color="#334155", linewidth=1)
    axes[0].set_title("Fraction of activation variance explained")
    axes[0].set_ylabel("FVE")
    secondary_labels = ["Cosine", "L2 ratio", "Relative bias"]
    secondary = [
        metrics["cossim"],
        metrics["l2_ratio"],
        metrics["relative_reconstruction_bias"],
    ]
    axes[1].bar(secondary_labels, secondary, color=[TOTAL, "#7c3aed", "#d97706"])
    axes[1].axhline(1, color="#334155", linestyle="--", label="Ideal ratio/bias")
    axes[1].set_title("Reconstruction geometry")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "reconstruction")


def smoothness_plot(report: dict[str, Any], figures: Path) -> None:
    metrics = report["activation_metrics"]
    definitions = [
        ("TV", "smoothness_tv_h", "smoothness_tv_l"),
        ("Lipschitz", "lipschitz_cont_h", "lipschitz_cont_l"),
        ("FFT", "fft_h", "fft_l"),
        ("Wavelet", "wavelet_h", "wavelet_l"),
        ("Multiscale", "multiscale_h", "multiscale_l"),
    ]
    high = np.asarray([metrics[h] for _, h, _ in definitions], dtype=float)
    low = np.asarray([metrics[l] for _, _, l in definitions], dtype=float)
    # Ratios make differently-scaled upstream metrics readable together.
    denominator = np.maximum(np.abs(high) + np.abs(low), 1e-12)
    high_share = high / denominator
    low_share = low / denominator
    positions = np.arange(len(definitions))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7))
    axes[0].bar(positions - width / 2, high, width, label="High", color=HIGH)
    axes[0].bar(positions + width / 2, low, width, label="Low", color=LOW)
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].set_title("T-SAE smoothness metrics (raw)")
    axes[0].set_ylabel("Lower generally means smoother")
    axes[1].bar(positions - width / 2, high_share, width, label="High", color=HIGH)
    axes[1].bar(positions + width / 2, low_share, width, label="Low", color=LOW)
    axes[1].set_title("Within-metric high/low contribution")
    axes[1].set_ylabel("Value / (|high| + |low|)")
    for axis in axes:
        axis.set_xticks(positions, [row[0] for row in definitions], rotation=18)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "smoothness")


def sparsity_plot(report: dict[str, Any], figures: Path) -> None:
    metrics = report["activation_metrics"]
    values = [metrics["l0"], metrics["sequence_l0"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].bar(["Per position L0", "Sequence L0"], values, color=[TOTAL, HIGH])
    axes[0].set_title("Sparse feature usage")
    axes[0].set_ylabel("Active features")
    axes[1].bar(
        ["Alive", "Dead"],
        [metrics["frac_alive"], 1.0 - metrics["frac_alive"]],
        color=[HIGH, "#cbd5e1"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Dictionary coverage")
    axes[1].set_ylabel("Feature fraction")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "sparsity")


def loss_recovered_plot(report: dict[str, Any], figures: Path) -> None:
    result = report.get("loss_recovered")
    if result is None:
        return
    labels = ["Original", "SAE reconstructed", "Zero ablation"]
    values = [
        result["loss_original"],
        result["loss_reconstructed"],
        result["loss_zero"],
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.bar(labels, values, color=[TOTAL, HIGH, "#dc2626"])
    ax.set_ylabel("Next-token cross entropy")
    ax.set_title(f"Loss recovered = {result['frac_recovered']:.3f}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "loss-recovered")


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (float, int)) else html.escape(str(value))


def write_html(output: Path, report: dict[str, Any]) -> None:
    metrics = report["activation_metrics"]
    recovered = report.get("loss_recovered")
    figures = [
        ("Training", "figures/training.png"),
        ("Reconstruction", "figures/reconstruction.png"),
        ("High/low temporal smoothness", "figures/smoothness.png"),
        ("Sparsity", "figures/sparsity.png"),
    ]
    if recovered is not None:
        figures.append(("LLM loss recovered", "figures/loss-recovered.png"))
    figure_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{path}" alt="{html.escape(title)}"></section>'
        for title, path in figures
    )
    recovered_value = "skipped" if recovered is None else fmt(recovered["frac_recovered"])
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>High/low JEPA-SAE — T-SAE evaluation</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--fg:#172033;--muted:#64748b;--line:#dbe2ea}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1120px;margin:auto;padding:36px 20px 70px}}.subtitle{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.metric{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.metric span{{display:block;color:var(--muted);font-size:.84rem}}.metric strong{{font-size:1.35rem}}
section{{margin-top:34px}}section img{{width:100%;background:white;border:1px solid var(--line);border-radius:10px}}
</style></head><body><main>
<h1>High/low endpoint JEPA-SAE</h1>
<p class="subtitle">Evaluation ports the metrics and formulas in AI4LIFE-GROUP/temporal-saes. The evaluated artifact is the final full-EMA high/low SAE; no unsplit SAE condition is present.</p>
<div class="metrics">
<div class="metric"><span>Full FVE</span><strong>{fmt(metrics['frac_variance_explained'])}</strong></div>
<div class="metric"><span>High-only FVE</span><strong>{fmt(metrics['frac_variance_explained_high'])}</strong></div>
<div class="metric"><span>Low-only FVE</span><strong>{fmt(metrics['frac_variance_explained_low'])}</strong></div>
<div class="metric"><span>Reconstruction cosine</span><strong>{fmt(metrics['cossim'])}</strong></div>
<div class="metric"><span>Per-position L0</span><strong>{fmt(metrics['l0'])}</strong></div>
<div class="metric"><span>Feature alive fraction</span><strong>{fmt(metrics['frac_alive'])}</strong></div>
<div class="metric"><span>LLM loss recovered</span><strong>{recovered_value}</strong></div>
</div>{figure_html}
<section><h2>Machine-readable artifacts</h2><p><code>analysis/temporal_sae_eval.json</code> and <code>analysis/temporal_sae_metrics.csv</code>.</p></section>
</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize T-SAE-compatible evaluation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output_dir) if args.output_dir else run_dir / "report"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = load_json(run_dir / "analysis" / "temporal_sae_eval.json")
    training_plot(run_dir, figures)
    reconstruction_plot(report, figures)
    smoothness_plot(report, figures)
    sparsity_plot(report, figures)
    loss_recovered_plot(report, figures)
    write_html(output, report)
    write_json(
        output / "visualization_summary.json",
        {
            "activation_metrics": report["activation_metrics"],
            "loss_recovered": report.get("loss_recovered"),
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    print(f"wrote report to {output / 'index.html'}")


if __name__ == "__main__":
    main()
