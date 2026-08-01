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
NULL = "#94a3b8"
POSITION = "#d97706"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_plot(run_dir: Path, figures: Path) -> None:
    history = load_json(run_dir / "model" / "training_report.json")["history"]
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    axes[0].plot(
        steps,
        [row["validation"]["ema_reconstruction_fvu"] for row in history],
        color=TOTAL,
        label="Full EMA SAE",
    )
    axes[0].plot(
        steps,
        [row["validation"]["ema_high_reconstruction_fvu"] for row in history],
        color=HIGH,
        linestyle="--",
        label="High only",
    )
    axes[1].plot(
        steps,
        [row["validation"]["code_cosine"] for row in history],
        color=HIGH,
    )
    axes[2].plot(
        steps,
        [row["validation"]["support_recall"] for row in history],
        color="#7c3aed",
        label="Support recall",
    )
    axes[2].plot(
        steps,
        [row["validation"]["support_precision"] for row in history],
        color="#db2777",
        label="Support precision",
    )
    axes[0].set_title("Endpoint reconstruction")
    axes[0].set_ylabel("FVU (lower is better)")
    axes[1].set_title("Endpoint-code prediction")
    axes[1].set_ylabel("Cosine (higher is better)")
    axes[2].set_title("Predicted sparse support")
    axes[2].set_ylabel("Score")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "training")


def sae_quality_plot(report: dict[str, Any], figures: Path) -> None:
    quality = report["standard_sae_quality"]
    recovered = report.get("loss_recovered")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    axes[0].bar(
        ["High", "Low", "Full"],
        [
            quality["high_only_fraction_variance_explained"],
            quality["low_only_fraction_variance_explained"],
            quality["fraction_variance_explained"],
        ],
        color=[HIGH, LOW, TOTAL],
    )
    axes[0].axhline(0, color="#334155", linewidth=1)
    axes[0].set_title("Variance explained")
    axes[0].set_ylabel("FVE (higher is better)")
    axes[1].bar(
        ["Cosine", "Alive"],
        [quality["reconstruction_cosine"], quality["alive_feature_fraction"]],
        color=[TOTAL, HIGH],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Reconstruction and dictionary use")
    if recovered is None:
        axes[2].text(0.5, 0.5, "Loss recovered skipped", ha="center", va="center")
        axes[2].set_axis_off()
    else:
        axes[2].bar(
            ["Original", "SAE recon", "Zero"],
            [
                recovered["loss_original"],
                recovered["loss_reconstructed"],
                recovered["loss_zero"],
            ],
            color=[TOTAL, HIGH, "#dc2626"],
        )
        axes[2].set_title(
            f"Loss recovered = {recovered['fraction_loss_recovered']:.3f}"
        )
        axes[2].set_ylabel("Next-token cross entropy")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "standard-sae-quality")


def forecast_plot(report: dict[str, Any], figures: Path) -> None:
    rows = sorted(
        report["forecast_validity"]["horizon_curve"], key=lambda row: row["horizon"]
    )
    horizon = np.asarray([row["horizon"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    axes[0].plot(
        horizon,
        [row["code_cosine"] for row in rows],
        marker="o",
        color=HIGH,
        label="Learned context predictor",
    )
    axes[0].plot(
        horizon,
        [row["shuffled_context_cosine"] for row in rows],
        color=NULL,
        linestyle="--",
        label="Shuffled context",
    )
    axes[0].plot(
        horizon,
        [row["position_only_cosine"] for row in rows],
        color=POSITION,
        linestyle=":",
        label="Position only",
    )
    axes[0].plot(
        horizon,
        [row["context_target_cosine"] for row in rows],
        color=TOTAL,
        alpha=0.7,
        label="Raw context code",
    )
    gain_shuffled = [row["context_gain_over_shuffled"]["mean"] for row in rows]
    gain_position = [
        row["context_gain_over_position_only"]["mean"] for row in rows
    ]
    axes[1].plot(horizon, gain_shuffled, marker="o", color=HIGH, label="vs shuffled")
    axes[1].plot(horizon, gain_position, marker="o", color=POSITION, label="vs position")
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[2].plot(
        horizon,
        [row["residual_prediction_fvu"] for row in rows],
        marker="o",
        color="#7c3aed",
    )
    axes[0].set_title("Endpoint-code forecast controls")
    axes[0].set_ylabel("Cosine with EMA endpoint high code")
    axes[1].set_title("Context-dependent forecast gain")
    axes[1].set_ylabel("Paired cosine gain")
    axes[2].set_title("Predicted high residual")
    axes[2].set_ylabel("FVU (lower is better)")
    for axis in axes:
        axis.set_xlabel("Forecast horizon (tokens)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "forecast-validity")


def probe_plot(report: dict[str, Any], figures: Path) -> None:
    probes = report["mmlu_probe_accuracy"]
    preferred = [
        "context_high",
        "predicted_endpoint_high",
        "endpoint_high",
        "context_low",
        "endpoint_low",
        "endpoint_full",
    ]
    labels = {
        "context_high": "Context high",
        "predicted_endpoint_high": "Predicted high",
        "endpoint_high": "Endpoint high",
        "context_low": "Context low",
        "endpoint_low": "Endpoint low",
        "endpoint_full": "Endpoint full",
    }
    colors = [HIGH, "#0d9488", "#047857", LOW, "#65a30d", TOTAL]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), sharey=False)
    for axis, (probe_axis, results) in zip(axes, probes.items()):
        names = [name for name in preferred if name in results]
        values = [results[name]["balanced_accuracy"] for name in names]
        chance = np.mean([results[name]["chance_accuracy"] for name in names])
        axis.bar(
            np.arange(len(names)), values, color=colors[: len(names)]
        )
        axis.axhline(chance, color="#334155", linestyle="--", label="Chance")
        axis.set_xticks(
            np.arange(len(names)), [labels[name] for name in names], rotation=35, ha="right"
        )
        axis.set_title(f"{probe_axis.capitalize()} probe")
        axis.set_ylabel("Locked-test balanced accuracy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "mmlu-probes")


def causal_plot(run_dir: Path, figures: Path) -> dict[str, Any] | None:
    conditions = {
        "Patch": (
            run_dir / "analysis" / "intervention-patch.jsonl",
            "delta_contrast_minus_target_logprob",
        ),
        "Ablate": (
            run_dir / "analysis" / "intervention-ablate.jsonl",
            "delta_answer_logprob",
        ),
        "Random ablate": (
            run_dir / "analysis" / "intervention-random.jsonl",
            "delta_answer_logprob",
        ),
    }
    summary: dict[str, Any] = {}
    for label, (path, metric) in conditions.items():
        rows = load_jsonl(path)
        values = np.asarray(
            [float(row[metric]) for row in rows if metric in row], dtype=float
        )
        if not len(values):
            continue
        summary[label] = {
            "metric": metric,
            "n": len(values),
            "mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0,
        }
    if not summary:
        return None
    labels = list(summary)
    means = [summary[label]["mean"] for label in labels]
    errors = [1.96 * summary[label]["standard_error"] for label in labels]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
        color=[HIGH, "#dc2626", NULL][: len(labels)],
    )
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_title("Causal effect of forecastable high features")
    ax.set_ylabel("Log-probability change (95% normal CI)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "causal-interventions")
    return summary


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (float, int)) else html.escape(str(value))


def write_html(
    output: Path,
    report: dict[str, Any],
    causal_summary: dict[str, Any] | None,
) -> None:
    quality = report["standard_sae_quality"]
    recovered = report.get("loss_recovered")
    forecast = report["forecast_validity"]
    longest = forecast["longest_horizon"]
    base_mmlu = report["base_model_mmlu_accuracy"].get("accuracy")
    figures = [
        ("Training", "figures/training.png"),
        ("Conventional SAE quality", "figures/standard-sae-quality.png"),
        ("Forecast validity and null controls", "figures/forecast-validity.png"),
        ("MMLU semantic/context/syntax probes", "figures/mmlu-probes.png"),
    ]
    if causal_summary is not None:
        figures.append(("Causal interventions", "figures/causal-interventions.png"))
    figure_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{path}" alt="{html.escape(title)}"></section>'
        for title, path in figures
    )
    recovered_value = (
        "skipped" if recovered is None else fmt(recovered["fraction_loss_recovered"])
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>High/low JEPA-SAE evaluation</title>
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
<p class="subtitle">Conventional SAE quality plus locked-test evidence that endpoint prediction uses context rather than a position-only or shuffled-context shortcut.</p>
<div class="metrics">
<div class="metric"><span>Full FVE</span><strong>{fmt(quality['fraction_variance_explained'])}</strong></div>
<div class="metric"><span>Reconstruction cosine</span><strong>{fmt(quality['reconstruction_cosine'])}</strong></div>
<div class="metric"><span>Alive features</span><strong>{fmt(quality['alive_feature_fraction'])}</strong></div>
<div class="metric"><span>Loss recovered</span><strong>{recovered_value}</strong></div>
<div class="metric"><span>Longest-horizon gain vs shuffled</span><strong>{fmt(longest['context_gain_over_shuffled']['mean'])}</strong></div>
<div class="metric"><span>Longest-horizon gain vs position</span><strong>{fmt(longest['context_gain_over_position_only']['mean'])}</strong></div>
<div class="metric"><span>Base-model MMLU</span><strong>{fmt(base_mmlu)}</strong></div>
</div>{figure_html}
<section><h2>Machine-readable artifacts</h2><p><code>analysis/transition_jepa_report.json</code>, <code>transition_horizon_metrics.csv</code>, and <code>mmlu_probe_accuracy.csv</code>.</p></section>
</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize standard SAE and endpoint-forecast validity results"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output_dir) if args.output_dir else run_dir / "report"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = load_json(run_dir / "analysis" / "transition_jepa_report.json")
    training_plot(run_dir, figures)
    sae_quality_plot(report, figures)
    forecast_plot(report, figures)
    probe_plot(report, figures)
    causal_summary = causal_plot(run_dir, figures)
    write_html(output, report, causal_summary)
    write_json(
        output / "visualization_summary.json",
        {
            "standard_sae_quality": report["standard_sae_quality"],
            "loss_recovered": report.get("loss_recovered"),
            "forecast_validity": report["forecast_validity"],
            "causal_interventions": causal_summary,
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    print(f"wrote report to {output / 'index.html'}")


if __name__ == "__main__":
    main()
