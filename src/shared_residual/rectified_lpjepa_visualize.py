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
RDM = "#7c3aed"


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
    report = load_json(run_dir / "model" / "training_report.json")
    history = report["history"]
    steps = [row["step"] for row in history]
    target_fraction = report["rgg_target"]["active_fraction"]
    sparse_target_fraction = (
        report["architecture"]["high_k"] / report["architecture"]["d_high"]
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.6))
    axes[0, 0].plot(
        steps,
        [row["validation"]["full_reconstruction_fvu"] for row in history],
        color=TOTAL,
        label="Full SAE",
    )
    axes[0, 0].plot(
        steps,
        [row["validation"]["high_reconstruction_fvu"] for row in history],
        color=HIGH,
        linestyle="--",
        label="High only",
    )
    axes[0, 1].plot(
        steps,
        [row["validation"]["invariance_loss"] for row in history],
        color=HIGH,
    )
    axes[0, 2].plot(
        steps,
        [row["validation"]["rdm_loss"] for row in history],
        color=RDM,
        label="Combined RDMReg",
    )
    axes[0, 2].plot(
        steps,
        [row["validation"]["axis_aligned_rdm_loss"] for row in history],
        color="#db2777",
        linestyle="--",
        label="Axis-aligned",
    )
    axes[0, 2].plot(
        steps,
        [row["validation"]["random_projection_rdm_loss"] for row in history],
        color="#8b5cf6",
        linestyle=":",
        label="Random projection",
    )
    axes[1, 0].plot(
        steps,
        [row["validation"]["high_active_fraction"] for row in history],
        color=HIGH,
        label="Sparse Top-K high",
    )
    axes[1, 0].plot(
        steps,
        [row["validation"]["dense_high_active_fraction"] for row in history],
        color="#db2777",
        label="Dense JEPA high",
    )
    axes[1, 0].axhline(
        target_fraction, color=RDM, linestyle="--", label="Dense RGG target"
    )
    axes[1, 0].axhline(
        sparse_target_fraction,
        color=HIGH,
        linestyle=":",
        label="Top-K target",
    )
    axes[1, 1].plot(
        steps,
        [row["validation"]["high_positive_cosine"] for row in history],
        color=HIGH,
        label="Sparse same span",
    )
    axes[1, 1].plot(
        steps,
        [row["validation"]["high_shuffled_cosine"] for row in history],
        color=NULL,
        linestyle="--",
        label="Shuffled sequence",
    )
    axes[1, 1].plot(
        steps,
        [row["validation"]["dense_high_positive_cosine"] for row in history],
        color="#db2777",
        linestyle=":",
        label="Dense same span",
    )
    axes[1, 2].plot(
        steps,
        [row["validation"]["swap_reconstruction_fvu"] for row in history],
        color="#d97706",
    )
    axes[0, 0].set_title("SAE reconstruction")
    axes[0, 0].set_ylabel("FVU (lower is better)")
    axes[0, 1].set_title("High-view invariance")
    axes[0, 1].set_ylabel("Normalized MSE")
    axes[0, 2].set_title("Rectified distribution matching")
    axes[0, 2].set_ylabel("Normalized W2")
    axes[1, 0].set_title("Dense candidates vs final Top-K")
    axes[1, 0].set_ylabel("Active fraction")
    axes[1, 1].set_title("Sparse evaluation vs dense training view")
    axes[1, 1].set_ylabel("High-code cosine")
    axes[1, 2].set_title("Same-span high-code swap")
    axes[1, 2].set_ylabel("FVU (lower is better)")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 2].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "training")


def sae_quality_plot(report: dict[str, Any], figures: Path) -> None:
    quality = report["standard_sae_quality"]
    recovered = report.get("loss_recovered")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    positions = np.arange(3)
    width = 0.58
    axes[0].bar(
        positions,
        [
            quality["high_only_fraction_variance_explained"],
            quality["low_only_fraction_variance_explained"],
            quality["fraction_variance_explained"],
        ],
        width,
        color=[HIGH, LOW, TOTAL],
    )
    axes[0].set_xticks(positions, ["High", "Low", "Full"])
    axes[0].axhline(0, color="#334155", linewidth=1)
    axes[0].set_title("Variance explained")
    axes[0].set_ylabel("FVE")
    positions2 = np.arange(3)
    axes[1].bar(
        positions2,
        [quality["reconstruction_cosine"], quality["alive_feature_fraction"], quality["high_topk_saturation_fraction"]],
        width,
        color=[TOTAL, HIGH, RDM],
    )
    axes[1].set_xticks(positions2, ["Cosine", "Alive", "Top-K exact"])
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
            f"Loss recovered: {recovered['fraction_loss_recovered']:.3f}"
        )
        axes[2].set_ylabel("Next-token cross entropy")
        axes[2].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "standard-sae-quality")


def invariance_plot(report: dict[str, Any], figures: Path) -> None:
    rows = report["view_invariance"]["distance_curve"]
    distance = np.asarray([row["distance"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
    axes[0].plot(
        distance,
        [row["high_positive_cosine"] for row in rows],
        marker="o",
        color=HIGH,
        label="High same span",
    )
    axes[0].plot(
        distance,
        [row["high_shuffled_cosine"] for row in rows],
        color=NULL,
        linestyle="--",
        label="High shuffled",
    )
    axes[0].plot(
        distance,
        [row["low_positive_cosine"] for row in rows],
        color=LOW,
        linestyle=":",
        label="Low same span",
    )
    axes[0].plot(
        distance,
        [row["dense_high_positive_cosine"] for row in rows],
        color="#db2777",
        linestyle="-.",
        label="Dense high same span",
    )
    axes[1].plot(
        distance,
        [row["high_margin"] for row in rows],
        color=HIGH,
        marker="o",
        label="Positive - shuffled",
    )
    axes[1].plot(
        distance,
        [row["dense_high_margin"] for row in rows],
        color="#db2777",
        linestyle=":",
        label="Dense training margin",
    )
    axes[1].fill_between(
        distance,
        [row["high_margin_ci95_low"] for row in rows],
        [row["high_margin_ci95_high"] for row in rows],
        color=HIGH,
        alpha=0.15,
    )
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[2].plot(
        distance,
        [row["reconstruction_fvu"] for row in rows],
        color=TOTAL,
        label="Ordinary reconstruction",
    )
    axes[2].plot(
        distance,
        [row["swap_reconstruction_fvu"] for row in rows],
        marker="o",
        color=HIGH,
        label="Same-span high swap",
    )
    axes[2].plot(
        distance,
        [row["shuffled_swap_fvu"] for row in rows],
        color=NULL,
        linestyle="--",
        label="Shuffled high swap",
    )
    axes[0].set_title("High/low view similarity")
    axes[0].set_ylabel("Cosine")
    axes[1].set_title("Same-span specificity")
    axes[1].set_ylabel("Positive minus shuffled cosine")
    axes[2].set_title("Exchangeability of high code")
    axes[2].set_ylabel("Reconstruction FVU")
    for axis in axes:
        axis.set_xlabel("Token distance")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "view-invariance")


def probe_plot(report: dict[str, Any], figures: Path) -> None:
    probes = report["mmlu_probe_accuracy"]
    preferred = [
        "high_mean",
        "endpoint_high",
        "low_mean",
        "endpoint_low",
        "endpoint_full",
    ]
    labels = {
        "high_mean": "Window high mean",
        "endpoint_high": "Endpoint high",
        "low_mean": "Window low mean",
        "endpoint_low": "Endpoint low",
        "endpoint_full": "Endpoint full",
    }
    colors = [HIGH, "#047857", LOW, "#65a30d", TOTAL]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    for axis, (probe_axis, results) in zip(axes, probes.items()):
        names = [name for name in preferred if name in results]
        values = [results[name]["balanced_accuracy"] for name in names]
        chance = np.mean([results[name]["chance_accuracy"] for name in names])
        axis.bar(np.arange(len(names)), values, color=colors[: len(names)])
        axis.axhline(chance, color="#334155", linestyle="--", label="Chance")
        axis.set_xticks(
            np.arange(len(names)),
            [labels[name] for name in names],
            rotation=35,
            ha="right",
        )
        axis.set_title(f"{probe_axis.capitalize()} probe")
        axis.set_ylabel("Locked-test balanced accuracy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "mmlu-probes")


def causal_plot(run_dir: Path, figures: Path) -> dict[str, Any] | None:
    conditions = {
        "Patch": (run_dir / "analysis" / "intervention-patch.jsonl", "delta_contrast_minus_target_logprob"),
        "Ablate": (run_dir / "analysis" / "intervention-ablate.jsonl", "delta_answer_logprob"),
        "Random ablate": (run_dir / "analysis" / "intervention-random.jsonl", "delta_answer_logprob"),
    }
    summary: dict[str, Any] = {}
    for label, (path, metric) in conditions.items():
        values = np.asarray(
            [float(row[metric]) for row in load_jsonl(path) if metric in row],
            dtype=float,
        )
        if not len(values):
            continue
        summary[label] = {
            "metric": metric,
            "n": len(values),
            "mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        }
    if not summary:
        return None
    labels = list(summary)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.bar(
        labels,
        [summary[label]["mean"] for label in labels],
        yerr=[1.96 * summary[label]["standard_error"] for label in labels],
        capsize=5,
        color=[HIGH, "#dc2626", NULL][: len(labels)],
    )
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_title("Causal effect of Rectified high features")
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
    invariant = report["view_invariance"]
    rdm = report["rdm_validation"]
    base_mmlu = report["base_model_mmlu_accuracy"].get("accuracy")
    figures = [
        ("Training", "figures/training.png"),
        ("Conventional SAE quality", "figures/standard-sae-quality.png"),
        ("Shared-view validity and null control", "figures/view-invariance.png"),
        ("MMLU semantic/context/syntax probes", "figures/mmlu-probes.png"),
    ]
    if causal_summary is not None:
        figures.append(("Causal interventions", "figures/causal-interventions.png"))
    figure_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{path}" alt="{html.escape(title)}"></section>'
        for title, path in figures
    )
    recovered_value = "skipped" if recovered is None else fmt(recovered["fraction_loss_recovered"])
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rectified LpJEPA-SAE evaluation</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--fg:#172033;--muted:#64748b;--line:#dbe2ea}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1120px;margin:auto;padding:36px 20px 70px}}.subtitle{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.metric{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.metric span{{display:block;color:var(--muted);font-size:.84rem}}.metric strong{{font-size:1.35rem}}
section{{margin-top:34px}}section img{{width:100%;background:white;border:1px solid var(--line);border-radius:10px}}
</style></head><body><main>
<h1>Predictor-free Rectified LpJEPA-SAE</h1>
<p class="subtitle">Dense ReLU high candidates learn view invariance and RDMReg. ReLU+Top-K high codes are the only codes used for reconstruction, evaluation, and intervention.</p>
<div class="metrics">
<div class="metric"><span>Full SAE FVE</span><strong>{fmt(quality['fraction_variance_explained'])}</strong></div>
<div class="metric"><span>Sparse high L0 / target</span><strong>{fmt(quality['high_l0'])} / {fmt(report['checkpoint']['config']['high_k'])}</strong></div>
<div class="metric"><span>Top-K exact fraction</span><strong>{fmt(quality['high_topk_saturation_fraction'])}</strong></div>
<div class="metric"><span>Dense / sparse high margin</span><strong>{fmt(invariant['overall_dense_high_margin']['mean'])} / {fmt(invariant['overall_high_margin']['mean'])}</strong></div>
<div class="metric"><span>Loss recovered</span><strong>{recovered_value}</strong></div>
<div class="metric"><span>Held-out RDMReg</span><strong>{fmt(rdm['rdm_loss'])}</strong></div>
<div class="metric"><span>Axis-aligned RDMReg</span><strong>{fmt(rdm['axis_aligned_rdm_loss'])}</strong></div>
<div class="metric"><span>Base-model MMLU</span><strong>{fmt(base_mmlu)}</strong></div>
</div>{figure_html}
<section><h2>Machine-readable artifacts</h2><p><code>analysis/rectified_lpjepa_report.json</code>, <code>distance_metrics.csv</code>, and <code>mmlu_probe_accuracy.csv</code>.</p></section>
</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize SAE quality and predictor-free Rectified LpJEPA validity"
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
    report = load_json(run_dir / "analysis" / "rectified_lpjepa_report.json")
    training_plot(run_dir, figures)
    sae_quality_plot(report, figures)
    invariance_plot(report, figures)
    probe_plot(report, figures)
    causal_summary = causal_plot(run_dir, figures)
    write_html(output, report, causal_summary)
    write_json(
        output / "visualization_summary.json",
        {
            "method": "predictor-free Rectified LpJEPA-SAE",
            "rgg_target": load_json(run_dir / "model" / "training_report.json")["rgg_target"],
            "standard_sae_quality": report["standard_sae_quality"],
            "loss_recovered": report.get("loss_recovered"),
            "view_invariance": report["view_invariance"],
            "rdm_validation": report["rdm_validation"],
            "causal_interventions": causal_summary,
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    print(f"wrote report to {output / 'index.html'}")


if __name__ == "__main__":
    main()
