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
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.6))
    axes[0, 0].plot(
        steps,
        [row["validation"]["ema_reconstruction_fvu"] for row in history],
        color=TOTAL,
        label="Full EMA SAE",
    )
    axes[0, 0].plot(
        steps,
        [row["validation"]["ema_high_reconstruction_fvu"] for row in history],
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
        label="Normalized RDMReg",
    )
    axes[1, 0].plot(
        steps,
        [row["validation"]["high_active_fraction"] for row in history],
        color=HIGH,
        label="Learned high",
    )
    axes[1, 0].axhline(
        target_fraction, color=RDM, linestyle="--", label="RGG target"
    )
    axes[1, 1].plot(
        steps,
        [row["validation"]["high_positive_cosine"] for row in history],
        color=HIGH,
        label="Same span",
    )
    axes[1, 1].plot(
        steps,
        [row["validation"]["high_shuffled_cosine"] for row in history],
        color=NULL,
        linestyle="--",
        label="Shuffled sequence",
    )
    axes[1, 2].plot(
        steps,
        [row["validation"]["swap_reconstruction_fvu"] for row in history],
        color="#d97706",
    )
    axes[0, 0].set_title("EMA reconstruction")
    axes[0, 0].set_ylabel("FVU (lower is better)")
    axes[0, 1].set_title("High-view invariance")
    axes[0, 1].set_ylabel("Normalized MSE")
    axes[0, 2].set_title("Rectified distribution matching")
    axes[0, 2].set_ylabel("Normalized sliced W2")
    axes[1, 0].set_title("High-code sparsity control")
    axes[1, 0].set_ylabel("Active fraction")
    axes[1, 1].set_title("Shared-view specificity")
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
    online = quality["online"]
    ema = quality["ema"]
    recovered = report.get("loss_recovered")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    positions = np.arange(3)
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        [
            online["high_only_fraction_variance_explained"],
            online["low_only_fraction_variance_explained"],
            online["fraction_variance_explained"],
        ],
        width,
        color="#0ea5e9",
        label="Online SAE",
    )
    axes[0].bar(
        positions + width / 2,
        [
            ema["high_only_fraction_variance_explained"],
            ema["low_only_fraction_variance_explained"],
            ema["fraction_variance_explained"],
        ],
        width,
        color=HIGH,
        label="EMA SAE",
    )
    axes[0].set_xticks(positions, ["High", "Low", "Full"])
    axes[0].axhline(0, color="#334155", linewidth=1)
    axes[0].set_title("Online vs EMA variance explained")
    axes[0].set_ylabel("FVE")
    positions2 = np.arange(3)
    axes[1].bar(
        positions2 - width / 2,
        [online["reconstruction_cosine"], online["alive_feature_fraction"], online["high_active_fraction"]],
        width,
        color="#0ea5e9",
        label="Online SAE",
    )
    axes[1].bar(
        positions2 + width / 2,
        [ema["reconstruction_cosine"], ema["alive_feature_fraction"], ema["high_active_fraction"]],
        width,
        color=HIGH,
        label="EMA SAE",
    )
    axes[1].set_xticks(positions2, ["Cosine", "Alive", "High active"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Reconstruction and dictionary use")
    if recovered is None:
        axes[2].text(0.5, 0.5, "Loss recovered skipped", ha="center", va="center")
        axes[2].set_axis_off()
    else:
        axes[2].bar(
            ["Original", "Online recon", "EMA recon", "Zero"],
            [
                recovered["loss_original"],
                recovered["loss_reconstructed_online"],
                recovered["loss_reconstructed_ema"],
                recovered["loss_zero"],
            ],
            color=[TOTAL, "#0ea5e9", HIGH, "#dc2626"],
        )
        axes[2].set_title(
            "Loss recovered: "
            f"online={recovered['fraction_loss_recovered_online']:.3f}, "
            f"EMA={recovered['fraction_loss_recovered_ema']:.3f}"
        )
        axes[2].set_ylabel("Next-token cross entropy")
        axes[2].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "standard-sae-quality")


def invariance_plot(report: dict[str, Any], figures: Path) -> None:
    rows = report["view_invariance"]["distance_curve"]
    distance = np.asarray([row["distance"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
    axes[0].plot(
        distance,
        [row["ema_high_positive_cosine"] for row in rows],
        marker="o",
        color=HIGH,
        label="EMA same span",
    )
    axes[0].plot(
        distance,
        [row["ema_high_shuffled_cosine"] for row in rows],
        color=NULL,
        linestyle="--",
        label="EMA shuffled",
    )
    axes[0].plot(
        distance,
        [row["ema_low_positive_cosine"] for row in rows],
        color=LOW,
        linestyle=":",
        label="EMA low same span",
    )
    axes[1].plot(
        distance,
        [row["online_high_margin"] for row in rows],
        color="#0ea5e9",
        marker="o",
        label="Online",
    )
    axes[1].fill_between(
        distance,
        [row["online_high_margin_ci95_low"] for row in rows],
        [row["online_high_margin_ci95_high"] for row in rows],
        color="#0ea5e9",
        alpha=0.15,
    )
    axes[1].plot(
        distance,
        [row["ema_high_margin"] for row in rows],
        color=HIGH,
        marker="o",
        label="EMA",
    )
    axes[1].fill_between(
        distance,
        [row["ema_high_margin_ci95_low"] for row in rows],
        [row["ema_high_margin_ci95_high"] for row in rows],
        color=HIGH,
        alpha=0.12,
    )
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[2].plot(
        distance,
        [row["ema_swap_reconstruction_fvu"] for row in rows],
        marker="o",
        color=HIGH,
        label="Same-span high swap",
    )
    axes[2].plot(
        distance,
        [row["ema_shuffled_swap_fvu"] for row in rows],
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
        "high_mean_online",
        "high_mean_ema",
        "endpoint_high_ema",
        "low_mean_ema",
        "endpoint_low_ema",
        "endpoint_full_ema",
    ]
    labels = {
        "high_mean_online": "Window high mean (online)",
        "high_mean_ema": "Window high mean (EMA)",
        "endpoint_high_ema": "Endpoint high (EMA)",
        "low_mean_ema": "Window low mean (EMA)",
        "endpoint_low_ema": "Endpoint low (EMA)",
        "endpoint_full_ema": "Endpoint full (EMA)",
    }
    colors = ["#0ea5e9", HIGH, "#047857", LOW, "#65a30d", TOTAL]
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
    online = quality["online"]
    ema = quality["ema"]
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
    online_recovered = "skipped" if recovered is None else fmt(recovered["fraction_loss_recovered_online"])
    ema_recovered = "skipped" if recovered is None else fmt(recovered["fraction_loss_recovered_ema"])
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
<p class="subtitle">Two exchangeable token positions are encoded by one online SAE. High codes are aligned directly and matched to a sparse Rectified Generalized Gaussian target; the EMA SAE is the final artifact, not a training teacher.</p>
<div class="metrics">
<div class="metric"><span>Online / EMA full FVE</span><strong>{fmt(online['fraction_variance_explained'])} / {fmt(ema['fraction_variance_explained'])}</strong></div>
<div class="metric"><span>Online / EMA high active</span><strong>{fmt(online['high_active_fraction'])} / {fmt(ema['high_active_fraction'])}</strong></div>
<div class="metric"><span>Online / EMA loss recovered</span><strong>{online_recovered} / {ema_recovered}</strong></div>
<div class="metric"><span>EMA positive-shuffled margin</span><strong>{fmt(invariant['overall_ema_high_margin']['mean'])}</strong></div>
<div class="metric"><span>Held-out RDMReg</span><strong>{fmt(rdm['rdm_loss'])}</strong></div>
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
