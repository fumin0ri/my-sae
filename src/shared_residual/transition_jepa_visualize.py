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

from .io import read_jsonl, torch_load, write_json


COLORS = {
    "joint": "#2563eb",
    "fixed": "#d97706",
    "k_only": "#64748b",
    "shuffled": "#94a3b8",
    "patch": "#7c3aed",
    "ablate": "#dc2626",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_plot(run_dir: Path, figures: Path) -> None:
    reports = {
        method: load_json(run_dir / method / "training_report.json")
        for method in ("joint", "fixed", "k_only")
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    labels = {
        "joint": "Joint JEPA-SAE",
        "fixed": "Fixed SAE + predictor",
        "k_only": "Offset only",
    }
    for method, report in reports.items():
        history = report["history"]
        steps = [row["step"] for row in history]
        color = COLORS[method]
        axes[0].plot(
            steps,
            [row["validation"]["reconstruction_fvu"] for row in history],
            color=color,
            label=labels[method],
        )
        axes[1].plot(
            steps,
            [row["validation"]["code_cosine"] for row in history],
            color=color,
            label=labels[method],
        )
        axes[2].plot(
            steps,
            [row["validation"]["support_recall"] for row in history],
            color=color,
            label=labels[method],
        )
    axes[0].set_title("Residual reconstruction")
    axes[0].set_ylabel("Validation FVU")
    axes[1].set_title("Latent forecast")
    axes[1].set_ylabel("Target-code cosine")
    axes[2].set_title("Sparse support recovery")
    axes[2].set_ylabel("Top-K support recall")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "training-curves")


def offset_plot(report: dict[str, Any], figures: Path) -> None:
    curves = report["locked_test_offset_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for method, label in (
        ("joint", "Joint JEPA-SAE"),
        ("fixed", "Fixed SAE + predictor"),
        ("k_only", "Offset only"),
    ):
        rows = curves[method]
        offsets = [row["offset"] for row in rows]
        axes[0].plot(
            offsets,
            [row["code_cosine"] for row in rows],
            marker="o",
            color=COLORS[method],
            label=label,
        )
        axes[1].plot(
            offsets,
            [row["context_gain"]["mean"] for row in rows],
            marker="o",
            color=COLORS[method],
            label=label,
        )
    joint = curves["joint"]
    axes[0].plot(
        [row["offset"] for row in joint],
        [row["shuffled_context_cosine"] for row in joint],
        marker="o",
        linestyle="--",
        color=COLORS["shuffled"],
        label="Joint, shuffled z₀",
    )
    axes[0].set_title("Offset-conditioned future-code forecast")
    axes[0].set_ylabel("Cosine with EMA target code")
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[1].set_title("Does the matching z₀ matter?")
    axes[1].set_ylabel("True-context minus shuffled cosine")
    for axis in axes:
        axis.set_xticks(range(1, 10))
        axis.set_xlabel("Future offset k")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "offset-forecast")


def support_innovation_plot(report: dict[str, Any], figures: Path) -> None:
    curves = report["locked_test_offset_curve"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for method, label in (
        ("joint", "Joint JEPA-SAE"),
        ("fixed", "Fixed SAE + predictor"),
        ("k_only", "Offset only"),
    ):
        rows = curves[method]
        offsets = [row["offset"] for row in rows]
        axes[0].plot(
            offsets,
            [row["support_precision"] for row in rows],
            marker="o",
            color=COLORS[method],
            label=label,
        )
        axes[1].plot(
            offsets,
            [row["support_recall"] for row in rows],
            marker="o",
            color=COLORS[method],
            label=label,
        )
        axes[2].plot(
            offsets,
            [row["innovation_energy_fraction"] for row in rows],
            marker="o",
            color=COLORS[method],
            label=label,
        )
    axes[0].set_title("Forecast support precision")
    axes[0].set_ylabel("Precision")
    axes[1].set_title("Forecast support recall")
    axes[1].set_ylabel("Recall")
    axes[2].set_title("Unforecastable residual")
    axes[2].set_ylabel("Innovation / target energy")
    for axis in axes:
        axis.set_xticks(range(1, 10))
        axis.set_xlabel("Future offset k")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "support-and-innovation")


def mmlu_probe_plot(report: dict[str, Any], figures: Path) -> None:
    probes = report["locked_test_mmlu_probes"]
    order = [
        ("Joint z0", "joint_z0", COLORS["joint"]),
        ("Standard SAE z0", "standard_sae_z0", COLORS["fixed"]),
        ("Joint predicted z9", "joint_predicted_z9", "#0f766e"),
        ("Raw h0", "raw_h0", "#334155"),
    ]
    titles = {
        "semantics": "Semantics: correct answer",
        "context": "Context: MMLU domain",
        "syntax": "Syntax: prompt format",
    }
    positions = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for axis, (probe_axis, title) in zip(axes, titles.items()):
        values = [
            probes[probe_axis][key]["accuracy"] for _, key, _ in order
        ]
        low = [
            values[index]
            - probes[probe_axis][key]["group_bootstrap"]["ci95_low"]
            for index, (_, key, _) in enumerate(order)
        ]
        high = [
            probes[probe_axis][key]["group_bootstrap"]["ci95_high"]
            - values[index]
            for index, (_, key, _) in enumerate(order)
        ]
        axis.bar(
            positions,
            values,
            color=[color for _, _, color in order],
        )
        axis.errorbar(
            positions,
            values,
            yerr=np.asarray([low, high]),
            fmt="none",
            color="black",
            capsize=4,
        )
        axis.axhline(
            probes[probe_axis]["joint_z0"]["chance_accuracy"],
            linestyle="--",
            color="#111827",
            label="Majority chance",
        )
        axis.set_xticks(
            positions,
            [label for label, _, _ in order],
            rotation=23,
            ha="right",
        )
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Locked-test linear-probe accuracy")
    fig.tight_layout()
    save_figure(fig, figures, "mmlu-probes")


def mmlu_model_plot(report: dict[str, Any], figures: Path) -> None:
    results = report["benchmark"]["base_model_accuracy"]
    context = results["by_context"]
    syntax = results["by_syntax"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for axis, values, title in (
        (axes[0], context, "Base LLM accuracy by MMLU context"),
        (axes[1], syntax, "Base LLM accuracy by syntax template"),
    ):
        labels = list(values)
        accuracy = [values[label]["accuracy"] for label in labels]
        axis.bar(
            np.arange(len(labels)),
            accuracy,
            color=COLORS["joint"],
        )
        axis.axhline(
            results["chance_accuracy"],
            linestyle="--",
            color="#111827",
            label="Random chance",
        )
        axis.set_xticks(
            np.arange(len(labels)),
            labels,
            rotation=20,
            ha="right",
        )
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Zero-shot answer accuracy")
    fig.tight_layout()
    save_figure(fig, figures, "mmlu-base-model")


def embedding_and_heatmap(run_dir: Path, figures: Path) -> None:
    bundle = torch_load(run_dir / "analysis" / "transition_visualization.pt")
    embedding = bundle["embedding"].float().numpy()
    labels = np.asarray(bundle["context_labels"])
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    cmap = plt.get_cmap("tab10")
    for index, label in enumerate(sorted(set(labels))):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=26,
            alpha=0.7,
            color=cmap(index),
            label=label,
        )
    ax.set_xlabel("Joint z₀ PC 1")
    ax.set_ylabel("Joint z₀ PC 2")
    ax.set_title("Locked-test JEPA-SAE context states")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "context-embedding")

    heat = bundle["feature_activations"].float().numpy().T
    semantic_labels = np.asarray(bundle["semantic_labels"])
    order = np.argsort(semantic_labels)
    scale = np.quantile(heat[heat > 0], 0.98) if np.any(heat > 0) else 1.0
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    image = ax.imshow(
        heat[:, order],
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=max(scale, 1e-8),
    )
    ax.set_yticks(range(len(bundle["feature_ids"])), bundle["feature_ids"])
    ax.set_ylabel("Forecastable SAE feature")
    ax.set_xlabel("Locked-test MMLU questions (sorted by correct answer)")
    ax.set_title("Top offset-9 forecast features")
    boundaries = (
        np.flatnonzero(
            semantic_labels[order][1:] != semantic_labels[order][:-1]
        )
        + 0.5
    )
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.8)
    fig.colorbar(image, ax=ax, label="Top-K forecast activation")
    fig.tight_layout()
    save_figure(fig, figures, "forecast-feature-heatmap")


def intervention_plot(
    analysis_dir: Path,
    figures: Path,
) -> dict[str, Any] | None:
    paths = {
        "Contradictory forecast patch": analysis_dir / "intervention-patch.jsonl",
        "Forecastable-feature ablation": analysis_dir / "intervention-ablate.jsonl",
        "Norm-matched random": analysis_dir / "intervention-random.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    rows = {label: read_jsonl(path) for label, path in paths.items()}
    values = {
        label: np.asarray(
            [float(row["delta_answer_logprob"]) for row in condition]
        )
        for label, condition in rows.items()
    }
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    violin = ax.violinplot(
        [values[label] for label in paths],
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(
        violin["bodies"],
        [COLORS["patch"], COLORS["ablate"], COLORS["shuffled"]],
    ):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    violin["cmeans"].set_color("black")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_xticks(range(1, 4), list(paths))
    ax.set_ylabel("Change in target-answer log probability")
    ax.set_title("Causal edits restricted to the forecastable component")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "causal-interventions")

    paired = (
        values["Forecastable-feature ablation"]
        - values["Norm-matched random"]
    )
    rng = np.random.default_rng(991)
    bootstrap = np.asarray(
        [
            rng.choice(paired, size=len(paired), replace=True).mean()
            for _ in range(5000)
        ]
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    signs = rng.choice([-1.0, 1.0], size=(10000, len(paired)))
    null = (signs * paired[None, :]).mean(axis=1)
    observed = float(paired.mean())
    return {
        label: {
            "n": len(group),
            "mean": float(group.mean()),
            "median": float(np.median(group)),
        }
        for label, group in values.items()
    } | {
        "learned_minus_random": {
            "mean": observed,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "sign_flip_p": float(
                (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
            ),
            "cohens_dz": float(
                observed / paired.std(ddof=1)
                if len(paired) > 1 and paired.std(ddof=1) > 0
                else 0.0
            ),
        }
    }


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def write_html(
    output_dir: Path,
    report: dict[str, Any],
    interventions: dict[str, Any] | None,
) -> None:
    comparison = report["primary_joint_minus_fixed_code_cosine"]
    joint_last = report["locked_test_offset_curve"]["joint"][-1]
    fixed_last = report["locked_test_offset_curve"]["fixed"][-1]
    k_only_last = report["locked_test_offset_curve"]["k_only"][-1]
    mmlu = report["benchmark"]["base_model_accuracy"]
    probes = report["locked_test_mmlu_probes"]
    figures = [
        ("Training dynamics", "figures/training-curves.png"),
        ("Offset forecast and context gain", "figures/offset-forecast.png"),
        ("Support and innovation", "figures/support-and-innovation.png"),
        ("MMLU semantics, context, and syntax", "figures/mmlu-probes.png"),
        ("Base LLM MMLU accuracy", "figures/mmlu-base-model.png"),
        ("JEPA-SAE context embedding", "figures/context-embedding.png"),
        ("Forecast feature heatmap", "figures/forecast-feature-heatmap.png"),
    ]
    if interventions is not None:
        figures.append(("Causal interventions", "figures/causal-interventions.png"))
    figure_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2>'
        f'<img src="{path}" alt="{html.escape(title)}"></section>'
        for title, path in figures
    )
    causal_html = ""
    if interventions is not None:
        learned = interventions["learned_minus_random"]
        causal_html = (
            "<section><h2>Causal summary</h2>"
            "<p>Forecastable-feature ablation minus norm-matched random: "
            f"mean={fmt(learned['mean'])}, 95% CI "
            f"[{fmt(learned['ci95_low'])}, {fmt(learned['ci95_high'])}], "
            f"sign-flip p={fmt(learned['sign_flip_p'])}.</p></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Offset-conditioned transition JEPA-SAE report</title>
<style>
:root {{ --bg:#f5f7fb; --card:#fff; --fg:#172033; --muted:#64748b; --line:#dbe2ea; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
main {{ max-width:1120px; margin:auto; padding:36px 20px 70px; }}
.subtitle {{ color:var(--muted); max-width:900px; }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:24px 0; }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.metric span {{ display:block; color:var(--muted); font-size:.84rem; }} .metric strong {{ display:block; font-size:1.35rem; margin-top:5px; }}
section {{ margin-top:34px; }} section img {{ width:100%; background:white; border:1px solid var(--line); border-radius:10px; }}
code {{ color:inherit; }}
</style></head><body><main>
<h1>Offset-conditioned transition JEPA-SAE</h1>
<p class="subtitle">A Top-K code at h₀ and offset k forecasts an EMA SAE code at hₖ. The target may change with k, while the predictor is restricted to one present state and one offset.</p>
<div class="metrics">
<div class="metric"><span>Joint − fixed cosine</span><strong>{fmt(comparison['mean'])}</strong></div>
<div class="metric"><span>Group-bootstrap 95% CI</span><strong>{fmt(comparison['ci95_low'])} to {fmt(comparison['ci95_high'])}</strong></div>
<div class="metric"><span>Joint offset-9 cosine</span><strong>{fmt(joint_last['code_cosine'])}</strong></div>
<div class="metric"><span>Fixed offset-9 cosine</span><strong>{fmt(fixed_last['code_cosine'])}</strong></div>
<div class="metric"><span>k-only offset-9 cosine</span><strong>{fmt(k_only_last['code_cosine'])}</strong></div>
<div class="metric"><span>Joint offset-9 context gain</span><strong>{fmt(joint_last['context_gain']['mean'])}</strong></div>
<div class="metric"><span>Base LLM MMLU answer accuracy</span><strong>{fmt(mmlu['accuracy'])}</strong></div>
<div class="metric"><span>Joint predicted z9 semantics</span><strong>{fmt(probes['semantics']['joint_predicted_z9']['accuracy'])}</strong></div>
<div class="metric"><span>Joint predicted z9 context</span><strong>{fmt(probes['context']['joint_predicted_z9']['accuracy'])}</strong></div>
<div class="metric"><span>Joint predicted z9 syntax</span><strong>{fmt(probes['syntax']['joint_predicted_z9']['accuracy'])}</strong></div>
</div>
{figure_html}
{causal_html}
<section><h2>Interpretation boundary</h2>
<p><code>P(z₀,k)</code> estimates what is forecastable before observing intervening tokens. It is not a deterministic transition operator. The innovation contains later token information, lexical realization, and unforecastable updates.</p></section>
<p class="subtitle">Machine-readable results: <code>analysis/transition_jepa_report.json</code>, <code>analysis/mmlu_probe_accuracy.csv</code>, <code>analysis/transition_offset_metrics.csv</code>, and <code>analysis/transition_visualization.pt</code>.</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(
        document,
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build figures and HTML for transition JEPA-SAE"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "report"
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = load_json(run_dir / "analysis" / "transition_jepa_report.json")
    training_plot(run_dir, figures)
    offset_plot(report, figures)
    support_innovation_plot(report, figures)
    mmlu_probe_plot(report, figures)
    mmlu_model_plot(report, figures)
    embedding_and_heatmap(run_dir, figures)
    interventions = intervention_plot(run_dir / "analysis", figures)
    write_html(output_dir, report, interventions)
    write_json(
        output_dir / "visualization_summary.json",
        {
            "primary_comparison": report[
                "primary_joint_minus_fixed_code_cosine"
            ],
            "offset_curve": report["locked_test_offset_curve"],
            "mmlu_probes": report["locked_test_mmlu_probes"],
            "base_mmlu_model": report["benchmark"]["base_model_accuracy"],
            "interventions": interventions,
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    print(f"wrote report to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
