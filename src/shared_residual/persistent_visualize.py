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
    "persistent": "#2563eb",
    "standard": "#d97706",
    "null": "#94a3b8",
    "raw": "#64748b",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_plot(
    proposed: dict[str, Any],
    baseline: dict[str, Any],
    figures: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for report, label, color in (
        (proposed, "Direct z₀→each zⱼ", COLORS["persistent"]),
        (baseline, "Standard SAE", COLORS["standard"]),
    ):
        history = report["history"]
        steps = [row["step"] for row in history]
        axes[0].plot(
            steps,
            [row["validation"]["reconstruction_fvu"] for row in history],
            color=color,
            label=label,
        )
        axes[1].plot(
            steps,
            [row["validation"]["code_cosine"] for row in history],
            color=color,
            label=label,
        )
    axes[0].set_title("Residual reconstruction")
    axes[0].set_ylabel("Validation FVU (lower is better)")
    axes[1].set_title("Individual code persistence")
    axes[1].set_ylabel("Mean z₀-to-zⱼ cosine")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "training-curves")


def offset_cosine_plot(report: dict[str, Any], figures: Path) -> None:
    curves = report["locked_test_offset_curve"]
    proposed = curves["persistent"]
    standard = curves["standard_sae"]
    offsets = [row["offset"] for row in proposed]
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    ax.plot(
        offsets,
        [row["positive_cosine"] for row in proposed],
        marker="o",
        color=COLORS["persistent"],
        label="Persistent SAE: same window",
    )
    ax.plot(
        offsets,
        [row["shuffled_cosine"] for row in proposed],
        marker="o",
        linestyle="--",
        color=COLORS["null"],
        label="Persistent SAE: different-window null",
    )
    ax.plot(
        offsets,
        [row["positive_cosine"] for row in standard],
        marker="s",
        color=COLORS["standard"],
        label="Standard SAE: same window",
    )
    ax.set_xticks(offsets)
    ax.set_xlabel("Target offset j (tokens after z₀)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Does z₀ persist to each zⱼ separately?")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "offset-cosine")


def support_plot(report: dict[str, Any], figures: Path) -> None:
    curves = report["locked_test_offset_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for key, label, color in (
        ("persistent", "Direct z₀→each zⱼ", COLORS["persistent"]),
        ("standard_sae", "Standard SAE", COLORS["standard"]),
    ):
        rows = curves[key]
        offsets = [row["offset"] for row in rows]
        axes[0].plot(
            offsets,
            [row["support_survival"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            offsets,
            [row["support_jaccard"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_title("Conditional feature survival")
    axes[0].set_ylabel("P(feature active at j | active at 0)")
    axes[1].set_title("Sparse-support overlap")
    axes[1].set_ylabel("Support Jaccard")
    for axis in axes:
        axis.set_xticks(range(1, 10))
        axis.set_xlabel("Target offset j")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "support-persistence")


def discrimination_plot(report: dict[str, Any], figures: Path) -> None:
    curves = report["locked_test_offset_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for key, label, color in (
        ("persistent", "Direct z₀→each zⱼ", COLORS["persistent"]),
        ("standard_sae", "Standard SAE", COLORS["standard"]),
    ):
        rows = curves[key]
        offsets = [row["offset"] for row in rows]
        axes[0].plot(
            offsets,
            [row["same_vs_shuffled_auc"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            offsets,
            [row["same_group_retrieval_at_1"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].axhline(0.5, linestyle="--", color=COLORS["null"])
    axes[0].set_title("Same-window vs shuffled discrimination")
    axes[0].set_ylabel("ROC AUC")
    axes[1].set_title("Cross-position state retrieval")
    axes[1].set_ylabel("Same-group retrieval@1")
    for axis in axes:
        axis.set_xticks(range(1, 10))
        axis.set_xlabel("Target offset j")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "window-discrimination")


def probe_plot(report: dict[str, Any], figures: Path) -> None:
    probes = report["locked_test_state_probes"]
    order = [
        ("Persistent z₀", "persistent_z0", COLORS["persistent"]),
        ("Standard SAE z₀", "standard_sae_z0", COLORS["standard"]),
        ("Raw h₀", "raw_h0", COLORS["raw"]),
    ]
    values = [probes[key]["accuracy"] for _, key, _ in order]
    low = [
        values[index] - probes[key]["group_bootstrap"]["ci95_low"]
        for index, (_, key, _) in enumerate(order)
    ]
    high = [
        probes[key]["group_bootstrap"]["ci95_high"] - values[index]
        for index, (_, key, _) in enumerate(order)
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    positions = np.arange(len(order))
    ax.bar(positions, values, color=[color for _, _, color in order])
    ax.errorbar(
        positions,
        values,
        yerr=np.asarray([low, high]),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(
        probes["persistent_z0"]["chance_accuracy"],
        linestyle="--",
        color="#334155",
        label="Majority chance",
    )
    ax.set_xticks(positions, [label for label, _, _ in order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Locked-test state accuracy")
    ax.set_title("Semantic content already present at token 0")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "state-probes")


def embedding_and_heatmap(run_dir: Path, figures: Path) -> None:
    bundle = torch_load(run_dir / "analysis" / "persistent_visualization.pt")
    embedding = bundle["embedding"].float().numpy()
    labels = np.asarray(bundle["labels"])
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    cmap = plt.get_cmap("tab10")
    for index, label in enumerate(sorted(set(labels))):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=28,
            alpha=0.7,
            color=cmap(index),
            label=label,
        )
    ax.set_xlabel("Persistent z₀ PC 1")
    ax.set_ylabel("Persistent z₀ PC 2")
    ax.set_title("Locked-test sparse context states")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "persistent-code-embedding")

    heat = bundle["feature_activations"].float().numpy().T
    order = np.argsort(labels)
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
    ax.set_ylabel("Persistent SAE feature")
    ax.set_xlabel("Locked-test windows (sorted by state)")
    ax.set_title("Most persistent z₀ features")
    boundaries = np.flatnonzero(labels[order][1:] != labels[order][:-1]) + 0.5
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.8)
    fig.colorbar(image, ax=ax, label="Activation at token 0")
    fig.tight_layout()
    save_figure(fig, figures, "persistent-feature-heatmap")


def intervention_plot(
    analysis_dir: Path,
    figures: Path,
) -> dict[str, Any] | None:
    paths = {
        "Contradictory z₀ patch": analysis_dir / "intervention-patch.jsonl",
        "Persistent z₀ ablation": analysis_dir / "intervention-ablate.jsonl",
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
    fig, ax = plt.subplots(figsize=(8.7, 4.6))
    violin = ax.violinplot(
        [values[label] for label in paths],
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(
        violin["bodies"],
        ["#7c3aed", "#dc2626", COLORS["null"]],
    ):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    violin["cmeans"].set_color("black")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_xticks(range(1, 4), list(paths))
    ax.set_ylabel("Change in target-answer log probability")
    ax.set_title("Causal writes of z₀ at future positions 1…9")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "causal-interventions")

    paired = values["Persistent z₀ ablation"] - values["Norm-matched random"]
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
    proposed = report["locked_test_offset_curve"]["persistent"]
    baseline = report["locked_test_offset_curve"]["standard_sae"]
    last = proposed[-1]
    last_baseline = baseline[-1]
    probe = report["locked_test_state_probes"]["persistent_z0"]
    diagnostics = report["collapse_diagnostics"]["persistent_z0"]
    figures = [
        ("Training dynamics", "figures/training-curves.png"),
        ("Individual z₀-to-zⱼ similarity", "figures/offset-cosine.png"),
        ("Sparse feature survival", "figures/support-persistence.png"),
        ("Different-window discrimination", "figures/window-discrimination.png"),
        ("Locked-test state probes", "figures/state-probes.png"),
        ("Sparse context-state embedding", "figures/persistent-code-embedding.png"),
        ("Persistent feature heatmap", "figures/persistent-feature-heatmap.png"),
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
            "<p>Persistent z₀ ablation minus norm-matched random: "
            f"mean={fmt(learned['mean'])}, 95% CI "
            f"[{fmt(learned['ci95_low'])}, {fmt(learned['ci95_high'])}], "
            f"sign-flip p={fmt(learned['sign_flip_p'])}, "
            f"Cohen's dz={fmt(learned['cohens_dz'])}.</p></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Direct persistent-SAE research report</title>
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
<h1>Direct persistent sparse states</h1>
<p class="subtitle">The context is exactly z₀. It is compared independently with each stop-gradient EMA target z₁…z₉. No target mean, Transformer predictor, or position embedding is used. Same-window scores are always shown beside a different-window null and a standard-SAE control.</p>
<div class="metrics">
<div class="metric"><span>Offset-9 cosine</span><strong>{fmt(last['positive_cosine'])}</strong></div>
<div class="metric"><span>Offset-9 shuffled null</span><strong>{fmt(last['shuffled_cosine'])}</strong></div>
<div class="metric"><span>Offset-9 standard SAE</span><strong>{fmt(last_baseline['positive_cosine'])}</strong></div>
<div class="metric"><span>Offset-9 survival</span><strong>{fmt(last['support_survival'])}</strong></div>
<div class="metric"><span>z₀ state-probe accuracy</span><strong>{fmt(probe['accuracy'])}</strong></div>
<div class="metric"><span>Dead feature fraction</span><strong>{fmt(diagnostics['dead_dimension_fraction'])}</strong></div>
</div>
{figure_html}
{causal_html}
<section><h2>Interpretation boundary</h2>
<p>These results identify a sparse state already present at the first token and stable across the following nine residual positions. They do not by themselves establish that the state is the model's complete or essential “thought”; causal intervention and replication across model, layer, task family, and seed remain necessary.</p></section>
<p class="subtitle">Machine-readable results: <code>analysis/persistent_report.json</code>, <code>analysis/persistent_offset_metrics.csv</code>, and <code>analysis/persistent_visualization.pt</code>.</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(
        document,
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build publication figures and an HTML persistent-SAE report"
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
    proposed_training = load_json(run_dir / "persistent" / "training_report.json")
    baseline_training = load_json(run_dir / "standard" / "training_report.json")
    report = load_json(run_dir / "analysis" / "persistent_report.json")
    training_plot(proposed_training, baseline_training, figures)
    offset_cosine_plot(report, figures)
    support_plot(report, figures)
    discrimination_plot(report, figures)
    probe_plot(report, figures)
    embedding_and_heatmap(run_dir, figures)
    interventions = intervention_plot(run_dir / "analysis", figures)
    write_html(output_dir, report, interventions)
    write_json(
        output_dir / "visualization_summary.json",
        {
            "offset_curve": report["locked_test_offset_curve"],
            "state_probes": report["locked_test_state_probes"],
            "collapse_diagnostics": report["collapse_diagnostics"],
            "interventions": interventions,
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    print(f"wrote report to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
