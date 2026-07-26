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
    "posthoc": "#d97706",
    "innovation": "#dc2626",
    "raw": "#64748b",
    "patch": "#7c3aed",
    "random": "#94a3b8",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_plot(
    joint: dict[str, Any],
    baseline: dict[str, Any],
    figures: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for report, label, color in (
        (joint, "JEPA-regularized SAE", COLORS["joint"]),
        (baseline, "Standard SAE + post-hoc predictor", COLORS["posthoc"]),
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
    axes[1].set_title("Masked future-code prediction")
    axes[1].set_ylabel("Validation cosine (higher is better)")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "training-curves")


def probe_plot(report: dict[str, Any], figures: Path) -> None:
    probes = report["locked_test_probes"]
    order = [
        ("Joint predictable", "joint_predictable_code", COLORS["joint"]),
        ("Post-hoc predictable", "posthoc_predictable_code", COLORS["posthoc"]),
        ("Joint target SAE", "joint_target_sae_code", "#0f766e"),
        ("Standard SAE", "standard_sae_code", "#65a30d"),
        ("Innovation", "joint_innovation_residual", COLORS["innovation"]),
        ("Raw residual", "raw_target_residual", COLORS["raw"]),
    ]
    values = [probes[key]["accuracy"] for _, key, _ in order]
    lows = [
        values[index] - probes[key]["group_bootstrap"]["ci95_low"]
        for index, (_, key, _) in enumerate(order)
    ]
    highs = [
        probes[key]["group_bootstrap"]["ci95_high"] - values[index]
        for index, (_, key, _) in enumerate(order)
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(order))
    ax.bar(x, values, color=[color for _, _, color in order])
    ax.errorbar(
        x,
        values,
        yerr=np.asarray([lows, highs]),
        color="black",
        fmt="none",
        capsize=4,
    )
    chance = probes["joint_predictable_code"]["chance_accuracy"]
    ax.axhline(chance, color="#334155", linestyle="--", label="Majority chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Locked-test accuracy")
    ax.set_xticks(x, [label for label, _, _ in order], rotation=20, ha="right")
    ax.set_title("Where is task state information?")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "locked-test-probes")


def gap_plot(report: dict[str, Any], figures: Path) -> None:
    joint = report["gap_curve"]["joint"]
    posthoc = report["gap_curve"]["posthoc"]
    shuffled = report["shuffled_context_null"]["joint"]
    gaps = sorted({int(row["gap"]) for row in joint})
    targets = sorted({int(row["target_size"]) for row in joint})
    matrices = []
    for rows in (joint, shuffled, posthoc):
        lookup = {
            (int(row["target_size"]), int(row["gap"])): float(row["code_cosine"])
            for row in rows
        }
        matrices.append(
            np.asarray(
                [[lookup[(target, gap)] for gap in gaps] for target in targets]
            )
        )
    matrices.append(matrices[0] - matrices[2])
    titles = [
        "JEPA-regularized SAE",
        "Shuffled-context null",
        "Standard SAE + post-hoc",
        "Joint minus post-hoc",
    ]
    cmaps = ["viridis", "viridis", "viridis", "coolwarm"]
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 3.9), constrained_layout=True)
    for axis, matrix, title, cmap in zip(axes, matrices, titles, cmaps):
        limit = max(abs(matrix.min()), abs(matrix.max())) if cmap == "coolwarm" else None
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=-limit if limit is not None else None,
            vmax=limit if limit is not None else None,
        )
        axis.set_xticks(range(len(gaps)), gaps)
        axis.set_yticks(range(len(targets)), targets)
        axis.set_xlabel("Gap (tokens)")
        axis.set_ylabel("Target span (tokens)")
        axis.set_title(title)
        for row in range(len(targets)):
            for column in range(len(gaps)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[row, column]) > 0.45 else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, shrink=0.8, label="Target-code cosine")
    save_figure(fig, figures, "gap-target-generalization")


def invariance_plot(report: dict[str, Any], figures: Path) -> None:
    values = report["paraphrase_and_semantic_invariance"]
    order = [
        ("Joint predictable", "joint_predictable_code", COLORS["joint"]),
        ("Post-hoc predictable", "posthoc_predictable_code", COLORS["posthoc"]),
        ("Innovation", "joint_innovation_residual", COLORS["innovation"]),
        ("Raw residual", "raw_target_residual", COLORS["raw"]),
    ]
    conditions = [
        ("Same problem\nparaphrase", "same_problem_paraphrase"),
        ("Different problem,\nsame state", "different_problem_same_state"),
        ("Different state", "different_state"),
    ]
    x = np.arange(len(conditions))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for index, (label, key, color) in enumerate(order):
        heights = [values[key][condition]["mean"] for _, condition in conditions]
        ax.bar(
            x + (index - 1.5) * width,
            heights,
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(x, [label for label, _ in conditions])
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Paraphrase invariance versus semantic sensitivity")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "invariance-specificity")


def decomposition_plot(report: dict[str, Any], figures: Path) -> None:
    energy = report["predictable_innovation_energy"]
    labels = ["Target centered", "Predictable write", "Innovation"]
    values = [
        energy["target_centered"],
        energy["predictable_write"],
        energy["innovation"],
    ]
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    bars = ax.bar(
        labels,
        values,
        color=[COLORS["raw"], COLORS["joint"], COLORS["innovation"]],
    )
    ax.set_ylabel("Mean squared L2 energy")
    ax.set_title(
        "Predictable / innovation decomposition "
        f"(prediction FVU={energy['prediction_fvu']:.2f})"
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2g}",
            ha="center",
            va="bottom",
        )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "predictable-innovation-energy")


def embedding_and_heatmap(
    run_dir: Path,
    report: dict[str, Any],
    figures: Path,
) -> None:
    bundle = torch_load(run_dir / "analysis" / "predictive_codes.pt")
    codes = bundle["joint"]["predicted_code"].float().numpy()
    test_indices = np.asarray(bundle["split"]["test_indices"], dtype=int)
    test_codes = codes[test_indices]
    metadata = bundle["metadata"]
    label_key = bundle["label_key"]
    labels = np.asarray([str(metadata[index][label_key]) for index in test_indices])
    centered = test_codes - test_codes.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    embedded = centered @ vh[:2].T
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    cmap = plt.get_cmap("tab10")
    for index, label in enumerate(sorted(set(labels))):
        mask = labels == label
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=24,
            alpha=0.65,
            color=cmap(index),
            label=label,
        )
    ax.set_xlabel("Predictable-code PC 1")
    ax.set_ylabel("Predictable-code PC 2")
    ax.set_title("Locked-test predictable sparse states")
    ax.legend(frameon=False, title=label_key)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    save_figure(fig, figures, "predictable-code-embedding")

    feature_ids = [
        int(row["feature_id"])
        for row in report["top_predictable_features"][:20]
    ]
    order = np.argsort(labels)
    heat = test_codes[order][:, feature_ids].T
    scale = np.quantile(heat[heat > 0], 0.98) if np.any(heat > 0) else 1.0
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    image = ax.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=max(scale, 1e-8),
    )
    ax.set_yticks(range(len(feature_ids)), feature_ids)
    ax.set_ylabel("Predictable SAE feature")
    ax.set_xlabel("Locked-test windows (sorted by state)")
    ax.set_title("Most-used predictable features")
    boundaries = np.flatnonzero(labels[order][1:] != labels[order][:-1]) + 0.5
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.8)
    fig.colorbar(image, ax=ax, label="Activation")
    fig.tight_layout()
    save_figure(fig, figures, "predictable-feature-heatmap")


def intervention_plot(
    analysis_dir: Path,
    figures: Path,
) -> dict[str, Any] | None:
    paths = {
        "Contradictory patch": analysis_dir / "intervention-patch.jsonl",
        "Predictable ablation": analysis_dir / "intervention-ablate.jsonl",
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
        [COLORS["patch"], COLORS["innovation"], COLORS["random"]],
    ):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    violin["cmeans"].set_color("black")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_xticks(range(1, 4), list(paths))
    ax.set_ylabel("Change in target-answer log probability")
    ax.set_title("Causal interventions in original residual space")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, "causal-interventions")

    paired = values["Predictable ablation"] - values["Norm-matched random"]
    rng = np.random.default_rng(991)
    bootstrap = np.asarray(
        [rng.choice(paired, size=len(paired), replace=True).mean() for _ in range(5000)]
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    signs = rng.choice([-1.0, 1.0], size=(10000, len(paired)))
    null = (signs * paired[None, :]).mean(axis=1)
    observed = float(paired.mean())
    summary: dict[str, Any] = {
        label: {
            "n": len(group),
            "mean": float(group.mean()),
            "median": float(np.median(group)),
        }
        for label, group in values.items()
    }
    summary["learned_minus_random"] = {
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
    patch = rows["Contradictory patch"]
    if patch and "delta_contrast_answer_logprob" in patch[0]:
        directional = np.asarray(
            [
                float(row["delta_contrast_answer_logprob"])
                - float(row["delta_answer_logprob"])
                for row in patch
            ]
        )
        summary["patch_directionality"] = {
            "mean_source_minus_target_delta": float(directional.mean()),
            "median_source_minus_target_delta": float(np.median(directional)),
        }
    return summary


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def write_html(
    output_dir: Path,
    report: dict[str, Any],
    interventions: dict[str, Any] | None,
) -> None:
    probes = report["locked_test_probes"]
    comparison = report["joint_minus_posthoc_probe_accuracy"]
    diagnostics = report["collapse_and_rank_diagnostics"][
        "joint_predictable_code"
    ]
    figures = [
        ("Training dynamics", "figures/training-curves.png"),
        ("Locked-test probes", "figures/locked-test-probes.png"),
        ("Gap and target-span generalization", "figures/gap-target-generalization.png"),
        ("Paraphrase invariance and specificity", "figures/invariance-specificity.png"),
        ("Predictable / innovation energy", "figures/predictable-innovation-energy.png"),
        ("Predictable-code embedding", "figures/predictable-code-embedding.png"),
        ("Feature activation heatmap", "figures/predictable-feature-heatmap.png"),
    ]
    if interventions is not None:
        figures.append(("Causal interventions", "figures/causal-interventions.png"))
    figure_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2>'
        f'<img src="{path}" alt="{html.escape(title)}"></section>'
        for title, path in figures
    )
    intervention_html = ""
    if interventions is not None:
        learned = interventions["learned_minus_random"]
        intervention_html = (
            "<section><h2>Causal summary</h2>"
            "<p>Predictable-feature ablation minus norm-matched random: "
            f"mean={fmt(learned['mean'])}, 95% CI "
            f"[{fmt(learned['ci95_low'])}, {fmt(learned['ci95_high'])}], "
            f"sign-flip p={fmt(learned['sign_flip_p'])}, "
            f"Cohen's dz={fmt(learned['cohens_dz'])}.</p></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive sparse residual research report</title>
<style>
:root {{ --bg:#f5f7fb; --card:#fff; --fg:#172033; --muted:#64748b; --line:#dbe2ea; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
main {{ max-width:1120px; margin:auto; padding:36px 20px 70px; }}
.subtitle {{ color:var(--muted); max-width:850px; }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:24px 0; }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.metric span {{ display:block; color:var(--muted); font-size:.84rem; }} .metric strong {{ display:block; font-size:1.35rem; margin-top:5px; }}
section {{ margin-top:34px; }} section img {{ width:100%; background:white; border:1px solid var(--line); border-radius:10px; }}
code {{ color:inherit; }}
</style></head><body><main>
<h1>Predictive sparse residual research report</h1>
<p class="subtitle">A feature counts as shared when a masked past context predicts it across a future gap. The decoder remains in the original LLM residual space, permitting causal tests.</p>
<div class="metrics">
<div class="metric"><span>Joint predictable state accuracy</span><strong>{fmt(probes['joint_predictable_code']['accuracy'])}</strong></div>
<div class="metric"><span>Post-hoc predictable accuracy</span><strong>{fmt(probes['posthoc_predictable_code']['accuracy'])}</strong></div>
<div class="metric"><span>Joint minus post-hoc</span><strong>{fmt(comparison['difference'])}</strong></div>
<div class="metric"><span>95% group-bootstrap CI</span><strong>{fmt(comparison['group_bootstrap_ci95_low'])} to {fmt(comparison['group_bootstrap_ci95_high'])}</strong></div>
<div class="metric"><span>Predictable effective rank</span><strong>{fmt(diagnostics['effective_rank'], 1)}</strong></div>
<div class="metric"><span>Dead feature fraction</span><strong>{fmt(diagnostics['dead_dimension_fraction'])}</strong></div>
</div>
{figure_html}
{intervention_html}
<p class="subtitle">Machine-readable results: <code>analysis/predictive_report.json</code> and <code>analysis/predictive_codes.pt</code>.</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(
        document,
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build publication figures and an HTML predictive-SAE report"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / "report"
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    joint_training = load_json(run_dir / "joint" / "training_report.json")
    baseline_training = load_json(run_dir / "posthoc" / "training_report.json")
    report = load_json(run_dir / "analysis" / "predictive_report.json")
    training_plot(joint_training, baseline_training, figures)
    probe_plot(report, figures)
    gap_plot(report, figures)
    invariance_plot(report, figures)
    decomposition_plot(report, figures)
    embedding_and_heatmap(run_dir, report, figures)
    interventions = intervention_plot(run_dir / "analysis", figures)
    write_html(output_dir, report, interventions)
    write_json(
        output_dir / "visualization_summary.json",
        {
            "locked_test": report["locked_test_probes"],
            "joint_minus_posthoc": report[
                "joint_minus_posthoc_probe_accuracy"
            ],
            "interventions": interventions,
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    print(f"wrote report to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
