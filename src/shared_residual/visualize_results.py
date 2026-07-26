from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_jsonl, write_json


SERIES = {
    "shared": "#3568c0",
    "mean": "#d97706",
    "last": "#6b7280",
    "null": "#9ca3af",
    "patch": "#7c3aed",
    "ablate": "#dc2626",
    "random": "#6b7280",
}


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    fig.savefig(figures_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def candidate_config_summary(
    candidates: list[dict[str, Any]],
) -> dict[tuple[int, int, int, float], dict[str, float]]:
    grouped: dict[tuple[int, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            int(row["layer"]),
            int(row["window_size"]),
            int(row["rank"]),
            float(row["ridge"]),
        )
        grouped[key].append(row)
    summary: dict[tuple[int, int, int, float], dict[str, float]] = {}
    for key, rows in grouped.items():
        icc = [float(row["validation"]["mean_icc"]) for row in rows]
        null = [
            float(row["validation"]["null_mean_icc"])
            for row in rows
            if row["validation"]["null_mean_icc"] is not None
        ]
        summary[key] = {
            "icc_mean": float(np.mean(icc)),
            "icc_std": float(np.std(icc, ddof=1)) if len(icc) > 1 else 0.0,
            "null_mean": float(np.mean(null)) if null else float("nan"),
            "null_gap": float(np.mean(icc) - np.mean(null))
            if null
            else float(np.mean(icc)),
            "shared_probe": float(
                np.mean(
                    [
                        row["validation"]["probe"]["shared_subspace"].get(
                            "accuracy", np.nan
                        )
                        for row in rows
                    ]
                )
            ),
            "mean_probe": float(
                np.mean(
                    [
                        row["validation"]["probe"][
                            "rank_matched_mean_pca"
                        ].get("accuracy", np.nan)
                        for row in rows
                    ]
                )
            ),
            "last_probe": float(
                np.mean(
                    [
                        row["validation"]["probe"][
                            "rank_matched_last_token_pca"
                        ].get("accuracy", np.nan)
                        for row in rows
                    ]
                )
            ),
        }
    return summary


def plot_layer_window(
    summary: dict[tuple[int, int, int, float], dict[str, float]],
    figures_dir: Path,
) -> list[dict[str, Any]]:
    layers = sorted({key[0] for key in summary})
    widths = sorted({key[1] for key in summary})
    best_rows: list[dict[str, Any]] = []
    matrix = np.full((len(widths), len(layers)), np.nan)
    for width_index, width in enumerate(widths):
        for layer_index, layer in enumerate(layers):
            options = [
                (key, metrics)
                for key, metrics in summary.items()
                if key[0] == layer and key[1] == width
            ]
            if not options:
                continue
            key, metrics = max(options, key=lambda item: item[1]["null_gap"])
            matrix[width_index, layer_index] = metrics["null_gap"]
            best_rows.append(
                {
                    "layer": layer,
                    "window_size": width,
                    "rank": key[2],
                    "ridge": key[3],
                    **metrics,
                }
            )
    fig, ax = plt.subplots(figsize=(max(7, len(layers) * 0.75), 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(layers)), labels=layers)
    ax.set_yticks(range(len(widths)), labels=widths)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Window width (tokens)")
    ax.set_title("Best validation ICC above position-shuffled null")
    for row_index in range(len(widths)):
        for column_index in range(len(layers)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < np.nanmax(matrix) * 0.72 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="ICC − null ICC")
    save_figure(fig, figures_dir, "layer-window-icc")
    return best_rows


def plot_selected_slice(
    summary: dict[tuple[int, int, int, float], dict[str, float]],
    selected: dict[str, Any],
    figures_dir: Path,
) -> None:
    layers = sorted({key[0] for key in summary})
    rows = []
    for layer in layers:
        key = (
            layer,
            int(selected["window_size"]),
            int(selected["rank"]),
            float(selected["ridge"]),
        )
        if key in summary:
            rows.append((layer, summary[key]))
    if not rows:
        return
    x = [row[0] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(
        x,
        [row[1]["icc_mean"] for row in rows],
        marker="o",
        color=SERIES["shared"],
        label="Observed shared ICC",
    )
    axes[0].plot(
        x,
        [row[1]["null_mean"] for row in rows],
        marker="x",
        linestyle="--",
        color=SERIES["null"],
        label="Position-shuffled null",
    )
    axes[0].axvline(
        selected["layer"],
        color=SERIES["patch"],
        linestyle=":",
        label="Selected layer",
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("ICC")
    axes[0].set_title("Shared signal across depth")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        x,
        [row[1]["shared_probe"] for row in rows],
        marker="o",
        color=SERIES["shared"],
        label="Shared subspace",
    )
    axes[1].plot(
        x,
        [row[1]["mean_probe"] for row in rows],
        marker="s",
        color=SERIES["mean"],
        label="Rank-matched mean PCA",
    )
    axes[1].plot(
        x,
        [row[1]["last_probe"] for row in rows],
        marker="^",
        color=SERIES["last"],
        label="Rank-matched last-token PCA",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("State decodability across depth")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        f"Selected slice: window={selected['window_size']}, "
        f"rank={selected['rank']}, ridge={selected['ridge']}"
    )
    save_figure(fig, figures_dir, "depth-profile")


def plot_locked_test(
    locked: dict[str, Any],
    figures_dir: Path,
) -> None:
    test = locked["test"]
    bootstrap = locked["icc_bootstrap"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    observed = float(test["mean_icc"])
    null = float(test["null_mean_icc"])
    lower = max(0.0, observed - float(bootstrap["ci95_low"]))
    upper = max(0.0, float(bootstrap["ci95_high"]) - observed)
    axes[0].bar(
        ["Observed", "Shuffled null"],
        [observed, null],
        color=[SERIES["shared"], SERIES["null"]],
    )
    axes[0].errorbar(
        [0],
        [observed],
        yerr=np.asarray([[lower], [upper]]),
        fmt="none",
        color="black",
        capsize=5,
    )
    axes[0].set_ylabel("Mean ICC")
    axes[0].set_title(
        f"Locked test (permutation p={test['permutation_p_value']:.3g})"
    )
    probes = test["probe"]
    names = ["Shared", "Mean PCA", "Last PCA"]
    values = [
        probes["shared_subspace"].get("accuracy", np.nan),
        probes["rank_matched_mean_pca"].get("accuracy", np.nan),
        probes["rank_matched_last_token_pca"].get("accuracy", np.nan),
    ]
    axes[1].bar(
        names,
        values,
        color=[SERIES["shared"], SERIES["mean"], SERIES["last"]],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Locked-test state decoding")
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    save_figure(fig, figures_dir, "locked-test")


def plot_spectrum(locked: dict[str, Any], figures_dir: Path) -> None:
    values = np.asarray(locked["generalized_eigenvalues"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        np.arange(1, len(values) + 1),
        values,
        marker="o",
        color=SERIES["shared"],
    )
    ax.axhline(0, color=SERIES["null"], linewidth=1)
    ax.set_xlabel("Shared component")
    ax.set_ylabel("Generalized signal/noise eigenvalue")
    ax.set_title("Selected subspace spectrum")
    ax.grid(alpha=0.2)
    save_figure(fig, figures_dir, "shared-spectrum")


def plot_interventions(
    research_dir: Path,
    figures_dir: Path,
) -> dict[str, Any] | None:
    paths = {
        "Contradictory patch": research_dir / "intervention-patch.jsonl",
        "Learned ablation": research_dir / "intervention-ablate.jsonl",
        "Random ablation": research_dir / "intervention-random.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    values = {
        label: [
            float(row["delta_answer_logprob"])
            for row in read_jsonl(path)
        ]
        for label, path in paths.items()
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(
        [values[label] for label in paths],
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(
        parts["bodies"],
        [SERIES["patch"], SERIES["ablate"], SERIES["random"]],
    ):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    parts["cmeans"].set_color("black")
    rng = np.random.default_rng(0)
    for position, label in enumerate(paths, 1):
        jitter = rng.normal(0, 0.035, len(values[label]))
        ax.scatter(
            position + jitter,
            values[label],
            s=12,
            alpha=0.45,
            color=[
                SERIES["patch"],
                SERIES["ablate"],
                SERIES["random"],
            ][position - 1],
        )
    ax.axhline(0, color=SERIES["null"], linewidth=1)
    ax.set_xticks(range(1, len(paths) + 1), labels=list(paths))
    ax.set_ylabel("Δ target-answer log probability")
    ax.set_title("Causal interventions")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figures_dir, "causal-interventions")
    summary = {
        label: {
            "n": len(group),
            "mean": float(np.mean(group)),
            "median": float(np.median(group)),
            "std": float(np.std(group, ddof=1)) if len(group) > 1 else 0.0,
        }
        for label, group in values.items()
    }
    patch_rows = read_jsonl(paths["Contradictory patch"])
    if patch_rows and "delta_contrast_answer_logprob" in patch_rows[0]:
        target_delta = np.asarray(
            [float(row["delta_answer_logprob"]) for row in patch_rows]
        )
        source_delta = np.asarray(
            [float(row["delta_contrast_answer_logprob"]) for row in patch_rows]
        )
        directional = source_delta - target_delta
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for index, (target, source) in enumerate(
            zip(target_delta, source_delta)
        ):
            ax.plot(
                [0, 1],
                [target, source],
                color=SERIES["null"],
                alpha=0.35,
                linewidth=1,
            )
            ax.scatter(0, target, color=SERIES["mean"], s=18)
            ax.scatter(1, source, color=SERIES["patch"], s=18)
        ax.axhline(0, color=SERIES["null"], linewidth=1)
        ax.set_xticks(
            [0, 1],
            labels=["Target-consistent answer", "Source-consistent answer"],
        )
        ax.set_ylabel("Δ answer log probability after patch")
        ax.set_title("Directionality of contradictory state patches")
        ax.grid(axis="y", alpha=0.2)
        save_figure(fig, figures_dir, "patch-directionality")
        summary["patch_directionality"] = {
            "n": len(directional),
            "mean_source_minus_target_delta": float(directional.mean()),
            "median_source_minus_target_delta": float(np.median(directional)),
        }
    learned = np.asarray(values["Learned ablation"])
    random_control = np.asarray(values["Random ablation"])
    paired = learned - random_control
    rng_test = np.random.default_rng(991)
    signs = rng_test.choice(
        np.asarray([-1.0, 1.0]),
        size=(10_000, len(paired)),
    )
    null_means = (signs * paired[None, :]).mean(axis=1)
    observed = float(paired.mean())
    p_value = float(
        (1 + np.sum(np.abs(null_means) >= abs(observed)))
        / (len(null_means) + 1)
    )
    bootstrap_means = np.asarray(
        [
            rng_test.choice(paired, size=len(paired), replace=True).mean()
            for _ in range(5_000)
        ]
    )
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    summary["paired_learned_minus_random"] = {
        "n": len(paired),
        "mean_difference": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "sign_flip_p_value": p_value,
        "cohens_dz": float(
            observed / paired.std(ddof=1)
            if len(paired) > 1 and paired.std(ddof=1) > 0
            else 0.0
        ),
    }
    return summary


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def build_html(
    output_dir: Path,
    selection: dict[str, Any],
    locked: dict[str, Any],
    best_rows: list[dict[str, Any]],
    intervention: dict[str, Any] | None,
) -> None:
    selected = selection["selected"]
    summary = selection["locked_test_summary"]
    figures = [
        ("Layer × window landscape", "figures/layer-window-icc.png"),
        ("Depth profile and baselines", "figures/depth-profile.png"),
        ("Locked test", "figures/locked-test.png"),
        ("Shared spectrum", "figures/shared-spectrum.png"),
    ]
    if intervention is not None:
        figures.append(
            ("Causal interventions", "figures/causal-interventions.png")
        )
        if "patch_directionality" in intervention:
            figures.append(
                ("Patch directionality", "figures/patch-directionality.png")
            )
    figure_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2>'
        f'<img src="{path}" alt="{html.escape(title)}"></section>'
        for title, path in figures
    )
    best_table = "\n".join(
        "<tr>"
        f"<td>{row['layer']}</td>"
        f"<td>{row['window_size']}</td>"
        f"<td>{row['rank']}</td>"
        f"<td>{fmt(row['icc_mean'])}</td>"
        f"<td>{fmt(row['null_mean'])}</td>"
        "</tr>"
        for row in sorted(best_rows, key=lambda row: (row["layer"], row["window_size"]))
    )
    intervention_html = ""
    if intervention is not None:
        intervention_html = "<h2>Intervention summary</h2><table><thead><tr><th>Condition</th><th>n</th><th>Mean Δ log p</th><th>Median</th></tr></thead><tbody>"
        intervention_html += "".join(
            f"<tr><td>{html.escape(label)}</td><td>{stats['n']}</td>"
            f"<td>{fmt(stats['mean'])}</td><td>{fmt(stats['median'])}</td></tr>"
            for label, stats in intervention.items()
            if label
            not in {"paired_learned_minus_random", "patch_directionality"}
        )
        intervention_html += "</tbody></table>"
        paired = intervention["paired_learned_minus_random"]
        intervention_html += (
            "<p>Paired learned-minus-random ablation: "
            f"mean={fmt(paired['mean_difference'])}, "
            f"95% CI={fmt(paired['ci95_low'])}–{fmt(paired['ci95_high'])}, "
            f"sign-flip p={fmt(paired['sign_flip_p_value'])}, "
            f"Cohen's dz={fmt(paired['cohens_dz'])}.</p>"
        )
        if "patch_directionality" in intervention:
            directional = intervention["patch_directionality"]
            intervention_html += (
                "<p>Contradictory patch directionality "
                "(source-answer Δ minus target-answer Δ): "
                f"mean={fmt(directional['mean_source_minus_target_delta'])}, "
                f"median={fmt(directional['median_source_minus_target_delta'])}."
                "</p>"
            )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shared residual-stream research report</title>
<style>
:root {{ color-scheme: light dark; --bg: #f7f8fa; --fg: #172033; --card: #ffffff; --muted: #5f6b7a; --border: #d9dee7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111827; --fg: #e5e7eb; --card: #1f2937; --muted: #aab2c0; --border: #374151; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--fg); }}
main {{ max-width: 1120px; margin: auto; padding: 32px 20px 64px; }}
h1 {{ margin-bottom: 8px; }} h2 {{ margin-top: 36px; }}
.subtitle {{ color: var(--muted); }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 24px 0; }}
.metric {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
.metric span {{ display: block; color: var(--muted); font-size: 0.86rem; }}
.metric strong {{ display: block; font-size: 1.35rem; margin-top: 5px; }}
section img {{ width: 100%; height: auto; background: white; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); }}
th, td {{ padding: 9px 11px; border-bottom: 1px solid var(--border); text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.table-wrap {{ overflow-x: auto; }}
code {{ color: inherit; }}
</style>
</head>
<body><main>
<h1>Shared residual-stream research report</h1>
<p class="subtitle">Nested grouped selection; one untouched outer test; causal controls when available.</p>
<div class="metrics">
<div class="metric"><span>Selected configuration</span><strong>L{selected['layer']} · T{selected['window_size']} · r{selected['rank']}</strong></div>
<div class="metric"><span>Locked-test ICC</span><strong>{fmt(summary['mean_icc'])}</strong></div>
<div class="metric"><span>95% group-bootstrap CI</span><strong>{fmt(summary['icc_ci95']['ci95_low'])}–{fmt(summary['icc_ci95']['ci95_high'])}</strong></div>
<div class="metric"><span>Permutation p</span><strong>{fmt(summary['permutation_p_value'])}</strong></div>
<div class="metric"><span>Shared-code accuracy</span><strong>{fmt(summary['shared_probe_accuracy'])}</strong></div>
<div class="metric"><span>Split-half similarity</span><strong>{fmt(summary['split_half_mean_squared_cosine'])}</strong></div>
</div>
{figure_html}
{intervention_html}
<h2>Best validation configuration per layer and window</h2>
<div class="table-wrap"><table><thead><tr><th>Layer</th><th>Window</th><th>Rank</th><th>ICC</th><th>Null ICC</th></tr></thead><tbody>
{best_table}
</tbody></table></div>
<p class="subtitle">Machine-readable inputs: <code>selection.json</code>, <code>locked_test.json</code>, and <code>candidates.jsonl</code>.</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create publication figures and an HTML research report"
    )
    parser.add_argument("--research-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    research_dir = Path(args.research_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(research_dir / "candidates.jsonl")
    selection = json.loads(
        (research_dir / "selection.json").read_text(encoding="utf-8")
    )
    locked = json.loads(
        (research_dir / "locked_test.json").read_text(encoding="utf-8")
    )
    summary = candidate_config_summary(candidates)
    best_rows = plot_layer_window(summary, figures_dir)
    plot_selected_slice(summary, selection["selected"], figures_dir)
    plot_locked_test(locked, figures_dir)
    plot_spectrum(locked, figures_dir)
    intervention = plot_interventions(research_dir, figures_dir)
    build_html(output_dir, selection, locked, best_rows, intervention)
    write_json(
        output_dir / "visualization_summary.json",
        {
            "selected": selection["selected"],
            "locked_test": selection["locked_test_summary"],
            "interventions": intervention,
            "figures": sorted(path.name for path in figures_dir.glob("*.png")),
        },
    )
    print(f"wrote report to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
