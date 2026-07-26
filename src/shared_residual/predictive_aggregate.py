from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import torch_load, write_json


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(report_path: Path) -> dict[str, Any]:
    analysis_dir = report_path.parent
    run_dir = analysis_dir.parent
    report = load_json(report_path)
    checkpoint = torch_load(run_dir / "joint" / "predictive_sae.pt")
    codes = torch_load(analysis_dir / "predictive_codes.pt")
    task_families = sorted(
        {
            str(row.get("task_family", "unknown"))
            for row in codes["metadata"]
        }
    )
    source = checkpoint.get("source_config", {})
    train_args = checkpoint.get("train_args", {})
    probes = report["locked_test_probes"]
    difference = report["joint_minus_posthoc_probe_accuracy"]
    joint_gap = report["gap_curve"]["joint"]
    posthoc_gap = report["gap_curve"]["posthoc"]
    invariance = report["paraphrase_and_semantic_invariance"][
        "joint_predictable_code"
    ]
    row: dict[str, Any] = {
        "run": str(run_dir),
        "task_family": ",".join(task_families),
        "model": source.get("model", "unknown"),
        "model_revision": source.get("resolved_model_revision"),
        "layer": source.get("layer"),
        "hook_point": source.get("hook_point"),
        "seed": train_args.get("seed"),
        "split_seed": train_args.get("split_seed"),
        "d_sae": checkpoint["config"].get("d_sae"),
        "k": checkpoint["config"].get("k"),
        "joint_accuracy": probes["joint_predictable_code"]["accuracy"],
        "posthoc_accuracy": probes["posthoc_predictable_code"]["accuracy"],
        "joint_minus_posthoc_accuracy": difference["difference"],
        "difference_ci95_low": difference["group_bootstrap_ci95_low"],
        "difference_ci95_high": difference["group_bootstrap_ci95_high"],
        "joint_mean_code_cosine": float(
            np.mean([item["code_cosine"] for item in joint_gap])
        ),
        "posthoc_mean_code_cosine": float(
            np.mean([item["code_cosine"] for item in posthoc_gap])
        ),
        "paraphrase_margin": invariance[
            "paraphrase_margin_over_same_state"
        ],
        "semantic_margin": invariance[
            "semantic_margin_same_over_different_state"
        ],
        "effective_rank": report["collapse_and_rank_diagnostics"][
            "joint_predictable_code"
        ]["effective_rank"],
        "dead_feature_fraction": report["collapse_and_rank_diagnostics"][
            "joint_predictable_code"
        ]["dead_dimension_fraction"],
    }
    visualization_path = run_dir / "report" / "visualization_summary.json"
    if visualization_path.exists():
        interventions = load_json(visualization_path).get("interventions")
        if interventions:
            causal = interventions.get("learned_minus_random", {})
            row.update(
                {
                    "causal_learned_minus_random": causal.get("mean"),
                    "causal_ci95_low": causal.get("ci95_low"),
                    "causal_ci95_high": causal.get("ci95_high"),
                    "causal_sign_flip_p": causal.get("sign_flip_p"),
                }
            )
    return row


def save_forest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [
        f"{row['task_family']} | {row['model']} L{row['layer']} seed {row['seed']}"
        for row in rows
    ]
    means = np.asarray([row["joint_minus_posthoc_accuracy"] for row in rows])
    lows = np.asarray([row["difference_ci95_low"] for row in rows])
    highs = np.asarray([row["difference_ci95_high"] for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, max(4.2, 0.42 * len(rows) + 1.5)))
    ax.errorbar(
        means,
        y,
        xerr=np.asarray([means - lows, highs - means]),
        fmt="o",
        color="#2563eb",
        ecolor="#94a3b8",
        capsize=3,
    )
    ax.axvline(0, color="#334155", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Joint minus post-hoc locked-test accuracy")
    ax.set_title("Predictive dictionary benefit across replication units")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "replication-forest.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / "replication-forest.pdf", bbox_inches="tight")
    plt.close(fig)


def write_html(rows: list[dict[str, Any]], output_dir: Path) -> None:
    table = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['task_family']))}</td>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{row['layer']}</td><td>{row['seed']}</td>"
        f"<td>{row['joint_accuracy']:.3f}</td>"
        f"<td>{row['posthoc_accuracy']:.3f}</td>"
        f"<td>{row['joint_minus_posthoc_accuracy']:.3f}</td>"
        f"<td>[{row['difference_ci95_low']:.3f}, {row['difference_ci95_high']:.3f}]</td>"
        f"<td>{row['joint_mean_code_cosine']:.3f}</td>"
        f"<td>{row['semantic_margin']:.3f}</td>"
        "</tr>"
        for row in rows
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive SAE replication study</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1300px;margin:auto;padding:32px;color:#172033}}
img{{max-width:100%}} table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px;border-bottom:1px solid #dbe2ea;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .scroll{{overflow-x:auto}}
</style></head><body><h1>Predictive SAE replication study</h1>
<p>Each row is a prespecified task, model, layer, and feature-learning seed replication unit.</p>
<img src="replication-forest.png" alt="Replication forest plot">
<div class="scroll"><table><thead><tr>
<th>Task</th><th>Model</th><th>Layer</th><th>Seed</th>
<th>Joint acc.</th><th>Post-hoc acc.</th><th>Difference</th><th>95% CI</th>
<th>Code cosine</th><th>Semantic margin</th></tr></thead><tbody>
{table}</tbody></table></div></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate prespecified predictive-SAE replication runs"
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.runs_root)
    reports = sorted(root.glob("**/analysis/predictive_report.json"))
    if not reports:
        raise ValueError(f"no predictive reports found under {root}")
    rows = [summarize_run(path) for path in reports]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "replication_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "replication_summary.json", rows)
    save_forest(rows, output_dir)
    write_html(rows, output_dir)
    print(f"aggregated {len(rows)} runs into {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
