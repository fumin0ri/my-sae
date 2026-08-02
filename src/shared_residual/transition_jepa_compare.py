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

from .io import write_json


CONDITIONS = (
    ("softplus", "Softplus", "#2563eb"),
    ("relu_topk", "ReLU + Top-K", "#d97706"),
    ("relu_topk_auxk", "ReLU + Top-K + AuxK", "#059669"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "evaluation": load_json(
            run_dir / "analysis" / "transition_jepa_report.json"
        ),
        "training": load_json(run_dir / "model" / "training_report.json"),
    }


def validate_comparison(runs: dict[str, dict[str, Any]]) -> None:
    fingerprints = {
        run["evaluation"]["checkpoint"]["data_fingerprint"]
        for run in runs.values()
    }
    if len(fingerprints) != 1:
        raise ValueError("comparison runs do not share one activation fingerprint")
    splits = {
        json.dumps(run["evaluation"]["split"], sort_keys=True)
        for run in runs.values()
    }
    if len(splits) != 1:
        raise ValueError("comparison runs do not share one locked evaluation split")
    expected = {
        "softplus": ("softplus", False),
        "relu_topk": ("relu_topk", False),
        "relu_topk_auxk": ("relu_topk", True),
    }
    reference: dict[str, Any] | None = None
    for key, run in runs.items():
        evaluation = run["evaluation"]
        training = run["training"]
        config = dict(evaluation["checkpoint"]["config"])
        output, auxk_enabled = expected[key]
        if config.get("predictor_output", "softplus") != output:
            raise ValueError(f"{key} has the wrong predictor output")
        auxk = training["architecture"].get("predictor_auxk", {})
        if bool(auxk.get("enabled", False)) != auxk_enabled:
            raise ValueError(f"{key} has the wrong AuxK condition")
        if training["horizon_balancing"]["mode"] != "inverse_probability":
            raise ValueError(f"{key} is not inverse-probability horizon weighted")
        config.pop("predictor_output", None)
        if reference is None:
            reference = config
        elif config != reference:
            raise ValueError(
                f"{key} changes model hyperparameters beyond predictor output"
            )


def condition_summary(run: dict[str, Any]) -> dict[str, Any]:
    evaluation = run["evaluation"]
    training = run["training"]
    config = evaluation["checkpoint"]["config"]
    diagnostics = evaluation["representation_diagnostics"]
    predicted = diagnostics["predicted_endpoint_high_online"]
    target = diagnostics["endpoint_high_ema"]
    probes = evaluation["mmlu_probe_accuracy"]
    d_high = round(config["d_sae"] * config["high_fraction"])
    final_validation = training["history"][-1]["validation"]
    return {
        "run_dir": run["run_dir"],
        "predictor_output": config.get("predictor_output", "softplus"),
        "predictor_auxk": training["architecture"].get("predictor_auxk"),
        "ema_sae_fve": evaluation["standard_sae_quality"]["ema"][
            "fraction_variance_explained"
        ],
        "ema_fraction_loss_recovered": (
            evaluation.get("loss_recovered") or {}
        ).get("fraction_loss_recovered_ema"),
        "predicted_active_dimensions": round(
            predicted["active_dimension_fraction"] * d_high
        ),
        "predicted_active_dimension_fraction": predicted[
            "active_dimension_fraction"
        ],
        "predicted_variance_participation_dimension": predicted[
            "variance_participation_dimension"
        ],
        "target_active_dimensions": round(
            target["active_dimension_fraction"] * d_high
        ),
        "target_variance_participation_dimension": target[
            "variance_participation_dimension"
        ],
        "predicted_context_probe_accuracy": probes["context"][
            "predicted_endpoint_high_online"
        ]["accuracy"],
        "predicted_syntax_probe_accuracy": probes["syntax"][
            "predicted_endpoint_high_online"
        ]["accuracy"],
        "predicted_semantics_probe_accuracy": probes["semantics"][
            "predicted_endpoint_high_online"
        ]["accuracy"],
        "final_validation_predictor_metrics": {
            key: value
            for key, value in final_validation.items()
            if key.startswith("predictor_")
            or key in {"code_cosine", "code_nrmse", "support_jaccard"}
        },
        "horizon_curve": evaluation["forecast_validity"]["horizon_curve"],
    }


def horizon_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label, _ in CONDITIONS:
        for point in summaries[key]["horizon_curve"]:
            rows.append(
                {
                    "condition": key,
                    "label": label,
                    "horizon": point["horizon"],
                    "online_code_cosine": point["online_code_cosine"],
                    "online_shuffled_context_cosine": point[
                        "online_shuffled_context_cosine"
                    ],
                    "horizon_only_cosine": point["horizon_only_cosine"],
                    "gain_over_shuffled": point["online_gain_over_shuffled"][
                        "mean"
                    ],
                    "gain_over_horizon_only": point[
                        "online_gain_over_horizon_only"
                    ]["mean"],
                    "support_jaccard": point["online_support_jaccard"],
                    "residual_prediction_fve": 1.0
                    - point["online_residual_prediction_fvu"],
                }
            )
    return rows


def plot_comparison(
    summaries: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.2))
    for key, label, color in CONDITIONS:
        curve = sorted(summaries[key]["horizon_curve"], key=lambda row: row["horizon"])
        horizons = [row["horizon"] for row in curve]
        axes[0, 0].plot(
            horizons,
            [row["online_code_cosine"] for row in curve],
            marker="o",
            label=label,
            color=color,
        )
        axes[0, 1].plot(
            horizons,
            [row["online_gain_over_shuffled"]["mean"] for row in curve],
            marker="o",
            label=label,
            color=color,
        )
        axes[0, 2].plot(
            horizons,
            [row["online_gain_over_horizon_only"]["mean"] for row in curve],
            marker="o",
            label=label,
            color=color,
        )
    axes[0, 0].set_title("Endpoint-code forecast")
    axes[0, 0].set_ylabel("Cosine")
    axes[0, 1].set_title("Problem-specific context gain")
    axes[0, 1].set_ylabel("Correct - shuffled cosine")
    axes[0, 2].set_title("Gain over learned horizon prior")
    axes[0, 2].set_ylabel("Correct - horizon-only cosine")
    axes[0, 1].axhline(0, color="#64748b", linewidth=1)
    axes[0, 2].axhline(0, color="#64748b", linewidth=1)

    labels = [label for _, label, _ in CONDITIONS]
    colors = [color for _, _, color in CONDITIONS]
    positions = np.arange(len(CONDITIONS))
    active_dimensions = [
        summaries[key]["predicted_active_dimensions"] for key, _, _ in CONDITIONS
    ]
    participation = [
        summaries[key]["predicted_variance_participation_dimension"]
        for key, _, _ in CONDITIONS
    ]
    sae_fve = [summaries[key]["ema_sae_fve"] for key, _, _ in CONDITIONS]
    recovered = [
        summaries[key]["ema_fraction_loss_recovered"] for key, _, _ in CONDITIONS
    ]
    axes[1, 0].bar(positions, active_dimensions, color=colors)
    axes[1, 0].set_title("Distinct predicted features on MMLU")
    axes[1, 0].set_ylabel("Active high dimensions")
    axes[1, 1].bar(positions, participation, color=colors)
    axes[1, 1].set_title("Predicted-code effective dimension")
    axes[1, 1].set_ylabel("Variance participation dimension")
    width = 0.36
    axes[1, 2].bar(
        positions - width / 2, sae_fve, width, label="EMA SAE FVE", color="#6366f1"
    )
    if all(value is not None for value in recovered):
        axes[1, 2].bar(
            positions + width / 2,
            recovered,
            width,
            label="EMA loss recovered",
            color="#14b8a6",
        )
    axes[1, 2].set_title("SAE fidelity controls")
    axes[1, 2].set_ylabel("Fraction")
    axes[1, 2].legend(frameon=False)

    for axis in axes[0]:
        axis.set_xlabel("Horizon")
        axis.grid(alpha=0.2)
    for axis in axes[1]:
        axis.set_xticks(positions, labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "predictor_comparison.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / "predictor_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def write_html(output_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for key, label, _ in CONDITIONS:
        summary = summaries[key]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{summary['ema_sae_fve']:.4f}</td>"
            f"<td>{summary['predicted_active_dimensions']}</td>"
            f"<td>{summary['predicted_variance_participation_dimension']:.2f}</td>"
            f"<td>{summary['predicted_context_probe_accuracy']:.4f}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Predictor AuxK comparison</title>
<style>body{{font-family:system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#0f172a}}
img{{width:100%;height:auto}}table{{border-collapse:collapse}}th,td{{padding:8px 12px;border:1px solid #cbd5e1;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
</head><body><h1>Predictor output and AuxK comparison</h1>
<p>All conditions use the same residual manifest, locked MMLU split, seed, and inverse-probability horizon weighting.</p>
<table><thead><tr><th>Condition</th><th>EMA SAE FVE</th><th>Predicted active dims</th><th>Predicted VPD</th><th>Context probe</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Forecast and collapse controls</h2><img src="predictor_comparison.png" alt="Predictor comparison plots">
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the three predictor conditions")
    parser.add_argument("--softplus-run", required=True)
    parser.add_argument("--relu-topk-run", required=True)
    parser.add_argument("--relu-topk-auxk-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = {
        "softplus": load_run(Path(args.softplus_run)),
        "relu_topk": load_run(Path(args.relu_topk_run)),
        "relu_topk_auxk": load_run(Path(args.relu_topk_auxk_run)),
    }
    validate_comparison(runs)
    summaries = {key: condition_summary(run) for key, run in runs.items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "protocol": {
                "conditions": [key for key, _, _ in CONDITIONS],
                "locked_data_fingerprint": next(
                    iter(
                        run["evaluation"]["checkpoint"]["data_fingerprint"]
                        for run in runs.values()
                    )
                ),
                "horizon_weighting": "inverse_probability",
            },
            "conditions": summaries,
        },
    )
    rows = horizon_rows(summaries)
    with (output_dir / "horizon_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(summaries, output_dir)
    write_html(output_dir, summaries)
    print(f"wrote comparison to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
