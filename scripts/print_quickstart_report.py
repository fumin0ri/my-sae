#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested(value: dict[str, Any], *keys: str, default: Any = "n/a") -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default="runs/quickstart/shared/report.json")
    args = parser.parse_args()
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    print("\nQuickstart result")
    print("-----------------")
    print(f"activation shape:       {report['shape']}")
    print(f"kept shared rank:       {report['kept_rank']}")
    print(f"held-out mean ICC:      {nested(report, 'test', 'mean_icc')}")
    print(
        "permutation p-value:    "
        f"{nested(report, 'permutation_control', 'p_value')}"
    )
    print(
        "split-half stability:   "
        f"{nested(report, 'split_half_stability', 'mean_squared_cosine')}"
    )
    print(
        "shared-code probe acc.: "
        f"{nested(report, 'probe', 'shared_subspace', 'accuracy')}"
    )
    print(
        "rank-matched PCA acc.:  "
        f"{nested(report, 'probe', 'rank_matched_mean_pca', 'accuracy')}"
    )
    print(f"\nFull report: {report_path}")
    print(
        "This is a pipeline smoke test. It is not evidence that the extracted "
        "subspace is a model's essential thought representation."
    )


if __name__ == "__main__":
    main()
