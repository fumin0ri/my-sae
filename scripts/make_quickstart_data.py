#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


STATES = ("amber", "cobalt", "jade", "violet")

TEMPLATES = (
    (
        "Memory task {case_id}. A controller writes the value {state} into register R. "
        "No later instruction changes R. Check the syntax, preserve the context, and "
        "prepare a short response. When asked, report the value stored in R. "
        "The value currently held in register R is"
    ),
    (
        "State tracking example {case_id}. The active mode has been set to {state}. "
        "The following operations are read-only and leave that mode unchanged. "
        "Review the request, retain the active mode, and get ready to answer. "
        "The active mode that should be returned is"
    ),
    (
        "Working-memory trial {case_id}. Store {state} as the current state variable. "
        "During the next few steps, do not update the variable. Continue checking, "
        "holding, and preparing the response while keeping the same state. "
        "The unchanged state variable is"
    ),
    (
        "Finite-state record {case_id}. After the last transition, the machine is in "
        "state {state}. All remaining steps are observations rather than transitions. "
        "Keep the result available while validating the record and preparing output. "
        "The machine state to report is"
    ),
)


def build_rows(n: int, seed: int) -> list[dict[str, object]]:
    if n < len(STATES) * 3:
        raise ValueError(f"--n must be at least {len(STATES) * 3}")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    balanced_states = [STATES[i % len(STATES)] for i in range(n)]
    rng.shuffle(balanced_states)
    for i, state in enumerate(balanced_states):
        template_id = rng.randrange(len(TEMPLATES))
        case_id = 10_000 + rng.randrange(90_000)
        rows.append(
            {
                "id": f"quickstart-{i:04d}",
                "state": state,
                "template_id": template_id,
                "text": TEMPLATES[template_id].format(
                    case_id=case_id,
                    state=state,
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a balanced smoke-test dataset for shared residual analysis"
    )
    parser.add_argument("--output", default="data/quickstart.jsonl")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(args.n, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} balanced prompts to {output}")


if __name__ == "__main__":
    main()
