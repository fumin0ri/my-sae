#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


STATES = ("amber", "cobalt", "jade", "violet")

TEMPLATES = (
    (
        "Finite-state problem {problem_id}. The register begins in state {start}. "
        "{operations} Apply every transition in order. Do not answer prematurely; "
        "retain the computed state while checking the transition sequence. "
        "After all transitions, the final register state is"
    ),
    (
        "Track a four-state controller for case {problem_id}. Its initial mode is "
        "{start}. {operations} Work through the updates in sequence, verify the "
        "result, and keep the final mode active until producing the answer. "
        "The controller's final mode is"
    ),
    (
        "Working-memory evaluation {problem_id}. Start with the value {start} in a "
        "cyclic four-value register. {operations} Silently execute the complete "
        "program, then hold the result fixed while preparing the response. "
        "The resulting value in the register is"
    ),
    (
        "Consider state-machine trace {problem_id}. Before execution the state is "
        "{start}. {operations} Follow the trace carefully without skipping an "
        "update. Check the computed state once and then report it. "
        "The state at the end of the trace is"
    ),
)

OPERATION_FORMS = (
    (
        "Advance the state by {amount} position{suffix} around the cycle.",
        "Move forward by {amount} cyclic step{suffix}.",
        "Apply a rotation of plus {amount} modulo four.",
    ),
    (
        "Update the register by adding {amount} modulo four.",
        "Shift the current mode ahead by {amount} place{suffix}.",
        "Execute a cyclic increment of {amount}.",
    ),
)


def operation_text(shifts: list[int], template_id: int, rng: random.Random) -> str:
    family = OPERATION_FORMS[template_id % len(OPERATION_FORMS)]
    parts = []
    for step, amount in enumerate(shifts, 1):
        form = family[rng.randrange(len(family))]
        rendered = form.format(
            amount=amount,
            suffix="" if amount == 1 else "s",
        )
        parts.append(f"Step {step}: {rendered}")
    return " ".join(parts)


def render_prompt(
    problem_id: int,
    start: int,
    shifts: list[int],
    template_id: int,
    rng: random.Random,
) -> str:
    return TEMPLATES[template_id].format(
        problem_id=problem_id,
        start=STATES[start],
        operations=operation_text(shifts, template_id, rng),
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grouped finite-state prompts and causal patch pairs"
    )
    parser.add_argument("--prompts-output", default="data/research/prompts.jsonl")
    parser.add_argument("--pairs-output", default="data/research/pairs.jsonl")
    parser.add_argument("--problems", type=int, default=128)
    parser.add_argument("--paraphrases", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.problems < 40:
        raise ValueError("use at least 40 independent problem groups")
    if not 1 <= args.paraphrases <= len(TEMPLATES):
        raise ValueError(f"--paraphrases must be in [1, {len(TEMPLATES)}]")

    rng = random.Random(args.seed)
    prompts: list[dict[str, object]] = []
    for problem_index in range(args.problems):
        final_state = problem_index % len(STATES)
        shifts = [rng.randint(1, 3) for _ in range(rng.randint(2, 5))]
        start = (final_state - sum(shifts)) % len(STATES)
        problem_id = 100_000 + problem_index
        group_id = f"fsm-{problem_index:05d}"
        for template_id in range(args.paraphrases):
            prompt = render_prompt(
                problem_id, start, shifts, template_id, random.Random(args.seed + problem_index * 31 + template_id)
            )
            row = {
                "id": f"{group_id}-p{template_id}",
                "group_id": group_id,
                "state": STATES[final_state],
                "initial_state": STATES[start],
                "shifts": shifts,
                "template_id": template_id,
                "text": prompt,
            }
            prompts.append(row)

    pairs: list[dict[str, object]] = []
    for pair_index in range(args.pairs):
        target_state = pair_index % len(STATES)
        source_state = (target_state + 1 + pair_index % 3) % len(STATES)
        shifts = [rng.randint(1, 3) for _ in range(rng.randint(2, 5))]
        target_start = (target_state - sum(shifts)) % len(STATES)
        source_start = (source_state - sum(shifts)) % len(STATES)
        target_text = render_prompt(
            800_000 + pair_index,
            target_start,
            shifts,
            0,
            random.Random(args.seed + 60_000 + pair_index),
        )
        source_text = render_prompt(
            900_000 + pair_index,
            source_start,
            shifts,
            0,
            random.Random(args.seed + 70_000 + pair_index),
        )
        pairs.append(
            {
                "id": f"patch-{pair_index:04d}",
                "source_state": STATES[source_state],
                "target_state": STATES[target_state],
                "source_text": source_text,
                "target_text": target_text,
                "answer": f" {STATES[target_state]}",
                "source_answer": f" {STATES[source_state]}",
            }
        )

    write_jsonl(Path(args.prompts_output), prompts)
    write_jsonl(Path(args.pairs_output), pairs)
    print(
        f"wrote {len(prompts)} prompts in {args.problems} leakage-safe groups "
        f"and {len(pairs)} causal pairs"
    )


if __name__ == "__main__":
    main()
