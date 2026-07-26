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

ARITHMETIC_TEMPLATES = (
    (
        "Modular arithmetic case {problem_id}. Begin with integer {start_index}. "
        "{operations} Carry out the complete calculation modulo four. Use the "
        "fixed output code 0=amber, 1=cobalt, 2=jade, 3=violet, and retain the "
        "computed code while checking the arithmetic. The mapped result state is"
    ),
    (
        "Evaluate residue program {problem_id}. The accumulator initially holds "
        "{start_index}. {operations} Reduce after every update modulo 4, verify "
        "the final residue, then translate it with 0 amber, 1 cobalt, 2 jade, "
        "3 violet. The final coded state is"
    ),
    (
        "Working-memory arithmetic trace {problem_id}. Initialize x to "
        "{start_index}. {operations} Apply every instruction sequentially in "
        "Z/4Z. Do not emit intermediate values. Map the held result through "
        "[amber, cobalt, jade, violet]. The answer state is"
    ),
    (
        "Consider modular computation {problem_id}, starting from "
        "{start_index}. {operations} Check the sum modulo four after the final "
        "instruction. The residue labels in order are amber, cobalt, jade, "
        "violet. Report only the resulting label, which is"
    ),
)

LOGIC_TEMPLATES = (
    (
        "Boolean register problem {problem_id}. Initially P is {p_value} and Q "
        "is {q_value}. {operations} Negation means logical NOT and every update "
        "is applied in order. Encode (false,false)=amber, (true,false)=cobalt, "
        "(false,true)=jade, (true,true)=violet. The final Boolean state is"
    ),
    (
        "Track propositions P and Q for case {problem_id}. Their starting truth "
        "values are P={p_value}, Q={q_value}. {operations} Execute all toggles "
        "without reordering them, hold the final pair, and use the code amber "
        "00, cobalt 10, jade 01, violet 11. The resulting code is"
    ),
    (
        "Symbolic negation trace {problem_id}. Start from P {p_value} and Q "
        "{q_value}. {operations} Treat each instruction as an exclusive state "
        "update, verify the final truth assignment, and map 00/10/01/11 to "
        "amber/cobalt/jade/violet. The retained state label is"
    ),
    (
        "Logical working-memory task {problem_id}. Before execution, P={p_value} "
        "and Q={q_value}. {operations} Follow the complete sequence of NOT "
        "operations. After checking the two final truth values, convert them "
        "using amber 00, cobalt 10, jade 01, violet 11. The final label is"
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


def arithmetic_operations(shifts: list[int], template_id: int) -> str:
    forms = (
        "Instruction {step}: add {amount}.",
        "Update {step}: increase the accumulator by {amount}.",
        "Operation {step}: x becomes x plus {amount}.",
    )
    return " ".join(
        forms[(template_id + step) % len(forms)].format(
            step=step + 1,
            amount=amount,
        )
        for step, amount in enumerate(shifts)
    )


def logic_toggle_mask(shifts: list[int]) -> int:
    mask = 0
    for amount in shifts:
        mask ^= {1: 1, 2: 2, 3: 3}[amount]
    return mask


def logic_operations(shifts: list[int], template_id: int) -> str:
    names = {
        1: ("Negate P.", "Replace P by NOT P.", "Toggle proposition P."),
        2: ("Negate Q.", "Replace Q by NOT Q.", "Toggle proposition Q."),
        3: (
            "Negate both P and Q.",
            "Replace each proposition by its negation.",
            "Toggle P and toggle Q.",
        ),
    }
    return " ".join(
        f"Step {step + 1}: {names[amount][(template_id + step) % 3]}"
        for step, amount in enumerate(shifts)
    )


def render_task_prompt(
    task_family: str,
    problem_id: int,
    start: int,
    shifts: list[int],
    template_id: int,
    rng: random.Random,
) -> str:
    if task_family == "fsm":
        return render_prompt(
            problem_id,
            start,
            shifts,
            template_id,
            rng,
        )
    if task_family == "arithmetic":
        return ARITHMETIC_TEMPLATES[template_id].format(
            problem_id=problem_id,
            start_index=start,
            operations=arithmetic_operations(shifts, template_id),
        )
    if task_family == "logic":
        return LOGIC_TEMPLATES[template_id].format(
            problem_id=problem_id,
            p_value="true" if start & 1 else "false",
            q_value="true" if start & 2 else "false",
            operations=logic_operations(shifts, template_id),
        )
    raise ValueError(f"unknown task family {task_family!r}")


def initial_state_for_final(
    task_family: str,
    final_state: int,
    shifts: list[int],
) -> int:
    if task_family in {"fsm", "arithmetic"}:
        return (final_state - sum(shifts)) % len(STATES)
    if task_family == "logic":
        return final_state ^ logic_toggle_mask(shifts)
    raise ValueError(f"unknown task family {task_family!r}")


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
    parser.add_argument(
        "--task-family",
        choices=["fsm", "arithmetic", "logic"],
        default="fsm",
    )
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
        start = initial_state_for_final(args.task_family, final_state, shifts)
        problem_id = 100_000 + problem_index
        group_id = f"{args.task_family}-{problem_index:05d}"
        for template_id in range(args.paraphrases):
            prompt = render_task_prompt(
                args.task_family,
                problem_id,
                start,
                shifts,
                template_id,
                random.Random(args.seed + problem_index * 31 + template_id),
            )
            row = {
                "id": f"{group_id}-p{template_id}",
                "group_id": group_id,
                "state": STATES[final_state],
                "initial_state": STATES[start],
                "shifts": shifts,
                "template_id": template_id,
                "task_family": args.task_family,
                "contains_explicit_negation": args.task_family == "logic",
                "text": prompt,
            }
            prompts.append(row)

    pairs: list[dict[str, object]] = []
    for pair_index in range(args.pairs):
        target_state = pair_index % len(STATES)
        source_state = (target_state + 1 + pair_index % 3) % len(STATES)
        shifts = [rng.randint(1, 3) for _ in range(rng.randint(2, 5))]
        target_start = initial_state_for_final(
            args.task_family,
            target_state,
            shifts,
        )
        source_start = initial_state_for_final(
            args.task_family,
            source_state,
            shifts,
        )
        target_text = render_task_prompt(
            args.task_family,
            800_000 + pair_index,
            target_start,
            shifts,
            0,
            random.Random(args.seed + 60_000 + pair_index),
        )
        source_text = render_task_prompt(
            args.task_family,
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
                "task_family": args.task_family,
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
