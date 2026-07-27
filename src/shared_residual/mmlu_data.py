from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import random
from pathlib import Path
from typing import Any, Iterable

from .io import write_jsonl


MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
ANSWER_LABELS = ("A", "B", "C", "D")

# Official broad categories from hendrycks/test/categories.py.
STEM = {
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
}
HUMANITIES = {
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
}
SOCIAL_SCIENCES = {
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
}
OTHER = {
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
}
CATEGORY_SUBJECTS = {
    "STEM": STEM,
    "humanities": HUMANITIES,
    "social_sciences": SOCIAL_SCIENCES,
    "other": OTHER,
}
SUBJECT_TO_CATEGORY = {
    subject: category
    for category, subjects in CATEGORY_SUBJECTS.items()
    for subject in subjects
}


def normalize_answer(value: Any) -> int:
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped in ANSWER_LABELS:
            return ANSWER_LABELS.index(stripped)
        return int(stripped)
    return int(value)


def reorder_choices(
    choices: list[str],
    answer: int,
    desired_answer: int,
    rng: random.Random,
) -> list[str]:
    if len(choices) != 4:
        raise ValueError("MMLU examples must contain exactly four choices")
    if not 0 <= answer < 4 or not 0 <= desired_answer < 4:
        raise ValueError("answer indices must lie in [0, 3]")
    correct = choices[answer]
    distractors = [
        choice for index, choice in enumerate(choices) if index != answer
    ]
    rng.shuffle(distractors)
    result: list[str | None] = [None] * 4
    result[desired_answer] = correct
    iterator = iter(distractors)
    for index in range(4):
        if result[index] is None:
            result[index] = next(iterator)
    return [str(choice) for choice in result]


def render_prompt(
    question: str,
    choices: list[str],
    subject: str,
    syntax_template: int,
) -> str:
    if len(choices) != 4:
        raise ValueError("MMLU prompts require four choices")
    subject_name = subject.replace("_", " ")
    labelled = list(zip(ANSWER_LABELS, choices))
    if syntax_template == 0:
        options = "\n".join(f"{label}. {choice}" for label, choice in labelled)
        return (
            f"The following is a multiple-choice question about {subject_name}.\n"
            f"Question: {question}\n{options}\nAnswer:"
        )
    if syntax_template == 1:
        options = " | ".join(
            f"({label}) {choice}" for label, choice in labelled
        )
        return (
            f"Domain={subject_name}\n{question}\nChoices: {options}\n"
            "Select exactly one option. Response:"
        )
    if syntax_template == 2:
        options = "\n".join(
            f"[OPTION {label}] {choice}" for label, choice in labelled
        )
        return (
            f"[SUBJECT] {subject_name}\n[QUESTION] {question}\n{options}\n"
            "[ANSWER]"
        )
    if syntax_template == 3:
        options = "\n".join(
            f"Candidate {label}: {choice}" for label, choice in labelled
        )
        return (
            f"In the field of {subject_name}, solve this problem:\n{question}\n"
            f"{options}\nThe correct option is"
        )
    raise ValueError("syntax_template must lie in [0, 3]")


def round_robin_subject_sample(
    rows: list[dict[str, Any]],
    maximum: int,
    seed: int,
) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        return list(rows)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
    rng = random.Random(seed)
    for subject_rows in by_subject.values():
        rng.shuffle(subject_rows)
    subjects = sorted(by_subject)
    selected: list[dict[str, Any]] = []
    while len(selected) < maximum:
        progressed = False
        for subject in subjects:
            if by_subject[subject]:
                selected.append(by_subject[subject].pop())
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
    return selected


def build_prompt_rows(
    raw_rows: Iterable[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        subject = str(raw["subject"])
        if subject not in SUBJECT_TO_CATEGORY:
            raise KeyError(f"unknown MMLU subject {subject!r}")
        indexed.append({**raw, "_source_index": index})

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in indexed:
        by_category[SUBJECT_TO_CATEGORY[str(row["subject"])]].append(row)

    prompts: list[dict[str, Any]] = []
    for category_index, category in enumerate(CATEGORY_SUBJECTS):
        category_rows = by_category[category]
        random.Random(seed + 1009 * category_index).shuffle(category_rows)
        for local_index, row in enumerate(category_rows):
            # Within each context category, the 4x4 answer-by-syntax design is
            # balanced in consecutive blocks of sixteen questions.
            syntax_template = local_index % 4
            desired_answer = (local_index // 4) % 4
            original_answer = normalize_answer(row["answer"])
            choices = reorder_choices(
                [str(choice) for choice in row["choices"]],
                original_answer,
                desired_answer,
                random.Random(seed + 104729 * int(row["_source_index"])),
            )
            subject = str(row["subject"])
            question_id = f"mmlu-{int(row['_source_index']):05d}"
            prompts.append(
                {
                    "id": question_id,
                    "question_id": question_id,
                    "subject": subject,
                    "context_category": category,
                    "semantic_answer": ANSWER_LABELS[desired_answer],
                    "syntax_template": f"template_{syntax_template}",
                    "syntax_template_id": syntax_template,
                    "original_answer": ANSWER_LABELS[original_answer],
                    "dataset": MMLU_DATASET,
                    "dataset_revision": MMLU_REVISION,
                    "dataset_split": MMLU_SPLIT,
                    "text": render_prompt(
                        str(row["question"]),
                        choices,
                        subject,
                        syntax_template,
                    ),
                }
            )
    prompts.sort(key=lambda row: str(row["question_id"]))
    return prompts


def build_causal_pairs(
    prompts: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 700_001)
    candidates = list(prompts)
    rng.shuffle(candidates)
    pairs: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    for target in candidates:
        matching = [
            source
            for source in candidates
            if source["question_id"] != target["question_id"]
            and source["context_category"] == target["context_category"]
            and source["syntax_template"] == target["syntax_template"]
            and source["semantic_answer"] != target["semantic_answer"]
        ]
        if not matching:
            continue
        source = rng.choice(matching)
        target_id = str(target["question_id"])
        if target_id in used_targets:
            continue
        used_targets.add(target_id)
        pairs.append(
            {
                "id": f"mmlu-patch-{len(pairs):04d}",
                "source_text": source["text"],
                "target_text": target["text"],
                "answer": f" {target['semantic_answer']}",
                "source_answer": f" {source['semantic_answer']}",
                "source_question_id": source["question_id"],
                "target_question_id": target["question_id"],
                "source_semantic_answer": source["semantic_answer"],
                "target_semantic_answer": target["semantic_answer"],
                "context_category": target["context_category"],
                "syntax_template": target["syntax_template"],
            }
        )
        if len(pairs) == count:
            break
    if len(pairs) != count:
        raise RuntimeError(f"requested {count} causal pairs, built {len(pairs)}")
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build balanced MMLU prompts for JEPA representation evaluation"
    )
    parser.add_argument("--prompts-output", required=True)
    parser.add_argument("--pairs-output", required=True)
    parser.add_argument("--dataset", default=MMLU_DATASET)
    parser.add_argument("--dataset-config", default=MMLU_CONFIG)
    parser.add_argument("--dataset-revision", default=MMLU_REVISION)
    parser.add_argument("--split", default=MMLU_SPLIT)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="0 uses the full 14,042-question MMLU test split",
    )
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_questions and args.max_questions < 160:
        raise ValueError("--max-questions must be 0 or at least 160")
    if args.pairs < 1:
        raise ValueError("--pairs must be positive")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "MMLU generation needs `datasets`; reinstall with "
            "`python -m pip install --upgrade -e .`"
        ) from error

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        revision=args.dataset_revision,
    )
    raw_rows = [dict(row) for row in dataset]
    selected = round_robin_subject_sample(
        raw_rows,
        args.max_questions,
        args.seed,
    )
    prompts = build_prompt_rows(selected, args.seed)
    # Record the actual requested source in case the CLI overrides the default.
    for row in prompts:
        row["dataset"] = args.dataset
        row["dataset_revision"] = args.dataset_revision
        row["dataset_split"] = args.split
    pairs = build_causal_pairs(prompts, args.pairs, args.seed)
    write_jsonl(Path(args.prompts_output), prompts)
    write_jsonl(Path(args.pairs_output), pairs)

    answer_counts = Counter(row["semantic_answer"] for row in prompts)
    context_counts = Counter(row["context_category"] for row in prompts)
    syntax_counts = Counter(row["syntax_template"] for row in prompts)
    print(
        f"wrote {len(prompts):,} MMLU questions and {len(pairs)} causal pairs\n"
        f"semantic={dict(sorted(answer_counts.items()))}\n"
        f"context={dict(sorted(context_counts.items()))}\n"
        f"syntax={dict(sorted(syntax_counts.items()))}"
    )


if __name__ == "__main__":
    main()
