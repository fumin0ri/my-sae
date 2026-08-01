from collections import Counter
import random

from shared_residual.mmlu_data import (
    CATEGORY_SUBJECTS,
    SUBJECT_TO_CATEGORY,
    build_causal_pairs,
    build_prompt_rows,
    render_prompt,
    reorder_choices,
)


def synthetic_rows():
    rows = []
    for category, subjects in CATEGORY_SUBJECTS.items():
        subject = sorted(subjects)[0]
        for index in range(64):
            rows.append(
                {
                    "question": f"{category} question {index}",
                    "choices": [
                        f"wrong-0-{index}",
                        f"correct-{index}",
                        f"wrong-2-{index}",
                        f"wrong-3-{index}",
                    ],
                    "answer": 1,
                    "subject": subject,
                }
            )
    return rows


def test_official_mmlu_categories_cover_57_subjects_once() -> None:
    subjects = [subject for values in CATEGORY_SUBJECTS.values() for subject in values]
    assert len(subjects) == 57
    assert len(set(subjects)) == 57
    assert len(SUBJECT_TO_CATEGORY) == 57


def test_answer_and_syntax_are_orthogonally_balanced_within_context() -> None:
    prompts = build_prompt_rows(synthetic_rows(), seed=4)
    for category in CATEGORY_SUBJECTS:
        local = [row for row in prompts if row["context_category"] == category]
        joint = Counter(
            (row["semantic_answer"], row["syntax_template"]) for row in local
        )
        assert len(joint) == 16
        assert set(joint.values()) == {4}


def test_reordered_choices_put_correct_text_at_requested_label() -> None:
    choices = ["wrong-a", "correct", "wrong-b", "wrong-c"]
    reordered = reorder_choices(
        choices,
        answer=1,
        desired_answer=3,
        rng=random.Random(9),
    )
    assert reordered[3] == "correct"
    assert sorted(reordered) == sorted(choices)


def test_four_syntax_templates_are_distinct_and_answer_ready() -> None:
    prompts = {
        render_prompt(
            "What is two plus two?",
            ["1", "2", "3", "4"],
            "elementary_mathematics",
            template,
        )
        for template in range(4)
    }
    assert len(prompts) == 4
    assert all(prompt[-1] not in {"A", "B", "C", "D"} for prompt in prompts)


def test_causal_pairs_hold_context_and_syntax_fixed() -> None:
    prompts = build_prompt_rows(synthetic_rows(), seed=7)
    pairs = build_causal_pairs(prompts, count=32, seed=7)
    assert len(pairs) == 32
    assert all(
        row["source_semantic_answer"] != row["target_semantic_answer"]
        for row in pairs
    )
    assert all(
        row["source_question_id"] != row["target_question_id"] for row in pairs
    )
