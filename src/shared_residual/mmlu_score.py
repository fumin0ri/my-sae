from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .io import read_jsonl, write_json
from .mmlu_data import ANSWER_LABELS
from .modeling import (
    forward_backbone,
    input_device,
    load_hf_model,
)
from .training import configure_accelerator


def accuracy_summary(
    truth: list[str],
    prediction: list[str],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    if not truth or len(truth) != len(prediction) or len(truth) != len(metadata):
        raise ValueError("truth, prediction, and metadata must have equal length")

    def grouped(key: str) -> dict[str, dict[str, float | int]]:
        indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(metadata):
            indices[str(row[key])].append(index)
        return {
            label: {
                "accuracy": sum(
                    prediction[index] == truth[index] for index in members
                )
                / len(members),
                "n": len(members),
            }
            for label, members in sorted(indices.items())
        }

    return {
        "accuracy": sum(left == right for left, right in zip(truth, prediction))
        / len(truth),
        "n": len(truth),
        "chance_accuracy": 0.25,
        "by_context": grouped("context_category"),
        "by_syntax": grouped("syntax_template"),
        "by_subject": grouped("subject"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the frozen base LLM on the balanced MMLU prompts"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("MMLU JSONL is empty")
    model, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device_map,
        args.trust_remote_code,
        args.revision,
        use_safetensors=True if args.use_safetensors else None,
    )
    tokenizer.truncation_side = "left"
    model_device = input_device(model)
    configure_accelerator(model_device)
    answer_token_ids = []
    for label in ANSWER_LABELS:
        ids = tokenizer(
            f" {label}",
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"answer continuation ' {label}' is not one token for this model"
            )
        answer_token_ids.append(ids[0])
    answer_tokens = torch.tensor(answer_token_ids, device=model_device)

    truth: list[str] = []
    prediction: list[str] = []
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc="base LLM MMLU",
    ):
        batch = rows[start : start + args.batch_size]
        encoded = tokenizer(
            [str(row["text"]) for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(model_device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = forward_backbone(model, encoded)
            hidden = outputs.last_hidden_state
            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            final_hidden = hidden[
                torch.arange(len(batch), device=model_device),
                last_positions,
            ]
            logits = model.get_output_embeddings()(final_hidden)
            choice_logits = logits.index_select(1, answer_tokens)
            predicted = choice_logits.argmax(dim=1).cpu().tolist()
        truth.extend(str(row["semantic_answer"]) for row in batch)
        prediction.extend(ANSWER_LABELS[index] for index in predicted)

    report = {
        "benchmark": {
            "dataset": rows[0].get("dataset"),
            "revision": rows[0].get("dataset_revision"),
            "split": rows[0].get("dataset_split"),
            "prompting": "zero-shot single-token A/B/C/D",
            "option_order": "balanced deterministic permutation",
        },
        "model": args.model,
        "requested_model_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        **accuracy_summary(truth, prediction, rows),
    }
    write_json(Path(args.output), report)
    print(
        f"base LLM MMLU accuracy={report['accuracy']:.4f} "
        f"on n={report['n']:,}"
    )


if __name__ == "__main__":
    main()
