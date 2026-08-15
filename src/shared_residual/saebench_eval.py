from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, cast

import torch

from .io import write_json
from .saebench_adapter import Component, load_saebench_adapter


SUPPORTED_EVALS = ("core", "sparse_probing")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("comma-separated selection cannot be empty")
    return values


def _load_official_core_outputs(
    output_dir: Path, selected_saes: list[tuple[str, Any]]
) -> list[dict[str, Any]]:
    """Load both newly computed and SAEBench-cached Core result files."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sae_release, _ in selected_saes:
        filename = f"{sae_release}_custom_sae_eval_results.json".replace("/", "_")
        path = output_dir / "core" / filename
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "unique_id": f"{sae_release}_custom_sae",
                "sae_set": sae_release,
                "sae_id": "custom_sae",
                "metrics": payload.get("eval_result_metrics", {}),
                "official_result": str(path),
            }
        )
    if missing:
        raise RuntimeError(
            "SAEBench Core did not produce results for every requested SAE: "
            + ", ".join(missing)
            + ". Inspect the preceding SAEBench error log."
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Rectified LpJEPA-SAE with SAEBench 0.6"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--components", default="full")
    parser.add_argument("--evals", default="core")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--sae-batch-size", type=int, default=64)
    parser.add_argument("--core-reconstruction-batches", type=int, default=200)
    parser.add_argument("--core-sparsity-batches", type=int, default=2000)
    parser.add_argument("--core-dataset", default="Skylion007/openwebtext")
    parser.add_argument("--compute-weight-metrics", action="store_true")
    parser.add_argument("--sparse-probe-train-size", type=int, default=4000)
    parser.add_argument("--sparse-probe-test-size", type=int, default=1000)
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-revision")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.context_size,
        args.llm_batch_size,
        args.sae_batch_size,
        args.core_reconstruction_batches,
        args.core_sparsity_batches,
        args.sparse_probe_train_size,
        args.sparse_probe_test_size,
    ) < 1:
        raise ValueError("SAEBench batch, context, and sample counts must be positive")
    components = _parse_csv(args.components)
    if len(components) != len(set(components)):
        raise ValueError("SAE component selection contains duplicates")
    invalid_components = sorted(set(components) - {"full", "high", "low"})
    if invalid_components:
        raise ValueError(f"unsupported SAE components: {invalid_components}")
    evals = _parse_csv(args.evals)
    if len(evals) != len(set(evals)):
        raise ValueError("SAEBench evaluation selection contains duplicates")
    invalid_evals = sorted(set(evals) - set(SUPPORTED_EVALS))
    if invalid_evals:
        raise ValueError(f"unsupported SAEBench evaluations: {invalid_evals}")
    try:
        saebench_version = version("sae-bench")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "SAEBench is not installed. Run: pip install -e '.[saebench]'"
        ) from error
    if saebench_version.split(".")[:2] != ["0", "6"]:
        raise RuntimeError(
            f"This adapter targets sae-bench 0.6.x, found {saebench_version}"
        )

    dtype = getattr(torch, args.dtype)
    selected_saes: list[tuple[str, Any]] = []
    checkpoint_metadata: dict[str, Any] | None = None
    for component_name in components:
        component = cast(Component, component_name)
        adapter, checkpoint = load_saebench_adapter(
            args.checkpoint,
            component=component,
            device=args.device,
            dtype=dtype,
            context_size=args.context_size,
            model_revision=args.model_revision,
        )
        selected_saes.append((f"rectified-lpjepa-{component}", adapter))
        checkpoint_metadata = checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "saebench_version": saebench_version,
        "checkpoint": str(Path(args.checkpoint)),
        "architecture_id": checkpoint_metadata.get("architecture_id")
        if checkpoint_metadata
        else None,
        "components": components,
        "evals": {},
        "settings": vars(args),
        "comparability_note": (
            "Compare checkpoints at matched L0; SAEBench metrics are often "
            "strongly sparsity-dependent. Weight-based O(d_sae^2) metrics are "
            "disabled by default for the 32k dictionary."
        ),
    }

    if "core" in evals:
        from sae_bench.evals.core.main import multiple_evals

        multiple_evals(
            selected_saes=selected_saes,
            n_eval_reconstruction_batches=args.core_reconstruction_batches,
            n_eval_sparsity_variance_batches=args.core_sparsity_batches,
            eval_batch_size_prompts=args.llm_batch_size,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=args.compute_weight_metrics,
            exclude_special_tokens_from_reconstruction=True,
            dataset=args.core_dataset,
            context_size=args.context_size,
            output_folder=str(output_dir / "core"),
            verbose=True,
            dtype=args.dtype,
            device=args.device,
            force_rerun=args.force_rerun,
        )
        summary["evals"]["core"] = _load_official_core_outputs(
            output_dir, selected_saes
        )

    if "sparse_probing" in evals:
        from sae_bench.evals.sparse_probing.eval_config import (
            SparseProbingEvalConfig,
        )
        from sae_bench.evals.sparse_probing.main import run_eval

        probe_config = SparseProbingEvalConfig(
            model_name=selected_saes[0][1].cfg.model_name,
            random_seed=args.seed,
            context_length=args.context_size,
            probe_train_set_size=args.sparse_probe_train_size,
            probe_test_set_size=args.sparse_probe_test_size,
            llm_batch_size=args.llm_batch_size,
            sae_batch_size=args.sae_batch_size,
            llm_dtype=args.dtype,
            lower_vram_usage=True,
        )
        probe_results = run_eval(
            probe_config,
            selected_saes,
            args.device,
            str(output_dir / "sparse_probing"),
            force_rerun=args.force_rerun,
            clean_up_activations=not args.save_activations,
            save_activations=args.save_activations,
            artifacts_path=str(output_dir / "artifacts"),
        )
        if not probe_results:
            raise RuntimeError("SAEBench Sparse Probing produced no results")
        summary["evals"]["sparse_probing"] = _jsonable(probe_results)

    write_json(output_dir / "saebench_summary.json", _jsonable(summary))
    print(f"wrote {output_dir / 'saebench_summary.json'}")


if __name__ == "__main__":
    main()
