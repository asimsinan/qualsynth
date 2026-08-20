#!/usr/bin/env python3
"""Run one normal full-QualSynth generation per dataset for practical cost/runtime evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

os.environ.setdefault("PYTORCH_MPS_METAL", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPS_DISABLE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.experiments.experiment_runner import ExperimentResult, ExperimentRunner
from src.qualsynth.utils.config_loader import ConfigLoader
from src.qualsynth.utils.reviewer_artifacts import (
    cost_runtime_row,
    summarize_cost_runtime,
    write_cost_runtime_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one normal QualSynth generation per dataset.")
    parser.add_argument("--datasets", nargs="+", help="Dataset names. Defaults to all configured datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for every dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reviewer_revision" / "cost_runtime" / "real_usecase",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing successful result JSONs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--validation-mode", choices=["standard", "auto", "high_dimensional"], default="auto")
    parser.add_argument("--disable-universal-validation", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_datasets(loader: ConfigLoader, requested: Optional[Iterable[str]]) -> list[str]:
    datasets = list(requested) if requested else loader.list_datasets()
    available = set(loader.list_datasets())
    missing = [dataset for dataset in datasets if dataset not in available]
    if missing:
        raise ValueError(f"Unknown datasets: {missing}")
    return sorted(datasets)


def load_existing_result(path: Path) -> Optional[ExperimentResult]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return ExperimentResult(**json.load(handle))
    except Exception:
        return None


def result_to_row(result: ExperimentResult, source_file: Path) -> dict[str, Any]:
    row = cost_runtime_row(asdict_like(result), source_file=source_file.relative_to(PROJECT_ROOT))
    metadata = result.metadata or {}
    target = row.get("target_samples") or 0
    validated = row.get("n_validated") or 0
    row.update(
        {
            "target_samples": target,
            "target_shortfall": max(0, int(target) - int(validated)) if target else 0,
            "target_samples_requested": metadata.get("target_samples"),
        }
    )
    return row


def asdict_like(result: ExperimentResult) -> dict[str, Any]:
    return {
        "dataset": result.dataset,
        "method": result.method,
        "seed": result.seed,
        "success": result.success,
        "error": result.error,
        "execution_time": result.execution_time,
        "generation_time": result.generation_time,
        "training_time": result.training_time,
        "evaluation_time": result.evaluation_time,
        "generation_cost": result.generation_cost,
        "llm_calls": result.llm_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "n_generated": result.n_generated,
        "metadata": result.metadata,
    }


def save_manifest(output_dir: Path, datasets: list[str], args: argparse.Namespace) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "datasets": datasets,
        "seed": args.seed,
        "method": "qualsynth",
        "model_name": args.model_name,
        "validation_mode": args.validation_mode,
        "universal_validation_enabled": not args.disable_universal_validation,
        "prompt_policy": "anchor",
        "validation_policy": "full",
        "selection_policy": "multi_objective",
        "sampling": {
            "max_iterations": args.max_iterations,
            "batch_size": args.batch_size,
        },
        "hardware_runtime_metadata": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
        },
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
    }
    with (output_dir / "real_usecase_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def save_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    per_dataset = pd.DataFrame(rows)
    per_dataset.to_csv(output_dir / "real_usecase_cost_runtime_per_dataset.csv", index=False)
    summary = summarize_cost_runtime(per_dataset)
    summary.to_csv(output_dir / "real_usecase_cost_runtime_summary.csv", index=False)
    write_cost_runtime_markdown(
        summary,
        output_dir / "real_usecase_cost_runtime_summary.md",
        title="Real-Usecase QualSynth Cost and Runtime Summary",
    )
    if not per_dataset.empty and "success" in per_dataset:
        failures = per_dataset.loc[~per_dataset["success"].astype(bool)].copy()
    else:
        failures = pd.DataFrame()
    if not failures.empty:
        failures.to_csv(output_dir / "real_usecase_failures.csv", index=False)


def main() -> int:
    args = parse_args()
    loader = ConfigLoader()
    datasets = resolve_datasets(loader, args.datasets)
    output_dir = resolve_path(args.output_dir)

    print("=" * 88)
    print("Real-Usecase QualSynth Cost/Runtime Runner")
    print("=" * 88)
    print(f"Datasets: {datasets}")
    print(f"Seed: {args.seed}")
    print(f"Total runs: {len(datasets)}")
    print(f"Output directory: {output_dir}")
    print("=" * 88)
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir, datasets, args)

    runner = ExperimentRunner(
        output_dir=str(output_dir),
        verbose=True,
        enable_universal_validation=not args.disable_universal_validation,
        validation_mode=args.validation_mode,
    )

    rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    for index, dataset in enumerate(datasets, start=1):
        result_path = output_dir / dataset / "qualsynth" / f"seed{args.seed}.json"
        print(f"\n[{index}/{len(datasets)}] {dataset} / qualsynth / seed {args.seed}")
        if args.resume:
            existing = load_existing_result(result_path)
            if existing and existing.success:
                print("  ↳ skipped (existing successful result)")
                rows.append(result_to_row(existing, result_path))
                skipped += 1
                continue

        result = runner.run_experiment(
            dataset_name=dataset,
            method_name="qualsynth",
            seed=args.seed,
            save_results=True,
            max_iterations_override=args.max_iterations,
            model_name_override=args.model_name,
            batch_size_override=args.batch_size,
        )
        rows.append(result_to_row(result, result_path))
        completed += int(result.success)
        failed += int(not result.success)
        save_outputs(output_dir, rows)

    save_outputs(output_dir, rows)

    print("\n" + "=" * 88)
    print("Real-Usecase Cost/Runtime Batch Complete")
    print("=" * 88)
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Summary:   {output_dir / 'real_usecase_cost_runtime_per_dataset.csv'}")
    print("=" * 88)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
