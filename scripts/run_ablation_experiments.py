#!/usr/bin/env python3
"""
Run full QualSynth ablation experiments.

This script executes reviewer-facing ablations for:
- anchor design
- dynamic few-shot cycling
- few-shot prompting
- validation thresholds
- multi-objective scoring weights

Outputs are written under `results/ablations/` by default, with one result JSON
per dataset / variant / seed plus a machine-readable manifest and summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# Keep heavyweight generators on CPU for stability.
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
from src.qualsynth.utils.config_loader import ConfigLoader, MethodConfig


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    overrides: Dict[str, Any]


ABLATION_VARIANTS: List[AblationVariant] = [
    AblationVariant(
        name="qualsynth_ablation_full",
        description="Reference QualSynth configuration from configs/methods/qualsynth.yaml.",
        overrides={},
    ),
    AblationVariant(
        name="qualsynth_ablation_anchor_random",
        description="Replace stratified anchors with random anchor selection.",
        overrides={"anchor_selection_strategy": "random"},
    ),
    AblationVariant(
        name="qualsynth_ablation_anchor_typical",
        description="Use typical anchors instead of stratified anchors.",
        overrides={"anchor_selection_strategy": "typical"},
    ),
    AblationVariant(
        name="qualsynth_ablation_anchor_kmeans_diverse",
        description="Use k-means diverse anchors instead of stratified anchors.",
        overrides={"anchor_selection_strategy": "kmeans_diverse"},
    ),
    AblationVariant(
        name="qualsynth_ablation_static_fewshot",
        description="Keep few-shot examples enabled but disable iteration-to-iteration cycling.",
        overrides={"dynamic_few_shot": False, "few_shot_selection_strategy": "mixed"},
    ),
    AblationVariant(
        name="qualsynth_ablation_no_fewshot",
        description="Disable few-shot prompting entirely.",
        overrides={"use_few_shot": False, "n_few_shot_examples": 0},
    ),
    AblationVariant(
        name="qualsynth_ablation_validation_strict",
        description="Tighten statistical validation thresholds.",
        overrides={
            "adaptive_std_threshold": 3.0,
            "adaptive_percentile_threshold": 0.99,
        },
    ),
    AblationVariant(
        name="qualsynth_ablation_validation_relaxed",
        description="Relax statistical validation thresholds.",
        overrides={
            "adaptive_std_threshold": 4.5,
            "adaptive_percentile_threshold": 0.999,
        },
    ),
    AblationVariant(
        name="qualsynth_ablation_no_diversity_objective",
        description="Remove the diversity term from multi-objective selection.",
        overrides={
            "diversity_weight": 0.0,
            "performance_weight": 1.0,
        },
    ),
    AblationVariant(
        name="qualsynth_ablation_no_fairness_objective",
        description="Legacy alias for the active quality/diversity-only objective mix.",
        overrides={
            "fairness_weight": 0.0,
            "diversity_weight": 0.50,
            "performance_weight": 0.50,
        },
    ),
    AblationVariant(
        name="qualsynth_ablation_quality_focused",
        description="Shift optimizer weights toward quality/performance.",
        overrides={
            "diversity_weight": 0.15,
            "performance_weight": 0.85,
        },
    ),
    AblationVariant(
        name="qualsynth_ablation_diversity_focused",
        description="Shift optimizer weights toward diversity.",
        overrides={
            "diversity_weight": 0.85,
            "performance_weight": 0.15,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full QualSynth ablation matrix.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to run. Defaults to all configured datasets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seeds to run. Defaults to the common seed set across selected datasets.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Variant names to run. Defaults to all ablation variants.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "ablations" / "full"),
        help="Directory for ablation outputs.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional max-iterations override for all variants.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override for all variants.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model override for all variants.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments with an existing successful result JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned matrix without running it.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print available ablation variants and exit.",
    )
    parser.add_argument(
        "--disable-universal-validation",
        action="store_true",
        help="Disable universal validation for baseline methods (not typically needed here).",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["standard", "auto", "high_dimensional"],
        default="auto",
        help="Universal validation mode to use for ablations (default: auto).",
    )
    return parser.parse_args()


def get_variant_map() -> Dict[str, AblationVariant]:
    return {variant.name: variant for variant in ABLATION_VARIANTS}


def resolve_datasets(loader: ConfigLoader, requested: Optional[Iterable[str]]) -> List[str]:
    datasets = list(requested) if requested else loader.list_datasets()
    available = set(loader.list_datasets())
    missing = [dataset for dataset in datasets if dataset not in available]
    if missing:
        raise ValueError(f"Unknown datasets: {missing}")
    return datasets


def resolve_seeds(loader: ConfigLoader, datasets: List[str], requested: Optional[Iterable[int]]) -> List[int]:
    if requested:
        return list(dict.fromkeys(int(seed) for seed in requested))

    common_seeds: Optional[set[int]] = None
    for dataset in datasets:
        dataset_seeds = set(loader.load_dataset_config(dataset).seeds)
        common_seeds = dataset_seeds if common_seeds is None else common_seeds & dataset_seeds

    if not common_seeds:
        raise ValueError("Could not resolve a common seed set across the selected datasets.")

    return sorted(common_seeds)


def resolve_variants(requested: Optional[Iterable[str]]) -> List[AblationVariant]:
    variant_map = get_variant_map()
    if requested:
        unknown = [name for name in requested if name not in variant_map]
        if unknown:
            raise ValueError(f"Unknown ablation variants: {unknown}")
        return [variant_map[name] for name in requested]
    return ABLATION_VARIANTS


def build_method_config(base_config: MethodConfig, variant: AblationVariant) -> MethodConfig:
    hyperparameters = deepcopy(base_config.hyperparameters or {})
    hyperparameters.update(variant.overrides)
    return MethodConfig(
        name=variant.name,
        type=base_config.type,
        description=variant.description,
        category=f"{base_config.category}_ablation",
        hyperparameters=hyperparameters,
        tuning_grid=deepcopy(base_config.tuning_grid),
        settings=deepcopy(base_config.settings),
        expected=deepcopy(base_config.expected),
        references=deepcopy(base_config.references),
        notes=f"Ablation variant derived from `{base_config.name}`.",
        components=deepcopy(base_config.components),
        strategy=base_config.strategy,
    )


def load_existing_result(result_path: Path) -> Optional[ExperimentResult]:
    if not result_path.exists():
        return None
    try:
        with result_path.open("r") as handle:
            data = json.load(handle)
        return ExperimentResult(**data)
    except Exception:
        return None


def result_to_row(result: ExperimentResult) -> Dict[str, Any]:
    avg_performance = result.avg_performance or {}
    avg_fairness = result.avg_fairness or {}
    metadata = result.metadata or {}
    return {
        "experiment_id": result.experiment_id,
        "dataset": result.dataset,
        "method": result.method,
        "seed": result.seed,
        "success": result.success,
        "error": result.error,
        "execution_time": result.execution_time,
        "n_generated": result.n_generated,
        "generation_time": result.generation_time,
        "training_time": result.training_time,
        "evaluation_time": result.evaluation_time,
        "generation_cost": result.generation_cost,
        "llm_calls": result.llm_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "avg_f1": avg_performance.get("f1"),
        "avg_roc_auc": avg_performance.get("roc_auc"),
        "avg_precision": avg_performance.get("precision"),
        "avg_recall": avg_performance.get("recall"),
        "avg_balanced_accuracy": avg_performance.get("balanced_accuracy"),
        "avg_dpd": avg_fairness.get("demographic_parity_difference"),
        "avg_eod": avg_fairness.get("equal_opportunity_difference"),
        "iterations": metadata.get("iterations"),
        "n_generated_raw": metadata.get("n_generated_raw"),
        "n_validated": metadata.get("n_validated"),
        "target_samples": metadata.get("target_samples"),
        "target_reached": metadata.get("target_reached"),
        "target_shortfall": metadata.get("target_shortfall"),
        "validation_rate": metadata.get("validation_rate"),
        "validation_mode": metadata.get("validation_metrics", {})
        .get("quality_stats", {})
        .get("validation_mode"),
        "converged": metadata.get("converged"),
        "convergence_reason": metadata.get("convergence_reason"),
        "timestamp": result.timestamp,
    }


def save_manifest(output_dir: Path, datasets: List[str], seeds: List[int], variants: List[AblationVariant]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "datasets": datasets,
        "seeds": seeds,
        "variants": [asdict(variant) for variant in variants],
    }
    with (output_dir / "ablation_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def save_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "ablation_summary.csv"
    summary_json = output_dir / "ablation_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", indent=2)


def main() -> int:
    args = parse_args()

    if args.list_variants:
        for variant in ABLATION_VARIANTS:
            print(f"{variant.name}: {variant.description}")
        return 0

    loader = ConfigLoader()
    datasets = resolve_datasets(loader, args.datasets)
    seeds = resolve_seeds(loader, datasets, args.seeds)
    variants = resolve_variants(args.variants)

    total_experiments = len(datasets) * len(seeds) * len(variants)
    print("=" * 88)
    print("QualSynth Ablation Runner")
    print("=" * 88)
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {[variant.name for variant in variants]}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {args.output_dir}")
    print(f"Universal validation: {not args.disable_universal_validation}")
    print(f"Validation mode: {args.validation_mode}")
    print("=" * 88)

    if args.dry_run:
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir, datasets, seeds, variants)

    base_method_config = loader.load_method_config("qualsynth")
    variant_method_configs = {
        variant.name: build_method_config(base_method_config, variant)
        for variant in variants
    }

    runner = ExperimentRunner(
        output_dir=str(output_dir),
        verbose=True,
        enable_universal_validation=not args.disable_universal_validation,
        validation_mode=args.validation_mode,
    )
    original_load_method_config = runner.config_loader.load_method_config

    def custom_load_method_config(method_name: str) -> MethodConfig:
        if method_name in variant_method_configs:
            return variant_method_configs[method_name]
        return original_load_method_config(method_name)

    runner.config_loader.load_method_config = custom_load_method_config  # type: ignore[assignment]

    rows: List[Dict[str, Any]] = []
    completed = 0
    skipped = 0
    failed = 0
    experiment_index = 0

    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                experiment_index += 1
                experiment_id = f"{dataset}_{variant.name}_seed{seed}"
                print(f"\n[{experiment_index}/{total_experiments}] {experiment_id}")

                result_path = output_dir / dataset / variant.name / f"seed{seed}.json"
                if args.resume:
                    existing_result = load_existing_result(result_path)
                    if existing_result and existing_result.success:
                        print("  ↳ skipped (existing successful result)")
                        rows.append(result_to_row(existing_result))
                        skipped += 1
                        continue

                result = runner.run_experiment(
                    dataset_name=dataset,
                    method_name=variant.name,
                    seed=seed,
                    save_results=True,
                    max_iterations_override=args.max_iterations,
                    model_name_override=args.model_name,
                    batch_size_override=args.batch_size,
                )
                rows.append(result_to_row(result))
                if result.success:
                    completed += 1
                else:
                    failed += 1

                save_summary(output_dir, rows)

    save_summary(output_dir, rows)

    print("\n" + "=" * 88)
    print("Ablation Batch Complete")
    print("=" * 88)
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Summary:   {output_dir / 'ablation_summary.csv'}")
    print("=" * 88)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
