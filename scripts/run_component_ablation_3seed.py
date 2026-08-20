#!/usr/bin/env python3
"""Run reviewer-facing 3-seed QualSynth component ablations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

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
from src.qualsynth.utils.config_loader import ConfigLoader, MethodConfig
from src.qualsynth.utils.reviewer_artifacts import (
    cost_runtime_row,
    summarize_cost_runtime,
    write_cost_runtime_markdown,
)


@dataclass(frozen=True)
class ComponentVariant:
    name: str
    description: str
    prompt_policy: str
    validation_policy: str
    selection_policy: str
    overrides: dict[str, Any]


COMPONENT_VARIANTS: list[ComponentVariant] = [
    ComponentVariant(
        name="qualsynth_component_full",
        description="Full QualSynth: anchor-centric prompting, full validation, multi-objective selection.",
        prompt_policy="anchor",
        validation_policy="full",
        selection_policy="multi_objective",
        overrides={},
    ),
    ComponentVariant(
        name="qualsynth_component_no_anchor_prompt",
        description="Generate from minority-class distribution summaries without anchor rows or local-copy instructions.",
        prompt_policy="no_anchor",
        validation_policy="full",
        selection_policy="multi_objective",
        overrides={"prompt_policy": "no_anchor"},
    ),
    ComponentVariant(
        name="qualsynth_component_no_validation_raw",
        description="Strict raw-generation stress test that bypasses repair, validators, duplicate filters, and optimizer.",
        prompt_policy="anchor",
        validation_policy="raw",
        selection_policy="generation_order",
        overrides={
            "validation_policy": "raw",
            "selection_policy": "generation_order",
            "enable_sota_dedup": False,
            "enable_adaptive_validation": False,
            "enable_statistical_validation": False,
        },
    ),
    ComponentVariant(
        name="qualsynth_component_no_objective",
        description="Full generation and validation, then deterministic generation-order selection without optimizer.",
        prompt_policy="anchor",
        validation_policy="full",
        selection_policy="generation_order",
        overrides={"selection_policy": "generation_order"},
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QualSynth component ablations with 3 seeds.")
    parser.add_argument("--datasets", nargs="+", help="Dataset names. Defaults to all configured datasets.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds. Defaults to first three common seeds.")
    parser.add_argument("--variants", nargs="+", help="Variant names. Defaults to all component variants.")
    parser.add_argument(
        "--skip-variants",
        nargs="+",
        default=None,
        help=(
            "Variant names to exclude from this run (applied AFTER --variants). "
            "Useful for resume runs that drop a component already covered with sufficient evidence "
            "(e.g. --skip-variants qualsynth_component_no_objective)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reviewer_revision" / "ablations" / "component_3seed",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing successful result JSONs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix without running experiments.")
    parser.add_argument("--no-archive", action="store_true", help="Do not archive legacy ablation_full outputs.")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--validation-mode", choices=["standard", "auto", "high_dimensional"], default="auto")
    parser.add_argument("--disable-universal-validation", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
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


def resolve_seeds(loader: ConfigLoader, datasets: list[str], requested: Optional[Iterable[int]]) -> list[int]:
    if requested:
        return list(dict.fromkeys(int(seed) for seed in requested))

    common: Optional[set[int]] = None
    for dataset in datasets:
        seeds = set(loader.load_dataset_config(dataset).seeds)
        common = seeds if common is None else common & seeds
    if not common:
        raise ValueError("Could not resolve a common seed set across selected datasets.")
    return sorted(common)[:3]


def resolve_variants(
    requested: Optional[Iterable[str]],
    skip: Optional[Iterable[str]] = None,
) -> list[ComponentVariant]:
    by_name = {variant.name: variant for variant in COMPONENT_VARIANTS}
    if not requested:
        selected = list(COMPONENT_VARIANTS)
    else:
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"Unknown component variants: {missing}")
        selected = [by_name[name] for name in requested]

    if skip:
        skip_set = set(skip)
        unknown = skip_set - set(by_name)
        if unknown:
            raise ValueError(f"Unknown component variants in --skip-variants: {sorted(unknown)}")
        selected = [variant for variant in selected if variant.name not in skip_set]
        if not selected:
            raise ValueError("All variants were filtered out by --skip-variants.")
    return selected


def build_method_config(base: MethodConfig, variant: ComponentVariant) -> MethodConfig:
    hyperparameters = deepcopy(base.hyperparameters or {})
    hyperparameters.update(
        {
            "prompt_policy": variant.prompt_policy,
            "validation_policy": variant.validation_policy,
            "selection_policy": variant.selection_policy,
        }
    )
    hyperparameters.update(variant.overrides)
    return MethodConfig(
        name=variant.name,
        type=base.type,
        description=variant.description,
        category=f"{base.category}_component_ablation",
        hyperparameters=hyperparameters,
        tuning_grid=deepcopy(base.tuning_grid),
        settings=deepcopy(base.settings),
        expected=deepcopy(base.expected),
        references=deepcopy(base.references),
        notes=f"Component ablation derived from `{base.name}`.",
        components=deepcopy(base.components),
        strategy=base.strategy,
    )


def archive_existing(no_archive: bool) -> Optional[Path]:
    if no_archive:
        return None
    legacy = PROJECT_ROOT / "results" / "reviewer_revision" / "ablation_full"
    if not legacy.exists():
        return None
    archive_root = PROJECT_ROOT / "results" / "reviewer_revision" / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"ablation_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(legacy), str(destination))
    return destination


def load_existing_result(path: Path) -> Optional[ExperimentResult]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return ExperimentResult(**json.load(handle))
    except Exception:
        return None


def result_to_summary_row(result: ExperimentResult, source_file: Path) -> dict[str, Any]:
    payload = asdict(result)
    row = cost_runtime_row(payload, source_file=source_file.relative_to(PROJECT_ROOT))
    avg_performance = result.avg_performance or {}
    avg_fairness = result.avg_fairness or {}
    row.update(
        {
            "experiment_id": result.experiment_id,
            "avg_f1": avg_performance.get("f1"),
            "avg_roc_auc": avg_performance.get("roc_auc"),
            "avg_precision": avg_performance.get("precision"),
            "avg_recall": avg_performance.get("recall"),
            "avg_balanced_accuracy": avg_performance.get("balanced_accuracy"),
            "avg_dpd": avg_fairness.get("demographic_parity_difference"),
            "avg_eod": avg_fairness.get("equal_opportunity_difference"),
            "timestamp": result.timestamp,
        }
    )
    return row


def save_manifest(
    output_dir: Path,
    datasets: list[str],
    seeds: list[int],
    variants: list[ComponentVariant],
    archive_path: Optional[Path],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "datasets": datasets,
        "seeds": seeds,
        "variants": [asdict(variant) for variant in variants],
        "skipped_variants": sorted(set(args.skip_variants)) if args.skip_variants else [],
        "model_name": args.model_name,
        "validation_mode": args.validation_mode,
        "universal_validation_enabled": not args.disable_universal_validation,
        "archive_path": str(archive_path.relative_to(PROJECT_ROOT)) if archive_path else None,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
    }
    manifest_path = output_dir / "component_ablation_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        history = existing.get("history", []) if isinstance(existing, dict) else []
        history.append({k: v for k, v in existing.items() if k != "history"})
        manifest["history"] = history
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def save_summaries(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "component_ablation_summary.csv", index=False)
    summary.to_json(output_dir / "component_ablation_summary.json", orient="records", indent=2)
    summary.to_csv(output_dir / "component_ablation_cost_runtime_per_run.csv", index=False)
    cost_summary = summarize_cost_runtime(summary)
    cost_summary.to_csv(output_dir / "component_ablation_cost_runtime_summary.csv", index=False)
    write_cost_runtime_markdown(
        cost_summary,
        output_dir / "component_ablation_cost_runtime_summary.md",
        title="Component Ablation Cost and Runtime Summary",
    )


def write_analysis_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        for name in [
            "variant_ranking_by_avg_metric.csv",
            "dataset_variant_summary.csv",
            "classifier_variant_summary.csv",
        ]:
            pd.DataFrame().to_csv(analysis_dir / name, index=False)
        return

    successful = df[df["success"]].copy()
    metric_cols = [
        "avg_f1",
        "avg_roc_auc",
        "avg_precision",
        "avg_recall",
        "avg_balanced_accuracy",
    ]
    available_metrics = [col for col in metric_cols if col in successful.columns]
    if successful.empty or not available_metrics:
        ranking = pd.DataFrame()
        dataset_summary = pd.DataFrame()
    else:
        ranking = (
            successful.groupby("method", dropna=False)[available_metrics]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("avg_f1" if "avg_f1" in available_metrics else available_metrics[0], ascending=False)
        )
        dataset_summary = (
            successful.groupby(["dataset", "method"], dropna=False)[available_metrics + ["n_generated"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["dataset", "method"])
        )
    ranking.to_csv(analysis_dir / "variant_ranking_by_avg_metric.csv", index=False)
    dataset_summary.to_csv(analysis_dir / "dataset_variant_summary.csv", index=False)

    classifier_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        result_path = PROJECT_ROOT / str(row.get("source_file", ""))
        if not result_path.exists():
            continue
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for classifier, metrics in (payload.get("performance_metrics") or {}).items():
            classifier_rows.append(
                {
                    "dataset": payload.get("dataset"),
                    "method": payload.get("method"),
                    "seed": payload.get("seed"),
                    "classifier": classifier,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and not str(key).startswith("avg_")
                    },
                }
            )
    classifier_df = pd.DataFrame(classifier_rows)
    if not classifier_df.empty:
        numeric_cols = [
            col
            for col in classifier_df.columns
            if col not in {"dataset", "method", "seed", "classifier"}
        ]
        classifier_df = (
            classifier_df.groupby(["dataset", "method", "classifier"], dropna=False)[numeric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
    classifier_df.to_csv(analysis_dir / "classifier_variant_summary.csv", index=False)


def write_quality_diagnostics(output_dir: Path, datasets: list[str], seeds: list[int], variants: list[ComponentVariant]) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    invalid_root = PROJECT_ROOT / "results" / "reviewer_revision" / "invalid_rows"
    invalid_root.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.analyze_quality_diagnostics import analyze_sample_set, load_train_minority
    except Exception as exc:
        pd.DataFrame([{"status": "diagnostics_import_failed", "error": str(exc)}]).to_csv(
            analysis_dir / "component_quality_diagnostics.csv",
            index=False,
        )
        return

    rows: list[dict[str, Any]] = []
    invalid_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for seed in seeds:
            try:
                train_minority = load_train_minority(dataset, seed)
            except Exception as exc:
                rows.append({"dataset": dataset, "seed": seed, "status": "train_load_failed", "error": str(exc)})
                continue
            for variant in variants:
                for sample_set in ["generated", "validated"]:
                    try:
                        row, invalid = analyze_sample_set(
                            dataset,
                            variant.name,
                            seed,
                            sample_set,
                            output_dir,
                            train_minority,
                        )
                    except Exception as exc:
                        row, invalid = {
                            "dataset": dataset,
                            "method": variant.name,
                            "seed": seed,
                            "sample_set": sample_set,
                            "status": "diagnostic_failed",
                            "error": str(exc),
                        }, pd.DataFrame()
                    rows.append(row)
                    if not invalid.empty:
                        invalid_frames.append(invalid)

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(analysis_dir / "component_quality_diagnostics.csv", index=False)
    invalid_examples = pd.concat(invalid_frames, ignore_index=True) if invalid_frames else pd.DataFrame()

    summary_path = output_dir / "component_ablation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        failures = summary.loc[~summary["success"].astype(bool)].copy() if "success" in summary else pd.DataFrame()
        if not failures.empty:
            failure_examples = failures[
                ["dataset", "method", "seed", "error", "source_file"]
            ].copy()
            failure_examples.insert(0, "invalid_reason", "downstream_failure")
            invalid_examples = pd.concat([invalid_examples, failure_examples], ignore_index=True)

    invalid_examples.to_csv(
        invalid_root / "component_ablation_invalid_row_examples.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()
    if args.list_variants:
        for variant in COMPONENT_VARIANTS:
            print(f"{variant.name}: {variant.description}")
        return 0

    loader = ConfigLoader()
    datasets = resolve_datasets(loader, args.datasets)
    seeds = resolve_seeds(loader, datasets, args.seeds)
    variants = resolve_variants(args.variants, args.skip_variants)
    output_dir = resolve_path(args.output_dir)
    total = len(datasets) * len(seeds) * len(variants)

    print("=" * 88)
    print("QualSynth Component Ablation Runner")
    print("=" * 88)
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {[variant.name for variant in variants]}")
    if args.skip_variants:
        print(f"Skipped variants: {sorted(set(args.skip_variants))}")
    print(f"Total experiments: {total}")
    print(f"Output directory: {output_dir}")
    print("=" * 88)
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_existing(args.no_archive)
    save_manifest(output_dir, datasets, seeds, variants, archive_path, args)

    base_method = loader.load_method_config("qualsynth")
    variant_configs = {variant.name: build_method_config(base_method, variant) for variant in variants}

    runner = ExperimentRunner(
        output_dir=str(output_dir),
        verbose=True,
        enable_universal_validation=not args.disable_universal_validation,
        validation_mode=args.validation_mode,
    )
    original_load = runner.config_loader.load_method_config

    def load_method_config(name: str) -> MethodConfig:
        if name in variant_configs:
            return variant_configs[name]
        return original_load(name)

    runner.config_loader.load_method_config = load_method_config  # type: ignore[assignment]

    rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    experiment_index = 0
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                experiment_index += 1
                result_path = output_dir / dataset / variant.name / f"seed{seed}.json"
                print(f"\n[{experiment_index}/{total}] {dataset} / {variant.name} / seed {seed}")
                if args.resume:
                    existing = load_existing_result(result_path)
                    if existing and existing.success:
                        print("  ↳ skipped (existing successful result)")
                        rows.append(result_to_summary_row(existing, result_path))
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
                rows.append(result_to_summary_row(result, result_path))
                completed += int(result.success)
                failed += int(not result.success)
                save_summaries(output_dir, rows)

    save_summaries(output_dir, rows)
    write_analysis_tables(output_dir, rows)
    write_quality_diagnostics(output_dir, datasets, seeds, variants)

    print("\n" + "=" * 88)
    print("Component Ablation Batch Complete")
    print("=" * 88)
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Summary:   {output_dir / 'component_ablation_summary.csv'}")
    print("=" * 88)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
