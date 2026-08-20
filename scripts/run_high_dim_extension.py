#!/usr/bin/env python3
"""Run reviewer-revision high-dimensional extension experiments.

Adds two evidence pieces on top of the existing high-dimensional benchmark
(`results/reviewer_revision/high_dimensional_benchmark`) to address Reviewer 1's
"high-dimensional datasets, such as gene expression data, where the number of
features reaches into the thousands. This would test whether the anchor-centric
prompting scales beyond low-dimensional settings":

1. **K-sweep on Alon** — same source data as the canonical `alon_colon` benchmark,
   prepared at k ∈ {50, 200, 500} via `scripts/prepare_high_dim_extension.py`. The
   k=50 splits are reused from the existing benchmark; k=200 / k=500 are new
   dataset names so the runner gets independent splits per k.
2. **Golub leukemia (k=50)** — a second canonical microarray benchmark
   (Golub et al. 1999), prepared with the same fold-safe pipeline as Alon.

This script is a *sibling*, not a replacement, of `run_component_ablation_3seed.py`.
The component ablation evaluates *which QualSynth pieces matter* for quality on
small/medium tabular data; this script evaluates *whether QualSynth still works*
on high-dimensional gene-expression data (and as feature count grows). They share
the same 3-seed paired structure so paired Wilcoxon analysis is valid here too.
"""

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
from typing import Any, Iterable, Optional

import pandas as pd

# Match the threading guards used elsewhere in reviewer-revision runs so timing
# numbers are comparable across scripts.
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


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HighDimDataset:
    """A logical (dataset_name, k, narrative_group) tuple for the runner.

    `narrative_group` lets the analysis script bucket results into the two
    reviewer narratives: the k-sweep on Alon and the cross-dataset (Alon vs
    Golub at k=50) comparison. Both are read off the same per-seed JSONs.
    """

    name: str
    k_selected: int
    narrative_group: str  # "alon_k_sweep" or "cross_dataset_k50"
    description: str


HIGH_DIM_DATASETS: list[HighDimDataset] = [
    HighDimDataset(
        name="alon_colon",
        k_selected=50,
        narrative_group="alon_k_sweep",
        description="Alon colon microarray, k=50 (reuses existing splits).",
    ),
    HighDimDataset(
        name="alon_colon_k200",
        k_selected=200,
        narrative_group="alon_k_sweep",
        description="Alon colon microarray, k=200 (new splits).",
    ),
    HighDimDataset(
        name="alon_colon_k500",
        k_selected=500,
        narrative_group="alon_k_sweep",
        description="Alon colon microarray, k=500 (new splits, prompt stress test).",
    ),
    HighDimDataset(
        name="golub_leukemia",
        k_selected=50,
        narrative_group="cross_dataset_k50",
        description="Golub ALL/AML leukemia microarray, k=50 (new dataset, new splits).",
    ),
    HighDimDataset(
        name="golub_leukemia_k200",
        k_selected=200,
        narrative_group="golub_k_sweep",
        description="Golub ALL/AML leukemia microarray, k=200 (new splits).",
    ),
    HighDimDataset(
        name="golub_leukemia_k500",
        k_selected=500,
        narrative_group="golub_k_sweep",
        description="Golub ALL/AML leukemia microarray, k=500 (prompt stress test).",
    ),
]

# Cross-dataset narrative also references alon_colon@k=50, but we don't duplicate
# the runs — the analysis script joins on (dataset, seed).
DEFAULT_METHODS: list[str] = ["qualsynth", "smote"]
OPTIONAL_METHODS: list[str] = ["ctgan", "tabddpm", "tabfairgdt"]
DEFAULT_SEEDS: list[int] = [42, 123, 456]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QualSynth high-dimensional extension experiments (Alon k-sweep + Golub).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Dataset names to include. Defaults to all four high-dim variants "
            "(alon_colon, alon_colon_k200, alon_colon_k500, golub_leukemia, "
            "golub_leukemia_k200, golub_leukemia_k500)."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Methods to include. Defaults to {DEFAULT_METHODS}. Optional: {OPTIONAL_METHODS}.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Seeds. Defaults to the 3-seed reviewer-revision footprint {DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reviewer_revision" / "high_dim_extension",
        help="Where to write per-seed JSONs, summary CSV, manifest, and analysis stub.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already have a successful per-seed JSON on disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix and exit without running anything.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not archive any pre-existing high_dim_extension outputs.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--validation-mode",
        choices=["standard", "auto", "high_dimensional"],
        default="auto",
        help="Universal-validator mode. `auto` picks high_dimensional when n_features ≥ 30 and n_train < n_features.",
    )
    parser.add_argument(
        "--disable-universal-validation",
        action="store_true",
        help="Bypass the universal validator (matches component-ablation flag for parity).",
    )
    parser.add_argument(
        "--list-matrix",
        action="store_true",
        help="Print the full (dataset × method × seed) matrix and exit.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_datasets(requested: Optional[Iterable[str]]) -> list[HighDimDataset]:
    by_name = {dataset.name: dataset for dataset in HIGH_DIM_DATASETS}
    if not requested:
        return list(HIGH_DIM_DATASETS)
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(
            f"Unknown high-dim datasets: {missing}. Available: {sorted(by_name)}."
        )
    return [by_name[name] for name in requested]


def resolve_methods(loader: ConfigLoader, requested: Iterable[str]) -> list[str]:
    available = set(loader.list_methods())
    methods = list(dict.fromkeys(requested))
    missing = [name for name in methods if name not in available]
    if missing:
        raise ValueError(f"Unknown methods: {missing}. Available: {sorted(available)}.")
    return methods


def archive_existing(output_dir: Path, no_archive: bool) -> Optional[Path]:
    if no_archive or not output_dir.exists():
        return None
    if not any(output_dir.iterdir()):
        return None
    archive_root = PROJECT_ROOT / "results" / "reviewer_revision" / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"high_dim_extension_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(output_dir), str(destination))
    return destination


def load_existing_result(path: Path) -> Optional[ExperimentResult]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return ExperimentResult(**json.load(handle))
    except Exception:
        return None


def result_to_summary_row(
    result: ExperimentResult,
    dataset: HighDimDataset,
    source_file: Path,
) -> dict[str, Any]:
    payload = asdict(result)
    row = cost_runtime_row(payload, source_file=source_file.relative_to(PROJECT_ROOT))
    avg_performance = result.avg_performance or {}
    avg_fairness = result.avg_fairness or {}
    row.update(
        {
            "experiment_id": result.experiment_id,
            "narrative_group": dataset.narrative_group,
            "k_selected": dataset.k_selected,
            "avg_f1": avg_performance.get("f1"),
            "avg_roc_auc": avg_performance.get("roc_auc"),
            "avg_precision": avg_performance.get("precision"),
            "avg_recall": avg_performance.get("recall"),
            "avg_balanced_accuracy": avg_performance.get("balanced_accuracy"),
            "avg_pr_auc": avg_performance.get("pr_auc"),
            "avg_mcc": avg_performance.get("mcc"),
            "avg_dpd": avg_fairness.get("demographic_parity_difference"),
            "avg_eod": avg_fairness.get("equal_opportunity_difference"),
            "timestamp": result.timestamp,
        }
    )
    return row


def save_manifest(
    output_dir: Path,
    datasets: list[HighDimDataset],
    methods: list[str],
    seeds: list[int],
    archive_path: Optional[Path],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "datasets": [asdict(dataset) for dataset in datasets],
        "methods": methods,
        "seeds": seeds,
        "narrative_groups": sorted({dataset.narrative_group for dataset in datasets}),
        "model_name": args.model_name,
        "validation_mode": args.validation_mode,
        "universal_validation_enabled": not args.disable_universal_validation,
        "archive_path": str(archive_path.relative_to(PROJECT_ROOT)) if archive_path else None,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
    }
    manifest_path = output_dir / "high_dim_extension_manifest.json"
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
    """Persist the summary, cost/runtime per-run table, and aggregated cost/runtime markdown.

    We persist on every iteration so a long run can be inspected mid-flight; the
    per-seed JSONs remain the authoritative source of truth either way.
    """

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "high_dim_extension_summary.csv", index=False)
    summary.to_json(output_dir / "high_dim_extension_summary.json", orient="records", indent=2)
    summary.to_csv(output_dir / "high_dim_extension_cost_runtime_per_run.csv", index=False)
    cost_summary = summarize_cost_runtime(summary)
    cost_summary.to_csv(output_dir / "high_dim_extension_cost_runtime_summary.csv", index=False)
    write_cost_runtime_markdown(
        cost_summary,
        output_dir / "high_dim_extension_cost_runtime_summary.md",
        title="High-Dimensional Extension Cost and Runtime Summary",
    )


def write_basic_analysis_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Lightweight tables consumed by `build_high_dim_extension_report.py`.

    The full quality-first analysis (paired Wilcoxon, quality diagnostics, etc.)
    lives in `build_high_dim_extension_report.py`. The runner only emits the basic
    aggregations so the run itself is self-contained even if the analysis script
    has not been invoked yet.
    """

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        for name in [
            "method_ranking_by_avg_metric.csv",
            "dataset_method_summary.csv",
            "k_sweep_summary.csv",
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
        "avg_pr_auc",
        "avg_mcc",
    ]
    available_metrics = [col for col in metric_cols if col in successful.columns]

    if successful.empty or not available_metrics:
        for name in [
            "method_ranking_by_avg_metric.csv",
            "dataset_method_summary.csv",
            "k_sweep_summary.csv",
        ]:
            pd.DataFrame().to_csv(analysis_dir / name, index=False)
        return

    ranking = (
        successful.groupby("method", dropna=False)[available_metrics]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("avg_f1" if "avg_f1" in available_metrics else available_metrics[0], ascending=False)
    )
    dataset_summary = (
        successful.groupby(["dataset", "method", "k_selected"], dropna=False)[available_metrics + ["n_generated"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["dataset", "method"])
    )
    # k_sweep table: only entries on the alon source pivoted across k.
    alon_only = successful[successful["narrative_group"] == "alon_k_sweep"].copy()
    if not alon_only.empty:
        k_sweep = (
            alon_only.groupby(["method", "k_selected"], dropna=False)[available_metrics]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["method", "k_selected"])
        )
    else:
        k_sweep = pd.DataFrame()

    ranking.to_csv(analysis_dir / "method_ranking_by_avg_metric.csv", index=False)
    dataset_summary.to_csv(analysis_dir / "dataset_method_summary.csv", index=False)
    k_sweep.to_csv(analysis_dir / "k_sweep_summary.csv", index=False)


def list_matrix(datasets: list[HighDimDataset], methods: list[str], seeds: list[int]) -> None:
    print(f"Datasets ({len(datasets)}): {[d.name for d in datasets]}")
    print(f"Methods ({len(methods)}): {methods}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Total experiments: {len(datasets) * len(methods) * len(seeds)}")


def main() -> int:
    args = parse_args()

    loader = ConfigLoader()
    datasets = resolve_datasets(args.datasets)
    methods = resolve_methods(loader, args.methods)
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    output_dir = resolve_path(args.output_dir)
    total = len(datasets) * len(methods) * len(seeds)

    print("=" * 88)
    print("QualSynth High-Dimensional Extension Runner")
    print("=" * 88)
    print(f"Datasets: {[d.name for d in datasets]}")
    print(f"Methods:  {methods}")
    print(f"Seeds:    {seeds}")
    print(f"Total experiments: {total}")
    print(f"Output directory:  {output_dir}")
    print("=" * 88)

    if args.list_matrix:
        list_matrix(datasets, methods, seeds)
        return 0
    if args.dry_run:
        return 0

    # We only archive when starting fresh; resume runs deliberately keep the
    # existing directory so per-seed JSONs are reused.
    archive_path = None if args.resume else archive_existing(output_dir, args.no_archive)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir, datasets, methods, seeds, archive_path, args)

    runner = ExperimentRunner(
        output_dir=str(output_dir),
        verbose=True,
        enable_universal_validation=not args.disable_universal_validation,
        validation_mode=args.validation_mode,
    )

    rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    experiment_index = 0

    # Iteration order: dataset → method → seed. This keeps all of one (dataset,
    # method) pair contiguous so we can spot context-limit / API failures early
    # at large k before burning the rest of the matrix.
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                experiment_index += 1
                result_path = output_dir / dataset.name / method / f"seed{seed}.json"
                print(
                    f"\n[{experiment_index}/{total}] {dataset.name} (k={dataset.k_selected}) / "
                    f"{method} / seed {seed}"
                )
                if args.resume:
                    existing = load_existing_result(result_path)
                    if existing and existing.success:
                        print("  ↳ skipped (existing successful result)")
                        rows.append(result_to_summary_row(existing, dataset, result_path))
                        skipped += 1
                        continue

                result = runner.run_experiment(
                    dataset_name=dataset.name,
                    method_name=method,
                    seed=seed,
                    save_results=True,
                    max_iterations_override=args.max_iterations,
                    model_name_override=args.model_name,
                    batch_size_override=args.batch_size,
                )
                rows.append(result_to_summary_row(result, dataset, result_path))
                completed += int(result.success)
                failed += int(not result.success)
                save_summaries(output_dir, rows)

    save_summaries(output_dir, rows)
    write_basic_analysis_tables(output_dir, rows)

    print("\n" + "=" * 88)
    print("High-Dimensional Extension Batch Complete")
    print("=" * 88)
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Summary:   {output_dir / 'high_dim_extension_summary.csv'}")
    print(
        "Next:      python scripts/build_high_dim_extension_report.py  "
        "(quality-first analysis report)"
    )
    print("=" * 88)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
