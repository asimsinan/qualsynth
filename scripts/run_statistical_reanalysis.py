#!/usr/bin/env python3
"""Reviewer-revision statistical reanalysis for benchmark result roots."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_ROOTS = [
    Path("results/reviewer_revision/high_dimensional_benchmark/experiments"),
]
DEFAULT_OUTPUT_DIR = Path("results/reviewer_revision/statistical_reanalysis")
PRIMARY_METHOD = "qualsynth"
METHOD_ORDER = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]
CLASSIFIER_ORDER = ["RandomForest", "XGBoost", "LogisticRegression"]
METRICS = ["accuracy", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "mcc"]


@dataclass(frozen=True)
class TestResult:
    family: str
    scope: str
    metric: str
    test: str
    statistic: float | None
    p_value: float | None
    n_blocks: int
    status: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reviewer-revision statistical reanalysis from benchmark result JSONs.",
    )
    parser.add_argument(
        "--result-root",
        action="append",
        type=Path,
        default=None,
        help="Result root with dataset/method/seed*.json layout. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for statistical reanalysis artifacts.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_ORDER,
        help="Methods to include, in rank/table order.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_result_files(result_roots: Iterable[Path]) -> Iterable[Path]:
    for root in result_roots:
        for dataset_dir in sorted(root.iterdir() if root.exists() else []):
            if not dataset_dir.is_dir() or dataset_dir.name == "logs":
                continue
            for method_dir in sorted(dataset_dir.iterdir()):
                if not method_dir.is_dir() or method_dir.name == "logs":
                    continue
                yield from sorted(method_dir.glob("seed*.json"))


def parse_seed(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("seed"):
        return None
    try:
        return int(stem.replace("seed", "", 1))
    except ValueError:
        return None


def build_metric_tables(result_files: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []

    for path in result_files:
        payload = load_json(path)
        dataset = payload.get("dataset") or path.parents[1].name
        method = payload.get("method") or path.parent.name
        seed = payload.get("seed") if payload.get("seed") is not None else parse_seed(path)
        metadata = payload.get("metadata", {}) or {}
        avg_performance = payload.get("avg_performance", {}) or {}

        row = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "success": bool(payload.get("success", False)),
            "result_json": str(path),
            "n_generated_raw": metadata.get("n_generated_raw"),
            "n_validated": metadata.get("n_validated"),
            "validation_rate": metadata.get("validation_rate"),
            "duplicate_ratio": metadata.get("duplicate_ratio"),
            "quality_pass_rate": metadata.get("quality_pass_rate"),
            "execution_time": payload.get("execution_time"),
            "generation_time": payload.get("generation_time"),
            "llm_calls": payload.get("llm_calls", 0),
            "total_tokens": payload.get("total_tokens", 0),
            "generation_cost": payload.get("generation_cost", 0.0),
        }
        for metric in METRICS:
            row[metric] = avg_performance.get(metric)
        per_run_rows.append(row)

        for classifier_name, classifier_metrics in (payload.get("performance_metrics", {}) or {}).items():
            classifier_row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "classifier": classifier_name,
                "result_json": str(path),
            }
            for metric in METRICS:
                classifier_row[metric] = classifier_metrics.get(metric)
            classifier_rows.append(classifier_row)

    return pd.DataFrame(per_run_rows), pd.DataFrame(classifier_rows)


def ordered_methods(frame: pd.DataFrame, methods: Sequence[str]) -> list[str]:
    present = set(frame["method"].dropna())
    return [method for method in methods if method in present]


def complete_pivot(
    frame: pd.DataFrame,
    index: str | list[str],
    methods: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    pivot = frame.pivot_table(index=index, columns="method", values=metric, aggfunc="mean")
    existing = [method for method in methods if method in pivot.columns]
    pivot = pivot.reindex(columns=existing)
    return pivot.dropna(axis=0, how="any")


def mean_ranks(pivot: pd.DataFrame) -> pd.DataFrame:
    if pivot.empty:
        return pd.DataFrame(columns=["method", "mean_rank"])
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    return (
        ranks.mean(axis=0)
        .rename("mean_rank")
        .reset_index()
        .rename(columns={"index": "method"})
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def safe_friedman(
    pivot: pd.DataFrame,
    family: str,
    scope: str,
    metric: str,
    note: str = "",
) -> TestResult:
    if pivot.shape[0] < 2:
        return TestResult(
            family=family,
            scope=scope,
            metric=metric,
            test="friedman",
            statistic=None,
            p_value=None,
            n_blocks=int(pivot.shape[0]),
            status="not_run_insufficient_blocks",
            note=note or "Friedman test requires at least two complete paired blocks.",
        )
    if pivot.shape[1] < 3:
        return TestResult(
            family=family,
            scope=scope,
            metric=metric,
            test="friedman",
            statistic=None,
            p_value=None,
            n_blocks=int(pivot.shape[0]),
            status="not_run_insufficient_methods",
            note=note or "Friedman test requires at least three methods.",
        )
    statistic, p_value = friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
    return TestResult(
        family=family,
        scope=scope,
        metric=metric,
        test="friedman",
        statistic=float(statistic),
        p_value=float(p_value),
        n_blocks=int(pivot.shape[0]),
        status="run",
        note=note,
    )


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float | None:
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return None
    ranks = rankdata(np.abs(diffs))
    positive = ranks[diffs > 0].sum()
    negative = ranks[diffs < 0].sum()
    n = len(diffs)
    return float((positive - negative) / (n * (n + 1) / 2.0))


def safe_wilcoxon(x: Sequence[float], y: Sequence[float]) -> tuple[float | None, float | None, str]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return None, None, "not_run_insufficient_pairs"
    if np.allclose(x_arr - y_arr, 0):
        return None, 1.0, "all_differences_zero"
    statistic, p_value = wilcoxon(x_arr, y_arr, alternative="two-sided", zero_method="wilcox")
    return float(statistic), float(p_value), "run"


def holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(valid.items()):
        corrected = min(1.0, (m - rank) * float(p_value))
        running_max = max(running_max, corrected)
        adjusted.loc[idx] = running_max
    return adjusted


def pairwise_against_primary(
    pivot: pd.DataFrame,
    family: str,
    scope_columns: dict[str, Any],
    metric: str,
    primary: str = PRIMARY_METHOD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if primary not in pivot.columns:
        return pd.DataFrame(rows)
    for method in pivot.columns:
        if method == primary:
            continue
        aligned = pivot[[primary, method]].dropna()
        statistic, p_value, status = safe_wilcoxon(aligned[primary], aligned[method])
        rows.append(
            {
                **scope_columns,
                "family": family,
                "metric": metric,
                "comparison": f"{primary}_vs_{method}",
                "baseline": method,
                "test": "wilcoxon_signed_rank",
                "statistic": statistic,
                "raw_p_value": p_value,
                "n_pairs": int(len(aligned)),
                "status": status,
                "primary_mean": float(aligned[primary].mean()) if not aligned.empty else np.nan,
                "baseline_mean": float(aligned[method].mean()) if not aligned.empty else np.nan,
                "mean_difference": float((aligned[primary] - aligned[method]).mean()) if not aligned.empty else np.nan,
                "rank_biserial": rank_biserial(aligned[primary], aligned[method]) if not aligned.empty else None,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_p_value"] = holm_adjust(result["raw_p_value"])
    return result


def build_per_dataset_pairwise(per_run: pd.DataFrame, methods: Sequence[str], metric: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in sorted(per_run["dataset"].dropna().unique()):
        pivot = complete_pivot(
            per_run.loc[per_run["dataset"] == dataset],
            index="seed",
            methods=methods,
            metric=metric,
        )
        frames.append(
            pairwise_against_primary(
                pivot,
                family=f"per_dataset_{metric}_pairwise__{dataset}",
                scope_columns={"dataset": dataset, "scope": "within_dataset_seed_pairs"},
                metric=metric,
            )
        )
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()


def build_dataset_friedman(per_run: pd.DataFrame, methods: Sequence[str], metric: str) -> tuple[pd.DataFrame, TestResult]:
    dataset_means = complete_pivot(per_run, index="dataset", methods=methods, metric=metric)
    result = safe_friedman(
        dataset_means,
        family=f"dataset_level_{metric}_omnibus",
        scope="dataset_mean_blocks",
        metric=metric,
        note="Dataset-level block uses one mean per dataset and method; this is the primary omnibus design when at least two datasets are available.",
    )
    return dataset_means.reset_index(), result


def build_classifier_block_stats(
    per_classifier: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, TestResult]:
    block_means = complete_pivot(
        per_classifier,
        index=["dataset", "classifier"],
        methods=methods,
        metric=metric,
    )
    omnibus = safe_friedman(
        block_means,
        family=f"dataset_classifier_{metric}_omnibus",
        scope="dataset_classifier_mean_blocks",
        metric=metric,
        note="Finer-grained dataset-classifier blocks requested by reviewers; interpret as sensitivity analysis because classifiers within a dataset are not fully independent.",
    )
    pairwise = pairwise_against_primary(
        block_means,
        family=f"dataset_classifier_{metric}_pairwise",
        scope_columns={"scope": "dataset_classifier_mean_blocks"},
        metric=metric,
    )
    return block_means.reset_index(), pairwise, omnibus


def summarize_performance(per_run: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "runs": ("seed", "count"),
        "successes": ("success", "sum"),
        "validation_rate_mean": ("validation_rate", "mean"),
        "duplicate_ratio_mean": ("duplicate_ratio", "mean"),
        "generation_time_mean": ("generation_time", "mean"),
        "llm_calls_sum": ("llm_calls", "sum"),
        "total_tokens_sum": ("total_tokens", "sum"),
    }
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return (
        per_run.groupby(["dataset", "method"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["dataset", "method"])
    )


def summarize_classifiers(per_classifier: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {"runs": ("seed", "count")}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return (
        per_classifier.groupby(["dataset", "classifier", "method"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["dataset", "classifier", "method"])
    )


def write_report(
    output_path: Path,
    performance_summary: pd.DataFrame,
    test_results: pd.DataFrame,
    per_dataset_f1: pd.DataFrame,
    classifier_pairwise_f1: pd.DataFrame,
) -> None:
    lines = [
        "# Statistical Reanalysis",
        "",
        "This bundle defines separate multiple-testing families for dataset-level omnibus tests, per-dataset QualSynth-vs-baseline paired tests, and dataset-classifier sensitivity analyses.",
        "",
        "## Performance Summary",
        "",
        performance_summary.to_markdown(index=False),
        "",
        "## Omnibus Tests",
        "",
        test_results.to_markdown(index=False) if not test_results.empty else "_No omnibus tests available._",
        "",
        "## Per-Dataset F1 Pairwise Tests",
        "",
        per_dataset_f1.to_markdown(index=False) if not per_dataset_f1.empty else "_No per-dataset pairwise tests available._",
        "",
        "## Dataset-Classifier F1 Pairwise Sensitivity",
        "",
        classifier_pairwise_f1.to_markdown(index=False) if not classifier_pairwise_f1.empty else "_No classifier-block pairwise tests available._",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    result_roots = [resolve_path(path) for path in (args.result_root or DEFAULT_RESULT_ROOTS)]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = list(iter_result_files(result_roots))
    if not result_files:
        raise RuntimeError(f"No result JSON files found in: {result_roots}")

    per_run, per_classifier = build_metric_tables(result_files)
    methods = ordered_methods(per_run, args.methods)
    if PRIMARY_METHOD not in methods:
        raise RuntimeError(f"Primary method '{PRIMARY_METHOD}' is missing from discovered results.")

    per_run = per_run.loc[per_run["method"].isin(methods)].copy()
    per_classifier = per_classifier.loc[per_classifier["method"].isin(methods)].copy()

    performance_summary = summarize_performance(per_run)
    classifier_summary = summarize_classifiers(per_classifier)

    test_results: list[TestResult] = []
    pairwise_frames: list[pd.DataFrame] = []
    classifier_pairwise_frames: list[pd.DataFrame] = []

    for metric in ["f1", "roc_auc"]:
        dataset_means, dataset_omnibus = build_dataset_friedman(per_run, methods, metric)
        dataset_means.to_csv(output_dir / f"dataset_means_{metric}.csv", index=False)
        mean_ranks(dataset_means.set_index("dataset")).to_csv(output_dir / f"dataset_mean_ranks_{metric}.csv", index=False)
        test_results.append(dataset_omnibus)

        per_dataset_pairwise = build_per_dataset_pairwise(per_run, methods, metric)
        if not per_dataset_pairwise.empty:
            per_dataset_pairwise.to_csv(output_dir / f"per_dataset_{metric}_wilcoxon_holm.csv", index=False)
            pairwise_frames.append(per_dataset_pairwise)

        classifier_blocks, classifier_pairwise, classifier_omnibus = build_classifier_block_stats(
            per_classifier,
            methods,
            metric,
        )
        classifier_blocks.to_csv(output_dir / f"dataset_classifier_blocks_{metric}.csv", index=False)
        mean_ranks(classifier_blocks.set_index(["dataset", "classifier"])).to_csv(
            output_dir / f"dataset_classifier_mean_ranks_{metric}.csv",
            index=False,
        )
        test_results.append(classifier_omnibus)
        if not classifier_pairwise.empty:
            classifier_pairwise.to_csv(output_dir / f"dataset_classifier_{metric}_wilcoxon_holm.csv", index=False)
            classifier_pairwise_frames.append(classifier_pairwise)

    all_pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    all_classifier_pairwise = (
        pd.concat(classifier_pairwise_frames, ignore_index=True)
        if classifier_pairwise_frames
        else pd.DataFrame()
    )
    test_results_frame = pd.DataFrame([result.__dict__ for result in test_results])

    families = pd.DataFrame(
        [
            {
                "family": "dataset_level_f1_omnibus",
                "description": "Friedman omnibus over one mean F1 block per dataset and method.",
                "correction": "none; one omnibus test",
            },
            {
                "family": "dataset_level_roc_auc_omnibus",
                "description": "Friedman omnibus over one mean ROC-AUC block per dataset and method.",
                "correction": "none; one omnibus test",
            },
            {
                "family": "per_dataset_f1_pairwise__<dataset>",
                "description": "Within-dataset paired seed-level Wilcoxon tests comparing QualSynth with each baseline.",
                "correction": "Holm correction within each dataset family.",
            },
            {
                "family": "per_dataset_roc_auc_pairwise__<dataset>",
                "description": "Within-dataset paired seed-level Wilcoxon tests comparing QualSynth with each baseline.",
                "correction": "Holm correction within each dataset family.",
            },
            {
                "family": "dataset_classifier_f1_pairwise",
                "description": "Sensitivity analysis over dataset-classifier mean blocks.",
                "correction": "Holm correction within metric family.",
            },
            {
                "family": "dataset_classifier_roc_auc_pairwise",
                "description": "Sensitivity analysis over dataset-classifier mean blocks.",
                "correction": "Holm correction within metric family.",
            },
        ]
    )

    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    per_classifier.to_csv(output_dir / "per_classifier_metrics.csv", index=False)
    performance_summary.to_csv(output_dir / "performance_summary.csv", index=False)
    classifier_summary.to_csv(output_dir / "classifier_summary.csv", index=False)
    test_results_frame.to_csv(output_dir / "omnibus_tests.csv", index=False)
    families.to_csv(output_dir / "multiple_testing_families.csv", index=False)
    if not all_pairwise.empty:
        all_pairwise.to_csv(output_dir / "all_per_dataset_pairwise_tests.csv", index=False)
    if not all_classifier_pairwise.empty:
        all_classifier_pairwise.to_csv(output_dir / "all_dataset_classifier_pairwise_tests.csv", index=False)

    manifest = {
        "result_roots": [str(path) for path in result_roots],
        "output_dir": str(output_dir),
        "datasets": sorted(per_run["dataset"].dropna().unique().tolist()),
        "methods": methods,
        "n_runs": int(len(per_run)),
        "n_classifier_rows": int(len(per_classifier)),
        "outputs": [
            "per_run_metrics.csv",
            "per_classifier_metrics.csv",
            "performance_summary.csv",
            "classifier_summary.csv",
            "omnibus_tests.csv",
            "multiple_testing_families.csv",
            "all_per_dataset_pairwise_tests.csv",
            "all_dataset_classifier_pairwise_tests.csv",
        ],
    }
    with open(output_dir / "statistical_reanalysis_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    write_report(
        output_dir / "statistical_reanalysis_report.md",
        performance_summary=performance_summary,
        test_results=test_results_frame,
        per_dataset_f1=all_pairwise.loc[all_pairwise["metric"] == "f1"] if not all_pairwise.empty else pd.DataFrame(),
        classifier_pairwise_f1=(
            all_classifier_pairwise.loc[all_classifier_pairwise["metric"] == "f1"]
            if not all_classifier_pairwise.empty
            else pd.DataFrame()
        ),
    )

    print(f"Wrote statistical reanalysis to: {output_dir}")
    print(performance_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
