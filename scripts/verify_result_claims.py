#!/usr/bin/env python3
"""Build authoritative benchmark tables from explicit result roots."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUALSYNTH_CORRECTION_ROOT = (
    PROJECT_ROOT / "results/reviewer_revision/canonical_dedup_correction/openrouter1"
)
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.data.splitting import encode_features, load_split
from src.qualsynth.utils.config_loader import ConfigLoader


DATASETS = [
    "german_credit",
    "breast_cancer",
    "pima_diabetes",
    "wine_quality",
    "yeast",
    "haberman",
    "thyroid",
    "htru2",
]
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
PRIMARY_METHODS = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]
LEGACY_METHODS = ["qualsynth", "smote", "ctgan", "tabfairgdt"]
SCOPES = {
    "five_method": PRIMARY_METHODS,
    "legacy_four_method": LEGACY_METHODS,
}
CLASSIFIER_ORDER = ["RandomForest", "XGBoost", "LogisticRegression"]


@dataclass(frozen=True)
class ResultSource:
    method: str
    result_dir: Path
    log_dir: Path
    source_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unify manuscript statistics from explicit benchmark result roots."
    )
    parser.add_argument(
        "--qualsynth-results-dir",
        default="results/openrouter_old",
        help="Directory containing complete archived QualSynth result JSONs.",
    )
    parser.add_argument(
        "--qualsynth-log-dir",
        default="results/openrouter_old/logs",
        help="Directory containing archived QualSynth CSV log artifacts.",
    )
    parser.add_argument(
        "--baseline-results-dir",
        default="results/experiments",
        help="Directory containing SMOTE/CTGAN/TabFairGDT result JSONs.",
    )
    parser.add_argument(
        "--baseline-log-dir",
        default="results/experiments/logs",
        help="Directory containing SMOTE/CTGAN/TabFairGDT CSV log artifacts.",
    )
    parser.add_argument(
        "--tabddpm-results-dir",
        default="results/reviewer_revision/tabddpm_main",
        help="Directory containing the regenerated TabDDPM result JSONs.",
    )
    parser.add_argument(
        "--tabddpm-log-dir",
        default="results/reviewer_revision/tabddpm_main/logs",
        help="Directory containing regenerated TabDDPM CSV log artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reviewer_revision/claim_verification",
        help="Directory for the rebuilt statistics bundle.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def get_sources(args: argparse.Namespace) -> Dict[str, ResultSource]:
    return {
        "qualsynth": ResultSource(
            method="qualsynth",
            result_dir=PROJECT_ROOT / args.qualsynth_results_dir,
            log_dir=PROJECT_ROOT / args.qualsynth_log_dir,
            source_label="qualsynth_archived",
        ),
        "smote": ResultSource(
            method="smote",
            result_dir=PROJECT_ROOT / args.baseline_results_dir,
            log_dir=PROJECT_ROOT / args.baseline_log_dir,
            source_label="baseline_json",
        ),
        "ctgan": ResultSource(
            method="ctgan",
            result_dir=PROJECT_ROOT / args.baseline_results_dir,
            log_dir=PROJECT_ROOT / args.baseline_log_dir,
            source_label="baseline_json",
        ),
        "tabfairgdt": ResultSource(
            method="tabfairgdt",
            result_dir=PROJECT_ROOT / args.baseline_results_dir,
            log_dir=PROJECT_ROOT / args.baseline_log_dir,
            source_label="baseline_json",
        ),
        "tabddpm": ResultSource(
            method="tabddpm",
            result_dir=PROJECT_ROOT / args.tabddpm_results_dir,
            log_dir=PROJECT_ROOT / args.tabddpm_log_dir,
            source_label="tabddpm_regenerated",
        ),
    }


def result_json_path(source: ResultSource, dataset: str, seed: int) -> Path:
    corrected = QUALSYNTH_CORRECTION_ROOT / dataset / source.method / f"seed{seed}.json"
    if source.method == "qualsynth" and corrected.exists():
        return corrected
    return source.result_dir / dataset / source.method / f"seed{seed}.json"


def generated_csv_path(source: ResultSource, dataset: str, seed: int) -> Path:
    corrected = QUALSYNTH_CORRECTION_ROOT / "logs" / f"{dataset}_{source.method}_seed{seed}_generated_samples.csv"
    if source.method == "qualsynth" and corrected.exists():
        return corrected
    return source.log_dir / f"{dataset}_{source.method}_seed{seed}_generated_samples.csv"


def validated_csv_path(source: ResultSource, dataset: str, seed: int) -> Path:
    corrected = QUALSYNTH_CORRECTION_ROOT / "logs" / f"{dataset}_{source.method}_seed{seed}_validated_samples.csv"
    if source.method == "qualsynth" and corrected.exists():
        return corrected
    return source.log_dir / f"{dataset}_{source.method}_seed{seed}_validated_samples.csv"


def load_result_payloads(sources: Dict[str, ResultSource]) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for method, source in sources.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                json_path = result_json_path(source, dataset, seed)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "result_json": str(json_path),
                        "source_label": source.source_label,
                        "exists": json_path.exists(),
                    }
                )

    frame = pd.DataFrame(rows)
    if not frame["exists"].all():
        missing = frame.loc[~frame["exists"], ["dataset", "method", "seed", "result_json"]]
        missing_text = "\n".join(
            f"- {row.dataset}/{row.method}/seed{row.seed}: {row.result_json}"
            for row in missing.itertuples(index=False)
        )
        raise RuntimeError(
            "Benchmark bundle is incomplete. Missing result JSONs:\n"
            f"{missing_text}"
        )
    return frame


def build_per_seed_rows(result_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[dict[str, Any]] = []
    classifier_rows: List[dict[str, Any]] = []

    for row in result_index.itertuples(index=False):
        payload = load_json(Path(row.result_json))
        avg_perf = payload.get("avg_performance", {}) or {}
        avg_fair = payload.get("avg_fairness", {}) or {}
        metric_rows.append(
            {
                "dataset": row.dataset,
                "method": row.method,
                "seed": row.seed,
                "source_label": row.source_label,
                "success": bool(payload.get("success", False)),
                "f1": avg_perf.get("f1"),
                "roc_auc": avg_perf.get("roc_auc"),
                "pr_auc": avg_perf.get("pr_auc"),
                "balanced_accuracy": avg_perf.get("balanced_accuracy"),
                "avg_demographic_parity_difference": avg_fair.get("avg_demographic_parity_difference"),
                "result_json": row.result_json,
            }
        )

        perf_by_classifier = payload.get("performance_metrics", {}) or {}
        for classifier in CLASSIFIER_ORDER:
            clf_metrics = perf_by_classifier.get(classifier)
            if not clf_metrics:
                continue
            classifier_rows.append(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "seed": row.seed,
                    "classifier": classifier,
                    "f1": clf_metrics.get("f1"),
                    "roc_auc": clf_metrics.get("roc_auc"),
                    "pr_auc": clf_metrics.get("pr_auc"),
                    "balanced_accuracy": clf_metrics.get("balanced_accuracy"),
                    "result_json": row.result_json,
                }
            )

    per_seed = pd.DataFrame(metric_rows)
    per_classifier = pd.DataFrame(classifier_rows)
    if per_seed.empty:
        raise RuntimeError("No per-seed metrics could be loaded from the result roots.")
    return per_seed, per_classifier


def summarize_completeness(result_index: pd.DataFrame) -> pd.DataFrame:
    return (
        result_index.groupby(["dataset", "method"], dropna=False)
        .agg(
            n_expected=("seed", "count"),
            n_jsons=("exists", "sum"),
        )
        .reset_index()
        .sort_values(["dataset", "method"])
    )


def compute_scope_dataset_means(
    per_seed: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    pivot = (
        per_seed.loc[per_seed["method"].isin(methods)]
        .groupby(["dataset", "method"], dropna=False)[metric]
        .mean()
        .reset_index()
        .pivot(index="dataset", columns="method", values=metric)
    )
    pivot = pivot[list(methods)]
    return pivot.reset_index()


def mean_ranks_from_dataset_means(dataset_means: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    pivot = dataset_means.set_index("dataset")[list(methods)]
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    return (
        ranks.mean(axis=0)
        .rename("mean_rank")
        .reset_index()
        .rename(columns={"index": "method"})
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def holm_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    m = len(ordered)
    running_max = 0.0
    for idx, (name, p_value) in enumerate(ordered):
        corrected = min(1.0, (m - idx) * p_value)
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return None
    ranks = rankdata(np.abs(diffs))
    positive = ranks[diffs > 0].sum()
    negative = ranks[diffs < 0].sum()
    n = len(diffs)
    return float((positive - negative) / (n * (n + 1) / 2.0))


def compute_global_stats(
    dataset_means: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
) -> Dict[str, Any]:
    pivot = dataset_means.set_index("dataset")[list(methods)]
    statistic, p_value = friedmanchisquare(*[pivot[col].values for col in methods])
    pairwise_rows: List[dict[str, Any]] = []
    raw_p_values: Dict[str, float] = {}

    for baseline in methods[1:]:
        q = pivot["qualsynth"].values
        b = pivot[baseline].values
        _, pairwise_p = wilcoxon(q, b, alternative="two-sided", zero_method="wilcox")
        raw_p_values[baseline] = float(pairwise_p)
        pairwise_rows.append(
            {
                "comparison": f"qualsynth_vs_{baseline}",
                "raw_p_value": float(pairwise_p),
                "rank_biserial": rank_biserial(q, b),
                "qualsynth_mean": float(np.mean(q)),
                "baseline_mean": float(np.mean(b)),
            }
        )

    adjusted = holm_correction(raw_p_values)
    for row in pairwise_rows:
        baseline = row["comparison"].replace("qualsynth_vs_", "")
        row["holm_p_value"] = adjusted[baseline]

    return {
        "metric": metric,
        "methods": list(methods),
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
        "pairwise": pairwise_rows,
    }


def compute_within_dataset_friedman(
    per_seed: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    filtered = per_seed.loc[per_seed["method"].isin(methods)]
    for dataset in DATASETS:
        pivot = (
            filtered.loc[filtered["dataset"] == dataset, ["seed", "method", metric]]
            .pivot(index="seed", columns="method", values=metric)
            .reindex(columns=list(methods))
        )
        if pivot.isna().any().any():
            continue
        statistic, p_value = friedmanchisquare(*[pivot[col].values for col in methods])
        rows.append(
            {
                "dataset": dataset,
                "friedman_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def compute_classifier_block_ranks(
    per_classifier: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_means = (
        per_classifier.loc[per_classifier["method"].isin(methods)]
        .groupby(["dataset", "classifier", "method"], dropna=False)[metric]
        .mean()
        .reset_index()
        .pivot(index=["dataset", "classifier"], columns="method", values=metric)
    )
    block_means = block_means[list(methods)]
    ranks = block_means.rank(axis=1, ascending=False, method="average")
    mean_ranks = (
        ranks.mean(axis=0)
        .rename("mean_rank")
        .reset_index()
        .rename(columns={"index": "method"})
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )
    return block_means.reset_index(), mean_ranks


def _needs_encoding(frame: pd.DataFrame) -> bool:
    return any(dtype == "object" for dtype in frame.dtypes)


def prepare_numeric_frame(dataset: str, seed: int, sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_raw = load_split(dataset, seed=seed, return_raw=True)
    split_encoded = load_split(dataset, seed=seed, return_raw=False)
    y_train = split_raw["y_train"]
    minority_label = y_train.value_counts().idxmin()
    train_encoded_minority = split_encoded["X_train"].loc[y_train == minority_label].reset_index(drop=True)
    preprocessor = split_raw.get("preprocessor")

    features = sample_frame.drop(columns=["target"], errors="ignore").copy()
    train_numeric = train_encoded_minority.copy().apply(pd.to_numeric, errors="coerce")
    train_numeric = train_numeric.fillna(train_numeric.median()).fillna(0)
    if features.empty:
        return train_numeric, pd.DataFrame(columns=train_numeric.columns)
    if _needs_encoding(features):
        numeric_samples = encode_features(features, preprocessor)
    else:
        numeric_samples = features.copy().apply(pd.to_numeric, errors="coerce")

    train_numeric = train_encoded_minority.copy()
    for column in train_numeric.columns:
        if column not in numeric_samples.columns:
            numeric_samples[column] = train_numeric[column].median()
    extra_columns = [column for column in numeric_samples.columns if column not in train_numeric.columns]
    if extra_columns:
        numeric_samples = numeric_samples.drop(columns=extra_columns)
    numeric_samples = numeric_samples[train_numeric.columns]

    train_numeric = train_numeric.apply(pd.to_numeric, errors="coerce")
    numeric_samples = numeric_samples.apply(pd.to_numeric, errors="coerce")
    train_numeric = train_numeric.fillna(train_numeric.median()).fillna(0)
    numeric_samples = numeric_samples.fillna(train_numeric.median()).fillna(0)
    return train_numeric, numeric_samples


def correlation_distance(train_df: pd.DataFrame, sample_df: pd.DataFrame) -> Optional[float]:
    if len(train_df) < 3 or len(sample_df) < 3 or train_df.shape[1] < 2:
        return None
    corr_train = train_df.corr(numeric_only=True).fillna(0)
    corr_sample = sample_df.corr(numeric_only=True).fillna(0)
    if corr_train.shape != corr_sample.shape:
        return None
    tri = np.triu_indices_from(corr_train, k=1)
    if len(tri[0]) == 0:
        return None
    diff = np.abs(corr_train.values[tri] - corr_sample.values[tri])
    return float(np.mean(diff))


def exact_duplicate_ratio(sample_df: pd.DataFrame) -> Optional[float]:
    if sample_df.empty:
        return None
    return float(sample_df.duplicated().mean())


def build_quality_audit_rows(sources: Dict[str, ResultSource]) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for method, source in sources.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                generated_path = generated_csv_path(source, dataset, seed)
                validated_path = validated_csv_path(source, dataset, seed)
                if not generated_path.exists():
                    raise RuntimeError(
                        f"Missing generated-sample CSV for {dataset}/{method}/seed{seed}: "
                        f"{generated_path}"
                    )
                generated_df = pd.read_csv(generated_path)
                if validated_path.exists():
                    validated_df = pd.read_csv(validated_path)
                else:
                    payload = load_json(result_json_path(source, dataset, seed))
                    n_validated = int(payload.get("metadata", {}).get("n_validated", 0) or 0)
                    if n_validated != 0:
                        raise RuntimeError(
                            f"Missing validated-sample CSV for {dataset}/{method}/seed{seed}: "
                            f"{validated_path}"
                        )
                    validated_df = pd.DataFrame(columns=generated_df.columns)
                train_numeric, validated_numeric = prepare_numeric_frame(dataset, seed, validated_df)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "source_label": source.source_label,
                        "n_generated_candidates": int(len(generated_df)),
                        "n_validated": int(len(validated_df)),
                        "post_validation_acceptance_rate": (
                            float(len(validated_df) / len(generated_df)) if len(generated_df) else np.nan
                        ),
                        "raw_exact_duplicate_rate": exact_duplicate_ratio(generated_df),
                        "validated_correlation_distance": correlation_distance(train_numeric, validated_numeric),
                        "generated_csv": str(generated_path),
                        "validated_csv": str(validated_path),
                    }
                )
    return pd.DataFrame(rows)


def summarize_quality_audit(quality_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        quality_rows.groupby("method", dropna=False)
        .agg(
            n_runs=("seed", "count"),
            mean_acceptance_rate=("post_validation_acceptance_rate", "mean"),
            mean_raw_duplicate_rate=("raw_exact_duplicate_rate", "mean"),
            mean_correlation_distance=("validated_correlation_distance", "mean"),
        )
        .reset_index()
        .sort_values("mean_correlation_distance")
        .reset_index(drop=True)
    )
    dataset_summary = (
        quality_rows.groupby(["dataset", "method"], dropna=False)
        .agg(
            mean_acceptance_rate=("post_validation_acceptance_rate", "mean"),
            mean_raw_duplicate_rate=("raw_exact_duplicate_rate", "mean"),
            mean_correlation_distance=("validated_correlation_distance", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "method"])
        .reset_index(drop=True)
    )
    return summary, dataset_summary


def build_fairness_rows(per_seed: pd.DataFrame, configs: ConfigLoader) -> tuple[pd.DataFrame, pd.DataFrame]:
    fairness_datasets = {
        dataset
        for dataset in DATASETS
        if configs.load_dataset_config(dataset).sensitive_attributes
    }
    fairness_rows = per_seed.loc[
        per_seed["dataset"].isin(fairness_datasets)
        & per_seed["avg_demographic_parity_difference"].notna(),
        ["dataset", "method", "seed", "avg_demographic_parity_difference", "result_json"],
    ].copy()
    fairness_summary = (
        fairness_rows.groupby("method", dropna=False)
        .agg(
            n_runs=("seed", "count"),
            mean_dpd=("avg_demographic_parity_difference", "mean"),
            median_dpd=("avg_demographic_parity_difference", "median"),
            max_dpd=("avg_demographic_parity_difference", "max"),
        )
        .reset_index()
        .sort_values("mean_dpd")
        .reset_index(drop=True)
    )
    return fairness_rows, fairness_summary


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"
    return frame.to_markdown(index=False)


def write_report(
    output_path: Path,
    completeness: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    scope_summaries: Dict[str, Dict[str, Any]],
) -> None:
    lines = [
        "# Reviewer Revision Statistics Bundle",
        "",
        "## Completeness",
        "",
        markdown_table(completeness),
        "",
        "## Fairness Summary",
        "",
        markdown_table(fairness_summary),
        "",
        "## Quality Audit Summary",
        "",
        markdown_table(quality_summary),
        "",
    ]

    for scope_name, summary in scope_summaries.items():
        lines.extend(
            [
                f"## {scope_name.replace('_', ' ').title()}",
                "",
                "### Mean Dataset Ranks",
                "",
                "#### F1",
                "",
                markdown_table(summary["mean_ranks_f1"]),
                "",
                "#### ROC-AUC",
                "",
                markdown_table(summary["mean_ranks_roc_auc"]),
                "",
                f"- Global Friedman F1 p-value: `{summary['global_f1']['friedman_p_value']:.6g}`",
                f"- Global Friedman ROC-AUC p-value: `{summary['global_roc_auc']['friedman_p_value']:.6g}`",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_scope_outputs(
    scope_dir: Path,
    methods: Sequence[str],
    per_seed: pd.DataFrame,
    per_classifier: pd.DataFrame,
) -> Dict[str, Any]:
    scope_dir.mkdir(parents=True, exist_ok=True)
    filtered_seed = per_seed.loc[per_seed["method"].isin(methods)].copy()
    filtered_classifier = per_classifier.loc[per_classifier["method"].isin(methods)].copy()

    dataset_means_f1 = compute_scope_dataset_means(filtered_seed, methods, "f1")
    dataset_means_roc = compute_scope_dataset_means(filtered_seed, methods, "roc_auc")
    mean_ranks_f1 = mean_ranks_from_dataset_means(dataset_means_f1, methods)
    mean_ranks_roc = mean_ranks_from_dataset_means(dataset_means_roc, methods)
    global_f1 = compute_global_stats(dataset_means_f1, methods, "f1")
    global_roc = compute_global_stats(dataset_means_roc, methods, "roc_auc")
    within_dataset_f1 = compute_within_dataset_friedman(filtered_seed, methods, "f1")
    classifier_block_f1, classifier_ranks_f1 = compute_classifier_block_ranks(filtered_classifier, methods, "f1")
    classifier_block_roc, classifier_ranks_roc = compute_classifier_block_ranks(filtered_classifier, methods, "roc_auc")

    dataset_means_f1.to_csv(scope_dir / "dataset_means_f1.csv", index=False)
    dataset_means_roc.to_csv(scope_dir / "dataset_means_roc_auc.csv", index=False)
    mean_ranks_f1.to_csv(scope_dir / "mean_ranks_f1.csv", index=False)
    mean_ranks_roc.to_csv(scope_dir / "mean_ranks_roc_auc.csv", index=False)
    within_dataset_f1.to_csv(scope_dir / "within_dataset_friedman_f1.csv", index=False)
    classifier_block_f1.to_csv(scope_dir / "per_dataset_classifier_f1.csv", index=False)
    classifier_block_roc.to_csv(scope_dir / "per_dataset_classifier_roc_auc.csv", index=False)
    classifier_ranks_f1.to_csv(scope_dir / "mean_ranks_per_dataset_classifier_f1.csv", index=False)
    classifier_ranks_roc.to_csv(scope_dir / "mean_ranks_per_dataset_classifier_roc_auc.csv", index=False)

    with open(scope_dir / "global_stats_f1.json", "w", encoding="utf-8") as handle:
        json.dump(global_f1, handle, indent=2)
    with open(scope_dir / "global_stats_roc_auc.json", "w", encoding="utf-8") as handle:
        json.dump(global_roc, handle, indent=2)

    return {
        "methods": list(methods),
        "mean_ranks_f1": mean_ranks_f1,
        "mean_ranks_roc_auc": mean_ranks_roc,
        "global_f1": global_f1,
        "global_roc_auc": global_roc,
    }


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = get_sources(args)
    for source in sources.values():
        ensure_exists(source.result_dir, f"{source.method} result directory")
        ensure_exists(source.log_dir, f"{source.method} log directory")

    result_index = load_result_payloads(sources)
    completeness = summarize_completeness(result_index)
    per_seed, per_classifier = build_per_seed_rows(result_index)
    configs = ConfigLoader()
    fairness_rows, fairness_summary = build_fairness_rows(per_seed, configs)
    quality_rows = build_quality_audit_rows(sources)
    quality_summary, quality_dataset_summary = summarize_quality_audit(quality_rows)

    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    per_classifier.to_csv(output_dir / "per_seed_classifier_metrics.csv", index=False)
    completeness.to_csv(output_dir / "completeness.csv", index=False)
    fairness_rows.to_csv(output_dir / "fairness_rows.csv", index=False)
    fairness_summary.to_csv(output_dir / "fairness_summary.csv", index=False)
    quality_rows.to_csv(output_dir / "quality_audit_rows.csv", index=False)
    quality_summary.to_csv(output_dir / "quality_audit_summary.csv", index=False)
    quality_dataset_summary.to_csv(output_dir / "quality_audit_dataset_summary.csv", index=False)

    scope_summaries: Dict[str, Dict[str, Any]] = {}
    for scope_name, methods in SCOPES.items():
        scope_summaries[scope_name] = write_scope_outputs(output_dir / scope_name, methods, per_seed, per_classifier)

    write_report(
        output_dir / "claim_verification_report.md",
        completeness=completeness,
        fairness_summary=fairness_summary,
        quality_summary=quality_summary,
        scope_summaries=scope_summaries,
    )

    manifest = {
        "sources": {
            method: {
                "result_dir": str(source.result_dir),
                "log_dir": str(source.log_dir),
                "source_label": source.source_label,
            }
            for method, source in sources.items()
        },
        "qualsynth_correction_overlay": str(
            QUALSYNTH_CORRECTION_ROOT.relative_to(PROJECT_ROOT)
        ),
        "scopes": {scope: methods for scope, methods in SCOPES.items()},
    }
    with open(output_dir / "bundle_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote rebuilt statistics bundle to: {output_dir}")


if __name__ == "__main__":
    main()
