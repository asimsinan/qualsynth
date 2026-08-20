#!/usr/bin/env python3
"""Build the Reviewer 3 six-method/no-augmentation statistics bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUALSYNTH_CORRECTION_ROOT = (
    PROJECT_ROOT / "results/reviewer_revision/canonical_dedup_correction/openrouter1"
)
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
CLASSIFIERS = ["RandomForest", "XGBoost", "LogisticRegression"]
CLASSIFIER_ALIASES = {
    "RandomForest": ("RandomForest", "random_forest"),
    "XGBoost": ("XGBoost", "xgboost"),
    "LogisticRegression": ("LogisticRegression", "logistic_regression"),
}
MAIN_METHODS = [
    "qualsynth",
    "smote",
    "ctgan",
    "tabfairgdt",
    "tabddpm",
    "no_augmentation",
]
ALL_METHODS = [*MAIN_METHODS, "no_augmentation_unweighted"]
METRICS = ["f1", "roc_auc"]
DISPLAY_NAMES = {
    "qualsynth": "QualSynth",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tabfairgdt": "TabFairGDT",
    "tabddpm": "TabDDPM",
    "no_augmentation": "NoAug-CW",
    "no_augmentation_unweighted": "NoAug-Unweighted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reviewer_revision/reviewer3_round2/no_augmentation/analysis"),
    )
    return parser.parse_args()


def source_path(method: str, dataset: str, seed: int) -> Path:
    if method == "qualsynth":
        root = PROJECT_ROOT / "results/openrouter1"
        corrected = QUALSYNTH_CORRECTION_ROOT / dataset / method / f"seed{seed}.json"
        if corrected.exists():
            return corrected
    elif method in {"smote", "ctgan", "tabfairgdt"}:
        root = PROJECT_ROOT / "results/experiments1"
    elif method == "tabddpm":
        root = PROJECT_ROOT / "results/reviewer_revision/tabddpm_main"
    else:
        root = PROJECT_ROOT / "results/reviewer_revision/reviewer3_round2/no_augmentation"
    return root / dataset / method / f"seed{seed}.json"


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completeness_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for method in ALL_METHODS:
        for dataset in DATASETS:
            for seed in SEEDS:
                path = source_path(method, dataset, seed)
                completeness_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "path": str(path),
                        "exists": path.exists(),
                    }
                )
                if not path.exists():
                    missing.append(path)
                    continue
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not payload.get("success", False):
                    raise RuntimeError(f"Unsuccessful result in analysis set: {path}")
                avg = payload.get("avg_performance") or {}
                run_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "f1": avg.get("f1"),
                        "roc_auc": avg.get("roc_auc"),
                        "result_json": str(path),
                    }
                )
                by_classifier = payload.get("performance_metrics") or {}
                for classifier in CLASSIFIERS:
                    metrics = next(
                        (
                            by_classifier[key]
                            for key in CLASSIFIER_ALIASES[classifier]
                            if key in by_classifier
                        ),
                        None,
                    )
                    if not metrics:
                        raise RuntimeError(f"Missing {classifier} metrics: {path}")
                    classifier_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "classifier": classifier,
                            "f1": metrics.get("f1"),
                            "roc_auc": metrics.get("roc_auc"),
                            "result_json": str(path),
                        }
                    )
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} result files:\n{preview}")
    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(classifier_rows),
        pd.DataFrame(completeness_rows),
    )


def pivot_means(
    frame: pd.DataFrame,
    index: str | list[str],
    methods: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    pivot = frame.pivot_table(index=index, columns="method", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=list(methods)).dropna(axis=0, how="any")
    if pivot.shape[1] != len(methods):
        raise RuntimeError(f"Incomplete method columns for {metric}: {list(pivot.columns)}")
    return pivot


def mean_ranks(pivot: pd.DataFrame) -> pd.DataFrame:
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    return (
        ranks.mean(axis=0)
        .rename("mean_rank")
        .reset_index()
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def rank_biserial(x: Iterable[float], y: Iterable[float]) -> float | None:
    differences = np.asarray(list(x), dtype=float) - np.asarray(list(y), dtype=float)
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]
    if len(differences) == 0:
        return None
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (len(differences) * (len(differences) + 1) / 2.0))


def safe_wilcoxon(x: Iterable[float], y: Iterable[float]) -> tuple[float | None, float]:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    difference = x_array[mask] - y_array[mask]
    if np.allclose(difference, 0):
        return None, 1.0
    statistic, p_value = wilcoxon(
        x_array[mask],
        y_array[mask],
        alternative="two-sided",
        zero_method="wilcox",
    )
    return float(statistic), float(p_value)


def holm_adjust(frame: pd.DataFrame, p_column: str = "raw_p_value") -> pd.DataFrame:
    result = frame.copy()
    result["holm_p_value"] = np.nan
    valid = result[p_column].dropna().sort_values()
    running_max = 0.0
    total = len(valid)
    for rank, (index, p_value) in enumerate(valid.items()):
        adjusted = min(1.0, (total - rank) * float(p_value))
        running_max = max(running_max, adjusted)
        result.loc[index, "holm_p_value"] = running_max
    return result


def global_statistics(pivot: pd.DataFrame, metric: str, scope: str) -> tuple[dict, pd.DataFrame]:
    statistic, p_value = friedmanchisquare(*[pivot[column] for column in pivot.columns])
    rows = []
    for baseline in pivot.columns:
        if baseline == "qualsynth":
            continue
        test_statistic, pairwise_p = safe_wilcoxon(pivot["qualsynth"], pivot[baseline])
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "comparison": f"qualsynth_vs_{baseline}",
                "baseline": baseline,
                "n_pairs": len(pivot),
                "statistic": test_statistic,
                "raw_p_value": pairwise_p,
                "qualsynth_mean": float(pivot["qualsynth"].mean()),
                "baseline_mean": float(pivot[baseline].mean()),
                "mean_difference": float((pivot["qualsynth"] - pivot[baseline]).mean()),
                "rank_biserial": rank_biserial(pivot["qualsynth"], pivot[baseline]),
            }
        )
    pairwise = holm_adjust(pd.DataFrame(rows))
    omnibus = {
        "scope": scope,
        "metric": metric,
        "methods": list(pivot.columns),
        "n_blocks": len(pivot),
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
    }
    return omnibus, pairwise


def per_dataset_qualsynth_noaug(per_run: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        subset = per_run.loc[
            (per_run["dataset"] == dataset)
            & per_run["method"].isin(["qualsynth", "no_augmentation"])
        ]
        pivot = subset.pivot(index="seed", columns="method", values=metric).dropna()
        statistic, p_value = safe_wilcoxon(pivot["qualsynth"], pivot["no_augmentation"])
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n_seed_pairs": len(pivot),
                "qualsynth_mean": float(pivot["qualsynth"].mean()),
                "noaug_cw_mean": float(pivot["no_augmentation"].mean()),
                "mean_difference": float(
                    (pivot["qualsynth"] - pivot["no_augmentation"]).mean()
                ),
                "statistic": statistic,
                "raw_p_value": p_value,
                "rank_biserial": rank_biserial(
                    pivot["qualsynth"], pivot["no_augmentation"]
                ),
            }
        )
    return holm_adjust(pd.DataFrame(rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)
    return display.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run, per_classifier, completeness = load_rows()
    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    per_classifier.to_csv(output_dir / "per_classifier_metrics.csv", index=False)
    completeness.to_csv(output_dir / "completeness.csv", index=False)

    omnibus_rows = []
    global_pairwise_frames = []
    report_sections = []
    for metric in METRICS:
        dataset_pivot = pivot_means(per_run, "dataset", MAIN_METHODS, metric)
        classifier_pivot = pivot_means(
            per_classifier,
            ["dataset", "classifier"],
            MAIN_METHODS,
            metric,
        )
        dataset_pivot.reset_index().to_csv(
            output_dir / f"six_method_dataset_means_{metric}.csv", index=False
        )
        classifier_pivot.reset_index().to_csv(
            output_dir / f"six_method_dataset_classifier_blocks_{metric}.csv", index=False
        )
        dataset_ranks = mean_ranks(dataset_pivot)
        classifier_ranks = mean_ranks(classifier_pivot)
        dataset_ranks.to_csv(output_dir / f"six_method_dataset_mean_ranks_{metric}.csv", index=False)
        classifier_ranks.to_csv(
            output_dir / f"six_method_dataset_classifier_mean_ranks_{metric}.csv", index=False
        )

        for scope, pivot in (
            ("eight_dataset_means", dataset_pivot),
            ("24_dataset_classifier_blocks", classifier_pivot),
        ):
            omnibus, pairwise = global_statistics(pivot, metric, scope)
            omnibus_rows.append(omnibus)
            global_pairwise_frames.append(pairwise)

        paired = per_dataset_qualsynth_noaug(per_run, metric)
        paired.to_csv(output_dir / f"qualsynth_vs_noaug_cw_by_dataset_{metric}.csv", index=False)
        report_sections.append(
            (metric, dataset_pivot.reset_index(), dataset_ranks, classifier_ranks, paired)
        )

    omnibus_frame = pd.DataFrame(omnibus_rows)
    global_pairwise = pd.concat(global_pairwise_frames, ignore_index=True)
    omnibus_frame.to_csv(output_dir / "six_method_omnibus_tests.csv", index=False)
    global_pairwise.to_csv(output_dir / "six_method_qualsynth_pairwise_holm.csv", index=False)

    classifier_summary = (
        per_classifier.groupby(["method", "classifier"])[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    classifier_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in classifier_summary.columns
    ]
    classifier_summary.to_csv(output_dir / "classifier_resolved_summary.csv", index=False)

    overall_summary = (
        per_run.groupby("method")[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    overall_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in overall_summary.columns
    ]
    overall_summary.to_csv(output_dir / "all_condition_arithmetic_summary.csv", index=False)

    report_lines = [
        "# Reviewer 3 no-augmentation analysis",
        "",
        "This bundle adds **NoAug-CW** as the sixth main comparator and keeps "
        "**NoAug-Unweighted** supplementary. Existing five-method artifacts are not overwritten.",
        "",
        "## Completeness",
        "",
        f"- Result JSONs: {len(completeness)} / {len(completeness)} present.",
        f"- Main six-method runs: {len(DATASETS) * len(SEEDS) * len(MAIN_METHODS)}.",
        f"- New no-augmentation runs: {len(DATASETS) * len(SEEDS) * 2}.",
        f"- New classifier outcomes: {len(DATASETS) * len(SEEDS) * 2 * len(CLASSIFIERS)}.",
        "",
        "## Arithmetic summaries",
        "",
        markdown_table(overall_summary),
        "",
        "## Omnibus tests",
        "",
        markdown_table(omnibus_frame),
        "",
        "## QualSynth pairwise tests with Holm correction",
        "",
        markdown_table(global_pairwise),
    ]
    for metric, means, dataset_ranks, classifier_ranks, paired in report_sections:
        report_lines.extend(
            [
                "",
                f"## {metric.upper()}",
                "",
                "### Six-method dataset means",
                "",
                markdown_table(means),
                "",
                "### Mean ranks over eight dataset blocks",
                "",
                markdown_table(dataset_ranks),
                "",
                "### Mean ranks over 24 dataset-classifier blocks",
                "",
                markdown_table(classifier_ranks),
                "",
                "### Seed-paired QualSynth vs NoAug-CW by dataset",
                "",
                markdown_table(paired),
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    artifact_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "name": "reviewer3_round2_no_augmentation_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_methods": MAIN_METHODS,
        "supplementary_method": "no_augmentation_unweighted",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "source_roots": {
            "qualsynth": "results/openrouter1",
            "qualsynth_correction_overlay": (
                "results/reviewer_revision/canonical_dedup_correction/openrouter1"
            ),
            "smote_ctgan_tabfairgdt": "results/experiments1",
            "tabddpm": "results/reviewer_revision/tabddpm_main",
            "no_augmentation": "results/reviewer_revision/reviewer3_round2/no_augmentation",
        },
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
            if path.name != "manifest.json"
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote no-augmentation analysis bundle to {output_dir}")


if __name__ == "__main__":
    main()
