#!/usr/bin/env python3
"""Summarize and rank QualSynth ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ablation seed JSON files and rank variants."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Ablation output directory containing dataset/variant/seed*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <input-dir>/analysis.",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        help="Primary performance metric for ranking (default: f1).",
    )
    parser.add_argument(
        "--secondary-metric",
        default="roc_auc",
        help="Secondary performance metric to report (default: roc_auc).",
    )
    parser.add_argument(
        "--fairness-metric",
        default="demographic_parity_difference",
        help="Fairness metric to report when available (default: demographic_parity_difference).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_or_nan(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean)) if clean else float("nan")


def build_records(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_rows: List[Dict[str, Any]] = []
    clf_rows: List[Dict[str, Any]] = []

    for json_path in sorted(input_dir.glob("*/*/seed*.json")):
        payload = load_json(json_path)
        if not payload.get("success"):
            continue

        dataset = payload["dataset"]
        variant = payload["method"]
        seed = payload["seed"]
        avg_performance = payload.get("avg_performance") or {}
        avg_fairness = payload.get("avg_fairness") or {}
        metadata = payload.get("metadata") or {}

        avg_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "execution_time": payload.get("execution_time"),
                "n_generated": payload.get("n_generated"),
                "generation_time": payload.get("generation_time"),
                "generation_cost": payload.get("generation_cost"),
                "validation_rate": metadata.get("validation_rate"),
                "converged": metadata.get("converged"),
                "target_samples": metadata.get("target_samples"),
                "target_reached": metadata.get("target_reached"),
                "target_shortfall": metadata.get("target_shortfall"),
                "iterations": metadata.get("iterations"),
                **{f"avg_{k}": v for k, v in avg_performance.items()},
                **{f"fair_{k}": v for k, v in avg_fairness.items()},
            }
        )

        performance_metrics = payload.get("performance_metrics") or {}
        fairness_metrics = payload.get("fairness_metrics") or {}
        for classifier, metrics in performance_metrics.items():
            fairness = fairness_metrics.get(classifier) or {}
            clf_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "classifier": classifier,
                    **metrics,
                    **{f"fair_{k}": v for k, v in fairness.items()},
                }
            )

    return pd.DataFrame(avg_rows), pd.DataFrame(clf_rows)


def compute_avg_rank(df: pd.DataFrame, value_col: str, group_cols: List[str]) -> pd.DataFrame:
    work = df.dropna(subset=[value_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["variant", "avg_rank"])

    work["rank"] = work.groupby(group_cols)[value_col].rank(ascending=False, method="average")
    return (
        work.groupby("variant", as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "avg_rank"})
        .sort_values(["avg_rank", "variant"], kind="stable")
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    avg_df, clf_df = build_records(input_dir)
    if avg_df.empty:
        raise SystemExit(f"No successful seed JSON files found in {input_dir}")

    avg_metric_col = f"avg_{args.metric}"
    avg_secondary_col = f"avg_{args.secondary_metric}"
    avg_fairness_col = f"fair_{args.fairness_metric}"
    if avg_fairness_col not in avg_df.columns:
        avg_df[avg_fairness_col] = np.nan

    variant_avg = (
        avg_df.groupby("variant", as_index=False)
        .agg(
            runs=("seed", "count"),
            datasets=("dataset", "nunique"),
            seeds=("seed", "nunique"),
            mean_metric=(avg_metric_col, "mean"),
            std_metric=(avg_metric_col, "std"),
            mean_secondary=(avg_secondary_col, "mean"),
            mean_fairness=(avg_fairness_col, "mean"),
            mean_generated=("n_generated", "mean"),
            mean_validation_rate=("validation_rate", "mean"),
            mean_execution_time=("execution_time", "mean"),
        )
    )
    variant_avg["std_metric"] = variant_avg["std_metric"].fillna(0.0)
    variant_avg = variant_avg.sort_values(
        ["mean_metric", "mean_secondary", "variant"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    variant_avg["rank_by_avg_metric"] = np.arange(1, len(variant_avg) + 1)

    avg_rank_df = compute_avg_rank(avg_df, avg_metric_col, ["dataset", "seed"])
    if not avg_rank_df.empty:
        variant_avg = variant_avg.merge(avg_rank_df, on="variant", how="left")

    dataset_variant_summary = (
        avg_df.groupby(["dataset", "variant"], as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_metric=(avg_metric_col, "mean"),
            std_metric=(avg_metric_col, "std"),
            mean_secondary=(avg_secondary_col, "mean"),
            mean_fairness=(avg_fairness_col, "mean"),
        )
        .sort_values(["dataset", "mean_metric"], ascending=[True, False], kind="stable")
    )
    dataset_variant_summary["std_metric"] = dataset_variant_summary["std_metric"].fillna(0.0)

    if clf_df.empty:
        classifier_variant_summary = pd.DataFrame()
    else:
        clf_fairness_col = f"fair_{args.fairness_metric}"
        if clf_fairness_col not in clf_df.columns:
            clf_df[clf_fairness_col] = np.nan
        classifier_variant_summary = (
            clf_df.groupby(["variant", "classifier"], as_index=False)
            .agg(
                runs=("seed", "count"),
                mean_metric=(args.metric, "mean"),
                std_metric=(args.metric, "std"),
                mean_secondary=(args.secondary_metric, "mean"),
                mean_fairness=(clf_fairness_col, "mean"),
            )
            .sort_values(["classifier", "mean_metric"], ascending=[True, False], kind="stable")
        )
        classifier_variant_summary["std_metric"] = classifier_variant_summary["std_metric"].fillna(0.0)

        classifier_rank_df = compute_avg_rank(clf_df, args.metric, ["dataset", "seed", "classifier"])
        if not classifier_rank_df.empty:
            classifier_variant_summary = classifier_variant_summary.merge(
                classifier_rank_df, on="variant", how="left"
            )

    avg_df.to_csv(output_dir / "per_run_avg_metrics.csv", index=False)
    clf_df.to_csv(output_dir / "per_run_classifier_metrics.csv", index=False)
    variant_avg.to_csv(output_dir / "variant_ranking_by_avg_metric.csv", index=False)
    dataset_variant_summary.to_csv(output_dir / "dataset_variant_summary.csv", index=False)
    if not classifier_variant_summary.empty:
        classifier_variant_summary.to_csv(
            output_dir / "classifier_variant_summary.csv", index=False
        )

    best = variant_avg.iloc[0]
    print("=" * 88)
    print("Ablation Summary")
    print("=" * 88)
    print(f"Input directory: {input_dir}")
    print(f"Primary metric:  {args.metric}")
    print(f"Secondary:       {args.secondary_metric}")
    print(f"Best variant:    {best['variant']}")
    print(f"Mean {args.metric}:      {best['mean_metric']:.4f}")
    if not pd.isna(best["mean_secondary"]):
        print(f"Mean {args.secondary_metric}: {best['mean_secondary']:.4f}")
    if not pd.isna(best["mean_fairness"]):
        print(f"Mean fairness ({args.fairness_metric}): {best['mean_fairness']:.4f}")
    if "avg_rank" in variant_avg.columns and not pd.isna(best.get("avg_rank")):
        print(f"Average rank:    {best['avg_rank']:.3f}")
    print(f"Analysis dir:    {output_dir}")
    print("-" * 88)
    print(variant_avg[["variant", "mean_metric", "mean_secondary", "mean_fairness"]].head(10).to_string(index=False))
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
