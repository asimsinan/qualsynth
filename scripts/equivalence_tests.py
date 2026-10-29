#!/usr/bin/env python3
"""Paired TOST equivalence tests for QualSynth versus each comparator.

Uses the eight dataset-mean blocks of the benchmark (per_run_metrics.csv written by
analyze_benchmark.py). For each comparator and margin delta, the two one-sided paired
t-tests are H0: mean(QualSynth - other) <= -delta and H0: mean(QualSynth - other) >= +delta;
equivalence holds when both reject at alpha = 0.05, which is equivalent to the 90%
confidence interval of the mean difference lying inside [-delta, +delta]. Holm correction
is applied across the five comparators within each metric and margin.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = ["smote", "ctgan", "tabfairgdt", "tabddpm", "no_augmentation"]
MARGINS = {"f1": [0.01, 0.02, 0.03], "roc_auc": [0.005, 0.01, 0.02]}


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def tost(diff, delta):
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    p_lower = 1 - stats.t.cdf((mean + delta) / se, df=n - 1)
    p_upper = stats.t.cdf((mean - delta) / se, df=n - 1)
    half = stats.t.ppf(0.95, df=n - 1) * se
    return mean, mean - half, mean + half, max(p_lower, p_upper)


def main():
    default_dir = PROJECT_ROOT / "results" / "analysis" / "benchmark"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-run", type=Path, default=default_dir / "per_run_metrics.csv")
    parser.add_argument("--output", type=Path, default=default_dir / "equivalence_tost.csv")
    args = parser.parse_args()

    per_run = pd.read_csv(args.per_run)
    means = per_run.groupby(["dataset", "method"])[["f1", "roc_auc"]].mean()
    rows = []
    for metric, margins in MARGINS.items():
        table = means[metric].unstack("method")
        for delta in margins:
            block = []
            for other in COMPARATORS:
                diff = (table["qualsynth"] - table[other]).values
                mean, low, high, p = tost(diff, delta)
                block.append({"metric": metric, "margin": delta, "comparator": other,
                              "mean_difference": mean, "ci90_low": low, "ci90_high": high,
                              "tost_p": p})
            for row, adj in zip(block, holm([r["tost_p"] for r in block])):
                row["tost_p_holm"] = float(adj)
                row["equivalent_holm"] = bool(adj < 0.05)
                rows.append(row)
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
