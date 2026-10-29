#!/usr/bin/env python3
"""Compute the benchmark statistics reported for the eight low-dimensional datasets.

Reads per-run metrics either from the CSV files written by evaluate_stored_runs.py
(--per-run-dir; this is how the reported values were produced) or from one result JSON per
(dataset, method, seed) in the run directories below, and writes, to
results/analysis/benchmark/:
  * per_run_metrics.csv and per_run_classifier_metrics.csv;
  * dataset_means.csv, dataset_sds.csv, and mean_ranks.csv;
  * statistics.json with Friedman tests and Holm-corrected Wilcoxon comparisons of
    QualSynth with each other method, using eight dataset means as blocks, for the five
    synthetic oversamplers and for the six strategies including NoAug-CW;
  * classifier_summary.csv (mean over datasets and seeds for each classifier);
  * per_dataset_seed_paired_tests.csv (QualSynth versus each comparator within each
    dataset, Holm-corrected across datasets).

Run directories (relative to the project root):
  results/qualsynth_main   QualSynth
  results/baselines        SMOTE, CTGAN, TabFairGDT
  results/tabddpm_main     TabDDPM
  results/no_augmentation  NoAug-CW (no_augmentation) and NoAug-Unweighted
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["breast_cancer", "german_credit", "haberman", "htru2", "pima_diabetes",
            "thyroid", "wine_quality", "yeast"]
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
SOURCES = {
    "qualsynth": "qualsynth_main",
    "smote": "baselines",
    "ctgan": "baselines",
    "tabfairgdt": "baselines",
    "tabddpm": "tabddpm_main",
    "no_augmentation": "no_augmentation",
    "no_augmentation_unweighted": "no_augmentation",
}
SYNTH = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]
SIX = SYNTH + ["no_augmentation"]
CLASSIFIERS = {"LogisticRegression": "logistic_regression", "RandomForest": "random_forest",
               "XGBoost": "xgboost"}
METRICS = ["f1", "roc_auc"]


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def load_runs(results_root):
    runs, classifier_rows = [], []
    for method, folder in SOURCES.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                path = results_root / folder / dataset / method / f"seed{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing run: {path}")
                payload = json.loads(path.read_text())
                perf = payload["avg_performance"]
                runs.append({"dataset": dataset, "method": method, "seed": seed,
                             "f1": perf["f1"], "roc_auc": perf["roc_auc"]})
                per_classifier = payload["performance_metrics"]
                for name, alias in CLASSIFIERS.items():
                    metrics = per_classifier.get(name, per_classifier.get(alias))
                    classifier_rows.append({"dataset": dataset, "method": method, "seed": seed,
                                            "classifier": name, "f1": metrics["f1"],
                                            "roc_auc": metrics["roc_auc"]})
    return pd.DataFrame(runs), pd.DataFrame(classifier_rows)


def dataset_level(per_run):
    means = per_run.groupby(["dataset", "method"])[METRICS].mean()
    report, ranks_out = {}, []
    for metric in METRICS:
        table = means[metric].unstack("method")
        report[metric] = {"overall_mean": table.mean().round(4).to_dict()}
        for name, methods in [("five_synthetic", SYNTH), ("six_strategies", SIX)]:
            sub = table[methods]
            ranks = sub.apply(lambda r: pd.Series(rankdata(-r.values), index=methods), axis=1)
            stat, p = friedmanchisquare(*[sub[m].values for m in methods])
            others = [m for m in methods if m != "qualsynth"]
            tests = [wilcoxon(sub["qualsynth"], sub[m]) for m in others]
            adjusted = holm([t.pvalue for t in tests])
            report[metric][name] = {
                "mean_ranks": ranks.mean().round(3).to_dict(),
                "friedman": {"statistic": float(stat), "p": float(p)},
                "qualsynth_versus": {
                    m: {"mean_difference": float((sub["qualsynth"] - sub[m]).mean()),
                        "W": float(t.statistic), "raw_p": float(t.pvalue), "holm_p": float(h)}
                    for m, t, h in zip(others, tests, adjusted)
                },
            }
            for method, value in ranks.mean().items():
                ranks_out.append({"metric": metric, "family": name, "method": method,
                                  "mean_rank": value})
    return report, pd.DataFrame(ranks_out)


def per_dataset_tests(per_run):
    rows = []
    for metric in METRICS:
        for method in [m for m in SIX if m != "qualsynth"]:
            block = []
            for dataset, frame in per_run.groupby("dataset"):
                wide = frame.pivot(index="seed", columns="method", values=metric)
                diff = wide["qualsynth"] - wide[method]
                p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(wide["qualsynth"], wide[method]).pvalue)
                block.append({"dataset": dataset, "metric": metric, "comparator": method,
                              "n_pairs": int(diff.notna().sum()),
                              "mean_difference": float(diff.mean()), "raw_p": p})
            for row, adj in zip(block, holm([r["raw_p"] for r in block])):
                row["holm_p"] = float(adj)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "analysis" / "benchmark")
    parser.add_argument("--per-run-dir", type=Path, default=None,
                        help="Directory with per_run_metrics.csv and per_run_classifier_metrics.csv "
                             "from evaluate_stored_runs.py")
    args = parser.parse_args()

    if args.per_run_dir is not None:
        per_run = pd.read_csv(args.per_run_dir / "per_run_metrics.csv")
        per_classifier = pd.read_csv(args.per_run_dir / "per_run_classifier_metrics.csv")
    else:
        per_run, per_classifier = load_runs(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output_dir / "per_run_metrics.csv", index=False)
    per_classifier.to_csv(args.output_dir / "per_run_classifier_metrics.csv", index=False)

    grouped = per_run.groupby(["dataset", "method"])[METRICS]
    grouped.mean().to_csv(args.output_dir / "dataset_means.csv")
    grouped.std().to_csv(args.output_dir / "dataset_sds.csv")
    report, ranks = dataset_level(per_run)
    ranks.to_csv(args.output_dir / "mean_ranks.csv", index=False)
    (args.output_dir / "statistics.json").write_text(json.dumps(report, indent=2))
    per_classifier.groupby(["classifier", "method"])[METRICS].mean().round(4).to_csv(
        args.output_dir / "classifier_summary.csv")
    per_dataset_tests(per_run).to_csv(args.output_dir / "per_dataset_seed_paired_tests.csv",
                                      index=False)

    for metric in METRICS:
        six = report[metric]["six_strategies"]
        print(f"{metric}: overall means {report[metric]['overall_mean']}")
        print(f"  Friedman (six strategies): chi2={six['friedman']['statistic']:.3f}, "
              f"p={six['friedman']['p']:.4f}")
        print("  Holm p, QualSynth versus:",
              {m: round(v["holm_p"], 4) for m, v in six["qualsynth_versus"].items()})
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
