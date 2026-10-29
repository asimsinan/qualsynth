#!/usr/bin/env python3
"""Evaluate every main-benchmark run from its stored synthetic rows in one pipeline.

For each method, dataset and seed, the training fold is augmented with the run's stored
synthetic rows and the three downstream classifiers are trained and evaluated on the
test fold with the current software environment. All methods, including QualSynth,
whose runs were originally evaluated with earlier library versions, are therefore scored
by the same code. Writes per_run_metrics.csv and per_run_classifier_metrics.csv in the
format read by analyze_benchmark.py (--per-run-dir).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fold_safe_preprocessing_check as base  # noqa: E402
from src.qualsynth.data.splitting import encode_features, load_split  # noqa: E402
from src.qualsynth.evaluation.classifiers import ClassifierPipeline  # noqa: E402
from src.qualsynth.experiments.experiment_runner import ExperimentRunner  # noqa: E402

METHODS = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm", "no_augmentation",
           "no_augmentation_unweighted"]
CLASSIFIERS = {"LogisticRegression": "LogisticRegression", "RandomForest": "RandomForest",
               "XGBoost": "XGBoost"}


def run_one(method, dataset, seed):
    split = load_split(dataset, seed=seed, split_dir=str(ROOT / "data/splits"))
    X_train = split["X_train"].copy().reset_index(drop=True)
    X_test = split["X_test"].copy().reset_index(drop=True)
    y_train, _, y_test, _ = ExperimentRunner._normalize_binary_labels(
        pd.Series(split["y_train"]).reset_index(drop=True),
        pd.Series(split["y_val"]).reset_index(drop=True),
        pd.Series(split["y_test"]).reset_index(drop=True))
    columns = list(X_train.columns)
    X_aug, y_aug = X_train, y_train
    if not method.startswith("no_augmentation"):
        path = base.first_existing([d / f"{dataset}_{method}_seed{seed}_validated_samples.csv"
                                    for d in base.SAMPLE_DIRS[method]])
        synthetic = pd.read_csv(path, float_precision="round_trip").drop(columns=["target"], errors="ignore")[columns]
        if split.get("preprocessor") is not None and (synthetic.dtypes == object).any():
            synthetic = encode_features(synthetic, split["preprocessor"])[columns]
        if len(synthetic):
            X_aug = pd.concat([X_train, synthetic], ignore_index=True)
            y_aug = pd.concat([y_train, pd.Series([1] * len(synthetic), name=y_train.name)], ignore_index=True)
    policy = "none" if method == "no_augmentation_unweighted" else "balanced"
    pipeline = ClassifierPipeline(random_state=seed, imbalance_policy=policy)
    pipeline.train(X_aug, y_aug, verbose=False)
    metrics = pipeline.evaluate(X_test, y_test, compute_fairness=False, verbose=False)
    avg = ExperimentRunner._average_metrics(metrics)
    run = {"dataset": dataset, "method": method, "seed": seed, "f1": avg["f1"], "roc_auc": avg["roc_auc"]}
    clf = [{"dataset": dataset, "method": method, "seed": seed, "classifier": name,
            "f1": float(metrics[name]["f1"]), "roc_auc": float(metrics[name]["roc_auc"])} for name in CLASSIFIERS]
    return run, clf


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "analysis" / "evaluation")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    results = Parallel(n_jobs=args.jobs)(delayed(run_one)(m, d, s)
                                         for m in METHODS for d in base.DATASETS for s in base.SEEDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r, _ in results]).to_csv(args.output_dir / "per_run_metrics.csv", index=False)
    pd.DataFrame([c for _, cl in results for c in cl]).to_csv(args.output_dir / "per_run_classifier_metrics.csv", index=False)
    print(pd.DataFrame([r for r, _ in results]).groupby("method")[["f1", "roc_auc"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
