#!/usr/bin/env python3
"""Fold-safe preprocessing sensitivity check for the main benchmark.

Mode "stored" retrains the downstream classifiers on the stored synthetic rows in the
archived feature representation and must reproduce the archived metrics exactly.
Mode "fold" standardises every non-categorical feature with a scaler fitted on the
training fold only (all eight datasets), re-expresses the stored synthetic rows of
QualSynth, CTGAN, TabFairGDT and TabDDPM in that space (an affine change, so their
generators need not be rerun), and regenerates SMOTE on the fold-standardised fold
because its nearest-neighbour search depends on feature scale.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qualsynth.data.splitting import encode_features, load_split  # noqa: E402
from src.qualsynth.evaluation.classifiers import ClassifierPipeline  # noqa: E402
from src.qualsynth.experiments.experiment_runner import ExperimentRunner  # noqa: E402

DATASETS = ["breast_cancer", "german_credit", "haberman", "htru2", "pima_diabetes",
            "thyroid", "wine_quality", "yeast"]
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
RES = ROOT / "results"
SAMPLE_DIRS = {
    "qualsynth": [RES / "qualsynth_main/logs"],
    "smote": [RES / "baselines/logs"],
    "ctgan": [RES / "baselines/logs"],
    "tabfairgdt": [RES / "baselines/logs"],
    "tabddpm": [RES / "tabddpm_main/logs"],
    "no_augmentation": [],
}
RESULT_DIRS = {
    "qualsynth": [RES / "qualsynth_main"],
    "smote": [RES / "baselines"], "ctgan": [RES / "baselines"],
    "tabfairgdt": [RES / "baselines"], "tabddpm": [RES / "tabddpm_main"],
    "no_augmentation": [RES / "no_augmentation"],
}


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[-1])


def stored_metrics(method, dataset, seed):
    import json
    path = first_existing([d / dataset / method / f"seed{seed}.json" for d in RESULT_DIRS[method]])
    perf = json.loads(path.read_text())["avg_performance"]
    return perf["f1"], perf["roc_auc"]


def run_one(method, dataset, seed, mode):
    split = load_split(dataset, seed=seed, split_dir=str(ROOT / "data/splits"))
    X_train = split["X_train"].copy().reset_index(drop=True)
    X_test = split["X_test"].copy().reset_index(drop=True)
    y_train, _, y_test, _ = ExperimentRunner._normalize_binary_labels(
        pd.Series(split["y_train"]).reset_index(drop=True),
        pd.Series(split["y_val"]).reset_index(drop=True),
        pd.Series(split["y_test"]).reset_index(drop=True),
    )
    columns = list(X_train.columns)
    preprocessor = split.get("preprocessor")
    categorical = set(getattr(preprocessor, "categorical_features", []) or [])
    numeric = [c for c in columns if c not in categorical]

    synthetic = None
    if method != "no_augmentation":
        path = first_existing([d / f"{dataset}_{method}_seed{seed}_validated_samples.csv"
                               for d in SAMPLE_DIRS[method]])
        synthetic = pd.read_csv(path, float_precision="round_trip").drop(columns=["target"], errors="ignore")[columns]
        if preprocessor is not None and (synthetic.dtypes == object).any():
            # QualSynth's German Credit rows are stored in raw form; the runner encoded
            # them with the split preprocessor before classifier training.
            synthetic = encode_features(synthetic, preprocessor)[columns]

    if mode == "fold":
        mean = X_train[numeric].astype(float).mean()
        std = X_train[numeric].astype(float).std(ddof=0).replace(0, 1.0)
        scale = lambda frame: frame.assign(**{c: (frame[c].astype(float) - mean[c]) / std[c] for c in numeric})
        X_train, X_test = scale(X_train), scale(X_test)
        if method == "smote":
            from src.qualsynth.baselines.smote import SMOTEBaseline
            target = ExperimentRunner._target_synthetic_count(y_train)
            X_res, _ = SMOTEBaseline(k_neighbors=5, sampling_strategy="auto", random_state=seed
                                     ).fit_resample(X_train, y_train, n_samples=target)
            synthetic = X_res.iloc[len(X_train):].reset_index(drop=True)
        elif synthetic is not None:
            synthetic = scale(synthetic)

    if synthetic is not None and len(synthetic):
        X_aug = pd.concat([X_train, synthetic], ignore_index=True)
        y_aug = pd.concat([y_train, pd.Series([1] * len(synthetic), name=y_train.name)], ignore_index=True)
    else:
        X_aug, y_aug = X_train, y_train
    pipeline = ClassifierPipeline(random_state=seed, imbalance_policy="balanced")
    pipeline.train(X_aug, y_aug, verbose=False)
    metrics = pipeline.evaluate(X_test, y_test, compute_fairness=False, verbose=False)
    avg = ExperimentRunner._average_metrics(metrics)
    f1_stored, auc_stored = stored_metrics(method, dataset, seed)
    return {"dataset": dataset, "method": method, "seed": seed, "f1": avg["f1"], "roc_auc": avg["roc_auc"],
            "f1_stored": f1_stored, "roc_auc_stored": auc_stored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stored", "fold"], required=True)
    parser.add_argument("--methods", nargs="+", default=list(SAMPLE_DIRS))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [(m, d, s) for m in args.methods for d in args.datasets for s in args.seeds]
    rows = Parallel(n_jobs=args.jobs)(delayed(run_one)(m, d, s, args.mode) for m, d, s in tasks)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    frame["df1"] = (frame.f1 - frame.f1_stored).abs()
    frame["dauc"] = (frame.roc_auc - frame.roc_auc_stored).abs()
    print(frame.groupby(["method"])[["df1", "dauc"]].max().round(6).to_string())


if __name__ == "__main__":
    main()
