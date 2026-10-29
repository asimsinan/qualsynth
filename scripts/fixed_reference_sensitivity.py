#!/usr/bin/env python3
"""Fixed z-reference sensitivity analysis for QualSynth (Reviewer 5, point 6).

During generation the z-score mean and standard deviation are computed on a reference
that grows with every admitted row. This analysis re-checks QualSynth's stored rows
against a fixed reference, drops the rows that fail the z-test (theta = 4), retrains the
three classifiers, and recomputes the benchmark comparisons. Two references are tested:
  fold      the training fold alone (the main benchmark's initial reference)
  minority  the training-fold minority rows only (the stricter reference suggested)
Rows removed here would have been replaced during a real fixed-reference run, so affected
runs end below the balancing target; the analysis bounds the effect of the drifted rows on
utility rather than reproducing a fixed-reference generation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fold_safe_preprocessing_check as base  # noqa: E402
from src.qualsynth.data.splitting import encode_features, load_split  # noqa: E402
from src.qualsynth.evaluation.classifiers import ClassifierPipeline  # noqa: E402
from src.qualsynth.experiments.experiment_runner import ExperimentRunner  # noqa: E402
from src.qualsynth.validation.universal_validator import UniversalValidator  # noqa: E402

THETA = 4.0


def run_one(dataset, seed, reference):
    split = load_split(dataset, seed=seed, split_dir=str(ROOT / "data/splits"))
    X_train = split["X_train"].copy().reset_index(drop=True)
    X_test = split["X_test"].copy().reset_index(drop=True)
    y_train, _, y_test, _ = ExperimentRunner._normalize_binary_labels(
        pd.Series(split["y_train"]).reset_index(drop=True),
        pd.Series(split["y_val"]).reset_index(drop=True),
        pd.Series(split["y_test"]).reset_index(drop=True))
    columns = list(X_train.columns)
    path = base.first_existing([d / f"{dataset}_qualsynth_seed{seed}_validated_samples.csv"
                                for d in base.SAMPLE_DIRS["qualsynth"]])
    synthetic = pd.read_csv(path, float_precision="round_trip").drop(columns=["target"], errors="ignore")[columns]
    if split.get("preprocessor") is not None and (synthetic.dtypes == object).any():
        synthetic = encode_features(synthetic, split["preprocessor"])[columns]
    synthetic = synthetic.reset_index(drop=True)
    ref = X_train if reference == "fold" else X_train[(y_train == 1).values]
    tested = columns
    if reference == "minority_numeric":
        # Exclude label-encoded categorical flags, whose minority-class SD can be near zero.
        categorical = set(getattr(split.get("preprocessor"), "categorical_features", []) or [])
        tested = [c for c in columns if c not in categorical]
    validator = UniversalValidator(statistical_std_threshold=THETA, verbose=False)
    keep = validator._validate_statistical(synthetic[tested], ref[tested].reset_index(drop=True))
    kept = synthetic[keep].reset_index(drop=True)
    X_aug = pd.concat([X_train, kept], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1] * len(kept), name=y_train.name)], ignore_index=True)
    pipeline = ClassifierPipeline(random_state=seed, imbalance_policy="balanced")
    pipeline.train(X_aug, y_aug, verbose=False)
    avg = ExperimentRunner._average_metrics(
        pipeline.evaluate(X_test, y_test, compute_fairness=False, verbose=False))
    return {"dataset": dataset, "method": "qualsynth", "seed": seed, "f1": avg["f1"],
            "roc_auc": avg["roc_auc"], "stored_rows": len(synthetic), "kept_rows": len(kept)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", choices=["fold", "minority", "minority_numeric"], required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(delayed(run_one)(d, s, args.reference)
                                      for d in base.DATASETS for s in base.SEEDS)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    kept = frame.groupby("dataset")[["stored_rows", "kept_rows"]].sum()
    kept["kept_pct"] = (100 * kept.kept_rows / kept.stored_rows).round(1)
    print(kept.to_string())


if __name__ == "__main__":
    main()
