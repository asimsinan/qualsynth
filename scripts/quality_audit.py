#!/usr/bin/env python3
"""Candidate-quality audit of the synthetic rows each method added to the training fold.

For every method, dataset, and seed, the stored synthetic rows are assessed with one
fixed diagnostic:
  * acceptance: share of rows kept by the QualSynth admission checks applied post hoc
    (standard mode, theta = 4.0, training-fold reference);
  * exact-duplicate rate of the generated rows;
  * mean absolute correlation difference (Delta_rho) between the stored rows and the
    minority class of the training fold.
Writes quality_audit_rows.csv and quality_audit_summary.csv to results/analysis/quality/.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.data.splitting import encode_features, load_split  # noqa: E402
from src.qualsynth.validation.universal_validator import UniversalValidator  # noqa: E402

DATASETS = ["breast_cancer", "german_credit", "haberman", "htru2", "pima_diabetes",
            "thyroid", "wine_quality", "yeast"]
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
LOG_DIRS = {
    "qualsynth": "qualsynth_main/logs",
    "smote": "baselines/logs",
    "ctgan": "baselines/logs",
    "tabfairgdt": "baselines/logs",
    "tabddpm": "tabddpm_main/logs",
}


def prepare_numeric_frame(dataset, seed, sample_frame):
    """Express sample rows and the training-fold minority in the same numeric space."""
    split_raw = load_split(dataset, seed=seed, return_raw=True)
    split_encoded = load_split(dataset, seed=seed, return_raw=False)
    y_train = split_raw["y_train"]
    minority_label = y_train.value_counts().idxmin()
    train_minority = split_encoded["X_train"].loc[y_train == minority_label].reset_index(drop=True)
    preprocessor = split_raw.get("preprocessor")

    features = sample_frame.drop(columns=["target"], errors="ignore").copy()
    if any(dtype == "object" for dtype in features.dtypes):
        samples = encode_features(features, preprocessor)
    else:
        samples = features.apply(pd.to_numeric, errors="coerce")
    for column in train_minority.columns:
        if column not in samples.columns:
            samples[column] = train_minority[column].median()
    samples = samples[train_minority.columns]

    train_numeric = train_minority.apply(pd.to_numeric, errors="coerce")
    train_numeric = train_numeric.fillna(train_numeric.median()).fillna(0)
    samples = samples.apply(pd.to_numeric, errors="coerce").fillna(train_numeric.median()).fillna(0)
    return train_numeric, samples


def correlation_difference(train_df, sample_df) -> Optional[float]:
    if len(train_df) < 3 or len(sample_df) < 3 or train_df.shape[1] < 2:
        return None
    corr_train = train_df.corr(numeric_only=True).fillna(0)
    corr_sample = sample_df.corr(numeric_only=True).fillna(0)
    upper = np.triu_indices_from(corr_train, k=1)
    return float(np.mean(np.abs(corr_train.values[upper] - corr_sample.values[upper])))


def acceptance(samples, X_train, y_train):
    validator = UniversalValidator(statistical_std_threshold=4.0, verbose=False,
                                   validation_mode="standard")
    X = samples[list(X_train.columns)].reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        result = validator.validate_and_select(X, pd.Series([1] * len(X)), X_train, y_train)
    return len(result.X_validated) / len(X) if len(X) else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "analysis" / "quality")
    args = parser.parse_args()

    rows = []
    for method, log_dir in LOG_DIRS.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                stem = args.results_root / log_dir / f"{dataset}_{method}_seed{seed}"
                generated = pd.read_csv(f"{stem}_generated_samples.csv")
                stored = pd.read_csv(f"{stem}_validated_samples.csv")
                split = load_split(dataset, seed=seed)
                X_train = split["X_train"].reset_index(drop=True)
                y_train = pd.Series(split["y_train"]).reset_index(drop=True)
                train_numeric, stored_numeric = prepare_numeric_frame(dataset, seed, stored)
                rows.append({
                    "dataset": dataset, "method": method, "seed": seed,
                    "n_stored": len(stored),
                    "acceptance": acceptance(stored_numeric, X_train, y_train),
                    "exact_duplicate_rate": float(generated.duplicated().mean()),
                    "correlation_difference": correlation_difference(train_numeric, stored_numeric),
                })
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "quality_audit_rows.csv", index=False)
    summary = frame.groupby("method").agg(
        n_runs=("seed", "count"),
        acceptance_pct=("acceptance", lambda s: 100 * s.mean()),
        exact_duplicate_pct=("exact_duplicate_rate", lambda s: 100 * s.mean()),
        correlation_difference=("correlation_difference", "mean"),
    )
    summary.to_csv(args.output_dir / "quality_audit_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
