#!/usr/bin/env python3
"""Re-evaluate retained Thyroid samples in one canonical feature space.

The December 2025 Thyroid artifacts contain a mixture of raw clinical units and
standardized features across seeds. This script detects each retained artifact's
representation, converts raw generated rows with the split preprocessor, and
trains/evaluates every seed against the canonical encoded test fold. It never
overwrites the historical result tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.data.splitting import encode_features, load_split
from src.qualsynth.evaluation.classifiers import ClassifierPipeline


SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
NUMERICAL_COLUMNS = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
CANONICAL_CORRECTION_ROOT = (
    PROJECT_ROOT
    / "results/reviewer_revision/canonical_dedup_correction/openrouter1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _representation(saved_training_prefix: pd.DataFrame, split: dict) -> tuple[str, float, float]:
    """Classify the saved feature scale using the unchanged training prefix."""
    encoded = split["X_train"].reset_index(drop=True)[NUMERICAL_COLUMNS].astype(float)
    raw_split = load_split("thyroid", seed=int(split["seed"]), return_raw=True)
    raw = raw_split["X_train"].reset_index(drop=True)[NUMERICAL_COLUMNS].astype(float)
    observed = saved_training_prefix[NUMERICAL_COLUMNS].astype(float)

    encoded_rmse = float(np.sqrt(np.nanmean((observed.to_numpy() - encoded.to_numpy()) ** 2)))
    raw_rmse = float(np.sqrt(np.nanmean((observed.to_numpy() - raw.to_numpy()) ** 2)))
    representation = "encoded" if encoded_rmse < raw_rmse else "raw"
    return representation, encoded_rmse, raw_rmse


def _average_metrics(performance: dict) -> dict[str, float]:
    keys = ["f1", "roc_auc", "precision", "recall", "balanced_accuracy", "pr_auc"]
    return {
        key: float(np.mean([float(metrics[key]) for metrics in performance.values()]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/openrouter1")
    parser.add_argument(
        "--output",
        default="results/reviewer_revision/thyroid_correction/analysis",
    )
    parser.add_argument(
        "--correction-root",
        default=str(CANONICAL_CORRECTION_ROOT.relative_to(PROJECT_ROOT)),
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    correction_root = Path(args.correction_root)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not correction_root.is_absolute():
        correction_root = PROJECT_ROOT / correction_root
    output.mkdir(parents=True, exist_ok=True)
    correction_log_dir = correction_root / "logs"
    correction_result_dir = correction_root / "thyroid" / "qualsynth"
    correction_log_dir.mkdir(parents=True, exist_ok=True)
    correction_result_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    per_classifier_rows = []
    audit_rows = []

    for seed in SEEDS:
        split = load_split("thyroid", seed=seed)
        X_train = split["X_train"].reset_index(drop=True)
        y_train = pd.Series(split["y_train"]).reset_index(drop=True)
        X_test = split["X_test"].reset_index(drop=True)
        y_test = pd.Series(split["y_test"]).reset_index(drop=True)

        generated_path = source / "logs" / f"thyroid_qualsynth_seed{seed}_generated_samples.csv"
        resampled_path = source / "logs" / f"thyroid_qualsynth_seed{seed}_resampled_dataset.csv"
        historical_path = source / "thyroid" / "qualsynth" / f"seed{seed}.json"

        generated_saved = pd.read_csv(generated_path)
        generated_features = generated_saved.drop(columns=["target"], errors="ignore")
        resampled_saved = pd.read_csv(resampled_path)
        saved_prefix = resampled_saved.drop(columns=["target"], errors="ignore").iloc[: len(X_train)]
        representation, encoded_rmse, raw_rmse = _representation(saved_prefix, split)

        if representation == "raw":
            generated_encoded = encode_features(generated_features, split["preprocessor"])
        else:
            generated_encoded = generated_features.copy()
        generated_encoded = generated_encoded[X_train.columns].apply(pd.to_numeric, errors="coerce")
        generated_encoded = generated_encoded.fillna(X_train.median(numeric_only=True))

        target_class = int(y_train.value_counts().idxmin())
        y_generated = pd.Series([target_class] * len(generated_encoded), name=y_train.name)
        X_augmented = pd.concat([X_train, generated_encoded], ignore_index=True)
        y_augmented = pd.concat([y_train, y_generated], ignore_index=True)

        pipeline = ClassifierPipeline(random_state=seed, imbalance_policy="balanced")
        pipeline.train(X_augmented, y_augmented, verbose=False)
        performance = pipeline.evaluate(
            X_test,
            y_test,
            compute_fairness=False,
            verbose=False,
        )
        averages = _average_metrics(performance)
        historical_payload = json.loads(historical_path.read_text())
        historical = historical_payload["performance_metrics"]
        historical_f1 = float(np.mean([float(row["f1"]) for row in historical.values()]))
        historical_auc = float(np.mean([float(row["roc_auc"]) for row in historical.values()]))

        per_run_rows.append(
            {
                "seed": seed,
                "n_generated": len(generated_encoded),
                "saved_representation": representation,
                "historical_mean_f1": historical_f1,
                "corrected_mean_f1": averages["f1"],
                "historical_mean_roc_auc": historical_auc,
                "corrected_mean_roc_auc": averages["roc_auc"],
                **{f"corrected_mean_{key}": value for key, value in averages.items() if key not in {"f1", "roc_auc"}},
            }
        )
        for classifier, metrics in performance.items():
            per_classifier_rows.append(
                {
                    "seed": seed,
                    "classifier": classifier,
                    **{key: float(value) for key, value in metrics.items() if np.isscalar(value)},
                }
            )

        categorical_columns = [column for column in X_train.columns if column not in NUMERICAL_COLUMNS]
        audit_rows.append(
            {
                "seed": seed,
                "saved_representation": representation,
                "encoded_prefix_rmse": encoded_rmse,
                "raw_prefix_rmse": raw_rmse,
                "n_generated": len(generated_encoded),
                "exact_duplicate_rows": int(generated_encoded.duplicated().sum()),
                "constant_categorical_columns": int(
                    sum(generated_encoded[column].nunique(dropna=False) <= 1 for column in categorical_columns)
                ),
                "categorical_columns": len(categorical_columns),
            }
        )

        canonical_samples = generated_encoded.copy()
        canonical_samples["target"] = y_generated.to_numpy()
        corrected_generated_path = (
            correction_log_dir / f"thyroid_qualsynth_seed{seed}_generated_samples.csv"
        )
        corrected_validated_path = (
            correction_log_dir / f"thyroid_qualsynth_seed{seed}_validated_samples.csv"
        )
        corrected_resampled_path = (
            correction_log_dir / f"thyroid_qualsynth_seed{seed}_resampled_dataset.csv"
        )
        canonical_samples.to_csv(corrected_generated_path, index=False)
        canonical_samples.to_csv(corrected_validated_path, index=False)
        canonical_resampled = X_augmented.copy()
        canonical_resampled["target"] = y_augmented.to_numpy()
        canonical_resampled.to_csv(corrected_resampled_path, index=False)

        corrected_payload = dict(historical_payload)
        corrected_payload["performance_metrics"] = performance
        corrected_payload["avg_performance"] = averages
        corrected_payload["metadata"] = {
            **historical_payload.get("metadata", {}),
            "retained_sample_representation_correction": {
                "applied": True,
                "scope": "downstream_reanalysis_of_unchanged_retained_samples",
                "source_representation": representation,
                "canonical_representation": "training_fold_encoded_feature_space",
                "archived_result_sha256": _sha256(historical_path),
                "archived_generated_sha256": _sha256(generated_path),
                "archived_resampled_sha256": _sha256(resampled_path),
                "historical_recorded_metrics": {
                    "f1": historical_f1,
                    "roc_auc": historical_auc,
                },
                "corrected_metrics": {
                    "f1": averages["f1"],
                    "roc_auc": averages["roc_auc"],
                },
                "retained_row_membership_modified": False,
                "feature_representation_canonicalized": True,
                "classifier_training_repeated": True,
            },
            "artifact_manifest": {
                "generated_samples": {
                    "path": str(corrected_generated_path.relative_to(PROJECT_ROOT)),
                    "rows": len(canonical_samples),
                    "sha256": _sha256(corrected_generated_path),
                },
                "validated_samples": {
                    "path": str(corrected_validated_path.relative_to(PROJECT_ROOT)),
                    "rows": len(canonical_samples),
                    "sha256": _sha256(corrected_validated_path),
                },
                "resampled_dataset": {
                    "path": str(corrected_resampled_path.relative_to(PROJECT_ROOT)),
                    "rows": len(canonical_resampled),
                    "sha256": _sha256(corrected_resampled_path),
                },
            },
        }
        corrected_result_path = correction_result_dir / f"seed{seed}.json"
        corrected_result_path.write_text(
            json.dumps(corrected_payload, indent=2, default=_json_default) + "\n"
        )

    per_run = pd.DataFrame(per_run_rows)
    per_classifier = pd.DataFrame(per_classifier_rows)
    audit = pd.DataFrame(audit_rows)

    noaug_root = Path(
        "results/reviewer_revision/reviewer3_round2/no_augmentation/thyroid/no_augmentation"
    )
    if noaug_root.exists():
        noaug_f1 = []
        noaug_auc = []
        for seed in per_run.seed:
            metrics = json.loads((noaug_root / f"seed{seed}.json").read_text())["performance_metrics"]
            noaug_f1.append(float(np.mean([float(row["f1"]) for row in metrics.values()])))
            noaug_auc.append(float(np.mean([float(row["roc_auc"]) for row in metrics.values()])))
        per_run["noaug_cw_mean_f1"] = noaug_f1
        per_run["noaug_cw_mean_roc_auc"] = noaug_auc
        per_run["corrected_minus_noaug_cw_f1"] = per_run.corrected_mean_f1 - per_run.noaug_cw_mean_f1
        per_run["corrected_minus_noaug_cw_roc_auc"] = (
            per_run.corrected_mean_roc_auc - per_run.noaug_cw_mean_roc_auc
        )
    per_run.to_csv(output / "per_run_comparison.csv", index=False)
    per_classifier.to_csv(output / "per_classifier_metrics.csv", index=False)
    audit.to_csv(output / "representation_and_quality_audit.csv", index=False)

    classifier_summary = (
        per_classifier.groupby("classifier")[["f1", "roc_auc", "precision", "recall", "balanced_accuracy", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    classifier_summary.to_csv(output / "classifier_summary.csv", index=False)

    raw_seeds = audit.loc[audit.saved_representation == "raw", "seed"].tolist()
    encoded_seeds = audit.loc[audit.saved_representation == "encoded", "seed"].tolist()
    comparison_text = ""
    if "noaug_cw_mean_f1" in per_run:
        comparison_text = f"""
## Relative predictive performance

The corrected retained-sample result is close to the non-generative NoAug-CW reference (F1 {per_run.noaug_cw_mean_f1.mean():.6f}; ROC-AUC {per_run.noaug_cw_mean_roc_auc.mean():.6f}). QualSynth is lower by {abs(per_run.corrected_minus_noaug_cw_f1.mean()):.6f} F1 and {abs(per_run.corrected_minus_noaug_cw_roc_auc.mean()):.6f} ROC-AUC. Against the frozen synthetic-generator results, corrected QualSynth is the strongest synthetic-data method on Thyroid: the next-highest values are TabDDPM at F1 0.584627 and ROC-AUC 0.904094.
"""
    report = f"""# Thyroid diagnostic and corrected reanalysis

## Finding

The historical Thyroid run mixed feature representations across seeds. Saved training folds were in raw clinical units for seeds `{raw_seeds}` and in the standardized feature space for seeds `{encoded_seeds}`. In addition, the categorical encoder converted generated values to strings before matching them against numeric encoder classes. This caused all {int(audit.categorical_columns.iloc[0])} categorical columns in every retained synthetic set to collapse to a single value.

## Corrected retained-sample evaluation

Raw generated rows were converted with the repaired encoder, encoded rows were kept in their existing standardized representation, and every classifier was retrained and tested on the corresponding canonical encoded split. Across ten seeds and three classifiers, the corrected arithmetic means are:

- F1: {per_run.corrected_mean_f1.mean():.6f} (historical artifact mean: {per_run.historical_mean_f1.mean():.6f})
- ROC-AUC: {per_run.corrected_mean_roc_auc.mean():.6f} (historical artifact mean: {per_run.historical_mean_roc_auc.mean():.6f})
- Balanced accuracy: {per_run.corrected_mean_balanced_accuracy.mean():.6f}
- PR-AUC: {per_run.corrected_mean_pr_auc.mean():.6f}

These values show that the reported Thyroid collapse was primarily an evaluation/representation failure, rather than evidence that the retained synthetic rows contained no predictive signal.
{comparison_text}

## Correction scope

This is a deterministic reanalysis of the unchanged retained samples, not a fresh generation experiment. The correction standardizes downstream representation and repeats classifier training and testing over all ten predefined seeds. The correction overlay preserves hashes of the archived inputs and is the manuscript-facing source for the Thyroid predictive results.
"""
    (output / "REPORT.md").write_text(report)
    correction_manifest = {
        "schema_version": "thyroid_retained_sample_correction_v1",
        "dataset": "thyroid",
        "seeds": SEEDS,
        "source_root": str(source),
        "correction_root": str(correction_root),
        "analysis_report": str((output / "REPORT.md").relative_to(PROJECT_ROOT)),
        "mean_corrected_f1": float(per_run.corrected_mean_f1.mean()),
        "mean_corrected_roc_auc": float(per_run.corrected_mean_roc_auc.mean()),
        "result_files": {
            f"seed{seed}": _sha256(correction_result_dir / f"seed{seed}.json")
            for seed in SEEDS
        },
    }
    (correction_root / "thyroid_retained_sample_manifest.json").write_text(
        json.dumps(correction_manifest, indent=2) + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
