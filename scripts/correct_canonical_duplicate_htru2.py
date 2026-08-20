#!/usr/bin/env python3
"""Correct the one serialization-equivalent duplicate in the archived HTRU2 run.

The archived QualSynth run for HTRU2/seed3141 contains two numeric rows that
differ only in floating-point text precision. They become equal on CSV parsing.
This script removes one row, retrains the original downstream classifiers on the
corrected augmented fold, and writes an auditable correction overlay without
altering the historical artifacts.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.evaluation.classifiers import ClassifierPipeline


DATASET = "htru2"
SEED = 3141
ARCHIVE_ROOT = PROJECT_ROOT / "results" / "openrouter1"
CORRECTION_ROOT = (
    PROJECT_ROOT / "results" / "reviewer_revision" / "canonical_dedup_correction" / "openrouter1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_metrics(by_classifier: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = by_classifier["RandomForest"].keys()
    return {
        key: float(np.mean([by_classifier[name][key] for name in by_classifier]))
        for key in keys
    }


def evaluate(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict]:
    pipeline = ClassifierPipeline(random_state=SEED, imbalance_policy="balanced")
    pipeline.train(X_train, y_train, verbose=False)
    metrics = pipeline.evaluate(X_test, y_test, compute_fairness=False, verbose=False)
    return metrics, mean_metrics(metrics)


def main() -> None:
    result_path = ARCHIVE_ROOT / DATASET / "qualsynth" / f"seed{SEED}.json"
    generated_path = ARCHIVE_ROOT / "logs" / f"{DATASET}_qualsynth_seed{SEED}_generated_samples.csv"
    validated_path = ARCHIVE_ROOT / "logs" / f"{DATASET}_qualsynth_seed{SEED}_validated_samples.csv"
    split_path = PROJECT_ROOT / "data" / "splits" / DATASET / f"split_seed{SEED}.pkl"

    with result_path.open(encoding="utf-8") as handle:
        archived_result = json.load(handle)
    with split_path.open("rb") as handle:
        split = pickle.load(handle)

    generated = pd.read_csv(generated_path)
    validated = pd.read_csv(validated_path)
    generated_corrected = generated.drop_duplicates(ignore_index=True)
    validated_corrected = validated.drop_duplicates(ignore_index=True)
    removed_generated = len(generated) - len(generated_corrected)
    removed_validated = len(validated) - len(validated_corrected)
    if (removed_generated, removed_validated) != (1, 1):
        raise RuntimeError(
            "Expected exactly one duplicate in both archived HTRU2 artifacts; "
            f"found generated={removed_generated}, validated={removed_validated}."
        )

    feature_columns = list(split["X_train"].columns)
    if list(validated_corrected.drop(columns=["target"]).columns) != feature_columns:
        raise RuntimeError("Validated artifact columns do not match the archived training fold.")

    X_train = split["X_train"].reset_index(drop=True)
    y_train = split["y_train"].reset_index(drop=True)
    X_test = split["X_test"]
    y_test = split["y_test"]

    # Rebuild the archived fold in the current evaluator before changing only
    # the duplicate row. The historical and current runtime stacks can differ
    # slightly, so both values are retained in the correction provenance.
    archived_X = pd.concat([X_train, validated.drop(columns=["target"])], ignore_index=True)
    archived_y = pd.concat([y_train, validated["target"].astype(y_train.dtype)], ignore_index=True)
    _, reproduced_average = evaluate(archived_X, archived_y, X_test, y_test)
    corrected_X = pd.concat(
        [X_train, validated_corrected.drop(columns=["target"])], ignore_index=True
    )
    corrected_y = pd.concat(
        [y_train, validated_corrected["target"].astype(y_train.dtype)], ignore_index=True
    )
    corrected_metrics, corrected_average = evaluate(corrected_X, corrected_y, X_test, y_test)

    output_log_dir = CORRECTION_ROOT / "logs"
    output_result_dir = CORRECTION_ROOT / DATASET / "qualsynth"
    output_log_dir.mkdir(parents=True, exist_ok=True)
    output_result_dir.mkdir(parents=True, exist_ok=True)
    corrected_generated_path = output_log_dir / generated_path.name
    corrected_validated_path = output_log_dir / validated_path.name
    corrected_resampled_path = output_log_dir / f"{DATASET}_qualsynth_seed{SEED}_resampled_dataset.csv"
    generated_corrected.to_csv(corrected_generated_path, index=False)
    validated_corrected.to_csv(corrected_validated_path, index=False)
    corrected_resampled = corrected_X.copy()
    corrected_resampled["target"] = corrected_y.to_numpy()
    corrected_resampled.to_csv(corrected_resampled_path, index=False)

    corrected_result = dict(archived_result)
    corrected_result["n_generated"] = int(len(validated_corrected))
    corrected_result["performance_metrics"] = corrected_metrics
    corrected_result["avg_performance"] = corrected_average
    corrected_result["metadata"] = {
        **archived_result.get("metadata", {}),
        "n_generated": int(len(validated_corrected)),
        "X_resampled_size": int(len(corrected_X)),
        "canonical_dedup_correction": {
            "applied": True,
            "dataset": DATASET,
            "seed": SEED,
            "reason": "One serialization-equivalent numeric duplicate in archived output",
            "rows_removed": 1,
            "duplicate_definition": "Exact equality after CSV numeric parsing",
            "archived_result_sha256": sha256(result_path),
            "archived_generated_sha256": sha256(generated_path),
            "archived_validated_sha256": sha256(validated_path),
            "reproduced_archived_metrics": {
                "f1": reproduced_average["f1"],
                "roc_auc": reproduced_average["roc_auc"],
            },
            "historical_recorded_metrics": {
                "f1": archived_result["avg_performance"]["f1"],
                "roc_auc": archived_result["avg_performance"]["roc_auc"],
            },
            "corrected_metrics": {
                "f1": corrected_average["f1"],
                "roc_auc": corrected_average["roc_auc"],
            },
        },
        "artifact_manifest": {
            "generated_samples": {
                "path": str(corrected_generated_path.relative_to(PROJECT_ROOT)),
                "rows": int(len(generated_corrected)),
                "sha256": sha256(corrected_generated_path),
            },
            "validated_samples": {
                "path": str(corrected_validated_path.relative_to(PROJECT_ROOT)),
                "rows": int(len(validated_corrected)),
                "sha256": sha256(corrected_validated_path),
            },
            "resampled_dataset": {
                "path": str(corrected_resampled_path.relative_to(PROJECT_ROOT)),
                "rows": int(len(corrected_resampled)),
                "sha256": sha256(corrected_resampled_path),
            },
        },
    }
    corrected_result_path = output_result_dir / result_path.name
    corrected_result_path.write_text(json.dumps(corrected_result, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "canonical_dedup_correction_v1",
        "correction_result": str(corrected_result_path.relative_to(PROJECT_ROOT)),
        "corrected_result_sha256": sha256(corrected_result_path),
        "rows_before": int(len(validated)),
        "rows_after": int(len(validated_corrected)),
        "duplicate_rows_after": int(validated_corrected.duplicated().sum()),
        "f1_before": reproduced_average["f1"],
        "f1_after": corrected_average["f1"],
        "roc_auc_before": reproduced_average["roc_auc"],
        "roc_auc_after": corrected_average["roc_auc"],
        "historical_f1": archived_result["avg_performance"]["f1"],
        "historical_roc_auc": archived_result["avg_performance"]["roc_auc"],
    }
    (CORRECTION_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
