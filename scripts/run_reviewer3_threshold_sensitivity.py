#!/usr/bin/env python3
"""Run Reviewer 3's post-hoc threshold and safety-diversity study.

Every condition is applied to the same stored, unvalidated candidate pool for a
dataset/seed pair.  Threshold estimation receives only the training-fold
minority reference.  Validation and test folds are used only after a condition
has been fixed, for downstream classifier evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, wasserstein_distance, wilcoxon
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from src.qualsynth.data.splitting import load_split  # noqa: E402
from src.qualsynth.evaluation.classifiers import ClassifierPipeline  # noqa: E402
from src.qualsynth.validation.threshold_calibration import (  # noqa: E402
    ThresholdCalibration,
    calibrate_minority_z_threshold,
)
from src.qualsynth.validation.universal_validator import UniversalValidator  # noqa: E402


DEFAULT_DATASETS = [
    "german_credit",
    "breast_cancer",
    "pima_diabetes",
    "wine_quality",
    "yeast",
    "haberman",
    "thyroid",
    "htru2",
]
DEFAULT_SEEDS = [42, 123, 456]
SOURCE_ROOT = PROJECT_ROOT / "results/reviewer_revision/ablations/component_3seed"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/reviewer_revision/reviewer3_round2/threshold_sensitivity"
)
CLASSIFIERS = ["RandomForest", "XGBoost", "LogisticRegression"]
BOOTSTRAP_METRICS = [
    "acceptance_rate",
    "selection_rate",
    "raw_range_violation_rate",
    "schema_violation_rate",
    "train_match_rate",
    "within_candidate_duplicate_rate",
    "numeric_equivalent_duplicate_rate",
    "numeric_equivalent_duplicate_survivor_rate",
    "statistical_rejection_rate",
    "standardized_wasserstein",
    "correlation_mae",
    "mean_nearest_minority_distance",
    "pairwise_diversity",
    "f1",
    "roc_auc",
]


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    parameter: float | None = None
    clip_to_minority_bounds: bool = False
    historical_status: str = "round2_sensitivity"


CONDITIONS = [
    Condition("std_3", "z_score", 3.0),
    Condition("std_4", "z_score", 4.0),
    Condition("std_5", "z_score", 5.0),
    Condition("percentile_0_99", "central_percentile", 0.99),
    Condition("percentile_0_995", "central_percentile", 0.995),
    Condition("percentile_0_999", "central_percentile", 0.999),
    Condition(
        "historical_preclip_std4",
        "z_score",
        4.0,
        clip_to_minority_bounds=True,
        historical_status="reconstructed_intended_path_not_proven_archived",
    ),
    Condition("calibrated_z95", "calibrated_z"),
    Condition("no_statistical_filter", "none"),
]


@dataclass
class RunInputs:
    dataset: str
    seed: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_minority: pd.DataFrame
    candidates: pd.DataFrame
    target_samples: int
    candidate_path: Path
    split_path: Path
    target_mapping: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--skip-classifiers",
        action="store_true",
        help="Build validation diagnostics without fitting downstream classifiers.",
    )
    parser.add_argument(
        "--no-validator-concordance",
        action="store_true",
        help="Skip checks against the active UniversalValidator z-gate.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def normalize_binary_labels(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    train = pd.Series(y_train).reset_index(drop=True)
    test = pd.Series(y_test).reset_index(drop=True)
    counts = train.value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError(f"Expected binary training labels, found {len(counts)}")
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    mapping = {minority_label: 1, majority_label: 0}

    def apply(values: pd.Series, name: str) -> pd.Series:
        normalized = values.map(mapping)
        if normalized.isna().any():
            unknown = values.loc[normalized.isna()].drop_duplicates().tolist()
            raise ValueError(f"Unknown labels in {name}: {unknown}")
        return normalized.astype(int).reset_index(drop=True)

    metadata = {
        "minority_original": jsonable(minority_label),
        "majority_original": jsonable(majority_label),
        "normalized_minority": 1,
        "normalized_majority": 0,
        "training_counts_original": {
            str(jsonable(label)): int(count) for label, count in counts.items()
        },
    }
    return apply(train, "train"), apply(test, "test"), metadata


def candidate_path(dataset: str, seed: int) -> Path:
    return (
        SOURCE_ROOT
        / "logs"
        / f"{dataset}_qualsynth_component_no_validation_raw_seed{seed}_generated_samples.csv"
    )


def load_run_inputs(dataset: str, seed: int) -> RunInputs:
    split_path = PROJECT_ROOT / "data/splits" / dataset / f"split_seed{seed}.pkl"
    split = load_split(dataset, seed=seed)
    X_train = split["X_train"].copy().reset_index(drop=True)
    X_test = split["X_test"].copy().reset_index(drop=True)
    y_train, y_test, mapping = normalize_binary_labels(
        split["y_train"],
        split["y_test"],
    )
    minority = X_train.loc[y_train == 1].copy().reset_index(drop=True)
    path = candidate_path(dataset, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = pd.read_csv(path)
    missing = [column for column in X_train.columns if column not in candidates.columns]
    extra = [column for column in candidates.columns if column not in X_train.columns]
    if missing or extra:
        raise ValueError(
            f"Candidate schema mismatch for {dataset}/seed{seed}: "
            f"missing={missing}, extra={extra}"
        )
    candidates = candidates.loc[:, X_train.columns].copy().reset_index(drop=True)
    for column in X_train.columns:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    counts = y_train.value_counts()
    target = int(counts.max() - counts.min())
    return RunInputs(
        dataset=dataset,
        seed=seed,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_minority=minority,
        candidates=candidates,
        target_samples=target,
        candidate_path=path,
        split_path=split_path,
        target_mapping=mapping,
    )


def minority_feature_profiles(run: RunInputs) -> pd.DataFrame:
    rows = []
    for column in run.X_minority.columns:
        values = pd.to_numeric(run.X_minority[column], errors="coerce")
        rows.append(
            {
                "dataset": run.dataset,
                "seed": run.seed,
                "feature": column,
                "minority_count": len(run.X_minority),
                "finite_count": int(np.isfinite(values).sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "skewness": float(values.skew()),
                "unique_count": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def finite_mask(frame: pd.DataFrame) -> np.ndarray:
    return np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)


def observed_range_violation_mask(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> np.ndarray:
    lower = reference.min(axis=0)
    upper = reference.max(axis=0)
    return ((frame < lower) | (frame > upper)).any(axis=1).to_numpy(dtype=bool)


def schema_margin_mask(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> np.ndarray:
    lower = reference.min(axis=0)
    upper = reference.max(axis=0)
    margin = ((upper - lower) * 0.2).clip(lower=0.5)
    within = (frame >= (lower - margin)) & (frame <= (upper + margin))
    return finite_mask(frame) & within.all(axis=1).to_numpy(dtype=bool)


def row_key(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(float(value) for value in values)


def exact_duplicate_masks(
    frame: pd.DataFrame,
    eligible: np.ndarray,
    reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_keys = {row_key(row) for row in reference.to_numpy(dtype=float)}
    seen: set[tuple[Any, ...]] = set()
    train_match = np.zeros(len(frame), dtype=bool)
    within_duplicate = np.zeros(len(frame), dtype=bool)
    for position, row in enumerate(frame.to_numpy(dtype=float)):
        if not eligible[position]:
            continue
        key = row_key(row)
        if key in train_keys:
            train_match[position] = True
        elif key in seen:
            within_duplicate[position] = True
        else:
            seen.add(key)
    return train_match, within_duplicate


def active_validator_duplicate_masks(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce UniversalValidator's MD5-over-``str(row.values)`` identity."""
    candidate_hashes = frame.apply(
        lambda row: hashlib.md5(str(row.values).encode()).hexdigest(),
        axis=1,
    )
    reference_hashes = reference.apply(
        lambda row: hashlib.md5(str(row.values).encode()).hexdigest(),
        axis=1,
    )
    train_match = candidate_hashes.isin(set(reference_hashes)).to_numpy(dtype=bool)
    within_duplicate = candidate_hashes.duplicated(keep="first").to_numpy(dtype=bool)
    within_duplicate &= ~train_match
    return train_match, within_duplicate


def z_score_mask(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    threshold: float,
    *,
    inclusive: bool = False,
) -> np.ndarray:
    means = reference.mean(axis=0)
    stds = reference.std(axis=0, ddof=1)
    active = stds > 0
    if not bool(active.any()):
        return np.ones(len(frame), dtype=bool)
    z_scores = ((frame.loc[:, active] - means.loc[active]) / stds.loc[active]).abs()
    comparison = z_scores <= threshold if inclusive else z_scores < threshold
    return comparison.all(axis=1).to_numpy(dtype=bool)


def percentile_mask(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    central_coverage: float,
) -> np.ndarray:
    tail = (1.0 - central_coverage) / 2.0
    lower = reference.quantile(tail, interpolation="linear")
    upper = reference.quantile(1.0 - tail, interpolation="linear")
    return ((frame >= lower) & (frame <= upper)).all(axis=1).to_numpy(dtype=bool)


def apply_condition(
    run: RunInputs,
    condition: Condition,
    calibration: ThresholdCalibration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply one condition using training-minority data only."""
    raw = run.candidates.copy()
    transformed = raw.copy()
    raw_finite = finite_mask(raw)
    raw_range_violation = observed_range_violation_mask(raw, run.X_minority)

    clipped_cells = np.zeros(raw.shape, dtype=bool)
    if condition.clip_to_minority_bounds:
        lower = run.X_minority.min(axis=0)
        upper = run.X_minority.max(axis=0)
        clipped = transformed.clip(lower=lower, upper=upper, axis=1)
        clipped_cells = ~np.isclose(
            transformed.to_numpy(dtype=float),
            clipped.to_numpy(dtype=float),
            equal_nan=True,
        )
        transformed = clipped

    schema_valid = schema_margin_mask(transformed, run.X_minority)
    train_match, within_duplicate = active_validator_duplicate_masks(
        transformed,
        run.X_minority,
    )
    numeric_train_match, numeric_within_duplicate = exact_duplicate_masks(
        transformed,
        schema_valid,
        run.X_minority,
    )
    dedup_valid = schema_valid & ~train_match & ~within_duplicate

    resolved_parameter: float | None = condition.parameter
    if condition.family == "z_score":
        statistical_valid = z_score_mask(
            transformed,
            run.X_minority,
            float(condition.parameter),
        )
    elif condition.family == "central_percentile":
        statistical_valid = percentile_mask(
            transformed,
            run.X_minority,
            float(condition.parameter),
        )
    elif condition.family == "calibrated_z":
        resolved_parameter = float(calibration.threshold)
        statistical_valid = z_score_mask(
            transformed,
            run.X_minority,
            resolved_parameter,
            inclusive=True,
        )
    elif condition.family == "none":
        statistical_valid = np.ones(len(transformed), dtype=bool)
    else:
        raise ValueError(f"Unsupported condition family: {condition.family}")

    accepted = dedup_valid & statistical_valid
    selected_positions = np.flatnonzero(accepted)[: run.target_samples]
    selected_mask = np.zeros(len(transformed), dtype=bool)
    selected_mask[selected_positions] = True

    stage = np.full(len(raw), "accepted", dtype=object)
    stage[~raw_finite] = "non_finite"
    stage[raw_finite & ~schema_valid] = "schema_margin"
    stage[schema_valid & train_match] = "train_match"
    stage[schema_valid & ~train_match & within_duplicate] = "within_candidate_duplicate"
    stage[dedup_valid & ~statistical_valid] = "statistical_gate"
    stage[accepted & ~selected_mask] = "target_cap"

    decisions = pd.DataFrame(
        {
            "dataset": run.dataset,
            "seed": run.seed,
            "condition": condition.name,
            "candidate_index": np.arange(len(raw)),
            "raw_finite": raw_finite,
            "raw_observed_range_violation": raw_range_violation,
            "schema_valid": schema_valid,
            "exact_train_match": train_match,
            "exact_within_candidate_duplicate": within_duplicate,
            "numeric_equivalent_train_match": numeric_train_match,
            "numeric_equivalent_within_candidate_duplicate": numeric_within_duplicate,
            "statistical_valid": statistical_valid,
            "accepted_pre_cap": accepted,
            "selected_for_augmentation": selected_mask,
            "clipped_any_cell": clipped_cells.any(axis=1),
            "rejection_stage": stage,
        }
    )
    accepted_frame = transformed.loc[accepted].copy().reset_index(drop=True)
    selected_frame = transformed.loc[selected_mask].copy().reset_index(drop=True)
    metadata = {
        "resolved_parameter": resolved_parameter,
        "clipped_cells": int(clipped_cells.sum()),
        "clipped_rows": int(clipped_cells.any(axis=1).sum()),
    }
    return decisions, accepted_frame, selected_frame, metadata


def assert_validator_concordance(
    run: RunInputs,
    condition: Condition,
    accepted_frame: pd.DataFrame,
    transformed_source: pd.DataFrame,
) -> None:
    if condition.family != "z_score":
        return
    validator = UniversalValidator(
        verbose=False,
        use_adaptive_threshold=False,
        statistical_std_threshold=float(condition.parameter),
        enable_semantic_dedup=False,
        validation_mode="standard",
        max_samples=None,
    )
    result = validator.validate_and_select(
        transformed_source,
        pd.Series(np.ones(len(transformed_source), dtype=int)),
        run.X_minority,
        pd.Series(np.ones(len(run.X_minority), dtype=int)),
        method_name=f"threshold_concordance_{condition.name}",
    )
    if result.n_after_quality != len(accepted_frame):
        raise AssertionError(
            f"Validator mismatch for {run.dataset}/seed{run.seed}/{condition.name}: "
            f"study={len(accepted_frame)}, validator={result.n_after_quality}"
        )


def standardized_frames(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    means = reference.mean(axis=0).to_numpy(dtype=float)
    stds = reference.std(axis=0, ddof=1).to_numpy(dtype=float)
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    reference_scaled = (reference.to_numpy(dtype=float) - means) / stds
    generated_scaled = (generated.to_numpy(dtype=float) - means) / stds
    return generated_scaled, reference_scaled


def distribution_metrics(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
) -> dict[str, float | int | None]:
    if generated.empty:
        return {
            "standardized_wasserstein": None,
            "correlation_mae": None,
            "mean_nearest_minority_distance": None,
            "median_nearest_minority_distance": None,
            "pairwise_diversity": None,
            "pairwise_diversity_sample_size": 0,
        }
    generated_scaled, reference_scaled = standardized_frames(generated, reference)
    feature_wasserstein = [
        wasserstein_distance(reference_scaled[:, index], generated_scaled[:, index])
        for index in range(reference_scaled.shape[1])
    ]

    reference_frame = pd.DataFrame(reference_scaled)
    generated_frame = pd.DataFrame(generated_scaled)
    active = [
        column
        for column in reference_frame.columns
        if reference_frame[column].nunique() > 1
        and generated_frame[column].nunique() > 1
    ]
    if len(active) >= 2:
        ref_corr = reference_frame[active].corr().to_numpy(dtype=float)
        gen_corr = generated_frame[active].corr().to_numpy(dtype=float)
        upper = np.triu_indices(len(active), k=1)
        corr_values = np.abs(ref_corr[upper] - gen_corr[upper])
        corr_values = corr_values[np.isfinite(corr_values)]
        correlation_mae = float(corr_values.mean()) if len(corr_values) else None
    else:
        correlation_mae = None

    neighbours = NearestNeighbors(n_neighbors=1)
    neighbours.fit(reference_scaled)
    distances, _ = neighbours.kneighbors(generated_scaled)
    nearest = distances[:, 0]

    sample_size = min(500, len(generated_scaled))
    if sample_size >= 2:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(generated_scaled), size=sample_size, replace=False)
        pairwise_diversity = float(pdist(generated_scaled[positions]).mean())
    else:
        pairwise_diversity = 0.0
    return {
        "standardized_wasserstein": float(np.mean(feature_wasserstein)),
        "correlation_mae": correlation_mae,
        "mean_nearest_minority_distance": float(np.mean(nearest)),
        "median_nearest_minority_distance": float(np.median(nearest)),
        "pairwise_diversity": pairwise_diversity,
        "pairwise_diversity_sample_size": sample_size,
    }


def selected_frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def evaluate_selected(
    run: RunInputs,
    selected: pd.DataFrame,
    cache: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], str, bool]:
    cache_key = f"{run.dataset}:{run.seed}:{selected_frame_sha256(selected)}"
    if cache_key in cache:
        return cache[cache_key], cache_key, True
    X_augmented = pd.concat([run.X_train, selected], ignore_index=True)
    y_augmented = pd.concat(
        [run.y_train, pd.Series(np.ones(len(selected), dtype=int))],
        ignore_index=True,
    )
    pipeline = ClassifierPipeline(random_state=run.seed, imbalance_policy="balanced")
    trained = pipeline.train(X_augmented, y_augmented, verbose=False)
    if set(trained) != set(CLASSIFIERS):
        raise RuntimeError(
            f"Classifier training incomplete for {run.dataset}/seed{run.seed}: {list(trained)}"
        )
    metrics = pipeline.evaluate(
        run.X_test,
        run.y_test,
        compute_fairness=False,
        verbose=False,
    )
    if set(metrics) != set(CLASSIFIERS):
        raise RuntimeError(
            f"Classifier evaluation incomplete for {run.dataset}/seed{run.seed}: {list(metrics)}"
        )
    cache[cache_key] = metrics
    return metrics, cache_key, False


def average_classifier_metric(
    metrics: dict[str, dict[str, Any]],
    name: str,
) -> float | None:
    values = [float(item[name]) for item in metrics.values() if name in item]
    return float(np.mean(values)) if values else None


def hierarchical_bootstrap_interval(
    frame: pd.DataFrame,
    metric: str,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    usable = frame.loc[:, ["dataset", metric]].dropna()
    datasets = usable["dataset"].drop_duplicates().tolist()
    if not datasets or resamples <= 0:
        return None, None
    grouped = {
        dataset: usable.loc[usable["dataset"] == dataset, metric].to_numpy(dtype=float)
        for dataset in datasets
    }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        values = []
        for dataset in sampled_datasets:
            group = grouped[str(dataset)]
            values.extend(rng.choice(group, size=len(group), replace=True).tolist())
        estimates.append(float(np.mean(values)))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def build_bootstrap_summary(
    per_run: pd.DataFrame,
    resamples: int,
) -> pd.DataFrame:
    rows = []
    for condition_index, condition in enumerate([item.name for item in CONDITIONS]):
        subset = per_run.loc[per_run["condition"] == condition]
        row: dict[str, Any] = {"condition": condition, "n_runs": len(subset)}
        for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
            values = pd.to_numeric(subset[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else None
            low, high = hierarchical_bootstrap_interval(
                subset.assign(**{metric: values}),
                metric,
                resamples,
                seed=7301 + 101 * condition_index + metric_index,
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def safe_wilcoxon(values: Sequence[float]) -> tuple[float | None, float]:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0 or np.allclose(differences, 0):
        return None, 1.0
    statistic, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return float(statistic), float(p_value)


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["holm_p_value"] = np.nan
    for metric, indices in result.groupby("metric").groups.items():
        ordered = result.loc[list(indices), "raw_p_value"].dropna().sort_values()
        running_max = 0.0
        total = len(ordered)
        for rank, (index, p_value) in enumerate(ordered.items()):
            adjusted = min(1.0, (total - rank) * float(p_value))
            running_max = max(running_max, adjusted)
            result.loc[index, "holm_p_value"] = running_max
    return result


def paired_vs_std4(per_run: pd.DataFrame, resamples: int) -> pd.DataFrame:
    metrics = [
        "acceptance_rate",
        "standardized_wasserstein",
        "correlation_mae",
        "mean_nearest_minority_distance",
        "pairwise_diversity",
        "f1",
        "roc_auc",
    ]
    rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        if condition.name == "std_4":
            continue
        for metric_index, metric in enumerate(metrics):
            pivot = per_run.pivot_table(
                index=["dataset", "seed"],
                columns="condition",
                values=metric,
                aggfunc="first",
            )
            if condition.name not in pivot.columns or "std_4" not in pivot.columns:
                paired = pd.DataFrame(
                    columns=["dataset", "seed", condition.name, "std_4", "difference"]
                )
            else:
                paired = pivot.loc[:, [condition.name, "std_4"]].dropna().reset_index()
                paired["difference"] = paired[condition.name] - paired["std_4"]
            dataset_pairs = (
                paired.groupby("dataset", as_index=False)[
                    [condition.name, "std_4", "difference"]
                ].mean()
                if len(paired)
                else paired
            )
            statistic, p_value = safe_wilcoxon(dataset_pairs["difference"].tolist())
            low, high = hierarchical_bootstrap_interval(
                paired.rename(columns={"difference": metric}),
                metric,
                resamples,
                seed=14011 + 101 * condition_index + metric_index,
            )
            rows.append(
                {
                    "condition": condition.name,
                    "reference": "std_4",
                    "metric": metric,
                    "n_dataset_pairs": len(dataset_pairs),
                    "n_seed_pairs": len(paired),
                    "condition_mean": (
                        float(paired[condition.name].mean()) if len(paired) else None
                    ),
                    "std4_mean": float(paired["std_4"].mean()) if len(paired) else None,
                    "mean_difference": (
                        float(paired["difference"].mean()) if len(paired) else None
                    ),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "wilcoxon_statistic": statistic,
                    "raw_p_value": p_value,
                }
            )
    return holm_adjust(pd.DataFrame(rows))


def minority_size_associations(per_run: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "acceptance_rate",
        "statistical_rejection_rate",
        "standardized_wasserstein",
        "pairwise_diversity",
        "f1",
        "roc_auc",
    ]
    predictors = ["minority_count", "mean_abs_feature_skewness", "max_abs_feature_skewness"]
    rows = []
    for condition in [item.name for item in CONDITIONS]:
        subset = per_run.loc[per_run["condition"] == condition]
        for predictor in predictors:
            for outcome in outcomes:
                pair = subset.loc[:, [predictor, outcome]].dropna()
                if len(pair) < 3 or pair[predictor].nunique() < 2 or pair[outcome].nunique() < 2:
                    rho, p_value = None, None
                else:
                    result = spearmanr(pair[predictor], pair[outcome])
                    rho, p_value = float(result.statistic), float(result.pvalue)
                rows.append(
                    {
                        "condition": condition,
                        "predictor": predictor,
                        "outcome": outcome,
                        "n": len(pair),
                        "spearman_rho": rho,
                        "raw_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)
    return display.to_markdown(index=False)


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition",
        "acceptance_rate_mean",
        "acceptance_rate_ci_low",
        "acceptance_rate_ci_high",
        "raw_range_violation_rate_mean",
        "standardized_wasserstein_mean",
        "correlation_mae_mean",
        "mean_nearest_minority_distance_mean",
        "pairwise_diversity_mean",
        "f1_mean",
        "f1_ci_low",
        "f1_ci_high",
        "roc_auc_mean",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
    ]
    return summary.loc[:, columns]


def write_report(
    output_dir: Path,
    per_run: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    dataset_summary = (
        per_run.groupby(["dataset", "condition"], as_index=False)[
            ["acceptance_rate", "f1", "roc_auc", "pairwise_diversity"]
        ]
        .mean()
    )
    reference_rows = paired.loc[
        paired["condition"].isin(
            ["std_3", "std_5", "historical_preclip_std4", "calibrated_z95"]
        )
    ]
    calibration_summary = (
        calibration.groupby("dataset", as_index=False)
        .agg(
            minority_count=("minority_count", "mean"),
            calibrated_threshold_mean=("threshold", "mean"),
            calibrated_threshold_min=("threshold", "min"),
            calibrated_threshold_max=("threshold", "max"),
            achieved_retention_mean=("achieved_retention", "mean"),
        )
    )
    by_condition = summary.set_index("condition")
    std3 = by_condition.loc["std_3"]
    std4 = by_condition.loc["std_4"]
    std5 = by_condition.loc["std_5"]
    percentile_low = min(
        float(by_condition.loc[name, "acceptance_rate_mean"])
        for name in ["percentile_0_99", "percentile_0_995", "percentile_0_999"]
    )
    percentile_high = max(
        float(by_condition.loc[name, "acceptance_rate_mean"])
        for name in ["percentile_0_99", "percentile_0_995", "percentile_0_999"]
    )
    target_summary = (
        per_run.groupby("condition", as_index=False)
        .agg(
            runs=("condition", "size"),
            target_attained_runs=("target_shortfall", lambda values: int((values == 0).sum())),
            mean_selected=("n_selected", "mean"),
            mean_target_shortfall=("target_shortfall", "mean"),
        )
    )
    std4_runs = per_run.loc[per_run["condition"] == "std_4"]
    numeric_survivors = int(std4_runs["n_numeric_equivalent_duplicate_survivors"].sum())
    std4_accepted = int(std4_runs["n_accepted_pre_cap"].sum())
    candidate_weighted_range_rate = float(
        std4_runs["n_raw_range_violations"].sum()
        / std4_runs["n_raw_candidates"].sum()
    )
    below_four = int((calibration["threshold"] < 4.0).sum())
    above_four = int((calibration["threshold"] > 4.0).sum())
    lines = [
        "# Reviewer 3 threshold reliability and safety-diversity study",
        "",
        "This is a post-hoc sensitivity study over stored, unvalidated QualSynth candidate "
        "pools. Every condition sees the same candidates for each dataset/seed. Calibration "
        "uses the training-fold minority only; validation/test labels do not select thresholds.",
        "",
        "## Interpretation boundaries",
        "",
        "- `std_4` is the active feature-wise rule tested without pre-clipping.",
        "- `historical_preclip_std4` reconstructs the intended clip-to-minority-bounds then "
        "4-sigma path. Sparse archived main-result metadata cannot prove that exact path ran.",
        "- Percentile conditions are counterfactual sensitivities. The historical 0.995 "
        "parameter was configured but inactive, so these rows are not historical evidence.",
        "- Distances and correlations are measured in the stored encoded feature space. "
        "They are diagnostics, not privacy guarantees or proof of semantic novelty.",
        "- Feature-wise z and percentile gates operate on every stored numeric encoded "
        "column to match the active low-dimensional implementation. For nominal columns, "
        "the ordering of encoded values is not intrinsically metric; percentile results "
        "therefore remain diagnostic rather than a proposed replacement rule.",
        "- Main acceptance reproduces the active MD5-over-string row identity. A separate "
        "numeric-equivalence audit treats signed zero (`-0.0` and `0.0`) as equal, exposing "
        "representation-level duplicate survivors without calling them semantic matches.",
        "- Confidence intervals use a hierarchical bootstrap: datasets are sampled first, "
        "then seeds within sampled datasets.",
        "- Wilcoxon tests use eight paired dataset means. Seeds are not treated as "
        "independent blocks; Holm correction is applied within each metric family.",
        "",
        "## Completeness",
        "",
        f"- Dataset/seed candidate pools: {per_run[['dataset', 'seed']].drop_duplicates().shape[0]}.",
        f"- Conditions per pool: {per_run['condition'].nunique()}.",
        f"- Condition-level runs: {len(per_run)}.",
        f"- Failures: {int(per_run['failure'].notna().sum())}.",
        "",
        "## Main findings",
        "",
        f"- Tightening the fixed gate from 4 to 3 sigma reduced mean acceptance from "
        f"{std4['acceptance_rate_mean']:.3f} to {std3['acceptance_rate_mean']:.3f}; "
        f"relaxing it to 5 sigma raised acceptance to {std5['acceptance_rate_mean']:.3f}. "
        f"Mean downstream F1 remained within {min(std3['f1_mean'], std4['f1_mean'], std5['f1_mean']):.3f}--"
        f"{max(std3['f1_mean'], std4['f1_mean'], std5['f1_mean']):.3f}, and ROC-AUC within "
        f"{min(std3['roc_auc_mean'], std4['roc_auc_mean'], std5['roc_auc_mean']):.3f}--"
        f"{max(std3['roc_auc_mean'], std4['roc_auc_mean'], std5['roc_auc_mean']):.3f}.",
        f"- The counterfactual percentile gates accepted only {percentile_low:.3f}--"
        f"{percentile_high:.3f} on average. The inactive historical 0.995 parameter "
        "therefore cannot be described as though it had been part of the benchmark gate.",
        f"- Training-only 95% calibration produced thresholds from "
        f"{calibration['threshold'].min():.3f} to {calibration['threshold'].max():.3f}; "
        f"{below_four}/{len(calibration)} pools were below 4 sigma and "
        f"{above_four}/{len(calibration)} were above it. A universal 4-sigma optimum is "
        "not supported, even though aggregate downstream utility was stable.",
        f"- The equal-pool mean raw minority-range violation rate was "
        f"{std4['raw_range_violation_rate_mean']:.3f}; candidate-weighted it was "
        f"{candidate_weighted_range_rate:.3f}. Clipping changed distribution/diversity "
        "diagnostics but had little aggregate effect on acceptance or predictive utility.",
        f"- Under the 4-sigma condition, the numeric-equivalence audit found "
        f"{numeric_survivors} signed-zero duplicate survivors among {std4_accepted} "
        "accepted rows. The rate is small, but it rules out literal zero-false-positive "
        "or guaranteed-novelty wording.",
        "- Every post-hoc condition fell short of the original balancing target in all "
        "24 pools because the frozen raw generator stopped before compensating for later "
        "deduplication/rejection. Utility results therefore measure fixed-pool sensitivity, "
        "not a fresh end-to-end rerun to equal class balance.",
        "",
        "## Condition summary with 95% bootstrap intervals",
        "",
        markdown_table(compact_summary(summary)),
        "",
        "## Training-only calibrated thresholds",
        "",
        markdown_table(calibration_summary),
        "",
        "## Target attainment in the fixed candidate pools",
        "",
        markdown_table(target_summary),
        "",
        "## Selected paired differences versus `std_4`",
        "",
        markdown_table(reference_rows),
        "",
        "## Dataset-resolved means",
        "",
        markdown_table(dataset_summary),
        "",
        "## Decision rule",
        "",
        "The manuscript must retain 4-sigma only as a historical heuristic if its "
        "acceptance, quality, diversity, and downstream utility remain reasonably stable. "
        "Otherwise it must be labeled as a fixed historical condition, with the "
        "training-only calibrated rule reported as sensitivity evidence. This post-hoc "
        "study does not redefine the headline algorithm or replace the main benchmark.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def condition_run_row(
    run: RunInputs,
    condition: Condition,
    calibration: ThresholdCalibration,
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    selected: pd.DataFrame,
    condition_metadata: dict[str, Any],
    distribution: dict[str, Any],
    classifier_metrics: dict[str, dict[str, Any]],
    classifier_cache_key: str | None,
    classifier_cache_hit: bool,
    profiles: pd.DataFrame,
    validator_concordant: bool | None,
) -> dict[str, Any]:
    n_raw = len(decisions)
    counts = decisions["rejection_stage"].value_counts()
    n_after_schema = int(decisions["schema_valid"].sum())
    n_after_dedup = int(
        (
            decisions["schema_valid"]
            & ~decisions["exact_train_match"]
            & ~decisions["exact_within_candidate_duplicate"]
        ).sum()
    )
    n_statistical = int(counts.get("statistical_gate", 0))
    numeric_duplicate = (
        decisions["numeric_equivalent_train_match"]
        | decisions["numeric_equivalent_within_candidate_duplicate"]
    )
    numeric_survivor = numeric_duplicate & decisions["accepted_pre_cap"]
    abs_skew = profiles["skewness"].abs().replace([np.inf, -np.inf], np.nan)
    return {
        "dataset": run.dataset,
        "seed": run.seed,
        "condition": condition.name,
        "family": condition.family,
        "historical_status": condition.historical_status,
        "configured_parameter": condition.parameter,
        "resolved_parameter": condition_metadata["resolved_parameter"],
        "clip_to_minority_bounds": condition.clip_to_minority_bounds,
        "calibration_scope": calibration.calibration_scope,
        "calibration_method": calibration.method,
        "calibration_target_retention": calibration.target_retention,
        "calibration_achieved_retention": calibration.achieved_retention,
        "minority_count": len(run.X_minority),
        "dimension": run.X_train.shape[1],
        "mean_abs_feature_skewness": float(abs_skew.mean()),
        "max_abs_feature_skewness": float(abs_skew.max()),
        "n_features_abs_skew_gt_1": int((abs_skew > 1.0).sum()),
        "target_samples": run.target_samples,
        "n_raw_candidates": n_raw,
        "n_non_finite": int(counts.get("non_finite", 0)),
        "n_raw_range_violations": int(
            decisions["raw_observed_range_violation"].sum()
        ),
        "raw_range_violation_rate": float(
            decisions["raw_observed_range_violation"].mean()
        ),
        "n_schema_violations": n_raw - n_after_schema,
        "schema_violation_rate": (n_raw - n_after_schema) / n_raw if n_raw else 0.0,
        "n_train_matches": int(decisions["exact_train_match"].sum()),
        "train_match_rate": float(decisions["exact_train_match"].mean()),
        "n_within_candidate_duplicates": int(
            decisions["exact_within_candidate_duplicate"].sum()
        ),
        "within_candidate_duplicate_rate": float(
            decisions["exact_within_candidate_duplicate"].mean()
        ),
        "n_numeric_equivalent_duplicates": int(numeric_duplicate.sum()),
        "numeric_equivalent_duplicate_rate": float(numeric_duplicate.mean()),
        "n_numeric_equivalent_duplicate_survivors": int(numeric_survivor.sum()),
        "numeric_equivalent_duplicate_survivor_rate": (
            int(numeric_survivor.sum()) / len(accepted) if len(accepted) else 0.0
        ),
        "n_after_dedup": n_after_dedup,
        "n_statistical_rejected": n_statistical,
        "statistical_rejection_rate": n_statistical / n_after_dedup if n_after_dedup else 0.0,
        "n_accepted_pre_cap": len(accepted),
        "acceptance_rate": len(accepted) / n_raw if n_raw else 0.0,
        "n_selected": len(selected),
        "selection_rate": len(selected) / n_raw if n_raw else 0.0,
        "target_shortfall": max(0, run.target_samples - len(selected)),
        "clipped_rows": condition_metadata["clipped_rows"],
        "clipped_cells": condition_metadata["clipped_cells"],
        "validator_concordant": validator_concordant,
        **distribution,
        "f1": average_classifier_metric(classifier_metrics, "f1"),
        "roc_auc": average_classifier_metric(classifier_metrics, "roc_auc"),
        "classifier_imbalance_policy": "balanced",
        "classifier_cache_key": classifier_cache_key,
        "classifier_cache_hit": classifier_cache_hit,
        "candidate_sha256": sha256_file(run.candidate_path),
        "split_sha256": sha256_file(run.split_path),
        "failure": None,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = output_dir / "candidate_decisions"
    selected_dir = output_dir / "selected_candidates"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    profile_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    classifier_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for dataset in args.datasets:
        if dataset not in DEFAULT_DATASETS:
            raise ValueError(f"Unknown Reviewer 3 dataset: {dataset}")
        for seed in args.seeds:
            print(f"[{dataset} seed{seed}] loading frozen split and candidate pool")
            run = load_run_inputs(dataset, seed)
            profiles = minority_feature_profiles(run)
            profile_frames.append(profiles)
            calibration = calibrate_minority_z_threshold(
                run.X_train,
                run.y_train,
                target_retention=0.95,
            )
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    **calibration.to_dict(),
                }
            )
            input_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "candidate_path": str(run.candidate_path.relative_to(PROJECT_ROOT)),
                    "candidate_sha256": sha256_file(run.candidate_path),
                    "candidate_rows": len(run.candidates),
                    "split_path": str(run.split_path.relative_to(PROJECT_ROOT)),
                    "split_sha256": sha256_file(run.split_path),
                    "train_rows": len(run.X_train),
                    "test_rows": len(run.X_test),
                    "minority_rows": len(run.X_minority),
                    "target_samples": run.target_samples,
                    "target_mapping": json.dumps(run.target_mapping, sort_keys=True),
                }
            )

            for condition in CONDITIONS:
                print(f"  - {condition.name}")
                try:
                    decisions, accepted, selected, metadata = apply_condition(
                        run,
                        condition,
                        calibration,
                    )
                    validator_concordant: bool | None = None
                    if condition.family == "z_score" and not args.no_validator_concordance:
                        transformed_source = run.candidates.copy()
                        if condition.clip_to_minority_bounds:
                            transformed_source = transformed_source.clip(
                                lower=run.X_minority.min(axis=0),
                                upper=run.X_minority.max(axis=0),
                                axis=1,
                            )
                        assert_validator_concordance(
                            run,
                            condition,
                            accepted,
                            transformed_source,
                        )
                        validator_concordant = True

                    distribution = distribution_metrics(
                        accepted,
                        run.X_minority,
                        seed=seed + sum(ord(char) for char in condition.name),
                    )
                    if args.skip_classifiers:
                        metrics: dict[str, dict[str, Any]] = {}
                        cache_key = None
                        cache_hit = False
                    else:
                        metrics, cache_key, cache_hit = evaluate_selected(
                            run,
                            selected,
                            classifier_cache,
                        )
                    run_row = condition_run_row(
                        run,
                        condition,
                        calibration,
                        decisions,
                        accepted,
                        selected,
                        metadata,
                        distribution,
                        metrics,
                        cache_key,
                        cache_hit,
                        profiles,
                        validator_concordant,
                    )
                    per_run_rows.append(run_row)
                    for classifier, values in metrics.items():
                        classifier_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "condition": condition.name,
                                "classifier": classifier,
                                "f1": values.get("f1"),
                                "roc_auc": values.get("roc_auc"),
                                "balanced_accuracy": values.get("balanced_accuracy"),
                                "precision": values.get("precision"),
                                "recall": values.get("recall"),
                                "n_selected": len(selected),
                                "classifier_cache_key": cache_key,
                                "classifier_cache_hit": cache_hit,
                            }
                        )
                    stem = f"{dataset}_seed{seed}_{condition.name}"
                    decisions.to_csv(decisions_dir / f"{stem}.csv", index=False)
                    selected.to_csv(selected_dir / f"{stem}.csv", index=False)
                except Exception as exc:
                    message = f"{dataset}/seed{seed}/{condition.name}: {type(exc).__name__}: {exc}"
                    print(f"    FAILED: {message}")
                    failures.append(message)
                    per_run_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition.name,
                            "family": condition.family,
                            "historical_status": condition.historical_status,
                            "failure": message,
                        }
                    )

    per_run = pd.DataFrame(per_run_rows)
    per_classifier = pd.DataFrame(classifier_rows)
    profiles = pd.concat(profile_frames, ignore_index=True)
    calibration_frame = pd.DataFrame(calibration_rows)
    inputs = pd.DataFrame(input_rows)

    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    per_classifier.to_csv(output_dir / "per_classifier_metrics.csv", index=False)
    profiles.to_csv(output_dir / "training_minority_feature_profiles.csv", index=False)
    calibration_frame.to_csv(output_dir / "training_only_calibration.csv", index=False)
    inputs.to_csv(output_dir / "input_manifest.csv", index=False)
    (output_dir / "condition_definitions.json").write_text(
        json.dumps([asdict(condition) for condition in CONDITIONS], indent=2) + "\n",
        encoding="utf-8",
    )

    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Threshold study retained {len(failures)} failures in {output_dir / 'failures.json'}"
        )

    summary = build_bootstrap_summary(per_run, args.bootstrap_resamples)
    paired = paired_vs_std4(per_run, args.bootstrap_resamples)
    associations = minority_size_associations(per_run)
    dataset_summary = (
        per_run.groupby(["dataset", "condition"], as_index=False)[BOOTSTRAP_METRICS]
        .mean()
    )
    summary.to_csv(output_dir / "condition_summary_bootstrap.csv", index=False)
    paired.to_csv(output_dir / "paired_vs_std4.csv", index=False)
    associations.to_csv(output_dir / "minority_size_associations.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_condition_summary.csv", index=False)
    write_report(output_dir, per_run, summary, paired, calibration_frame)

    artifact_paths = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "name": "reviewer3_round2_threshold_sensitivity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(Path(__file__)),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "candidate_source": str(SOURCE_ROOT.relative_to(PROJECT_ROOT)),
        "calibration_scope": "training_minority_only",
        "classifier_imbalance_policy": "balanced",
        "validator_reference_population": "training_fold_minority",
        "percentile_historical_status": "configured_but_inactive_counterfactual_only",
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "method": "hierarchical_dataset_then_seed",
            "confidence_level": 0.95,
        },
        "classifier_evaluation_skipped": bool(args.skip_classifiers),
        "condition_runs": len(per_run),
        "classifier_outcomes": len(per_classifier),
        "failures": failures,
        "top_level_artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote threshold sensitivity bundle to {output_dir}")
    print(
        f"Completed {len(per_run)} condition runs and {len(per_classifier)} classifier outcomes"
    )


if __name__ == "__main__":
    main()
