#!/usr/bin/env python3
"""Sample-level quality diagnostics for reviewer-revision benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.data.splitting import encode_features, load_split

DEFAULT_RESULT_ROOT = Path("results/reviewer_revision/high_dimensional_benchmark/experiments")
DEFAULT_OUTPUT_DIR = Path("results/reviewer_revision/quality_diagnostics")
METHOD_ORDER = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sample-level distribution, correlation, constraint, NN, and invalid-row diagnostics.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Result root with dataset/method/seed*.json layout and logs/*.csv sample artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for diagnostics artifacts.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_ORDER,
        help="Methods to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to include.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_datasets(result_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in result_root.iterdir()
        if path.is_dir() and path.name != "logs"
    )


def sample_path(result_root: Path, dataset: str, method: str, seed: int, kind: str) -> Path:
    return result_root / "logs" / f"{dataset}_{method}_seed{seed}_{kind}_samples.csv"


def result_path(result_root: Path, dataset: str, method: str, seed: int) -> Path:
    return result_root / dataset / method / f"seed{seed}.json"


def load_train_minority(dataset: str, seed: int) -> pd.DataFrame:
    split = load_split(dataset, seed=seed, return_raw=False)
    X_train = split["X_train"].copy().reset_index(drop=True)
    y_train = pd.Series(split["y_train"]).reset_index(drop=True)
    minority_label = y_train.value_counts().idxmin()
    return X_train.loc[y_train == minority_label].reset_index(drop=True)


def align_numeric_samples(dataset: str, seed: int, frame: pd.DataFrame, train_numeric: pd.DataFrame) -> pd.DataFrame:
    sample = frame.drop(columns=["target"], errors="ignore").copy()
    split_raw = load_split(dataset, seed=seed, return_raw=True)
    preprocessor = split_raw.get("preprocessor")

    if sample.empty:
        return pd.DataFrame(columns=train_numeric.columns)

    if any(dtype == "object" for dtype in sample.dtypes) and preprocessor is not None:
        sample = encode_features(sample, preprocessor)
    else:
        sample = sample.apply(pd.to_numeric, errors="coerce")

    for column in train_numeric.columns:
        if column not in sample.columns:
            sample[column] = train_numeric[column].median()
    extra_columns = [column for column in sample.columns if column not in train_numeric.columns]
    if extra_columns:
        sample = sample.drop(columns=extra_columns)

    sample = sample[train_numeric.columns]
    train_fill = train_numeric.median(numeric_only=True).fillna(0)
    return sample.apply(pd.to_numeric, errors="coerce").fillna(train_fill).fillna(0)


def standardize(train: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_num = train.apply(pd.to_numeric, errors="coerce")
    sample_num = sample.apply(pd.to_numeric, errors="coerce")
    center = train_num.median(numeric_only=True).fillna(0)
    scale = train_num.std(numeric_only=True).replace(0, np.nan).fillna(1.0)
    train_scaled = (train_num.fillna(center) - center) / scale
    sample_scaled = (sample_num.fillna(center) - center) / scale
    return train_scaled, sample_scaled, center, scale


def distribution_similarity(train_scaled: pd.DataFrame, sample_scaled: pd.DataFrame) -> dict[str, float | None]:
    if sample_scaled.empty or train_scaled.empty:
        return {
            "mean_wasserstein_distance": None,
            "median_wasserstein_distance": None,
            "mean_ks_statistic": None,
            "max_ks_statistic": None,
        }

    wasserstein_values: list[float] = []
    ks_values: list[float] = []
    for column in train_scaled.columns:
        train_values = train_scaled[column].to_numpy(dtype=float)
        sample_values = sample_scaled[column].to_numpy(dtype=float)
        train_values = train_values[np.isfinite(train_values)]
        sample_values = sample_values[np.isfinite(sample_values)]
        if len(train_values) == 0 or len(sample_values) == 0:
            continue
        wasserstein_values.append(float(wasserstein_distance(train_values, sample_values)))
        ks_values.append(float(ks_2samp(train_values, sample_values).statistic))

    return {
        "mean_wasserstein_distance": float(np.mean(wasserstein_values)) if wasserstein_values else None,
        "median_wasserstein_distance": float(np.median(wasserstein_values)) if wasserstein_values else None,
        "mean_ks_statistic": float(np.mean(ks_values)) if ks_values else None,
        "max_ks_statistic": float(np.max(ks_values)) if ks_values else None,
    }


def correlation_rmse(train_scaled: pd.DataFrame, sample_scaled: pd.DataFrame) -> float | None:
    if len(train_scaled) < 3 or len(sample_scaled) < 3 or train_scaled.shape[1] < 2:
        return None
    train_corr = train_scaled.corr(numeric_only=True).fillna(0).to_numpy()
    sample_corr = sample_scaled.corr(numeric_only=True).fillna(0).to_numpy()
    if train_corr.shape != sample_corr.shape:
        return None
    tri = np.triu_indices_from(train_corr, k=1)
    if len(tri[0]) == 0:
        return None
    diff = train_corr[tri] - sample_corr[tri]
    return float(np.sqrt(np.mean(diff**2)))


def nearest_neighbor_diagnostics(train_scaled: pd.DataFrame, sample_scaled: pd.DataFrame) -> dict[str, float | None]:
    if len(train_scaled) < 2 or sample_scaled.empty:
        return {
            "mean_nn_distance_to_train": None,
            "median_nn_distance_to_train": None,
            "min_nn_distance_to_train": None,
            "train_nn_q95": None,
            "nn_distance_q95_ratio": None,
            "exact_train_match_rate": None,
        }

    train_array = train_scaled.to_numpy(dtype=float)
    sample_array = sample_scaled.to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_array)
    sample_distances = nn.kneighbors(sample_array, n_neighbors=1, return_distance=True)[0][:, 0]

    train_nn = NearestNeighbors(n_neighbors=2)
    train_nn.fit(train_array)
    train_distances = train_nn.kneighbors(train_array, n_neighbors=2, return_distance=True)[0][:, 1]
    train_q95 = float(np.quantile(train_distances, 0.95))

    exact_matches = pairwise_distances(sample_array, train_array, metric="euclidean").min(axis=1) <= 1e-12
    return {
        "mean_nn_distance_to_train": float(np.mean(sample_distances)),
        "median_nn_distance_to_train": float(np.median(sample_distances)),
        "min_nn_distance_to_train": float(np.min(sample_distances)),
        "train_nn_q95": train_q95,
        "nn_distance_q95_ratio": float(np.mean(sample_distances) / train_q95) if train_q95 > 0 else None,
        "exact_train_match_rate": float(np.mean(exact_matches)),
    }


def pca_space_diagnostics(train_scaled: pd.DataFrame, sample_scaled: pd.DataFrame) -> dict[str, float | int | None]:
    if len(train_scaled) < 3 or sample_scaled.empty:
        return {
            "pca_components": None,
            "pca_explained_variance": None,
            "pca_mean_nn_distance": None,
            "pca_reconstruction_error_mean": None,
        }
    n_components = min(10, len(train_scaled) - 1, train_scaled.shape[1])
    if n_components < 1:
        return {
            "pca_components": None,
            "pca_explained_variance": None,
            "pca_mean_nn_distance": None,
            "pca_reconstruction_error_mean": None,
        }
    pca = PCA(n_components=n_components, random_state=0)
    train_latent = pca.fit_transform(train_scaled)
    sample_latent = pca.transform(sample_scaled)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_latent)
    latent_distances = nn.kneighbors(sample_latent, n_neighbors=1, return_distance=True)[0][:, 0]
    reconstructed = pca.inverse_transform(sample_latent)
    reconstruction_error = np.mean((sample_scaled.to_numpy(dtype=float) - reconstructed) ** 2, axis=1)
    return {
        "pca_components": int(n_components),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "pca_mean_nn_distance": float(np.mean(latent_distances)),
        "pca_reconstruction_error_mean": float(np.mean(reconstruction_error)),
    }


def constraint_diagnostics(raw_frame: pd.DataFrame, sample_scaled: pd.DataFrame) -> dict[str, float | int]:
    if raw_frame.empty:
        return {
            "missing_cell_rate": 0.0,
            "nonfinite_cell_rate": 0.0,
            "high_z_row_violation_rate": 0.0,
            "high_z_cell_rate": 0.0,
            "invalid_row_count": 0,
        }
    numeric = raw_frame.drop(columns=["target"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().to_numpy()
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    high_z = np.abs(sample_scaled.to_numpy(dtype=float)) > 5.0
    invalid_rows = np.any(missing | ~finite | high_z, axis=1)
    return {
        "missing_cell_rate": float(np.mean(missing)),
        "nonfinite_cell_rate": float(np.mean(~finite)),
        "high_z_row_violation_rate": float(np.mean(np.any(high_z, axis=1))) if len(high_z) else 0.0,
        "high_z_cell_rate": float(np.mean(high_z)) if high_z.size else 0.0,
        "invalid_row_count": int(np.sum(invalid_rows)),
    }


def exact_duplicate_rate(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float(frame.drop(columns=["target"], errors="ignore").duplicated().mean())


def invalid_rows(frame: pd.DataFrame, sample_scaled: pd.DataFrame, max_rows: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    numeric = frame.drop(columns=["target"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().to_numpy()
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    high_z = np.abs(sample_scaled.to_numpy(dtype=float)) > 5.0
    invalid_mask = np.any(missing | ~finite | high_z, axis=1)
    invalid = frame.loc[invalid_mask].head(max_rows).copy()
    if invalid.empty:
        return invalid
    invalid.insert(0, "invalid_reason", "missing_or_nonfinite_or_abs_z_gt_5")
    return invalid


def analyze_sample_set(
    dataset: str,
    method: str,
    seed: int,
    kind: str,
    result_root: Path,
    train_minority: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    path = sample_path(result_root, dataset, method, seed, kind)
    if not path.exists():
        return {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "sample_set": kind,
            "status": "missing_sample_csv",
            "sample_csv": str(path),
        }, pd.DataFrame()

    raw = pd.read_csv(path)
    sample_numeric = align_numeric_samples(dataset, seed, raw, train_minority)
    train_scaled, sample_scaled, _, _ = standardize(train_minority, sample_numeric)

    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "sample_set": kind,
        "status": "ok",
        "n_samples": int(len(raw)),
        "n_features": int(sample_numeric.shape[1]),
        "sample_csv": str(path),
        "exact_duplicate_rate": exact_duplicate_rate(raw),
    }
    row.update(distribution_similarity(train_scaled, sample_scaled))
    row["feature_correlation_rmse"] = correlation_rmse(train_scaled, sample_scaled)
    row.update(nearest_neighbor_diagnostics(train_scaled, sample_scaled))
    row.update(pca_space_diagnostics(train_scaled, sample_scaled))
    row.update(constraint_diagnostics(raw, sample_scaled))

    invalid = invalid_rows(raw, sample_scaled)
    if not invalid.empty:
        invalid.insert(0, "sample_set", kind)
        invalid.insert(0, "seed", seed)
        invalid.insert(0, "method", method)
        invalid.insert(0, "dataset", dataset)
    return row, invalid


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "n_samples",
        "exact_duplicate_rate",
        "mean_wasserstein_distance",
        "median_wasserstein_distance",
        "mean_ks_statistic",
        "max_ks_statistic",
        "feature_correlation_rmse",
        "mean_nn_distance_to_train",
        "median_nn_distance_to_train",
        "min_nn_distance_to_train",
        "nn_distance_q95_ratio",
        "exact_train_match_rate",
        "pca_mean_nn_distance",
        "pca_reconstruction_error_mean",
        "missing_cell_rate",
        "nonfinite_cell_rate",
        "high_z_row_violation_rate",
        "high_z_cell_rate",
        "invalid_row_count",
    ]
    aggregations = {"runs": ("seed", "count")}
    for column in numeric_cols:
        if column in rows.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    return (
        rows.loc[rows["status"] == "ok"]
        .groupby(["dataset", "sample_set", "method"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["dataset", "sample_set", "method"])
    )


def write_report(output_path: Path, summary: pd.DataFrame, missing: pd.DataFrame, invalid_examples: pd.DataFrame) -> None:
    lines = [
        "# Quality Diagnostics",
        "",
        "Uniform sample-level diagnostics computed from generated and validated sample CSVs.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No summary rows._",
        "",
        "## Missing Inputs",
        "",
        missing.to_markdown(index=False) if not missing.empty else "_No missing sample CSVs._",
        "",
        "## Invalid Row Examples",
        "",
        invalid_examples.head(20).to_markdown(index=False) if not invalid_examples.empty else "_No invalid rows detected by missing/nonfinite/abs(z)>5 checks._",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    result_root = resolve_path(args.result_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(result_root)
    rows: list[dict[str, Any]] = []
    invalid_frames: list[pd.DataFrame] = []

    for dataset in datasets:
        for seed in args.seeds:
            train_minority = load_train_minority(dataset, seed)
            for method in args.methods:
                if not result_path(result_root, dataset, method, seed).exists():
                    continue
                for kind in ["generated", "validated"]:
                    row, invalid = analyze_sample_set(dataset, method, seed, kind, result_root, train_minority)
                    rows.append(row)
                    if not invalid.empty:
                        invalid_frames.append(invalid)

    diagnostics = pd.DataFrame(rows)
    summary = summarize(diagnostics) if not diagnostics.empty else pd.DataFrame()
    missing = diagnostics.loc[diagnostics["status"] != "ok"].copy() if not diagnostics.empty else pd.DataFrame()
    invalid_examples = pd.concat(invalid_frames, ignore_index=True) if invalid_frames else pd.DataFrame()

    diagnostics.to_csv(output_dir / "quality_diagnostics_rows.csv", index=False)
    summary.to_csv(output_dir / "quality_diagnostics_summary.csv", index=False)
    missing.to_csv(output_dir / "quality_diagnostics_missing_inputs.csv", index=False)
    invalid_examples.to_csv(output_dir / "quality_diagnostics_invalid_examples.csv", index=False)

    manifest = {
        "result_root": str(result_root),
        "output_dir": str(output_dir),
        "datasets": datasets,
        "methods": args.methods,
        "seeds": args.seeds,
        "n_rows": int(len(diagnostics)),
        "n_missing_inputs": int(len(missing)),
        "n_invalid_examples": int(len(invalid_examples)),
        "diagnostic_definitions": {
            "distribution_similarity": "Mean standardized per-feature Wasserstein distance and KS statistic against minority-class training samples.",
            "feature_correlation_preservation": "RMSE between upper-triangle feature-correlation matrices for minority training and synthetic samples.",
            "constraint_violations": "Missing, nonfinite, and abs(z)>5 checks using minority-training medians and standard deviations.",
            "nearest_neighbor_distance": "Euclidean distance from synthetic samples to nearest minority-training sample in standardized feature space.",
            "pca_space": "High-dimensional sensitivity diagnostics using PCA fitted on minority-training samples.",
        },
    }
    with open(output_dir / "quality_diagnostics_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    write_report(output_dir / "quality_diagnostics_report.md", summary, missing, invalid_examples)
    print(f"Wrote quality diagnostics to: {output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
