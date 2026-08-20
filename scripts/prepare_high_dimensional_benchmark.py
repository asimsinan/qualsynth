"""
Prepare the Alon colon tumor microarray benchmark for reviewer-revision runs.

The raw dataset is downloaded from the UGR ELVIRA repository and converted into
fold-safe train/validation/test splits. Feature selection is fit only on each
training split, then the selected feature subset and scaler are reused for that
seed's validation/test data and all downstream methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import ssl
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.data.preprocessing import FoldSafeHighDimensionalPreprocessor

DEFAULT_URL = "https://leo.ugr.es/elvira/DBCRepository/ColonTumor/ColonTumor.zip"
DEFAULT_SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]


def _download_source(url: str) -> bytes:
    try:
        with urlopen(url, timeout=60) as response:
            return response.read()
    except (URLError, ssl.SSLError):
        # The UGR mirror currently has an expired certificate in some Python
        # environments. Keep the fallback explicit and record it in metadata.
        context = ssl._create_unverified_context()
        with urlopen(url, timeout=60, context=context) as response:
            return response.read()


def _parse_elvira_dbc(zip_bytes: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(zip_bytes)) as archive:
        dbc_name = next(name for name in archive.namelist() if name.lower().endswith(".dbc"))
        text = archive.read(dbc_name).decode("utf-8", errors="replace")

    if "cases = (" not in text:
        raise ValueError("Could not locate cases section in colonTumor.dbc")

    cases_block = text.split("cases = (", 1)[1].rsplit(");", 1)[0]
    rows: list[list[Any]] = []
    for line in cases_block.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped.startswith("[") or not stripped.endswith("]"):
            continue
        parts = [part.strip() for part in stripped[1:-1].split(",")]
        if len(parts) < 2:
            continue
        label = parts[0]
        values = [float(value) for value in parts[1:]]
        rows.append([label, *values])

    if not rows:
        raise ValueError("No sample rows were parsed from colonTumor.dbc")

    n_features = len(rows[0]) - 1
    if any(len(row) != n_features + 1 for row in rows):
        raise ValueError("Parsed rows have inconsistent feature counts")

    columns = ["source_label"] + [f"gene_{index:04d}" for index in range(1, n_features + 1)]
    df = pd.DataFrame(rows, columns=columns)
    # UGR labels: negative=tumor, positive=normal. Use 1=normal because the
    # current QualSynth workflow treats the generated minority batch as class 1.
    df["target"] = df["source_label"].map({"negative": 0, "positive": 1}).astype(int)
    return df.drop(columns=["source_label"])


def _prepare_one_seed(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_selected_features: int,
    output_dir: Path,
) -> dict[str, Any]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=0.6,
        stratify=y,
        random_state=seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=0.5,
        stratify=y_temp,
        random_state=seed,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X.columns, index=X_val.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    k = min(n_selected_features, X_train_imputed.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train_imputed, y_train)
    selected_features = X_train_imputed.columns[selector.get_support()].tolist()

    scaler = StandardScaler()
    X_train_selected = pd.DataFrame(
        scaler.fit_transform(X_train_imputed[selected_features]),
        columns=selected_features,
        index=X_train.index,
    )
    X_val_selected = pd.DataFrame(
        scaler.transform(X_val_imputed[selected_features]),
        columns=selected_features,
        index=X_val.index,
    )
    X_test_selected = pd.DataFrame(
        scaler.transform(X_test_imputed[selected_features]),
        columns=selected_features,
        index=X_test.index,
    )

    selector_scores = {
        feature: float(score)
        for feature, score in zip(X.columns, selector.scores_)
        if feature in selected_features and not np.isnan(score)
    }
    imputer_statistics = {
        feature: float(value)
        for feature, value in zip(X.columns, imputer.statistics_)
        if feature in selected_features and not np.isnan(value)
    }
    preprocessor = FoldSafeHighDimensionalPreprocessor(
        feature_names=selected_features,
        selected_features=selected_features,
        original_feature_count=X.shape[1],
        selected_feature_count=len(selected_features),
        imputer_statistics=imputer_statistics,
        selector_scores=selector_scores,
        scaler=scaler,
    )

    split_data = {
        "X_train": X_train_selected.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "X_val": X_val_selected.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "X_test": X_test_selected.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "seed": seed,
        "preprocessor": preprocessor,
        "metadata": {
            "feature_selection": "SelectKBest(f_classif)",
            "feature_selection_scope": "fit on training split only",
            "original_feature_count": int(X.shape[1]),
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
        },
    }

    split_path = output_dir / f"split_seed{seed}.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(split_data, handle)

    return {
        "seed": seed,
        "split_file": str(split_path.relative_to(PROJECT_ROOT)),
        "class_distribution": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "validation": {str(k): int(v) for k, v in y_val.value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
    }


def _resource_snapshot() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(PROJECT_ROOT))
        return {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "memory_bytes": int(memory.total),
            "available_memory_bytes": int(memory.available),
            "disk_free_bytes": int(disk.free),
            "python_version": platform.python_version(),
        }
    except Exception as exc:  # pragma: no cover - resource snapshot is best effort
        return {"error": str(exc), "python_version": platform.python_version()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Alon colon high-dimensional benchmark splits.")
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--dataset-name", default="alon_colon")
    parser.add_argument("--selected-features", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "results" / "reviewer_revision" / "high_dimensional_benchmark")
    parser.add_argument("--save-raw-csv", action="store_true", help="Optionally save the converted raw CSV locally.")
    args = parser.parse_args()

    split_output_dir = (args.split_dir / args.dataset_name).resolve()
    metadata_dir = args.metadata_dir.resolve()
    split_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = _download_source(args.source_url)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    df = _parse_elvira_dbc(source_bytes)
    X = df.drop(columns=["target"])
    y = df["target"]

    if args.save_raw_csv:
        raw_path = PROJECT_ROOT / "data" / "raw" / f"{args.dataset_name}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)

    split_metadata = [
        _prepare_one_seed(X, y, seed, args.selected_features, split_output_dir)
        for seed in args.seeds
    ]

    metadata = {
        "dataset": args.dataset_name,
        "prepared_at": datetime.now().isoformat(),
        "source_url": args.source_url,
        "source_sha256": source_sha256,
        "source_bytes": len(source_bytes),
        "source_files": ["colonTumor.dbc", "colonTumor.x"],
        "license_and_redistribution_note": (
            "The UGR repository provides transformed .data/.names/Elvira-format files for research use, "
            "but no explicit redistribution license was identified on the dataset page. For public archive, "
            "archive this preparation script, source URL, SHA-256 hash, split metadata, and results rather than "
            "redistributing raw downloaded data unless permission is confirmed."
        ),
        "reference": "Alon U. et al. Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays. PNAS 96:6745-6750, 1999.",
        "samples": int(len(df)),
        "original_feature_count": int(X.shape[1]),
        "selected_feature_count": int(min(args.selected_features, X.shape[1])),
        "class_mapping": {"0": "tumor/negative", "1": "normal/positive"},
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "feature_selection": {
            "method": "SelectKBest(f_classif)",
            "scope": "fit independently inside each training split",
            "reason": "prompt-size control and leakage prevention for high-dimensional gene expression data",
        },
        "splits": split_metadata,
        "resource_snapshot": _resource_snapshot(),
    }

    metadata_path = metadata_dir / f"{args.dataset_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Prepared {args.dataset_name}: {len(df)} samples, {X.shape[1]} original features")
    print(f"Selected {metadata['selected_feature_count']} fold-specific features per seed")
    print(f"Wrote splits to {split_output_dir}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
