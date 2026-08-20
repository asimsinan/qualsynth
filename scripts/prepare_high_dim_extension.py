"""Prepare reviewer-revision high-dimensional extension splits.

Adds two pieces of evidence on top of the existing `prepare_high_dimensional_benchmark.py`
(which prepares the canonical Alon colon benchmark at k=50):

1. **Alon k-sweep** at k ∈ {200, 500} — same source data as the existing Alon benchmark,
   different `--selected-features` so we can demonstrate that anchor-centric prompting
   still produces structurally faithful synthetic data as the prompted feature count
   grows. Each k value lives under its own dataset name (`alon_colon_k200`,
   `alon_colon_k500`) with its own `data/splits/<name>/split_seed*.pkl` files.
2. **Golub leukemia** (`golub_leukemia`) — a second canonical microarray benchmark
   from Golub et al. (Science, 1999), parsed from the same UGR Elvira `.dbc`
   format. Uses the same fold-safe SelectKBest(f_classif) preprocessing the Alon
   benchmark already uses; this addresses Reviewer 1's "datasets" (plural) request
   without inventing an evaluation pipeline.

This script is intentionally a sibling, not a replacement, of
`prepare_high_dimensional_benchmark.py`. The original script is referenced by
`configs/experiments/high_dimensional_benchmark.yaml` and existing reproducibility
bundles, so we keep it untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import ssl
import sys
from dataclasses import dataclass
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

DEFAULT_SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HighDimSource:
    """A high-dimensional source dataset that ships in the UGR Elvira `.dbc` format.

    `label_mapping` maps the raw Elvira class string to an integer target. We pin
    the mapping per dataset so the minority class always lands on `target == 1`
    (the value QualSynth's generation workflow expects).
    """

    key: str
    description: str
    candidate_urls: tuple[str, ...]
    label_mapping: dict[str, int]
    minority_class_name: str
    majority_class_name: str
    citation: str
    license_note: str


SOURCES: dict[str, HighDimSource] = {
    "alon": HighDimSource(
        key="alon",
        description="Alon et al. (PNAS, 1999) colon tumor microarray (62 samples × 2000 genes).",
        candidate_urls=(
            "https://leo.ugr.es/elvira/DBCRepository/ColonTumor/ColonTumor.zip",
        ),
        # UGR labels: negative=tumor (40, majority), positive=normal (22, minority).
        label_mapping={"negative": 0, "positive": 1},
        minority_class_name="normal",
        majority_class_name="tumor",
        citation=(
            "Alon U. et al. Broad patterns of gene expression revealed by clustering "
            "analysis of tumor and normal colon tissues probed by oligonucleotide arrays. "
            "PNAS 96:6745-6750, 1999."
        ),
        license_note=(
            "UGR Elvira repository provides Elvira-format .dbc/.x files for research use; "
            "no explicit redistribution license is published. Archive this script, source URL, "
            "SHA-256, split metadata, and results rather than redistributing raw downloads."
        ),
    ),
    "golub": HighDimSource(
        key="golub",
        description="Golub et al. (Science, 1999) ALL/AML leukemia microarray (72 samples × 7129 genes).",
        # The UGR Elvira mirror at /elvira/DBCRepository/Leukemia/ALLAML.html links
        # the canonical ALL-AML_Leukemia.zip; we keep a few historical alternatives
        # as fallbacks in case the upstream rearranges its layout again.
        candidate_urls=(
            "https://leo.ugr.es/elvira/DBCRepository/Leukemia/ALL-AML_Leukemia.zip",
            "https://leo.ugr.es/elvira/DBCRepository/Leukemia/ALLAML.zip",
            "https://leo.ugr.es/elvira/DBCRepository/LeukemiaAllAml/LeukemiaAllAml.zip",
        ),
        # Elvira convention: AML (acute myeloid leukemia) is the smaller class
        # (25 of 72 samples) and is what we want at target == 1.
        label_mapping={"ALL": 0, "AML": 1, "all": 0, "aml": 1, "negative": 0, "positive": 1},
        minority_class_name="AML",
        majority_class_name="ALL",
        citation=(
            "Golub T. R. et al. Molecular classification of cancer: class discovery and "
            "class prediction by gene expression monitoring. Science 286:531-537, 1999."
        ),
        license_note=(
            "UGR Elvira repository provides Elvira-format .dbc/.x files for research use; "
            "no explicit redistribution license is published. Archive this script, source URL, "
            "SHA-256, split metadata, and results rather than redistributing raw downloads."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Source acquisition + parsing
# ---------------------------------------------------------------------------


def _download_source(url: str) -> bytes:
    """Fetch a remote URL with a permissive TLS fallback for the UGR mirror."""

    try:
        with urlopen(url, timeout=60) as response:
            return response.read()
    except (URLError, ssl.SSLError):
        # UGR's certificate has been intermittently expired across Python builds.
        # The fallback is recorded in metadata and the SHA-256 keeps integrity verifiable.
        context = ssl._create_unverified_context()
        with urlopen(url, timeout=60, context=context) as response:
            return response.read()


def _resolve_remote_source(
    candidate_urls: tuple[str, ...],
    explicit_url: str | None,
) -> tuple[str, bytes]:
    """Return the first URL that returns a non-empty body, plus the body."""

    urls = (explicit_url,) if explicit_url else candidate_urls
    last_error: Exception | None = None
    for url in urls:
        if not url:
            continue
        try:
            payload = _download_source(url)
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            continue
        if payload:
            return url, payload
    raise RuntimeError(
        f"Could not download high-dimensional source from any candidate URL: {urls}; "
        f"last error: {last_error!r}"
    )


def _parse_elvira_dbc(zip_bytes: bytes, label_mapping: dict[str, int]) -> pd.DataFrame:
    """Parse a UGR Elvira `.dbc` zip into a (features + target) DataFrame.

    The Alon and Golub DBCs share a common Elvira structure:
        ...
        cases = (
            [<class_label>, <feature_1>, ..., <feature_n>],
            ...
        );

    We look up `<class_label>` in `label_mapping` so callers can pin the
    minority class onto `target == 1`.
    """

    with ZipFile(BytesIO(zip_bytes)) as archive:
        dbc_candidates = [name for name in archive.namelist() if name.lower().endswith(".dbc")]
        if not dbc_candidates:
            raise ValueError(f"No .dbc file in archive; got {archive.namelist()!r}")
        # When a UGR archive ships with separate Train/Test/Total .dbcs (Golub
        # ALL-AML, NervousSystem, etc.), we want the union of all samples, which
        # the upstream calls "Total". Falling back to the largest .dbc keeps
        # archives that ship a single combined file (Alon colon) on the same
        # code path while still defaulting to the most-complete file when several
        # exist.
        def _pick_combined_dbc(names: list[str]) -> str:
            preferred = [name for name in names if "total" in name.lower()]
            if preferred:
                return preferred[0]
            return max(names, key=lambda name: archive.getinfo(name).file_size)

        dbc_name = _pick_combined_dbc(dbc_candidates)
        text = archive.read(dbc_name).decode("utf-8", errors="replace")

    if "cases = (" not in text:
        raise ValueError(f"Could not locate `cases = (` block in {dbc_name}")

    cases_block = text.split("cases = (", 1)[1].rsplit(");", 1)[0]
    rows: list[list[Any]] = []
    unknown_labels: set[str] = set()
    for line in cases_block.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped.startswith("[") or not stripped.endswith("]"):
            continue
        parts = [part.strip() for part in stripped[1:-1].split(",")]
        if len(parts) < 2:
            continue
        raw_label = parts[0].strip().strip("'\"")
        if raw_label not in label_mapping:
            unknown_labels.add(raw_label)
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric feature value in {dbc_name}: {exc}") from exc
        rows.append([raw_label, *values])

    if unknown_labels:
        raise ValueError(
            f"Encountered class labels {sorted(unknown_labels)!r} in {dbc_name} that are "
            f"not in label_mapping {sorted(label_mapping)!r}; refusing to silently drop them."
        )
    if not rows:
        raise ValueError(f"No sample rows parsed from {dbc_name}")

    n_features = len(rows[0]) - 1
    if any(len(row) != n_features + 1 for row in rows):
        raise ValueError(f"Inconsistent feature counts in {dbc_name}")

    columns = ["source_label"] + [f"gene_{index:05d}" for index in range(1, n_features + 1)]
    df = pd.DataFrame(rows, columns=columns)
    df["target"] = df["source_label"].map(label_mapping).astype(int)
    return df.drop(columns=["source_label"])


def _load_local_csv(path: Path) -> pd.DataFrame:
    """Load a precomputed CSV mirror (used when the UGR mirror is unreachable)."""

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"CSV mirror at {path} must contain a `target` column.")
    return df


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------


def _prepare_one_seed(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_selected_features: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Replicates the canonical Alon prep pipeline for one seed, parameterized by k."""

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
            "train": {str(k_): int(v) for k_, v in y_train.value_counts().sort_index().items()},
            "validation": {str(k_): int(v) for k_, v in y_val.value_counts().sort_index().items()},
            "test": {str(k_): int(v) for k_, v in y_test.value_counts().sort_index().items()},
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a high-dimensional reviewer-revision split set. Use --source alon for "
            "the existing colon source (run with k=200 / k=500 to get the Alon k-sweep splits) "
            "and --source golub for the Golub leukemia ALL/AML benchmark."
        ),
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        required=True,
        help="Built-in source descriptor (defines URL, label mapping, citation).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help=(
            "Logical dataset name written to data/splits/<name>/split_seed*.pkl. "
            "Must match the YAML config under configs/datasets/<name>.yaml."
        ),
    )
    parser.add_argument(
        "--selected-features",
        type=int,
        required=True,
        help="k for SelectKBest(f_classif). Fit independently on each training split.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds. Default mirrors the 3-seed component ablation (42, 123, 456).",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Override the built-in candidate URLs (for archive runs / pinned snapshots).",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help=(
            "Path to a precomputed CSV mirror with a `target` column. Use this when the "
            "Elvira mirror is unreachable; SHA-256 of the CSV is recorded in metadata."
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reviewer_revision" / "high_dim_extension" / "splits",
    )
    parser.add_argument(
        "--save-raw-csv",
        action="store_true",
        help="Optionally cache the parsed dataset as data/raw/<dataset_name>.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = SOURCES[args.source]
    split_output_dir = (args.split_dir / args.dataset_name).resolve()
    metadata_dir = args.metadata_dir.resolve()
    split_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_kind: str
    source_used: str
    source_sha256: str
    if args.source_csv:
        csv_bytes = args.source_csv.read_bytes()
        source_sha256 = hashlib.sha256(csv_bytes).hexdigest()
        df = _load_local_csv(args.source_csv)
        source_kind = "local_csv"
        source_used = str(args.source_csv.resolve())
    else:
        source_used, source_bytes = _resolve_remote_source(source.candidate_urls, args.source_url)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        df = _parse_elvira_dbc(source_bytes, source.label_mapping)
        source_kind = "elvira_dbc"

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
        "source_descriptor": source.key,
        "source_kind": source_kind,
        "source_used": source_used,
        "source_sha256": source_sha256,
        "license_and_redistribution_note": source.license_note,
        "reference": source.citation,
        "samples": int(len(df)),
        "original_feature_count": int(X.shape[1]),
        "requested_selected_features": int(args.selected_features),
        "selected_feature_count": int(min(args.selected_features, X.shape[1])),
        "minority_class_name": source.minority_class_name,
        "majority_class_name": source.majority_class_name,
        "label_mapping": source.label_mapping,
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
    print(f"Selected k = {metadata['selected_feature_count']} fold-specific features per seed")
    print(f"Source: {source_used} (kind={source_kind}, sha256={source_sha256[:12]}…)")
    print(f"Wrote splits to {split_output_dir}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
