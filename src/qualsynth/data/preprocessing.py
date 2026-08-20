"""Dataset loading and preprocessing for the source-tree experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_SPECS: Dict[str, Dict[str, object]] = {
    "german_credit": {
        "filename": "german_credit.csv",
        "target_column": "class",
        "categorical_features": [
            "checking_status",
            "credit_history",
            "purpose",
            "savings_status",
            "employment",
            "personal_status",
            "other_parties",
            "property_magnitude",
            "other_payment_plans",
            "housing",
            "job",
            "own_telephone",
            "foreign_worker",
        ],
        "numerical_features": [
            "duration",
            "credit_amount",
            "installment_rate",
            "residence_since",
            "age",
            "existing_credits",
            "num_dependents",
        ],
        "protected_attributes": ["age", "personal_status"],
    },
    "breast_cancer": {"filename": "breast_cancer.csv", "target_column": "target"},
    "pima_diabetes": {"filename": "pima_diabetes.csv", "target_column": "target"},
    "wine_quality": {"filename": "wine_quality.csv", "target_column": "target"},
    "yeast": {"filename": "yeast.csv", "target_column": "target"},
    "haberman": {"filename": "haberman.csv", "target_column": "target"},
    "thyroid": {"filename": "thyroid.csv", "target_column": "target"},
    "htru2": {"filename": "htru2.csv", "target_column": "target"},
    "alon_colon": {"filename": "alon_colon.csv", "target_column": "target"},
}


@dataclass
class FoldSafeHighDimensionalPreprocessor:
    """Minimal preprocessor for fold-safe high-dimensional split artifacts."""

    feature_names: List[str]
    selected_features: List[str]
    original_feature_count: int
    selected_feature_count: int
    target_column: str = "target"
    categorical_features: List[str] | None = None
    numerical_features: List[str] | None = None
    protected_attributes: List[str] | None = None
    label_encoders: Dict[str, Any] | None = None
    imputer_statistics: Dict[str, float] | None = None
    selector_scores: Dict[str, float] | None = None
    scaler: StandardScaler | None = None

    def __post_init__(self) -> None:
        self.categorical_features = self.categorical_features or []
        self.numerical_features = self.numerical_features or list(self.selected_features)
        self.protected_attributes = self.protected_attributes or []
        self.label_encoders = self.label_encoders or {}


class DatasetPreprocessor:
    """Preprocess one dataset while retaining encoding metadata for reuse."""

    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.protected_attributes: List[str] = list(
            DATASET_SPECS[dataset_name].get("protected_attributes", [])
        )
        self.target_column: str | None = None

    def load_and_preprocess(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(filepath)
        spec = DATASET_SPECS[self.dataset_name]
        target_column = str(spec["target_column"])
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {filepath}")

        y = self._build_target(df[target_column])
        X = df.drop(columns=[target_column]).copy()

        explicit_categorical = list(spec.get("categorical_features", []))
        explicit_numerical = list(spec.get("numerical_features", []))
        inferred_categorical = [
            column for column in X.columns if X[column].dtype == "object" and column not in explicit_categorical
        ]

        self.categorical_features = [column for column in explicit_categorical + inferred_categorical if column in X.columns]
        if explicit_numerical:
            self.numerical_features = [column for column in explicit_numerical if column in X.columns]
        else:
            self.numerical_features = [
                column for column in X.columns if column not in self.categorical_features
            ]

        for column in self.categorical_features:
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].fillna("missing").astype(str))
            self.label_encoders[column] = encoder

        for column in self.numerical_features:
            X[column] = pd.to_numeric(X[column], errors="coerce")
            fill_value = X[column].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            X[column] = X[column].fillna(fill_value)

        if self.numerical_features:
            X[self.numerical_features] = self.scaler.fit_transform(X[self.numerical_features])

        self.feature_names = X.columns.tolist()
        self.target_column = target_column
        return X, y

    def _build_target(self, target_series: pd.Series) -> pd.Series:
        if self.dataset_name == "german_credit":
            return (pd.to_numeric(target_series, errors="coerce").fillna(1).astype(int) == 2).astype(int)
        return pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)

    def get_feature_info(self) -> Dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "protected_attributes": self.protected_attributes,
            "target_column": self.target_column,
            "n_features": len(self.feature_names),
        }


def load_dataset(
    dataset_name: str,
    data_dir: str = "data/raw",
    return_preprocessor: bool = False,
):
    """Load one configured raw dataset and return encoded features and labels."""

    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    filename = str(DATASET_SPECS[dataset_name]["filename"])
    filepath = Path(data_dir) / filename
    preprocessor = DatasetPreprocessor(dataset_name)
    X, y = preprocessor.load_and_preprocess(str(filepath))
    feature_info = preprocessor.get_feature_info()
    if return_preprocessor:
        return X, y, feature_info, preprocessor
    return X, y, feature_info


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[4]
    data_dir = project_root / "data" / "raw"

    for dataset_name in DATASET_SPECS:
        X, y, info = load_dataset(dataset_name, str(data_dir))
        print(f"{dataset_name}: X={X.shape}, y={y.value_counts().to_dict()}, info={info['n_features']}")
