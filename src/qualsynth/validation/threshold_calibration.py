"""Training-only calibration helpers for Reviewer 3 threshold sensitivity.

The calibrator deliberately accepts only the training fold and its labels. It
uses leave-one-out minority statistics so a candidate observation is not used
to estimate its own mean and standard deviation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    target_retention: float
    achieved_retention: float
    minority_label: Any
    minority_count: int
    numeric_feature_count: int
    finite_leave_one_out_scores: int
    calibration_scope: str = "training_minority_only"
    method: str = "leave_one_out_max_abs_z_quantile"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calibrate_minority_z_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    target_retention: float = 0.95,
    minimum_threshold: float = 1.0,
    maximum_threshold: float = 10.0,
) -> ThresholdCalibration:
    """Calibrate a max-feature absolute-z threshold from training minority rows.

    No validation or test data can be supplied to this API. Each minority row
    is scored against mean/std estimates from the other minority rows, then the
    requested empirical retention quantile is clipped to a documented range.
    """
    if not 0.0 < target_retention <= 1.0:
        raise ValueError("target_retention must be in (0, 1]")
    if minimum_threshold <= 0 or maximum_threshold < minimum_threshold:
        raise ValueError("Invalid threshold bounds")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal length")

    labels = pd.Series(y_train).reset_index(drop=True)
    counts = labels.value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError("Binary training labels are required")
    minority_label = counts.idxmin()
    minority_mask = labels == minority_label
    minority = X_train.reset_index(drop=True).loc[minority_mask]
    numeric = minority.select_dtypes(include=[np.number]).astype(float)
    if numeric.shape[1] == 0:
        raise ValueError("At least one numeric feature is required")
    if len(numeric) < 4:
        raise ValueError(
            "At least four minority training rows are required for leave-one-out calibration"
        )

    row_scores = []
    for position in range(len(numeric)):
        reference = numeric.drop(numeric.index[position])
        row = numeric.iloc[position]
        means = reference.mean(axis=0)
        stds = reference.std(axis=0, ddof=1).replace(0.0, np.nan)
        z_scores = ((row - means).abs() / stds).replace([np.inf, -np.inf], np.nan)
        finite = z_scores.dropna()
        if not finite.empty:
            row_scores.append(float(finite.max()))

    if not row_scores:
        raise ValueError("Calibration produced no finite leave-one-out z scores")

    raw_threshold = float(
        np.quantile(np.asarray(row_scores, dtype=float), target_retention, method="higher")
    )
    threshold = float(np.clip(raw_threshold, minimum_threshold, maximum_threshold))
    achieved_retention = float(np.mean(np.asarray(row_scores) <= threshold))
    return ThresholdCalibration(
        threshold=threshold,
        target_retention=target_retention,
        achieved_retention=achieved_retention,
        minority_label=minority_label,
        minority_count=len(numeric),
        numeric_feature_count=numeric.shape[1],
        finite_leave_one_out_scores=len(row_scores),
    )


def minority_reference(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, Any]:
    """Return only the training-fold minority rows and their original label."""
    labels = pd.Series(y_train).reset_index(drop=True)
    counts = labels.value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError("Binary training labels are required")
    minority_label = counts.idxmin()
    return X_train.reset_index(drop=True).loc[labels == minority_label].copy(), minority_label

