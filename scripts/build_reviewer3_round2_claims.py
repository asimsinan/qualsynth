#!/usr/bin/env python3
"""Freeze Reviewer 3 evidence into one machine-readable manuscript claim bundle.

The builder is deliberately fail-closed: it will not emit a claim freeze until the
no-augmentation, threshold, and paired-backend matrices all pass their completeness
and provenance gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = PROJECT_ROOT / "results/reviewer_revision"
ROUND2_ROOT = REVISION_ROOT / "reviewer3_round2"
NOAUG_ROOT = ROUND2_ROOT / "no_augmentation/analysis"
THRESHOLD_ROOT = ROUND2_ROOT / "threshold_sensitivity"
BACKEND_ROOT = ROUND2_ROOT / "backend_sensitivity/core"
HISTORICAL_ROOT = REVISION_ROOT / "claim_verification_refreshed_tabddpm"
COMPONENT_ROOT = REVISION_ROOT / "ablations/component_3seed/analysis"
HIGH_DIM_ROOT = REVISION_ROOT / "high_dim_extension/analysis"
OUTPUT_ROOT = ROUND2_ROOT / "claim_freeze"
MACRO_PATH = PROJECT_ROOT / "sreport/round2_claims.tex"
GENERATED_TEX_ROOT = PROJECT_ROOT / "sreport/generated"

MAIN_METHODS = [
    "qualsynth",
    "smote",
    "ctgan",
    "tabfairgdt",
    "tabddpm",
    "no_augmentation",
]
SYNTHETIC_METHODS = MAIN_METHODS[:-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def read_json(path: Path) -> dict[str, Any]:
    with require_file(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def one(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def gate_inputs() -> dict[str, Any]:
    noaug_completeness = read_csv(NOAUG_ROOT / "completeness.csv")
    threshold_manifest = read_json(THRESHOLD_ROOT / "manifest.json")
    backend_manifest = read_json(BACKEND_ROOT / "analysis_manifest.json")
    backend_runs = read_csv(BACKEND_ROOT / "per_run_backend_metrics.csv")
    backend_report = require_file(BACKEND_ROOT / "REPORT.md").read_text(encoding="utf-8")

    noaug_ok = len(noaug_completeness) == 560 and bool(noaug_completeness["exists"].all())
    threshold_ok = (
        int(threshold_manifest.get("condition_runs", 0)) == 216
        and int(threshold_manifest.get("classifier_outcomes", 0)) == 648
        and not threshold_manifest.get("failures")
    )
    backend_ok = (
        int(backend_manifest.get("run_count", 0)) == 24
        and int(backend_manifest.get("successful_runs", 0)) == 24
        and not backend_manifest.get("failures")
        and len(backend_runs) == 24
        and bool(backend_runs["success"].all())
        and bool(backend_runs["stage_artifacts_complete"].all())
        and bool(backend_runs["candidate_stage_semantics_present"].all())
        and bool((backend_runs["reported_calls"] == backend_runs["request_calls"]).all())
        and bool(backend_runs["target_reached"].all())
        and bool((backend_runs["target_shortfall"] == 0).all())
        and bool((backend_runs["selected_rows"] == backend_runs["target_samples"]).all())
        and "- Gate status: PASS." in backend_report
    )
    gates = {
        "no_augmentation": noaug_ok,
        "threshold_sensitivity": threshold_ok,
        "backend_sensitivity": backend_ok,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Claim freeze refused; failed evidence gates: {failed}")
    return {
        "gates": gates,
        "noaug_completeness": noaug_completeness,
        "threshold_manifest": threshold_manifest,
        "backend_manifest": backend_manifest,
        "backend_runs": backend_runs,
    }


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "performance_f1": read_csv(NOAUG_ROOT / "six_method_dataset_means_f1.csv"),
        "performance_roc_auc": read_csv(NOAUG_ROOT / "six_method_dataset_means_roc_auc.csv"),
        "performance_f1_ranks": read_csv(NOAUG_ROOT / "six_method_dataset_mean_ranks_f1.csv"),
        "performance_roc_auc_ranks": read_csv(
            NOAUG_ROOT / "six_method_dataset_mean_ranks_roc_auc.csv"
        ),
        "performance_classifier_summary": read_csv(
            NOAUG_ROOT / "classifier_resolved_summary.csv"
        ).loc[lambda frame: frame["method"].isin(MAIN_METHODS)],
        "performance_arithmetic_all": read_csv(NOAUG_ROOT / "all_condition_arithmetic_summary.csv"),
        "performance_arithmetic_summary": read_csv(
            NOAUG_ROOT / "all_condition_arithmetic_summary.csv"
        ).loc[lambda frame: frame["method"].isin(MAIN_METHODS)],
        "performance_per_run": read_csv(NOAUG_ROOT / "per_run_metrics.csv").loc[
            lambda frame: frame["method"].isin(MAIN_METHODS)
        ],
        "performance_omnibus": read_csv(NOAUG_ROOT / "six_method_omnibus_tests.csv"),
        "performance_pairwise": read_csv(NOAUG_ROOT / "six_method_qualsynth_pairwise_holm.csv"),
        "noaug_pairs_f1": read_csv(NOAUG_ROOT / "qualsynth_vs_noaug_cw_by_dataset_f1.csv"),
        "noaug_pairs_roc_auc": read_csv(
            NOAUG_ROOT / "qualsynth_vs_noaug_cw_by_dataset_roc_auc.csv"
        ),
        "threshold_conditions": read_csv(THRESHOLD_ROOT / "condition_summary_bootstrap.csv"),
        "threshold_dataset_conditions": read_csv(THRESHOLD_ROOT / "dataset_condition_summary.csv"),
        "threshold_per_run": read_csv(THRESHOLD_ROOT / "per_run_metrics.csv"),
        "threshold_calibration": read_csv(THRESHOLD_ROOT / "training_only_calibration.csv"),
        "threshold_paired": read_csv(THRESHOLD_ROOT / "paired_vs_std4.csv"),
        "threshold_minority_associations": read_csv(
            THRESHOLD_ROOT / "minority_size_associations.csv"
        ),
        "backend_per_run": read_csv(BACKEND_ROOT / "per_run_backend_metrics.csv"),
        "backend_model_summary": read_csv(BACKEND_ROOT / "model_summary.csv"),
        "backend_paired": read_csv(BACKEND_ROOT / "paired_differences.csv"),
        "quality_audit": read_csv(HISTORICAL_ROOT / "quality_audit_summary.csv"),
        "component_ablation": read_csv(COMPONENT_ROOT / "component_paired_tests.csv"),
        "high_dimensional": read_csv(HIGH_DIM_ROOT / "k_sweep_table.csv"),
        "high_dimensional_cross_dataset": read_csv(HIGH_DIM_ROOT / "cross_dataset_table.csv"),
        "historical_cost": read_csv(REVISION_ROOT / "cost_runtime_manuscript_table.csv"),
    }


def selected_claims(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    f1_ranks = tables["performance_f1_ranks"]
    auc_ranks = tables["performance_roc_auc_ranks"]
    arithmetic = tables["performance_arithmetic_summary"]
    arithmetic_all = tables["performance_arithmetic_all"]
    omnibus = tables["performance_omnibus"]
    pairwise = tables["performance_pairwise"]
    threshold = tables["threshold_conditions"]
    calibration = tables["threshold_calibration"]
    threshold_per_run = tables["threshold_per_run"]
    backend_runs = tables["backend_per_run"]
    backend_summary = tables["backend_model_summary"]
    backend_paired = tables["backend_paired"]

    main_omnibus = omnibus.loc[omnibus["scope"] == "eight_dataset_means"]
    q_vs_noaug_f1 = one(
        pairwise,
        scope="eight_dataset_means",
        metric="f1",
        baseline="no_augmentation",
    )
    q_vs_noaug_auc = one(
        pairwise,
        scope="eight_dataset_means",
        metric="roc_auc",
        baseline="no_augmentation",
    )
    q_vs_ctgan_auc = one(
        pairwise,
        scope="eight_dataset_means",
        metric="roc_auc",
        baseline="ctgan",
    )
    q_vs_tabddpm_auc = one(
        pairwise,
        scope="eight_dataset_means",
        metric="roc_auc",
        baseline="tabddpm",
    )
    z3 = one(threshold, condition="std_3")
    z4 = one(threshold, condition="std_4")
    z5 = one(threshold, condition="std_5")
    calibrated = one(threshold, condition="calibrated_z95")
    percentiles = threshold.loc[threshold["condition"].str.startswith("percentile_")]
    calibration_below = int((calibration["threshold"] < 4.0).sum())
    calibration_above = int((calibration["threshold"] > 4.0).sum())

    backend_by_method = backend_runs.groupby("method", as_index=False).agg(
        runs=("success", "size"),
        target_rows=("selected_rows", "sum"),
        request_calls=("request_calls", "sum"),
        total_tokens=("total_tokens", "sum"),
        cost_usd=("cost_usd", "sum"),
        generation_time_seconds=("generation_time_seconds", "sum"),
        numeric_within_duplicates=("numeric_within_duplicates", "sum"),
        numeric_train_matches=("numeric_train_matches", "sum"),
    )

    return {
        "study_counts": {
            "datasets": 8,
            "seeds": 10,
            "main_methods": 6,
            "downstream_classifiers": 3,
            "main_generation_runs": 480,
            "main_classifier_outcomes": 1440,
            "new_no_augmentation_runs": 160,
            "new_no_augmentation_classifier_outcomes": 480,
            "threshold_candidate_pools": 24,
            "threshold_conditions": 9,
            "threshold_condition_runs": 216,
            "threshold_classifier_outcomes": 648,
            "backend_runs": 24,
            "backend_target_rows": int(backend_runs["selected_rows"].sum()),
        },
        "six_method_performance": {
            "arithmetic_summary": records(arithmetic),
            "supplementary_noaug_unweighted": records(
                arithmetic_all.loc[arithmetic_all["method"] == "no_augmentation_unweighted"]
            )[0],
            "f1_mean_ranks": records(f1_ranks),
            "roc_auc_mean_ranks": records(auc_ranks),
            "f1_friedman": records(main_omnibus.loc[main_omnibus["metric"] == "f1"])[0],
            "roc_auc_friedman": records(main_omnibus.loc[main_omnibus["metric"] == "roc_auc"])[0],
            "qualsynth_vs_noaug_cw_f1": {
                key: clean_value(q_vs_noaug_f1[key]) for key in q_vs_noaug_f1.index
            },
            "qualsynth_vs_noaug_cw_roc_auc": {
                key: clean_value(q_vs_noaug_auc[key]) for key in q_vs_noaug_auc.index
            },
            "qualsynth_vs_ctgan_roc_auc": {
                key: clean_value(q_vs_ctgan_auc[key]) for key in q_vs_ctgan_auc.index
            },
            "qualsynth_vs_tabddpm_roc_auc": {
                key: clean_value(q_vs_tabddpm_auc[key])
                for key in q_vs_tabddpm_auc.index
            },
        },
        "threshold_sensitivity": {
            "std_3": {key: clean_value(z3[key]) for key in z3.index},
            "std_4": {key: clean_value(z4[key]) for key in z4.index},
            "std_5": {key: clean_value(z5[key]) for key in z5.index},
            "calibrated_z95": {key: clean_value(calibrated[key]) for key in calibrated.index},
            "percentile_acceptance_min": float(percentiles["acceptance_rate_mean"].min()),
            "percentile_acceptance_max": float(percentiles["acceptance_rate_mean"].max()),
            "calibrated_threshold_min": float(calibration["threshold"].min()),
            "calibrated_threshold_max": float(calibration["threshold"].max()),
            "calibrated_thresholds_below_4": calibration_below,
            "calibrated_thresholds_above_4": calibration_above,
            "numeric_equivalent_duplicate_survivors_std4": int(
                threshold_per_run.loc[
                    threshold_per_run["condition"] == "std_4",
                    "n_numeric_equivalent_duplicate_survivors",
                ].sum()
            ),
            "accepted_rows_std4": int(
                threshold_per_run.loc[
                    threshold_per_run["condition"] == "std_4",
                    "n_accepted_pre_cap",
                ].sum()
            ),
            "fixed_pool_target_attained_runs": int(
                (threshold_per_run["target_shortfall"] == 0).sum()
            ),
        },
        "backend_sensitivity": {
            "model_means": records(backend_summary),
            "paired_differences": records(backend_paired),
            "totals_by_model": records(backend_by_method),
            "total_cost_usd": float(backend_runs["cost_usd"].sum()),
            "total_generation_time_seconds": float(backend_runs["generation_time_seconds"].sum()),
        },
        "interpretation_limits": {
            "backend": "Two current routes test sensitivity, not universal backend invariance.",
            "threshold": "Fixed-pool post-hoc sensitivity; not fresh target-completing generation.",
            "privacy": "Local deployment can reduce third-party disclosure but is not automatically private.",
        },
    }


def build_macros(claims: dict[str, Any]) -> dict[str, str]:
    performance = claims["six_method_performance"]
    threshold = claims["threshold_sensitivity"]
    backend = claims["backend_sensitivity"]
    counts = claims["study_counts"]

    def rank(metric: str, method: str) -> float:
        rows = performance[f"{metric}_mean_ranks"]
        return float(next(row["mean_rank"] for row in rows if row["method"] == method))

    backend_means = {row["method"]: row for row in backend["model_means"]}
    backend_totals = {row["method"]: row for row in backend["totals_by_model"]}
    backend_differences = {row["metric"]: row for row in backend["paired_differences"]}
    arithmetic = {row["method"]: row for row in performance["arithmetic_summary"]}
    q_noaug_f1 = performance["qualsynth_vs_noaug_cw_f1"]
    q_noaug_auc = performance["qualsynth_vs_noaug_cw_roc_auc"]
    q_ctgan_auc = performance["qualsynth_vs_ctgan_roc_auc"]
    q_tabddpm_auc = performance["qualsynth_vs_tabddpm_roc_auc"]
    noaug_unweighted = performance["supplementary_noaug_unweighted"]
    f1_friedman = performance["f1_friedman"]
    auc_friedman = performance["roc_auc_friedman"]
    return {
        "RThreeMainGenerationRuns": str(counts["main_generation_runs"]),
        "RThreeMainMethodRuns": str(counts["main_generation_runs"]),
        "RThreeMainClassifierOutcomes": str(counts["main_classifier_outcomes"]),
        "RThreeNewNoAugRuns": str(counts["new_no_augmentation_runs"]),
        "RThreeNewNoAugClassifierOutcomes": str(counts["new_no_augmentation_classifier_outcomes"]),
        "RThreeNoAugFOneRank": fmt(rank("f1", "no_augmentation"), 2),
        "RThreeQualSynthFOneRank": fmt(rank("f1", "qualsynth"), 2),
        "RThreeNoAugAUCRank": fmt(rank("roc_auc", "no_augmentation"), 2),
        "RThreeQualSynthAUCRank": fmt(rank("roc_auc", "qualsynth"), 2),
        "RThreeNoAugFOneMean": fmt(arithmetic["no_augmentation"]["f1_mean"], 3),
        "RThreeQualSynthFOneMean": fmt(arithmetic["qualsynth"]["f1_mean"], 3),
        "RThreeNoAugAUCMean": fmt(arithmetic["no_augmentation"]["roc_auc_mean"], 3),
        "RThreeQualSynthAUCMean": fmt(arithmetic["qualsynth"]["roc_auc_mean"], 3),
        "RThreeNoAugUnweightedFOneMean": fmt(noaug_unweighted["f1_mean"], 3),
        "RThreeNoAugUnweightedAUCMean": fmt(noaug_unweighted["roc_auc_mean"], 3),
        "RThreeQualSynthNoAugFOneDelta": fmt(q_noaug_f1["mean_difference"], 4),
        "RThreeQualSynthNoAugFOneHolmP": fmt(q_noaug_f1["holm_p_value"], 4),
        "RThreeQualSynthNoAugAUCDelta": fmt(q_noaug_auc["mean_difference"], 4),
        "RThreeQualSynthNoAugAUCHolmP": fmt(q_noaug_auc["holm_p_value"], 4),
        "RThreeQualSynthCtganAUCHolmP": fmt(q_ctgan_auc["holm_p_value"], 4),
        "RThreeQualSynthTabddpmAUCHolmP": fmt(
            q_tabddpm_auc["holm_p_value"], 4
        ),
        "RThreeFOneFriedmanStatistic": fmt(f1_friedman["friedman_statistic"], 2),
        "RThreeFOneFriedmanP": fmt(f1_friedman["friedman_p_value"], 4),
        "RThreeAUCFriedmanStatistic": fmt(auc_friedman["friedman_statistic"], 2),
        "RThreeAUCFriedmanP": fmt(auc_friedman["friedman_p_value"], 4),
        "RThreeThresholdPools": str(counts["threshold_candidate_pools"]),
        "RThreeThresholdConditions": str(counts["threshold_conditions"]),
        "RThreeThresholdRuns": str(counts["threshold_condition_runs"]),
        "RThreeStdThreeAcceptance": fmt(threshold["std_3"]["acceptance_rate_mean"], 3),
        "RThreeStdFourAcceptance": fmt(threshold["std_4"]["acceptance_rate_mean"], 3),
        "RThreeStdFiveAcceptance": fmt(threshold["std_5"]["acceptance_rate_mean"], 3),
        "RThreePercentileAcceptanceMin": fmt(threshold["percentile_acceptance_min"], 3),
        "RThreePercentileAcceptanceMax": fmt(threshold["percentile_acceptance_max"], 3),
        "RThreeThresholdFOneMin": fmt(
            min(threshold[name]["f1_mean"] for name in ["std_3", "std_4", "std_5"]),
            3,
        ),
        "RThreeThresholdFOneMax": fmt(
            max(threshold[name]["f1_mean"] for name in ["std_3", "std_4", "std_5"]),
            3,
        ),
        "RThreeCalibratedThresholdMin": fmt(threshold["calibrated_threshold_min"], 3),
        "RThreeCalibratedThresholdMax": fmt(threshold["calibrated_threshold_max"], 3),
        "RThreeCalibratedBelowFour": str(threshold["calibrated_thresholds_below_4"]),
        "RThreeCalibratedAboveFour": str(threshold["calibrated_thresholds_above_4"]),
        "RThreeStdFourDuplicateSurvivors": str(
            threshold["numeric_equivalent_duplicate_survivors_std4"]
        ),
        "RThreeStdFourAcceptedRows": str(threshold["accepted_rows_std4"]),
        "RThreeThresholdTargetAttainedRuns": str(threshold["fixed_pool_target_attained_runs"]),
        "RThreeConfidenceLevel": "95",
        "RThreeBackendRuns": str(counts["backend_runs"]),
        "RThreeBackendRows": str(counts["backend_target_rows"]),
        "RThreeGemmaFOne": fmt(backend_means["qualsynth_r3_gemma"]["f1"], 3),
        "RThreeLunaFOne": fmt(backend_means["qualsynth_r3_luna"]["f1"], 3),
        "RThreeGemmaAUC": fmt(backend_means["qualsynth_r3_gemma"]["roc_auc"], 3),
        "RThreeLunaAUC": fmt(backend_means["qualsynth_r3_luna"]["roc_auc"], 3),
        "RThreeGemmaAcceptance": fmt(
            backend_means["qualsynth_r3_gemma"]["validation_acceptance_rate"], 3
        ),
        "RThreeLunaAcceptance": fmt(
            backend_means["qualsynth_r3_luna"]["validation_acceptance_rate"], 3
        ),
        "RThreeGemmaCalls": str(int(backend_totals["qualsynth_r3_gemma"]["request_calls"])),
        "RThreeLunaCalls": str(int(backend_totals["qualsynth_r3_luna"]["request_calls"])),
        "RThreeGemmaTokensK": fmt(
            backend_totals["qualsynth_r3_gemma"]["total_tokens"] / 1000.0,
            1,
        ),
        "RThreeLunaTokensK": fmt(
            backend_totals["qualsynth_r3_luna"]["total_tokens"] / 1000.0,
            1,
        ),
        "RThreeGemmaRuntimeHours": fmt(
            backend_totals["qualsynth_r3_gemma"]["generation_time_seconds"] / 3600.0,
            2,
        ),
        "RThreeLunaRuntimeHours": fmt(
            backend_totals["qualsynth_r3_luna"]["generation_time_seconds"] / 3600.0,
            2,
        ),
        "RThreeGemmaCost": fmt(backend_totals["qualsynth_r3_gemma"]["cost_usd"], 2),
        "RThreeLunaCost": fmt(backend_totals["qualsynth_r3_luna"]["cost_usd"], 2),
        "RThreeGemmaDuplicates": str(
            int(backend_totals["qualsynth_r3_gemma"]["numeric_within_duplicates"])
        ),
        "RThreeLunaDuplicates": str(
            int(backend_totals["qualsynth_r3_luna"]["numeric_within_duplicates"])
        ),
        "RThreeGemmaTrainMatches": str(
            int(backend_totals["qualsynth_r3_gemma"]["numeric_train_matches"])
        ),
        "RThreeLunaTrainMatches": str(
            int(backend_totals["qualsynth_r3_luna"]["numeric_train_matches"])
        ),
        "RThreeBackendFOneDelta": fmt(backend_differences["f1"]["luna_minus_gemma_mean"], 4),
        "RThreeBackendFOneCILow": fmt(
            backend_differences["f1"]["hierarchical_bootstrap_ci_low"], 4
        ),
        "RThreeBackendFOneCIHigh": fmt(
            backend_differences["f1"]["hierarchical_bootstrap_ci_high"], 4
        ),
        "RThreeBackendAUCDelta": fmt(backend_differences["roc_auc"]["luna_minus_gemma_mean"], 4),
        "RThreeBackendAUCCILow": fmt(
            backend_differences["roc_auc"]["hierarchical_bootstrap_ci_low"], 4
        ),
        "RThreeBackendAUCCIHigh": fmt(
            backend_differences["roc_auc"]["hierarchical_bootstrap_ci_high"], 4
        ),
        "RThreeBackendAcceptanceDelta": fmt(
            backend_differences["validation_acceptance_rate"]["luna_minus_gemma_mean"], 4
        ),
        "RThreeBackendAcceptanceCILow": fmt(
            backend_differences["validation_acceptance_rate"]["hierarchical_bootstrap_ci_low"],
            4,
        ),
        "RThreeBackendAcceptanceCIHigh": fmt(
            backend_differences["validation_acceptance_rate"]["hierarchical_bootstrap_ci_high"],
            4,
        ),
        "RThreeBackendYieldDelta": fmt(
            backend_differences["row_yield"]["luna_minus_gemma_mean"], 4
        ),
        "RThreeBackendYieldCILow": fmt(
            backend_differences["row_yield"]["hierarchical_bootstrap_ci_low"], 4
        ),
        "RThreeBackendYieldCIHigh": fmt(
            backend_differences["row_yield"]["hierarchical_bootstrap_ci_high"], 4
        ),
        "RThreeBackendWassersteinDelta": fmt(
            backend_differences["standardized_wasserstein"]["luna_minus_gemma_mean"], 4
        ),
        "RThreeBackendWassersteinCILow": fmt(
            backend_differences["standardized_wasserstein"]["hierarchical_bootstrap_ci_low"],
            4,
        ),
        "RThreeBackendWassersteinCIHigh": fmt(
            backend_differences["standardized_wasserstein"]["hierarchical_bootstrap_ci_high"],
            4,
        ),
        "RThreeBackendCost": fmt(backend["total_cost_usd"], 2),
    }


def correction_families() -> list[dict[str, Any]]:
    return [
        {
            "id": "primary_six_method_f1",
            "unit": "eight dataset means",
            "omnibus": "Friedman across six methods",
            "post_hoc": "five QualSynth-versus-comparator Wilcoxon tests",
            "correction": "Holm within the F1 family",
        },
        {
            "id": "primary_six_method_roc_auc",
            "unit": "eight dataset means",
            "omnibus": "Friedman across six methods",
            "post_hoc": "five QualSynth-versus-comparator Wilcoxon tests",
            "correction": "Holm within the ROC-AUC family",
        },
        {
            "id": "noaug_dataset_seed_pairs",
            "unit": "ten paired seeds per dataset",
            "post_hoc": "QualSynth versus NoAug-CW in eight datasets",
            "correction": "Holm across datasets, separately for F1 and ROC-AUC",
        },
        {
            "id": "threshold_vs_std4",
            "unit": "eight paired dataset means; seeds averaged within dataset",
            "post_hoc": "each sensitivity condition versus std_4",
            "correction": "Holm within each outcome metric",
        },
        {
            "id": "backend_sensitivity",
            "unit": "four datasets with three paired seeds",
            "analysis": "hierarchical paired bootstrap confidence intervals",
            "correction": "none; descriptive sensitivity estimates, no hypothesis tests",
        },
        {
            "id": "component_ablation",
            "unit": "paired dataset-seed-classifier or dataset-seed runs as archived",
            "correction": "archived Holm family across component metrics",
        },
    ]


METHOD_DISPLAY = {
    "qualsynth": "QualSynth",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tabfairgdt": "TabFairGDT",
    "tabddpm": "TabDDPM",
    "no_augmentation": "NoAug-CW",
}

DATASET_DISPLAY = {
    "breast_cancer": "Breast Cancer",
    "german_credit": "German Credit",
    "haberman": "Haberman",
    "htru2": "HTRU2",
    "pima_diabetes": "Pima Diabetes",
    "thyroid": "Thyroid",
    "wine_quality": "Wine Quality",
    "yeast": "Yeast",
}


def generated_header() -> list[str]:
    return [
        "% Generated by scripts/build_reviewer3_round2_claims.py; do not edit.",
    ]


def performance_tex(tables: dict[str, pd.DataFrame], metric: str) -> str:
    per_run = tables["performance_per_run"]
    summary = (
        per_run.groupby(["dataset", "method"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    ranks = tables[f"performance_{metric}_ranks"].set_index("method")["mean_rank"]
    synthetic_dataset_means = (
        per_run.loc[per_run["method"].isin(SYNTHETIC_METHODS)]
        .groupby(["dataset", "method"])[metric]
        .mean()
        .unstack()
    )
    synthetic_ranks = synthetic_dataset_means.rank(
        axis=1, ascending=False, method="average"
    ).mean(axis=0)
    metric_label = "F1" if metric == "f1" else "ROC-AUC"
    label_suffix = "f1" if metric == "f1" else "auc"
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        f"\\caption{{{metric_label} by dataset for two complementary comparisons: "
        "five synthetic oversampling methods and the non-generative NoAug-CW "
        "reference (mean $\\pm$ standard deviation across ten seed-level classifier "
        "means). Bold identifies the best synthetic result. Overall ranks additionally "
        "compare the six training strategies on predictive utility.}",
        f"\\label{{tab:round2_{label_suffix}}}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{5}{c}{Synthetic oversampling methods} & "
        "\\multicolumn{1}{c}{Non-generative reference} \\\\ ",
        "\\cmidrule(lr){2-6}\\cmidrule(l){7-7}",
        "Dataset & QualSynth & SMOTE & CTGAN & TabFairGDT & TabDDPM & NoAug-CW \\\\",
        "\\midrule",
    ]
    for dataset in DATASET_DISPLAY:
        subset = summary.loc[summary["dataset"] == dataset].set_index("method")
        best = float(subset.loc[SYNTHETIC_METHODS, "mean"].max())
        cells = []
        for method in MAIN_METHODS:
            mean = float(subset.loc[method, "mean"])
            std = float(subset.loc[method, "std"])
            cell = f"{mean:.3f}$\\pm${std:.3f}"
            if method in SYNTHETIC_METHODS and math.isclose(
                mean, best, rel_tol=0.0, abs_tol=1e-12
            ):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{DATASET_DISPLAY[dataset]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\midrule"])
    synthetic_rank_cells = []
    best_synthetic_rank = float(synthetic_ranks.loc[SYNTHETIC_METHODS].min())
    for method in SYNTHETIC_METHODS:
        rank_value = float(synthetic_ranks.loc[method])
        rank_cell = f"{rank_value:.3f}"
        if math.isclose(rank_value, best_synthetic_rank, rel_tol=0.0, abs_tol=1e-12):
            rank_cell = f"\\textbf{{{rank_cell}}}"
        synthetic_rank_cells.append(rank_cell)

    best_rank = float(ranks.loc[MAIN_METHODS].min())
    rank_cells = []
    for method in MAIN_METHODS:
        rank_value = float(ranks.loc[method])
        rank_cell = f"{rank_value:.3f}"
        if math.isclose(rank_value, best_rank, rel_tol=0.0, abs_tol=1e-12):
            rank_cell = f"\\textbf{{{rank_cell}}}"
        rank_cells.append(rank_cell)
    lines.extend(
        [
            "Synthetic-only average rank & "
            + " & ".join(synthetic_rank_cells)
            + " & -- \\\\ ",
            "Overall predictive-utility rank & " + " & ".join(rank_cells) + " \\\\ ",
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def statistical_tex(tables: dict[str, pd.DataFrame]) -> str:
    omnibus = tables["performance_omnibus"]
    pairwise = tables["performance_pairwise"]
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Dataset-level predictive-utility inference across six training "
        "strategies. The five synthetic oversamplers form the methodological "
        "comparison; NoAug-CW is a non-generative reference. Friedman tests use eight "
        "dataset means. Pairwise Wilcoxon tests compare QualSynth with the four other "
        "synthetic methods and, separately, with NoAug-CW; Holm correction is applied "
        "to the five comparisons within each metric.}",
        "\\label{tab:round2_stats}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Metric & Friedman $\\chi_F^2$ & $p$ & $\\Delta$ vs NoAug-CW & "
        "$p_H$ vs NoAug-CW & $p_H$ vs CTGAN \\\\",
        "\\midrule",
    ]
    for metric, label in [("f1", "F1"), ("roc_auc", "ROC-AUC")]:
        o = one(omnibus, scope="eight_dataset_means", metric=metric)
        noaug = one(
            pairwise,
            scope="eight_dataset_means",
            metric=metric,
            baseline="no_augmentation",
        )
        ctgan = one(
            pairwise,
            scope="eight_dataset_means",
            metric=metric,
            baseline="ctgan",
        )
        lines.append(
            f"{label} & {float(o['friedman_statistic']):.3f} & "
            f"{float(o['friedman_p_value']):.4f} & "
            f"{float(noaug['mean_difference']):+.4f} & "
            f"{float(noaug['holm_p_value']):.4f} & "
            f"{float(ctgan['holm_p_value']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def threshold_tex(tables: dict[str, pd.DataFrame]) -> str:
    summary = tables["threshold_conditions"].set_index("condition")
    calibration = (
        tables["threshold_calibration"]
        .groupby("dataset", as_index=False)
        .agg(
            minority_count=("minority_count", "first"),
            numeric_feature_count=("numeric_feature_count", "first"),
            threshold_mean=("threshold", "mean"),
            threshold_min=("threshold", "min"),
            threshold_max=("threshold", "max"),
            achieved_retention_mean=("achieved_retention", "mean"),
        )
        .set_index("dataset")
    )
    conditions = [
        ("std_3", "3-sigma"),
        ("std_4", "4-sigma"),
        ("std_5", "5-sigma"),
        ("percentile_0_995", "Central 99.5\\% interval"),
        ("calibrated_z95", "Train-calibrated 95\\%"),
        ("no_statistical_filter", "No statistical filter"),
    ]
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Fixed-pool validation sensitivity over 24 dataset--seed candidate "
        "pools. Values are equal-pool means; lower Wasserstein distance and higher "
        "pairwise diversity indicate different sides of the safety--diversity trade-off.}",
        "\\label{tab:round2_threshold}",
        "\\scriptsize",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Condition & Acceptance & Wasserstein & Pairwise diversity & F1 & ROC-AUC \\\\",
        "\\midrule",
    ]
    for condition, label in conditions:
        row = summary.loc[condition]
        lines.append(
            f"{label} & {row['acceptance_rate_mean']:.3f} & "
            f"{row['standardized_wasserstein_mean']:.3f} & "
            f"{row['pairwise_diversity_mean']:.3f} & {row['f1_mean']:.3f} & "
            f"{row['roc_auc_mean']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[ht!]",
            "\\centering",
            "\\caption{Training-only threshold calibration by dataset. Minority count "
            "and numeric dimension are constant across the three seeds; cutoffs and "
            "retention are seed summaries. The non-monotonic pattern indicates that "
            "minority size alone does not determine the calibrated cutoff.}",
            "\\label{tab:round2_threshold_calibration}",
            "\\small",
            "\\begin{tabular}{@{}lrrrrr@{}}",
            "\\toprule",
            "Dataset & Minority $n$ & Numeric $d$ & Mean cutoff & Cutoff range & Mean retention \\\\",
            "\\midrule",
        ]
    )
    for dataset in DATASET_DISPLAY:
        row = calibration.loc[dataset]
        lines.append(
            f"{DATASET_DISPLAY[dataset]} & {int(row['minority_count'])} & "
            f"{int(row['numeric_feature_count'])} & {row['threshold_mean']:.3f} & "
            f"{row['threshold_min']:.3f}--{row['threshold_max']:.3f} & "
            f"{row['achieved_retention_mean']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def backend_tex(tables: dict[str, pd.DataFrame]) -> str:
    summary = tables["backend_model_summary"].set_index("method")
    runs = tables["backend_per_run"]
    totals = runs.groupby("method").agg(
        total_tokens=("total_tokens", "sum"),
        generation_time_seconds=("generation_time_seconds", "sum"),
        cost_usd=("cost_usd", "sum"),
    )
    labels = [
        ("qualsynth_r3_gemma", "Gemma 3 27B"),
        ("qualsynth_r3_luna", "GPT-5.6 Luna Pro"),
    ]
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Paired backend sensitivity across four datasets and three seeds. "
        "All runs used the same provider-neutral request contract and implementation. "
        "Calls and quality/utility metrics are run means; tokens, generation time, "
        "and recorded provider charge are totals.}",
        "\\label{tab:round2_backend}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        "\\toprule",
        "Backend & Calls/run & Tokens (k) & Time (h) & Acceptance & Duplicate rate & Train-match rate & Wasserstein & F1 & ROC-AUC & Cost (USD) \\\\",
        "\\midrule",
    ]
    for method, label in labels:
        row = summary.loc[method]
        total = totals.loc[method]
        lines.append(
            f"{label} & {row['request_calls']:.1f} & "
            f"{float(total['total_tokens']) / 1000.0:.1f} & "
            f"{float(total['generation_time_seconds']) / 3600.0:.2f} & "
            f"{row['validation_acceptance_rate']:.3f} & "
            f"{row['numeric_within_duplicate_rate']:.4f} & "
            f"{row['numeric_train_match_rate']:.4f} & "
            f"{row['standardized_wasserstein']:.3f} & {row['f1']:.3f} & "
            f"{row['roc_auc']:.3f} & {float(total['cost_usd']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def classifier_tex(tables: dict[str, pd.DataFrame]) -> str:
    summary = tables["performance_classifier_summary"].set_index(["classifier", "method"])
    classifiers = [
        ("LogisticRegression", "Logistic Regression"),
        ("RandomForest", "Random Forest"),
        ("XGBoost", "XGBoost"),
    ]
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Classifier-resolved predictive utility for five synthetic "
        "oversamplers and the non-generative NoAug-CW reference. Each cell reports "
        "mean F1 / mean ROC-AUC across eight datasets and ten seeds.}",
        "\\label{tab:round2_classifier}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{5}{c}{Synthetic oversampling methods} & "
        "\\multicolumn{1}{c}{Non-generative reference} \\\\ ",
        "\\cmidrule(lr){2-6}\\cmidrule(l){7-7}",
        "Classifier & QualSynth & SMOTE & CTGAN & TabFairGDT & TabDDPM & NoAug-CW \\\\",
        "\\midrule",
    ]
    for classifier, label in classifiers:
        cells = []
        for method in MAIN_METHODS:
            row = summary.loc[(classifier, method)]
            cells.append(f"{row['f1_mean']:.3f} / {row['roc_auc_mean']:.3f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def quality_tex(tables: dict[str, pd.DataFrame]) -> str:
    quality = tables["quality_audit"].set_index("method")
    methods = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Candidate-quality audit over 80 runs per method. Exact "
        "duplicate rates are measured after canonicalized numeric screening and are not semantic "
        "privacy or universal-novelty guarantees. Acceptance uses QualSynth-oriented "
        "checks and is therefore diagnostic rather than a neutral utility comparison.}",
        "\\label{tab:round2_quality}",
        "\\small",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Method & Acceptance (\\%) & Exact duplicates (\\%) & Correlation distance \\\\",
        "\\midrule",
    ]
    for method in methods:
        row = quality.loc[method]
        lines.append(
            f"{METHOD_DISPLAY[method]} & {100 * row['mean_acceptance_rate']:.2f} & "
            f"{100 * row['mean_raw_duplicate_rate']:.4f} & "
            f"{row['mean_correlation_distance']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def component_tex(tables: dict[str, pd.DataFrame]) -> str:
    frame = tables["component_ablation"]
    selections = [
        (
            "qualsynth_component_no_anchor_prompt",
            "corr_distance",
            "Anchor prompting",
            "Correlation distance",
        ),
        ("qualsynth_component_no_validation_raw", "duplicate_rate", "Validation", "Duplicate rate"),
        (
            "qualsynth_component_no_validation_raw",
            "wasserstein_norm",
            "Validation",
            "Wasserstein distance",
        ),
        (
            "qualsynth_component_no_objective",
            "corr_distance",
            "Optional selector",
            "Correlation distance",
        ),
        (
            "qualsynth_component_no_objective",
            "mean_pairwise_dist",
            "Optional selector",
            "Pairwise distance",
        ),
    ]
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Paired component ablation. The comparison mean is the component-removed "
        "variant; $p_H$ is the Holm-adjusted Wilcoxon value.}",
        "\\label{tab:round2_component}",
        "\\small",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Component & Metric & Full & Removed & Difference & $p_H$ \\\\",
        "\\midrule",
    ]
    for comparison, metric, component, label in selections:
        row = one(frame, comparison=comparison, metric=metric)
        scientific = abs(float(row["mean_cmp"])) >= 1e6

        def number(value: Any) -> str:
            return f"{float(value):.2e}" if scientific else f"{float(value):.3f}"

        lines.append(
            f"{component} & {label} & {number(row['mean_ref'])} & "
            f"{number(row['mean_cmp'])} & {number(row['delta'])} & "
            f"{float(row['wilcoxon_p_holm']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def high_dimensional_tex(tables: dict[str, pd.DataFrame]) -> str:
    sweep = tables["high_dimensional"]
    cross = tables["high_dimensional_cross_dataset"]
    selections: list[tuple[pd.Series, str]] = []
    for dataset, k, methods in [
        ("alon_colon", 50, ["qualsynth", "smote", "tabddpm"]),
        ("alon_colon_k200", 200, ["qualsynth", "smote"]),
        ("alon_colon_k500", 500, ["qualsynth", "smote"]),
    ]:
        for method in methods:
            selections.append(
                (one(sweep, dataset=dataset, k_selected=k, method=method), f"Alon $k={k}$")
            )
    for method in ["qualsynth", "smote"]:
        selections.append((one(cross, dataset="golub_leukemia", method=method), "Golub $k=50$"))
    for dataset, k in [("golub_leukemia_k200", 200), ("golub_leukemia_k500", 500)]:
        selections.append(
            (one(sweep, dataset=dataset, k_selected=k, method="qualsynth"), f"Golub $k={k}$")
        )
    lines = generated_header() + [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{High-dimensional extension (three-seed means). Lower is better for "
        "Wasserstein, correlation distance, duplicate rate, and range violations.}",
        "\\label{tab:round2_high_dim}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}llr@{\\hspace{0.8em}}rrrrr@{}}",
        "\\toprule",
        "Dataset / setting & Method & Wasserstein & Corr. distance & Duplicate & Range violation & F1 & ROC-AUC \\\\",
        "\\midrule",
    ]
    for row, setting in selections:
        lines.append(
            f"{setting} & {METHOD_DISPLAY[row['method']]} & "
            f"{row['wasserstein_norm']:.3f} & {row['corr_distance']:.3f} & "
            f"{row['duplicate_rate']:.3f} & {row['range_violation_rate']:.3f} & "
            f"{row['avg_f1']:.3f} & {row['avg_roc_auc']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def historical_cost_tex(tables: dict[str, pd.DataFrame]) -> str:
    frame = tables["historical_cost"].set_index("dataset")
    lines = generated_header() + [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Per-run runtime and request-volume analysis. QualSynth calls and "
        "tokens come from instrumented full-pipeline component runs.}",
        "\\label{tab:round2_cost}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Dataset & QualSynth (s) & SMOTE (s) & CTGAN (s) & TabDDPM (s) & Calls & Tokens (k) \\\\",
        "\\midrule",
    ]
    for dataset in DATASET_DISPLAY:
        row = frame.loc[dataset]
        lines.append(
            f"{DATASET_DISPLAY[dataset]} & "
            f"{row['qualsynth_mean_generation_time_seconds']:.1f} & "
            f"{row['smote_mean_generation_time_seconds']:.1f} & "
            f"{row['ctgan_mean_generation_time_seconds']:.1f} & "
            f"{row['tabddpm_mean_generation_time_seconds']:.1f} & "
            f"{row['qualsynth_mean_llm_calls']:.1f} & "
            f"{row['qualsynth_mean_total_tokens_k']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def write_generated_tex(tables: dict[str, pd.DataFrame]) -> list[Path]:
    GENERATED_TEX_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "round2_performance_f1.tex": performance_tex(tables, "f1"),
        "round2_performance_roc_auc.tex": performance_tex(tables, "roc_auc"),
        "round2_statistical_tests.tex": statistical_tex(tables),
        "round2_threshold_sensitivity.tex": threshold_tex(tables),
        "round2_backend_sensitivity.tex": backend_tex(tables),
        "round2_classifier_summary.tex": classifier_tex(tables),
        "round2_quality_audit.tex": quality_tex(tables),
        "round2_component_ablation.tex": component_tex(tables),
        "round2_high_dimensional.tex": high_dimensional_tex(tables),
        "round2_historical_cost.tex": historical_cost_tex(tables),
    }
    paths = []
    for filename, content in payloads.items():
        path = GENERATED_TEX_ROOT / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def source_files() -> list[Path]:
    return [
        PROJECT_ROOT / "configs/experiments/reviewer3_backend_sensitivity.yaml",
        PROJECT_ROOT / "configs/experiments/reviewer3_no_augmentation.yaml",
        PROJECT_ROOT / "configs/experiments/reviewer3_threshold_sensitivity.yaml",
        PROJECT_ROOT / "configs/methods/qualsynth.yaml",
        PROJECT_ROOT / "configs/methods/no_augmentation.yaml",
        PROJECT_ROOT / "configs/methods/no_augmentation_unweighted.yaml",
        PROJECT_ROOT / "configs/methods/qualsynth_r3_gemma.yaml",
        PROJECT_ROOT / "configs/methods/qualsynth_r3_luna.yaml",
        PROJECT_ROOT / "scripts/build_reviewer3_round2_claims.py",
        PROJECT_ROOT / "scripts/analyze_reviewer3_no_augmentation.py",
        PROJECT_ROOT / "scripts/reanalyze_thyroid_consistent_encoding.py",
        PROJECT_ROOT / "scripts/run_reviewer3_backend_sensitivity.py",
        PROJECT_ROOT / "scripts/run_reviewer3_threshold_sensitivity.py",
        PROJECT_ROOT / "scripts/analyze_reviewer3_backend_sensitivity.py",
        PROJECT_ROOT / "scripts/run_experiments.py",
        PROJECT_ROOT / "scripts/run_openrouter_experiments.py",
        PROJECT_ROOT / "src/qualsynth/core/iterative_workflow.py",
        PROJECT_ROOT / "src/qualsynth/evaluation/classifiers.py",
        PROJECT_ROOT / "src/qualsynth/generator.py",
        PROJECT_ROOT / "src/qualsynth/generators/counterfactual_generator.py",
        PROJECT_ROOT / "src/qualsynth/validation/threshold_calibration.py",
        PROJECT_ROOT / "docs/REVIEWER3_METHOD_CODE_CONCORDANCE.md",
        PROJECT_ROOT / "reviewer_comments.md",
        REVISION_ROOT
        / "canonical_dedup_correction/openrouter1/thyroid_retained_sample_manifest.json",
        REVISION_ROOT / "thyroid_correction/analysis/REPORT.md",
        PROJECT_ROOT / "sreport/main_round1_received.tex",
        ROUND2_ROOT / "provenance_manifest.json",
        NOAUG_ROOT / "manifest.json",
        NOAUG_ROOT / "completeness.csv",
        NOAUG_ROOT / "six_method_dataset_means_f1.csv",
        NOAUG_ROOT / "six_method_dataset_means_roc_auc.csv",
        NOAUG_ROOT / "six_method_dataset_mean_ranks_f1.csv",
        NOAUG_ROOT / "six_method_dataset_mean_ranks_roc_auc.csv",
        NOAUG_ROOT / "classifier_resolved_summary.csv",
        NOAUG_ROOT / "all_condition_arithmetic_summary.csv",
        NOAUG_ROOT / "per_run_metrics.csv",
        NOAUG_ROOT / "six_method_omnibus_tests.csv",
        NOAUG_ROOT / "six_method_qualsynth_pairwise_holm.csv",
        NOAUG_ROOT / "qualsynth_vs_noaug_cw_by_dataset_f1.csv",
        NOAUG_ROOT / "qualsynth_vs_noaug_cw_by_dataset_roc_auc.csv",
        THRESHOLD_ROOT / "manifest.json",
        THRESHOLD_ROOT / "condition_summary_bootstrap.csv",
        THRESHOLD_ROOT / "dataset_condition_summary.csv",
        THRESHOLD_ROOT / "per_run_metrics.csv",
        THRESHOLD_ROOT / "training_only_calibration.csv",
        THRESHOLD_ROOT / "paired_vs_std4.csv",
        THRESHOLD_ROOT / "minority_size_associations.csv",
        BACKEND_ROOT / "analysis_manifest.json",
        BACKEND_ROOT / "REPORT.md",
        BACKEND_ROOT / "per_run_backend_metrics.csv",
        BACKEND_ROOT / "model_summary.csv",
        BACKEND_ROOT / "paired_differences.csv",
        HISTORICAL_ROOT / "bundle_manifest.json",
        HISTORICAL_ROOT / "quality_audit_summary.csv",
        COMPONENT_ROOT / "component_paired_tests.csv",
        HIGH_DIM_ROOT / "k_sweep_table.csv",
        HIGH_DIM_ROOT / "cross_dataset_table.csv",
        REVISION_ROOT / "cost_runtime_manuscript_table.csv",
    ]


def write_outputs(
    gate: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    claims: dict[str, Any],
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    macros = build_macros(claims)
    source_manifest = {
        str(require_file(path).relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in source_files()
    }
    bundle = {
        "schema_version": "reviewer3_round2_claims_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_gates": gate["gates"],
        "claims": claims,
        "manuscript_macros": macros,
        "tables": {name: records(frame) for name, frame in tables.items()},
        "correction_families": correction_families(),
        "source_sha256": source_manifest,
    }
    claim_path = OUTPUT_ROOT / "round2_claims.json"
    claim_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    for name, frame in tables.items():
        frame.to_csv(OUTPUT_ROOT / f"table_{name}.csv", index=False)

    macro_lines = [
        "% Generated by scripts/build_reviewer3_round2_claims.py; do not edit.",
        f"% round2_claims.json SHA-256: {sha256_file(claim_path)}",
    ]
    macro_lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    MACRO_PATH.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    generated_tex_paths = write_generated_tex(tables)

    completeness = pd.DataFrame(
        [
            {
                "evidence_family": "no_augmentation_all_methods",
                "expected": 560,
                "observed": len(gate["noaug_completeness"]),
                "failures": int((~gate["noaug_completeness"]["exists"]).sum()),
                "gate_pass": gate["gates"]["no_augmentation"],
            },
            {
                "evidence_family": "threshold_condition_runs",
                "expected": 216,
                "observed": gate["threshold_manifest"]["condition_runs"],
                "failures": len(gate["threshold_manifest"].get("failures", [])),
                "gate_pass": gate["gates"]["threshold_sensitivity"],
            },
            {
                "evidence_family": "backend_core_runs",
                "expected": 24,
                "observed": gate["backend_manifest"]["run_count"],
                "failures": len(gate["backend_manifest"].get("failures", [])),
                "gate_pass": gate["gates"]["backend_sensitivity"],
            },
        ]
    )
    completeness.to_csv(OUTPUT_ROOT / "completeness_manifest.csv", index=False)
    failures = {
        "no_augmentation": records(
            gate["noaug_completeness"].loc[~gate["noaug_completeness"]["exists"]]
        ),
        "threshold_sensitivity": gate["threshold_manifest"].get("failures", []),
        "backend_sensitivity": gate["backend_manifest"].get("failures", []),
    }
    (OUTPUT_ROOT / "failure_manifest.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "correction_families.json").write_text(
        json.dumps(correction_families(), indent=2) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Reviewer 3 round-2 claim freeze",
        "",
        "All three evidence gates passed. Numerical manuscript prose must use the "
        "generated `sreport/round2_claims.tex` macros or the frozen CSV tables.",
        "",
        "## Main six-method benchmark",
        "",
        f"- NoAug-CW / QualSynth mean F1: {macros['RThreeNoAugFOneMean']} / "
        f"{macros['RThreeQualSynthFOneMean']}.",
        f"- NoAug-CW / QualSynth mean F1 ranks: {macros['RThreeNoAugFOneRank']} / "
        f"{macros['RThreeQualSynthFOneRank']}.",
        f"- NoAug-CW / QualSynth mean ROC-AUC ranks: {macros['RThreeNoAugAUCRank']} / "
        f"{macros['RThreeQualSynthAUCRank']}.",
        f"- QualSynth versus NoAug-CW mean differences (F1 / ROC-AUC): "
        f"{macros['RThreeQualSynthNoAugFOneDelta']} / "
        f"{macros['RThreeQualSynthNoAugAUCDelta']}.",
        "",
        "## Threshold sensitivity",
        "",
        f"- Fixed 3/4/5-sigma acceptance: {macros['RThreeStdThreeAcceptance']} / "
        f"{macros['RThreeStdFourAcceptance']} / {macros['RThreeStdFiveAcceptance']}.",
        f"- Training-only calibrated threshold range: "
        f"{macros['RThreeCalibratedThresholdMin']}--"
        f"{macros['RThreeCalibratedThresholdMax']}.",
        f"- Numeric-equivalence survivors under 4 sigma: "
        f"{macros['RThreeStdFourDuplicateSurvivors']} among "
        f"{macros['RThreeStdFourAcceptedRows']} accepted rows.",
        "",
        "## Backend sensitivity",
        "",
        f"- Complete paired runs / selected rows: {macros['RThreeBackendRuns']} / "
        f"{macros['RThreeBackendRows']}.",
        f"- Gemma / Luna mean F1: {macros['RThreeGemmaFOne']} / " f"{macros['RThreeLunaFOne']}.",
        f"- Gemma / Luna mean ROC-AUC: {macros['RThreeGemmaAUC']} / " f"{macros['RThreeLunaAUC']}.",
        f"- Gemma / Luna total calls: {macros['RThreeGemmaCalls']} / "
        f"{macros['RThreeLunaCalls']}.",
        f"- Gemma / Luna total tokens (thousands): {macros['RThreeGemmaTokensK']} / "
        f"{macros['RThreeLunaTokensK']}.",
        f"- Gemma / Luna generation time (hours): "
        f"{macros['RThreeGemmaRuntimeHours']} / {macros['RThreeLunaRuntimeHours']}.",
        f"- Gemma / Luna observed provider cost (USD): {macros['RThreeGemmaCost']} / "
        f"{macros['RThreeLunaCost']}.",
        f"- Combined observed provider cost: USD {macros['RThreeBackendCost']}.",
        "",
        "Backend flexibility is architectural, whereas these results characterize only "
        "two tested routes. Local deployment can reduce third-party disclosure but does "
        "not itself guarantee privacy.",
    ]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_files = sorted(
        path
        for path in [*OUTPUT_ROOT.iterdir(), MACRO_PATH, *generated_tex_paths]
        if path.is_file()
        and path.name not in {"freeze_manifest.json", "manuscript_claim_audit.json"}
    )
    freeze_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "builder_sha256": sha256_file(Path(__file__)),
        "inputs": source_manifest,
        "outputs": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in output_files
        },
    }
    (OUTPUT_ROOT / "freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    gate = gate_inputs()
    tables = load_tables()
    claims = selected_claims(tables)
    write_outputs(gate, tables, claims)
    print(f"Wrote immutable claim bundle to {OUTPUT_ROOT}")
    print(f"Wrote manuscript macros to {MACRO_PATH}")


if __name__ == "__main__":
    main()
