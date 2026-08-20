#!/usr/bin/env python3
"""Analyze the paired Reviewer 3 backend pilot or core matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.data.splitting import load_split  # noqa: E402


METHODS = ["qualsynth_r3_gemma", "qualsynth_r3_luna"]
DISPLAY = {
    "qualsynth_r3_gemma": "Gemma 3 27B",
    "qualsynth_r3_luna": "GPT-5.6 Luna Pro",
}
# Usage records can include a token-price estimate when a route incurred no
# billable provider charge. Manuscript cost figures use recorded charges; the
# estimate is retained separately for auditability.
RECORDED_PROVIDER_CHARGE_OVERRIDES = {
    "qualsynth_r3_gemma": 0.0,
}
PILOT_DATASETS = ["haberman", "breast_cancer"]
CORE_DATASETS = ["haberman", "breast_cancer", "pima_diabetes", "yeast"]
PILOT_SEEDS = [42]
CORE_SEEDS = [42, 123, 456]
BASE_ROOT = PROJECT_ROOT / "results/reviewer_revision/reviewer3_round2/backend_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "core"], required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_csv(path: Path) -> int:
    return len(pd.read_csv(path)) if path.exists() else 0


def normalize_labels(y_train: pd.Series) -> pd.Series:
    labels = pd.Series(y_train).reset_index(drop=True)
    counts = labels.value_counts()
    if len(counts) != 2:
        raise ValueError("Binary labels required")
    return labels.map({counts.idxmin(): 1, counts.idxmax(): 0}).astype(int)


def align_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing accepted-candidate columns: {missing}")
    aligned = frame.loc[:, columns].copy()
    for column in columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    return aligned


def exact_audit(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, float | int]:
    if generated.empty:
        return {
            "numeric_within_duplicates": 0,
            "numeric_train_matches": 0,
            "numeric_within_duplicate_rate": 0.0,
            "numeric_train_match_rate": 0.0,
        }
    train_keys = {tuple(float(value) for value in row) for row in reference.to_numpy()}
    seen: set[tuple[float, ...]] = set()
    train_matches = 0
    duplicates = 0
    for row in generated.to_numpy(dtype=float):
        key = tuple(float(value) for value in row)
        if key in train_keys:
            train_matches += 1
        elif key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return {
        "numeric_within_duplicates": duplicates,
        "numeric_train_matches": train_matches,
        "numeric_within_duplicate_rate": duplicates / len(generated),
        "numeric_train_match_rate": train_matches / len(generated),
    }


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
            "pairwise_diversity": None,
        }
    means = reference.mean(axis=0).to_numpy(dtype=float)
    stds = reference.std(axis=0, ddof=1).to_numpy(dtype=float)
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    ref = (reference.to_numpy(dtype=float) - means) / stds
    gen = (generated.to_numpy(dtype=float) - means) / stds
    wasserstein = float(
        np.mean(
            [wasserstein_distance(ref[:, index], gen[:, index]) for index in range(ref.shape[1])]
        )
    )
    ref_frame = pd.DataFrame(ref)
    gen_frame = pd.DataFrame(gen)
    active = [
        column
        for column in ref_frame.columns
        if ref_frame[column].nunique() > 1 and gen_frame[column].nunique() > 1
    ]
    if len(active) >= 2:
        ref_corr = ref_frame[active].corr().to_numpy()
        gen_corr = gen_frame[active].corr().to_numpy()
        upper = np.triu_indices(len(active), k=1)
        differences = np.abs(ref_corr[upper] - gen_corr[upper])
        differences = differences[np.isfinite(differences)]
        correlation = float(differences.mean()) if len(differences) else None
    else:
        correlation = None
    nearest = NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(gen)[0][:, 0]
    sample_size = min(500, len(gen))
    if sample_size >= 2:
        positions = np.random.default_rng(seed).choice(len(gen), size=sample_size, replace=False)
        pairwise = float(pdist(gen[positions]).mean())
    else:
        pairwise = 0.0
    return {
        "standardized_wasserstein": wasserstein,
        "correlation_mae": correlation,
        "mean_nearest_minority_distance": float(nearest.mean()),
        "pairwise_diversity": pairwise,
    }


def load_request_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def result_row(root: Path, dataset: str, seed: int, method: str) -> dict[str, Any]:
    result_path = root / dataset / method / f"seed{seed}.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    payload = read_json(result_path)
    metadata = payload.get("metadata") or {}
    provenance = metadata.get("run_provenance") or {}
    usage = provenance.get("usage") or {}
    stopping = provenance.get("stopping") or {}
    logs = root / "logs"
    base = f"{dataset}_{method}_seed{seed}"
    paths = {
        stage: logs / f"{base}_{stage}.csv"
        for stage in [
            "parsed_candidates",
            "validation_input_candidates",
            "accepted_candidates",
            "rejected_candidates",
            "generated_samples",
        ]
    }
    request_path = logs / f"{base}_llm_requests.jsonl"
    requests = load_request_records(request_path)
    split = load_split(dataset, seed=seed)
    X_train = split["X_train"].copy().reset_index(drop=True)
    y_train = normalize_labels(split["y_train"])
    minority = X_train.loc[y_train == 1].copy().reset_index(drop=True)
    accepted_raw = pd.read_csv(paths["accepted_candidates"])
    accepted = align_numeric(accepted_raw, list(X_train.columns))
    selected_raw = pd.read_csv(paths["generated_samples"])
    selected = align_numeric(selected_raw, list(X_train.columns))
    requested_rows = int(sum(record.get("requested_row_count", 0) or 0 for record in requests))
    parsed_rows_by_request = int(sum(record.get("parsed_row_count", 0) or 0 for record in requests))
    parse_successes = int(sum(bool(record.get("parse_success")) for record in requests))
    validation_input_rows = count_csv(paths["validation_input_candidates"])
    accepted_rows = len(accepted)
    rejected_rows = count_csv(paths["rejected_candidates"])
    prompts = sorted(
        {record.get("prompt_sha256") for record in requests if record.get("prompt_sha256")}
    )
    sampling_hashes = sorted(
        {json.dumps(record.get("sampling") or {}, sort_keys=True) for record in requests}
    )
    # Quality metrics describe the rows actually selected for augmentation. The
    # full accepted stage remains the denominator for validator acceptance.
    exact = exact_audit(selected, minority)
    distribution = distribution_metrics(selected, minority, seed)
    f1 = (payload.get("avg_performance") or {}).get("f1")
    roc_auc = (payload.get("avg_performance") or {}).get("roc_auc")
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "model_display": DISPLAY[method],
        "success": bool(payload.get("success")),
        "error": payload.get("error"),
        "requested_model": provenance.get("requested_model_slug"),
        "resolved_model": provenance.get("resolved_model_slug"),
        "provider": provenance.get("resolved_provider"),
        "request_calls": len(requests),
        "reported_calls": int(usage.get("calls", payload.get("llm_calls", 0)) or 0),
        "parse_successful_calls": parse_successes,
        "parse_call_success_rate": parse_successes / len(requests) if requests else 0.0,
        "requested_rows": requested_rows,
        "parsed_rows_by_request": parsed_rows_by_request,
        "row_yield": parsed_rows_by_request / requested_rows if requested_rows else 0.0,
        "parsed_artifact_rows": count_csv(paths["parsed_candidates"]),
        "validation_input_rows": validation_input_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "validation_acceptance_rate": (
            accepted_rows / validation_input_rows if validation_input_rows else 0.0
        ),
        "selected_rows": len(selected),
        "target_samples": int(stopping.get("target_samples", 0) or 0),
        "target_reached": bool(stopping.get("target_reached")),
        "target_shortfall": int(stopping.get("target_shortfall", 0) or 0),
        "iterations": int(stopping.get("iterations", 0) or 0),
        "finish_reasons": json.dumps([record.get("finish_reason") for record in requests]),
        "prompt_hashes": json.dumps(prompts),
        "sampling_hashes": json.dumps(sampling_hashes),
        "sampling_sequence": json.dumps(
            [
                record.get("sampling") or {}
                for record in requests
                if int(record.get("attempt", 1) or 1) == 1
            ],
            sort_keys=True,
        ),
        "common_contract_sha256": (provenance.get("run_context") or {}).get(
            "common_contract_sha256"
        ),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int((provenance.get("reasoning") or {}).get("tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "estimated_cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
        "cost_usd": RECORDED_PROVIDER_CHARGE_OVERRIDES.get(
            method, float(usage.get("cost_usd", 0.0) or 0.0)
        ),
        "generation_time_seconds": float(usage.get("generation_time_seconds", 0.0) or 0.0),
        "f1": f1,
        "roc_auc": roc_auc,
        **exact,
        **distribution,
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": sha256_file(result_path),
        "request_path": str(request_path.relative_to(PROJECT_ROOT)),
        "request_sha256": sha256_file(request_path),
        "stage_artifacts_complete": all(path.exists() for path in paths.values()),
        "candidate_stage_semantics_present": bool(metadata.get("candidate_stage_semantics")),
    }


def core_targets() -> pd.DataFrame:
    rows = []
    for dataset in CORE_DATASETS:
        for seed in CORE_SEEDS:
            labels = pd.Series(load_split(dataset, seed=seed)["y_train"])
            counts = labels.value_counts()
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "target_samples": int(counts.max() - counts.min()),
                }
            )
    return pd.DataFrame(rows)


def pilot_cost_projection(per_run: pd.DataFrame) -> pd.DataFrame:
    targets = core_targets()
    scenarios = [
        ("perfect_yield", 1.0),
        ("pilot_observed_yield", None),
        ("conservative_50pct_yield", 0.5),
    ]
    rows = []
    for method in METHODS:
        observed = per_run.loc[per_run["method"] == method]
        pilot_yield = float(observed["selected_rows"].sum() / observed["requested_rows"].sum())
        cost_per_requested = float(observed["cost_usd"].sum() / observed["requested_rows"].sum())
        per_run_costs = observed["cost_usd"] / observed["requested_rows"].replace(0, np.nan)
        mean_seconds_per_call = float(observed["generation_time_seconds"].mean())
        min_seconds_per_call = float(observed["generation_time_seconds"].min())
        max_seconds_per_call = float(observed["generation_time_seconds"].max())
        for scenario, fixed_yield in scenarios:
            resolved_yield = pilot_yield if fixed_yield is None else fixed_yield
            calls = int(
                sum(
                    math.ceil(target / (20.0 * resolved_yield))
                    for target in targets["target_samples"]
                )
            )
            requested_rows = calls * 20
            rows.append(
                {
                    "method": method,
                    "model_display": DISPLAY[method],
                    "scenario": scenario,
                    "assumed_selected_yield": resolved_yield,
                    "core_target_rows": int(targets["target_samples"].sum()),
                    "projected_calls": calls,
                    "projected_requested_rows": requested_rows,
                    "projected_cost_usd": requested_rows * cost_per_requested,
                    "projected_cost_low_usd": requested_rows * float(per_run_costs.min()),
                    "projected_cost_high_usd": requested_rows * float(per_run_costs.max()),
                    "projected_runtime_hours": calls * mean_seconds_per_call / 3600.0,
                    "projected_runtime_low_hours": calls * min_seconds_per_call / 3600.0,
                    "projected_runtime_high_hours": calls * max_seconds_per_call / 3600.0,
                    "projection_basis": "two one-call batch5 pilots; core uses batch20",
                }
            )
    return pd.DataFrame(rows)


def model_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "parse_call_success_rate",
        "row_yield",
        "validation_acceptance_rate",
        "target_reached",
        "numeric_within_duplicate_rate",
        "numeric_train_match_rate",
        "standardized_wasserstein",
        "correlation_mae",
        "mean_nearest_minority_distance",
        "pairwise_diversity",
        "request_calls",
        "total_tokens",
        "reasoning_tokens",
        "generation_time_seconds",
        "cost_usd",
        "f1",
        "roc_auc",
    ]
    return per_run.groupby(["method", "model_display"], as_index=False)[metrics].mean()


def hierarchical_mean_ci(
    paired: pd.DataFrame,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a paired mean by resampling datasets, then seeds within dataset."""
    if paired.empty or resamples <= 0:
        return (math.nan, math.nan)
    datasets = paired.index.get_level_values("dataset").unique().to_numpy()
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        sampled_values: list[float] = []
        for dataset in sampled_datasets:
            values = paired.xs(dataset, level="dataset").to_numpy(dtype=float)
            sampled_values.extend(rng.choice(values, size=len(values), replace=True).tolist())
        draws[index] = float(np.mean(sampled_values))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_differences(
    per_run: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int = 20260820,
) -> pd.DataFrame:
    metrics = [
        "row_yield",
        "validation_acceptance_rate",
        "selected_rows",
        "standardized_wasserstein",
        "correlation_mae",
        "mean_nearest_minority_distance",
        "pairwise_diversity",
        "request_calls",
        "total_tokens",
        "generation_time_seconds",
        "cost_usd",
        "f1",
        "roc_auc",
    ]
    rows = []
    rng = np.random.default_rng(bootstrap_seed)
    for metric in metrics:
        pivot = per_run.pivot(index=["dataset", "seed"], columns="method", values=metric).dropna()
        differences = pivot["qualsynth_r3_luna"] - pivot["qualsynth_r3_gemma"]
        ci_low, ci_high = hierarchical_mean_ci(
            differences,
            resamples=bootstrap_resamples,
            rng=rng,
        )
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(pivot),
                "n_datasets": pivot.index.get_level_values("dataset").nunique(),
                "gemma_mean": float(pivot["qualsynth_r3_gemma"].mean()),
                "luna_mean": float(pivot["qualsynth_r3_luna"].mean()),
                "luna_minus_gemma_mean": float(differences.mean()),
                "hierarchical_bootstrap_ci_low": ci_low,
                "hierarchical_bootstrap_ci_high": ci_high,
                "luna_minus_gemma_min": float(differences.min()),
                "luna_minus_gemma_max": float(differences.max()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def paired_overlap_concordance(per_run: pd.DataFrame, column: str) -> bool:
    """Compare ordered request records over each model pair's shared call prefix."""
    for _, group in per_run.groupby(["dataset", "seed"]):
        if set(group["method"]) != set(METHODS):
            return False
        sequences = {
            row.method: json.loads(getattr(row, column)) for row in group.itertuples(index=False)
        }
        overlap = min(len(sequences[method]) for method in METHODS)
        if overlap == 0:
            return False
        if any(
            sequences[METHODS[0]][index] != sequences[METHODS[1]][index] for index in range(overlap)
        ):
            return False
    return True


def build_report(
    stage: str,
    per_run: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    projection: pd.DataFrame | None,
    bootstrap_resamples: int,
    transport_incident_present: bool,
    completion_continuation_present: bool,
) -> str:
    expected = 4 if stage == "pilot" else 24
    target_complete = (
        per_run["target_reached"]
        & (per_run["target_shortfall"] == 0)
        & (per_run["selected_rows"] == per_run["target_samples"])
    )
    complete = bool(
        len(per_run) == expected
        and per_run["success"].all()
        and per_run["stage_artifacts_complete"].all()
        and per_run["candidate_stage_semantics_present"].all()
        and (per_run["reported_calls"] == per_run["request_calls"]).all()
        and (stage == "pilot" or target_complete.all())
    )
    contract_concordance = all(
        group["common_contract_sha256"].notna().all()
        and group["common_contract_sha256"].nunique() == 1
        for _, group in per_run.groupby(["dataset", "seed"])
    )
    sampling_concordance = paired_overlap_concordance(per_run, "sampling_sequence")
    lines = [
        f"# Reviewer 3 backend sensitivity — {stage}",
        "",
        "The two current OpenRouter routes use one provider-neutral CSV request contract "
        "and identical non-model settings. The historical `:free` Gemma route returned "
        "HTTP 404 and is provenance only; current paired runs use `google/gemma-3-27b-it`.",
        "",
        "## Completeness and contract checks",
        "",
        f"- Expected/completed runs: {expected}/{len(per_run)}.",
        f"- Successful runs: {int(per_run['success'].sum())}/{len(per_run)}.",
        f"- Stage artifacts and explicit semantics complete: "
        f"{int((per_run['stage_artifacts_complete'] & per_run['candidate_stage_semantics_present']).sum())}/{len(per_run)}.",
        f"- Request-count reconciliation: "
        f"{int((per_run['reported_calls'] == per_run['request_calls']).sum())}/{len(per_run)}.",
        (
            "- Exact target completion: not required for the one-call pilot "
            f"({int(target_complete.sum())}/{len(per_run)} reached the full target)."
            if stage == "pilot"
            else f"- Exact target completion: {int(target_complete.sum())}/{len(per_run)}."
        ),
        f"- Paired common request-contract hashes identical within dataset/seed: {contract_concordance}.",
        f"- Actual sampling records identical over every paired shared-call prefix: {sampling_concordance}.",
        f"- Gate status: {'PASS' if complete and contract_concordance and sampling_concordance else 'FAIL'}.",
        *(
            [
                "- Transport incident record: one incomplete Gemma/Pima seed-456 "
                "response-socket hang was interrupted and rerun once under unchanged "
                "scientific and transport settings; the interrupted console log and "
                "incident note are retained.",
            ]
            if stage == "core" and transport_incident_present
            else []
        ),
        *(
            [
                "- Completion continuation record: one Luna/Yeast seed-456 run "
                "first stalled five rows short and was resumed from its retained "
                "checkpoint. Reset/restart attempts are preserved and excluded; "
                "the included continuation passed the shared-prefix sampling gate."
            ]
            if stage == "core" and completion_continuation_present
            else []
        ),
        "",
        "## Per-run evidence",
        "",
        markdown_table(
            per_run[
                [
                    "dataset",
                    "model_display",
                    "provider",
                    "request_calls",
                    "requested_rows",
                    "parsed_rows_by_request",
                    "row_yield",
                    "accepted_rows",
                    "validation_acceptance_rate",
                    "target_reached",
                    "target_shortfall",
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "estimated_cost_usd",
                    "cost_usd",
                    "generation_time_seconds",
                    "f1",
                    "roc_auc",
                ]
            ]
        ),
        "",
        "## Model means",
        "",
        markdown_table(summary),
        "",
        "## Paired Luna-minus-Gemma differences",
        "",
        f"Intervals are 95% hierarchical bootstrap intervals ({bootstrap_resamples:,} "
        "resamples), with datasets resampled first and seeds resampled within each "
        "selected dataset. They describe paired sensitivity and are not multiplicity-"
        "adjusted hypothesis tests.",
        "",
        markdown_table(paired),
        "",
        "## Interpretation boundary",
        "",
        "Backend flexibility is architectural: users can choose compatible hosted or local "
        "models. These paired results test empirical sensitivity only for two current routes. "
        "They cannot establish universal backend independence. Local deployment can reduce "
        "third-party disclosure, but privacy still depends on logging, retention, access "
        "control, runtime, and infrastructure configuration. Actual prompt hashes can diverge "
        "after the two model trajectories require different remaining-row quotas; the frozen "
        "common-contract hash and sampling records therefore define protocol concordance.",
    ]
    if stage == "pilot" and projection is not None:
        observed = projection.loc[projection["scenario"] == "pilot_observed_yield"]
        lines.extend(
            [
                "",
                "## Core-matrix cost and runtime projection",
                "",
                "The projection covers 3,420 target rows across 24 runs. It extrapolates "
                "from only four one-call, batch-5 pilots to batch-20 core calls; it is a "
                "gate estimate, not a budget guarantee.",
                "",
                markdown_table(projection),
                "",
                f"Under pilot-observed row yield, projected combined provider cost is "
                f"${observed['projected_cost_usd'].sum():.2f} and sequential generation "
                f"time is {observed['projected_runtime_hours'].sum():.2f} hours. The "
                "conservative 50% scenario is reported above before enabling the paid core flag.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root or (BASE_ROOT / args.stage)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    datasets = PILOT_DATASETS if args.stage == "pilot" else CORE_DATASETS
    seeds = PILOT_SEEDS if args.stage == "pilot" else CORE_SEEDS
    rows = [
        result_row(root, dataset, seed, method)
        for dataset in datasets
        for seed in seeds
        for method in METHODS
    ]
    per_run = pd.DataFrame(rows)
    summary = model_summary(per_run)
    paired = paired_differences(
        per_run,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    projection = pilot_cost_projection(per_run) if args.stage == "pilot" else None
    per_run.to_csv(root / "per_run_backend_metrics.csv", index=False)
    summary.to_csv(root / "model_summary.csv", index=False)
    paired.to_csv(root / "paired_differences.csv", index=False)
    if projection is not None:
        projection.to_csv(root / "core_cost_runtime_projection.csv", index=False)
    report = build_report(
        args.stage,
        per_run,
        summary,
        paired,
        projection,
        args.bootstrap_resamples,
        (root / "TRANSPORT_INCIDENTS.md").exists(),
        (root / "COMPLETION_CONTINUATIONS.md").exists(),
    )
    (root / "REPORT.md").write_text(report, encoding="utf-8")

    artifacts = sorted(
        path for path in root.iterdir() if path.is_file() and path.name != "analysis_manifest.json"
    )
    continuation_root = root / "completion_continuation_seed456"
    retained_excluded = (
        sorted(path for path in continuation_root.rglob("*") if path.is_file())
        if continuation_root.is_dir()
        else []
    )
    transport_log = (
        root
        / "logs"
        / "pima_diabetes_qualsynth_r3_gemma_seed456_interrupted_transport_hang_console.log"
    )
    if transport_log.is_file():
        retained_excluded.append(transport_log)
    manifest = {
        "name": f"reviewer3_backend_sensitivity_{args.stage}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(Path(__file__)),
        "stage": args.stage,
        "datasets": datasets,
        "seeds": seeds,
        "methods": METHODS,
        "run_count": len(per_run),
        "successful_runs": int(per_run["success"].sum()),
        "hierarchical_bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": 20260820,
            "levels": ["dataset", "seed_within_dataset"],
        },
        "failures": per_run.loc[
            ~per_run["success"], ["dataset", "seed", "method", "error"]
        ].to_dict("records"),
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts
        },
        "retained_excluded_artifacts": {
            str(path.relative_to(root)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(retained_excluded))
        },
    }
    (root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote backend {args.stage} analysis to {root}")


if __name__ == "__main__":
    main()
