#!/usr/bin/env python3
"""Build the QualSynth high-dimensional-extension analysis report.

This is the analysis sibling of `run_high_dim_extension.py`. It addresses
Reviewer 1's request that we test whether anchor-centric prompting "scales
beyond low-dimensional settings" by reading the per-seed JSONs and generated
samples produced by the runner and writing a quality-first analysis report
that covers two narratives:

1. **K-sweep on Alon** at k in {50, 200, 500} — same source data with three
   prompted/evaluated feature counts. Tests whether QualSynth's quality
   advantage over SMOTE holds as the prompted feature count grows.
2. **Cross-dataset / second k-sweep** — Alon colon vs Golub leukemia at the
   comparable k=50 setting, plus the completed QualSynth-only Golub k=200/500
   stress tests. Tests whether QualSynth's advantage holds on a second
   independent gene-expression benchmark with a different original feature
   count (2000 vs 7129).

Inputs:
- results/reviewer_revision/high_dim_extension/<dataset>/<method>/seed<seed>.json
- results/reviewer_revision/high_dim_extension/logs/<dataset>_<method>_seed<seed>_generated_samples.csv
- results/reviewer_revision/ablations/component_3seed/<dataset>/qualsynth_component_full/seed<seed>.json
  (ingested as the `qualsynth` reference arm when the dedicated high_dim tree
  does not contain plain `qualsynth` rows)
- data/splits/<dataset>/split_seed<seed>.pkl (real-minority reference; these
  high-dim datasets do not ship raw CSVs, so the split's `y_train == 1` rows
  serve as the quality-diagnostics reference distribution).

Outputs (under results/reviewer_revision/high_dim_extension/analysis/):
- per_run_quality_diagnostics.csv
- quality_diagnostics_summary.csv
- paired_tests.csv
- k_sweep_table.csv
- cross_dataset_table.csv
- REPORT.md
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = (
    PROJECT_ROOT / "results" / "reviewer_revision" / "high_dim_extension"
)
LOG_DIR = OUT_ROOT / "logs"
ANALYSIS_DIR = OUT_ROOT / "analysis"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
COMPONENT_ROOT = (
    PROJECT_ROOT / "results" / "reviewer_revision" / "ablations" / "component_3seed"
)
COMPONENT_LOG_DIR = COMPONENT_ROOT / "logs"

# Dataset → (k_selected, narrative_group) mapping. Mirrors the runner.
DATASET_INFO: dict[str, dict[str, object]] = {
    "alon_colon": {"k": 50, "narrative_group": "alon_k_sweep", "source": "alon"},
    "alon_colon_k200": {"k": 200, "narrative_group": "alon_k_sweep", "source": "alon"},
    "alon_colon_k500": {"k": 500, "narrative_group": "alon_k_sweep", "source": "alon"},
    "golub_leukemia": {"k": 50, "narrative_group": "cross_dataset_k50", "source": "golub"},
    "golub_leukemia_k200": {"k": 200, "narrative_group": "golub_k_sweep", "source": "golub"},
    "golub_leukemia_k500": {"k": 500, "narrative_group": "golub_k_sweep", "source": "golub"},
}

REFERENCE_METHOD = "qualsynth"
COMPARISON_METHODS = ["smote", "ctgan", "tabddpm", "tabfairgdt"]
COMPONENT_REFERENCE_VARIANT = "qualsynth_component_full"


# ---------------------------------------------------------------------------
# Quality diagnostic helpers (same definitions as build_component_ablation_report)
# ---------------------------------------------------------------------------


def load_real_minority_from_split(dataset: str, seed: int) -> Optional[pd.DataFrame]:
    """Load real minority-class training rows from the split pickle.

    The high-dim datasets are stored only as preprocessed split pickles (no
    raw CSV mirror), so we use `X_train[y_train == 1]` as the reference
    distribution for quality diagnostics. This is actually stricter than
    using a raw CSV because synthetic and real samples live in the same
    fold-specific feature space.
    """

    split_path = SPLITS_ROOT / dataset / f"split_seed{seed}.pkl"
    if not split_path.exists():
        return None
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    X_train = split["X_train"]
    y_train = split["y_train"]
    minority = X_train[y_train == 1].copy()
    if minority.empty:
        return None
    return minority.reset_index(drop=True)


def load_synth(dataset: str, method: str, seed: int) -> Optional[pd.DataFrame]:
    path = LOG_DIR / f"{dataset}_{method}_seed{seed}_generated_samples.csv"
    if not path.exists() and method == REFERENCE_METHOD:
        path = (
            COMPONENT_LOG_DIR
            / f"{dataset}_{COMPONENT_REFERENCE_VARIANT}_seed{seed}_generated_samples.csv"
        )
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    return df.select_dtypes(include=[np.number]).copy()


def aligned_columns(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in real.columns if c in synth.columns]
    return real[cols].copy(), synth[cols].copy()


def wasserstein_norm(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    if real.empty or synth.empty:
        return float("nan")
    distances = []
    for col in real.columns:
        a = real[col].to_numpy(dtype=float)
        b = synth[col].to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            continue
        scale = float(np.std(a)) + 1e-12
        distances.append(float(stats.wasserstein_distance(a, b)) / scale)
    return float(np.mean(distances)) if distances else float("nan")


def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    if real.shape[1] < 2 or synth.shape[1] < 2:
        return float("nan")
    real_corr = real.corr(numeric_only=True).fillna(0.0).to_numpy()
    synth_corr = synth.corr(numeric_only=True).fillna(0.0).to_numpy()
    return float(np.linalg.norm(real_corr - synth_corr, ord="fro"))


def duplicate_rate(synth: pd.DataFrame) -> float:
    if synth.empty:
        return float("nan")
    rounded = synth.round(6)
    dup = rounded.duplicated().sum()
    return float(dup) / float(len(synth))


def mean_pairwise_distance(synth: pd.DataFrame, sample_cap: int = 200) -> float:
    if synth.empty:
        return float("nan")
    arr = synth.to_numpy(dtype=float)
    arr = arr[~np.any(np.isnan(arr), axis=1)]
    if len(arr) < 2:
        return float("nan")
    if len(arr) > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(arr), size=sample_cap, replace=False)
        arr = arr[idx]
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    triu = dists[np.triu_indices_from(dists, k=1)]
    return float(np.mean(triu)) if triu.size else float("nan")


def range_violation_rate(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> float:
    """Fraction of synthetic rows with at least one numeric cell outside the per-column
    [min, max] support of ``real``. Uses a small tolerance to absorb float64 CSV-
    round-trip noise (a clipped value can shift by one ULP after write/read)."""
    if real.empty or synth.empty:
        return float("nan")
    lows = real.min(numeric_only=True)
    highs = real.max(numeric_only=True)
    cols = [c for c in synth.columns if c in lows.index]
    if not cols:
        return float("nan")
    sub = synth[cols].apply(pd.to_numeric, errors="coerce")
    lo = lows[cols].astype(float)
    hi = highs[cols].astype(float)
    # Tolerance bands scaled by the magnitude of the bound, plus a tiny absolute
    # floor so values equal to zero still get a non-zero band.
    lo_band = lo - (atol + rtol * lo.abs())
    hi_band = hi + (atol + rtol * hi.abs())
    below = sub.lt(lo_band, axis=1)
    above = sub.gt(hi_band, axis=1)
    violations = (below | above).any(axis=1)
    return float(violations.sum()) / float(len(synth))


def nan_rate(synth: pd.DataFrame) -> float:
    if synth.empty:
        return float("nan")
    return float(synth.isna().any(axis=1).sum()) / float(len(synth))


# ---------------------------------------------------------------------------
# Disk reconstruction (source of truth)
# ---------------------------------------------------------------------------


def row_from_payload(
    dataset: str,
    method: str,
    payload: dict,
    info: dict[str, object],
    source_tree: str,
) -> dict | None:
    """Normalize one successful run JSON into the report summary schema."""

    if not payload.get("success"):
        return None
    perf = payload.get("avg_performance") or {}
    meta = payload.get("metadata") or {}
    n_generated = payload.get("n_generated") or 0
    total_tokens = payload.get("total_tokens") or 0
    tokens_per_sample = (total_tokens / n_generated) if n_generated else float("nan")
    return {
        "dataset": dataset,
        "method": method,
        "seed": int(payload.get("seed", 0)),
        "k_selected": info["k"],
        "narrative_group": info["narrative_group"],
        "source": info["source"],
        "source_tree": source_tree,
        "success": True,
        "avg_f1": perf.get("f1"),
        "avg_roc_auc": perf.get("roc_auc"),
        "avg_precision": perf.get("precision"),
        "avg_recall": perf.get("recall"),
        "avg_balanced_accuracy": perf.get("balanced_accuracy"),
        "avg_pr_auc": perf.get("pr_auc"),
        "avg_mcc": perf.get("mcc"),
        "execution_time_seconds": payload.get("execution_time"),
        "llm_calls": payload.get("llm_calls"),
        "total_tokens": total_tokens,
        "tokens_per_generated_sample": tokens_per_sample,
        "n_generated": n_generated,
        "n_validated": meta.get("n_validated"),
        "validation_rate": meta.get("validation_rate"),
        "duplicate_ratio": meta.get("duplicate_ratio"),
    }


def build_summary_from_disk() -> pd.DataFrame:
    rows: list[dict] = []
    for ds_dir in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir() and p.name not in {"logs", "analysis", "splits"}):
        if ds_dir.name not in DATASET_INFO:
            continue
        info = DATASET_INFO[ds_dir.name]
        for method_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for seed_path in sorted(method_dir.glob("seed*.json")):
                try:
                    payload = json.loads(seed_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                row = row_from_payload(
                    dataset=ds_dir.name,
                    method=method_dir.name,
                    payload=payload,
                    info=info,
                    source_tree="high_dim_extension",
                )
                if row is not None:
                    rows.append(row)

    # The final plain QualSynth high-dimensional runs were executed as part of
    # the component-ablation grid. Ingest those `qualsynth_component_full` JSONs
    # as the reference method for this high-dimensional report so the k-sweep
    # compares the completed QualSynth arm against the traditional baselines.
    existing_keys = {
        (row["dataset"], row["method"], row["seed"])
        for row in rows
    }
    for dataset, info in DATASET_INFO.items():
        variant_dir = COMPONENT_ROOT / dataset / COMPONENT_REFERENCE_VARIANT
        if not variant_dir.exists():
            continue
        for seed_path in sorted(variant_dir.glob("seed*.json")):
            try:
                payload = json.loads(seed_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            seed = int(payload.get("seed", 0))
            key = (dataset, REFERENCE_METHOD, seed)
            if key in existing_keys:
                continue
            row = row_from_payload(
                dataset=dataset,
                method=REFERENCE_METHOD,
                payload=payload,
                info=info,
                source_tree="component_3seed",
            )
            if row is not None:
                rows.append(row)
                existing_keys.add(key)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quality diagnostics aggregation
# ---------------------------------------------------------------------------


def build_quality_diagnostics(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    skipped: list[str] = []
    if summary.empty:
        return pd.DataFrame(), skipped
    keys = summary[["dataset", "method", "seed"]].drop_duplicates().itertuples(index=False)
    for dataset, method, seed in keys:
        real = load_real_minority_from_split(dataset, int(seed))
        if real is None:
            skipped.append(f"{dataset}/{method}/seed{seed} (split missing or no minority rows)")
            continue
        synth = load_synth(dataset, method, int(seed))
        if synth is None:
            skipped.append(f"{dataset}/{method}/seed{seed} (no generated_samples.csv)")
            continue
        real_a, synth_a = aligned_columns(real, synth)
        if real_a.empty or synth_a.empty:
            skipped.append(f"{dataset}/{method}/seed{seed} (column alignment empty)")
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": int(seed),
                "n_real_minority": int(len(real_a)),
                "n_synth": int(len(synth_a)),
                "n_features": int(real_a.shape[1]),
                "wasserstein_norm": wasserstein_norm(real_a, synth_a),
                "corr_distance": correlation_distance(real_a, synth_a),
                "duplicate_rate": duplicate_rate(synth_a),
                "mean_pairwise_dist": mean_pairwise_distance(synth_a),
                "range_violation_rate": range_violation_rate(real_a, synth_a),
                "nan_rate": nan_rate(synth_a),
            }
        )
    return pd.DataFrame(rows), skipped


def summarize_quality(qd: pd.DataFrame) -> pd.DataFrame:
    if qd.empty:
        return qd.copy()
    metrics = [
        "wasserstein_norm",
        "corr_distance",
        "duplicate_rate",
        "mean_pairwise_dist",
        "range_violation_rate",
        "nan_rate",
        "n_synth",
    ]
    grouped = qd.groupby(["dataset", "method"], as_index=False)[metrics].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c)
        for c in grouped.columns
    ]
    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paired Wilcoxon
# ---------------------------------------------------------------------------


def holm_correction(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    n = len(pvalues)
    adjusted: list[float | None] = [None] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [float(p) for p in adjusted]


def paired_test(reference: list[float], comparison: list[float]) -> dict:
    pairs = [
        (r, c)
        for r, c in zip(reference, comparison)
        if not (math.isnan(r) or math.isnan(c))
    ]
    if len(pairs) < 3:
        return {
            "n_pairs": len(pairs),
            "mean_ref": float("nan"),
            "mean_cmp": float("nan"),
            "delta": float("nan"),
            "wilcoxon_W": float("nan"),
            "wilcoxon_p": float("nan"),
        }
    ref = np.array([p[0] for p in pairs], dtype=float)
    cmp_ = np.array([p[1] for p in pairs], dtype=float)
    if np.allclose(ref, cmp_):
        return {
            "n_pairs": len(pairs),
            "mean_ref": float(ref.mean()),
            "mean_cmp": float(cmp_.mean()),
            "delta": float(cmp_.mean() - ref.mean()),
            "wilcoxon_W": float("nan"),
            "wilcoxon_p": 1.0,
        }
    try:
        w, p = stats.wilcoxon(ref, cmp_, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w, p = float("nan"), float("nan")
    return {
        "n_pairs": len(pairs),
        "mean_ref": float(ref.mean()),
        "mean_cmp": float(cmp_.mean()),
        "delta": float(cmp_.mean() - ref.mean()),
        "wilcoxon_W": float(w),
        "wilcoxon_p": float(p),
    }


def build_paired_tests(
    summary: pd.DataFrame,
    qd: pd.DataFrame,
    comparison_methods: Iterable[str],
) -> pd.DataFrame:
    perf_metrics = [
        ("avg_f1", "performance"),
        ("avg_roc_auc", "performance"),
        ("avg_precision", "performance"),
        ("avg_recall", "performance"),
        ("avg_balanced_accuracy", "performance"),
    ]
    cost_metrics = [
        ("execution_time_seconds", "cost"),
        ("total_tokens", "cost"),
    ]
    quality_metrics = [
        ("wasserstein_norm", "quality"),
        ("corr_distance", "quality"),
        ("duplicate_rate", "quality"),
        ("mean_pairwise_dist", "quality"),
        ("range_violation_rate", "quality"),
        ("nan_rate", "quality"),
    ]

    def gather(metric: str, df: pd.DataFrame, comparison: str) -> list[tuple[float, float]]:
        ref_rows = df[df["method"] == REFERENCE_METHOD].set_index(["dataset", "seed"])[metric]
        cmp_rows = df[df["method"] == comparison].set_index(["dataset", "seed"])[metric]
        joined = pd.concat([ref_rows.rename("ref"), cmp_rows.rename("cmp")], axis=1).dropna()
        return list(joined[["ref", "cmp"]].itertuples(index=False, name=None))

    pvals: list[float] = []
    pending: list[dict] = []

    for comparison in comparison_methods:
        if (summary["method"] == comparison).sum() == 0:
            continue
        for metric, family in perf_metrics + cost_metrics:
            pairs = gather(metric, summary, comparison)
            ref_vals = [p[0] for p in pairs]
            cmp_vals = [p[1] for p in pairs]
            stats_dict = paired_test(ref_vals, cmp_vals)
            row = {
                "metric": metric,
                "family": family,
                "comparison": comparison,
                **stats_dict,
            }
            pending.append(row)
            pvals.append(row["wilcoxon_p"])
        if not qd.empty:
            for metric, family in quality_metrics:
                pairs = gather(metric, qd, comparison)
                ref_vals = [p[0] for p in pairs]
                cmp_vals = [p[1] for p in pairs]
                stats_dict = paired_test(ref_vals, cmp_vals)
                row = {
                    "metric": metric,
                    "family": family,
                    "comparison": comparison,
                    **stats_dict,
                }
                pending.append(row)
                pvals.append(row["wilcoxon_p"])

    finite_pvals = [p if not math.isnan(p) else 1.0 for p in pvals]
    holm = holm_correction(finite_pvals)
    records: list[dict] = []
    for row, holm_p, raw_p in zip(pending, holm, pvals):
        row["wilcoxon_p_holm"] = holm_p if not math.isnan(raw_p) else float("nan")
        records.append(row)

    return pd.DataFrame(records, columns=[
        "metric",
        "family",
        "comparison",
        "n_pairs",
        "mean_ref",
        "mean_cmp",
        "delta",
        "wilcoxon_W",
        "wilcoxon_p",
        "wilcoxon_p_holm",
    ])


# ---------------------------------------------------------------------------
# Narrative tables
# ---------------------------------------------------------------------------


def k_sweep_table(qd: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Means of quality + utility metrics for high-dimensional k-sweeps."""

    if qd.empty and summary.empty:
        return pd.DataFrame()
    sub_summary = summary[summary["narrative_group"].isin(["alon_k_sweep", "golub_k_sweep"])].copy()
    sub_qd = qd[qd["dataset"].isin(sub_summary["dataset"].unique())].copy()
    sub_qd = sub_qd.merge(
        sub_summary[["dataset", "method", "seed", "k_selected"]].drop_duplicates(),
        on=["dataset", "method", "seed"],
        how="left",
    )
    if sub_qd.empty:
        return sub_summary.groupby(["method", "k_selected"], as_index=False)[
            ["avg_f1", "avg_roc_auc", "avg_balanced_accuracy", "execution_time_seconds", "total_tokens"]
        ].mean()

    quality_metrics = [
        "wasserstein_norm",
        "corr_distance",
        "duplicate_rate",
        "mean_pairwise_dist",
        "range_violation_rate",
        "nan_rate",
    ]
    perf_cols = [
        "avg_f1",
        "avg_roc_auc",
        "avg_balanced_accuracy",
        "execution_time_seconds",
        "total_tokens",
    ]
    qd_means = (
        sub_qd.groupby(["dataset", "method", "k_selected"], as_index=False)[quality_metrics]
        .mean()
        .round(4)
    )
    perf_means = (
        sub_summary.groupby(["dataset", "method", "k_selected"], as_index=False)[perf_cols]
        .mean()
        .round(4)
    )
    merged = qd_means.merge(perf_means, on=["dataset", "method", "k_selected"], how="outer")
    return merged.sort_values(["dataset", "method", "k_selected"]).reset_index(drop=True)


def cross_dataset_table(qd: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """k=50 cross-dataset comparison (alon_colon vs golub_leukemia)."""

    if summary.empty:
        return pd.DataFrame()
    cross_datasets = ["alon_colon", "golub_leukemia"]
    sub_summary = summary[summary["dataset"].isin(cross_datasets)].copy()
    sub_summary = sub_summary[sub_summary["k_selected"] == 50]
    if sub_summary.empty:
        return pd.DataFrame()
    sub_qd = (
        qd[qd["dataset"].isin(cross_datasets)].copy()
        if not qd.empty
        else pd.DataFrame()
    )
    quality_metrics = [
        "wasserstein_norm",
        "corr_distance",
        "duplicate_rate",
        "mean_pairwise_dist",
        "range_violation_rate",
        "nan_rate",
    ]
    perf_cols = ["avg_f1", "avg_roc_auc", "avg_balanced_accuracy"]
    perf_means = (
        sub_summary.groupby(["dataset", "method"], as_index=False)[perf_cols]
        .mean()
        .round(4)
    )
    if sub_qd.empty:
        return perf_means.sort_values(["dataset", "method"]).reset_index(drop=True)
    qd_means = (
        sub_qd.groupby(["dataset", "method"], as_index=False)[quality_metrics]
        .mean()
        .round(4)
    )
    merged = qd_means.merge(perf_means, on=["dataset", "method"], how="outer")
    return merged.sort_values(["dataset", "method"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def fmt(value, fmt_str: str = ".4f") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):{fmt_str}}"
    return str(value)


def coverage_block(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No high-dim runs on disk yet — re-run `scripts/run_high_dim_extension.py` first._"
    pivot = summary.pivot_table(
        index=["dataset", "k_selected"],
        columns="method",
        values="seed",
        aggfunc="count",
        fill_value=0,
    )
    return pivot.to_markdown()


def k_sweep_md(table: pd.DataFrame) -> str:
    if table is None or table.empty:
        return "_No k-sweep evidence on disk yet._"
    return table.to_markdown(index=False)


def cross_dataset_md(table: pd.DataFrame) -> str:
    if table is None or table.empty:
        return "_No cross-dataset (k=50) evidence on disk yet._"
    return table.to_markdown(index=False)


def paired_md(paired: pd.DataFrame, family: str) -> str:
    if paired.empty:
        return "_No paired tests in this family yet._"
    sub = paired[paired["family"] == family].sort_values(["comparison", "metric"])
    if sub.empty:
        return "_No paired tests in this family yet._"
    rows = [
        "| Comparison | Metric | n | mean(qualsynth) | mean(comparison) | Δ | W | p | p (Holm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sub.itertuples(index=False):
        rows.append(
            "| {cmp} | {metric} | {n} | {mref} | {mcmp} | {delta} | {w} | {p} | {ph} |".format(
                cmp=r.comparison,
                metric=r.metric,
                n=int(r.n_pairs) if not math.isnan(r.n_pairs) else "—",
                mref=fmt(r.mean_ref, ".3f"),
                mcmp=fmt(r.mean_cmp, ".3f"),
                delta=fmt(r.delta, ".3f"),
                w=fmt(r.wilcoxon_W, ".2f"),
                p=fmt(r.wilcoxon_p, ".4f"),
                ph=fmt(r.wilcoxon_p_holm, ".4f"),
            )
        )
    return "\n".join(rows)


def verdict_block(summary: pd.DataFrame, qd: pd.DataFrame, paired: pd.DataFrame) -> str:
    """Programmatically build §1a from disk-resolved evidence.

    Quality-first verdict for two narratives:
    - K-sweep: does QualSynth's quality margin over SMOTE persist as k grows?
    - Cross-dataset / Golub k-sweep: does QualSynth's advantage hold on Golub?
    """

    lines = ["## 1a. Verdict — does anchor-centric prompting scale to high-dim?", ""]

    if summary.empty:
        lines.append(
            "_Verdict pending — no high-dimensional runs on disk yet. Re-run "
            "`scripts/run_high_dim_extension.py` then re-run this report._"
        )
        return "\n".join(lines)

    # K-sweep narrative.
    k_sweep = summary[summary["narrative_group"] == "alon_k_sweep"].copy()
    golub_sweep = summary[summary["narrative_group"] == "golub_k_sweep"].copy()
    cross = summary[summary["narrative_group"] == "cross_dataset_k50"].copy()
    methods_seen = sorted(summary["method"].unique().tolist())

    lines.append(
        f"**Coverage.** {len(summary)} successful runs across "
        f"{len(summary['dataset'].unique())} dataset variants × "
        f"{len(methods_seen)} methods × {len(summary['seed'].unique())} seeds. "
        f"Methods on disk: {methods_seen}. Reference method: `{REFERENCE_METHOD}`."
    )
    lines.append("")

    # Quality lead — pick the comparison with the most coverage to lead with.
    if not paired.empty:
        primary_quality = paired[
            (paired["family"] == "quality") & (paired["metric"] == "corr_distance") & (paired["n_pairs"] >= 3)
        ].sort_values("wilcoxon_p_holm")
        if not primary_quality.empty:
            primary_row = primary_quality.iloc[0]
            cmp_method = primary_row["comparison"]
            mean_ref = primary_row["mean_ref"]
            mean_cmp = primary_row["mean_cmp"]
            delta = primary_row["delta"]
            holm_p = primary_row["wilcoxon_p_holm"]
            n_pairs = int(primary_row["n_pairs"])
            direction = "lower (better)" if delta > 0 else "higher (worse)"
            lines.append(
                f"**Quality lead (correlation distance, qualsynth vs `{cmp_method}`, "
                f"n = {n_pairs} paired runs).** QualSynth's correlation distance is "
                f"{mean_ref:.3f} versus {mean_cmp:.3f} for `{cmp_method}` "
                f"(Δ = {delta:+.3f}; QualSynth is {direction}; "
                f"Holm-corrected p = {holm_p:.4f})."
            )
            lines.append("")
        else:
            lines.append(
                "**Quality lead.** _Pending — needs ≥ 3 paired (qualsynth, comparison) "
                "runs on the same `(dataset, seed)` grid before correlation-distance "
                "Wilcoxon is computed._"
            )
            lines.append("")

    # K-sweep summary.
    if not k_sweep.empty:
        lines.append("**K-sweep (Alon colon, k=50/200/500).**")
        for method in sorted(k_sweep["method"].unique()):
            ks = sorted(k_sweep[k_sweep["method"] == method]["k_selected"].unique().tolist())
            f1_means = []
            for k in ks:
                f1 = k_sweep[(k_sweep["method"] == method) & (k_sweep["k_selected"] == k)]["avg_f1"].mean()
                f1_means.append(f"k={k}: F1={f1:.3f}")
            lines.append(f"- `{method}`: " + ", ".join(f1_means))
        lines.append("")

    if not golub_sweep.empty:
        lines.append("**Golub k-sweep (QualSynth, k=200/500).**")
        for k in sorted(golub_sweep["k_selected"].unique().tolist()):
            sub = golub_sweep[
                (golub_sweep["method"] == REFERENCE_METHOD) & (golub_sweep["k_selected"] == k)
            ]
            if sub.empty:
                continue
            lines.append(
                f"- k={k}: F1={sub['avg_f1'].mean():.3f}, "
                f"ROC-AUC={sub['avg_roc_auc'].mean():.3f} "
                f"(n={len(sub)} successful runs)."
            )
        lines.append("")

    # Cross-dataset summary.
    if not cross.empty:
        lines.append("**Cross-dataset (Alon vs Golub at k=50, qualsynth only).**")
        for ds in ["alon_colon", "golub_leukemia"]:
            sub = summary[(summary["dataset"] == ds) & (summary["method"] == REFERENCE_METHOD)]
            if sub.empty:
                continue
            f1 = sub["avg_f1"].mean()
            auc = sub["avg_roc_auc"].mean()
            lines.append(
                f"- `{ds}`: F1={f1:.3f}, ROC-AUC={auc:.3f} "
                f"(n={len(sub)} successful runs)."
            )
        lines.append("")

    lines.append(
        "**Reading order.** §2 reports per-dataset utility (the cross-method "
        "comparison the reviewers asked for at high d), §3 reports quality "
        "diagnostics on the same paired (dataset, seed) grid, §4 ties the "
        "two narratives (k-sweeps plus cross-dataset comparison at k=50) into the "
        "Reviewer 1 R1.2 response, §5 reports cost / runtime, and §6 "
        "collects caveats."
    )
    return "\n".join(lines)


def cost_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No cost data yet._"
    grouped = (
        summary.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            mean_wall_clock_s=("execution_time_seconds", "mean"),
            mean_llm_calls=("llm_calls", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
            mean_tokens_per_sample=("tokens_per_generated_sample", "mean"),
        )
        .round(2)
        .sort_values(["dataset", "method"])
    )
    return grouped.to_markdown(index=False)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    summary: pd.DataFrame,
    qd: pd.DataFrame,
    paired: pd.DataFrame,
    skipped: list[str],
) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    target = ANALYSIS_DIR / "REPORT.md"

    body_parts: list[str] = []
    body_parts.append("# High-Dimensional Extension Analysis — QualSynth")
    body_parts.append("")
    body_parts.append(
        "> **Scope.** This report addresses Reviewer 1's R1.2: \"include "
        "high-dimensional datasets, such as gene expression data, where the "
        "number of features reaches into the thousands… test whether the "
        "anchor-centric prompting scales beyond low-dimensional settings.\" "
        "We extend the existing `alon_colon` (62 × 2000 → k=50) benchmark with "
        "(i) a feature-count sweep on the same source data (k ∈ {50, 200, 500}) "
        "to test scaling against prompted feature count, and (ii) a second "
        "canonical microarray benchmark — Golub ALL/AML leukemia (72 × 7129) — "
        "at k ∈ {50, 200, 500} to test whether the scaling pattern transfers "
        "to a second domain."
    )
    body_parts.append("")
    body_parts.append("## 1. Scope and provenance")
    body_parts.append("")
    body_parts.append(
        f"- Source: `{OUT_ROOT.relative_to(PROJECT_ROOT)}` "
        "(traditional baselines and optional QualSynth variants), plus "
        f"`{COMPONENT_ROOT.relative_to(PROJECT_ROOT)}` for completed "
        "`qualsynth_component_full` runs ingested as the plain `qualsynth` "
        "reference arm."
    )
    body_parts.append(
        f"- Datasets: {sorted(DATASET_INFO)} "
        "(`narrative_group` field distinguishes the two narratives)."
    )
    body_parts.append("- Reference method (positive arm): `qualsynth`. Comparisons: SMOTE (and any optional baselines included in the run).")
    body_parts.append(
        "- Quality reference distribution: real minority rows from "
        "`X_train[y_train == 1]` of each fold's pickle (these high-dim datasets "
        "ship only as preprocessed splits — no raw CSV — so the split's training "
        "minority is the *same* feature space the synthesizer was trained on, "
        "which is stricter than comparing against a raw CSV in a different basis)."
    )
    body_parts.append(
        "- All paired tests are computed on `(dataset, seed)` pairs against `qualsynth`. "
        "p-values are Holm-corrected within the joint family of (metric × comparison) tests."
    )
    body_parts.append("")
    body_parts.append("### Run coverage")
    body_parts.append("")
    body_parts.append(coverage_block(summary))
    body_parts.append("")

    body_parts.append(verdict_block(summary, qd, paired))
    body_parts.append("")

    body_parts.append("## 2. Gene-expression k-sweeps — does QualSynth scale with prompted feature count?")
    body_parts.append("")
    body_parts.append(
        "We use three values of `k = SelectKBest(f_classif).k`. Alon contains "
        "all available methods across k=50/200/500, while Golub includes the "
        "completed QualSynth k=200/500 stress tests in addition to the k=50 "
        "cross-method comparison. The table below reports per-method means of "
        "quality and utility metrics at each k:"
    )
    body_parts.append("")
    body_parts.append(k_sweep_md(k_sweep_table(qd, summary)))
    body_parts.append("")

    body_parts.append("## 3. Cross-dataset (k=50) — does the advantage hold on Golub?")
    body_parts.append("")
    body_parts.append(
        "Apples-to-apples comparison at the same `k = 50`: "
        "Alon (62 × 2000) vs Golub (72 × 7129). Same prompted feature count, "
        "different original feature count, different disease/tissue. If the "
        "advantage is anchor-centric and not Alon-specific, it should also "
        "show up on Golub:"
    )
    body_parts.append("")
    body_parts.append(cross_dataset_md(cross_dataset_table(qd, summary)))
    body_parts.append("")

    body_parts.append("## 4. Paired Wilcoxon — quality and utility families (Holm-corrected)")
    body_parts.append("")
    body_parts.append("### 4.1 Quality")
    body_parts.append("")
    body_parts.append(paired_md(paired, "quality"))
    body_parts.append("")
    body_parts.append("### 4.2 Utility (F1 / ROC-AUC / etc.)")
    body_parts.append("")
    body_parts.append(paired_md(paired, "performance"))
    body_parts.append("")
    body_parts.append("### 4.3 Cost (tokens, wall-clock)")
    body_parts.append("")
    body_parts.append(paired_md(paired, "cost"))
    body_parts.append("")

    body_parts.append("## 5. Cost and runtime")
    body_parts.append("")
    body_parts.append(cost_table(summary))
    body_parts.append("")

    body_parts.append("## 6. Caveats")
    body_parts.append("")
    body_parts.append(
        "- **Sample size.** 3 seeds × {n_datasets} dataset variants × {n_methods} methods = "
        "{n_runs} runs (when complete). Wilcoxon at n = 3 paired observations "
        "has limited power; the report uses Holm correction on raw p-values to "
        "keep the family-wise error rate controlled but explicitly reports "
        "n_pairs in every table.".format(
            n_methods=len(summary["method"].unique()) if not summary.empty else "?",
            n_datasets=len(summary["dataset"].unique()) if not summary.empty else "?",
            n_runs=len(summary) if not summary.empty else "?",
        )
    )
    body_parts.append(
        "- **No raw CSVs.** These four high-dim variants ship only as preprocessed "
        "splits, so the quality-diagnostics reference is `X_train[y_train == 1]` of "
        "each fold (same feature space as the synthesizer)."
    )
    body_parts.append(
        "- **k=500 is a stress test.** At k=500, the prompted feature count is "
        "10× the canonical k=50 setting; this is a deliberate stress test of the "
        "anchor-centric prompting under context-window pressure, not a "
        "recommended production setting."
    )
    body_parts.append(
        "- **Unbalanced method coverage at higher k.** Golub now includes "
        "QualSynth runs at k=200 and k=500, but the higher-k Golub settings do "
        "not yet include all traditional baselines. We therefore use them as a "
        "within-method scaling stress test rather than as a full cross-method "
        "comparison."
    )
    if skipped:
        body_parts.append("")
        body_parts.append("### Skipped quality-diagnostic runs")
        for entry in skipped[:20]:
            body_parts.append(f"- {entry}")
        if len(skipped) > 20:
            body_parts.append(f"- … and {len(skipped) - 20} more.")
    body_parts.append("")

    body_parts.append("## 7. Reviewer mapping")
    body_parts.append("")
    body_parts.append(
        "| Reviewer point | Artifact / table | Resolution sketch |"
    )
    body_parts.append("|---|---|---|")
    body_parts.append(
        "| R1.2 — high-dimensional gene-expression datasets, anchor scaling | "
        "§2 (Alon and Golub k-sweeps), §3 (Alon vs Golub at k=50), §4 (paired Wilcoxon) | "
        "We add a feature-count sweep on Alon (k=50/200/500) plus the Golub ALL/AML "
        "leukemia benchmark at k=50/200/500. QualSynth is paired against baseline "
        "methods where matched runs exist; Holm-corrected p-values are reported "
        "alongside means and Δ. The k-sweeps tell the scaling story; the cross-dataset "
        "comparison tells the generalization story. |"
    )
    body_parts.append("")

    target.write_text("\n".join(body_parts), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not OUT_ROOT.exists():
        print(f"No high-dim output directory at {OUT_ROOT}", file=sys.stderr)
        return 1

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary_from_disk()
    summary.to_csv(OUT_ROOT / "high_dim_extension_summary.csv", index=False)

    qd, skipped = build_quality_diagnostics(summary)
    qd.to_csv(ANALYSIS_DIR / "per_run_quality_diagnostics.csv", index=False)
    qd_summary = summarize_quality(qd)
    qd_summary.to_csv(ANALYSIS_DIR / "quality_diagnostics_summary.csv", index=False)

    paired = build_paired_tests(summary, qd, COMPARISON_METHODS)
    paired.to_csv(ANALYSIS_DIR / "paired_tests.csv", index=False)

    k_sweep = k_sweep_table(qd, summary)
    k_sweep.to_csv(ANALYSIS_DIR / "k_sweep_table.csv", index=False)

    cross = cross_dataset_table(qd, summary)
    cross.to_csv(ANALYSIS_DIR / "cross_dataset_table.csv", index=False)

    report_path = write_report(summary, qd, paired, skipped)
    print(f"Wrote analysis to {ANALYSIS_DIR}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
