"""
Aggregate reviewer-revision cost and runtime metrics from experiment artifacts.

The script reads per-seed result artifacts written by ExperimentRunner and,
when available, archived progress logs from the original benchmark. It writes:
  - per-run CSV
  - dataset/method summary CSV
  - compact Markdown table for reviewer-response planning
  - manuscript-ready per-dataset timing/token tables
  - JSON metadata with totals and pricing assumptions
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOTS = (
    PROJECT_ROOT / "results" / "reviewer_revision" / "experiments",
    PROJECT_ROOT / "results" / "verification",
    PROJECT_ROOT / "results" / "openrouter",
)
DEFAULT_PROGRESS_ROOTS = (PROJECT_ROOT / "results" / "logs1",)
DEFAULT_COMPONENT_TOKEN_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "reviewer_revision"
    / "ablations"
    / "component_3seed"
    / "component_ablation_cost_runtime_summary.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "reviewer_revision"
MANUSCRIPT_DATASETS = (
    "breast_cancer",
    "german_credit",
    "haberman",
    "htru2",
    "pima_diabetes",
    "thyroid",
    "wine_quality",
    "yeast",
)
MANUSCRIPT_METHODS = ("qualsynth", "smote", "ctgan", "tabddpm")
DATASET_DISPLAY_NAMES = {
    "breast_cancer": "Breast Cancer",
    "german_credit": "German Credit",
    "haberman": "Haberman",
    "htru2": "HTRU2",
    "pima_diabetes": "Pima Diabetes",
    "thyroid": "Thyroid",
    "wine_quality": "Wine Quality",
    "yeast": "Yeast",
}


def _load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata = payload.get("metadata") or {}
    generation_cost = float(payload.get("generation_cost") or metadata.get("generation_cost") or 0.0)

    return {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "success": bool(payload.get("success", False)),
        "execution_time_seconds": _as_float(payload.get("execution_time")),
        "generation_time_seconds": _as_float(payload.get("generation_time")),
        "training_time_seconds": _as_float(payload.get("training_time", metadata.get("training_time"))),
        "evaluation_time_seconds": _as_float(payload.get("evaluation_time", metadata.get("evaluation_time"))),
        "generation_cost_usd": generation_cost,
        "llm_calls": _as_int(payload.get("llm_calls", metadata.get("llm_calls"))),
        "prompt_tokens": _as_int(payload.get("prompt_tokens", metadata.get("prompt_tokens"))),
        "completion_tokens": _as_int(payload.get("completion_tokens", metadata.get("completion_tokens"))),
        "total_tokens": _as_int(payload.get("total_tokens", metadata.get("total_tokens"))),
        "n_generated": _as_int(payload.get("n_generated", metadata.get("n_validated"))),
        "n_generated_raw": _as_int(metadata.get("n_generated_raw")),
        "n_validated": _as_int(metadata.get("n_validated", payload.get("n_generated"))),
        "target_samples": _as_int(metadata.get("target_samples")),
        "model_name": metadata.get("model_name"),
    }


def _load_progress_result(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    experiment_id = str(payload.get("experiment_id") or path.stem)
    dataset = payload.get("dataset")
    method = payload.get("method")
    if not dataset or not method or "ablation" in experiment_id:
        return None

    apply_step = next(
        (step for step in payload.get("steps", []) if step.get("name") == "apply_method"),
        {},
    )
    apply_result = apply_step.get("result") or {}
    result = payload.get("result") or {}

    return {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "dataset": dataset,
        "method": method,
        "seed": payload.get("seed"),
        "success": payload.get("status") == "completed",
        "execution_time_seconds": _as_float(result.get("execution_time")),
        "generation_time_seconds": _as_float(apply_result.get("generation_time")),
        "training_time_seconds": 0.0,
        "evaluation_time_seconds": 0.0,
        "generation_cost_usd": _as_float(apply_result.get("cost")),
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_generated": _as_int(apply_result.get("n_generated")),
        "n_generated_raw": 0,
        "n_validated": _as_int(apply_result.get("n_generated")),
        "target_samples": 0,
        "model_name": None,
    }


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def _as_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _discover_result_files(result_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in result_roots:
        if root.exists():
            files.extend(root.glob("**/seed*.json"))
    return sorted(set(files))


def _discover_progress_files(progress_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in progress_roots:
        if root.exists():
            files.extend(root.glob("*_progress.json"))
    return sorted(set(files))


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    successful = runs[runs["success"]].copy()
    if successful.empty:
        return pd.DataFrame()

    grouped = successful.groupby(["dataset", "method"], dropna=False)
    summary = grouped.agg(
        n_runs=("source_file", "count"),
        mean_execution_time_seconds=("execution_time_seconds", "mean"),
        std_execution_time_seconds=("execution_time_seconds", "std"),
        mean_generation_time_seconds=("generation_time_seconds", "mean"),
        mean_training_time_seconds=("training_time_seconds", "mean"),
        mean_evaluation_time_seconds=("evaluation_time_seconds", "mean"),
        mean_generated_samples=("n_generated", "mean"),
        total_generated_samples=("n_generated", "sum"),
        mean_llm_calls=("llm_calls", "mean"),
        total_llm_calls=("llm_calls", "sum"),
        mean_total_tokens=("total_tokens", "mean"),
        total_tokens=("total_tokens", "sum"),
        mean_prompt_tokens=("prompt_tokens", "mean"),
        mean_completion_tokens=("completion_tokens", "mean"),
        total_generation_cost_usd=("generation_cost_usd", "sum"),
        mean_generation_cost_usd=("generation_cost_usd", "mean"),
    ).reset_index()

    summary["seconds_per_generated_sample"] = summary.apply(
        lambda row: (
            row["mean_generation_time_seconds"] / row["mean_generated_samples"]
            if row["mean_generated_samples"] > 0
            else 0.0
        ),
        axis=1,
    )
    return summary.sort_values(["dataset", "method"]).reset_index(drop=True)


def _load_qualsynth_token_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(path)
    if "method" not in summary.columns:
        return pd.DataFrame()
    full = summary[summary["method"] == "qualsynth_component_full"].copy()
    return full[
        [
            "dataset",
            "mean_llm_calls",
            "mean_total_tokens",
            "mean_generation_cost_usd",
        ]
    ]


def _build_manuscript_table(
    timing_summary: pd.DataFrame,
    token_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in MANUSCRIPT_DATASETS:
        row: dict[str, Any] = {"dataset": dataset}
        for method in MANUSCRIPT_METHODS:
            match = timing_summary[
                (timing_summary["dataset"] == dataset)
                & (timing_summary["method"] == method)
            ]
            key = f"{method}_mean_generation_time_seconds"
            row[key] = (
                float(match["mean_generation_time_seconds"].iloc[0])
                if not match.empty
                else 0.0
            )

        token_match = token_summary[token_summary["dataset"] == dataset]
        if token_match.empty:
            row["qualsynth_mean_llm_calls"] = 0.0
            row["qualsynth_mean_total_tokens"] = 0.0
            row["qualsynth_mean_generation_cost_usd"] = 0.0
        else:
            row["qualsynth_mean_llm_calls"] = float(token_match["mean_llm_calls"].iloc[0])
            row["qualsynth_mean_total_tokens"] = float(token_match["mean_total_tokens"].iloc[0])
            row["qualsynth_mean_generation_cost_usd"] = float(
                token_match["mean_generation_cost_usd"].iloc[0]
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    table["qualsynth_mean_total_tokens_k"] = table["qualsynth_mean_total_tokens"] / 1000.0
    return table


def _write_manuscript_outputs(table: pd.DataFrame, output_dir: Path) -> None:
    csv_path = output_dir / "cost_runtime_manuscript_table.csv"
    markdown_path = output_dir / "cost_runtime_manuscript_table.md"
    tex_path = output_dir / "cost_runtime_manuscript_table_rows.tex"

    table.to_csv(csv_path, index=False)

    display = pd.DataFrame(
        {
            "Dataset": table["dataset"].map(DATASET_DISPLAY_NAMES).fillna(table["dataset"]),
            "QualSynth time (s)": table["qualsynth_mean_generation_time_seconds"].round(1),
            "SMOTE time (s)": table["smote_mean_generation_time_seconds"].round(1),
            "CTGAN time (s)": table["ctgan_mean_generation_time_seconds"].round(1),
            "TabDDPM time (s)": table["tabddpm_mean_generation_time_seconds"].round(1),
            "QS calls": table["qualsynth_mean_llm_calls"].round(1),
            "QS tokens (k)": table["qualsynth_mean_total_tokens_k"].round(1),
            "Local cost (USD)": table["qualsynth_mean_generation_cost_usd"].round(2),
        }
    )
    markdown = [
        "# Manuscript Cost/Runtime Table",
        "",
        "Generated by `scripts/analyze_cost_runtime.py` from archived progress logs and the instrumented full-pipeline QualSynth token summary.",
        "",
        display.to_markdown(index=False),
        "",
    ]
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    tex_rows = []
    for row in display.itertuples(index=False):
        tex_rows.append(
            f"{row[0]} & {row[1]:.1f} & {row[2]:.1f} & "
            f"{row[3]:.1f} & {row[4]:.1f} & {row[5]:.1f} & {row[6]:.1f} & "
            f"{row[7]:.2f} \\\\"
        )
    tex_path.write_text("\n".join(tex_rows) + "\n", encoding="utf-8")


def _write_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        output_path.write_text(
            "# Cost and Runtime Summary\n\nNo successful experiment result files were found.\n",
            encoding="utf-8",
        )
        return

    display = summary[
        [
            "dataset",
            "method",
            "n_runs",
            "mean_generation_time_seconds",
            "mean_training_time_seconds",
            "mean_evaluation_time_seconds",
            "mean_generated_samples",
            "mean_llm_calls",
            "mean_total_tokens",
            "mean_generation_cost_usd",
        ]
    ].copy()
    numeric_cols = [col for col in display.columns if col not in {"dataset", "method", "n_runs"}]
    display[numeric_cols] = display[numeric_cols].round(3)
    markdown = [
        "# Cost and Runtime Summary",
        "",
        "This artifact is generated from per-seed experiment JSON files. Local LM Studio runs default to zero monetary API cost; token totals remain reported for reproducibility and optional pricing conversion.",
        "",
        display.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(markdown), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate QualSynth cost/runtime metrics.")
    parser.add_argument(
        "--result-root",
        action="append",
        type=Path,
        default=None,
        help="Root directory containing per-seed result JSON files. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated reviewer-revision analysis artifacts.",
    )
    parser.add_argument(
        "--progress-root",
        action="append",
        type=Path,
        default=None,
        help="Root directory containing archived *_progress.json timing logs. Can be repeated.",
    )
    parser.add_argument(
        "--component-token-summary",
        type=Path,
        default=DEFAULT_COMPONENT_TOKEN_SUMMARY,
        help="Component-ablation cost/runtime summary used for QualSynth call/token counts.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir = output_dir.resolve()
    result_roots = args.result_root or list(DEFAULT_RESULT_ROOTS)
    result_roots = [
        path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        for path in result_roots
    ]
    progress_roots = args.progress_root or list(DEFAULT_PROGRESS_ROOTS)
    progress_roots = [
        path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        for path in progress_roots
    ]
    component_token_summary = (
        args.component_token_summary
        if args.component_token_summary.is_absolute()
        else PROJECT_ROOT / args.component_token_summary
    ).resolve()

    result_files = _discover_result_files(result_roots)
    progress_files = _discover_progress_files(progress_roots)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [_load_result(path) for path in result_files]
    progress_rows = [
        row for path in progress_files if (row := _load_progress_result(path)) is not None
    ]
    runs = pd.DataFrame([*result_rows, *progress_rows])
    if runs.empty:
        runs = pd.DataFrame(
            columns=[
                "source_file",
                "dataset",
                "method",
                "seed",
                "success",
                "execution_time_seconds",
                "generation_time_seconds",
                "training_time_seconds",
                "evaluation_time_seconds",
                "generation_cost_usd",
                "llm_calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "n_generated",
                "n_generated_raw",
                "n_validated",
                "target_samples",
                "model_name",
            ]
        )

    summary = _summarize_runs(runs)

    per_run_csv = output_dir / "cost_runtime_per_run.csv"
    summary_csv = output_dir / "cost_runtime_summary.csv"
    markdown_path = output_dir / "cost_runtime_summary.md"
    metadata_path = output_dir / "cost_runtime_metadata.json"

    runs.to_csv(per_run_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _write_markdown(summary, markdown_path)

    progress_runs = pd.DataFrame(progress_rows)
    progress_summary = _summarize_runs(progress_runs) if not progress_runs.empty else pd.DataFrame()
    token_summary = _load_qualsynth_token_summary(component_token_summary)
    manuscript_table = _build_manuscript_table(progress_summary, token_summary)
    _write_manuscript_outputs(manuscript_table, output_dir)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "result_roots": [str(path) for path in result_roots],
        "progress_roots": [str(path) for path in progress_roots],
        "component_token_summary": str(component_token_summary),
        "n_result_files": len(result_files),
        "n_progress_files": len(progress_files),
        "n_successful_runs": int(runs["success"].sum()) if "success" in runs else 0,
        "outputs": {
            "per_run_csv": str(per_run_csv.relative_to(PROJECT_ROOT)),
            "summary_csv": str(summary_csv.relative_to(PROJECT_ROOT)),
            "markdown": str(markdown_path.relative_to(PROJECT_ROOT)),
            "manuscript_table_csv": str(
                (output_dir / "cost_runtime_manuscript_table.csv").relative_to(PROJECT_ROOT)
            ),
            "manuscript_table_markdown": str(
                (output_dir / "cost_runtime_manuscript_table.md").relative_to(PROJECT_ROOT)
            ),
            "manuscript_table_tex_rows": str(
                (output_dir / "cost_runtime_manuscript_table_rows.tex").relative_to(PROJECT_ROOT)
            ),
        },
        "pricing_note": "LM Studio local API calls are recorded as $0 by default. Set LLM_INPUT_COST_PER_1M_TOKENS and LLM_OUTPUT_COST_PER_1M_TOKENS before future runs to attach nonzero token-based API cost.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Loaded {len(result_files)} result files")
    print(f"Loaded {len(progress_files)} progress files")
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {output_dir / 'cost_runtime_manuscript_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
