"""Shared reviewer-revision artifact helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def as_float(value: Any) -> float:
    """Return a finite float, using 0.0 for missing or invalid values."""
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def as_int(value: Any) -> int:
    """Return an int, using 0 for missing or invalid values."""
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def cost_runtime_row(result: Mapping[str, Any], source_file: str | Path | None = None) -> dict[str, Any]:
    """Flatten one experiment result payload into cost/runtime columns."""
    metadata = result.get("metadata") or {}
    n_generated = as_int(result.get("n_generated", metadata.get("n_validated")))
    total_tokens = as_int(result.get("total_tokens", metadata.get("total_tokens")))
    generation_cost = as_float(result.get("generation_cost", metadata.get("generation_cost")))

    return {
        "source_file": str(source_file) if source_file is not None else "",
        "dataset": result.get("dataset"),
        "method": result.get("method"),
        "seed": result.get("seed"),
        "success": bool(result.get("success", False)),
        "error": result.get("error"),
        "execution_time_seconds": as_float(result.get("execution_time")),
        "generation_time_seconds": as_float(result.get("generation_time")),
        "training_time_seconds": as_float(result.get("training_time", metadata.get("training_time"))),
        "evaluation_time_seconds": as_float(result.get("evaluation_time", metadata.get("evaluation_time"))),
        "generation_cost_usd": generation_cost,
        "llm_calls": as_int(result.get("llm_calls", metadata.get("llm_calls"))),
        "prompt_tokens": as_int(result.get("prompt_tokens", metadata.get("prompt_tokens"))),
        "completion_tokens": as_int(
            result.get("completion_tokens", metadata.get("completion_tokens"))
        ),
        "total_tokens": total_tokens,
        "n_generated": n_generated,
        "n_generated_raw": as_int(metadata.get("n_generated_raw")),
        "n_validated": as_int(metadata.get("n_validated", result.get("n_generated"))),
        "target_samples": as_int(metadata.get("target_samples")),
        "target_shortfall": as_int(metadata.get("target_shortfall")),
        "validation_rate": metadata.get("validation_rate"),
        "model_name": metadata.get("model_name"),
        "prompt_policy": metadata.get("prompt_policy"),
        "validation_policy": metadata.get("validation_policy"),
        "selection_policy": metadata.get("selection_policy"),
        "cost_per_generated_sample": generation_cost / n_generated if n_generated else 0.0,
        "tokens_per_generated_sample": total_tokens / n_generated if n_generated else 0.0,
    }


def summarize_cost_runtime(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize flattened cost/runtime rows by method and dataset."""
    if rows.empty:
        return pd.DataFrame()

    successful = rows[rows["success"]].copy()
    if successful.empty:
        return pd.DataFrame()

    grouped = successful.groupby(["dataset", "method"], dropna=False)
    summary = grouped.agg(
        n_runs=("source_file", "count"),
        mean_execution_time_seconds=("execution_time_seconds", "mean"),
        mean_generation_time_seconds=("generation_time_seconds", "mean"),
        mean_training_time_seconds=("training_time_seconds", "mean"),
        mean_evaluation_time_seconds=("evaluation_time_seconds", "mean"),
        mean_generated_samples=("n_generated", "mean"),
        total_generated_samples=("n_generated", "sum"),
        mean_llm_calls=("llm_calls", "mean"),
        total_llm_calls=("llm_calls", "sum"),
        mean_prompt_tokens=("prompt_tokens", "mean"),
        mean_completion_tokens=("completion_tokens", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        total_tokens=("total_tokens", "sum"),
        mean_generation_cost_usd=("generation_cost_usd", "mean"),
        total_generation_cost_usd=("generation_cost_usd", "sum"),
        mean_cost_per_generated_sample=("cost_per_generated_sample", "mean"),
        mean_tokens_per_generated_sample=("tokens_per_generated_sample", "mean"),
    ).reset_index()
    return summary.sort_values(["dataset", "method"]).reset_index(drop=True)


def write_cost_runtime_markdown(
    summary: pd.DataFrame,
    output_path: Path,
    title: str = "Cost and Runtime Summary",
) -> None:
    """Write a compact reviewer-facing Markdown cost/runtime table."""
    if summary.empty:
        output_path.write_text(f"# {title}\n\nNo successful runs were found.\n", encoding="utf-8")
        return

    display_cols = [
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
        "mean_cost_per_generated_sample",
    ]
    display = summary[[col for col in display_cols if col in summary.columns]].copy()
    numeric_cols = [col for col in display.columns if col not in {"dataset", "method", "n_runs"}]
    display[numeric_cols] = display[numeric_cols].round(3)
    markdown = [
        f"# {title}",
        "",
        "Local LM Studio runs report measured wall-clock/token usage. Monetary cost remains zero unless token pricing environment variables were set before generation.",
        "",
        display.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(markdown), encoding="utf-8")
