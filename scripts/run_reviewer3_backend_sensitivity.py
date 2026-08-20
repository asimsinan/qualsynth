#!/usr/bin/env python3
"""Run the paired Reviewer 3 Gemma/Luna backend conditions."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualsynth.experiments.experiment_runner import ExperimentRunner  # noqa: E402
from src.qualsynth.utils.config_loader import ConfigLoader  # noqa: E402


METHODS = ["qualsynth_r3_gemma", "qualsynth_r3_luna"]
PILOT_DATASETS = ["haberman", "breast_cancer"]
CORE_DATASETS = ["haberman", "breast_cancer", "pima_diabetes", "yeast"]
PILOT_SEEDS = [42]
CORE_SEEDS = [42, 123, 456]
BASE_OUTPUT = (
    PROJECT_ROOT
    / "results/reviewer_revision/reviewer3_round2/backend_sensitivity"
)
MODEL_SPECIFIC_KEYS = {"model_name", "reasoning_effort"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "core"], required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--continue-shortfalls",
        action="store_true",
        help=(
            "Resume a retained target-shortfall condition from its validated-sample "
            "CSV without changing the frozen scientific settings."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--approve-paid-core",
        action="store_true",
        help="Required for the paid 24-run core matrix after pilot review.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_and_validate_methods() -> tuple[dict[str, Any], str]:
    loader = ConfigLoader()
    configs = {name: loader.load_method_config(name) for name in METHODS}
    common_payloads = {}
    for name, config in configs.items():
        parameters = dict(config.hyperparameters or {})
        common_payloads[name] = {
            key: value
            for key, value in parameters.items()
            if key not in MODEL_SPECIFIC_KEYS
        }
    reference = common_payloads[METHODS[0]]
    for name in METHODS[1:]:
        if common_payloads[name] != reference:
            left = set(reference.items()) - set(common_payloads[name].items())
            right = set(common_payloads[name].items()) - set(reference.items())
            raise ValueError(
                f"Non-model configuration drift for {name}: "
                f"reference_only={sorted(left)}, condition_only={sorted(right)}"
            )
    return configs, canonical_sha256(reference)


def result_path(output_root: Path, dataset: str, method: str, seed: int) -> Path:
    return output_root / dataset / method / f"seed{seed}.json"


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_matches_condition(
    payload: dict[str, Any],
    dataset: str,
    method: str,
    seed: int,
    expected_model: str,
    expected_iterations: int,
) -> bool:
    provenance = (payload.get("metadata") or {}).get("run_provenance") or {}
    stopping = provenance.get("stopping") or {}
    exact_core_target = bool(
        stopping.get("target_reached")
        and int(payload.get("n_generated") or -1)
        == int(stopping.get("target_samples") or -2)
    )
    return bool(
        payload.get("success")
        and payload.get("dataset") == dataset
        and payload.get("method") == method
        and int(payload.get("seed")) == seed
        and provenance.get("requested_model_slug") == expected_model
        and (
            (expected_iterations == 0 and exact_core_target)
            or int(stopping.get("iterations") or -1) == expected_iterations
        )
    )


def summarized_result(path: Path, payload: dict[str, Any], status: str) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    provenance = metadata.get("run_provenance") or {}
    usage = provenance.get("usage") or {}
    stopping = provenance.get("stopping") or {}
    return {
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "status": status,
        "success": payload.get("success"),
        "error": payload.get("error"),
        "requested_model": provenance.get("requested_model_slug"),
        "resolved_model": provenance.get("resolved_model_slug"),
        "provider": provenance.get("resolved_provider"),
        "calls": usage.get("calls", payload.get("llm_calls")),
        "prompt_tokens": usage.get("prompt_tokens", payload.get("prompt_tokens")),
        "completion_tokens": usage.get(
            "completion_tokens", payload.get("completion_tokens")
        ),
        "reasoning_tokens": (provenance.get("reasoning") or {}).get("tokens"),
        "total_tokens": usage.get("total_tokens", payload.get("total_tokens")),
        "cost_usd": usage.get("cost_usd", payload.get("generation_cost")),
        "generation_time_seconds": usage.get(
            "generation_time_seconds", payload.get("generation_time")
        ),
        "target_samples": stopping.get("target_samples", metadata.get("target_samples")),
        "target_reached": stopping.get("target_reached", metadata.get("target_reached")),
        "target_shortfall": stopping.get(
            "target_shortfall", metadata.get("target_shortfall")
        ),
        "iterations": stopping.get("iterations", metadata.get("iterations")),
        "result_path": str(path.relative_to(PROJECT_ROOT)),
        "result_sha256": sha256_file(path),
    }


def continuation_row_count(
    output_root: Path,
    dataset: str,
    method: str,
    seed: int,
    payload: dict[str, Any],
) -> int:
    """Validate the checkpoint used for an explicit shortfall continuation."""
    stopping = ((payload.get("metadata") or {}).get("run_provenance") or {}).get(
        "stopping"
    ) or {}
    selected = int(payload.get("n_generated") or 0)
    target = int(stopping.get("target_samples") or 0)
    if selected <= 0 or target <= selected:
        raise ValueError(
            "--continue-shortfalls requires an existing positive target shortfall"
        )
    checkpoint = (
        output_root
        / "logs"
        / f"{dataset}_{method}_seed{seed}_validated_samples.csv"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Continuation checkpoint is missing: {checkpoint}")
    with checkpoint.open("r", encoding="utf-8") as handle:
        checkpoint_rows = max(sum(1 for _ in handle) - 1, 0)
    if checkpoint_rows != selected:
        raise ValueError(
            f"Continuation checkpoint has {checkpoint_rows} rows; result records {selected}"
        )
    return checkpoint_rows


def merge_continuation_accounting(
    path: Path,
    base_payload: dict[str, Any],
    base_sha256: str,
    continuation_from_rows: int,
) -> dict[str, Any]:
    """Combine cumulative timing and stopping metadata after a resumed tail."""
    payload = load_result(path)
    metadata = payload.setdefault("metadata", {})
    base_metadata = base_payload.get("metadata") or {}
    provenance = metadata.setdefault("run_provenance", {})
    base_provenance = base_metadata.get("run_provenance") or {}
    usage = provenance.setdefault("usage", {})
    base_usage = base_provenance.get("usage") or {}
    stopping = provenance.setdefault("stopping", {})
    base_stopping = base_provenance.get("stopping") or {}

    continuation_generation_time = float(payload.get("generation_time") or 0.0)
    base_generation_time = float(
        base_usage.get("generation_time_seconds")
        or base_payload.get("generation_time")
        or 0.0
    )
    combined_generation_time = base_generation_time + continuation_generation_time
    payload["generation_time"] = combined_generation_time
    metadata["generation_time"] = combined_generation_time
    usage["generation_time_seconds"] = combined_generation_time

    continuation_execution_time = float(payload.get("execution_time") or 0.0)
    base_execution_time = float(base_payload.get("execution_time") or 0.0)
    payload["execution_time"] = base_execution_time + continuation_execution_time

    combined_iterations = int(base_stopping.get("iterations") or 0) + int(
        stopping.get("iterations") or 0
    )
    stopping["iterations"] = combined_iterations
    metadata["iterations"] = combined_iterations
    metadata["completion_continuation"] = {
        "continued": True,
        "base_result_sha256": base_sha256,
        "continuation_from_rows": continuation_from_rows,
        "resume_generation_override": True,
        "scientific_settings_changed": False,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Load the existing OpenRouter environment before running."
        )
    if args.stage == "core" and not args.approve_paid_core:
        raise RuntimeError(
            "The paid core matrix is gated. Review the completed pilot report and rerun "
            "with --approve-paid-core."
        )

    configs, common_contract_sha256 = load_and_validate_methods()
    datasets = args.datasets or (
        PILOT_DATASETS if args.stage == "pilot" else CORE_DATASETS
    )
    seeds = args.seeds or (PILOT_SEEDS if args.stage == "pilot" else CORE_SEEDS)
    allowed_datasets = set(PILOT_DATASETS if args.stage == "pilot" else CORE_DATASETS)
    if not set(datasets).issubset(allowed_datasets):
        raise ValueError(
            f"Datasets outside the frozen {args.stage} matrix: "
            f"{sorted(set(datasets) - allowed_datasets)}"
        )
    output_root = args.output_root or (BASE_OUTPUT / args.stage)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    max_iterations = 1 if args.stage == "pilot" else 0
    batch_size = 5 if args.stage == "pilot" else 20
    run_rows = []
    failures = []
    for dataset in datasets:
        for seed in seeds:
            for method in METHODS:
                path = result_path(output_root, dataset, method, seed)
                expected_model = str(configs[method].hyperparameters["model_name"])
                existing: dict[str, Any] | None = None
                existing_sha256: str | None = None
                if not args.no_resume and path.exists():
                    existing_sha256 = sha256_file(path)
                    existing = load_result(path)
                    if result_matches_condition(
                        existing,
                        dataset,
                        method,
                        seed,
                        expected_model,
                        max_iterations,
                    ):
                        print(f"SKIP {dataset}/seed{seed}/{method}: verified existing result")
                        run_rows.append(summarized_result(path, existing, "verified_existing"))
                        continue

                continuation_rows = 0
                if args.continue_shortfalls:
                    if existing is None:
                        raise ValueError(
                            "--continue-shortfalls requires an existing result artifact"
                        )
                    continuation_rows = continuation_row_count(
                        output_root,
                        dataset,
                        method,
                        seed,
                        existing,
                    )

                print(f"RUN  {dataset}/seed{seed}/{method} ({expected_model})")
                runner = ExperimentRunner(
                    output_dir=str(output_root),
                    verbose=not args.quiet,
                    classifier_imbalance_policy="balanced",
                    run_context={
                        "experiment_name": "reviewer3_backend_sensitivity",
                        "stage": args.stage,
                        "common_contract_sha256": common_contract_sha256,
                        "completion_continuation": bool(continuation_rows),
                        "continuation_from_rows": continuation_rows,
                    },
                )
                run_kwargs = {
                    "dataset_name": dataset,
                    "method_name": method,
                    "seed": seed,
                    "max_iterations_override": max_iterations,
                    "batch_size_override": batch_size,
                    "resume_generation_override": True if continuation_rows else None,
                }
                if args.quiet:
                    console_path = (
                        output_root
                        / "logs"
                        / f"{dataset}_{method}_seed{seed}_console.log"
                    )
                    console_path.parent.mkdir(parents=True, exist_ok=True)
                    console_mode = "a" if continuation_rows else "w"
                    with console_path.open(console_mode, encoding="utf-8") as console:
                        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                            result = runner.run_experiment(**run_kwargs)
                    print(
                        f"DONE {dataset}/seed{seed}/{method}: "
                        f"success={result.success}, calls={result.llm_calls}, "
                        f"selected={result.n_generated}, cost=${result.generation_cost:.6f}"
                    )
                else:
                    result = runner.run_experiment(**run_kwargs)
                if not path.exists():
                    raise RuntimeError(f"Expected result artifact was not written: {path}")
                if continuation_rows:
                    if existing is None or existing_sha256 is None:
                        raise RuntimeError("Continuation base result was not retained")
                    payload = merge_continuation_accounting(
                        path,
                        existing,
                        existing_sha256,
                        continuation_rows,
                    )
                else:
                    payload = load_result(path)
                status = "completed" if result.success else "failed"
                run_rows.append(summarized_result(path, payload, status))
                if not result.success:
                    failures.append(
                        f"{dataset}/seed{seed}/{method}: {result.error or 'unknown failure'}"
                    )

    summary = {
        "name": "reviewer3_backend_sensitivity_execution",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "seeds": seeds,
        "methods": METHODS,
        "max_iterations": max_iterations,
        "batch_size": batch_size,
        "common_request_contract": "single_user_csv_v1",
        "common_non_model_contract_sha256": common_contract_sha256,
        "unsupported_parameter_policy": "fail_fast",
        "paid_core_gate_approved": bool(args.approve_paid_core),
        "runs": run_rows,
        "failures": failures,
    }
    summary_path = output_root / "execution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    if failures:
        raise RuntimeError(f"Backend study retained {len(failures)} failed runs")


if __name__ == "__main__":
    main()
