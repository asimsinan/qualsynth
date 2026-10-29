"""Experiment runner used by benchmark and ablation scripts."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.qualsynth.data.splitting import load_split
from src.qualsynth.evaluation.classifiers import ClassifierPipeline
from src.qualsynth.utils.config_loader import ConfigLoader, MethodConfig


@dataclass
class ExperimentResult:
    """Serializable payload for one dataset / method / seed run."""

    experiment_id: str
    dataset: str
    method: str
    seed: int
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    generation_time: float = 0.0
    training_time: float = 0.0
    evaluation_time: float = 0.0
    generation_cost: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_generated: int = 0
    performance_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)
    avg_performance: Dict[str, float] = field(default_factory=dict)
    avg_fairness: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentRunner:
    """Run one synthetic-data experiment and persist JSON/CSV artifacts."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        enable_universal_validation: bool = False,
        validation_mode: str = "standard",
        classifier_imbalance_policy: str = "balanced",
        run_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir or "results/experiments")
        self.log_dir = self.output_dir / "logs"
        self.verbose = verbose
        self.enable_universal_validation = enable_universal_validation
        self.validation_mode = validation_mode
        self.classifier_imbalance_policy = str(classifier_imbalance_policy).strip().lower()
        if self.classifier_imbalance_policy not in ClassifierPipeline.VALID_IMBALANCE_POLICIES:
            allowed = ", ".join(sorted(ClassifierPipeline.VALID_IMBALANCE_POLICIES))
            raise ValueError(
                f"Unknown classifier_imbalance_policy={classifier_imbalance_policy!r}; "
                f"expected one of: {allowed}."
            )
        self.run_context = dict(run_context or {})
        self.config_loader = ConfigLoader()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        dataset_name: str,
        method_name: str,
        seed: int,
        save_results: bool = True,
        max_iterations_override: Optional[int] = None,
        model_name_override: Optional[str] = None,
        batch_size_override: Optional[int] = None,
        resume_generation_override: Optional[bool] = None,
    ) -> ExperimentResult:
        started_at = time.time()
        experiment_id = f"{dataset_name}_{method_name}_seed{seed}"
        X_generated: Optional[pd.DataFrame] = None
        X_validated: Optional[pd.DataFrame] = None
        generation_meta: Dict[str, Any] = {}
        validation_meta: Dict[str, Any] = {}
        artifact_payload: Dict[str, Any] = {}
        method_config: Optional[MethodConfig] = None
        split_provenance: Dict[str, Any] = {}
        classifier_policy = self.classifier_imbalance_policy

        try:
            dataset_config = self.config_loader.load_dataset_config(dataset_name)
            method_config = self.config_loader.load_method_config(method_name)
            classifier_policy = str(
                (method_config.hyperparameters or {}).get(
                    "classifier_imbalance_policy",
                    self.classifier_imbalance_policy,
                )
            ).strip().lower()
            split = load_split(
                dataset_name,
                seed=seed,
                include_sensitive_indicators=True,
                dataset_config=dataset_config,
            )

            X_train = split["X_train"].copy().reset_index(drop=True)
            y_train = pd.Series(split["y_train"]).reset_index(drop=True)
            X_val = split["X_val"].copy().reset_index(drop=True)
            y_val = pd.Series(split["y_val"]).reset_index(drop=True)
            X_test = split["X_test"].copy().reset_index(drop=True)
            y_test = pd.Series(split["y_test"]).reset_index(drop=True)
            # Align sensitive attributes with the reset X/y indices above
            sensitive_test = split.get("sensitive_test")
            if sensitive_test is not None:
                sensitive_test = sensitive_test.reset_index(drop=True)
            sensitive_train = split.get("sensitive_train")
            if sensitive_train is not None:
                sensitive_train = sensitive_train.reset_index(drop=True)

            y_train, y_val, y_test, target_mapping = self._normalize_binary_labels(
                y_train,
                y_val,
                y_test,
            )
            split_provenance = self._build_split_provenance(
                dataset_name=dataset_name,
                seed=seed,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                target_mapping=target_mapping,
            )

            target_samples = self._target_synthetic_count(y_train)
            X_generated, y_generated, generation_meta = self._generate_samples(
                method_name=method_name,
                method_config=method_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                sensitive_train=sensitive_train,
                target_samples=target_samples,
                seed=seed,
                dataset_name=dataset_name,
                max_iterations_override=max_iterations_override,
                model_name_override=model_name_override,
                batch_size_override=batch_size_override,
                resume_generation_override=resume_generation_override,
            )
            artifact_payload = generation_meta.pop("_artifact_payload", {})

            if method_name.lower().startswith("qualsynth") and target_samples > 0 and X_generated.empty:
                raise RuntimeError(
                    "QualSynth generated zero candidate rows; the run is a failed generation, "
                    "not a successful no-augmentation result."
                )

            if generation_meta.get("validation_policy") == "raw":
                X_validated, y_validated, validation_meta = X_generated, y_generated, {
                    "validation_applied": False,
                    "validation_rate": 1.0 if not X_generated.empty else 0.0,
                    "outer_validation_bypassed": True,
                    "outer_validation_bypass_reason": "raw validation policy",
                }
            else:
                X_validated, y_validated, validation_meta = self._maybe_validate(
                    X_generated=X_generated,
                    y_generated=y_generated,
                    X_train=X_train,
                    y_train=y_train,
                    method_name=method_name,
                )

            if X_validated.empty:
                # Preserve the exact original fold and its dtypes for the
                # no-augmentation comparator (and any zero-target edge case).
                X_augmented = X_train.copy()
                y_augmented = y_train.copy()
            else:
                X_augmented = pd.concat([X_train, X_validated], ignore_index=True)
                y_augmented = pd.concat([y_train, y_validated], ignore_index=True)

            pipeline = ClassifierPipeline(
                random_state=seed,
                imbalance_policy=classifier_policy,
            )
            training_started_at = time.time()
            pipeline.train(X_augmented, y_augmented, verbose=self.verbose)
            training_time = time.time() - training_started_at
            evaluation_started_at = time.time()
            performance_metrics = pipeline.evaluate(
                X_test,
                y_test,
                sensitive_features=sensitive_test,
                compute_fairness=sensitive_test is not None,
                verbose=self.verbose,
            )
            evaluation_time = time.time() - evaluation_started_at

            avg_performance = self._average_metrics(performance_metrics)
            avg_fairness = self._average_fairness(performance_metrics)
            metadata = {
                "target_samples": target_samples,
                "n_generated_raw": len(X_generated),
                "n_validated": len(X_validated),
                "training_time": training_time,
                "evaluation_time": evaluation_time,
                "universal_validation_enabled": self.enable_universal_validation,
                "classifier_imbalance_policy": classifier_policy,
                "classifier_policy_details": pipeline.get_imbalance_policy_metadata(),
                "split_provenance": split_provenance,
                **generation_meta,
                **validation_meta,
            }
            metadata["run_provenance"] = self._build_run_provenance(
                dataset_name=dataset_name,
                method_name=method_name,
                seed=seed,
                method_config=method_config,
                split_provenance=split_provenance,
                classifier_policy=classifier_policy,
                metadata=metadata,
            )
            self._assert_required_run_metadata(metadata["run_provenance"])

            result = ExperimentResult(
                experiment_id=experiment_id,
                dataset=dataset_name,
                method=method_name,
                seed=seed,
                success=True,
                execution_time=time.time() - started_at,
                generation_time=float(generation_meta.get("generation_time", 0.0)),
                training_time=training_time,
                evaluation_time=evaluation_time,
                generation_cost=float(generation_meta.get("generation_cost", 0.0) or 0.0),
                llm_calls=int(generation_meta.get("llm_calls", 0) or 0),
                prompt_tokens=int(generation_meta.get("prompt_tokens", 0) or 0),
                completion_tokens=int(generation_meta.get("completion_tokens", 0) or 0),
                total_tokens=int(generation_meta.get("total_tokens", 0) or 0),
                n_generated=len(X_validated),
                performance_metrics=performance_metrics,
                fairness_metrics=avg_fairness,
                avg_performance=avg_performance,
                avg_fairness=avg_fairness,
                metadata=metadata,
            )
            if save_results:
                self._save_artifacts(
                    result,
                    X_generated,
                    X_validated,
                    dataset_name,
                    method_name,
                    seed,
                    artifact_payload=artifact_payload,
                )
            return result

        except Exception as exc:
            if method_config is not None and split_provenance:
                failure_provenance = self._build_run_provenance(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    seed=seed,
                    method_config=method_config,
                    split_provenance=split_provenance,
                    classifier_policy=classifier_policy,
                    metadata={**generation_meta, **validation_meta},
                )
            else:
                failure_provenance = None
            failure_metadata = {
                "traceback": traceback.format_exc(),
                **generation_meta,
                **validation_meta,
            }
            if failure_provenance is not None:
                failure_metadata["run_provenance"] = failure_provenance
            result = ExperimentResult(
                experiment_id=experiment_id,
                dataset=dataset_name,
                method=method_name,
                seed=seed,
                success=False,
                error=str(exc),
                execution_time=time.time() - started_at,
                generation_time=float(generation_meta.get("generation_time", 0.0) or 0.0),
                generation_cost=float(generation_meta.get("generation_cost", 0.0) or 0.0),
                llm_calls=int(generation_meta.get("llm_calls", 0) or 0),
                prompt_tokens=int(generation_meta.get("prompt_tokens", 0) or 0),
                completion_tokens=int(generation_meta.get("completion_tokens", 0) or 0),
                total_tokens=int(generation_meta.get("total_tokens", 0) or 0),
                n_generated=len(X_validated) if X_validated is not None else 0,
                metadata=failure_metadata,
            )
            if save_results:
                generated_for_save = (
                    X_generated
                    if X_generated is not None
                    else pd.DataFrame()
                )
                validated_for_save = (
                    X_validated
                    if X_validated is not None
                    else pd.DataFrame(columns=generated_for_save.columns)
                )
                self._save_artifacts(
                    result,
                    generated_for_save,
                    validated_for_save,
                    dataset_name,
                    method_name,
                    seed,
                    artifact_payload=artifact_payload,
                )
            if self.verbose:
                print(f"Experiment failed: {experiment_id}: {exc}")
            return result

    def _generate_samples(
        self,
        method_name: str,
        method_config: MethodConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_train: Optional[pd.DataFrame],
        target_samples: int,
        seed: int,
        dataset_name: str,
        max_iterations_override: Optional[int],
        model_name_override: Optional[str],
        batch_size_override: Optional[int],
        resume_generation_override: Optional[bool],
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        method_key = method_name.lower()
        params = dict(method_config.hyperparameters or {})
        started_at = time.time()

        if method_key.startswith("no_augmentation"):
            X_empty, y_empty = self._empty_generated(X_train, y_train)
            return X_empty, y_empty, {
                "generation_time": 0.0,
                "generation_cost": 0.0,
                "augmentation_applied": False,
                "augmentation_policy": "original_training_fold_only",
                "target_samples": target_samples,
                "target_reached": True,
                "target_shortfall": 0,
                "iterations": 0,
                "converged": True,
                "convergence_reason": "No augmentation requested",
                "validation_policy": "not_applicable",
                "selection_policy": "not_applicable",
                "validation_mode": "not_applicable",
            }

        if target_samples <= 0:
            X_empty, y_empty = self._empty_generated(X_train, y_train)
            return X_empty, y_empty, {"generation_time": 0.0, "generation_cost": 0.0}

        if method_key.startswith("qualsynth"):
            return self._run_qualsynth(
                params,
                X_train,
                y_train,
                target_samples,
                started_at,
                max_iterations_override,
                model_name_override,
                batch_size_override,
                resume_generation_override,
                dataset_name,
                method_name,
                seed,
            )
        if method_key == "smote":
            return self._run_smote(params, X_train, y_train, target_samples, seed, started_at)
        if method_key == "ctgan":
            return self._run_ctgan(params, X_train, y_train, target_samples, seed, started_at)
        if method_key == "tabfairgdt":
            return self._run_tabfairgdt(
                params, X_train, y_train, sensitive_train, target_samples, seed, started_at
            )
        if method_key == "tabddpm":
            return self._run_tabddpm(
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                target_samples,
                seed,
                dataset_name,
            )
        raise ValueError(f"Unsupported method: {method_name}")

    def _run_qualsynth(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_samples: int,
        started_at: float,
        max_iterations_override: Optional[int],
        model_name_override: Optional[str],
        batch_size_override: Optional[int],
        resume_generation_override: Optional[bool],
        dataset_name: str,
        method_name: str,
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        from src.qualsynth.generator import QualSynthGenerator

        max_iterations = max_iterations_override
        if max_iterations is None:
            max_iterations = int(params.get("max_iterations", 0) or 0)

        api_base = params.get("api_base")
        api_key = params.get("api_key")
        if api_base and "openrouter" in str(api_base).lower() and not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

        generator = QualSynthGenerator(
            model_name=model_name_override or params.get("model_name", "gemma3:12b"),
            temperature=float(params.get("temperature", 0.7)),
            top_p=float(params.get("top_p", 0.95)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            max_output_tokens=int(params.get("max_output_tokens", 8192)),
            reasoning_effort=params.get("reasoning_effort"),
            strict_request_contract=bool(params.get("strict_request_contract", False)),
            resume_generation=(
                bool(resume_generation_override)
                if resume_generation_override is not None
                else bool(params.get("resume_generation", True))
            ),
            max_iterations=max_iterations,
            min_iterations=int(params.get("min_iterations", 3)),
            stall_iterations=int(params.get("stall_iterations", 10)),
            batch_size=batch_size_override or int(params.get("batch_size", 20)),
            target_ratio=1.0,
            validation_threshold=float(params.get("adaptive_std_threshold", 4.5)),
            duplicate_threshold=float(params.get("duplicate_threshold", 0.10)),
            quality_threshold=float(params.get("quality_threshold", 0.5)),
            enable_adaptive_validation=bool(params.get("enable_adaptive_validation", False)),
            adaptive_percentile_threshold=float(params.get("adaptive_percentile_threshold", 0.995)),
            enable_diversity_first_selection=bool(params.get("enable_diversity_first_selection", False)),
            diversity_first_ratio=float(params.get("diversity_first_ratio", 0.5)),
            enable_statistical_validation=bool(params.get("enable_statistical_validation", True)),
            validation_mode=str(params.get("validation_mode", self.validation_mode)),
            diversity_weight=float(params.get("diversity_weight", 0.20)),
            performance_weight=float(params.get("performance_weight", 0.20)),
            enable_sota_dedup=bool(params.get("enable_sota_dedup", True)),
            enable_semantic_dedup=bool(params.get("enable_semantic_dedup", False)),
            anchor_selection_strategy=str(params.get("anchor_selection_strategy", "stratified")),
            use_few_shot=bool(params.get("use_few_shot", True)),
            n_few_shot_examples=int(params.get("n_few_shot_examples", 5)),
            diversity_prompt_strength=str(params.get("diversity_prompt_strength", "high")),
            prompt_policy=str(params.get("prompt_policy", "anchor")),
            validation_policy=str(params.get("validation_policy", "full")),
            selection_policy=str(params.get("selection_policy", "multi_objective")),
            clip_to_minority_class=bool(params.get("clip_to_minority_class", False)),
            api_base=api_base,
            api_key=api_key,
            dataset_name=dataset_name,
            method_name=method_name,
            seed=seed,
            output_dir=str(self.output_dir),
            verbose=self.verbose,
        )
        # Categorical features declared in the dataset schema are excluded from anchor strata.
        schema_preprocessor = load_split(dataset_name, seed=seed).get("preprocessor")
        generator.declared_categorical_features = list(
            getattr(schema_preprocessor, "categorical_features", None) or []
        )
        X_syn, y_syn = generator.fit_generate(X_train, y_train, n_samples=target_samples)
        workflow_result = getattr(generator, "last_workflow_result_", None)
        request_records = list(getattr(workflow_result, "request_records", []) or [])
        request_summaries = [
            {key: value for key, value in record.items() if key != "raw_response"}
            for record in request_records
        ]
        resolved_models = sorted(
            {
                str(record.get("resolved_model_slug"))
                for record in request_records
                if record.get("resolved_model_slug")
            }
        )
        resolved_providers = sorted(
            {
                str(record.get("resolved_provider"))
                for record in request_records
                if record.get("resolved_provider")
            }
        )
        endpoint_classes = sorted(
            {
                str(record.get("endpoint_class"))
                for record in request_records
                if record.get("endpoint_class")
            }
        )
        prompt_hashes = sorted(
            {
                str(record.get("prompt_sha256"))
                for record in request_records
                if record.get("prompt_sha256")
            }
        )
        effective_config = {
            key: value
            for key, value in generator.get_params().items()
            if key not in {"api_key"}
        }
        effective_config.update(
            {
                "api_base": self._redact_endpoint(api_base),
                "target_samples": target_samples,
                "dataset_name": dataset_name,
                "method_name": method_name,
                "seed": seed,
            }
        )
        n_selected = (
            len(workflow_result.X_generated)
            if workflow_result is not None and getattr(workflow_result, "X_generated", None) is not None
            else len(X_syn)
        )
        return (
            X_syn.reset_index(drop=True),
            y_syn.reset_index(drop=True),
            {
                "generation_time": time.time() - started_at,
                "generation_cost": float(getattr(workflow_result, "total_cost", 0.0) or 0.0),
                "llm_calls": int(getattr(workflow_result, "llm_calls", 0) or 0),
                "prompt_tokens": int(getattr(workflow_result, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(workflow_result, "completion_tokens", 0) or 0),
                "reasoning_tokens": int(getattr(workflow_result, "reasoning_tokens", 0) or 0),
                "total_tokens": int(getattr(workflow_result, "total_tokens", 0) or 0),
                "model_name": generator.model_name,
                "requested_model_slug": generator.model_name,
                "resolved_model_slugs": resolved_models,
                "resolved_providers": resolved_providers,
                "endpoint_classes": endpoint_classes or [self._classify_endpoint(api_base)],
                "request_date_utc": (
                    request_records[0].get("request_date_utc")
                    if request_records
                    else datetime.now(timezone.utc).isoformat()
                ),
                "prompt_hashes": prompt_hashes,
                "effective_config_sha256": self._canonical_sha256(effective_config),
                "effective_config": effective_config,
                "request_summaries": request_summaries,
                "output_format": "csv",
                "message_contract": "single_user_csv_v1",
                "candidate_stage_semantics": {
                    "parsed": "provider response rows before encoding, clipping, or repair",
                    "validation_input": "rows after encoding, clipping, and deterministic repair",
                    "accepted": "validation-input rows that passed active validation gates",
                    "rejected": "validation-input rows not represented in the accepted row multiset",
                },
                "reasoning_effort": generator.reasoning_effort,
                "max_iterations": max_iterations,
                "target_samples": target_samples,
                "target_reached": bool(n_selected >= target_samples),
                "target_shortfall": max(0, target_samples - n_selected),
                "iterations": int(getattr(workflow_result, "total_iterations", 0) or 0),
                "converged": bool(getattr(workflow_result, "converged", False)),
                "convergence_reason": getattr(workflow_result, "convergence_reason", None),
                "batch_size": generator.batch_size,
                "validation_rate": getattr(generator, "validation_rate_", None),
                "prompt_policy": generator.prompt_policy,
                "validation_policy": generator.validation_policy,
                "selection_policy": generator.selection_policy,
                "validation_mode": generator.validation_mode,
                "adaptive_std_threshold": generator.validation_threshold,
                "adaptive_percentile_threshold_configured": generator.adaptive_percentile_threshold,
                "adaptive_percentile_threshold_executed": False,
                "clip_reference_population": "minority_training_fold",
                "stall_iterations": int(params.get("stall_iterations", 10)),
                "_artifact_payload": {
                    "parsed_candidates": getattr(workflow_result, "X_parsed", pd.DataFrame()),
                    "validation_input_candidates": getattr(
                        workflow_result,
                        "X_validation_input",
                        pd.DataFrame(),
                    ),
                    "accepted_candidates": getattr(workflow_result, "X_accepted", pd.DataFrame()),
                    "rejected_candidates": getattr(workflow_result, "X_rejected", pd.DataFrame()),
                    "request_records": request_records,
                },
            },
        )

    def _run_smote(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_samples: int,
        seed: int,
        started_at: float,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        from src.qualsynth.baselines.smote import SMOTEBaseline

        generator = SMOTEBaseline(
            k_neighbors=int(params.get("k_neighbors", 5)),
            sampling_strategy=params.get("sampling_strategy", "auto"),
            random_state=seed,
        )
        X_resampled, y_resampled = generator.fit_resample(X_train, y_train, n_samples=target_samples)
        return self._synthetic_tail(X_resampled, y_resampled, X_train, y_train, started_at)

    def _run_ctgan(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_samples: int,
        seed: int,
        started_at: float,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        from src.qualsynth.baselines.ctgan_baseline import CTGANBaseline

        generator = CTGANBaseline(
            epochs=int(params.get("epochs", 300)),
            batch_size=int(params.get("batch_size", 500)),
            generator_dim=tuple(params.get("generator_dim", [256, 256])),
            discriminator_dim=tuple(params.get("discriminator_dim", [256, 256])),
            generator_lr=float(params.get("generator_lr", 2e-4)),
            discriminator_lr=float(params.get("discriminator_lr", 2e-4)),
            discriminator_steps=int(params.get("discriminator_steps", 1)),
            verbose=bool(params.get("verbose", False)),
            random_state=seed,
        )
        X_resampled, y_resampled = generator.fit_resample(X_train, y_train, n_samples=target_samples)
        return self._synthetic_tail(X_resampled, y_resampled, X_train, y_train, started_at)

    def _run_tabfairgdt(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_train: Optional[pd.DataFrame],
        target_samples: int,
        seed: int,
        started_at: float,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        from src.qualsynth.baselines.tabfairgdt import TabFairGDT

        generator = TabFairGDT(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            fairness_weight=float(params.get("fairness_weight", 0.5)),
            random_state=seed,
        )
        target_class = y_train.value_counts().idxmin()
        generated = generator.generate(
            X_train,
            y_train,
            n_samples=target_samples,
            sensitive_features=sensitive_train,
            target_class=target_class,
        )
        y_syn = pd.Series([target_class] * len(generated.samples), name=y_train.name)
        return (
            generated.samples.reset_index(drop=True),
            y_syn.reset_index(drop=True),
            {
                "generation_time": float(generated.generation_time or (time.time() - started_at)),
                "generation_cost": 0.0,
                "fairness_score": generated.fairness_score,
            },
        )

    def _run_tabddpm(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_samples: int,
        seed: int,
        dataset_name: str,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        from src.qualsynth.baselines.tabddpm_baseline import TabDDPMBaseline

        numerical_features = list(X_train.select_dtypes(include=[np.number]).columns)
        categorical_features = [col for col in X_train.columns if col not in numerical_features]
        generator = TabDDPMBaseline(
            repo_dir=params.get("repo_dir"),
            workspace_root=params.get("workspace_root"),
            model_type=params.get("model_type", "mlp"),
            hidden_dims=tuple(params.get("hidden_dims", [256, 256])),
            dropout=float(params.get("dropout", 0.0)),
            steps=int(params.get("steps", 1000)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
            batch_size=int(params.get("batch_size", 256)),
            sample_batch_size=int(params.get("sample_batch_size", 512)),
            num_timesteps=int(params.get("num_timesteps", 1000)),
            gaussian_loss_type=params.get("gaussian_loss_type", "mse"),
            scheduler=params.get("scheduler", "cosine"),
            normalization=params.get("normalization", "quantile"),
            device=params.get("device", "cpu"),
            verbose=bool(params.get("verbose", False)),
            keep_workspace=bool(params.get("keep_workspace", False)),
            random_state=seed,
        )
        generated = generator.fit_resample(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            n_samples=target_samples,
            dataset_name=dataset_name,
        )
        return (
            generated.samples.reset_index(drop=True),
            generated.labels.reset_index(drop=True),
            {
                "generation_time": float(generated.generation_time),
                "generation_cost": 0.0,
                "workspace": generated.workspace,
            },
        )

    def _maybe_validate(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method_name: str,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        if not self.enable_universal_validation or X_generated.empty:
            return X_generated, y_generated, {
                "validation_applied": False,
                "validation_rate": 1.0 if not X_generated.empty else 0.0,
            }

        from src.qualsynth.validation.universal_validator import UniversalValidator

        validator = UniversalValidator(
            max_samples=len(X_generated),
            verbose=self.verbose,
            validation_mode=self.validation_mode,
        )
        validation = validator.validate_and_select(X_generated, y_generated, X_train, y_train, method_name=method_name)
        return validation.X_validated, validation.y_validated, {
            "validation_applied": True,
            "validation_rate": validation.overall_pass_rate,
            "n_after_dedup": validation.n_after_dedup,
            "n_after_quality": validation.n_after_quality,
            "n_after_selection": validation.n_after_selection,
            "duplicate_ratio": validation.duplicate_ratio,
            "quality_pass_rate": validation.quality_pass_rate,
            "selection_rate": validation.selection_rate,
            "validation_metrics": validation.metrics,
        }

    def _save_artifacts(
        self,
        result: ExperimentResult,
        X_generated: pd.DataFrame,
        X_validated: pd.DataFrame,
        dataset_name: str,
        method_name: str,
        seed: int,
        artifact_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        artifact_payload = dict(artifact_payload or {})
        base_name = f"{dataset_name}_{method_name}_seed{seed}"
        artifact_manifest: Dict[str, Dict[str, Any]] = {}

        dataframes = {
            "generated_samples": X_generated,
            "validated_samples": X_validated,
            "parsed_candidates": artifact_payload.get("parsed_candidates"),
            "validation_input_candidates": artifact_payload.get(
                "validation_input_candidates"
            ),
            "accepted_candidates": artifact_payload.get("accepted_candidates"),
            "rejected_candidates": artifact_payload.get("rejected_candidates"),
        }
        for label, dataframe in dataframes.items():
            if not isinstance(dataframe, pd.DataFrame):
                continue
            path = self.log_dir / f"{base_name}_{label}.csv"
            dataframe.to_csv(path, index=False)
            artifact_manifest[label] = self._artifact_descriptor(path, rows=len(dataframe))

        request_records = artifact_payload.get("request_records") or []
        if request_records:
            request_path = self.log_dir / f"{base_name}_llm_requests.jsonl"
            with request_path.open("w", encoding="utf-8") as handle:
                for record in request_records:
                    handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            artifact_manifest["llm_requests"] = self._artifact_descriptor(
                request_path,
                rows=len(request_records),
            )

        result.metadata["artifact_manifest"] = artifact_manifest
        self._save_result_json(result, dataset_name, method_name, seed)

    def _save_failure_sample_csvs(
        self,
        X_generated: Optional[pd.DataFrame],
        X_validated: Optional[pd.DataFrame],
        dataset_name: str,
        method_name: str,
        seed: int,
    ) -> None:
        if X_generated is not None:
            X_generated.to_csv(
                self.log_dir / f"{dataset_name}_{method_name}_seed{seed}_generated_samples.csv",
                index=False,
            )
        if X_validated is not None:
            X_validated.to_csv(
                self.log_dir / f"{dataset_name}_{method_name}_seed{seed}_validated_samples.csv",
                index=False,
            )

    def _save_result_json(self, result: ExperimentResult, dataset_name: str, method_name: str, seed: int) -> None:
        result_dir = self.output_dir / dataset_name / method_name
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / f"seed{seed}.json").open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(asdict(result)), handle, indent=2)

    @staticmethod
    def _normalize_binary_labels(
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        """Normalize the training-fold minority to 1 and majority to 0."""
        counts = y_train.value_counts(dropna=False)
        if len(counts) != 2:
            raise ValueError(f"Expected two training classes; found {len(counts)}")
        minority_label = counts.idxmin()
        majority_label = counts.idxmax()
        mapping = {minority_label: 1, majority_label: 0}

        def normalize(values: pd.Series, split_name: str) -> pd.Series:
            normalized = values.map(mapping)
            if normalized.isna().any():
                unknown = values[normalized.isna()].drop_duplicates().tolist()
                raise ValueError(f"Unknown labels in {split_name}: {unknown}")
            normalized = normalized.astype(int)
            normalized.name = values.name
            return normalized.reset_index(drop=True)

        metadata = {
            "minority_original": _to_jsonable(minority_label),
            "majority_original": _to_jsonable(majority_label),
            "normalized_minority": 1,
            "normalized_majority": 0,
            "training_counts_original": {
                str(_to_jsonable(label)): int(count)
                for label, count in counts.items()
            },
        }
        return (
            normalize(y_train, "train"),
            normalize(y_val, "validation"),
            normalize(y_test, "test"),
            metadata,
        )

    def _build_split_provenance(
        self,
        dataset_name: str,
        seed: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "dataset": dataset_name,
            "seed": int(seed),
            "target_mapping": target_mapping,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "train_sha256": self._frame_pair_sha256(X_train, y_train),
            "validation_sha256": self._frame_pair_sha256(X_val, y_val),
            "test_sha256": self._frame_pair_sha256(X_test, y_test),
            "calibration_scope": "training_fold_only",
        }

    def _build_run_provenance(
        self,
        dataset_name: str,
        method_name: str,
        seed: int,
        method_config: MethodConfig,
        split_provenance: Dict[str, Any],
        classifier_policy: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        effective_config = metadata.get("effective_config") or {
            "method": method_name,
            "hyperparameters": {
                key: value
                for key, value in (method_config.hyperparameters or {}).items()
                if key != "api_key"
            },
            "classifier_imbalance_policy": classifier_policy,
            "validation_mode": metadata.get("validation_mode", self.validation_mode),
        }
        resolved_models = metadata.get("resolved_model_slugs") or []
        resolved_providers = metadata.get("resolved_providers") or []
        endpoint_classes = metadata.get("endpoint_classes") or []
        sampling_source = effective_config
        provenance = {
            "schema_version": "qualsynth_run_v1",
            "dataset": dataset_name,
            "method": method_name,
            "seed": int(seed),
            "requested_model_slug": metadata.get("requested_model_slug"),
            "resolved_model_slug": resolved_models[0] if len(resolved_models) == 1 else resolved_models or None,
            "resolved_provider": (
                resolved_providers[0]
                if len(resolved_providers) == 1
                else resolved_providers or None
            ),
            "endpoint_class": (
                endpoint_classes[0]
                if len(endpoint_classes) == 1
                else endpoint_classes or "not_applicable"
            ),
            "request_date_utc": metadata.get("request_date_utc")
            or datetime.now(timezone.utc).isoformat(),
            "prompt_hashes": metadata.get("prompt_hashes", []),
            "config_sha256": metadata.get("effective_config_sha256")
            or self._canonical_sha256(effective_config),
            "sampling": {
                "temperature": sampling_source.get("temperature"),
                "top_p": sampling_source.get("top_p"),
                "presence_penalty": sampling_source.get("presence_penalty"),
                "frequency_penalty": sampling_source.get("frequency_penalty"),
                "max_output_tokens": sampling_source.get("max_output_tokens"),
            },
            "output_format": metadata.get("output_format", "not_applicable"),
            "reasoning": {
                "effort": metadata.get("reasoning_effort"),
                "tokens": int(metadata.get("reasoning_tokens", 0) or 0),
            },
            "policies": {
                "classifier_imbalance": classifier_policy,
                "selection": metadata.get("selection_policy", "not_applicable"),
                "validation": metadata.get("validation_policy", "not_applicable"),
                "validation_mode": metadata.get("validation_mode", "not_applicable"),
                "validation_std_threshold": metadata.get("adaptive_std_threshold"),
                "percentile_configured": metadata.get(
                    "adaptive_percentile_threshold_configured"
                ),
                "percentile_executed": metadata.get(
                    "adaptive_percentile_threshold_executed", False
                ),
                "clip_reference_population": metadata.get("clip_reference_population"),
                "stall_iterations": metadata.get("stall_iterations"),
            },
            "split": split_provenance,
            "usage": {
                "calls": int(metadata.get("llm_calls", 0) or 0),
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "reasoning_tokens": int(metadata.get("reasoning_tokens", 0) or 0),
                "total_tokens": int(metadata.get("total_tokens", 0) or 0),
                "cost_usd": float(metadata.get("generation_cost", 0.0) or 0.0),
                "generation_time_seconds": float(metadata.get("generation_time", 0.0) or 0.0),
            },
            "stopping": {
                "target_samples": int(metadata.get("target_samples", 0) or 0),
                "target_reached": bool(metadata.get("target_reached", False)),
                "target_shortfall": int(metadata.get("target_shortfall", 0) or 0),
                "iterations": int(metadata.get("iterations", 0) or 0),
                "converged": bool(metadata.get("converged", False)),
                "reason": metadata.get("convergence_reason"),
            },
            "run_context": self.run_context,
        }
        return provenance

    @staticmethod
    def _assert_required_run_metadata(provenance: Dict[str, Any]) -> None:
        required = {
            "schema_version",
            "dataset",
            "method",
            "seed",
            "requested_model_slug",
            "resolved_model_slug",
            "resolved_provider",
            "endpoint_class",
            "request_date_utc",
            "prompt_hashes",
            "config_sha256",
            "sampling",
            "output_format",
            "reasoning",
            "policies",
            "split",
            "usage",
            "stopping",
            "run_context",
        }
        missing = sorted(required.difference(provenance))
        if missing:
            raise ValueError(f"Required run provenance keys are missing: {missing}")
        if not provenance.get("config_sha256"):
            raise ValueError("Required run provenance config_sha256 is empty")
        if provenance.get("usage", {}).get("calls", 0) > 0:
            llm_required = ("requested_model_slug", "resolved_model_slug", "prompt_hashes")
            empty = [key for key in llm_required if not provenance.get(key)]
            if empty:
                raise ValueError(f"LLM run provenance fields are empty: {empty}")

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        payload = json.dumps(
            _to_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _frame_pair_sha256(cls, X: pd.DataFrame, y: pd.Series) -> str:
        frame = X.copy()
        frame["__target__"] = y.to_numpy()
        descriptor = {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "row_hashes": pd.util.hash_pandas_object(frame, index=True).astype(str).tolist(),
        }
        return cls._canonical_sha256(descriptor)

    @staticmethod
    def _classify_endpoint(api_base: Optional[str]) -> str:
        normalized = (api_base or "").lower()
        if "openrouter" in normalized:
            return "openrouter"
        if "localhost:11434" in normalized or "ollama" in normalized:
            return "local_ollama"
        if "localhost" in normalized or "127.0.0.1" in normalized:
            return "local_openai_compatible"
        if "openai.com" in normalized:
            return "openai"
        return "openai_compatible_remote" if normalized else "not_recorded"

    @staticmethod
    def _redact_endpoint(api_base: Optional[str]) -> Optional[str]:
        if not api_base:
            return None
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(str(api_base))
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))

    def _artifact_descriptor(self, path: Path, rows: int) -> Dict[str, Any]:
        return {
            "path": str(path),
            "rows": int(rows),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @staticmethod
    def _empty_generated(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        return pd.DataFrame(columns=X_train.columns), pd.Series(dtype=y_train.dtype, name=y_train.name)

    @staticmethod
    def _synthetic_tail(
        X_resampled: pd.DataFrame,
        y_resampled: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        started_at: float,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        return (
            X_resampled.iloc[len(X_train):].reset_index(drop=True),
            y_resampled.iloc[len(y_train):].reset_index(drop=True),
            {"generation_time": time.time() - started_at, "generation_cost": 0.0},
        )

    @staticmethod
    def _target_synthetic_count(y_train: pd.Series) -> int:
        counts = y_train.value_counts()
        if len(counts) < 2:
            return 0
        return int(counts.max() - counts.min())

    @staticmethod
    def _average_metrics(classifier_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        excluded = ("avg_",)
        values: Dict[str, list[float]] = {}
        for metrics in classifier_metrics.values():
            for key, value in metrics.items():
                if key.startswith(excluded):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and not math.isnan(float(value)):
                    values.setdefault(key, []).append(float(value))
        return {key: float(np.mean(items)) for key, items in values.items() if items}

    @staticmethod
    def _average_fairness(classifier_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        fairness_keys = (
            "avg_demographic_parity_difference",
            "avg_equalized_odds_difference",
            "max_fairness_violation",
        )
        values: Dict[str, list[float]] = {key: [] for key in fairness_keys}
        for metrics in classifier_metrics.values():
            for key in fairness_keys:
                value = metrics.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)) and not math.isnan(float(value)):
                    values[key].append(float(value))
        return {key: float(np.mean(items)) for key, items in values.items() if items}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    try:
        if not isinstance(value, (str, bytes)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
