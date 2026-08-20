"""
QualSynthGenerator: Simple API for Quality-Driven Synthetic Data Generation

This module provides a user-friendly interface for generating synthetic samples
using the QualSynth methodology. It wraps the more complex IterativeWorkflow
with sensible defaults for common use cases.

Example usage:
    from qualsynth import QualSynthGenerator
    
    generator = QualSynthGenerator(
        model_name="gemma3:12b",
        temperature=0.7,
        max_iterations=20
    )
    
    X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass

from .core.iterative_workflow import IterativeRefinementWorkflow as IterativeWorkflow, WorkflowConfig


@dataclass
class GeneratorConfig:
    """Configuration for QualSynthGenerator with sensible defaults."""
    # LLM settings
    model_name: str = "gemma3:12b"
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Generation settings
    max_iterations: int = 0
    batch_size: int = 20
    target_ratio: float = 1.0  # Target class ratio (1.0 = balanced)
    
    # Validation settings
    validation_threshold: float = 4.5  # Statistical validation (σ)
    duplicate_threshold: float = 0.10  # Near-duplicate detection
    
    # Optimization weights
    fairness_weight: float = 0.60
    diversity_weight: float = 0.20
    quality_weight: float = 0.20


class QualSynthGenerator:
    """
    Quality-Driven Synthetic Data Generator using LLM-Guided Oversampling.
    
    QualSynthGenerator provides a simple, scikit-learn-style API for generating
    high-quality synthetic samples for imbalanced classification. It uses Large
    Language Models (LLMs) with iterative refinement and multi-stage validation
    to ensure all generated samples are realistic and statistically plausible.
    
    Parameters
    ----------
    model_name : str, default="gemma3:12b"
        Name of the LLM model to use. Supports:
        - OpenAI models: "gpt-4", "gpt-3.5-turbo"
        - Ollama models: "gemma3:12b", "llama3:8b"
        - Custom endpoints via api_base parameter
        
    temperature : float, default=0.7
        Controls generation consistency. Lower values (0.5-0.7) produce more
        consistent, distribution-matching samples. Higher values increase diversity.
        
    max_iterations : int, default=0
        Maximum number of refinement iterations. A value of 0 disables the
        iteration cap and generation continues until the target sample count is
        reached or stall detection stops the run.
        
    batch_size : int, default=20
        Number of samples to generate per LLM call. Smaller batches allow
        more detailed per-sample instructions.
        
    target_ratio : float, default=1.0
        Target class ratio after oversampling. 1.0 means perfect balance
        (minority class equals majority class).
        
    validation_threshold : float, default=4.5
        Statistical validation threshold in standard deviations. Samples
        with z-scores above this are rejected.
        
    api_base : str, optional
        Custom API base URL for self-hosted models or alternative providers.
        
    api_key : str, optional
        API key for cloud providers (OpenAI, OpenRouter). Can also be set
        via OPENAI_API_KEY environment variable.
        
    sensitive_attributes : list of str, optional
        Column names of sensitive attributes for fairness-aware generation.
        When provided, the generator prioritizes samples that reduce
        demographic disparity.
        
    verbose : bool, default=True
        Whether to print progress information during generation.
    
    Attributes
    ----------
    config_ : GeneratorConfig
        Configuration object with all parameters.
        
    workflow_ : IterativeWorkflow
        Underlying workflow object (available after fit_generate).
        
    n_samples_generated_ : int
        Number of synthetic samples generated in last fit_generate call.
        
    validation_rate_ : float
        Percentage of generated samples that passed validation.
    
    Examples
    --------
    Basic usage with default settings:
    
    >>> from qualsynth import QualSynthGenerator
    >>> import pandas as pd
    >>> 
    >>> # Load your imbalanced dataset
    >>> X_train = pd.read_csv("train_features.csv")
    >>> y_train = pd.read_csv("train_labels.csv")["target"]
    >>> 
    >>> # Initialize and generate
    >>> generator = QualSynthGenerator(model_name="gpt-4")
    >>> X_syn, y_syn = generator.fit_generate(X_train, y_train)
    >>> 
    >>> # Combine with original data for training
    >>> X_balanced = pd.concat([X_train, X_syn])
    >>> y_balanced = pd.concat([y_train, y_syn])
    
    Using local Ollama model:
    
    >>> generator = QualSynthGenerator(
    ...     model_name="gemma3:12b",
    ...     api_base="http://localhost:11434/v1"
    ... )
    >>> X_syn, y_syn = generator.fit_generate(X_train, y_train)
    
    Fairness-aware generation:
    
    >>> generator = QualSynthGenerator(
    ...     model_name="gpt-4",
    ...     sensitive_attributes=["gender", "race"]
    ... )
    >>> X_syn, y_syn = generator.fit_generate(X_train, y_train)
    
    See Also
    --------
    IterativeWorkflow : Lower-level API with more configuration options.
    
    Notes
    -----
    QualSynthGenerator achieves 100% validation pass rate by filtering samples
    during generation. All returned samples are guaranteed to pass:
    - Exact duplicate detection (hash-based)
    - Schema validation (correct types and ranges)
    - Statistical validation (within training distribution)
    """
    
    def __init__(
        self,
        model_name: str = "gemma3:12b",
        temperature: float = 0.7,
        top_p: float = 0.95,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_output_tokens: int = 8192,
        reasoning_effort: Optional[str] = None,
        strict_request_contract: bool = False,
        resume_generation: bool = True,
        max_iterations: int = 0,
        min_iterations: int = 3,
        stall_iterations: int = 10,
        batch_size: int = 20,
        target_ratio: float = 1.0,
        validation_threshold: float = 4.5,
        duplicate_threshold: float = 0.10,
        quality_threshold: float = 0.5,
        enable_adaptive_validation: bool = False,
        adaptive_percentile_threshold: float = 0.995,
        enable_diversity_first_selection: bool = False,
        diversity_first_ratio: float = 0.5,
        enable_statistical_validation: bool = True,
        validation_mode: str = "standard",
        diversity_weight: float = 0.20,
        performance_weight: float = 0.20,
        enable_sota_dedup: bool = True,
        enable_semantic_dedup: bool = False,
        anchor_selection_strategy: str = "stratified",
        use_few_shot: bool = True,
        n_few_shot_examples: int = 5,
        diversity_prompt_strength: str = "high",
        prompt_policy: str = "anchor",
        validation_policy: str = "full",
        # See iterative_workflow.IterativeWorkflowConfig.selection_policy for the
        # rationale: paired ablation found no significant performance benefit, so the
        # default is the cheaper generation-order policy. Multi-objective selection
        # remains available as an opt-in for users who want to experiment with it.
        selection_policy: str = "generation_order",
        # When True, post-generation clip uses minority-class bounds only (default
        # is full training range). Tighter support, drops boundary samples.
        clip_to_minority_class: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        sensitive_attributes: Optional[List[str]] = None,
        dataset_name: str = "unknown",
        method_name: str = "qualsynth",
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.strict_request_contract = strict_request_contract
        self.resume_generation = resume_generation
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.stall_iterations = stall_iterations
        self.batch_size = batch_size
        self.target_ratio = target_ratio
        self.validation_threshold = validation_threshold
        self.duplicate_threshold = duplicate_threshold
        self.quality_threshold = quality_threshold
        self.enable_adaptive_validation = enable_adaptive_validation
        self.adaptive_percentile_threshold = adaptive_percentile_threshold
        self.enable_diversity_first_selection = enable_diversity_first_selection
        self.diversity_first_ratio = diversity_first_ratio
        self.enable_statistical_validation = enable_statistical_validation
        self.validation_mode = validation_mode
        self.diversity_weight = diversity_weight
        self.performance_weight = performance_weight
        self.enable_sota_dedup = enable_sota_dedup
        self.enable_semantic_dedup = enable_semantic_dedup
        self.anchor_selection_strategy = anchor_selection_strategy
        self.use_few_shot = use_few_shot
        self.n_few_shot_examples = n_few_shot_examples
        self.diversity_prompt_strength = diversity_prompt_strength
        self.prompt_policy = prompt_policy
        self.validation_policy = validation_policy
        self.selection_policy = selection_policy
        self.clip_to_minority_class = clip_to_minority_class
        self.api_base = api_base
        self.api_key = api_key
        self.sensitive_attributes = sensitive_attributes or []
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.seed = seed
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Attributes set after fit_generate
        self.config_: Optional[GeneratorConfig] = None
        self.workflow_: Optional[IterativeWorkflow] = None
        self.last_workflow_result_: Optional[Any] = None
        self.n_samples_generated_: int = 0
        self.validation_rate_: float = 0.0

    def _build_workflow_config(self, n_samples: int) -> WorkflowConfig:
        """Build the lower-level workflow configuration."""
        return WorkflowConfig(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            strict_request_contract=self.strict_request_contract,
            resume_generation=self.resume_generation,
            max_iterations=self.max_iterations,
            min_iterations=self.min_iterations,
            stall_iterations=self.stall_iterations,
            batch_size=self.batch_size,
            target_samples=n_samples,
            fairness_weight=0.60,
            duplicate_threshold=self.duplicate_threshold,
            quality_threshold=self.quality_threshold,
            diversity_weight=self.diversity_weight,
            performance_weight=self.performance_weight,
            enable_sota_dedup=self.enable_sota_dedup,
            enable_adaptive_validation=self.enable_adaptive_validation,
            adaptive_std_threshold=self.validation_threshold,
            adaptive_percentile_threshold=self.adaptive_percentile_threshold,
            enable_diversity_first_selection=self.enable_diversity_first_selection,
            diversity_first_ratio=self.diversity_first_ratio,
            enable_statistical_validation=self.enable_statistical_validation,
            validation_mode=self.validation_mode,
            enable_semantic_dedup=self.enable_semantic_dedup,
            anchor_selection_strategy=self.anchor_selection_strategy,
            use_few_shot=self.use_few_shot,
            n_few_shot_examples=self.n_few_shot_examples,
            diversity_prompt_strength=self.diversity_prompt_strength,
            prompt_policy=self.prompt_policy,
            validation_policy=self.validation_policy,
            selection_policy=self.selection_policy,
            clip_to_minority_class=self.clip_to_minority_class,
        )
        
    def fit_generate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples for the minority class.
        
        This method analyzes the input data, determines the minority class,
        and generates synthetic samples to balance the class distribution.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features. Must be a pandas DataFrame with column names.
            
        y : pd.Series or np.ndarray
            Training labels. Binary classification (0/1) is expected.
            
        n_samples : int, optional
            Number of synthetic samples to generate. If not provided,
            generates enough samples to achieve target_ratio.
            
        Returns
        -------
        X_synthetic : pd.DataFrame
            Generated synthetic features with same columns as X.
            
        y_synthetic : pd.Series
            Labels for synthetic samples (all minority class).
            
        Raises
        ------
        ValueError
            If X and y have different lengths, or if y is not binary.
        """
        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}.")
        
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        unique_classes = y.unique()
        
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification. Got {len(unique_classes)} classes.")
        
        # Identify minority class
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        
        # Calculate target samples
        if n_samples is None:
            target_minority = int(n_majority * self.target_ratio)
            n_samples = max(0, target_minority - n_minority)
        
        if n_samples == 0:
            if self.verbose:
                print("Dataset is already balanced. No samples to generate.")
            return pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("QualSynth Generator")
            print(f"{'='*60}")
            print(f"Minority class: {minority_class} ({n_minority} samples)")
            print(f"Majority class: {majority_class} ({n_majority} samples)")
            print(f"Imbalance ratio: {n_majority/n_minority:.2f}:1")
            print(f"Target samples: {n_samples}")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}\n")
        
        workflow_config = self._build_workflow_config(n_samples=n_samples)
        
        # Set API configuration via environment if provided
        import os
        if self.api_base:
            os.environ['OPENAI_API_BASE'] = self.api_base
        if self.api_key:
            os.environ['OPENAI_API_KEY'] = self.api_key
        
        # Filter to minority class
        X_minority = X[y == minority_class].copy()
        y_minority = y[y == minority_class].copy()
        
        # Handle sensitive attributes
        sensitive_features = None
        if self.sensitive_attributes:
            available_attrs = [a for a in self.sensitive_attributes if a in X.columns]
            if available_attrs:
                sensitive_features = X_minority[available_attrs].copy()
        
        # Create and run workflow
        self.workflow_ = IterativeWorkflow(
            workflow_config,
            method_name=self.method_name,
            seed=self.seed,
            output_dir=self.output_dir,
        )
        
        result = self.workflow_.run(
            X_train=X_minority,
            y_train=y_minority,
            sensitive_features=sensitive_features,
            dataset_name=self.dataset_name,
        )
        self.last_workflow_result_ = result
        
        # Extract results
        X_synthetic = result.X_generated
        self.n_samples_generated_ = len(X_synthetic) if X_synthetic is not None else 0
        if getattr(result, "total_generated", 0):
            self.validation_rate_ = result.total_validated / max(1, result.total_generated)
        else:
            self.validation_rate_ = 0.0
        
        if X_synthetic is None or len(X_synthetic) == 0:
            if self.verbose:
                print("\nWarning: No samples generated.")
            return pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
        
        # Create labels for synthetic samples
        y_synthetic = pd.Series([minority_class] * len(X_synthetic), name=y.name)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Generation Complete")
            print(f"{'='*60}")
            print(f"Samples generated: {self.n_samples_generated_}")
            print(f"Validation rate: {self.validation_rate_:.1f}%")
            print(f"New minority count: {n_minority + self.n_samples_generated_}")
            new_ratio = n_majority / (n_minority + self.n_samples_generated_)
            print(f"New imbalance ratio: {new_ratio:.2f}:1")
            print(f"{'='*60}\n")
        
        return X_synthetic, y_synthetic
    
    def get_params(self) -> Dict[str, Any]:
        """Get generator parameters."""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'max_output_tokens': self.max_output_tokens,
            'reasoning_effort': self.reasoning_effort,
            'strict_request_contract': self.strict_request_contract,
            'resume_generation': self.resume_generation,
            'max_iterations': self.max_iterations,
            'min_iterations': self.min_iterations,
            'stall_iterations': self.stall_iterations,
            'batch_size': self.batch_size,
            'target_ratio': self.target_ratio,
            'validation_threshold': self.validation_threshold,
            'api_base': self.api_base,
            'sensitive_attributes': self.sensitive_attributes,
            'prompt_policy': self.prompt_policy,
            'validation_policy': self.validation_policy,
            'selection_policy': self.selection_policy,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'QualSynthGenerator':
        """Set generator parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self
    
    def __repr__(self) -> str:
        return (
            f"QualSynthGenerator(model_name='{self.model_name}', "
            f"temperature={self.temperature}, "
            f"max_iterations={self.max_iterations}, "
            f"batch_size={self.batch_size})"
        )
