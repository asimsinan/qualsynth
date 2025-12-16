"""
Iterative Refinement Workflow for Qualsynth

This module orchestrates the complete iterative refinement process:
1. Dataset Profiler → analyze characteristics
2. Fairness Auditor → detect violations, set targets
3. Schema Profiler → extract constraints
4. Diversity Planner → identify sparse regions
5. ITERATION LOOP:
   a. Prompt Builder → create fairness-aware prompts
   b. Counterfactual Generator → generate samples
   c. Validator → validate samples
   d. Multi-Objective Optimizer → select best samples
   e. Fairness Re-Auditor → evaluate and provide feedback
   f. Adjust strategy and repeat until convergence
"""

import pandas as pd
import numpy as np
import os
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings
from pathlib import Path

# Import all Qualsynth components
try:
    from ..modules.dataset_profiler import DatasetProfiler
    from ..modules.fairness_auditor import FairnessAuditor
    from ..modules.schema_profiler import SchemaProfiler
    from ..modules.diversity_planner import DiversityPlanner
    from ..modules.validator import Validator
    from ..data.splitting import encode_features
    from ..modules.optimizer import MultiObjectiveOptimizer
    from ..modules.fairness_reauditor import FairnessReAuditor
    from ..generators.counterfactual_generator import CounterfactualGenerator
    from ..utils.sota_duplicate_prevention import SOTADuplicatePrevention
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.modules.dataset_profiler import DatasetProfiler
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    from src.qualsynth.modules.schema_profiler import SchemaProfiler
    from src.qualsynth.modules.diversity_planner import DiversityPlanner
    from src.qualsynth.modules.validator import Validator
    from src.qualsynth.modules.optimizer import MultiObjectiveOptimizer
    from src.qualsynth.modules.fairness_reauditor import FairnessReAuditor
    from src.qualsynth.generators.counterfactual_generator import CounterfactualGenerator
    from src.qualsynth.utils.sota_duplicate_prevention import SOTADuplicatePrevention


@dataclass
class WorkflowConfig:
    """Configuration for iterative refinement workflow."""
    # LLM configuration
    model_name: str = "gemma3-m4-fast"
    temperature: float = 0.7
    top_p: float = 0.95
    presence_penalty: float = 0.6
    frequency_penalty: float = 0.6
    
    # Generation parameters
    target_samples: int = 100
    batch_size: int = 20
    max_iterations: int = 0  # 0 = no limit, loop until target reached
    min_iterations: int = 3
    stall_iterations: int = 10
    
    # Fairness parameters
    fairness_threshold: float = 0.05
    fairness_weight: float = 0.6
    counterfactual_ratio: float = 0.7
    improvement_threshold: float = 0.005
    
    # Validation parameters
    duplicate_threshold: float = 0.90
    quality_threshold: float = 0.5
    
    # Optimization parameters
    diversity_weight: float = 0.2
    performance_weight: float = 0.2
    
    # Duplicate prevention
    enable_sota_dedup: bool = True
    sota_diversity_threshold: float = 0.15
    sota_max_memory_size: int = 200
    sota_memory_strategy: str = 'sliding_window'  # Options: sliding_window, diversity_preserving, cluster_based
    
    # Adaptive validation
    enable_adaptive_validation: bool = False
    adaptive_std_threshold: float = 4.5
    enable_diversity_first_selection: bool = False
    diversity_first_ratio: float = 0.5
    enable_diversity_prompting: bool = True
    diversity_prompt_strength: str = "high"  # Options: low, medium, high
    
    # Statistical validation control
    enable_statistical_validation: bool = True
    
    # Anchor selection strategy
    anchor_selection_strategy: str = "stratified"  # Options: stratified, typical, kmeans_diverse, random
    
    # Semantic dedup (disabled by default for distribution matching)
    enable_semantic_dedup: bool = False


@dataclass
class WorkflowResult:
    """Result of iterative refinement workflow."""
    # Status
    success: bool = False  # Overall success flag
    error: Optional[str] = None  # Error message if failed
    
    # Generated samples
    X_generated: pd.DataFrame = field(default_factory=pd.DataFrame)
    y_generated: pd.Series = field(default_factory=pd.Series)
    
    # Metadata
    total_iterations: int = 0
    total_generated: int = 0
    total_validated: int = 0
    final_selected: int = 0
    
    # Fairness metrics
    initial_fairness: Dict[str, float] = field(default_factory=dict)
    final_fairness: Dict[str, float] = field(default_factory=dict)
    fairness_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    avg_quality_score: float = 0.0
    avg_diversity_score: float = 0.0
    avg_fairness_score: float = 0.0
    
    # Convergence
    converged: bool = False
    convergence_reason: str = ""
    
    # Iteration history
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cost tracking
    total_cost: float = 0.0
    total_tokens: int = 0


class IterativeRefinementWorkflow:
    """
    Orchestrates the complete iterative refinement workflow.
    
    This is the main orchestration layer that:
    1. Coordinates all 8 deterministic tools
    2. Manages the iterative generation loop
    3. Applies feedback-driven improvements
    4. Handles small dataset specialization
    5. Tracks metrics and convergence
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None, method_name: str = "qualsynth", seed: Optional[int] = None, preprocessor=None, output_dir: Optional[str] = None):
        """
        Initialize workflow with configuration.
        
        Args:
            config: Workflow configuration (uses defaults if None)
            method_name: Name of the method (default: "qualsynth")
            seed: Random seed for reproducibility and CSV naming
            preprocessor: DatasetPreprocessor for encoding RAW LLM output to training format
            output_dir: Directory for saving CSV outputs (default: results/experiments/logs)
        """
        self.config = config or WorkflowConfig()
        self.method_name = method_name
        self.seed = seed
        self.preprocessor = preprocessor  # For encoding RAW LLM output
        self.output_dir = Path(output_dir) if output_dir else None  # Custom output directory
        
        # Initialize components
        self.dataset_profiler = DatasetProfiler()
        self.fairness_auditor = FairnessAuditor(
            fairness_threshold=self.config.fairness_threshold
        )
        self.schema_profiler = SchemaProfiler()
        self.diversity_planner = DiversityPlanner()
        
        # Initialize validator (adaptive or standard)
        if self.config.enable_adaptive_validation:
            from src.qualsynth.validation.adaptive_validator import AdaptiveValidator
            self.validator = None  # Will use adaptive validator instead
            enable_stat_val = getattr(self.config, 'enable_statistical_validation', True)
            self.adaptive_validator = AdaptiveValidator(
                duplicate_threshold=self.config.duplicate_threshold,
                quality_threshold=self.config.quality_threshold,
                adaptive_std_threshold=self.config.adaptive_std_threshold,
                adaptive_percentile_threshold=getattr(self.config, 'adaptive_percentile_threshold', 0.99),
                diversity_weight=self.config.diversity_weight,
                fairness_weight=self.config.fairness_weight,
                performance_weight=self.config.performance_weight,
                enable_diversity_first_selection=self.config.enable_diversity_first_selection,
                diversity_first_ratio=self.config.diversity_first_ratio,
                verbose=True,
                enable_statistical_validation=enable_stat_val
            )
            stat_status = "ENABLED" if enable_stat_val else "DISABLED"
            print(f"🔬 Adaptive Validator: ENABLED ({self.config.adaptive_std_threshold}σ, {getattr(self.config, 'adaptive_percentile_threshold', 0.99)*100:.1f}th percentile)")
            print(f"   📊 Statistical Validation: {stat_status}")
        else:
            self.validator = Validator()
            self.adaptive_validator = None
            print("📋 Standard Validator: ENABLED (3σ, quality-first)")
        
        self.optimizer = MultiObjectiveOptimizer(
            fairness_weight=self.config.fairness_weight,
            diversity_weight=self.config.diversity_weight,
            quality_weight=self.config.performance_weight  # performance_weight used as quality_weight
        )
        self.reauditor = FairnessReAuditor(
            fairness_threshold=self.config.fairness_threshold,
            max_iterations=self.config.max_iterations,
            improvement_threshold=self.config.improvement_threshold,
            min_iterations=self.config.min_iterations
        )
        
        # Initialize duplicate prevention system
        if self.config.enable_sota_dedup:
            enable_semantic = getattr(self.config, 'enable_semantic_dedup', False)
            
            self.sota_dedup = SOTADuplicatePrevention(
                semantic_similarity_threshold=0.90,
                feature_diversity_threshold=self.config.sota_diversity_threshold,
                enable_semantic_dedup=enable_semantic,
                enable_hash_dedup=True,
                enable_feature_diversity=False,
                verbose=True,
                max_memory_size=self.config.sota_max_memory_size,
                memory_strategy=self.config.sota_memory_strategy
            )
            print("🔬 Duplicate Prevention: ENABLED")
            print(f"   - Hash dedup: True, Semantic dedup: {enable_semantic}")
            print(f"   - Memory: {self.config.sota_memory_strategy} (max={self.config.sota_max_memory_size})")
        else:
            self.sota_dedup = None
            print("⚠️  Duplicate Prevention: DISABLED")
        
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        dataset_name: str = "unknown"
    ) -> WorkflowResult:
        """
        Run the complete iterative refinement workflow.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_features: Sensitive attributes
            dataset_name: Name of dataset
        
        Returns:
            WorkflowResult with generated samples and metrics
        """
        # Initialize state
        all_generated_samples = []
        all_generated_labels = []
        total_generated = 0
        total_validated = 0
        result = WorkflowResult()
        start_iteration = 1
        
        print("="*80)
        print("Qualsynth ITERATIVE REFINEMENT WORKFLOW")
        print("="*80)
        print()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # RESUME SUPPORT: Load existing samples from CSV if available
        # This allows multi-day runs on rate-limited APIs like OpenRouter
        # ═══════════════════════════════════════════════════════════════════════════
        if self.output_dir:
            csv_output_dir = Path(self.output_dir) / 'logs'
            base_name = f"{dataset_name}_qualsynth_seed{self.seed}" if self.seed else f"{dataset_name}_qualsynth"
            existing_csv = csv_output_dir / f"{base_name}_validated_samples.csv"
            
            if existing_csv.exists():
                try:
                    existing_df = pd.read_csv(existing_csv)
                    if len(existing_df) > 0:
                        # Remove target column if present
                        if 'target' in existing_df.columns:
                            existing_labels = existing_df['target']
                            existing_df = existing_df.drop(columns=['target'])
                        else:
                            existing_labels = pd.Series([1] * len(existing_df))
                        
                        # Add to accumulated samples
                        # NOTE: Data is RAW from CSV - will be encoded in experiment_runner
                        all_generated_samples.append(existing_df)
                        all_generated_labels.append(existing_labels)
                        total_validated = len(existing_df)
                        
                        # Calculate remaining samples needed
                        remaining_samples = self.config.target_samples - total_validated
                        
                        # Start from iteration 1 - we just need to generate remaining samples
                        # The iteration count is for NEW iterations, not total historical
                        start_iteration = 1
                        
                        print("🔄 RESUME MODE: Loading existing samples")
                        print(f"   📁 Found: {existing_csv}")
                        print(f"   📊 Existing samples: {total_validated}")
                        print(f"   🎯 Target: {self.config.target_samples}")
                        print(f"   📈 Progress: {total_validated}/{self.config.target_samples} ({100*total_validated/self.config.target_samples:.1f}%)")
                        print(f"   📉 Remaining: {remaining_samples} samples to generate")
                        print(f"   ▶️  Starting generation loop (up to {self.config.max_iterations} iterations)")
                        print()
                        
                        # Check if already complete
                        if total_validated >= self.config.target_samples:
                            print("✅ Target already reached! No more iterations needed.")
                            result.samples = existing_df
                            result.labels = existing_labels
                            result.validation_rate = 1.0
                            result.total_iterations = 0
                            return result
                except Exception as e:
                    print(f"⚠️  Could not load existing samples: {e}")
                    print("   Starting fresh...")
                    all_generated_samples = []
                    all_generated_labels = []
                    total_validated = 0
                    start_iteration = 1
        print()
        
        # STEP 1: Dataset Profiler
        print("STEP 1: Dataset Profiler")
        print("-"*80)
        dataset_profile = self.dataset_profiler.profile(
            X_train, y_train, sensitive_features, dataset_name
        )
        print(f"Strategy: {dataset_profile.recommended_strategy}")
        print(f"Imbalance: {dataset_profile.imbalance_ratio:.2f}:1")
        
        if self.sota_dedup is not None:
            print(f"   🔧 Dedup: Hash={self.sota_dedup.enable_hash_dedup}, Semantic={self.sota_dedup.enable_semantic_dedup}")
        
        print()
        
        # STEP 2: Fairness Auditor (Initial)
        print("STEP 2: Fairness Auditor (Initial)")
        print("-"*80)
        initial_audit = self.fairness_auditor.audit(
            X_train, y_train, sensitive_features, dataset_name
        )
        print(f"Violations: {len(initial_audit.violations)}")
        for violation in initial_audit.violations:
            print(f"  • {violation.attribute} ({violation.metric.upper()}): "
                  f"{violation.value:.4f} ({violation.severity})")
            result.initial_fairness[f"{violation.attribute}_{violation.metric}"] = violation.value
        print()
        
        # STEP 3: Schema Profiler
        print("STEP 3: Schema Profiler")
        print("-"*80)
        schema_profile = self.schema_profiler.profile(
            X_train, y_train, sensitive_features, initial_audit, dataset_name
        )
        print(f"Features profiled: {len(schema_profile.features)}")
        print(f"Fairness constraints: {len(schema_profile.fairness_constraints)}")
        print()
        
        # STEP 4: Diversity Planner
        print("STEP 4: Diversity Planner")
        print("-"*80)
        diversity_plan = self.diversity_planner.plan(
            X_train, y_train, sensitive_features, initial_audit, dataset_name
        )
        print(f"Sparse regions identified: {len(diversity_plan.global_sparse_regions)}")
        print(f"Fairness-aware regions: {sum(len(regions) for regions in diversity_plan.group_sparse_regions.values())}")
        print()
        
        # STEP 4b: Data Mode Info
        print("STEP 4b: Data Mode")
        print("-"*80)
        print(f"   📊 Data format: RAW (human-readable values)")
        print(f"   📈 Features: {len(X_train.columns)}")
        print(f"   🔄 Encoding: Applied at training time via preprocessor")
        
        # Fit dedup on training data for proper distance normalization
        if self.sota_dedup is not None:
            try:
                from ..modules.schema_profiler import FeatureType
            except ImportError:
                from src.qualsynth.modules.schema_profiler import FeatureType
            
            categorical_features = [
                name for name, feature in schema_profile.features.items()
                if feature.type in [FeatureType.CATEGORICAL_NOMINAL, FeatureType.CATEGORICAL_ORDINAL, FeatureType.BINARY]
            ]
            self.sota_dedup.fit(X_train, categorical_features)
            print(f"   🔬 Dedup fitted on training data")
        print()
        
        # STEP 5: Iterative Generation Loop
        print("="*80)
        print("ITERATIVE GENERATION LOOP")
        print("="*80)
        print()
        
        # Initialize accumulation lists for all samples (only if not resuming)
        if not all_generated_samples:
            all_generated_samples = []  # Selected samples (after optimization)
            all_generated_labels = []
        all_validated_samples = list(all_generated_samples)  # Copy existing for validated tracking
        all_validated_labels = list(all_generated_labels)
        
        # Initialize variables before loop (in case loop doesn't execute)
        iteration = start_iteration - 1 if start_iteration > 1 else 0
        reaudit_result = None  # Will be set in the loop
        stall_counter = 0  # Track iterations without progress
        last_sample_count = sum(len(df) for df in all_generated_samples) if all_generated_samples else 0
        
        # TARGET-BASED LOOP: Run until target reached (no iteration limit)
        while True:
            iteration += 1
            
            # Safety limit check (only if max_iterations > 0)
            if self.config.max_iterations > 0 and iteration > self.config.max_iterations:
                print(f"\n{'='*80}")
                print(f"⚠️  SAFETY LIMIT: Max iterations ({self.config.max_iterations}) reached")
                print(f"{'='*80}\n")
                result.convergence_reason = f"Safety limit reached ({self.config.max_iterations} iterations)"
                break
            
            # Calculate current progress (sum of all samples across all DataFrames)
            current_sample_count = sum(len(df) for df in all_generated_samples) if all_generated_samples else 0
            progress_pct = current_sample_count / self.config.target_samples * 100 if self.config.target_samples > 0 else 0
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration} | Progress: {current_sample_count}/{self.config.target_samples} ({progress_pct:.1f}%)")
            print(f"{'='*80}\n")
            
            # Calculate and display current imbalance ratio
            minority_count = sum(y_train == 1)
            majority_count = sum(y_train == 0)
            current_validated = sum(len(df) for df in all_generated_samples) if all_generated_samples else 0
            current_minority = minority_count + current_validated
            current_ratio = majority_count / current_minority if current_minority > 0 else float('inf')
            target_ratio = 1.0
            
            print(f"   📊 Current Class Balance:")
            print(f"      Majority: {majority_count} | Minority: {minority_count} + {current_validated} generated = {current_minority}")
            print(f"      Current Ratio: {current_ratio:.3f}:1 | Target: {target_ratio:.1f}:1")
            print(f"      Gap to 1:1: {max(0, majority_count - current_minority)} samples")
            print()
            
            iteration_data = {
                'iteration': iteration,
                'batch_size': self.config.batch_size,
                'generated': 0,
                'validated': 0,
                'selected': 0
            }
            
            # Calculate adaptive batch size based on context window and feature count
            # This ensures we use optimal batch size for the model and dataset
            # Calculate remaining samples needed
            if 'all_generated_samples' in locals() and len(all_generated_samples) > 0:
                try:
                    X_all_gen_temp = pd.concat(all_generated_samples, axis=0, ignore_index=True)
                    n_samples_needed = max(0, self.config.target_samples - len(X_all_gen_temp))
                except Exception:
                    n_samples_needed = self.config.target_samples
            else:
                n_samples_needed = self.config.target_samples
            
            # If we've reached target, we can stop (but still check convergence)
            if n_samples_needed <= 0:
                print(f"   ✅ Target samples reached: {self.config.target_samples}")
                # Continue to check convergence, but don't generate more
            
            effective_batch_size = self._calculate_adaptive_batch_size(
                X_train=X_train,
                n_samples_needed=max(n_samples_needed, 10),  # At least 10 to ensure generation continues
                configured_batch_size=self.config.batch_size,
                n_few_shot_examples=getattr(self.config, 'n_few_shot_examples', 5)
            )
            
            # Update batch size for this iteration
            original_batch_size = self.config.batch_size
            self.config.batch_size = effective_batch_size
            
            # Log target progress
            if 'all_generated_samples' in locals() and len(all_generated_samples) > 0:
                try:
                    X_all_gen_temp = pd.concat(all_generated_samples, axis=0, ignore_index=True)
                    current_count = len(X_all_gen_temp)
                    progress_pct = (current_count / self.config.target_samples * 100) if self.config.target_samples > 0 else 0
                    print(f"   📊 Progress: {current_count}/{self.config.target_samples} samples ({progress_pct:.1f}%)")
                except Exception:
                    pass
            
            if effective_batch_size != original_batch_size:
                print(f"   📊 Adaptive batch size: {original_batch_size} → {effective_batch_size}")
            sys.stdout.flush()  # Force flush after batch size calculation
            
            # 5a. Counterfactual Generator (builds its own prompt)
            print(f"Step {iteration}A: Generate Samples")
            print("-"*80)
            print(f"   🎯 Target: {self.config.batch_size} samples")
            print(f"   🤖 Model: {self.config.model_name}")
            print(f"   🌡️  Temperature: {self.config.temperature}")
            sys.stdout.flush()
            
            generator = CounterfactualGenerator(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                batch_size=self.config.batch_size,
                top_p=self.config.top_p,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                anchor_selection_strategy=self.config.anchor_selection_strategy  # Pass anchor strategy
            )
            
            # Pass preprocessor for encoding (if available)
            if hasattr(self, 'preprocessor') and self.preprocessor:
                generator.set_preprocessor(self.preprocessor)
            
            try:
                result = generator.generate(
                    dataset_name=dataset_name,
                    X_train=X_train,
                    y_train=y_train,
                    n_samples=self.config.batch_size,
                    fairness_report=initial_audit,
                    schema_report=schema_profile,
                    diversity_plan=diversity_plan,
                    iteration=iteration  # Pass iteration number
                )
                
                if result.samples.empty:
                    print("   ⚠️  No samples generated in this iteration")
                    if result.errors:
                        print(f"   Errors: {result.errors}")
                    print("   ⚠️  Continuing to next iteration...")
                    # Don't break - continue to next iteration to retry
                    continue
                
                X_gen_batch = result.samples
                y_gen_batch = pd.Series([1] * len(X_gen_batch))  # Assume minority class
                
                # Log generation result
                print(f"   ✅ Generated: {len(X_gen_batch)} RAW samples")
                if len(X_gen_batch) > 0:
                    print(f"   📊 Columns: {len(X_gen_batch.columns)}")
                sys.stdout.flush()
                
                # Post-processing: Clip numerical values to training data range
                if len(X_gen_batch) > 0:
                    n_clipped_total = 0
                    for col in X_gen_batch.columns:
                        if col in X_train.columns:
                            # Only clip numerical columns
                            if pd.api.types.is_numeric_dtype(X_train[col]):
                                col_min = X_train[col].min()
                                col_max = X_train[col].max()
                                
                                # Convert to numeric and clip
                                X_gen_batch[col] = pd.to_numeric(X_gen_batch[col], errors='coerce')
                                
                                # Count values that need clipping
                                below_min = (X_gen_batch[col] < col_min).sum()
                                above_max = (X_gen_batch[col] > col_max).sum()
                                n_clipped = below_min + above_max
                                
                                if n_clipped > 0:
                                    X_gen_batch[col] = X_gen_batch[col].clip(lower=col_min, upper=col_max)
                                    n_clipped_total += n_clipped
                    
                    if n_clipped_total > 0:
                        print(f"   ✂️  Clipped {n_clipped_total} out-of-range values to training bounds")
                
                if hasattr(result, 'llm_calls'):
                    print(f"   📞 LLM calls: {result.llm_calls}")
                if hasattr(result, 'total_tokens') and result.total_tokens > 0:
                    print(f"   🎫 Tokens: {result.total_tokens}")
                iteration_data['generated'] = len(X_gen_batch)
                total_generated += len(X_gen_batch)
                print()
                
            except Exception as e:
                print(f"❌ Generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                break
            
            # 5c. Encode generated samples back to training format
            if hasattr(self, 'preprocessor') and self.preprocessor:
                X_gen_encoded = encode_features(X_gen_batch, self.preprocessor)
            else:
                X_gen_encoded = X_gen_batch
            
            # 5c.1. Fill NaN values with reasonable defaults before validation
            # This improves validation success rate by ensuring all features have values
            if X_gen_encoded is not None and len(X_gen_encoded) > 0:
                X_gen_encoded = X_gen_encoded.copy()
                
                # Get encoded X_train for fill values (X_train is RAW, need encoded version)
                if hasattr(self, 'preprocessor') and self.preprocessor:
                    X_train_for_fill = encode_features(X_train, self.preprocessor)
                else:
                    X_train_for_fill = X_train
                
                for col in X_gen_encoded.columns:
                    if X_gen_encoded[col].isna().any():
                        n_missing = X_gen_encoded[col].isna().sum()
                        if n_missing > 0:
                            # Fill with median for numerical, mode for categorical
                            if pd.api.types.is_numeric_dtype(X_gen_encoded[col]):
                                fill_value = X_train_for_fill[col].median() if col in X_train_for_fill.columns else 0.0
                            else:
                                fill_value = X_train_for_fill[col].mode()[0] if col in X_train_for_fill.columns and len(X_train_for_fill[col].mode()) > 0 else 0
                            X_gen_encoded[col].fillna(fill_value, inplace=True)
                            if n_missing == len(X_gen_encoded):
                                print(f"   ⚠️  Filled {n_missing} NaN values in '{col}' with {fill_value}")
            
            # 5d. Validator (Adaptive or Standard)
            print(f"Step {iteration}C: Validate Samples")
            print("-"*80)
            print(f"   🔍 Validating {len(X_gen_encoded)} samples...")
            
            if self.config.enable_adaptive_validation and self.adaptive_validator is not None:
                # Use adaptive validator (diversity-preserving)
                print("   🔬 Using Adaptive Validator (4.5σ, diversity-first)")
                
                # CRITICAL: Combine X_train with accumulated samples to prevent cross-iteration duplicates
                X_train_encoded = X_train if not hasattr(self, 'preprocessor') else encode_features(X_train, self.preprocessor)
                if all_generated_samples:
                    X_accumulated = pd.concat(all_generated_samples, axis=0, ignore_index=True)
                    y_accumulated = pd.concat(all_generated_labels, axis=0, ignore_index=True)
                    X_reference = pd.concat([X_train_encoded, X_accumulated], axis=0, ignore_index=True)
                    y_reference = pd.concat([y_train.reset_index(drop=True), y_accumulated], axis=0, ignore_index=True)
                else:
                    X_reference = X_train_encoded
                    y_reference = y_train
                
                # Adaptive validator handles everything: dedup, validation, selection
                adaptive_result = self.adaptive_validator.validate_and_select(
                    X_generated=X_gen_encoded,
                    y_generated=pd.Series([1] * len(X_gen_encoded)),
                    X_train=X_reference,  # Now includes accumulated samples!
                    y_train=y_reference,
                    sensitive_features=sensitive_features,
                    method_name="Qualsynth"
                )
                
                X_valid = adaptive_result.X_validated
                y_valid = adaptive_result.y_validated
                
                print(f"   ✅ Adaptive validation complete:")
                print(f"      Original: {adaptive_result.n_original}")
                print(f"      After dedup: {adaptive_result.n_after_dedup} ({adaptive_result.duplicate_ratio*100:.1f}% duplicates)")
                print(f"      After quality: {adaptive_result.n_after_quality} ({adaptive_result.quality_pass_rate*100:.1f}% pass)")
                print(f"      Final selected: {adaptive_result.n_after_selection} ({adaptive_result.overall_pass_rate*100:.1f}% overall)")
                
                if len(X_valid) == 0:
                    print("⚠️  No valid samples after adaptive validation")
                    print("   ⚠️  Continuing to next iteration...")
                    continue
                
            else:
                # Use standard validator (original behavior)
                print("   📋 Using Standard Validator (3σ, quality-first)")
                
                # CRITICAL: Combine X_train with accumulated samples to prevent cross-iteration duplicates
                X_train_encoded = X_train if not hasattr(self, 'preprocessor') else encode_features(X_train, self.preprocessor)
                if all_generated_samples:
                    X_accumulated = pd.concat(all_generated_samples, axis=0, ignore_index=True)
                    X_reference = pd.concat([X_train_encoded, X_accumulated], axis=0, ignore_index=True)
                else:
                    X_reference = X_train_encoded
                
                validation_report = self.validator.validate(
                    samples=X_gen_encoded,
                    schema=schema_profile,
                    existing_data=X_reference,  # Now includes accumulated samples!
                    fairness_constraints=initial_audit.fairness_targets if hasattr(initial_audit, 'fairness_targets') else None
                )
                
                # ValidationReport.valid_samples is an int count, not a list
                # Get actual valid samples from X_gen_encoded (validation just filters)
                valid_indices = [
                    i for i, res in enumerate(validation_report.results) if res.is_valid
                ]
                valid_sample_list = X_gen_encoded.iloc[valid_indices].to_dict('records') if valid_indices else []
                
                if not valid_sample_list or len(valid_sample_list) == 0:
                    print("⚠️  No valid samples in this iteration")
                    print(f"Total validated: {validation_report.valid_samples}/{validation_report.total_samples}")
                    if len(validation_report.results) > 0:
                        print(f"Validation errors (first 5):")
                        for i, res in enumerate(validation_report.results[:5]):
                            if not res.is_valid:
                                print(f"  Sample {i}: {res.errors}")
                    print("   ⚠️  Continuing to next iteration...")
                    # Don't break - continue to next iteration to retry
                    continue
                
                X_valid = pd.DataFrame(valid_sample_list)
                y_valid = pd.Series([1] * len(X_valid))
            
            print(f"   ✅ Valid: {len(X_valid)}/{len(X_gen_batch)} ({len(X_valid)/len(X_gen_batch)*100:.1f}%)")
            if len(X_valid) < len(X_gen_batch):
                invalid_count = len(X_gen_batch) - len(X_valid)
                print(f"   ❌ Invalid: {invalid_count} samples rejected")
            iteration_data['validated'] = len(X_valid)
            total_validated += len(X_valid)
            
                # Apply Duplicate Prevention
            print()
            print(f"Step {iteration}C2: Apply Duplicate Prevention")
            print("-"*80)
            
            if self.config.enable_sota_dedup and self.sota_dedup is not None:
                print("   🔬 Using multi-layer duplicate prevention")
                
                # Get categorical features for Gower distance
                try:
                    from ..modules.schema_profiler import FeatureType
                except ImportError:
                    from src.qualsynth.modules.schema_profiler import FeatureType
                
                categorical_features = [
                    name for name, feature in schema_profile.features.items()
                    if feature.type in [FeatureType.CATEGORICAL_NOMINAL, FeatureType.CATEGORICAL_ORDINAL, FeatureType.BINARY]
                ]
                
                valid_samples = X_valid.to_dict('records')
                sota_result = self.sota_dedup.filter_duplicates(
                    samples=valid_samples,
                    categorical_features=categorical_features
                )
                
                # Convert back to DataFrame
                X_valid_filtered = pd.DataFrame(sota_result.final_samples)
                
                # Create filter stats for compatibility
                filter_stats = {
                    'original_count': sota_result.original_count,
                    'filtered_count': sota_result.filtered_count,
                    'exact_duplicates': sota_result.exact_duplicates_removed,
                    'semantic_duplicates': sota_result.semantic_duplicates_removed,
                    'duplicate_rate': sota_result.duplicate_rate,
                    'diversity_score': sota_result.diversity_score,
                    'constant_columns': []
                }
                
                print(f"   ✅ Filtering complete:")
                print(f"      - Original: {sota_result.original_count}")
                print(f"      - Filtered: {sota_result.filtered_count}")
                print(f"      - Exact dupes: {sota_result.exact_duplicates_removed}")
                print(f"      - Semantic dupes: {sota_result.semantic_duplicates_removed}")
                print(f"      - Duplicate rate: {sota_result.duplicate_rate*100:.1f}%")
                print(f"      - Diversity score: {sota_result.diversity_score:.3f}")
            else:
                print("   ⚠️  Duplicate prevention disabled - no filtering applied")
                X_valid_filtered = X_valid
                filter_stats = {
                    'original_count': len(X_valid),
                    'filtered_count': len(X_valid),
                    'exact_duplicates': 0,
                    'semantic_duplicates': 0,
                    'duplicate_rate': 0.0,
                    'diversity_score': 0.0,
                    'constant_columns': []
                }
            
            # Check for constant columns (critical issue)
            if filter_stats.get('constant_columns'):
                constant_cols = filter_stats['constant_columns']
                print(f"   🚨 CRITICAL: {len(constant_cols)} constant columns detected!")
                print(f"   These columns will cause issues: {constant_cols[:5]}")
                print(f"   ⚠️  Skipping this batch - will regenerate with stricter instructions")
                # Don't use these samples - continue to next iteration
                continue
            
            # Update valid samples and labels
            if len(X_valid_filtered) < len(X_valid):
                print(f"   🔍 Diversity filter removed {len(X_valid) - len(X_valid_filtered)} samples")
                X_valid = X_valid_filtered
                y_valid = pd.Series([1] * len(X_valid))
            
            if len(X_valid) == 0:
                print("   ⚠️  No samples passed diversity filter, continuing to next iteration...")
                continue
            
            print(f"   ✅ Diversity Score: {filter_stats.get('diversity_score', 0):.3f}")
            
            # Accumulate validated samples (before selection)
            all_validated_samples.append(X_valid.copy())
            all_validated_labels.append(y_valid.copy())
            
            print()
            
            # 5d. Multi-Objective Optimizer
            print(f"Step {iteration}D: Optimize Selection")
            print("-"*80)
            print(f"   🎯 Selecting best {min(self.config.batch_size, len(X_valid))} from {len(X_valid)} valid samples...")
            print(f"   ⚖️  Weights: Quality={self.config.performance_weight:.2f}, "
                  f"Diversity={self.config.diversity_weight:.2f}, "
                  f"Fairness={self.config.fairness_weight:.2f}")
            optimization_result = self.optimizer.optimize(
                candidates_df=X_valid,
                n_samples=min(self.config.batch_size, len(X_valid)),
                existing_data=X_train,
                fairness_targets=initial_audit.fairness_targets if hasattr(initial_audit, 'fairness_targets') else None,
                diversity_plan=diversity_plan,
                schema=schema_profile
            )
            
            X_selected = optimization_result.selected_df
            y_selected = pd.Series([1] * len(X_selected)) if X_selected is not None and len(X_selected) > 0 else pd.Series([])
            
            print(f"   ✅ Selected: {len(X_selected)} samples")
            if hasattr(optimization_result, 'avg_quality_score'):
                print(f"   📊 Avg quality: {optimization_result.avg_quality_score:.3f}")
            if hasattr(optimization_result, 'avg_diversity_score'):
                print(f"   🎨 Avg diversity: {optimization_result.avg_diversity_score:.3f}")
            if hasattr(optimization_result, 'avg_fairness_score'):
                print(f"   ⚖️  Avg fairness: {optimization_result.avg_fairness_score:.3f}")
            iteration_data['selected'] = len(X_selected)
            
            print()
            
            # Add to accumulated samples
            all_generated_samples.append(X_selected)
            all_generated_labels.append(y_selected)
            
            # 5e. Fairness Re-Auditor
            print(f"Step {iteration}E: Re-Audit Fairness")
            print("-"*80)
            
            # Combine all generated samples so far
            X_all_gen = pd.concat(all_generated_samples, axis=0, ignore_index=True)
            y_all_gen = pd.concat(all_generated_labels, axis=0, ignore_index=True)
            
            print(f"   📊 Total accumulated: {len(X_all_gen)} samples across {iteration} iteration(s)")
            print(f"   🎯 Target: {self.config.target_samples} samples")
            print(f"   ⏳ Evaluating fairness metrics...")
            
            reaudit_result = self.reauditor.reaudit(
                X_original=X_train,
                y_original=y_train,
                sensitive_features_original=sensitive_features,
                X_generated=X_all_gen,
                y_generated=y_all_gen,
                original_fairness_report=initial_audit,
                iteration=iteration,
                total_generated=len(X_all_gen)
            )
            
            print()
            self.reauditor.print_summary(reaudit_result)
            
            # Update iteration data
            iteration_data['fairness_improvement'] = reaudit_result.dpd_improvement
            iteration_data['converged'] = reaudit_result.converged
            iteration_data['convergence_reason'] = reaudit_result.convergence_reason
            if hasattr(result, 'iteration_history'):
                result.iteration_history.append(iteration_data)
            
            # Print iteration summary
            print()
            print(f"{'='*80}")
            print(f"ITERATION {iteration} SUMMARY")
            print(f"{'='*80}")
            print(f"   Generated: {iteration_data['generated']} → Validated: {iteration_data['validated']} → Selected: {iteration_data['selected']}")
            print(f"   Total accumulated: {len(X_all_gen)}/{self.config.target_samples} samples ({len(X_all_gen)/self.config.target_samples*100:.1f}%)")
            
            # Calculate and display ratio improvement
            minority_count_summary = sum(y_train == 1)
            majority_count_summary = sum(y_train == 0)
            current_minority_summary = minority_count_summary + len(X_all_gen)
            current_ratio_summary = majority_count_summary / current_minority_summary if current_minority_summary > 0 else float('inf')
            initial_ratio = majority_count_summary / minority_count_summary if minority_count_summary > 0 else float('inf')
            ratio_improvement = initial_ratio - current_ratio_summary
            gap_remaining = max(0, majority_count_summary - current_minority_summary)
            
            print(f"   Class Balance: {current_ratio_summary:.3f}:1 (started: {initial_ratio:.2f}:1, target: 1.00:1)")
            print(f"   Ratio Improvement: {ratio_improvement:.3f} | Gap to 1:1: {gap_remaining} samples")
            
            if reaudit_result.dpd_improvement:
                avg_improvement = np.mean(list(reaudit_result.dpd_improvement.values()))
                print(f"   Fairness improvement: {avg_improvement:.4f} (avg DPD reduction)")
            if reaudit_result.converged:
                print(f"   ⚠️  Fairness converged: {reaudit_result.convergence_reason} (but continuing to reach target samples)")
            print(f"{'='*80}")
            print()
            
            # Save CSVs after each iteration (without .pkl checkpoint)
            if len(all_generated_samples) > 0 and len(all_validated_samples) > 0:
                # Concatenate lists into DataFrames
                X_all_gen = pd.concat(all_generated_samples, axis=0, ignore_index=True)
                y_all_gen = pd.concat(all_generated_labels, axis=0, ignore_index=True)
                X_all_valid = pd.concat(all_validated_samples, axis=0, ignore_index=True)
                y_all_valid = pd.concat(all_validated_labels, axis=0, ignore_index=True)
                
                self._save_iteration_csv(
                    selected_df=X_all_gen,
                    selected_labels=y_all_gen,
                    validated_df=X_all_valid,
                    validated_labels=y_all_valid,
                    X_train=X_train,
                    y_train=y_train,
                    dataset_name=dataset_name,
                    iteration=iteration
                )
            
            # Check if target reached
            if len(X_all_gen) >= self.config.target_samples:
                result.converged = True
                result.convergence_reason = f"Target samples ({self.config.target_samples}) reached"
                print(f"   ✅ TARGET REACHED: {len(X_all_gen)}/{self.config.target_samples} samples generated")
                break
            
            # Stall detection
            current_sample_count = len(X_all_gen)
            if current_sample_count <= last_sample_count:
                stall_counter += 1
                print(f"   ⚠️  No progress this iteration (stall count: {stall_counter}/{self.config.stall_iterations})")
                if stall_counter >= self.config.stall_iterations:
                    result.convergence_reason = f"Stalled: No progress for {stall_counter} iterations"
                    print(f"   ❌ STALLED: No new samples for {stall_counter} consecutive iterations")
                    break
            else:
                stall_counter = 0  # Reset on progress
            last_sample_count = current_sample_count
        
        # Finalize results
        if all_generated_samples:
            result.X_generated = pd.concat(all_generated_samples, axis=0, ignore_index=True)
            result.y_generated = pd.concat(all_generated_labels, axis=0, ignore_index=True)
            result.total_iterations = iteration
            result.total_generated = total_generated
            result.total_validated = total_validated
            result.final_selected = len(result.X_generated)
            
            # Final fairness metrics
            if reaudit_result:
                result.final_fairness = {
                    **{f"{k}_dpd": v for k, v in reaudit_result.new_dpd.items()},
                    **{f"{k}_eod": v for k, v in reaudit_result.new_eod.items()}
                }
                result.fairness_improvement = {
                    **{f"{k}_dpd": v for k, v in reaudit_result.dpd_improvement.items()},
                    **{f"{k}_eod": v for k, v in reaudit_result.eod_improvement.items()}
                }
        
        # Print final summary
        self._print_final_summary(result, dataset_name)
        
        # Check if we reached the desired number
        if hasattr(result, 'X_generated') and result.X_generated is not None and len(result.X_generated) > 0:
            if len(result.X_generated) >= self.config.target_samples:
                print(f"✅ Target reached: {len(result.X_generated)} >= {self.config.target_samples} samples")
            else:
                shortfall = self.config.target_samples - len(result.X_generated)
                percentage = (len(result.X_generated) / self.config.target_samples * 100) if self.config.target_samples > 0 else 0
                print(f"⚠️  Target NOT reached: {len(result.X_generated)}/{self.config.target_samples} samples ({percentage:.1f}%)")
                print(f"   Shortfall: {shortfall} samples")
                if result.convergence_reason and "Stalled" in result.convergence_reason:
                    print(f"   Reason: Generation stalled (validation rate too low or LLM not generating valid samples)")
                elif result.convergence_reason and "Safety limit" in result.convergence_reason:
                    print(f"   Reason: Safety iteration limit reached")
                else:
                    print(f"   Consider checking validation rate or LLM quality")
            result.success = True
        else:
            result.success = False
            result.error = "No samples generated"
        
        return result

    
    def _save_iteration_csv(
        self,
        selected_df: pd.DataFrame,
        selected_labels: pd.Series,
        validated_df: pd.DataFrame,
        validated_labels: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dataset_name: str,
        iteration: int
    ):
        """
        Save CSV files after each iteration:
        1. generated_samples.csv - selected samples after optimization
        2. validated_samples.csv - validated samples before selection
        3. resampled_dataset.csv - original training data + selected samples
        """
        try:
            from pathlib import Path
            from src.qualsynth.data.splitting import decode_features
            
            print(f"\n   💾 Saving iteration CSVs - {len(selected_df)} selected, {len(validated_df)} validated")
            sys.stdout.flush()
            
            # Decode samples for human-readable CSV
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                selected_df_raw = decode_features(selected_df, self.preprocessor)
                validated_df_raw = decode_features(validated_df, self.preprocessor)
                print(f"   📝 Decoded samples for CSV")
            else:
                selected_df_raw = selected_df
                validated_df_raw = validated_df
            
            # Determine output directory (use custom if set, otherwise default)
            if self.output_dir:
                csv_output_dir = self.output_dir / "logs"
            else:
                csv_output_dir = Path("results/experiments/logs")
            csv_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Base filename: {dataset}_{method}_seed{X}
            base_name = f"{dataset_name}_{self.method_name}_seed{self.seed if self.seed is not None else 'unknown'}"
            
            # Prepare selected samples DataFrame with target column
            generated_df = selected_df_raw.copy()
            if len(selected_labels) == len(generated_df):
                generated_df['target'] = selected_labels.values
            else:
                generated_df['target'] = 1
                print(f"   ⚠️  Selected labels length mismatch, defaulting to target=1")
            
            # 1. Save generated_samples.csv
            generated_csv = csv_output_dir / f"{base_name}_generated_samples.csv"
            generated_df.to_csv(generated_csv, index=False)
            print(f"   ✅ Saved: {generated_csv.name} ({len(generated_df)} selected samples)")
            
            # 2. Save validated_samples.csv
            validated_samples_df = validated_df_raw.copy()
            if len(validated_labels) == len(validated_samples_df):
                validated_samples_df['target'] = validated_labels.values
            else:
                validated_samples_df['target'] = 1
                print(f"   ⚠️  Validated labels length mismatch, defaulting to target=1")
            
            validated_csv = csv_output_dir / f"{base_name}_validated_samples.csv"
            validated_samples_df.to_csv(validated_csv, index=False)
            print(f"   ✅ Saved: {validated_csv.name} ({len(validated_samples_df)} validated samples)")
            
            # 3. Save resampled_dataset.csv
            X_resampled = pd.concat([X_train, selected_df_raw], axis=0, ignore_index=True)
            y_resampled = pd.concat([y_train, selected_labels], axis=0, ignore_index=True)
            
            resampled_df = X_resampled.copy()
            resampled_df['target'] = y_resampled.values
            
            resampled_csv = csv_output_dir / f"{base_name}_resampled_dataset.csv"
            resampled_df.to_csv(resampled_csv, index=False)
            print(f"   ✅ Saved: {resampled_csv.name} ({len(resampled_df)} = {len(X_train)} orig + {len(selected_df_raw)} selected)")
            
            sys.stdout.flush()
                    
        except Exception as e:
            print(f"   ⚠️  Failed to save iteration CSVs: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    def _calculate_adaptive_dedup_threshold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dataset_profile: Any
    ) -> float:
        """
        Calculate adaptive duplicate detection threshold based on dataset characteristics.
        
        Considers: dataset size, intrinsic dimensionality, class imbalance, and feature density.
        
        Returns:
            Adaptive semantic similarity threshold (0.75 - 0.95)
        """
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        
        base_threshold = 0.80
        
        # Factor 1: Dataset size (logarithmic scaling)
        n_samples = len(X_train)
        size_factor = np.log10(max(n_samples, 100)) / 5.0
        size_adjustment = 0.10 * size_factor
        
        # Factor 2: Intrinsic dimensionality
        n_features = X_train.shape[1]
        try:
            X_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X_sample.select_dtypes(include=[np.number]))
            
            if X_normalized.shape[1] > 1:
                pca = PCA(n_components=min(X_normalized.shape[1], 50))
                pca.fit(X_normalized)
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                intrinsic_dim = np.argmax(cumsum_variance >= 0.95) + 1
                dim_ratio = intrinsic_dim / n_features
                dim_adjustment = 0.05 * (1 - dim_ratio)
            else:
                dim_adjustment = 0.0
        except Exception:
            dim_adjustment = 0.0
        
        # Factor 3: Class imbalance
        imbalance_ratio = dataset_profile.imbalance_ratio if hasattr(dataset_profile, 'imbalance_ratio') else 1.0
        imbalance_factor = min(imbalance_ratio / 10.0, 1.0)
        imbalance_adjustment = -0.05 * imbalance_factor
        
        # Factor 4: Feature space density
        try:
            X_minority = X_train[y_train == y_train.value_counts().idxmin()]
            X_minority_sample = X_minority.sample(n=min(200, len(X_minority)), random_state=42)
            scaler = StandardScaler()
            X_minority_norm = scaler.fit_transform(X_minority_sample.select_dtypes(include=[np.number]))
            
            if len(X_minority_norm) > 5:
                nn = NearestNeighbors(n_neighbors=min(6, len(X_minority_norm)))
                nn.fit(X_minority_norm)
                distances, _ = nn.kneighbors(X_minority_norm)
                avg_5nn_distance = np.mean(distances[:, -1])
                density_factor = 1.0 / (1.0 + avg_5nn_distance)
                density_adjustment = 0.05 * density_factor
            else:
                density_adjustment = 0.0
        except Exception:
            density_adjustment = 0.0
        
        adaptive_threshold = base_threshold + size_adjustment + dim_adjustment + imbalance_adjustment + density_adjustment
        adaptive_threshold = np.clip(adaptive_threshold, 0.75, 0.95)
        
        print(f"   📊 Adaptive Threshold: {adaptive_threshold:.3f}")
        print(f"      Size: {size_adjustment:+.3f}, Dim: {dim_adjustment:+.3f}, Imbalance: {imbalance_adjustment:+.3f}, Density: {density_adjustment:+.3f}")
        
        return float(adaptive_threshold)
    
    def _print_final_summary(self, result: WorkflowResult, dataset_name: str):
        """Print final workflow summary."""
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE")
        print("="*80)
        print()
        
        print(f"Dataset: {dataset_name}")
        print(f"Total iterations: {result.total_iterations if hasattr(result, 'total_iterations') else 0}")
        print(f"Total generated: {result.total_generated if hasattr(result, 'total_generated') else 0}")
        print(f"Total validated: {result.total_validated if hasattr(result, 'total_validated') else 0}")
        print(f"Final selected: {result.final_selected if hasattr(result, 'final_selected') else 0}")
        print()
        
        if hasattr(result, 'converged') and result.converged:
            print(f"✅ CONVERGED: {result.convergence_reason}")
        else:
            print("⚠️  Did not converge")
        print()
        
        if hasattr(result, 'fairness_improvement') and result.fairness_improvement:
            print("FAIRNESS IMPROVEMENT:")
            for metric, improvement in result.fairness_improvement.items():
                if improvement != 0:
                    status = "✅" if improvement > 0 else "❌"
                    print(f"  {status} {metric}: {improvement:+.4f}")
        
        print()
        print("="*80)
    
    def _calculate_adaptive_batch_size(
        self,
        X_train: pd.DataFrame,
        n_samples_needed: int,
        configured_batch_size: int,
        n_few_shot_examples: int = 5
    ) -> int:
        """
        Calculate adaptive batch size based on model context window, features, and target samples.
        """
        n_features = len(X_train.columns)
        estimated_tokens_per_sample = (n_features * 18) + 50
        few_shot_tokens = n_few_shot_examples * estimated_tokens_per_sample
        
        model_name = self.config.model_name
        is_ollama = (
            os.getenv('OLLAMA_MODEL') is not None or 
            'ollama' in str(model_name).lower() or 
            'localhost:11434' in str(model_name).lower()
        )
        
        if is_ollama:
            context_window = 256000
            available_tokens = context_window - 3000 - few_shot_tokens
        else:
            context_window = 131000
            available_tokens = context_window - 3000 - few_shot_tokens
        
        max_samples_by_context = int((available_tokens * 0.8) / estimated_tokens_per_sample)
        candidate_batch_size = min(configured_batch_size, max_samples_by_context)
        optimal_batch_size = min(candidate_batch_size, n_samples_needed)
        optimal_batch_size = max(5, optimal_batch_size)
        
        is_large_model = (
            '27b' in str(self.config.model_name).lower() or
            'openrouter' in str(os.getenv('OPENAI_API_BASE', '')).lower() or
            configured_batch_size > 50
        )
        
        if is_large_model:
            max_model_batch = min(configured_batch_size, max_samples_by_context)
        else:
            max_model_batch = 50
        
        effective_batch_size = min(optimal_batch_size, max_model_batch)
        estimated_batches = int(np.ceil(n_samples_needed / effective_batch_size)) if effective_batch_size > 0 else 1
        
        if estimated_batches > 20 and max_samples_by_context > effective_batch_size:
            target_batches = 15
            suggested_batch_size = int(np.ceil(n_samples_needed / target_batches))
            effective_batch_size = min(suggested_batch_size, max_samples_by_context, max_model_batch)
            print(f"   ⚠️  Optimizing to ~{int(np.ceil(n_samples_needed / effective_batch_size))} batches")
            sys.stdout.flush()
        
        print(f"   📊 Dynamic batch size calculation:")
        print(f"      - Features: {n_features}")
        print(f"      - Estimated tokens/sample: ~{estimated_tokens_per_sample}")
        print(f"      - Few-shot examples: {n_few_shot_examples} (~{few_shot_tokens:,} tokens)")
        print(f"      - Context window: {context_window:,} tokens")
        print(f"      - Available for output: ~{available_tokens:,} tokens")
        print(f"      - Max samples by context: ~{max_samples_by_context}")
        print(f"      - Configured batch size: {configured_batch_size}")
        print(f"      - Effective batch size: {effective_batch_size}")
        print(f"      - Estimated batches needed: {estimated_batches}")
        sys.stdout.flush()
        
        return effective_batch_size


if __name__ == "__main__":
    from src.qualsynth.data.splitting import load_split
    
    print("="*80)
    print("Testing Iterative Refinement Workflow")
    print("="*80)
    
    dataset_name = 'german_credit'
    sensitive_cols = ['sex', 'age']
    
    split_data = load_split(dataset_name, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
    sensitive_features = X_train[available_sensitive_cols]
    
    config = WorkflowConfig(
        model_name="gemma3-m4-fast",
        target_samples=30,
        batch_size=15,
        max_iterations=2
    )
    
    workflow = IterativeRefinementWorkflow(config)
    
    print("✅ Workflow initialized successfully")
    print(f"Components: {len([c for c in dir(workflow) if not c.startswith('_')])}")
    print("\nNote: Full workflow test requires LLM API.")

