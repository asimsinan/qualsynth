"""
Counterfactual Generator for Qualsynth

This component generates synthetic samples using LLMs with fairness-aware prompting.
Uses CSV output format for faster generation and robust parsing with CleverCSV.
"""

import pandas as pd
import numpy as np
import hashlib
import json
import time
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from io import StringIO

# CSV parsing handled by pandas (robust with on_bad_lines='skip')

try:
    from ..prompts.prompt_builder import PromptBuilder
    from ..utils.llm_config import get_llm_config
    from ..utils.diversity_maximizer import DiversityMaximizer, DiversityConfig
except ImportError:
    from src.qualsynth.prompts.prompt_builder import PromptBuilder
    from src.qualsynth.utils.llm_config import get_llm_config
    from src.qualsynth.utils.diversity_maximizer import DiversityMaximizer, DiversityConfig


@dataclass
class GenerationResult:
    """Result of sample generation."""
    samples: pd.DataFrame
    n_requested: int
    n_generated: int
    n_valid: int
    generation_time: float
    llm_calls: int
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestContractError(RuntimeError):
    """Raised when a backend rejects the frozen comparative request contract."""


class CounterfactualGenerator:
    """
    LLM-based counterfactual sample generator with fairness-aware prompting.
    
    Uses CSV output format for:
    - 2-3x faster generation (fewer tokens)
    - Robust parsing with CleverCSV
    - Truncation-safe (each row is independent)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        batch_size: int = 20,
        max_retries: int = 3,
        top_p: float = 0.95,
        presence_penalty: float = 0.6,
        frequency_penalty: float = 0.6,
        **kwargs
    ):
        """
        Initialize counterfactual generator.
        
        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            batch_size: Number of samples to generate per batch
            max_retries: Maximum number of retries on failure
            top_p: Nucleus sampling parameter (0.0-1.0)
            presence_penalty: Penalize token repetition (0.0-2.0)
            frequency_penalty: Penalize common tokens (0.0-2.0)
            **kwargs: Additional arguments including:
                - anchor_selection_strategy: Strategy for anchor selection ('typical', 'stratified', 'kmeans_diverse')
        """
        self.model_name = model_name
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_retries = max_retries  
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_output_tokens = int(kwargs.get('max_output_tokens', 8192))
        self.reasoning_effort = kwargs.get('reasoning_effort')
        self.strict_request_contract = bool(kwargs.get('strict_request_contract', False))
        self.verbose = kwargs.get('verbose', True)
        self.prompt_policy = kwargs.get('prompt_policy', 'anchor')
        self.raw_generation = bool(kwargs.get('raw_generation', False))
        
        # Initialize LLM config
        self.llm_config = get_llm_config(
            model_name=model_name,
            temperature=temperature
        )
        self.model_name = self.llm_config["config_list"][0]["model"]
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            use_chain_of_thought=False,
            use_counterfactual=None,
            use_few_shot=kwargs.get('use_few_shot', True),
            n_few_shot_examples=kwargs.get('n_few_shot_examples', 5),
            dynamic_few_shot=kwargs.get('dynamic_few_shot', True),
            few_shot_selection_strategy=kwargs.get('few_shot_selection_strategy', 'mixed'),
        )
        
        # Initialize Diversity Maximizer
        anchor_strategy = kwargs.get('anchor_selection_strategy', 'stratified')
        diversity_config = DiversityConfig(
            enable_column_permutation=True,  # GReaT-style column shuffling
            temperature_schedule="cosine",   # High temp early, low later
            base_temperature=temperature,
            max_temperature=min(1.2, temperature + 0.3),
            enable_dpp_selection=True,       # DPP for diverse subset selection
            enable_anti_similarity=True,     # Filter too-similar samples
            min_distance_threshold=0.15,     # Minimum distance between samples
            n_anchors=12,
            anchor_rotation_strategy=anchor_strategy  # Use config parameter
        )
        self.diversity_maximizer = DiversityMaximizer(diversity_config)
        self.diversity_fitted = False
        self.previous_anchors = None
        
        if self.verbose:
            print(f"   🎯 Anchor selection strategy: {anchor_strategy}")
            print(f"   🧭 Prompt policy: {self.prompt_policy}")
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Preprocessor for encoding generated samples (optional)
        self.preprocessor = None
    
    def set_preprocessor(self, preprocessor):
        """Set the preprocessor for encoding generated samples."""
        self.preprocessor = preprocessor
    
    def generate(
        self,
        dataset_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        fairness_report: Any,
        schema_report: Any,
        diversity_plan: Any,
        iteration: int = 1
    ) -> GenerationResult:
        """
        Generate synthetic samples using LLM with CSV output.
        """
        start_time = time.time()
        
        # Pass training data to diversity maximizer for discriminative anchor selection
        if hasattr(self.diversity_maximizer, 'set_training_data'):
            self.diversity_maximizer.set_training_data(X_train, y_train)
        
        CSV_BATCH_SIZE = 30
        
        all_samples = []
        all_errors = []
        total_llm_calls = 0
        total_tokens_used = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0
        request_records = []
        
        seen_hashes = set()
        
        n_calls = max(1, (n_samples + CSV_BATCH_SIZE - 1) // CSV_BATCH_SIZE)
        
        print(f"\n📊 Generating {n_samples} samples in {n_calls} CSV batch(es) of ~{CSV_BATCH_SIZE} each...")
        print(f"   ✨ Using CSV output")
        sys.stdout.flush()
        
        columns = list(X_train.columns)
        
        for call_idx in range(n_calls):
            samples_needed = n_samples - len(all_samples)
            if samples_needed <= 0:
                break
            
            current_batch_size = min(CSV_BATCH_SIZE, samples_needed)
            
            batch_result = self._generate_csv_batch(
                columns=columns,
                n_samples=current_batch_size,
                batch_idx=call_idx,
                X_train=X_train,
                y_train=y_train,
                schema_profile=schema_report,
                diversity_plan=diversity_plan,
                fairness_feedback=fairness_report,
                iteration=iteration
            )
            
            for sample in batch_result['samples']:
                try:
                    sample_hash = hash(tuple(sorted(sample.items())))
                    if sample_hash not in seen_hashes:
                        seen_hashes.add(sample_hash)
                        all_samples.append(sample)
                except (TypeError, AttributeError):
                    all_samples.append(sample)
            
            if batch_result['errors']:
                all_errors.extend(batch_result['errors'])
            
            total_llm_calls += batch_result['llm_calls']
            total_tokens_used += batch_result['total_tokens']
            total_prompt_tokens += batch_result.get('prompt_tokens', 0)
            total_completion_tokens += batch_result.get('completion_tokens', 0)
            total_reasoning_tokens += batch_result.get('reasoning_tokens', 0)
            total_cost += batch_result.get('cost', 0.0)
            request_records.extend(batch_result.get('request_records', []))
        
        print(f"   📊 Total samples collected: {len(all_samples)}")
        
        if all_samples:
            df_samples = pd.DataFrame(all_samples)
            
            expected_cols = set(X_train.columns)
            df_cols = set(df_samples.columns)
            
            missing_cols = expected_cols - df_cols
            if missing_cols and not self.raw_generation:
                print(f"   ⚠️  Missing {len(missing_cols)} columns, filling with NaN")
                for col in missing_cols:
                    df_samples[col] = np.nan
            
            if not self.raw_generation:
                df_samples = df_samples[[c for c in X_train.columns if c in df_samples.columns]]
                df_samples = df_samples.reindex(columns=X_train.columns, fill_value=np.nan)
            
            print(f"   ✅ DataFrame created: {len(df_samples)} rows, {len(df_samples.columns)} columns")
            
            # Anti-similarity filter
            if (
                not self.raw_generation
                and self.diversity_maximizer.config.enable_anti_similarity
                and len(df_samples) > 1
            ):
                n_before = len(df_samples)
                df_samples = self.diversity_maximizer.filter_by_anti_similarity(df_samples)
                n_filtered = n_before - len(df_samples)
                if n_filtered > 0:
                    print(f"   🔍 Anti-similarity filter: removed {n_filtered} too-similar samples")
            
            # DPP selection (if we have more samples than needed)
            if (
                not self.raw_generation
                and self.diversity_maximizer.config.enable_dpp_selection
                and len(df_samples) > n_samples
            ):
                n_before = len(df_samples)
                df_samples = self.diversity_maximizer.select_diverse_subset_dpp(df_samples, n_samples)
                print(f"   🎯 DPP selection: selected {len(df_samples)} most diverse from {n_before}")
            
            # Compute and log diversity metrics
            if len(df_samples) > 1:
                diversity_metrics = self.diversity_maximizer.compute_diversity_score(df_samples)
                print(f"   📊 Diversity metrics:")
                print(f"      - Numerical CV: {diversity_metrics['numerical_cv']:.1f}%")
                print(f"      - Categorical entropy: {diversity_metrics['categorical_entropy']:.1f}%")
                print(f"      - Mean inter-sample distance: {diversity_metrics['mean_distance']:.3f}")
                print(f"      - Overall diversity score: {diversity_metrics['overall_diversity']:.1f}")
        else:
            df_samples = pd.DataFrame(columns=X_train.columns)
            print(f"   ❌ No samples generated")
        
        generation_time = time.time() - start_time
        
        self.total_calls += total_llm_calls
        self.total_tokens += total_tokens_used
        self.total_cost += total_cost
        
        return GenerationResult(
            samples=df_samples,
            n_requested=n_samples,
            n_generated=len(all_samples),
            n_valid=len(all_samples),
            generation_time=generation_time,
            llm_calls=total_llm_calls,
            total_tokens=total_tokens_used,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            reasoning_tokens=total_reasoning_tokens,
            cost=total_cost,
            errors=all_errors,
            metadata={
                'request_records': request_records,
                'requested_model_slug': self.model_name,
                'output_format': 'csv',
                'strict_request_contract': self.strict_request_contract,
                'reasoning_effort': self.reasoning_effort,
            },
        )
    
    def _generate_csv_batch(
        self,
        columns: List[str],
        n_samples: int,
        batch_idx: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        schema_profile: Any,
        diversity_plan: Any,
        fairness_feedback: Any,
        iteration: int
    ) -> Dict[str, Any]:
        """Generate a batch of samples in CSV format."""
        request_records: List[Dict[str, Any]] = []
        errors: List[str] = []
        attempts_made = 0
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        total_cost = 0.0

        for retry in range(self.max_retries):
            prompt = self._build_csv_prompt(
                columns=columns,
                n_samples=n_samples,
                X_train=X_train,
                y_train=y_train,
                schema_profile=schema_profile,
                diversity_plan=diversity_plan,
                fairness_feedback=fairness_feedback,
                iteration=iteration
            )
            prompt_sha256 = self._prompt_hash(prompt)
            try:
                print(f"\n   Batch {batch_idx}, Attempt {retry + 1}/{self.max_retries}")
                print(f"   Prompt: {len(prompt['system'])} + {len(prompt['user'])} chars")
                sys.stdout.flush()

                attempts_made += 1
                llm_result = self._call_llm_for_csv(prompt)
                total_tokens += int(llm_result.get('total_tokens', 0) or 0)
                prompt_tokens += int(llm_result.get('prompt_tokens', 0) or 0)
                completion_tokens += int(llm_result.get('completion_tokens', 0) or 0)
                reasoning_tokens += int(llm_result.get('reasoning_tokens', 0) or 0)
                total_cost += float(llm_result.get('cost', 0.0) or 0.0)

                # Use permuted_columns for parsing (matches LLM output order)
                permuted_columns = prompt.get('permuted_columns', columns)
                samples = self._parse_csv_response(llm_result['content'], permuted_columns)
                content = llm_result.get('content', '')
                request_records.append({
                    **llm_result.get('request_metadata', {}),
                    'batch_index': batch_idx,
                    'attempt': retry + 1,
                    'requested_row_count': n_samples,
                    'prompt_sha256': prompt_sha256,
                    'raw_response': content,
                    'raw_response_sha256': hashlib.sha256(
                        content.encode('utf-8')
                    ).hexdigest(),
                    'parsed_row_count': len(samples),
                    'parse_success': bool(samples),
                })

                if len(samples) > 0:
                    print(f"   ✅ Parsed {len(samples)} samples from CSV")

                    # Samples are already in RAW format - no range validation needed
                    # The encode_features() step at training time will handle type conversion

                    return {
                        'samples': samples,
                        'llm_calls': attempts_made,
                        'total_tokens': total_tokens,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'reasoning_tokens': reasoning_tokens,
                        'cost': total_cost,
                        'request_records': request_records,
                        'errors': errors,
                    }
                else:
                    errors.append(
                        f"Batch {batch_idx}, attempt {retry + 1}: response parsed zero rows"
                    )
                    print(f"   ⚠️  No valid samples parsed, retrying...")

            except RequestContractError:
                # A comparative pilot must expose an unsupported parameter
                # immediately; retrying the same rejected contract obscures the
                # compatibility result and can incur unnecessary API calls.
                raise
            except Exception as e:
                error_message = (
                    f"Batch {batch_idx}, attempt {retry + 1}: "
                    f"{type(e).__name__}: {e}"
                )
                errors.append(error_message)
                request_records.append({
                    'batch_index': batch_idx,
                    'attempt': retry + 1,
                    'requested_row_count': n_samples,
                    'requested_model_slug': self.model_name,
                    'prompt_sha256': prompt_sha256,
                    'parsed_row_count': 0,
                    'parse_success': False,
                    'request_error': error_message,
                })
                print(f"   ❌ Batch {batch_idx} error: {e}")
                import traceback
                traceback.print_exc()
                if retry < self.max_retries - 1:
                    time.sleep(2 ** retry)
        
        return {
            'samples': [],
            'llm_calls': attempts_made,
            'total_tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'reasoning_tokens': reasoning_tokens,
            'cost': total_cost,
            'request_records': request_records,
            'errors': errors or [f"Batch {batch_idx}: All retries failed"]
        }

    @staticmethod
    def _prompt_hash(prompt: Dict[str, str]) -> str:
        """Hash the exact role content without persisting sensitive prompts."""
        payload = {
            'system': prompt.get('system', ''),
            'user': prompt.get('user', ''),
            'permuted_columns': prompt.get('permuted_columns', []),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()
    
    def _build_csv_prompt(
        self,
        columns: List[str],
        n_samples: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        schema_profile: Any,
        diversity_plan: Any,
        fairness_feedback: Any,
        iteration: int,
        max_iterations: int = 36
    ) -> Dict[str, str]:
        if self.prompt_policy == "no_anchor":
            return self._build_no_anchor_csv_prompt(
                columns=columns,
                n_samples=n_samples,
                X_train=X_train,
                y_train=y_train,
                iteration=iteration,
                max_iterations=max_iterations,
            )

        if not self.diversity_fitted:
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            self.diversity_maximizer.fit(X_train, categorical_features=categorical_cols)
            self.diversity_fitted = True
        
        # Column permutation
        permuted_columns = self.diversity_maximizer.get_permuted_columns(columns, iteration)
        
        # Temperature scheduling
        scheduled_temp = self.diversity_maximizer.get_scheduled_temperature(iteration, max_iterations)
        self.temperature = scheduled_temp
        
        # Reset indices
        X_train_reset = X_train.reset_index(drop=True)
        y_train_reset = y_train.reset_index(drop=True)
        
        minority_mask = y_train_reset == 1
        X_minority = X_train_reset[minority_mask]
        if len(X_minority) == 0:
            X_minority = X_train_reset
        
        # Anchor rotation
        n_anchors = self.diversity_maximizer.config.n_anchors
        anchors = self.diversity_maximizer.select_diverse_anchors(
            X_minority, 
            n_anchors=min(n_anchors, len(X_minority)),
            iteration=iteration,
            previous_anchors=self.previous_anchors
        )
        
        # Store for next iteration
        self.previous_anchors = anchors.copy()
        
        # Format anchors as CSV
        csv_header = ",".join(permuted_columns)
        
        def format_row(row):
            values = []
            for col in permuted_columns:
                val = row[col]
                if pd.isna(val):
                    values.append("")
                elif isinstance(val, (int, np.integer)):
                    values.append(str(int(val)))
                elif isinstance(val, (float, np.floating)):
                    if abs(val) >= 100:
                        values.append(f"{val:.0f}")
                    elif abs(val) >= 1:
                        values.append(f"{val:.1f}")
                    else:
                        values.append(f"{val:.2f}")
                else:
                    str_val = str(val)
                    if "," in str_val:
                        str_val = f'"{str_val}"'
                    values.append(str_val)
            return ",".join(values)
        
        anchor_rows = [format_row(row) for _, row in anchors.iterrows()]
        
        # Create anchor pairs for interpolation
        anchor_pairs = []
        anchor_list = list(anchors.iterrows())
        for i in range(min(5, len(anchor_list))):  # Create 5 pairs
            idx1 = i
            idx2 = (i + 1) % len(anchor_list)
            pair_str = f"  Pair {i+1}: Anchor {idx1+1} ↔ Anchor {idx2+1}"
            anchor_pairs.append(pair_str)
        
        # Build feature knowledge with distribution statistics
        feature_knowledge = []
        distribution_constraints = []
        categorical_features = []
        numerical_features = []
        
        for col in permuted_columns:
            col_data = X_minority[col].dropna()
            n_unique = col_data.nunique()
            
            if n_unique > 20 or col_data.dtype in ['float64', 'float32']:
                try:
                    col_min = col_data.min()
                    col_max = col_data.max()
                    col_mean = col_data.mean()
                    col_median = col_data.median()
                    col_std = col_data.std()
                    col_p10 = col_data.quantile(0.10)
                    col_p25 = col_data.quantile(0.25)
                    col_p75 = col_data.quantile(0.75)
                    col_p90 = col_data.quantile(0.90)
                    numerical_features.append(col)
                    
                    # Feature knowledge with distribution info
                    # Use appropriate precision: 2 decimals for small values, 0 for large
                    def fmt(v):
                        if abs(v) < 10:
                            return f"{v:.2f}"
                        elif abs(v) < 100:
                            return f"{v:.1f}"
                        else:
                            return f"{v:.0f}"
                    
                    feature_knowledge.append(
                        f"  • {col}: STRICT range [{fmt(col_min)}, {fmt(col_max)}], "
                        f"mean={fmt(col_mean)}, median={fmt(col_median)}, std={fmt(col_std)}"
                    )
                    
                    distribution_constraints.append(
                        f"  • {col}:\n"
                        f"      - MEDIAN target: {fmt(col_median)} (center most values here!)\n"
                        f"      - 50% of values MUST be in [{fmt(col_p25)}, {fmt(col_p75)}]\n"
                        f"      - 80% of values MUST be in [{fmt(col_p10)}, {fmt(col_p90)}]\n"
                        f"      - Only 10% should exceed {fmt(col_p90)}\n"
                        f"      - Only 10% should be below {fmt(col_p10)}\n"
                        f"      - NEVER exceed [{fmt(col_min)}, {fmt(col_max)}]"
                    )
                except (TypeError, ValueError):
                    pass
            else:
                # Categorical feature - add frequency distribution
                unique_vals = sorted(col_data.unique())[:8]
                value_counts = col_data.value_counts(normalize=True)
                categorical_features.append(col)
                
                # Show top categories with their frequencies
                top_cats = value_counts.head(4)
                freq_str = ", ".join([f"{v}({p*100:.0f}%)" for v, p in top_cats.items()])
                
                feature_knowledge.append(f"  • {col}: valid values = {unique_vals}")
                distribution_constraints.append(
                    f"  • {col}: Use categories proportionally: {freq_str}"
                )
        
        # Anchor-centric generation
        samples_per_anchor = max(1, n_samples // len(anchors))
        extra_samples = n_samples - (samples_per_anchor * len(anchors))
        
        # Build anchor-specific generation instructions
        anchor_instructions = []
        for i, (_, anchor_row) in enumerate(anchors.iterrows()):
            # How many samples for this anchor
            n_for_this_anchor = samples_per_anchor + (1 if i < extra_samples else 0)
            
            # Format anchor values with labels for clarity
            anchor_values = []
            for col in permuted_columns:
                val = anchor_row[col]
                if pd.isna(val):
                    anchor_values.append(f"{col}=")
                elif isinstance(val, (int, np.integer)):
                    anchor_values.append(f"{col}={int(val)}")
                elif isinstance(val, (float, np.floating)):
                    if abs(val) >= 100:
                        anchor_values.append(f"{col}={val:.0f}")
                    elif abs(val) >= 1:
                        anchor_values.append(f"{col}={val:.1f}")
                    else:
                        anchor_values.append(f"{col}={val:.2f}")
        else:
            anchor_values.append(f"{col}={val}")
            
            anchor_str = ", ".join(anchor_values)
            anchor_instructions.append(
                f"ANCHOR {i+1}: {anchor_str}\n"
                f"   → Generate {n_for_this_anchor} variations (modify ONLY 1-2 features, keep rest EXACTLY as shown)"
            )
        
        # Identify which features can be varied
        vary_instructions = []
        for col in numerical_features[:5]:  # Top 5 numerical features
            vary_instructions.append(f"  • {col}: vary by ±10-15% of anchor value")
        for col in categorical_features[:3]:  # Top 3 categorical features
            vary_instructions.append(f"  • {col}: 80% keep same, 20% switch to another valid value")
        
        samples_per_anchor = max(1, n_samples // len(anchors))
        extra_samples = n_samples - (samples_per_anchor * len(anchors))
        
        # Build anchor-specific generation instructions
        anchor_instructions = []
        for i, (_, anchor_row) in enumerate(anchors.iterrows()):
            n_for_this_anchor = samples_per_anchor + (1 if i < extra_samples else 0)
            
            # Format anchor values with labels
            anchor_values = []
            for col in permuted_columns:
                val = anchor_row[col]
                if pd.isna(val):
                    anchor_values.append(f"{col}=")
                elif isinstance(val, (int, np.integer)):
                    anchor_values.append(f"{col}={int(val)}")
                elif isinstance(val, (float, np.floating)):
                    if abs(val) >= 100:
                        anchor_values.append(f"{col}={val:.0f}")
                    elif abs(val) >= 1:
                        anchor_values.append(f"{col}={val:.1f}")
                    else:
                        anchor_values.append(f"{col}={val:.2f}")
                else:
                    anchor_values.append(f"{col}={val}")
            
            anchor_str = ", ".join(anchor_values)
            anchor_instructions.append(
                f"ANCHOR {i+1}: {anchor_str}\n"
                f"   → Generate {n_for_this_anchor} variations (modify ONLY 1-2 features, keep rest EXACTLY as shown)"
            )
        
        # Build dynamic example from first anchor
        first_anchor = anchors.iloc[0]
        
        # Get first 2 numerical features for the example
        num_feats = [col for col in numerical_features[:2] if col in permuted_columns]
        if len(num_feats) >= 2:
            f1, f2 = num_feats[0], num_feats[1]
            v1, v2 = first_anchor[f1], first_anchor[f2]
            
            def fmt(v):
                if abs(v) >= 100:
                    return f"{v:.0f}"
                elif abs(v) >= 1:
                    return f"{v:.1f}"
                else:
                    return f"{v:.2f}"
            
            # Small variations (±10-15%)
            v1_up = v1 * 1.12
            v1_down = v1 * 0.88
            v2_up = v2 * 1.10
            
            dynamic_example = f"""EXAMPLE (using ANCHOR 1: {f1}={fmt(v1)}, {f2}={fmt(v2)}, ...):
  ✅ GOOD: {f1}={fmt(v1_up)}, {f2}={fmt(v2)}, ... (varied only {f1} by ~12%)
  ✅ GOOD: {f1}={fmt(v1)}, {f2}={fmt(v2_up)}, ... (varied only {f2} by ~10%)
  ✅ GOOD: {f1}={fmt(v1_down)}, {f2}={fmt(v2_up)}, ... (varied both slightly)
  ❌ BAD:  Changing 3+ features or making large changes (>30%)"""
        else:
            dynamic_example = "EXAMPLE: Vary 1-2 features by ±10-15%, keep all others exactly as anchor."

        system_prompt = f"""You are an expert synthetic data generator using ANCHOR-CENTRIC generation.

🎯 GOAL: Generate variations of REAL samples to preserve feature correlations.

⚠️ WHY ANCHOR-CENTRIC GENERATION:
- Real samples have CORRECT correlations between features
- If you generate features independently, you LOSE these correlations
- Lost correlations = BAD classifier performance
- By varying ONLY 1-2 features from a real anchor, correlations are PRESERVED

🔒 THE GOLDEN RULE:
For each generated sample:
1. Start with an EXACT COPY of the assigned anchor
2. Modify ONLY 1-2 features (small variation)
3. Keep ALL OTHER FEATURES EXACTLY as in the anchor
4. This preserves the natural relationships between features

📋 OUTPUT FORMAT:
- Output ONLY CSV data rows (NO header, NO explanations)
- Each row = one sample
- Exactly {len(permuted_columns)} comma-separated values per row
- Column order: {csv_header}
- Generate exactly {n_samples} rows total

⚠️ WHAT NOT TO DO:
- DON'T generate features independently
- DON'T use "typical" values from statistics
- DON'T create samples from scratch
- DON'T modify more than 2 features per sample

✅ WHAT TO DO:
- Start with anchor values
- Pick 1-2 features to vary slightly
- Keep everything else IDENTICAL to anchor"""

        user_prompt = f"""Generate {n_samples} samples as SMALL VARIATIONS of the anchors below.

═══════════════════════════════════════════════════════════════════
📊 COLUMN ORDER (must match exactly):
═══════════════════════════════════════════════════════════════════
{csv_header}

═══════════════════════════════════════════════════════════════════
📚 VALID RANGES (for validation only - stay close to anchor values):
═══════════════════════════════════════════════════════════════════
{chr(10).join(feature_knowledge)}

═══════════════════════════════════════════════════════════════════
🎯 ANCHORS - Generate variations of THESE EXACT samples:
═══════════════════════════════════════════════════════════════════

{chr(10).join(anchor_instructions)}

═══════════════════════════════════════════════════════════════════
✏️ HOW TO VARY (pick 1-2 features per sample):
═══════════════════════════════════════════════════════════════════
{chr(10).join(vary_instructions)}

═══════════════════════════════════════════════════════════════════
🔒 PRESERVE CORRELATIONS
═══════════════════════════════════════════════════════════════════

For EACH sample you generate:
1. COPY the assigned anchor's values EXACTLY
2. Pick ONLY 1-2 features to modify
3. Apply SMALL variation (±10-15% for numerical, same/similar for categorical)
4. Keep ALL OTHER features UNCHANGED from anchor

{dynamic_example}

The goal is to create NEIGHBORS of the anchor, NOT completely new samples.

═══════════════════════════════════════════════════════════════════
🚀 NOW GENERATE {n_samples} ANCHOR-VARIATION ROWS (NO HEADER):
═══════════════════════════════════════════════════════════════════
"""
        
        return {
            'system': system_prompt,
            'user': user_prompt,
            'permuted_columns': permuted_columns  # Return for parsing
        }

    def _build_no_anchor_csv_prompt(
        self,
        columns: List[str],
        n_samples: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        iteration: int,
        max_iterations: int = 36,
    ) -> Dict[str, str]:
        """Build a distribution-summary prompt with no anchor rows or local-copy instructions."""
        if not self.diversity_fitted:
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            self.diversity_maximizer.fit(X_train, categorical_features=categorical_cols)
            self.diversity_fitted = True

        permuted_columns = self.diversity_maximizer.get_permuted_columns(columns, iteration)
        self.temperature = self.diversity_maximizer.get_scheduled_temperature(iteration, max_iterations)
        csv_header = ",".join(permuted_columns)

        X_train_reset = X_train.reset_index(drop=True)
        y_train_reset = y_train.reset_index(drop=True)
        minority_mask = y_train_reset == 1
        X_minority = X_train_reset[minority_mask]
        if len(X_minority) == 0:
            X_minority = X_train_reset

        feature_lines = []
        for col in permuted_columns:
            col_data = X_minority[col].dropna()
            n_unique = col_data.nunique()
            if n_unique > 20 or col_data.dtype in ['float64', 'float32', 'int64', 'int32']:
                try:
                    col_min = col_data.min()
                    col_max = col_data.max()
                    col_mean = col_data.mean()
                    col_median = col_data.median()
                    col_std = col_data.std()

                    def fmt(value):
                        try:
                            if abs(value) < 10:
                                return f"{value:.2f}"
                            if abs(value) < 100:
                                return f"{value:.1f}"
                            return f"{value:.0f}"
                        except TypeError:
                            return str(value)

                    feature_lines.append(
                        f"  - {col}: range [{fmt(col_min)}, {fmt(col_max)}], "
                        f"mean {fmt(col_mean)}, median {fmt(col_median)}, std {fmt(col_std)}"
                    )
                except (TypeError, ValueError):
                    feature_lines.append(f"  - {col}: use realistic values observed in the training distribution")
            else:
                value_counts = col_data.value_counts(normalize=True).head(6)
                freq = ", ".join(f"{value}({prob*100:.0f}%)" for value, prob in value_counts.items())
                feature_lines.append(f"  - {col}: categories/frequencies {freq}")

        system_prompt = f"""You are an expert synthetic tabular data generator.

GOAL: Generate realistic minority-class tabular rows from dataset-level feature summaries.

IMPORTANT:
- Do not use anchor examples.
- Do not copy or locally perturb any specific training row.
- Generate plausible complete rows from the feature ranges and category frequencies.
- Preserve broad statistical relationships by using realistic combinations, not independent random noise.

OUTPUT FORMAT:
- Output ONLY CSV data rows.
- Do not include a header or explanations.
- Each row must contain exactly {len(permuted_columns)} comma-separated values.
- Column order: {csv_header}
- Generate exactly {n_samples} rows."""

        user_prompt = f"""Generate {n_samples} realistic minority-class samples from these feature summaries.

COLUMN ORDER:
{csv_header}

FEATURE SUMMARIES:
{chr(10).join(feature_lines)}

GENERATION GUIDANCE:
- Keep numerical values inside the listed ranges unless the distribution summary clearly implies edge values.
- Use categorical values in approximately the listed frequencies.
- Produce varied rows across the distribution.
- Do not mention anchors, source rows, or examples.

NOW GENERATE {n_samples} CSV ROWS ONLY:
"""

        return {
            'system': system_prompt,
            'user': user_prompt,
            'permuted_columns': permuted_columns,
        }
    
    def _call_llm_for_csv(self, prompt: Dict[str, str]) -> Dict[str, Any]:
        """Call LLM API for CSV output (no JSON mode)."""
        from openai import OpenAI
        import httpx
        import time as time_module
        
        config = self.llm_config.get('config_list', [{}])[0]
        
        ollama_model_env = os.getenv('OLLAMA_MODEL')
        openai_api_base_env = os.getenv('OPENAI_API_BASE', '')
        
        is_ollama = (
            ollama_model_env is not None or
            'localhost:11434' in openai_api_base_env.lower() or
            'ollama' in openai_api_base_env.lower()
        )
        
        # Check if using OpenRouter (cloud API with rate limits)
        is_openrouter = 'openrouter' in openai_api_base_env.lower()
        
        if is_ollama:
            model_name = ollama_model_env or config.get('model', self.model_name)
            api_base = openai_api_base_env or 'http://localhost:11434/v1'
            api_key = os.getenv('OPENAI_API_KEY', 'not-needed')
        else:
            api_base = config.get('api_base') or openai_api_base_env or 'http://localhost:11434/v1'
            api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY', 'not-needed')
            model_name = config.get('model', self.model_name)

        if api_base and not api_base.rstrip('/').endswith('/v1'):
            api_base = f"{api_base.rstrip('/')}/v1"

        # Freeze one provider-neutral role contract for comparative studies.
        # A single user turn is supported by both the Gemma and GPT OpenRouter
        # routes and avoids a model-specific prompt repair.
        messages = [{
            "role": "user",
            "content": f"{prompt['system']}\n\n{prompt['user']}"
        }]
        
        timeout = httpx.Timeout(300.0, read=300.0)
        client = OpenAI(
            api_key=api_key, 
            base_url=api_base,
            timeout=timeout
        )
        
        # Rate limiting for OpenRouter (20 req/min = 3 sec between requests)
        rate_limit_delay = float(os.getenv('OPENROUTER_RATE_LIMIT_DELAY', '0'))
        if is_openrouter and rate_limit_delay > 0:
            print(f"   ⏳ Rate limit delay: {rate_limit_delay:.1f}s")
            sys.stdout.flush()
            time_module.sleep(rate_limit_delay)
        
        print(f"   🤖 Calling {model_name}")
        sys.stdout.flush()
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_output_tokens,
        }
        
        # Disable reasoning/thinking when the backend exposes a toggle.
        is_groq = 'groq' in api_base.lower()
        normalized_model_name = model_name.lower()
        is_qwen_thinking_model = "qwen3" in normalized_model_name or "qwen3.5" in normalized_model_name
        extra_body = {}
        if is_groq:
            extra_body["reasoning_effort"] = "low"
        if is_qwen_thinking_model:
            # LM Studio's OpenAI-compatible API respects Qwen thinking control
            # most reliably through chat_template_kwargs.
            extra_body["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if extra_body:
            request_params["extra_body"] = extra_body
        
        # Retry logic with exponential backoff for rate limits
        max_retries = 10
        base_wait = 60  # Start with 60 seconds for rate limits
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**request_params)
                
                content = response.choices[0].message.content
                
                usage = response.usage if hasattr(response, 'usage') else None
                total_tokens = usage.total_tokens if usage else 0
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                completion_details = getattr(usage, 'completion_tokens_details', None) if usage else None
                reasoning_tokens = (
                    getattr(completion_details, 'reasoning_tokens', 0) or 0
                    if completion_details is not None
                    else 0
                )
                response_payload = (
                    response.model_dump()
                    if hasattr(response, 'model_dump')
                    else {}
                )
                usage_payload = response_payload.get('usage') or {}
                reported_cost = getattr(usage, 'cost', None) if usage else None
                if reported_cost is None:
                    reported_cost = usage_payload.get('cost')
                estimated_cost = self._estimate_call_cost(prompt_tokens, completion_tokens)
                cost = float(reported_cost) if reported_cost is not None else estimated_cost
                choice = response.choices[0]
                resolved_model = getattr(response, 'model', None) or model_name
                resolved_provider = (
                    getattr(response, 'provider', None)
                    or response_payload.get('provider')
                )
                endpoint_class = self._classify_endpoint(api_base)
                
                print(f"   ✅ Response received: {len(content)} chars, {total_tokens} tokens")
                
                return {
                    'content': content,
                    'total_tokens': total_tokens,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'reasoning_tokens': reasoning_tokens,
                    'cost': cost,
                    'request_metadata': {
                        'requested_model_slug': model_name,
                        'resolved_model_slug': resolved_model,
                        'resolved_provider': resolved_provider,
                        'endpoint_class': endpoint_class,
                        'request_date_utc': datetime.now(timezone.utc).isoformat(),
                        'message_contract': 'single_user_csv_v1',
                        'output_format': 'csv',
                        'sampling': {
                            'temperature': self.temperature,
                            'top_p': self.top_p,
                            'presence_penalty': self.presence_penalty,
                            'frequency_penalty': self.frequency_penalty,
                            'max_output_tokens': self.max_output_tokens,
                        },
                        'reasoning_effort': self.reasoning_effort,
                        'finish_reason': getattr(choice, 'finish_reason', None),
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'reasoning_tokens': reasoning_tokens,
                        'total_tokens': total_tokens,
                        'estimated_cost_usd': cost,
                        'cost_source': 'provider_reported' if reported_cost is not None else 'local_estimate',
                    },
                }
                
            except Exception as e:
                error_str = str(e).lower()

                if self.strict_request_contract and self._is_contract_rejection(e, error_str):
                    raise RequestContractError(
                        f"Backend rejected the frozen request contract for {model_name}: {e}"
                    ) from e
                
                # Check for rate limit errors (429)
                if '429' in str(e) or 'rate' in error_str or 'limit' in error_str or 'quota' in error_str:
                    # Parse wait time from error message if available
                    import re
                    wait_match = re.search(r'try again in (\d+)m?(\d*)s?', str(e), re.IGNORECASE)
                    if wait_match:
                        minutes = int(wait_match.group(1)) if wait_match.group(1) else 0
                        seconds = int(wait_match.group(2)) if wait_match.group(2) else 0
                        wait_time = minutes * 60 + seconds + 10  # Add 10s buffer
                    else:
                        # Exponential backoff: 60s, 120s, 240s, ...
                        wait_time = base_wait * (2 ** attempt)
                    
                    # Cap at 30 minutes for daily limits
                    if 'day' in error_str or 'daily' in error_str or 'tpd' in error_str:
                        wait_time = max(wait_time, 300)  # At least 5 minutes for daily limits
                        print(f"   ⚠️  Daily token limit hit. Waiting {wait_time//60}m {wait_time%60}s...")
                    else:
                        print(f"   ⚠️  Rate limit (attempt {attempt+1}/{max_retries}). Waiting {wait_time//60}m {wait_time%60}s...")
                    
                    sys.stdout.flush()
                    time_module.sleep(wait_time)
                    continue
                
                # Check for timeout errors
                elif 'timeout' in error_str or 'timed out' in error_str:
                    wait_time = 30 * (attempt + 1)
                    print(f"   ⚠️  Timeout (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                    sys.stdout.flush()
                    time_module.sleep(wait_time)
                    continue
                
                # Other errors - don't retry
                else:
                    print(f"   ❌ API Error: {e}")
                    raise
        
        # All retries exhausted
        raise Exception(f"Failed after {max_retries} retries due to rate limits")

    @staticmethod
    def _classify_endpoint(api_base: str) -> str:
        normalized = (api_base or '').lower()
        if 'openrouter' in normalized:
            return 'openrouter'
        if 'localhost:11434' in normalized or 'ollama' in normalized:
            return 'local_ollama'
        if 'localhost' in normalized or '127.0.0.1' in normalized:
            return 'local_openai_compatible'
        if 'openai.com' in normalized:
            return 'openai'
        return 'openai_compatible_remote'

    @staticmethod
    def _is_contract_rejection(exc: Exception, error_text: str) -> bool:
        status_code = getattr(exc, 'status_code', None)
        parameter_terms = (
            'unsupported parameter',
            'unknown parameter',
            'unrecognized parameter',
            'invalid parameter',
            'not supported',
        )
        return status_code in {400, 404, 422} or any(term in error_text for term in parameter_terms)
    
    def _parse_csv_response(self, content: str, expected_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Parse CSV response from LLM.
        
        Handles:
        - Missing header (LLM outputs data directly)
        - Code blocks (```csv ... ```)
        - Messy formatting
        """
        samples = []
        
        content = content.strip()
        
        # Remove code blocks if present
        if content.startswith("```"):
            lines = content.split('\n')
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            content = '\n'.join(lines[start_idx:end_idx])
        
        if not content.strip():
            print("   ⚠️  Empty response after cleanup")
            return []
        
        # Check if first line is header or data
        # If first line doesn't contain expected column names, prepend header
        first_line = content.split('\n')[0].strip()
        expected_header = ",".join(expected_columns)
        
        # Check if first line looks like data (contains values like A11, A12, numbers)
        # or like header (contains column names like 'checking_status', 'duration')
        has_header = any(col in first_line for col in expected_columns[:3])
        
        if not has_header:
            print(f"   ⚠️  No header detected - prepending expected columns")
            content = expected_header + "\n" + content
        
        # Use pandas for CSV parsing
        try:
            df = pd.read_csv(
                StringIO(content),
                on_bad_lines='skip',
                skipinitialspace=True,
                encoding='utf-8'
            )
            if df is not None and len(df) > 0:
                samples = df.to_dict('records')
                print(f"   ✅ Pandas parsed {len(samples)} rows with {len(df.columns)} columns")
                return samples
        except Exception as e:
            print(f"   ⚠️  Pandas failed: {e}, trying manual parse...")
        
        # Last resort: manual parsing
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return []
        
        header = [h.strip().strip('"') for h in lines[0].split(',')]
        
        for line in lines[1:]:
            if not line.strip():
                continue
            
            values = [v.strip().strip('"') for v in line.split(',')]
            
            if len(values) == len(header):
                sample = {}
                for col, val in zip(header, values):
                    try:
                        if '.' in val:
                            sample[col] = float(val)
                        else:
                            sample[col] = int(val)
                    except (ValueError, TypeError):
                        sample[col] = val
                samples.append(sample)
        
        if samples:
            print(f"   ✅ Manual parse got {len(samples)} rows")
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_tokens_per_call': self.total_tokens / max(1, self.total_calls)
        }

    @staticmethod
    def _estimate_call_cost(prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate monetary cost from optional per-million-token environment prices."""
        input_cost_per_million = float(os.getenv('LLM_INPUT_COST_PER_1M_TOKENS', '0') or 0)
        output_cost_per_million = float(os.getenv('LLM_OUTPUT_COST_PER_1M_TOKENS', '0') or 0)
        return (
            (prompt_tokens / 1_000_000.0) * input_cost_per_million
            + (completion_tokens / 1_000_000.0) * output_cost_per_million
        )
