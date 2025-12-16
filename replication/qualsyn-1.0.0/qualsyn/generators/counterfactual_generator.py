"""
Counterfactual Generator for Qualsynth

This component generates synthetic samples using LLMs with fairness-aware prompting.
Uses CSV output format for faster generation and robust parsing with CleverCSV.
"""

import pandas as pd
import numpy as np
import time
import os
import sys
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
    cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        model_name: str = "gemma3-m4-fast",
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
        self.verbose = kwargs.get('verbose', True)
        
        # Initialize LLM config
        self.llm_config = get_llm_config(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            use_chain_of_thought=False,
            use_counterfactual=None,
            use_few_shot=True
        )
        
        # Initialize SOTA Diversity Maximizer
        # Default to "stratified" for balanced distribution coverage
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
        
        CSV_BATCH_SIZE = 30
        
        all_samples = []
        all_errors = []
        total_llm_calls = 0
        total_tokens_used = 0
        
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
        
        print(f"   📊 Total samples collected: {len(all_samples)}")
        
        if all_samples:
            df_samples = pd.DataFrame(all_samples)
            
            expected_cols = set(X_train.columns)
            df_cols = set(df_samples.columns)
            
            missing_cols = expected_cols - df_cols
            if missing_cols:
                print(f"   ⚠️  Missing {len(missing_cols)} columns, filling with NaN")
                for col in missing_cols:
                    df_samples[col] = np.nan
            
            df_samples = df_samples[[c for c in X_train.columns if c in df_samples.columns]]
            df_samples = df_samples.reindex(columns=X_train.columns, fill_value=np.nan)
            
            print(f"   ✅ DataFrame created: {len(df_samples)} rows, {len(df_samples.columns)} columns")
            
            # ═══════════════════════════════════════════════════════════════════
            # SOTA TECHNIQUE 4: ANTI-SIMILARITY FILTER
            # ═══════════════════════════════════════════════════════════════════
            if self.diversity_maximizer.config.enable_anti_similarity and len(df_samples) > 1:
                n_before = len(df_samples)
                df_samples = self.diversity_maximizer.filter_by_anti_similarity(df_samples)
                n_filtered = n_before - len(df_samples)
                if n_filtered > 0:
                    print(f"   🔍 Anti-similarity filter: removed {n_filtered} too-similar samples")
            
            # ═══════════════════════════════════════════════════════════════════
            # SOTA TECHNIQUE 5: DPP SELECTION (if we have more samples than needed)
            # ═══════════════════════════════════════════════════════════════════
            if self.diversity_maximizer.config.enable_dpp_selection and len(df_samples) > n_samples:
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
        
        return GenerationResult(
            samples=df_samples,
            n_requested=n_samples,
            n_generated=len(all_samples),
            n_valid=len(all_samples),
            generation_time=generation_time,
            llm_calls=total_llm_calls,
            total_tokens=total_tokens_used,
            errors=all_errors
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
        
        for retry in range(self.max_retries):
            try:
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
                
                print(f"\n   Batch {batch_idx}, Attempt {retry + 1}/{self.max_retries}")
                print(f"   Prompt: {len(prompt['system'])} + {len(prompt['user'])} chars")
                sys.stdout.flush()
                
                llm_result = self._call_llm_for_csv(prompt)
                
                # Use permuted_columns for parsing (matches LLM output order)
                permuted_columns = prompt.get('permuted_columns', columns)
                samples = self._parse_csv_response(llm_result['content'], permuted_columns)
                
                if len(samples) > 0:
                    print(f"   ✅ Parsed {len(samples)} samples from CSV")
                    
                    # Samples are already in RAW format - no range validation needed
                    # The encode_features() step at training time will handle type conversion
                    
                    return {
                        'samples': samples,
                        'llm_calls': 1,
                        'total_tokens': llm_result.get('total_tokens', 0),
                        'errors': []
                    }
                else:
                    print(f"   ⚠️  No valid samples parsed, retrying...")
            
            except Exception as e:
                print(f"   ❌ Batch {batch_idx} error: {e}")
                import traceback
                traceback.print_exc()
                if retry < self.max_retries - 1:
                    time.sleep(2 ** retry)
        
        return {
            'samples': [],
            'llm_calls': self.max_retries,
            'total_tokens': 0,
            'errors': [f"Batch {batch_idx}: All retries failed"]
        }
    
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
        """Build SOTA DIVERSITY-MAXIMIZED prompt for sample generation.
        
        SOTA TECHNIQUES (based on GReaT, G2, TabDDPM research):
        1. COLUMN PERMUTATION: Shuffle column order each batch (GReaT)
        2. TEMPERATURE SCHEDULING: Cosine annealing for exploration → refinement
        3. ANCHOR ROTATION: Different anchors each iteration (avoid repetition)
        4. INTERPOLATION: Blend between anchor pairs (SMOTE-like)
        5. DPP SELECTION: Post-generation diversity maximization
        """
        # ═══════════════════════════════════════════════════════════════════
        # FIT DIVERSITY MAXIMIZER (once)
        # ═══════════════════════════════════════════════════════════════════
        if not self.diversity_fitted:
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            self.diversity_maximizer.fit(X_train, categorical_features=categorical_cols)
            self.diversity_fitted = True
        
        # ═══════════════════════════════════════════════════════════════════
        # SOTA TECHNIQUE 1: COLUMN PERMUTATION (GReaT)
        # ═══════════════════════════════════════════════════════════════════
        permuted_columns = self.diversity_maximizer.get_permuted_columns(columns, iteration)
        
        # ═══════════════════════════════════════════════════════════════════
        # SOTA TECHNIQUE 2: TEMPERATURE SCHEDULING
        # ═══════════════════════════════════════════════════════════════════
        scheduled_temp = self.diversity_maximizer.get_scheduled_temperature(iteration, max_iterations)
        self.temperature = scheduled_temp  # Update for this batch
        
        minority_mask = y_train == 1
        X_minority = X_train[minority_mask]
        if len(X_minority) == 0:
            X_minority = X_train
        
        # ═══════════════════════════════════════════════════════════════════
        # SOTA TECHNIQUE 3: ANCHOR ROTATION (avoid repeating same anchors)
        # ═══════════════════════════════════════════════════════════════════
        n_anchors = self.diversity_maximizer.config.n_anchors
        anchors = self.diversity_maximizer.select_diverse_anchors(
            X_minority, 
            n_anchors=min(n_anchors, len(X_minority)),
            iteration=iteration,
            previous_anchors=self.previous_anchors
        )
        
        # Store for next iteration
        self.previous_anchors = anchors.copy()
        
        # ═══════════════════════════════════════════════════════════════════
        # FORMAT ANCHORS AS CSV (using permuted column order)
        # ═══════════════════════════════════════════════════════════════════
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
        
        # ═══════════════════════════════════════════════════════════════════
        # IMPROVEMENT 4: CREATE ANCHOR PAIRS FOR INTERPOLATION
        # ═══════════════════════════════════════════════════════════════════
        anchor_pairs = []
        anchor_list = list(anchors.iterrows())
        for i in range(min(5, len(anchor_list))):  # Create 5 pairs
            idx1 = i
            idx2 = (i + 1) % len(anchor_list)
            pair_str = f"  Pair {i+1}: Anchor {idx1+1} ↔ Anchor {idx2+1}"
            anchor_pairs.append(pair_str)
        
        # ═══════════════════════════════════════════════════════════════════
        # BUILD FEATURE KNOWLEDGE WITH DISTRIBUTION STATISTICS
        # Key insight: LLM needs to know the MINORITY CLASS distribution
        # BUG FIX: Use X_minority statistics, NOT X_train (overall) statistics!
        # ═══════════════════════════════════════════════════════════════════
        feature_knowledge = []
        distribution_constraints = []
        categorical_features = []
        numerical_features = []
        
        for col in permuted_columns:
            # CRITICAL FIX: Use minority class data for distribution statistics
            col_data = X_minority[col].dropna()
            n_unique = col_data.nunique()
            
            if n_unique > 20 or col_data.dtype in ['float64', 'float32']:
                # Numerical feature
                try:
                    col_min = col_data.min()
                    col_max = col_data.max()
                    col_mean = col_data.mean()
                    col_median = col_data.median()
                    col_std = col_data.std()
                    numerical_features.append(col)
                    
                    # Use appropriate precision
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
                except (TypeError, ValueError):
                    pass
            else:
                # Categorical feature
                unique_vals = sorted(col_data.unique())[:8]
                categorical_features.append(col)
                feature_knowledge.append(f"  • {col}: valid values = {unique_vals}")
        
        # ═══════════════════════════════════════════════════════════════════
        # ANCHOR-CENTRIC GENERATION PROMPT (CORRELATION-PRESERVING)
        # Key insight: Generate variations of REAL samples to preserve correlations
        # This fixes the correlation reversal problem (e.g., age↔hours)
        # ═══════════════════════════════════════════════════════════════════
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
        
        system_prompt = f"""You are an expert synthetic data generator using ANCHOR-CENTRIC generation.

🎯 CRITICAL GOAL: Generate variations of REAL samples to preserve feature correlations.

⚠️ WHY ANCHOR-CENTRIC GENERATION:
- Real samples have CORRECT correlations between features (e.g., age↔hours, education↔occupation)
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

        # ═══════════════════════════════════════════════════════════════════
        # ANCHOR-CENTRIC USER PROMPT
        # ═══════════════════════════════════════════════════════════════════
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
🔒 CRITICAL: PRESERVE CORRELATIONS
═══════════════════════════════════════════════════════════════════

For EACH sample you generate:
1. COPY the assigned anchor's values EXACTLY
2. Pick ONLY 1-2 features to modify
3. Apply SMALL variation (±10-15% for numerical, same/similar for categorical)
4. Keep ALL OTHER features UNCHANGED from anchor

EXAMPLE (if anchor has age=45, hours=48, education=Bachelors, occupation=Prof-specialty):
  ✅ GOOD: age=47, hours=48, education=Bachelors, occupation=Prof-specialty (varied only age)
  ✅ GOOD: age=45, hours=50, education=Bachelors, occupation=Prof-specialty (varied only hours)
  ✅ GOOD: age=43, hours=46, education=Bachelors, occupation=Prof-specialty (varied age+hours slightly)
  ❌ BAD:  age=28, hours=35, education=HS-grad, occupation=Sales (too many changes!)
  ❌ BAD:  age=45, hours=60, education=Masters, occupation=Exec-managerial (3+ features changed)

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
    
    def _call_llm_for_csv(self, prompt: Dict[str, str]) -> Dict[str, Any]:
        """Call LLM API for CSV output (no JSON mode)."""
        from openai import OpenAI
        import httpx
        
        messages = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ]
        
        config = self.llm_config.get('config_list', [{}])[0]
        
        ollama_model_env = os.getenv('OLLAMA_MODEL')
        openai_api_base_env = os.getenv('OPENAI_API_BASE', '')
        
        is_ollama = (
            ollama_model_env is not None or
            'localhost:11434' in openai_api_base_env.lower() or
            'ollama' in openai_api_base_env.lower()
        )
        
        if is_ollama:
            model_name = ollama_model_env or config.get('model', self.model_name)
            api_base = openai_api_base_env or 'http://localhost:11434/v1'
            api_key = os.getenv('OPENAI_API_KEY', 'not-needed')
        else:
            api_base = config.get('api_base') or openai_api_base_env or 'http://localhost:11434/v1'
            api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY', 'not-needed')
            model_name = config.get('model', self.model_name)
        
        timeout = httpx.Timeout(300.0, read=300.0)
        client = OpenAI(
            api_key=api_key, 
            base_url=api_base,
            timeout=timeout
        )
        
        print(f"   🤖 Calling {model_name}")
        sys.stdout.flush()
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=8192,
        )
        
        content = response.choices[0].message.content
        
        usage = response.usage if hasattr(response, 'usage') else None
        total_tokens = usage.total_tokens if usage else 0
        
        print(f"   ✅ Response received: {len(content)} chars, {total_tokens} tokens")
        
        return {
            'content': content,
            'total_tokens': total_tokens
        }
    
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
