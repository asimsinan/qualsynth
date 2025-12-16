"""
Main Prompt Builder for Qualsynth

This module assembles complete prompts from all components.
"""

import pandas as pd
from typing import Dict, List, Any, Optional

try:
    from .base_templates import BaseTemplates
    from .fairness_templates import FairnessTemplates
    from .counterfactual_templates import CounterfactualTemplates
    from .few_shot_builder import FewShotBuilder
    from .constraint_encoder import ConstraintEncoder
    from ..utils.value_transformer import ValueTransformer
except ImportError:
    from base_templates import BaseTemplates
    from fairness_templates import FairnessTemplates
    from counterfactual_templates import CounterfactualTemplates
    from few_shot_builder import FewShotBuilder
    from constraint_encoder import ConstraintEncoder
    try:
        from src.qualsynth.utils.value_transformer import ValueTransformer
    except ImportError:
        ValueTransformer = None


class PromptBuilder:
    """
    Main prompt builder that assembles complete prompts for LLM generation.
    
    Combines:
    - Base templates (system, task, output format)
    - Fairness templates (fairness instructions, target groups)
    - Counterfactual templates (counterfactual reasoning)
    - Few-shot examples (representative samples)
    - Constraint encoding (schema, logical, fairness, diversity)
    """
    
    def __init__(
        self,
        strategy: str = "STANDARD",
        use_chain_of_thought: bool = True,
        use_few_shot: bool = True,
        use_counterfactual: bool = False,
        n_few_shot_examples: int = 5,
        enable_diversity_prompting: bool = False,
        diversity_prompt_strength: str = "medium",
        value_transformer: Optional[Any] = None
    ):
        """
        Initialize PromptBuilder.
        
        Args:
            strategy: Generation strategy (STANDARD, FAIRNESS_FIRST, DIVERSITY_FIRST, etc.)
            use_chain_of_thought: Whether to include chain-of-thought prompting
            use_few_shot: Whether to include few-shot examples
            use_counterfactual: Whether to include counterfactual examples
            n_few_shot_examples: Number of few-shot examples
            enable_diversity_prompting: Whether to add explicit diversity instructions
            diversity_prompt_strength: Strength of diversity prompting ("low", "medium", "high")
            value_transformer: ValueTransformer for real-world value generation
        """
        self.strategy = strategy
        self.use_chain_of_thought = use_chain_of_thought
        self.use_few_shot = use_few_shot
        self.use_counterfactual = use_counterfactual
        self.n_few_shot_examples = n_few_shot_examples
        self.enable_diversity_prompting = enable_diversity_prompting
        self.diversity_prompt_strength = diversity_prompt_strength
        self.value_transformer = value_transformer
    
    def build_prompt(
        self,
        # Dataset info
        dataset_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        target_class: int = 1,
        
        # Reports from deterministic tools
        dataset_profile: Optional[Any] = None,
        fairness_report: Optional[Any] = None,
        schema_report: Optional[Any] = None,
        diversity_plan: Optional[Any] = None,
        
        # Additional parameters
        sensitive_features: Optional[pd.DataFrame] = None,
        target_group: Optional[Dict[str, Any]] = None,
        iteration: int = 1,
        feedback: Optional[str] = None,
        existing_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Build complete prompt for LLM generation.
        
        Args:
            dataset_name: Name of the dataset
            X_train: Training features
            y_train: Training labels
            n_samples: Number of samples to generate
            target_class: Target class (usually minority)
            dataset_profile: Profile from DatasetProfiler
            fairness_report: Report from FairnessAuditor
            schema_report: Report from SchemaProfiler
            diversity_plan: Plan from DiversityPlanner
            sensitive_features: Sensitive features DataFrame
            target_group: Target group specification
            iteration: Current iteration number
            feedback: Feedback from previous iteration
        
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            dataset_name=dataset_name,
            X_train=X_train,
            y_train=y_train,
            n_samples=n_samples,
            target_class=target_class,
            dataset_profile=dataset_profile,
            fairness_report=fairness_report,
            schema_report=schema_report,
            diversity_plan=diversity_plan,
            sensitive_features=sensitive_features,
            target_group=target_group,
            iteration=iteration,
            feedback=feedback
        )
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt."""
        sections = []
        
        # Base system prompt
        sections.append(BaseTemplates.get_system_prompt(self.strategy))
        
        return "\n\n".join(sections)
    
    def _build_user_prompt(
        self,
        dataset_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        target_class: int,
        dataset_profile: Optional[Any],
        fairness_report: Optional[Any],
        schema_report: Optional[Any],
        diversity_plan: Optional[Any],
        sensitive_features: Optional[pd.DataFrame],
        target_group: Optional[Dict[str, Any]],
        iteration: int,
        feedback: Optional[str],
        existing_samples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build user prompt."""
        sections = []
        
        # 1. Task description
        minority_class_name = "positive class"
        sections.append(BaseTemplates.get_task_description(
            dataset_name=dataset_name,
            n_samples=n_samples,
            target_class=target_class,
            minority_class_name=minority_class_name
        ))
        
        # 2. Diversity instructions (if enabled) - SOTA FEATURE
        if self.enable_diversity_prompting:
            sections.append(self._get_diversity_prompt(self.diversity_prompt_strength))
            
            # IMPROVEMENT #1: Add verbalized sampling prompt for batch diversity
            feature_names = X_train.columns.tolist()
            sections.append(BaseTemplates.get_verbalized_sampling_prompt(
                n_samples=n_samples,
                feature_names=feature_names,
                include_probability=True
            ))
            
            # Add iteration-aware diversity reminder
            sections.append(BaseTemplates.get_diversity_reminder(iteration=iteration))
        
        # 3. Fairness instructions (if fairness-aware strategy)
        if self.strategy in ["FAIRNESS_FIRST", "EXTREME_FAIRNESS"] and fairness_report:
            # Fairness priority
            priority = "high" if self.strategy == "EXTREME_FAIRNESS" else "medium"
            sections.append(FairnessTemplates.get_fairness_priority_prompt(priority))
            
            # Target group specification
            if hasattr(fairness_report, 'fairness_targets'):
                sections.append(FairnessTemplates.get_target_group_specification(
                    fairness_report.fairness_targets,
                    detailed=True
                ))
            
            # Bias mitigation guidelines
            sections.append(FairnessTemplates.get_bias_mitigation_guidelines())
            
            # Fairness-first strategy
            if self.strategy == "FAIRNESS_FIRST":
                sections.append(FairnessTemplates.get_fairness_first_strategy_prompt())
        
        # 4. Counterfactual instructions (if enabled)
        if self.use_counterfactual and target_group:
            sections.append(CounterfactualTemplates.get_counterfactual_explanation())
            
            protected_attrs = list(target_group.keys())
            sections.append(CounterfactualTemplates.get_counterfactual_instructions(
                protected_attrs,
                target_group
            ))
            
            sections.append(CounterfactualTemplates.get_minimal_change_principle())
            
            # Counterfactual strategy
            n_counterfactual = int(n_samples * 0.7)  # 70% counterfactual
            n_novel = n_samples - n_counterfactual
            sections.append(CounterfactualTemplates.get_counterfactual_strategy_prompt(
                n_counterfactual,
                n_novel
            ))
        
        # 4. Constraints
        if schema_report:
            constraints_text = ConstraintEncoder.encode_all_constraints(
                schema_report,
                diversity_plan
            )
            sections.append(constraints_text)
            
            # Add feature variation reminder if we have schema info
            if hasattr(schema_report, 'feature_types') or hasattr(schema_report, 'categorical_features'):
                variation_reminder = """
⚠️  FEATURE VARIATION REQUIREMENT:
Each feature in your generated samples MUST have VARIATION across all {n_samples} samples.

DO NOT:
- Use the same value for a feature across all samples (creates constant columns)
- Repeat the same feature value pattern
- Generate samples where multiple features have constant values

DO:
- Use different values for each feature across your samples
- If a feature has categories, use multiple different categories
- If a feature has a range, use values across the range
- Ensure every feature shows variation in your {n_samples} samples

Example: If generating 20 samples and a feature has 5 possible values, use at least 3-4 different values across the 20 samples, not just 1 value."""
                sections.append(variation_reminder.format(n_samples=n_samples))
            
            # Add real-world value instructions
            # LLM generates values in human-understandable ranges (age=35, hours=40)
            # ValueTransformer normalizes them after generation
            if self.value_transformer:
                sections.append(self._get_real_world_value_instructions())
        
        # 5. Few-shot examples (if enabled) - IMPROVEMENT #2: Dynamic selection
        if self.use_few_shot and len(X_train) > 0:
            # Representative samples with iteration-aware dynamic selection
            # This ensures different examples are shown each iteration to prevent
            # LLM mode collapse from seeing the same patterns repeatedly
            representative_samples = FewShotBuilder.select_representative_samples(
                X_train, y_train,
                target_class=target_class,
                n_samples=self.n_few_shot_examples,
                sensitive_features=sensitive_features,
                target_group=target_group,
                iteration=iteration,  # IMPROVEMENT #2: Pass iteration for dynamic selection
                selection_strategy='mixed'  # Rotate through strategies
            )
            
            if len(representative_samples) > 0:
                # CRITICAL: Denormalize few-shot examples to match the real-world ranges
                # shown in the prompt. This ensures consistency between schema and examples.
                if self.value_transformer and self.value_transformer.is_fitted:
                    representative_samples = self.value_transformer.denormalize_for_prompt(
                        representative_samples
                    )
                
                sections.append(FewShotBuilder.format_examples(
                    representative_samples,
                    include_annotations=True,
                    target_group=target_group
                ))
            
            # Sparse region examples (if diversity plan available)
            if diversity_plan:
                sparse_examples = FewShotBuilder.format_sparse_region_examples(
                    diversity_plan,
                    X_train,
                    n_examples_per_region=2
                )
                if sparse_examples:
                    sections.append(sparse_examples)
            
            # Counterfactual examples (if enabled)
            if self.use_counterfactual and target_group and sensitive_features is not None:
                # Get base samples from majority group
                protected_attr = list(target_group.keys())[0]
                target_value = target_group[protected_attr]
                
                # Find original value (opposite of target)
                if protected_attr in sensitive_features.columns:
                    unique_values = sensitive_features[protected_attr].unique()
                    original_value = [v for v in unique_values if v != target_value][0] if len(unique_values) > 1 else target_value
                    
                    base_samples = X_train[(y_train == target_class) & 
                                          (sensitive_features[protected_attr] == original_value)].head(3)
                    
                    if len(base_samples) > 0:
                        cf_examples = FewShotBuilder.format_counterfactual_examples(
                            base_samples,
                            protected_attr,
                            original_value,
                            target_value,
                            n_examples=2
                        )
                        sections.append(cf_examples)
        
        # 6. Chain-of-thought (if enabled)
        if self.use_chain_of_thought:
            sections.append(BaseTemplates.get_chain_of_thought_prompt())
        
        # 7. Output format
        feature_names = X_train.columns.tolist()
        sections.append(BaseTemplates.get_output_format(
            feature_names,
            include_validation=True
        ))
        
        # 8. Fairness validation checklist (if fairness-aware)
        if self.strategy in ["FAIRNESS_FIRST", "EXTREME_FAIRNESS"] and fairness_report:
            if hasattr(fairness_report, 'fairness_targets'):
                sections.append(FairnessTemplates.get_fairness_validation_checklist(
                    fairness_report.fairness_targets
                ))
        
        # 9. Counterfactual validation (if enabled)
        if self.use_counterfactual:
            sections.append(CounterfactualTemplates.get_counterfactual_validation_checklist())
        
        # 10. General instructions
        sections.append(BaseTemplates.get_general_instructions())
        
        # 11. Feedback from previous iteration (if any)
        if iteration > 1 and feedback:
            sections.append(f"""FEEDBACK FROM PREVIOUS ITERATION:

{feedback}

Please address this feedback in your current generation.""")
        
        # 12. Error handling
        sections.append(BaseTemplates.get_error_handling_instructions())
        
        # 13. Diversity reminder (if existing samples provided)
        if existing_samples and len(existing_samples) > 0:
            diversity_reminder = f"""
⚠️  DIVERSITY REQUIREMENT - CRITICAL:
You have already generated {len(existing_samples)} samples. The samples you generate now MUST be DIFFERENT from all previous samples.

DO NOT generate samples that are:
- Exact duplicates of previous samples
- Near-duplicates (same values for most features)
- Samples with constant feature values

ENSURE DIVERSITY:
- Vary ALL features across your {n_samples} samples
- Make each sample unique and distinct
- Cover different combinations of feature values
- Avoid repeating patterns from previous samples

Remember: Diversity is as important as quality. Generate {n_samples} UNIQUE samples."""
            sections.append(diversity_reminder)
        
        # 14. FINAL REMINDER: Output format + validation checklist (critical - placed at end)
        sections.append(f"""🚨 FINAL REMINDER - READ THIS CAREFULLY 🚨

You MUST generate EXACTLY {n_samples} samples. Returning an empty array ({{"samples":[]}}) is NOT acceptable.

Your response MUST be ONLY a valid JSON array with {n_samples} sample objects. Nothing else.

DO NOT include:
- Any reasoning or thinking process
- Any explanations or commentary
- Any markdown formatting
- Any text before or after the JSON
- An empty array (you MUST generate {n_samples} samples)
- Samples with constant feature values

Your response should look EXACTLY like this:
[{{"field1": value1, "field2": value2, ...}}, {{"field1": value1, ...}}, ... ({n_samples} total objects)]

That's it. Just the JSON array with {n_samples} samples. Nothing more, nothing less.

⚠️  REMEMBER: Generate {n_samples} samples. Do NOT return an empty array.

🚨 ANTI-DUPLICATION REQUIREMENT (CRITICAL):
Before returning, verify each sample is SIGNIFICANTLY DIFFERENT from all others:
1. Pick any 2 samples from your {n_samples} samples
2. Count how many features have DIFFERENT values between them
3. If < 50% of features differ, samples are TOO SIMILAR - regenerate one
4. Repeat for multiple sample pairs to ensure diversity

Example GOOD diversity (20 features, comparing 2 samples):
✅ Sample 1: [0.2, 0.5, 0.8, 0.3, 0.7, 0.4, 0.9, 0.1, 0.6, 0.2, ...]
✅ Sample 2: [0.8, 0.1, 0.3, 0.9, 0.2, 0.7, 0.1, 0.8, 0.3, 0.6, ...]
→ 15+ features differ (75% different) → PASS

Example BAD diversity (20 features, comparing 2 samples):
❌ Sample 1: [0.5, 0.5, 0.5, 0.3, 0.7, 0.4, 0.9, 0.1, 0.6, 0.2, ...]
❌ Sample 2: [0.5, 0.5, 0.5, 0.3, 0.8, 0.4, 0.9, 0.1, 0.7, 0.2, ...]
→ Only 3 features differ (15% different) → FAIL → REGENERATE

🎯 VALIDATION CHECKLIST (Verify BEFORE returning):
✓ Array contains EXACTLY {n_samples} objects
✓ Each object has ALL required features
✓ All values are WITHIN valid ranges/categories
✓ NO two objects are identical (exact duplicates)
✓ EVERY pair of samples differs in at least 50% of features (Gower distance > 0.15)
✓ EVERY feature varies across objects (no constant columns)
✓ Categorical features: Multiple categories used (not just one)
✓ Numerical features: Values span full range (low/medium/high)
✓ Maximum diversity achieved (samples explore different regions)

If ANY checklist item fails, FIX IT before returning.""")
        
        return "\n\n".join(sections)
    
    def get_prompt_summary(self, prompt: Dict[str, str]) -> str:
        """
        Get a summary of the prompt (for logging/debugging).
        
        Args:
            prompt: Prompt dictionary
        
        Returns:
            Summary string
        """
        system_len = len(prompt['system'])
        user_len = len(prompt['user'])
        total_len = system_len + user_len
        
        return f"""Prompt Summary:
- Strategy: {self.strategy}
- System prompt: {system_len} chars
- User prompt: {user_len} chars
- Total: {total_len} chars
- Chain-of-thought: {self.use_chain_of_thought}
- Few-shot: {self.use_few_shot}
- Counterfactual: {self.use_counterfactual}"""
    
    def _get_diversity_prompt(self, strength: str = "medium") -> str:
        """
        Generate diversity-focused prompt instructions.
        
        Args:
            strength: Strength of diversity emphasis ("low", "medium", "high")
        
        Returns:
            Diversity prompt text
        """
        if strength == "low":
            return """
## 🎨 Diversity Guidelines

Generate samples that vary across different feature values. Avoid creating similar samples.
"""
        elif strength == "medium":
            return """
## 🎨 DIVERSITY REQUIREMENTS (IMPORTANT)

**Primary Goal: MAXIMIZE DIVERSITY**

Your generated samples MUST be diverse across ALL features:

1. **Feature Variation**: Each sample should have different feature values
   - Vary numerical features across their full range
   - Use different categorical values for each sample
   - Avoid clustering around common values

2. **Avoid Similarity**: Do NOT generate samples that are similar to each other
   - Each sample should be unique and distinct
   - Spread samples across the entire feature space
   - Explore edge cases and rare combinations

3. **Balance Exploration**: 
   - 50% of samples: Explore underrepresented regions
   - 50% of samples: Novel, creative combinations

**Remember: Diversity is MORE important than similarity to training data!**
"""
        else:  # high
            return """
## 🎨 CRITICAL: MAXIMUM DIVERSITY REQUIREMENT

**⚠️  PRIMARY OBJECTIVE: GENERATE HIGHLY DIVERSE SAMPLES ⚠️**

This is your MOST IMPORTANT task. Diversity is the #1 priority.

### 🚫 ANTI-DUPLICATE RULES (MANDATORY):

**YOU WILL BE REJECTED IF YOUR SAMPLES ARE TOO SIMILAR!**

- Each sample MUST be < 95% similar to any other sample
- Vary AT LEAST 50% of feature values between consecutive samples
- Use DIFFERENT combinations of categorical values (age groups, occupations, etc.)
- Spread numerical values across the FULL range (min, median, max, quartiles)
- **NEVER copy patterns from previous samples**

### Mandatory Diversity Rules:

1. **FEATURE SPACE EXPLORATION** (Required)
   - Spread samples across the ENTIRE feature range
   - Use MIN, MAX, and intermediate values for numerical features
   - Use ALL possible categories, not just common ones
   - Create UNUSUAL but valid feature combinations
   - **Rotate through different value ranges** (low → mid → high → extreme)

2. **ANTI-SIMILARITY ENFORCEMENT** (Required)
   - Each sample MUST be significantly different from others
   - Avoid generating samples with similar patterns
   - If you notice similarity, STOP and generate something completely different
   - Think: "How can I make this sample as different as possible?"
   - **Track which combinations you've used and avoid repeating them**

3. **DIVERSITY METRICS** (You will be evaluated on):
   - Feature variance: Samples should have high variance across features
   - Inter-sample distance: Samples should be far apart in feature space
   - Coverage: Samples should cover different regions of the feature space
   - **Duplicate rate: < 5% duplicates (you WILL be penalized for high duplicate rates!)**

4. **CREATIVE EXPLORATION** (Encouraged):
   - Generate samples in underexplored regions
   - Create rare but valid combinations
   - Push boundaries while respecting constraints
   - Be creative and bold with feature values
   - **Mix common and rare values in unexpected ways**

5. **SYSTEMATIC VARIATION** (New):
   - For each batch, divide samples into groups:
     * 25% with low numerical values
     * 25% with medium numerical values  
     * 25% with high numerical values
     * 25% with extreme/edge case values
   - Rotate through ALL categorical options systematically
   - **Never generate 2 consecutive samples with similar profiles**

### Diversity Checklist (Before Generating EACH sample):
- [ ] Are my samples spread across the full feature range?
- [ ] Did I use diverse categorical values?
- [ ] Are my samples significantly different from each other?
- [ ] Did I explore edge cases and rare combinations?
- [ ] **Did I check that this sample is NOT similar to previous ones?**
- [ ] **Did I vary at least 50% of features from the last sample?**

**REMEMBER: A few highly diverse samples are MORE valuable than many similar samples!**
**Your success will be measured by diversity metrics, not quantity!**
**DUPLICATES WILL BE REJECTED - Focus on quality over quantity!**
"""
    
    def _get_real_world_value_instructions(self) -> str:
        """
        Get instructions for generating STATISTICALLY REPRESENTATIVE values.
        
        This is the PROBABILITY-DRIVEN approach where:
        1. LLM generates values that MATCH the training distribution
        2. Values are in human-understandable ranges (age=35, hours=40)
        3. ValueTransformer normalizes them after generation
        
        Key insight: Random oversampling works because duplicates are 
        statistically perfect. LLM samples should also match the distribution.
        
        Returns:
            Instructions string with distribution statistics
        """
        if not self.value_transformer or not self.value_transformer.is_fitted:
            return ""
        
        lines = [
            "📊 DISTRIBUTION-MATCHING GENERATION (CRITICAL)",
            "",
            "🎯 YOUR GOAL: Generate samples that MATCH the training data distribution.",
            "DO NOT explore extremes or generate unusual combinations.",
            "Generate samples that look like they came from the SAME distribution as the training data.",
            ""
        ]
        
        # Get distribution statistics
        stats = self.value_transformer.get_distribution_statistics()
        ranges = self.value_transformer.get_real_world_ranges()
        
        # Group by type
        continuous_features = []
        categorical_features = []
        
        for name, info in stats.items():
            if info['type'] == 'continuous':
                continuous_features.append({
                    'name': name,
                    'min': info['min'],
                    'max': info['max'],
                    'p10': info['p10'],
                    'p25': info['p25'],
                    'p50': info['p50'],  # median
                    'p75': info['p75'],
                    'p90': info['p90'],
                })
            else:
                cats = info['categories'][:8]
                freqs = info.get('frequencies', {})
                categorical_features.append({
                    'name': name, 
                    'categories': cats,
                    'frequencies': freqs
                })
        
        # CONTINUOUS FEATURES - Distribution matching
        if continuous_features:
            lines.append("📈 CONTINUOUS FEATURES - MATCH THESE DISTRIBUTIONS:")
            lines.append("")
            for f in continuous_features[:6]:  # Top 6 continuous features
                lines.append(f"  {f['name']}:")
                lines.append(f"    Range: {f['min']:.0f} to {f['max']:.0f}")
                lines.append(f"    Distribution: 10th={f['p10']:.0f}, 25th={f['p25']:.0f}, MEDIAN={f['p50']:.0f}, 75th={f['p75']:.0f}, 90th={f['p90']:.0f}")
                lines.append(f"    → ~50% of samples should be between {f['p25']:.0f} and {f['p75']:.0f}")
                lines.append(f"    → ~80% of samples should be between {f['p10']:.0f} and {f['p90']:.0f}")
                lines.append(f"    → MOST COMMON value: around {f['p50']:.0f} (median)")
                lines.append("")
            
            lines.append("  🎯 DISTRIBUTION MATCHING RULES:")
            lines.append("    • Generate MOST values near the median (center of distribution)")
            lines.append("    • Generate FEWER values near the extremes")
            lines.append("    • DO NOT uniformly spread values across the range")
            lines.append("    • Think: 'What would a TYPICAL sample look like?'")
            lines.append("")
        
        # CATEGORICAL FEATURES - Frequency matching
        if categorical_features:
            lines.append("📋 CATEGORICAL FEATURES - MATCH THESE FREQUENCIES:")
            lines.append("")
            for f in categorical_features[:6]:
                cats_str = ", ".join(map(str, f['categories']))
                lines.append(f"  {f['name']}: {{{cats_str}}}")
                if f['frequencies']:
                    # Show top 3 most frequent categories
                    sorted_freqs = sorted(f['frequencies'].items(), key=lambda x: x[1], reverse=True)[:3]
                    freq_str = ", ".join([f"{cat}:{freq*100:.0f}%" for cat, freq in sorted_freqs])
                    lines.append(f"    Most common: {freq_str}")
                lines.append("")
            
            lines.append("  🎯 FREQUENCY MATCHING RULES:")
            lines.append("    • Use common categories MORE OFTEN")
            lines.append("    • Use rare categories LESS OFTEN")
            lines.append("    • Match the approximate proportions from training data")
            lines.append("")
        
        # VALIDATION RULES
        lines.extend([
            "🚨 VALIDATION RULES (samples will be REJECTED if violated):",
            "",
            "  ⛔ REJECTION TRIGGERS:",
            "    • Values outside min-max range",
            "    • Normalized 0-1 values when real-world expected",
            "    • Statistical outliers (>4σ from mean)",
            "",
            "  ✅ ACCEPTANCE CRITERIA:",
            "    • Values within valid range",
            "    • Distribution matches training data",
            "    • No extreme outliers",
            "",
        ])
        
        # EXAMPLES
        if continuous_features:
            f = continuous_features[0]
            lines.extend([
                f"📝 EXAMPLE for {f['name']}:",
                f"   ❌ WRONG (uniform spread): {f['min']:.0f}, {f['max']:.0f}, {f['min']:.0f}, {f['max']:.0f}",
                f"   ❌ WRONG (all same): {f['p50']:.0f}, {f['p50']:.0f}, {f['p50']:.0f}, {f['p50']:.0f}",
                f"   ✅ CORRECT (distribution-matching): {f['p50']:.0f}, {f['p25']:.0f}, {f['p75']:.0f}, {f['p50']:.0f}",
                f"      (most values near median, some variation)",
                ""
            ])
        
        lines.extend([
            "⚡ THINK LIKE THIS:",
            "  'What does a TYPICAL sample from this dataset look like?'",
            "  'What values are MOST COMMON in the training data?'",
            "  'Generate samples that could have come from the same source.'",
        ])
        
        return "\n".join(lines)
    
    def set_value_transformer(self, transformer: Any) -> None:
        """
        Set the ValueTransformer for real-world value generation.
        
        Args:
            transformer: Fitted ValueTransformer instance
        """
        self.value_transformer = transformer


if __name__ == "__main__":
    # Test prompt builder
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.dataset_profiler import DatasetProfiler
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    from src.qualsynth.modules.schema_profiler import SchemaProfiler
    from src.qualsynth.modules.diversity_planner import DiversityPlanner
    
    print("="*70)
    print("Testing Prompt Builder")
    print("="*70)
    
    # Load German Credit dataset
    dataset_name = 'german_credit'
    sensitive_cols = ['race', 'sex']
    
    split_data = load_split(dataset_name, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
    if available_sensitive_cols:
        sensitive_features = X_train[available_sensitive_cols]
    
    # Run all deterministic tools
    profiler = DatasetProfiler()
    dataset_profile = profiler.profile(X_train, y_train, sensitive_features, dataset_name)
    
    auditor = FairnessAuditor(fairness_threshold=0.05)
    fairness_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
    
    schema_profiler = SchemaProfiler()
    schema_report = schema_profiler.profile(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=fairness_report.fairness_targets,
        dataset_name=dataset_name
    )
    
    planner = DiversityPlanner()
    diversity_plan = planner.plan(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=fairness_report.fairness_targets,
        dataset_name=dataset_name
    )
    
    # Test 1: Standard prompt
    print("\n\nTEST 1: Standard Prompt")
    print("-"*70)
    
    builder1 = PromptBuilder(
        strategy="STANDARD",
        use_chain_of_thought=True,
        use_few_shot=True,
        use_counterfactual=False
    )
    
    prompt1 = builder1.build_prompt(
        dataset_name=dataset_name,
        X_train=X_train,
        y_train=y_train,
        n_samples=100,
        target_class=1,
        dataset_profile=dataset_profile,
        fairness_report=fairness_report,
        schema_report=schema_report,
        diversity_plan=diversity_plan,
        sensitive_features=sensitive_features,
        target_group={'sex': 0}
    )
    
    print(builder1.get_prompt_summary(prompt1))
    print("\nSystem Prompt (first 500 chars):")
    print(prompt1['system'][:500], "...")
    print("\nUser Prompt (first 1000 chars):")
    print(prompt1['user'][:1000], "...")
    
    # Test 2: Fairness-first prompt
    print("\n\nTEST 2: Fairness-First Prompt")
    print("-"*70)
    
    builder2 = PromptBuilder(
        strategy="FAIRNESS_FIRST",
        use_chain_of_thought=True,
        use_few_shot=True,
        use_counterfactual=True
    )
    
    prompt2 = builder2.build_prompt(
        dataset_name=dataset_name,
        X_train=X_train,
        y_train=y_train,
        n_samples=100,
        target_class=1,
        dataset_profile=dataset_profile,
        fairness_report=fairness_report,
        schema_report=schema_report,
        diversity_plan=diversity_plan,
        sensitive_features=sensitive_features,
        target_group={'sex': 0}
    )
    
    print(builder2.get_prompt_summary(prompt2))
    print("\nSystem Prompt (first 500 chars):")
    print(prompt2['system'][:500], "...")
    print("\nUser Prompt (first 1000 chars):")
    print(prompt2['user'][:1000], "...")
    
    # Test 3: Save full prompt to file
    print("\n\nTEST 3: Save Full Prompt to File")
    print("-"*70)
    
    with open('test_prompt_full.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("SYSTEM PROMPT\n")
        f.write("="*70 + "\n\n")
        f.write(prompt2['system'])
        f.write("\n\n" + "="*70 + "\n")
        f.write("USER PROMPT\n")
        f.write("="*70 + "\n\n")
        f.write(prompt2['user'])
    
    print("Full prompt saved to test_prompt_full.txt")
    
    print("\n\n✅ Prompt Builder Test Complete")

