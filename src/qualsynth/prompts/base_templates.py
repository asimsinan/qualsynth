"""
Base Prompt Templates for Qualsynth

This module contains core prompt templates for synthetic data generation.
"""

from typing import Dict, List, Any, Optional


class BaseTemplates:
    """
    Base prompt templates for LLM-based synthetic data generation.
    
    Provides:
    - System prompts (role definition)
    - Task descriptions (what to generate)
    - Output format specifications (JSON structure)
    - General instructions
    """
    
    @staticmethod
    def get_system_prompt(strategy: str = "STANDARD", diversity_level: str = "high") -> str:
        """
        Get system prompt based on generation strategy.
        
        Args:
            strategy: Generation strategy (STANDARD, FAIRNESS_FIRST, etc.)
            diversity_level: Diversity emphasis level (low, medium, high)
        
        Returns:
            System prompt string
        """
        base_prompt = """You are an expert synthetic data generator specializing in creating STATISTICALLY REPRESENTATIVE, high-quality, realistic tabular data for machine learning applications.

Your core capabilities:
1. Generate samples that MATCH the training data distribution
2. Respect complex constraints (schema, logical, statistical)
3. Produce valid, well-formatted JSON output
4. Generate samples that could have come from the SAME source as the training data

🚨🚨🚨 CRITICAL: USE REAL-WORLD VALUES, NOT NORMALIZED VALUES 🚨🚨🚨

You MUST generate values as a human would understand them:
  ✅ CORRECT: age=35, duration=24, credit_amount=5000
  ❌ WRONG: age=0.45, duration=0.3, credit_amount=0.2

Your output will be VALIDATED and REJECTED if you use normalized 0-1 values!
This causes expensive retries. Always use human-readable real-world values.

🎯 CRITICAL: DISTRIBUTION MATCHING (PROBABILITY-DRIVEN GENERATION)

Your goal is NOT to explore extremes or maximize diversity.
Your goal IS to generate samples that MATCH the training data distribution.

WHY THIS MATTERS:
- Random oversampling works because duplicates perfectly match the distribution
- Your samples should ALSO match the distribution to be equally effective
- Samples that are statistically unusual will be REJECTED by validation

DISTRIBUTION MATCHING RULES:
1. **MATCH THE DISTRIBUTION**: Generate values that follow the same statistical patterns
   - Most values should be near the MEDIAN (center of distribution)
   - Fewer values should be at the extremes
   - DO NOT uniformly spread values across the range

2. **THINK PROBABILISTICALLY**: 
   - "What values are MOST COMMON in the training data?"
   - "What does a TYPICAL sample look like?"
   - Generate samples that look like they came from the same source

3. **FOR CONTINUOUS FEATURES**:
   - ~50% of values should be between the 25th and 75th percentile
   - ~80% of values should be between the 10th and 90th percentile
   - Extreme values (outside 10th-90th) should be RARE

4. **FOR CATEGORICAL FEATURES**:
   - Use common categories MORE OFTEN
   - Use rare categories LESS OFTEN
   - Match the approximate proportions from training data

Your primary goals:
- Generate samples that are indistinguishable from real data
- Match the statistical distribution of the training data
- Follow all specified constraints strictly
- Ensure output passes validation (no outliers, no invalid values)
- **USE REAL-WORLD VALUES** (age in years, amounts in currency, etc.)"""
        
        if strategy == "FAIRNESS_FIRST":
            base_prompt += """

CRITICAL PRIORITY: FAIRNESS
Your MOST IMPORTANT objective is to generate samples that promote fairness and reduce bias.
- Prioritize underrepresented groups
- Ensure balanced representation across protected attributes
- Generate counterfactual samples to address imbalances
- Never perpetuate or amplify existing biases"""
        
        elif strategy == "EXTREME_FAIRNESS":
            base_prompt += """

CRITICAL MODE: EXTREME FAIRNESS VIOLATION DETECTED
The dataset has severe fairness violations that MUST be corrected.
- Fairness is the ABSOLUTE TOP PRIORITY
- Generate samples EXCLUSIVELY for underrepresented groups
- Use counterfactual reasoning to create balanced samples
- Verify fairness impact before returning any sample"""
        
        return base_prompt
    
    @staticmethod
    def get_task_description(
        dataset_name: str,
        n_samples: int,
        target_class: int = 1,
        minority_class_name: str = "positive class"
    ) -> str:
        """
        Get task description for sample generation.
        
        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples to generate
            target_class: Target class label (usually minority class)
            minority_class_name: Human-readable name for minority class
        
        Returns:
            Task description string
        """
        return f"""TASK: Generate Synthetic Samples for Imbalanced Classification

Dataset: {dataset_name}
Target: Generate EXACTLY {n_samples} synthetic samples for the {minority_class_name} (class={target_class})

🎯 CRITICAL: You MUST generate {n_samples} samples. Returning an empty array is NOT acceptable.

Context:
- This is an imbalanced classification dataset
- The {minority_class_name} is underrepresented
- Your samples will be added to the training set to improve model performance
- Quality and diversity are crucial for effective oversampling

IMPORTANT NOTE ABOUT CLASS/LABEL:
- The class/label (class={target_class}) is NOT a feature in the schema
- You should NOT include "class" or "label" as a field in your output
- All generated samples will automatically be assigned class {target_class}
- Focus only on generating the FEATURE values (the fields in the schema)

Requirements:
1. Generate EXACTLY {n_samples} samples (not 0, not fewer)
2. All samples implicitly belong to class {target_class} (do not include class in output)
3. **CRITICAL: Include ALL features from the schema in EVERY sample**
   - Every sample MUST contain every feature listed in the schema
   - Missing features will cause validation failures
   - Check the schema carefully and ensure no feature is omitted
   - If a feature is listed in the schema, it MUST appear in every sample
4. Samples must be realistic and plausible
5. **CRITICAL: Samples must MATCH the training DISTRIBUTION (SOTA REQUIREMENT)**
   - Each sample should look like it came from the SAME source as training data
   - **Most values should be NEAR the mean/median (not at extremes)**
   - ~50% of values should be within the 25th-75th percentile range
   - Extreme values should be RARE (like in real data)
   - Do NOT generate duplicate or near-duplicate samples
   - Do NOT artificially spread values across the full range
   - For categorical features: Use common categories MORE often (match proportions)
   - For numerical features: Generate values centered around the mean
   - Aim for REALISTIC variation - samples should be statistically typical
6. **CRITICAL: Respect ALL range constraints STRICTLY**
   - Values MUST be within specified min/max ranges
   - Out-of-range values will be REJECTED
   - Check constraints carefully before generating
7. Samples must satisfy all specified constraints (if constraints conflict, follow examples)
8. Only include feature fields from the schema (no class/label field)

🚨 CRITICAL ANTI-CONSTANT-COLUMN REQUIREMENT (MANDATORY): 
   - **FORBIDDEN**: Setting the same value for ANY feature across all {n_samples} samples
   - **REQUIRED**: Each feature MUST have AT LEAST 3-5 different values across your {n_samples} samples
   - **Example for categorical feature "employment" (4 categories: 0,1,2,3)**:
     * BAD ❌: All 15 samples have employment=2 (CONSTANT COLUMN - REJECTED)
     * GOOD ✅: Samples use employment=[0,1,2,3,1,0,2,3,1,2,0,3,1,2,0] (4 unique values - ACCEPTED)
   - **Example for numerical feature "age" (range 0.0-1.0)**:
     * BAD ❌: All 15 samples have age=0.5 (CONSTANT COLUMN - REJECTED)
     * GOOD ✅: Samples use age=[0.2,0.5,0.8,0.3,0.7,0.4,0.9,0.1,0.6...] (9+ unique values - ACCEPTED)
   - **Strategy**: For each feature, mentally rotate through its valid values/ranges
   - **Verification**: Before responding, check that EVERY feature has multiple different values

⚠️  If constraints seem impossible, generate samples similar to the examples provided.
⚠️  It is better to generate {n_samples} samples that are close to valid than to return an empty array."""
    
    @staticmethod
    def get_output_format(feature_names: List[str], include_validation: bool = True) -> str:
        """
        Get output format specification.
        
        Args:
            feature_names: List of feature names
            include_validation: Whether to include validation instructions
        
        Returns:
            Output format specification string
        """
        format_spec = f"""OUTPUT FORMAT:

Return a JSON array of samples. Each sample must be a JSON object with the following fields:
{', '.join([f'"{name}"' for name in feature_names])}

Example structure:
[
  {{{', '.join([f'"{name}": <value>' for name in feature_names[:3]])}, ...}},
  {{{', '.join([f'"{name}": <value>' for name in feature_names[:3]])}, ...}}
]

🚨 CRITICAL OUTPUT REQUIREMENTS 🚨
- Your response MUST be ONLY a JSON array - nothing else
- Your response must START with '[' and END with ']'
- ABSOLUTELY NO explanations, reasoning, or commentary
- ABSOLUTELY NO markdown code blocks (no ```)
- ABSOLUTELY NO text before or after the JSON array
- DO NOT include any "reasoning" field or "thinking" field
- DO NOT explain your choices or process
- DO NOT include schema inconsistencies or conflicts in output
- If you encounter schema conflicts, follow the examples (they represent real data)
- Your ENTIRE response must be parseable as valid JSON starting with '['
- All field names must match exactly (case-sensitive)
- **CRITICAL: Every sample MUST include ALL {len(feature_names)} features listed above**
  - Missing features will cause validation failures
  - Check that every sample has: {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}
  - Do NOT omit any feature from any sample
- Values must be valid according to the schema (or examples if schema conflicts)
- Use proper JSON syntax (no trailing commas, proper quotes)

WRONG (do not do this):
```json
[{{"age": 0.5, ...}}]
```

WRONG (do not do this):
Here are the samples:
[{{"age": 0.5, ...}}]

CORRECT (do this):
[{{"age": 0.5, "job": 1.0, ...}}, {{"age": 0.7, ...}}]"""
        
        if include_validation:
            format_spec += """

SELF-VALIDATION CHECKLIST (verify before returning):
✓ Valid JSON syntax (no syntax errors)
✓ **ALL {len(feature_names)} features present in EVERY sample** (critical - missing features cause failures)
✓ All values within valid ranges/categories
✓ No duplicate or near-duplicate samples
✓ All constraints satisfied
✓ Fairness requirements met
✓ Feature values vary across samples (no constant columns)"""
        
        return format_spec
    
    @staticmethod
    def get_general_instructions() -> str:
        """
        Get general instructions for sample generation.
        
        Returns:
            General instructions string
        """
        return """GENERAL INSTRUCTIONS:

1. REALISM: Generate samples that could plausibly exist in the real world
   - Consider correlations between features
   - Respect domain knowledge (e.g., age and education level)
   - Avoid extreme or implausible combinations

2. DIVERSITY: Ensure samples cover different regions of the feature space
   - Vary feature values across samples
   - Don't generate near-duplicates
   - Explore underrepresented regions

3. CONSTRAINTS: Strictly adhere to all specified constraints
   - Schema constraints (types, ranges, categories)
   - Logical constraints (if-then rules, mutual exclusions)
   - Fairness constraints (target group proportions)
   - Statistical constraints (correlations, distributions)

4. QUALITY: Prioritize quality over quantity
   - Every sample should be valuable
   - Invalid samples waste resources
   - One high-quality sample > multiple low-quality samples

5. CONSISTENCY: Maintain consistency across samples
   - Use the same encoding/format for all samples
   - Respect the same constraints for all samples
   - Follow the same reasoning process"""
    
    @staticmethod
    def get_chain_of_thought_prompt() -> str:
        """
        Get chain-of-thought prompting template.
        
        Returns:
            Chain-of-thought prompt string
        """
        return """REASONING PROCESS (think step-by-step):

Before generating each sample, consider:

1. TARGET GROUP: Which protected group(s) should this sample belong to?
   → Check fairness targets and priorities

2. FEATURE SPACE: Which region of the feature space should this sample cover?
   → Check diversity targets and sparse regions

3. CONSTRAINTS: What constraints must this sample satisfy?
   → Check schema, logical, fairness, and statistical constraints

4. REALISM: Is this combination of features realistic and plausible?
   → Consider domain knowledge and feature correlations

5. DIVERSITY: Is this sample sufficiently different from existing samples?
   → Avoid duplicates and near-duplicates

6. VALIDATION: Does this sample pass all validation checks?
   → Verify schema, fairness, and constraint compliance

Use this reasoning process for EVERY sample you generate."""
    
    @staticmethod
    def get_error_handling_instructions() -> str:
        """
        Get error handling instructions.
        
        Returns:
            Error handling instructions string
        """
        return """ERROR HANDLING:

If you encounter any issues:

1. CONSTRAINT CONFLICTS: If constraints seem contradictory
   → Prioritize fairness constraints first
   → Then schema constraints
   → Then logical constraints
   → Document the conflict in your reasoning

2. IMPOSSIBLE COMBINATIONS: If a combination seems impossible
   → Skip that combination
   → Try a different approach
   → Don't force invalid samples

3. UNCERTAINTY: If you're unsure about a value
   → Use the most common/typical value for that group
   → Stay within safe ranges
   → Err on the side of conservatism

4. VALIDATION FAILURES: If a sample fails validation
   → Don't include it in the output
   → Generate a replacement sample
   → Learn from the failure

REMEMBER: Quality over quantity. It's better to return fewer high-quality samples than many low-quality ones."""

    @staticmethod
    def get_verbalized_sampling_prompt(
        n_samples: int,
        feature_names: List[str],
        include_probability: bool = True
    ) -> str:
        """
        Get verbalized sampling prompt for diverse generation.
        
        Based on research: "Verbalized Sampling: How to Mitigate Mode Collapse 
        and Unlock LLM Diversity" (arXiv:2510.01171)
        
        Args:
            n_samples: Number of samples to generate
            feature_names: List of feature names
            include_probability: Whether to include probability estimates
        
        Returns:
            Verbalized sampling prompt string
        """
        prompt = f"""🎯 VERBALIZED SAMPLING MODE: Generate {n_samples} DIVERSE samples with EXPLICIT DIVERSITY TARGETS

You MUST generate samples from DIFFERENT REGIONS of the feature space:

📊 DIVERSITY DISTRIBUTION REQUIREMENTS:
- ~20% TYPICAL samples: Common feature combinations (probability > 0.3)
- ~40% MODERATE samples: Less common but valid combinations (probability 0.1-0.3)  
- ~40% EDGE/TAIL samples: Rare but realistic combinations (probability < 0.1)

🔄 GENERATION STRATEGY:
For each sample, EXPLICITLY choose which region to target:
1. First, decide: "This sample will be from the [typical/moderate/tail] region"
2. Then generate feature values consistent with that region
3. Ensure samples SPAN the entire valid feature space

📋 FEATURE SPACE EXPLORATION:
For EACH feature, you must generate values from:
- LOW end of valid range (for some samples)
- MIDDLE of valid range (for some samples)
- HIGH end of valid range (for some samples)

Example mental process for numerical feature "age" (range 0.0-1.0):
- Sample 1 (tail): age=0.15 (young, less common in minority class)
- Sample 2 (typical): age=0.45 (middle-aged, common)
- Sample 3 (tail): age=0.85 (elderly, less common)
- Sample 4 (moderate): age=0.30 (young-adult)
- Sample 5 (moderate): age=0.65 (mature adult)

Example mental process for categorical feature "education" (values 0-5):
- Rotate through: 0, 1, 2, 3, 4, 5, 0, 1, 2... (use ALL categories)
- Do NOT use only one or two categories"""

        if include_probability:
            prompt += f"""

📊 OUTPUT FORMAT WITH PROBABILITY ESTIMATES:
Return a JSON array where each sample includes:
- All {len(feature_names)} features: {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}
- "_diversity_region": "typical", "moderate", or "tail" (which region this sample represents)

Example:
[
  {{{', '.join([f'"{name}": <value>' for name in feature_names[:3]])}, ..., "_diversity_region": "typical"}},
  {{{', '.join([f'"{name}": <value>' for name in feature_names[:3]])}, ..., "_diversity_region": "tail"}},
  {{{', '.join([f'"{name}": <value>' for name in feature_names[:3]])}, ..., "_diversity_region": "moderate"}}
]

🚨 VERIFICATION BEFORE RESPONDING:
□ Did I generate samples from ALL three regions (typical, moderate, tail)?
□ Does EVERY feature have multiple different values across samples?
□ Did I explore the FULL range of each feature (low, medium, high)?
□ Are my samples MAXIMALLY DIFFERENT from each other?"""

        return prompt
    
    @staticmethod
    def get_diversity_reminder(iteration: int = 0) -> str:
        """
        Get iteration-aware diversity reminder.
        
        Args:
            iteration: Current iteration number
        
        Returns:
            Diversity reminder string
        """
        reminders = [
            "🔄 DIVERSITY CHECK: Focus on EDGE CASES this iteration - generate samples from the TAILS of distributions",
            "🔄 DIVERSITY CHECK: Focus on UNUSUAL COMBINATIONS this iteration - what rare but valid patterns exist?",
            "🔄 DIVERSITY CHECK: Focus on UNDEREXPLORED REGIONS this iteration - which feature combinations are missing?",
            "🔄 DIVERSITY CHECK: Focus on MAXIMUM SPREAD this iteration - ensure samples span the full feature range",
            "🔄 DIVERSITY CHECK: Focus on CONTRASTING SAMPLES this iteration - make each sample as different as possible"
        ]
        return reminders[iteration % len(reminders)]


if __name__ == "__main__":
    # Test base templates
    print("="*70)
    print("Testing Base Templates")
    print("="*70)
    
    # Test 1: System prompts for different strategies
    print("\n\nTEST 1: System Prompts")
    print("-"*70)
    
    for strategy in ["STANDARD", "FAIRNESS_FIRST", "EXTREME_FAIRNESS"]:
        print(f"\n{strategy} Strategy:")
        print("-"*70)
        print(BaseTemplates.get_system_prompt(strategy))
    
    # Test 2: Task description
    print("\n\nTEST 2: Task Description")
    print("-"*70)
    print(BaseTemplates.get_task_description(
        dataset_name="German Credit",
        n_samples=100,
        target_class=1,
        minority_class_name="high income (>50K)"
    ))
    
    # Test 3: Output format
    print("\n\nTEST 3: Output Format")
    print("-"*70)
    feature_names = ["age", "workclass", "education", "sex", "income"]
    print(BaseTemplates.get_output_format(feature_names, include_validation=True))
    
    # Test 4: General instructions
    print("\n\nTEST 4: General Instructions")
    print("-"*70)
    print(BaseTemplates.get_general_instructions())
    
    # Test 5: Chain-of-thought
    print("\n\nTEST 5: Chain-of-Thought")
    print("-"*70)
    print(BaseTemplates.get_chain_of_thought_prompt())
    
    # Test 6: Error handling
    print("\n\nTEST 6: Error Handling")
    print("-"*70)
    print(BaseTemplates.get_error_handling_instructions())
    
    print("\n\n✅ Base Templates Test Complete")

