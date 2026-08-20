"""
Counterfactual Reasoning Templates for Qualsynth

This module contains counterfactual generation prompting strategies.
"""

from typing import Dict, List, Any, Optional, Tuple


class CounterfactualTemplates:
    """
    Counterfactual reasoning templates for fairness-aware generation.
    
    Provides:
    - Counterfactual reasoning instructions
    - Protected attribute variation strategies
    - "What if" scenario templates
    - Minimal change principles
    """
    
    @staticmethod
    def get_counterfactual_explanation() -> str:
        """
        Get explanation of counterfactual generation.
        
        Returns:
            Counterfactual explanation string
        """
        return """COUNTERFACTUAL GENERATION:

What is a counterfactual?
→ A sample that is IDENTICAL to an existing sample EXCEPT for the protected attribute

Why generate counterfactuals?
→ To create balanced datasets where protected attributes don't determine outcomes
→ To reduce bias by showing that similar individuals can have different protected attributes
→ To improve fairness by balancing representation

Example:
Original sample: {age: 35, education: 13, sex: 1, income: >50K}
Counterfactual:  {age: 35, education: 13, sex: 0, income: >50K}
                                          ↑ ONLY THIS CHANGED

The counterfactual shows: "What if this person had sex=0 instead of sex=1?"
Everything else stays the same (or very similar)."""
    
    @staticmethod
    def get_counterfactual_instructions(
        protected_attributes: List[str],
        target_values: Dict[str, Any]
    ) -> str:
        """
        Get counterfactual generation instructions.
        
        Args:
            protected_attributes: List of protected attribute names
            target_values: Target values for protected attributes
        
        Returns:
            Counterfactual instructions string
        """
        lines = ["COUNTERFACTUAL GENERATION INSTRUCTIONS:", ""]
        
        lines.append("STEP 1: SELECT BASE SAMPLE")
        lines.append("→ Choose an existing sample from the dataset")
        lines.append("→ Preferably from the majority group")
        lines.append("→ Should be realistic and high-quality")
        lines.append("")
        
        lines.append("STEP 2: IDENTIFY PROTECTED ATTRIBUTES TO CHANGE")
        for attr in protected_attributes:
            if attr in target_values:
                lines.append(f"→ Change {attr} to {target_values[attr]}")
        lines.append("")
        
        lines.append("STEP 3: KEEP EVERYTHING ELSE SIMILAR")
        lines.append("→ Maintain all other feature values (or very close)")
        lines.append("→ Preserve correlations between non-protected features")
        lines.append("→ Keep the same outcome/class label")
        lines.append("")
        
        lines.append("STEP 4: MAKE MINIMAL ADJUSTMENTS (if needed)")
        lines.append("→ If some features are correlated with protected attributes:")
        lines.append("  • Make small adjustments to maintain realism")
        lines.append("  • Keep changes minimal (within 10-20% of original)")
        lines.append("  • Document why adjustments were needed")
        lines.append("")
        
        lines.append("RESULT: A counterfactual sample that:")
        lines.append("✓ Has the target protected attribute value(s)")
        lines.append("✓ Is very similar to the base sample in all other aspects")
        lines.append("✓ Is realistic and plausible")
        lines.append("✓ Helps balance the dataset")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_minimal_change_principle() -> str:
        """
        Get minimal change principle explanation.
        
        Returns:
            Minimal change principle string
        """
        return """MINIMAL CHANGE PRINCIPLE:

When generating counterfactuals, make the SMALLEST possible changes:

1. PROTECTED ATTRIBUTES: Change to target value
   Example: sex: 1 → 0

2. HIGHLY CORRELATED FEATURES: Small adjustments (if necessary)
   Example: If occupation is correlated with sex, adjust slightly
   occupation: 5 → 4 (small change)

3. INDEPENDENT FEATURES: Keep exactly the same
   Example: age, education, hours-per-week stay identical

4. OUTCOME: Keep the same (usually)
   Example: income: >50K → >50K (no change)

WHY MINIMAL CHANGES?
→ Demonstrates that protected attributes shouldn't determine outcomes
→ Creates realistic samples (not random)
→ Preserves feature correlations (except for protected attributes)
→ Improves model fairness without sacrificing quality

GUIDELINE: Change ≤ 3 features total (protected attributes + minimal adjustments)"""
    
    @staticmethod
    def get_counterfactual_examples(
        base_sample: Optional[Dict[str, Any]] = None,
        protected_attr: str = "sex",
        original_value: Any = 1,
        target_value: Any = 0
    ) -> str:
        """
        Get counterfactual generation examples.
        
        Args:
            base_sample: Base sample to use for example
            protected_attr: Protected attribute name
            original_value: Original value of protected attribute
            target_value: Target value of protected attribute
        
        Returns:
            Counterfactual examples string
        """
        if base_sample is None:
            # Default example
            return f"""COUNTERFACTUAL EXAMPLE:

Base Sample (from majority group, {protected_attr}={original_value}):
{{
  "age": 0.45,
  "workclass": 4,
  "education-num": 13,
  "{protected_attr}": {original_value},
  "marital-status": 2,
  "occupation": 5,
  "hours-per-week": 0.82,
  "capital-gain": 0.15
}}

Counterfactual Sample ({protected_attr}={target_value}):
{{
  "age": 0.45,           ← SAME
  "workclass": 4,        ← SAME
  "education-num": 13,   ← SAME
  "{protected_attr}": {target_value},  ← CHANGED (target group)
  "marital-status": 2,   ← SAME
  "occupation": 4,       ← SLIGHTLY ADJUSTED (if occupation correlates with {protected_attr})
  "hours-per-week": 0.82, ← SAME
  "capital-gain": 0.15   ← SAME
}}

Changes made:
1. {protected_attr}: {original_value} → {target_value} (PRIMARY CHANGE)
2. occupation: 5 → 4 (MINIMAL ADJUSTMENT, if needed)

Everything else: IDENTICAL"""
        
        else:
            # Use provided sample
            lines = [f"COUNTERFACTUAL EXAMPLE:", ""]
            lines.append(f"Base Sample ({protected_attr}={original_value}):")
            lines.append("{")
            for key, value in base_sample.items():
                lines.append(f'  "{key}": {value},')
            lines.append("}")
            lines.append("")
            lines.append(f"Counterfactual Sample ({protected_attr}={target_value}):")
            lines.append("{")
            for key, value in base_sample.items():
                if key == protected_attr:
                    lines.append(f'  "{key}": {target_value},  ← CHANGED')
                else:
                    lines.append(f'  "{key}": {value},  ← SAME')
            lines.append("}")
            
            return "\n".join(lines)
    
    @staticmethod
    def get_what_if_scenarios(
        protected_attributes: List[str],
        target_values: Dict[str, Any]
    ) -> str:
        """
        Get "what if" scenario templates.
        
        Args:
            protected_attributes: List of protected attributes
            target_values: Target values for protected attributes
        
        Returns:
            "What if" scenarios string
        """
        lines = ['"WHAT IF" REASONING:', ""]
        
        lines.append("For each sample you generate, ask yourself:")
        lines.append("")
        
        for attr in protected_attributes:
            if attr in target_values:
                lines.append(f'→ "What if this person had {attr}={target_values[attr]}?"')
        
        lines.append("")
        lines.append("Then:")
        lines.append("1. Keep all other features the same (or very similar)")
        lines.append("2. Make minimal adjustments for realism")
        lines.append("3. Maintain the same outcome/class")
        lines.append("")
        lines.append("This creates samples that demonstrate:")
        lines.append("✓ Protected attributes don't determine outcomes")
        lines.append("✓ Similar individuals can have different protected attributes")
        lines.append("✓ The model should treat both groups fairly")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_counterfactual_validation_checklist() -> str:
        """
        Get counterfactual validation checklist.
        
        Returns:
            Counterfactual validation checklist string
        """
        return """COUNTERFACTUAL VALIDATION CHECKLIST:

Before returning counterfactual samples, verify:

1. PROTECTED ATTRIBUTE CHANGED:
   ✓ Protected attribute(s) have target value
   ✓ Only protected attributes changed (or minimal adjustments)

2. MINIMAL CHANGES:
   ✓ ≤ 3 features changed total
   ✓ Non-protected features mostly identical
   ✓ Changes are small (within 10-20% of original)

3. REALISM MAINTAINED:
   ✓ Sample is still realistic and plausible
   ✓ Feature correlations preserved (except protected attributes)
   ✓ No implausible combinations

4. OUTCOME PRESERVED:
   ✓ Same class label as base sample (usually)
   ✓ Demonstrates fairness (similar features → similar outcome)

5. QUALITY PARITY:
   ✓ Counterfactual is as high-quality as base sample
   ✓ No degradation in realism or plausibility"""
    
    @staticmethod
    def get_counterfactual_strategy_prompt(
        n_counterfactuals: int,
        n_novel: int
    ) -> str:
        """
        Get counterfactual strategy prompt.
        
        Args:
            n_counterfactuals: Number of counterfactual samples to generate
            n_novel: Number of novel (non-counterfactual) samples to generate
        
        Returns:
            Counterfactual strategy prompt string
        """
        total = n_counterfactuals + n_novel
        cf_pct = (n_counterfactuals / total * 100) if total > 0 else 0
        
        return f"""COUNTERFACTUAL GENERATION STRATEGY:

Generate {total} samples total:
→ {n_counterfactuals} COUNTERFACTUAL samples ({cf_pct:.0f}%)
→ {n_novel} NOVEL samples ({100-cf_pct:.0f}%)

COUNTERFACTUAL SAMPLES ({n_counterfactuals}):
1. Select {n_counterfactuals} diverse base samples from majority group
2. For each base sample:
   • Change protected attribute(s) to target value
   • Keep all other features the same (or very similar)
   • Make minimal adjustments for realism
3. Result: {n_counterfactuals} samples that are counterfactuals of existing samples

NOVEL SAMPLES ({n_novel}):
1. Generate {n_novel} entirely new samples (not based on existing samples)
2. Ensure they belong to target group
3. Maximize diversity (cover different regions)
4. Maintain high quality and realism

This strategy balances:
✓ Fairness (counterfactuals reduce bias)
✓ Diversity (novel samples cover new regions)
✓ Quality (both types maintain high standards)"""
    
    @staticmethod
    def get_intersectional_counterfactual_prompt(
        protected_attributes: List[str],
        target_values: Dict[str, Any]
    ) -> str:
        """
        Get intersectional counterfactual prompt.
        
        Args:
            protected_attributes: List of protected attributes
            target_values: Target values for protected attributes
        
        Returns:
            Intersectional counterfactual prompt string
        """
        if len(protected_attributes) <= 1:
            return ""
        
        lines = ["INTERSECTIONAL COUNTERFACTUALS:", ""]
        lines.append(f"You have {len(protected_attributes)} protected attributes: {', '.join(protected_attributes)}")
        lines.append("")
        lines.append("Generate counterfactuals that address intersectional underrepresentation:")
        lines.append("")
        
        lines.append("APPROACH 1: Change all protected attributes simultaneously")
        change_str = ", ".join([f"{attr}={target_values.get(attr, '?')}" for attr in protected_attributes])
        lines.append(f"→ Base sample → Counterfactual with {change_str}")
        lines.append("")
        
        lines.append("APPROACH 2: Change one protected attribute at a time")
        for attr in protected_attributes:
            if attr in target_values:
                lines.append(f"→ Base sample → Counterfactual with {attr}={target_values[attr]}")
        lines.append("")
        
        lines.append("RECOMMENDATION: Use APPROACH 1 for most samples (addresses intersectionality)")
        lines.append("Use APPROACH 2 for some samples (provides more diversity)")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test counterfactual templates
    print("="*70)
    print("Testing Counterfactual Templates")
    print("="*70)
    
    # Test 1: Counterfactual explanation
    print("\n\nTEST 1: Counterfactual Explanation")
    print("-"*70)
    print(CounterfactualTemplates.get_counterfactual_explanation())
    
    # Test 2: Counterfactual instructions
    print("\n\nTEST 2: Counterfactual Instructions")
    print("-"*70)
    protected_attrs = ["sex", "race"]
    target_vals = {"sex": 0, "race": 2}
    print(CounterfactualTemplates.get_counterfactual_instructions(protected_attrs, target_vals))
    
    # Test 3: Minimal change principle
    print("\n\nTEST 3: Minimal Change Principle")
    print("-"*70)
    print(CounterfactualTemplates.get_minimal_change_principle())
    
    # Test 4: Counterfactual examples
    print("\n\nTEST 4: Counterfactual Examples")
    print("-"*70)
    print(CounterfactualTemplates.get_counterfactual_examples(
        protected_attr="sex",
        original_value=1,
        target_value=0
    ))
    
    # Test 5: What-if scenarios
    print("\n\nTEST 5: What-If Scenarios")
    print("-"*70)
    print(CounterfactualTemplates.get_what_if_scenarios(protected_attrs, target_vals))
    
    # Test 6: Counterfactual validation
    print("\n\nTEST 6: Counterfactual Validation Checklist")
    print("-"*70)
    print(CounterfactualTemplates.get_counterfactual_validation_checklist())
    
    # Test 7: Counterfactual strategy
    print("\n\nTEST 7: Counterfactual Strategy")
    print("-"*70)
    print(CounterfactualTemplates.get_counterfactual_strategy_prompt(
        n_counterfactuals=70,
        n_novel=30
    ))
    
    # Test 8: Intersectional counterfactuals
    print("\n\nTEST 8: Intersectional Counterfactuals")
    print("-"*70)
    print(CounterfactualTemplates.get_intersectional_counterfactual_prompt(
        protected_attrs, target_vals
    ))
    
    print("\n\n✅ Counterfactual Templates Test Complete")

