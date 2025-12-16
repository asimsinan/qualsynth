"""
Fairness-Aware Prompt Templates for Qualsynth

This module contains fairness-specific prompting strategies.
"""

from typing import Dict, List, Any, Optional


class FairnessTemplates:
    """
    Fairness-aware prompt templates for LLM-based generation.
    
    Provides:
    - Fairness instructions and guidelines
    - Target group specifications
    - Balance requirements
    - Bias mitigation strategies
    """
    
    @staticmethod
    def get_fairness_priority_prompt(priority: str = "high") -> str:
        """
        Get fairness priority prompt based on severity.
        
        Args:
            priority: Priority level (high, medium, low)
        
        Returns:
            Fairness priority prompt string
        """
        if priority == "high":
            return """🚨 CRITICAL FAIRNESS PRIORITY 🚨

This dataset has SEVERE fairness violations that MUST be addressed.

YOUR PRIMARY OBJECTIVE: Generate samples that reduce bias and promote fairness.

Fairness is MORE IMPORTANT than:
- Performance optimization
- Sample diversity
- Generation speed

Every sample you generate should:
1. Belong to an underrepresented group
2. Help balance the dataset
3. Reduce existing fairness violations
4. NOT perpetuate or amplify bias"""
        
        elif priority == "medium":
            return """⚠️  IMPORTANT: FAIRNESS-AWARE GENERATION ⚠️

This dataset has fairness concerns that should be addressed.

Balance your objectives:
- Fairness: High priority (but not exclusive)
- Diversity: Important for coverage
- Quality: Maintain high standards

Generate samples that:
1. Prioritize underrepresented groups
2. Help improve fairness metrics
3. Maintain diversity and quality"""
        
        else:  # low
            return """ℹ️  FAIRNESS CONSIDERATION

While generating samples, be mindful of fairness:
- Avoid amplifying existing biases
- Consider representation across protected groups
- Generate balanced samples when possible"""
    
    @staticmethod
    def get_target_group_specification(
        fairness_targets: List[Any],
        detailed: bool = True
    ) -> str:
        """
        Get target group specification from fairness targets.
        
        Args:
            fairness_targets: List of fairness targets from FairnessAuditor
            detailed: Whether to include detailed statistics
        
        Returns:
            Target group specification string
        """
        if not fairness_targets:
            return "No specific fairness targets. Generate diverse samples across all groups."
        
        lines = ["TARGET GROUPS (Fairness-Driven):", ""]
        
        for i, target in enumerate(fairness_targets, 1):
            if not hasattr(target, 'attribute'):
                continue
            
            attr = target.attribute
            target_group = target.target_group
            priority = target.priority if hasattr(target, 'priority') else "medium"
            
            priority_emoji = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
            
            lines.append(f"{priority_emoji} Target {i}: {attr} = {target_group}")
            
            if detailed and hasattr(target, 'current_proportion'):
                current = target.current_proportion * 100
                target_prop = target.target_proportion * 100 if hasattr(target, 'target_proportion') else 50.0
                samples_needed = target.samples_needed if hasattr(target, 'samples_needed') else 0
                
                lines.append(f"   Current: {current:.1f}% | Target: {target_prop:.1f}%")
                lines.append(f"   Samples needed: ~{samples_needed}")
                lines.append(f"   Priority: {priority.upper()}")
            
            lines.append("")
        
        lines.append("INSTRUCTION:")
        lines.append("Generate samples that belong to these target groups.")
        lines.append("If multiple targets exist, distribute samples across them based on priority.")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_bias_mitigation_guidelines() -> str:
        """
        Get bias mitigation guidelines.
        
        Returns:
            Bias mitigation guidelines string
        """
        return """BIAS MITIGATION GUIDELINES:

1. AVOID STEREOTYPES:
   ✗ Don't assume correlations based on protected attributes
   ✗ Don't generate samples that reinforce stereotypes
   ✓ Generate diverse samples within each protected group

2. BALANCE REPRESENTATION:
   ✓ Ensure underrepresented groups are well-represented
   ✓ Generate samples across all subgroups
   ✓ Don't over-represent already dominant groups

3. FEATURE INDEPENDENCE:
   ✓ Protected attributes should not determine other features
   ✓ Generate realistic combinations regardless of protected attributes
   ✓ Avoid proxy discrimination (using correlated features as proxies)

4. INTERSECTIONALITY:
   ✓ Consider multiple protected attributes simultaneously
   ✓ Address intersectional underrepresentation
   ✓ Generate samples for minority subgroups (e.g., female + young)

5. QUALITY PARITY:
   ✓ Maintain same quality standards across all groups
   ✓ Don't generate lower-quality samples for minority groups
   ✓ Ensure all groups have realistic, plausible samples"""
    
    @staticmethod
    def get_fairness_validation_checklist(fairness_targets: List[Any]) -> str:
        """
        Get fairness validation checklist.
        
        Args:
            fairness_targets: List of fairness targets
        
        Returns:
            Fairness validation checklist string
        """
        lines = ["FAIRNESS VALIDATION CHECKLIST:", ""]
        lines.append("Before returning your samples, verify:")
        lines.append("")
        
        if fairness_targets:
            lines.append("1. TARGET GROUP COMPLIANCE:")
            for target in fairness_targets:
                if hasattr(target, 'attribute'):
                    attr = target.attribute
                    target_group = target.target_group
                    lines.append(f"   ✓ All/most samples have {attr} = {target_group}")
            lines.append("")
        
        lines.append("2. NO BIAS AMPLIFICATION:")
        lines.append("   ✓ Samples don't reinforce stereotypes")
        lines.append("   ✓ No discriminatory patterns")
        lines.append("   ✓ Balanced representation within target groups")
        lines.append("")
        
        lines.append("3. INTERSECTIONAL FAIRNESS:")
        lines.append("   ✓ Multiple protected attributes considered")
        lines.append("   ✓ Minority subgroups represented")
        lines.append("   ✓ No intersectional bias")
        lines.append("")
        
        lines.append("4. QUALITY PARITY:")
        lines.append("   ✓ Same quality across all groups")
        lines.append("   ✓ Realistic samples for all groups")
        lines.append("   ✓ No quality degradation for minority groups")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_fairness_examples(
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None
    ) -> str:
        """
        Get fairness examples (what to do and what not to do).
        
        Args:
            positive_examples: List of positive examples
            negative_examples: List of negative examples
        
        Returns:
            Fairness examples string
        """
        lines = ["FAIRNESS EXAMPLES:", ""]
        
        if positive_examples:
            lines.append("✓ GOOD PRACTICES:")
            for ex in positive_examples:
                lines.append(f"  • {ex}")
            lines.append("")
        else:
            lines.append("✓ GOOD PRACTICES:")
            lines.append("  • Generate samples for underrepresented sex=0 group")
            lines.append("  • Vary other features (age, education) within target group")
            lines.append("  • Avoid stereotypical combinations")
            lines.append("  • Ensure realistic, high-quality samples")
            lines.append("")
        
        if negative_examples:
            lines.append("✗ BAD PRACTICES:")
            for ex in negative_examples:
                lines.append(f"  • {ex}")
            lines.append("")
        else:
            lines.append("✗ BAD PRACTICES:")
            lines.append("  • Generating samples for already over-represented groups")
            lines.append("  • Assuming low education for minority groups")
            lines.append("  • Creating stereotypical combinations")
            lines.append("  • Lower quality samples for minority groups")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_fairness_metrics_explanation() -> str:
        """
        Get explanation of fairness metrics.
        
        Returns:
            Fairness metrics explanation string
        """
        return """FAIRNESS METRICS (For Your Understanding):

Your generated samples will be evaluated using:

1. DEMOGRAPHIC PARITY DIFFERENCE (DPD):
   - Measures: Difference in positive prediction rates between groups
   - Goal: DPD close to 0 (equal positive rates)
   - Your impact: Generate samples that balance positive rates

2. EQUALIZED ODDS DIFFERENCE (EOD):
   - Measures: Difference in true positive AND false positive rates
   - Goal: EOD close to 0 (equal error rates)
   - Your impact: Generate realistic samples (not just positive class)

3. EQUAL OPPORTUNITY DIFFERENCE:
   - Measures: Difference in true positive rates (recall)
   - Goal: Close to 0 (equal opportunity for positive outcome)
   - Your impact: Generate high-quality positive samples for minority groups

Your samples should help REDUCE these metrics (bring them closer to 0)."""
    
    @staticmethod
    def get_fairness_first_strategy_prompt() -> str:
        """
        Get fairness-first strategy prompt.
        
        Returns:
            Fairness-first strategy prompt string
        """
        return """FAIRNESS-FIRST STRATEGY:

This generation uses a FAIRNESS-FIRST approach:

STEP 1: IDENTIFY TARGET GROUP
→ Determine which protected group(s) need samples
→ Check fairness targets and priorities

STEP 2: GENERATE FOR TARGET GROUP
→ ALL samples must belong to target group(s)
→ Set protected attribute(s) first
→ Then generate other features

STEP 3: ENSURE DIVERSITY WITHIN GROUP
→ Vary non-protected features
→ Cover different regions of feature space
→ Avoid duplicates within target group

STEP 4: VALIDATE FAIRNESS IMPACT
→ Verify all samples belong to target group
→ Check for bias amplification
→ Ensure quality parity

REMEMBER: Fairness is the PRIMARY objective. Other objectives (diversity, quality) are important but secondary."""


if __name__ == "__main__":
    # Test fairness templates
    from dataclasses import dataclass
    
    @dataclass
    class MockFairnessTarget:
        attribute: str
        target_group: int
        priority: str
        current_proportion: float
        target_proportion: float
        samples_needed: int
    
    print("="*70)
    print("Testing Fairness Templates")
    print("="*70)
    
    # Test 1: Fairness priority prompts
    print("\n\nTEST 1: Fairness Priority Prompts")
    print("-"*70)
    
    for priority in ["high", "medium", "low"]:
        print(f"\n{priority.upper()} Priority:")
        print("-"*70)
        print(FairnessTemplates.get_fairness_priority_prompt(priority))
    
    # Test 2: Target group specification
    print("\n\nTEST 2: Target Group Specification")
    print("-"*70)
    
    mock_targets = [
        MockFairnessTarget(
            attribute="sex",
            target_group=0,
            priority="high",
            current_proportion=0.362,
            target_proportion=0.50,
            samples_needed=1929
        ),
        MockFairnessTarget(
            attribute="race",
            target_group=2,
            priority="medium",
            current_proportion=0.15,
            target_proportion=0.25,
            samples_needed=500
        )
    ]
    
    print(FairnessTemplates.get_target_group_specification(mock_targets, detailed=True))
    
    # Test 3: Bias mitigation guidelines
    print("\n\nTEST 3: Bias Mitigation Guidelines")
    print("-"*70)
    print(FairnessTemplates.get_bias_mitigation_guidelines())
    
    # Test 4: Fairness validation checklist
    print("\n\nTEST 4: Fairness Validation Checklist")
    print("-"*70)
    print(FairnessTemplates.get_fairness_validation_checklist(mock_targets))
    
    # Test 5: Fairness examples
    print("\n\nTEST 5: Fairness Examples")
    print("-"*70)
    print(FairnessTemplates.get_fairness_examples())
    
    # Test 6: Fairness metrics explanation
    print("\n\nTEST 6: Fairness Metrics Explanation")
    print("-"*70)
    print(FairnessTemplates.get_fairness_metrics_explanation())
    
    # Test 7: Fairness-first strategy
    print("\n\nTEST 7: Fairness-First Strategy")
    print("-"*70)
    print(FairnessTemplates.get_fairness_first_strategy_prompt())
    
    print("\n\n✅ Fairness Templates Test Complete")

