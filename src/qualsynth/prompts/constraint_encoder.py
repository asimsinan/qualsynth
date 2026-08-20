"""
Constraint Encoder for Qualsynth

This module converts schema, fairness, and diversity constraints into natural language.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from ..modules.schema_profiler import FeatureType
except ImportError:
    from src.qualsynth.modules.schema_profiler import FeatureType


class ConstraintEncoder:
    """
    Converts constraints into natural language for LLM prompts.
    
    Handles:
    - Schema constraints (types, ranges, categories)
    - Logical constraints (if-then rules, correlations)
    - Fairness constraints (target proportions)
    - Diversity constraints (sparse regions)
    """
    
    @staticmethod
    def encode_schema_constraints(schema_report: Any) -> str:
        """
        Encode schema constraints as natural language.
        
        Args:
            schema_report: Schema report from SchemaProfiler
        
        Returns:
            Schema constraints string
        """
        if not schema_report or not hasattr(schema_report, 'features'):
            return "No schema constraints specified."
        
        lines = ["🔒 STRICT SCHEMA CONSTRAINTS (MUST OBEY - SOTA REQUIREMENT):", ""]
        lines.append("Each sample MUST satisfy these feature specifications.")
        lines.append("Values outside ranges/categories will be IMMEDIATELY REJECTED.")
        lines.append("")
        
        for feature_name, feature_info in schema_report.features.items():
            # Handle FeatureSchema object
            feature_type = getattr(feature_info, 'type', FeatureType.NUMERICAL_CONTINUOUS)
            
            # Feature header
            lines.append(f"• {feature_name}:")
            
            # Type and range/categories
            if feature_type == FeatureType.BINARY:
                values = getattr(feature_info, 'categories', [0, 1])
                lines.append(f"  Type: Binary")
                lines.append(f"  ✓ VALID VALUES: {{{', '.join(map(str, values))}}}")
                lines.append(f"  ✗ INVALID: Any value not in {{{', '.join(map(str, values))}}}")
                lines.append(f"  🎯 DIVERSITY: Use BOTH values across your samples, not just one")
            
            elif feature_type in [FeatureType.CATEGORICAL_NOMINAL, FeatureType.CATEGORICAL_ORDINAL]:
                categories = getattr(feature_info, 'categories', [])
                ordinal_str = " (ordinal)" if feature_type == FeatureType.CATEGORICAL_ORDINAL else ""
                lines.append(f"  Type: Categorical{ordinal_str}")
                lines.append(f"  ✓ VALID CATEGORIES: {{{', '.join(map(str, categories))}}}")
                lines.append(f"  ✗ INVALID: Any category not in the list above")
                lines.append(f"  🎯 DIVERSITY: Use MULTIPLE categories across samples, not just one")
            
            elif feature_type == FeatureType.NUMERICAL_DISCRETE:
                min_val = getattr(feature_info, 'min', 0)
                max_val = getattr(feature_info, 'max', 100)
                lines.append(f"  Type: Numerical (discrete/integer)")
                lines.append(f"  ✓ VALID RANGE: [{min_val}, {max_val}] (integers only)")
                lines.append(f"  ✗ INVALID: Values < {min_val} or > {max_val}")
                lines.append(f"  🎯 DIVERSITY: Generate values across FULL range (low, medium, high)")
            
            elif feature_type == FeatureType.NUMERICAL_CONTINUOUS:
                min_val = getattr(feature_info, 'min', 0.0)
                max_val = getattr(feature_info, 'max', 1.0)
                lines.append(f"  Type: Numerical (continuous/float)")
                lines.append(f"  ✓ VALID RANGE: [{min_val:.2f}, {max_val:.2f}]")
                lines.append(f"  ✗ INVALID: Values < {min_val:.2f} or > {max_val:.2f}")
                lines.append(f"  🎯 DIVERSITY: Generate values spanning FULL range (not clustered)")
            
            # Outliers (if any)
            outliers = getattr(feature_info, 'outliers', [])
            if outliers:
                lines.append(f"  ⚠️  Avoid outlier values: {outliers}")
            
            lines.append("")
        
        lines.append("🚨 CRITICAL VALIDATION RULES:")
        lines.append("1. ALL values MUST be within specified ranges/categories")
        lines.append("2. Out-of-range values = IMMEDIATE REJECTION")
        lines.append("3. Each feature MUST vary across samples (no constant columns)")
        lines.append("4. Categorical features: Use MULTIPLE categories, not just one")
        lines.append("5. Numerical features: Cover FULL range (low/medium/high values)")
        lines.append("")
        lines.append("✓ VALID: Values within ranges, diverse across samples")
        lines.append("✗ INVALID: Out-of-range values, constant columns, low diversity")
        
        return "\n".join(lines)
    
    @staticmethod
    def encode_logical_constraints(schema_report: Any) -> str:
        """
        Encode logical constraints as natural language.
        
        Args:
            schema_report: Schema report from SchemaProfiler
        
        Returns:
            Logical constraints string
        """
        if not schema_report or not hasattr(schema_report, 'logical_constraints'):
            return ""
        
        constraints = schema_report.logical_constraints
        if not constraints:
            return ""
        
        lines = ["LOGICAL CONSTRAINTS:", ""]
        lines.append("Samples must satisfy these logical rules:")
        lines.append("")
        
        # Mutual exclusions
        if 'mutual_exclusions' in constraints and constraints['mutual_exclusions']:
            lines.append("Mutual Exclusions (only one can be true):")
            for exclusion in constraints['mutual_exclusions']:
                features = getattr(exclusion, 'features', [])
                lines.append(f"  • {' OR '.join(features)} (not both)")
            lines.append("")
        
        # Implications
        if 'implications' in constraints and constraints['implications']:
            lines.append("If-Then Rules:")
            for impl in constraints['implications']:
                condition = getattr(impl, 'condition', '')
                consequence = getattr(impl, 'consequence', '')
                lines.append(f"  • IF {condition} THEN {consequence}")
            lines.append("")
        
        # Correlations
        if 'correlations' in constraints and constraints['correlations']:
            lines.append("Strong Correlations (maintain these relationships):")
            for corr in constraints['correlations']:
                feat1 = getattr(corr, 'feature1', '')
                feat2 = getattr(corr, 'feature2', '')
                strength = getattr(corr, 'correlation', 0.0)
                direction = "positively" if strength > 0 else "negatively"
                lines.append(f"  • {feat1} and {feat2} are {direction} correlated ({strength:.2f})")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def encode_fairness_constraints(schema_report: Any) -> str:
        """
        Encode fairness constraints as natural language.
        
        Args:
            schema_report: Schema report from SchemaProfiler
        
        Returns:
            Fairness constraints string
        """
        if not schema_report or not hasattr(schema_report, 'fairness_constraints'):
            return ""
        
        constraints = schema_report.fairness_constraints
        if not constraints:
            return ""
        
        lines = ["FAIRNESS CONSTRAINTS:", ""]
        
        # Balance constraints
        if 'balance_constraints' in constraints and constraints['balance_constraints']:
            lines.append("Target Proportions (for fairness):")
            for balance in constraints['balance_constraints']:
                attr = getattr(balance, 'attribute', '')
                group = getattr(balance, 'group', '')
                target_prop = getattr(balance, 'target_proportion', 0.5) * 100
                lines.append(f"  • {attr}={group}: Target {target_prop:.1f}% of samples")
            lines.append("")
        
        # Counterfactual constraints
        if 'counterfactual_constraints' in constraints and constraints['counterfactual_constraints']:
            lines.append("Counterfactual Requirements:")
            for cf in constraints['counterfactual_constraints']:
                attr = getattr(cf, 'attribute', '')
                values = getattr(cf, 'values', [])
                lines.append(f"  • Generate counterfactuals for {attr}: {values}")
            lines.append("")
        
        lines.append("PRIORITY: Fairness constraints are HIGH PRIORITY.")
        lines.append("Ensure generated samples meet these fairness requirements.")
        
        return "\n".join(lines)
    
    @staticmethod
    def encode_diversity_targets(diversity_plan: Any) -> str:
        """
        Encode diversity targets as natural language.
        
        Args:
            diversity_plan: Diversity plan from DiversityPlanner
        
        Returns:
            Diversity targets string
        """
        if not diversity_plan or not hasattr(diversity_plan, 'diversity_targets'):
            return ""
        
        targets = diversity_plan.diversity_targets
        if not targets:
            return ""
        
        lines = ["DIVERSITY TARGETS:", ""]
        lines.append("Focus on these sparse/underrepresented regions:")
        lines.append("")
        
        for i, target in enumerate(targets, 1):
            group = getattr(target, 'group', '')
            region_id = getattr(target, 'region_id', i)
            density = getattr(target, 'density', 0.0)
            samples_needed = getattr(target, 'samples_needed', 0)
            
            lines.append(f"Region {region_id} ({group}):")
            lines.append(f"  Density: {density:.3f} (sparse)")
            lines.append(f"  Samples needed: ~{samples_needed}")
            
            # Region characteristics
            chars = getattr(target, 'characteristics', None)
            if chars:
                lines.append(f"  Characteristics: {chars}")
            
            lines.append("")
        
        lines.append("GOAL: Generate diverse samples that cover these sparse regions.")
        lines.append("Avoid generating samples in already dense regions.")
        
        return "\n".join(lines)
    
    @staticmethod
    def encode_all_constraints(
        schema_report: Any,
        diversity_plan: Optional[Any] = None
    ) -> str:
        """
        Encode all constraints into a single prompt section.
        
        Args:
            schema_report: Schema report from SchemaProfiler
            diversity_plan: Diversity plan from DiversityPlanner (optional)
        
        Returns:
            Complete constraints string
        """
        sections = []
        
        # Schema constraints
        schema_str = ConstraintEncoder.encode_schema_constraints(schema_report)
        if schema_str:
            sections.append(schema_str)
        
        # Logical constraints
        logical_str = ConstraintEncoder.encode_logical_constraints(schema_report)
        if logical_str:
            sections.append(logical_str)
        
        # Fairness constraints
        fairness_str = ConstraintEncoder.encode_fairness_constraints(schema_report)
        if fairness_str:
            sections.append(fairness_str)
        
        # Diversity targets
        if diversity_plan:
            diversity_str = ConstraintEncoder.encode_diversity_targets(diversity_plan)
            if diversity_str:
                sections.append(diversity_str)
        
        return "\n\n".join(sections)


if __name__ == "__main__":
    # Test constraint encoder
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    from src.qualsynth.modules.schema_profiler import SchemaProfiler
    from src.qualsynth.modules.diversity_planner import DiversityPlanner
    
    print("="*70)
    print("Testing Constraint Encoder")
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
    
    # Run fairness audit
    auditor = FairnessAuditor(fairness_threshold=0.05)
    audit_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
    
    # Profile schema
    schema_profiler = SchemaProfiler()
    schema = schema_profiler.profile(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=audit_report.fairness_targets,
        dataset_name=dataset_name
    )
    
    # Create diversity plan
    planner = DiversityPlanner()
    diversity_plan = planner.plan(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=audit_report.fairness_targets,
        dataset_name=dataset_name
    )
    
    # Test 1: Schema constraints
    print("\n\nTEST 1: Schema Constraints")
    print("-"*70)
    print(ConstraintEncoder.encode_schema_constraints(schema))
    
    # Test 2: Logical constraints
    print("\n\nTEST 2: Logical Constraints")
    print("-"*70)
    logical_str = ConstraintEncoder.encode_logical_constraints(schema)
    if logical_str:
        print(logical_str)
    else:
        print("No logical constraints found")
    
    # Test 3: Fairness constraints
    print("\n\nTEST 3: Fairness Constraints")
    print("-"*70)
    fairness_str = ConstraintEncoder.encode_fairness_constraints(schema)
    if fairness_str:
        print(fairness_str)
    else:
        print("No fairness constraints found")
    
    # Test 4: Diversity targets
    print("\n\nTEST 4: Diversity Targets")
    print("-"*70)
    diversity_str = ConstraintEncoder.encode_diversity_targets(diversity_plan)
    if diversity_str:
        print(diversity_str)
    else:
        print("No diversity targets found")
    
    # Test 5: All constraints
    print("\n\nTEST 5: All Constraints Combined")
    print("-"*70)
    all_constraints = ConstraintEncoder.encode_all_constraints(schema, diversity_plan)
    print(all_constraints[:1000], "...")  # Print first 1000 chars
    
    print("\n\n✅ Constraint Encoder Test Complete")

