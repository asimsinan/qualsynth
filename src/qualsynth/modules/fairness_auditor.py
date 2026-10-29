"""
Fairness Auditor for Qualsynth Framework (FIRST STEP)

This component runs FIRST in the fairness-first workflow to:
1. Analyze current fairness violations in the dataset
2. Identify underrepresented protected groups
3. Set fairness targets for generation
4. Provide guidance for counterfactual generation

This is NOT an optional feedback loop - it's a mandatory first step that
guides all downstream generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import warnings

# Fairlearn imports
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    warnings.warn("Fairlearn not available. Fairness metrics will be limited.")

# AIF360 imports (optional)
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False


@dataclass
class FairnessViolation:
    """Details of a specific fairness violation."""
    attribute: str
    metric: str  # 'dpd', 'eod', 'eopd'
    value: float
    threshold: float
    violation_factor: float  # How many times over threshold
    privileged_group: Any
    unprivileged_group: Any
    privileged_rate: float
    unprivileged_rate: float
    severity: str  # 'moderate', 'severe', 'catastrophic'


@dataclass
class FairnessTarget:
    """Target for fairness-aware generation."""
    attribute: str
    target_group: Any  # The underrepresented group to prioritize
    current_proportion: float  # Current proportion in minority class
    target_proportion: float  # Target proportion to achieve
    n_samples_needed: int  # Number of samples needed for this group
    priority: str  # 'high', 'medium', 'low'


@dataclass
class FairnessAuditReport:
    """
    Comprehensive fairness audit report.
    
    This is the output of the FIRST STEP in fairness-first workflow.
    """
    # Violations
    violations: List[FairnessViolation] = field(default_factory=list)
    max_violation: float = 0.0
    overall_severity: str = "none"
    
    # Group analysis
    underrepresented_groups: List[str] = field(default_factory=list)
    group_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Targets
    fairness_targets: List[FairnessTarget] = field(default_factory=list)
    
    # Recommendations
    counterfactual_pairs: List[Tuple[Any, Any]] = field(default_factory=list)
    generation_strategy: str = "standard"
    fairness_weight: float = 0.5
    
    # Metrics
    all_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Summary
    passed: bool = False
    summary: str = ""


class FairnessAuditor:
    """
    Fairness Auditor - FIRST STEP in fairness-first Qualsynth workflow.
    
    This component runs BEFORE any generation to:
    1. Identify fairness violations in the original dataset
    2. Set targets for fairness-aware generation
    3. Provide guidance for counterfactual generation
    
    This is NOT optional - it's mandatory for fairness-first approach.
    """
    
    def __init__(
        self,
        fairness_threshold: float = 0.05,
        use_aif360: bool = True
    ):
        """
        Initialize Fairness Auditor.
        
        Args:
            fairness_threshold: Maximum acceptable fairness metric difference (default: 0.05)
            use_aif360: Whether to use AIF360 for additional metrics (default: True)
        """
        self.fairness_threshold = fairness_threshold
        self.use_aif360 = use_aif360 and AIF360_AVAILABLE
        
        if not FAIRLEARN_AVAILABLE:
            raise ImportError(
                "Fairlearn is required for FairnessAuditor. "
                "Install with: pip install fairlearn"
            )
    
    def audit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame,
        dataset_name: Optional[str] = None
    ) -> FairnessAuditReport:
        """
        Perform comprehensive fairness audit (FIRST STEP).
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_features: Sensitive attributes for fairness analysis
            dataset_name: Optional dataset name for logging
        
        Returns:
            FairnessAuditReport with violations, targets, and recommendations
        """
        report = FairnessAuditReport()
        
        # 1. Analyze group distributions
        report.group_distributions = self._analyze_group_distributions(
            X, y, sensitive_features
        )
        
        # 2. Detect fairness violations
        report.violations = self._detect_violations(
            X, y, sensitive_features
        )
        
        # 3. Calculate overall severity
        if report.violations:
            report.max_violation = max(v.value for v in report.violations)
            report.overall_severity = self._categorize_severity(report.max_violation)
        
        # 4. Identify underrepresented groups
        report.underrepresented_groups = self._identify_underrepresented_groups(
            X, y, sensitive_features
        )
        
        # 5. Set fairness targets
        report.fairness_targets = self._set_fairness_targets(
            X, y, sensitive_features, report.violations, report.group_distributions
        )
        
        # 6. Generate counterfactual pairs
        report.counterfactual_pairs = self._generate_counterfactual_pairs(
            sensitive_features
        )
        
        # 7. Recommend generation strategy
        report.generation_strategy, report.fairness_weight = self._recommend_strategy(
            report.overall_severity, report.max_violation
        )
        
        # 8. Check if passed
        report.passed = report.max_violation < self.fairness_threshold
        
        # 9. Generate summary
        report.summary = self._generate_summary(report)
        
        return report
    
    def _analyze_group_distributions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Analyze distribution of sensitive groups in minority vs majority."""
        distributions = {}
        
        # Handle None sensitive features
        if sensitive_features is None or len(sensitive_features.columns) == 0:
            return distributions
        
        minority_mask = y == 1
        
        for col in sensitive_features.columns:
            if col not in X.columns:
                continue
            
            distributions[col] = {}
            
            # Minority class distribution
            minority_dist = X[minority_mask][col].value_counts(normalize=True)
            # Majority class distribution
            majority_dist = X[~minority_mask][col].value_counts(normalize=True)
            
            for group in X[col].unique():
                distributions[col][f'minority_{group}'] = minority_dist.get(group, 0.0)
                distributions[col][f'majority_{group}'] = majority_dist.get(group, 0.0)
        
        return distributions
    
    def _detect_violations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame
    ) -> List[FairnessViolation]:
        """Detect fairness violations using demographic parity."""
        violations = []
        
        # Handle None sensitive features
        if sensitive_features is None or len(sensitive_features.columns) == 0:
            return violations
        
        for col in sensitive_features.columns:
            if col not in X.columns:
                continue
            
            groups = X[col].unique()
            if len(groups) != 2:
                # For now, only handle binary sensitive attributes
                continue
            
            g1, g2 = groups
            
            # Demographic Parity Difference
            p1 = y[X[col] == g1].mean()
            p2 = y[X[col] == g2].mean()
            dpd = abs(p1 - p2)
            
            if dpd > self.fairness_threshold:
                # Determine which is privileged (higher positive rate)
                privileged_group = g1 if p1 > p2 else g2
                unprivileged_group = g2 if p1 > p2 else g1
                privileged_rate = max(p1, p2)
                unprivileged_rate = min(p1, p2)
                
                violation_factor = dpd / self.fairness_threshold
                severity = self._categorize_severity(dpd)
                
                violations.append(FairnessViolation(
                    attribute=col,
                    metric='dpd',
                    value=dpd,
                    threshold=self.fairness_threshold,
                    violation_factor=violation_factor,
                    privileged_group=privileged_group,
                    unprivileged_group=unprivileged_group,
                    privileged_rate=privileged_rate,
                    unprivileged_rate=unprivileged_rate,
                    severity=severity
                ))
        
        return violations
    
    def _categorize_severity(self, violation: float) -> str:
        """Categorize violation severity."""
        if violation < self.fairness_threshold:
            return "none"
        elif violation < 0.2:
            return "moderate"  # 2-4x
        elif violation < 0.5:
            return "severe"  # 4-10x
        else:
            return "catastrophic"  # >10x
    
    def _identify_underrepresented_groups(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame
    ) -> List[str]:
        """Identify underrepresented groups in minority class."""
        underrep = []
        
        # Handle None sensitive features
        if sensitive_features is None or len(sensitive_features.columns) == 0:
            return []
        
        minority_mask = y == 1
        X_minority = X[minority_mask]
        X_majority = X[~minority_mask]
        
        for col in sensitive_features.columns:
            if col not in X.columns:
                continue
            
            minority_dist = X_minority[col].value_counts(normalize=True)
            majority_dist = X_majority[col].value_counts(normalize=True)
            
            for group in minority_dist.index:
                if group in majority_dist.index:
                    # If this group is significantly underrepresented in minority
                    if minority_dist[group] < majority_dist[group] * 0.7:
                        underrep.append(f"{col}={group}")
        
        return underrep
    
    def _set_fairness_targets(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame,
        violations: List[FairnessViolation],
        distributions: Dict[str, Dict[str, float]]
    ) -> List[FairnessTarget]:
        """
        Set fairness targets for generation.
        
        For each violation, determine how many samples of which group
        need to be generated to reduce the violation.
        """
        targets = []
        
        minority_mask = y == 1
        n_minority = minority_mask.sum()
        
        for violation in violations:
            attr = violation.attribute
            
            # Calculate how many samples of unprivileged group needed
            # to balance the rates
            
            # Current counts
            minority_unprivileged = ((X[attr] == violation.unprivileged_group) & minority_mask).sum()
            minority_privileged = ((X[attr] == violation.privileged_group) & minority_mask).sum()
            
            # Current proportions
            current_prop = minority_unprivileged / n_minority if n_minority > 0 else 0
            
            # Target: balance the proportions to reduce DPD
            # Ideally, we want proportions to match majority class
            majority_unprivileged = ((X[attr] == violation.unprivileged_group) & ~minority_mask).sum()
            majority_privileged = ((X[attr] == violation.privileged_group) & ~minority_mask).sum()
            n_majority = (~minority_mask).sum()
            
            target_prop = majority_unprivileged / n_majority if n_majority > 0 else 0.5
            
            # How many samples needed?
            # This is a simplified calculation - we need to generate more of the unprivileged group
            n_samples_needed = max(0, int((target_prop - current_prop) * n_minority * 2))
            
            # Priority based on severity
            if violation.severity == "catastrophic":
                priority = "high"
            elif violation.severity == "severe":
                priority = "high"
            elif violation.severity == "moderate":
                priority = "medium"
            else:
                priority = "low"
            
            targets.append(FairnessTarget(
                attribute=attr,
                target_group=violation.unprivileged_group,
                current_proportion=current_prop,
                target_proportion=target_prop,
                n_samples_needed=n_samples_needed,
                priority=priority
            ))
        
        return targets
    
    def _generate_counterfactual_pairs(
        self,
        sensitive_features: pd.DataFrame
    ) -> List[Tuple[Any, Any]]:
        """
        Generate counterfactual pairs for generation.
        
        For each binary sensitive attribute, create pairs (A, B) where
        we should generate similar samples with different sensitive attribute values.
        """
        pairs = []
        
        # Handle None sensitive features
        if sensitive_features is None or len(sensitive_features.columns) == 0:
            return pairs
        
        for col in sensitive_features.columns:
            unique_vals = sensitive_features[col].unique()
            if len(unique_vals) == 2:
                pairs.append((col, tuple(unique_vals)))
        
        return pairs
    
    def _recommend_strategy(
        self,
        severity: str,
        max_violation: float
    ) -> Tuple[str, float]:
        """
        Recommend generation strategy based on violation severity.
        
        Returns:
            (strategy_name, fairness_weight)
        """
        if severity == "catastrophic":
            return "fairness_first", 0.8
        elif severity == "severe":
            return "fairness_first", 0.7
        elif severity == "moderate":
            return "fairness_aware", 0.5
        else:
            return "standard", 0.3
    
    def _generate_summary(self, report: FairnessAuditReport) -> str:
        """Generate human-readable summary."""
        if report.passed:
            return "✅ No significant fairness violations detected."
        
        summary_parts = []
        
        if report.violations:
            summary_parts.append(
                f"🚨 {len(report.violations)} fairness violation(s) detected "
                f"(max: {report.max_violation:.4f}, {report.max_violation/self.fairness_threshold:.1f}x over threshold)"
            )
        
        if report.underrepresented_groups:
            summary_parts.append(
                f"⚠️ {len(report.underrepresented_groups)} underrepresented group(s): "
                f"{', '.join(report.underrepresented_groups[:3])}{'...' if len(report.underrepresented_groups) > 3 else ''}"
            )
        
        if report.fairness_targets:
            total_samples = sum(t.n_samples_needed for t in report.fairness_targets)
            summary_parts.append(
                f"🎯 {len(report.fairness_targets)} fairness target(s) set "
                f"(~{total_samples} samples needed)"
            )
        
        summary_parts.append(
            f"📋 Recommended strategy: {report.generation_strategy.upper()} "
            f"(fairness_weight={report.fairness_weight})"
        )
        
        return "\n".join(summary_parts)
    
    def print_report(
        self,
        report: FairnessAuditReport,
        dataset_name: str = "Dataset"
    ) -> None:
        """Print comprehensive fairness audit report."""
        print(f"\n{'='*70}")
        print(f"FAIRNESS AUDIT REPORT (FIRST STEP): {dataset_name}")
        print(f"{'='*70}")
        
        # Overall status
        status = "✅ PASSED" if report.passed else "❌ FAILED"
        print(f"\n📊 OVERALL STATUS: {status}")
        print(f"   Max violation: {report.max_violation:.4f} ({report.max_violation/self.fairness_threshold:.1f}x)")
        print(f"   Severity: {report.overall_severity.upper()}")
        
        # Violations
        if report.violations:
            print(f"\n🚨 FAIRNESS VIOLATIONS ({len(report.violations)}):")
            for v in report.violations:
                print(f"\n   {v.attribute} ({v.metric.upper()}):")
                print(f"      Value: {v.value:.4f} (threshold: {v.threshold:.4f})")
                print(f"      Violation: {v.violation_factor:.1f}x over threshold")
                print(f"      Severity: {v.severity.upper()}")
                print(f"      Privileged group ({v.privileged_group}): {v.privileged_rate:.4f}")
                print(f"      Unprivileged group ({v.unprivileged_group}): {v.unprivileged_rate:.4f}")
        else:
            print(f"\n✅ No fairness violations detected")
        
        # Underrepresented groups
        if report.underrepresented_groups:
            print(f"\n⚠️  UNDERREPRESENTED GROUPS ({len(report.underrepresented_groups)}):")
            for group in report.underrepresented_groups[:10]:
                print(f"   • {group}")
            if len(report.underrepresented_groups) > 10:
                print(f"   ... and {len(report.underrepresented_groups) - 10} more")
        
        # Fairness targets
        if report.fairness_targets:
            print(f"\n🎯 FAIRNESS TARGETS ({len(report.fairness_targets)}):")
            for target in report.fairness_targets:
                print(f"\n   {target.attribute} = {target.target_group}:")
                print(f"      Current proportion: {target.current_proportion:.4f}")
                print(f"      Target proportion: {target.target_proportion:.4f}")
                print(f"      Samples needed: ~{target.n_samples_needed}")
                print(f"      Priority: {target.priority.upper()}")
        
        # Counterfactual pairs
        if report.counterfactual_pairs:
            print(f"\n🔄 COUNTERFACTUAL PAIRS ({len(report.counterfactual_pairs)}):")
            for attr, (val1, val2) in report.counterfactual_pairs:
                print(f"   • {attr}: {val1} ↔ {val2}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        print(f"   Strategy: {report.generation_strategy.upper()}")
        print(f"   Fairness weight: {report.fairness_weight}")
        print(f"   Use counterfactual generation: {'YES' if report.fairness_targets else 'NO'}")
        
        # Summary
        print(f"\n📝 SUMMARY:")
        for line in report.summary.split('\n'):
            print(f"   {line}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test the Fairness Auditor
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    
    print("="*70)
    print("Testing Fairness Auditor (FIRST STEP)")
    print("="*70)
    
    auditor = FairnessAuditor(fairness_threshold=0.05)
    
    # Test on all 3 datasets
    datasets = [
        ('german_credit', ['personal_status']),
        ('german_credit', ['personal_status', 'age'])
    ]
    
    for dataset_name, sensitive_cols in datasets:
        print(f"\n\n{'='*70}")
        print(f"AUDITING: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load data
        split_data = load_split(dataset_name, seed=42)
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        
        # Get sensitive features
        available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
        if not available_sensitive_cols:
            # Try to find any categorical column
            categorical_cols = X_train.select_dtypes(include=['int', 'float']).columns.tolist()
            if categorical_cols:
                available_sensitive_cols = [categorical_cols[0]]
        
        if not available_sensitive_cols:
            print(f"⚠️ No sensitive features found for {dataset_name}")
            continue
        
        sensitive_features = X_train[available_sensitive_cols]
        
        # Audit fairness
        report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
        
        # Print report
        auditor.print_report(report, dataset_name)
    
    print("\n✅ Fairness Auditor Test Complete")

