"""
Fairness Re-Auditor for Qualsynth

This component re-audits fairness after sample generation to:
1. Verify that generated samples improved fairness
2. Compare against original fairness targets
3. Provide feedback for iterative refinement
4. Determine convergence criteria
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
except ImportError:
    warnings.warn("Fairlearn not installed. Install with: pip install fairlearn")
    demographic_parity_difference = None
    equalized_odds_difference = None


@dataclass
class ReauditResult:
    """Result of fairness re-audit."""
    # Fairness metrics (after augmentation)
    new_dpd: Dict[str, float] = field(default_factory=dict)
    new_eod: Dict[str, float] = field(default_factory=dict)
    
    # Comparison with original
    original_dpd: Dict[str, float] = field(default_factory=dict)
    original_eod: Dict[str, float] = field(default_factory=dict)
    
    # Improvement metrics
    dpd_improvement: Dict[str, float] = field(default_factory=dict)
    eod_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Target achievement
    targets_met: Dict[str, bool] = field(default_factory=dict)
    overall_targets_met: bool = False
    
    # Convergence
    converged: bool = False
    convergence_reason: str = ""
    
    # Feedback for next iteration
    feedback: str = ""
    recommended_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    n_iterations: int = 0
    total_samples_generated: int = 0


class FairnessReAuditor:
    """
    Re-audits fairness after sample generation and provides feedback.
    
    This component:
    1. Evaluates fairness of augmented dataset (original + generated)
    2. Compares against original fairness violations
    3. Checks if fairness targets are met
    4. Provides feedback for iterative refinement
    5. Determines convergence criteria
    """
    
    def __init__(
        self,
        fairness_threshold: float = 0.05,
        max_iterations: int = 5,
        improvement_threshold: float = 0.005,  # Lowered from 0.01 to 0.005
        convergence_patience: int = 2,
        min_iterations: int = 3  # NEW: Minimum iterations before allowing convergence
    ):
        """
        Initialize Fairness Re-Auditor.
        
        Args:
            fairness_threshold: Threshold for fairness violations (e.g., 0.05 = 5%)
            max_iterations: Maximum number of refinement iterations
            improvement_threshold: Minimum improvement required to continue (e.g., 0.005 = 0.5%)
            convergence_patience: Number of iterations without improvement before stopping
            min_iterations: Minimum iterations before allowing convergence (default: 3)
        """
        self.fairness_threshold = fairness_threshold
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.convergence_patience = convergence_patience
        self.min_iterations = min_iterations
    
    def reaudit(
        self,
        # Original data
        X_original: pd.DataFrame,
        y_original: pd.Series,
        sensitive_features_original: pd.DataFrame,
        
        # Generated samples
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        
        # Original fairness report
        original_fairness_report: Any,
        
        # Iteration info
        iteration: int = 1,
        total_generated: int = 0,
        
        # Optional: trained model for evaluation
        trained_model: Optional[Any] = None
    ) -> ReauditResult:
        """
        Re-audit fairness after generation.
        
        Args:
            X_original: Original training features
            y_original: Original training labels
            sensitive_features_original: Original sensitive features
            X_generated: Generated samples
            y_generated: Generated labels
            original_fairness_report: Original fairness audit report
            iteration: Current iteration number
            total_generated: Total samples generated so far
            trained_model: Optional trained model for evaluation
        
        Returns:
            ReauditResult with fairness metrics and feedback
        """
        result = ReauditResult(
            n_iterations=iteration,
            total_samples_generated=total_generated
        )
        
        # Create augmented dataset
        X_augmented = pd.concat([X_original, X_generated], axis=0, ignore_index=True)
        y_augmented = pd.concat([y_original, y_generated], axis=0, ignore_index=True)
        
        # Handle None sensitive features
        if sensitive_features_original is None or len(sensitive_features_original.columns) == 0:
            # No sensitive features - skip fairness re-audit
            result.converged = True
            result.convergence_reason = "No sensitive features to audit"
            return result
        
        # Get sensitive features for generated samples
        sensitive_features_generated = X_generated[sensitive_features_original.columns]
        sensitive_features_augmented = pd.concat(
            [sensitive_features_original, sensitive_features_generated],
            axis=0,
            ignore_index=True
        )
        
        # Get original metrics
        if hasattr(original_fairness_report, 'violations'):
            for violation in original_fairness_report.violations:
                attr = violation.attribute
                if violation.metric == 'dpd':
                    result.original_dpd[attr] = abs(violation.value)
                elif violation.metric == 'eod':
                    result.original_eod[attr] = abs(violation.value)
                elif violation.metric == 'eopd':
                    # Equal Opportunity Difference
                    result.original_eod[attr] = abs(violation.value)
        
        # Calculate new fairness metrics
        if trained_model is not None:
            # Use trained model predictions
            y_pred = trained_model.predict(X_augmented)
        else:
            # Use ground truth labels as proxy
            y_pred = y_augmented
        
        # Calculate DPD and EOD for each sensitive attribute
        for attr in sensitive_features_augmented.columns:
            try:
                # Demographic Parity Difference
                if demographic_parity_difference is not None:
                    dpd = demographic_parity_difference(
                        y_augmented,
                        y_pred,
                        sensitive_features=sensitive_features_augmented[attr]
                    )
                    result.new_dpd[attr] = abs(dpd)
                    
                    # Calculate improvement
                    if attr in result.original_dpd:
                        improvement = result.original_dpd[attr] - result.new_dpd[attr]
                        result.dpd_improvement[attr] = improvement
                        
                        # Check if target met
                        result.targets_met[f"{attr}_dpd"] = result.new_dpd[attr] <= self.fairness_threshold
                
                # Equalized Odds Difference
                if equalized_odds_difference is not None:
                    eod = equalized_odds_difference(
                        y_augmented,
                        y_pred,
                        sensitive_features=sensitive_features_augmented[attr]
                    )
                    result.new_eod[attr] = abs(eod)
                    
                    # Calculate improvement
                    if attr in result.original_eod:
                        improvement = result.original_eod[attr] - result.new_eod[attr]
                        result.eod_improvement[attr] = improvement
                        
                        # Check if target met
                        result.targets_met[f"{attr}_eod"] = result.new_eod[attr] <= self.fairness_threshold
            
            except Exception as e:
                warnings.warn(f"Error calculating fairness metrics for {attr}: {str(e)}")
        
        # Determine overall target achievement
        if result.targets_met:
            result.overall_targets_met = all(result.targets_met.values())
        
        # Check convergence
        result.converged, result.convergence_reason = self._check_convergence(
            result, iteration
        )
        
        # Generate feedback
        result.feedback = self._generate_feedback(result, original_fairness_report)
        
        # Generate recommended adjustments
        result.recommended_adjustments = self._generate_adjustments(result)
        
        return result
    
    def _check_convergence(
        self,
        result: ReauditResult,
        iteration: int
    ) -> Tuple[bool, str]:
        """
        Check if convergence criteria are met.
        
        Args:
            result: Current reaudit result
            iteration: Current iteration number
        
        Returns:
            Tuple of (converged, reason)
        """
        # Check 0: Minimum iterations not met - NEVER converge early
        if iteration < self.min_iterations:
            return False, f"Continuing (minimum {self.min_iterations} iterations required)"
        
        # Check 1: All targets met (only after minimum iterations)
        if result.overall_targets_met:
            return True, "All fairness targets achieved"
        
        # Check 2: Maximum iterations reached
        if iteration >= self.max_iterations:
            return True, f"Maximum iterations ({self.max_iterations}) reached"
        
        # Check 3: Insufficient improvement (only after minimum iterations + patience)
        if iteration >= self.min_iterations + self.convergence_patience:
            if result.dpd_improvement or result.eod_improvement:
                max_improvement = max(
                    list(result.dpd_improvement.values()) + list(result.eod_improvement.values())
                )
                if max_improvement < self.improvement_threshold:
                    return True, f"Insufficient improvement (<{self.improvement_threshold})"
        
        # Check 4: Fairness degradation
        if result.dpd_improvement or result.eod_improvement:
            min_improvement = min(
                list(result.dpd_improvement.values()) + list(result.eod_improvement.values())
            )
            if min_improvement < -0.02:  # Fairness got worse by >2%
                return True, "Fairness degradation detected"
        
        # Not converged
        return False, "Continue refinement"
    
    def _generate_feedback(
        self,
        result: ReauditResult,
        original_report: Any
    ) -> str:
        """
        Generate human-readable feedback for next iteration.
        
        Args:
            result: Current reaudit result
            original_report: Original fairness report
        
        Returns:
            Feedback string
        """
        lines = []
        
        lines.append(f"ITERATION {result.n_iterations} FEEDBACK:")
        lines.append("")
        
        # Overall status
        if result.overall_targets_met:
            lines.append("✅ SUCCESS: All fairness targets achieved!")
        elif result.converged:
            lines.append(f"⚠️  CONVERGED: {result.convergence_reason}")
        else:
            lines.append("🔄 CONTINUE: Further refinement needed")
        
        lines.append("")
        lines.append("FAIRNESS METRICS:")
        
        # DPD metrics
        for attr, new_dpd in result.new_dpd.items():
            original_dpd = result.original_dpd.get(attr, 0.0)
            improvement = result.dpd_improvement.get(attr, 0.0)
            target_met = result.targets_met.get(f"{attr}_dpd", False)
            
            status = "✅" if target_met else "❌"
            arrow = "↓" if improvement > 0 else "↑" if improvement < 0 else "→"
            
            lines.append(f"  {status} {attr} DPD: {original_dpd:.4f} {arrow} {new_dpd:.4f} "
                        f"(change: {improvement:+.4f})")
        
        # EOD metrics
        for attr, new_eod in result.new_eod.items():
            original_eod = result.original_eod.get(attr, 0.0)
            improvement = result.eod_improvement.get(attr, 0.0)
            target_met = result.targets_met.get(f"{attr}_eod", False)
            
            status = "✅" if target_met else "❌"
            arrow = "↓" if improvement > 0 else "↑" if improvement < 0 else "→"
            
            lines.append(f"  {status} {attr} EOD: {original_eod:.4f} {arrow} {new_eod:.4f} "
                        f"(change: {improvement:+.4f})")
        
        lines.append("")
        
        # Recommendations
        if not result.converged:
            lines.append("RECOMMENDATIONS FOR NEXT ITERATION:")
            
            # Find attributes that need more work
            needs_work = []
            for attr in result.new_dpd.keys():
                if not result.targets_met.get(f"{attr}_dpd", False):
                    needs_work.append((attr, result.new_dpd[attr], 'DPD'))
            
            for attr in result.new_eod.keys():
                if not result.targets_met.get(f"{attr}_eod", False):
                    needs_work.append((attr, result.new_eod[attr], 'EOD'))
            
            if needs_work:
                lines.append("  Focus on:")
                for attr, value, metric in needs_work:
                    lines.append(f"    • {attr} {metric}: {value:.4f} (target: <{self.fairness_threshold})")
            
            lines.append("")
            lines.append("  Suggested actions:")
            lines.append("    • Increase generation focus on underrepresented groups")
            lines.append("    • Use more counterfactual samples")
            lines.append("    • Adjust fairness weight in optimizer")
        
        return "\n".join(lines)
    
    def _generate_adjustments(
        self,
        result: ReauditResult
    ) -> Dict[str, Any]:
        """
        Generate recommended adjustments for next iteration.
        
        Args:
            result: Current reaudit result
        
        Returns:
            Dictionary of recommended adjustments
        """
        adjustments = {}
        
        if result.converged:
            return adjustments
        
        # Adjust fairness weight based on progress
        avg_dpd_improvement = np.mean(list(result.dpd_improvement.values())) if result.dpd_improvement else 0
        
        if avg_dpd_improvement < 0:
            # Fairness got worse, increase fairness weight
            adjustments['fairness_weight'] = 0.8
            adjustments['counterfactual_ratio'] = 0.9  # More counterfactuals
        elif avg_dpd_improvement < self.improvement_threshold:
            # Slow improvement, moderately increase fairness focus
            adjustments['fairness_weight'] = 0.7
            adjustments['counterfactual_ratio'] = 0.8
        else:
            # Good improvement, maintain current approach
            adjustments['fairness_weight'] = 0.6
            adjustments['counterfactual_ratio'] = 0.7
        
        # Identify which groups need more samples
        priority_groups = []
        for attr in result.new_dpd.keys():
            if not result.targets_met.get(f"{attr}_dpd", False):
                # Find which group is underrepresented
                if hasattr(result, 'group_proportions'):
                    priority_groups.append(attr)
        
        if priority_groups:
            adjustments['priority_attributes'] = priority_groups
        
        # Adjust batch size based on iteration
        if result.n_iterations > 3:
            # Later iterations: smaller, more focused batches
            adjustments['batch_size'] = 25
        else:
            # Early iterations: larger batches
            adjustments['batch_size'] = 50
        
        return adjustments
    
    def print_summary(self, result: ReauditResult):
        """Print a summary of the reaudit result."""
        print("="*70)
        print(f"FAIRNESS RE-AUDIT - ITERATION {result.n_iterations}")
        print("="*70)
        print()
        print(result.feedback)
        print()
        
        if result.recommended_adjustments:
            print("RECOMMENDED ADJUSTMENTS:")
            for key, value in result.recommended_adjustments.items():
                print(f"  • {key}: {value}")
            print()
        
        print("="*70)


if __name__ == "__main__":
    # Test fairness re-auditor
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    
    print("="*70)
    print("Testing Fairness Re-Auditor")
    print("="*70)
    
    # Load dataset
    dataset_name = 'german_credit'
    sensitive_cols = ['race', 'sex']
    
    split_data = load_split(dataset_name, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
    if available_sensitive_cols:
        sensitive_features = X_train[available_sensitive_cols]
    
    # Run initial fairness audit
    print("\nStep 1: Initial Fairness Audit")
    print("-"*70)
    
    auditor = FairnessAuditor(fairness_threshold=0.05)
    original_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
    
    print(f"Original violations: {len(original_report.violations)}")
    for violation in original_report.violations:
        dpd_value = getattr(violation, 'dpd', getattr(violation, 'demographic_parity_difference', 0.0))
        print(f"  • {violation.attribute}: DPD={dpd_value:.4f}")
    
    # Simulate generated samples (use minority class samples as proxy)
    print("\nStep 2: Simulate Generation")
    print("-"*70)
    
    minority_samples = X_train[y_train == 1].head(100)
    minority_labels = y_train[y_train == 1].head(100)
    
    print(f"Generated {len(minority_samples)} samples")
    
    # Re-audit
    print("\nStep 3: Re-Audit Fairness")
    print("-"*70)
    
    reauditor = FairnessReAuditor(
        fairness_threshold=0.05,
        max_iterations=5,
        improvement_threshold=0.01
    )
    
    reaudit_result = reauditor.reaudit(
        X_original=X_train,
        y_original=y_train,
        sensitive_features_original=sensitive_features,
        X_generated=minority_samples,
        y_generated=minority_labels,
        original_fairness_report=original_report,
        iteration=1,
        total_generated=100
    )
    
    # Print summary
    reauditor.print_summary(reaudit_result)
    
    print("\n✅ Fairness Re-Auditor Test Complete")

