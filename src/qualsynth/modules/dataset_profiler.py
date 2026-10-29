"""
Dataset Profiler for Qualsynth Framework

This component analyzes dataset characteristics and recommends adaptive strategies
for fairness-first generation based on:
- Dataset size (small, medium, large)
- Imbalance ratio (moderate, high, extreme)
- Fairness violations (moderate, severe, catastrophic)
- Feature complexity (simple, moderate, complex)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DatasetSize(Enum):
    """Dataset size categories"""
    TINY = "tiny"  # <500 samples
    SMALL = "small"  # 500-5000 samples
    MEDIUM = "medium"  # 5000-50000 samples
    LARGE = "large"  # >50000 samples


class ImbalanceLevel(Enum):
    """Imbalance ratio categories"""
    MODERATE = "moderate"  # 1.5:1 to 3:1
    HIGH = "high"  # 3:1 to 7:1
    EXTREME = "extreme"  # >7:1


class FairnessViolationLevel(Enum):
    """Fairness violation severity"""
    NONE = "none"  # <0.05 (within threshold)
    MODERATE = "moderate"  # 0.05-0.2 (2-4x)
    SEVERE = "severe"  # 0.2-0.5 (4-10x)
    CATASTROPHIC = "catastrophic"  # >0.5 (>10x)


class StrategyType(Enum):
    """Recommended Qualsynth strategy types"""
    STANDARD = "standard"
    EXTREME_FAIRNESS = "extreme_fairness"
    HIGH_IMBALANCE = "high_imbalance"
    CONSERVATIVE = "conservative"  # For extreme fairness violations


@dataclass
class DatasetProfile:
    """
    Comprehensive dataset profile with recommendations.
    """
    # Basic characteristics
    n_samples: int
    n_features: int
    n_minority: int
    n_majority: int
    imbalance_ratio: float
    
    # Categorizations
    size_category: DatasetSize
    imbalance_level: ImbalanceLevel
    
    # Fairness analysis (if sensitive features provided)
    fairness_violations: Optional[Dict[str, float]] = None
    fairness_level: Optional[FairnessViolationLevel] = None
    underrepresented_groups: Optional[List[str]] = None
    
    # Feature complexity
    n_categorical: int = 0
    n_numerical: int = 0
    feature_complexity: str = "moderate"
    
    # Recommendations
    recommended_strategy: StrategyType = StrategyType.STANDARD
    strategy_config: Dict[str, Any] = None
    
    # Warnings
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.strategy_config is None:
            self.strategy_config = {}
        if self.warnings is None:
            self.warnings = []


class DatasetProfiler:
    """
    Analyzes dataset characteristics and recommends adaptive Qualsynth strategies.
    
    This is the FIRST component in the fairness-first workflow, providing
    critical information for all downstream modules.
    """
    
    def __init__(self, fairness_threshold: float = 0.05):
        """
        Initialize Dataset Profiler.
        
        Args:
            fairness_threshold: Maximum acceptable fairness metric difference (default: 0.05)
        """
        self.fairness_threshold = fairness_threshold
    
    def profile(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None,
        dataset_name: Optional[str] = None
    ) -> DatasetProfile:
        """
        Analyze dataset and recommend strategy.
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_features: Optional sensitive attributes for fairness analysis
            dataset_name: Optional dataset name for logging
        
        Returns:
            DatasetProfile with comprehensive analysis and recommendations
        """
        # Basic statistics
        n_samples = len(X)
        n_features = X.shape[1]
        n_minority = (y == 1).sum()
        n_majority = (y == 0).sum()
        imbalance_ratio = n_majority / n_minority if n_minority > 0 else float('inf')
        
        # Categorize size
        size_category = self._categorize_size(n_samples)
        
        # Categorize imbalance
        imbalance_level = self._categorize_imbalance(imbalance_ratio)
        
        # Feature analysis
        n_categorical, n_numerical = self._analyze_features(X)
        feature_complexity = self._assess_feature_complexity(n_categorical, n_numerical, n_features)
        
        # Fairness analysis (if sensitive features provided)
        fairness_violations = None
        fairness_level = None
        underrepresented_groups = None
        
        if sensitive_features is not None and not sensitive_features.empty:
            fairness_violations = self._analyze_fairness(X, y, sensitive_features)
            fairness_level = self._categorize_fairness_violations(fairness_violations)
            underrepresented_groups = self._identify_underrepresented_groups(
                X, y, sensitive_features
            )
        
        # Create profile
        profile = DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_minority=n_minority,
            n_majority=n_majority,
            imbalance_ratio=imbalance_ratio,
            size_category=size_category,
            imbalance_level=imbalance_level,
            fairness_violations=fairness_violations,
            fairness_level=fairness_level,
            underrepresented_groups=underrepresented_groups,
            n_categorical=n_categorical,
            n_numerical=n_numerical,
            feature_complexity=feature_complexity
        )
        
        # Recommend strategy
        self._recommend_strategy(profile)
        
        # Add warnings
        self._add_warnings(profile)
        
        return profile
    
    def _categorize_size(self, n_samples: int) -> DatasetSize:
        """Categorize dataset size."""
        if n_samples < 500:
            return DatasetSize.TINY
        elif n_samples < 5000:
            return DatasetSize.SMALL
        elif n_samples < 50000:
            return DatasetSize.MEDIUM
        else:
            return DatasetSize.LARGE
    
    def _categorize_imbalance(self, ratio: float) -> ImbalanceLevel:
        """Categorize imbalance level."""
        if ratio < 3.0:
            return ImbalanceLevel.MODERATE
        elif ratio < 7.0:
            return ImbalanceLevel.HIGH
        else:
            return ImbalanceLevel.EXTREME
    
    def _analyze_features(self, X: pd.DataFrame) -> Tuple[int, int]:
        """Count categorical and numerical features."""
        n_categorical = 0
        n_numerical = 0
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                n_categorical += 1
            elif X[col].nunique() < 10:
                n_categorical += 1
            else:
                n_numerical += 1
        
        return n_categorical, n_numerical
    
    def _assess_feature_complexity(self, n_cat: int, n_num: int, n_total: int) -> str:
        """Assess feature complexity."""
        if n_total < 10:
            return "simple"
        elif n_total > 50:
            return "complex"
        else:
            return "moderate"
    
    def _analyze_fairness(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze fairness violations in the dataset.
        
        Computes demographic parity difference for each sensitive attribute.
        """
        violations = {}
        
        for col in sensitive_features.columns:
            if col not in X.columns:
                continue
            
            # Compute demographic parity difference
            groups = X[col].unique()
            if len(groups) == 2:
                g1, g2 = groups
                p1 = y[X[col] == g1].mean()
                p2 = y[X[col] == g2].mean()
                dpd = abs(p1 - p2)
                violations[f'{col}_dpd'] = dpd
        
        return violations
    
    def _categorize_fairness_violations(
        self,
        violations: Dict[str, float]
    ) -> FairnessViolationLevel:
        """Categorize fairness violation severity."""
        if not violations:
            return FairnessViolationLevel.NONE
        
        max_violation = max(violations.values())
        
        if max_violation < self.fairness_threshold:
            return FairnessViolationLevel.NONE
        elif max_violation < 0.2:
            return FairnessViolationLevel.MODERATE
        elif max_violation < 0.5:
            return FairnessViolationLevel.SEVERE
        else:
            return FairnessViolationLevel.CATASTROPHIC
    
    def _identify_underrepresented_groups(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame
    ) -> List[str]:
        """Identify underrepresented minority subgroups."""
        underrep = []
        
        # Get minority samples
        minority_mask = y == 1
        X_minority = X[minority_mask]
        
        for col in sensitive_features.columns:
            if col not in X.columns:
                continue
            
            # Check distribution in minority vs majority
            minority_dist = X_minority[col].value_counts(normalize=True)
            majority_dist = X[~minority_mask][col].value_counts(normalize=True)
            
            for group in minority_dist.index:
                if group in majority_dist.index:
                    # If this group is significantly underrepresented in minority
                    if minority_dist[group] < majority_dist[group] * 0.5:
                        underrep.append(f"{col}={group}")
        
        return underrep
    
    def _recommend_strategy(self, profile: DatasetProfile) -> None:
        """
        Recommend Qualsynth strategy based on dataset characteristics.
        
        Updates profile.recommended_strategy and profile.strategy_config in-place.
        """
        # Decision logic based on baseline findings
        
        # CONSERVATIVE: Catastrophic fairness requiring conservative approach
        if profile.fairness_level == FairnessViolationLevel.CATASTROPHIC:
            profile.recommended_strategy = StrategyType.CONSERVATIVE
            profile.strategy_config = {
                'max_synthetic_ratio': 1.0,  # Allow full 1:1 balance
                'diversity_threshold': 0.85,  # High diversity requirement
                'fairness_iterations': 5,  # Multiple passes
                'constraint_strictness': 'high',
                'validation_passes': 3,
                'fairness_weight': 0.7,  # Prioritize fairness
                'performance_weight': 0.2,
                'diversity_weight': 0.1,
                'counterfactual_generation': True
            }
        
        # EXTREME_FAIRNESS: Severe fairness violations
        elif profile.fairness_level == FairnessViolationLevel.SEVERE:
            profile.recommended_strategy = StrategyType.EXTREME_FAIRNESS
            profile.strategy_config = {
                'max_synthetic_ratio': 1.0,  # Can generate up to 1:1
                'diversity_threshold': 0.75,
                'fairness_iterations': 5,
                'constraint_strictness': 'medium',
                'validation_passes': 2,
                'fairness_weight': 0.8,  # Heavily prioritize fairness
                'performance_weight': 0.15,
                'diversity_weight': 0.05,
                'counterfactual_generation': True,
                'fairness_first_mode': True
            }
        
        # HIGH_IMBALANCE: Extreme imbalance ratio
        elif profile.imbalance_level == ImbalanceLevel.EXTREME:
            profile.recommended_strategy = StrategyType.HIGH_IMBALANCE
            profile.strategy_config = {
                'max_synthetic_ratio': 1.0,
                'diversity_threshold': 0.8,
                'fairness_iterations': 3,
                'constraint_strictness': 'medium',
                'validation_passes': 2,
                'fairness_weight': 0.5,
                'performance_weight': 0.35,
                'diversity_weight': 0.15,
                'duplicate_threshold': 0.03  # Stricter duplicate detection
            }
        
        # STANDARD: Default strategy
        else:
            profile.recommended_strategy = StrategyType.STANDARD
            profile.strategy_config = {
                'max_synthetic_ratio': 1.0,
                'diversity_threshold': 0.75,
                'fairness_iterations': 3,
                'constraint_strictness': 'medium',
                'validation_passes': 1,
                'fairness_weight': 0.5,
                'performance_weight': 0.3,
                'diversity_weight': 0.2
            }
    
    def _add_warnings(self, profile: DatasetProfile) -> None:
        """Add warnings based on dataset characteristics."""
        # Small dataset warning
        if profile.size_category == DatasetSize.TINY:
            profile.warnings.append(
                f"⚠️ TINY dataset ({profile.n_samples} samples). "
                "High risk of overfitting. Using conservative generation."
            )
        elif profile.size_category == DatasetSize.SMALL:
            profile.warnings.append(
                f"⚠️ SMALL dataset ({profile.n_samples} samples). "
                "Using conservative generation to prevent overfitting."
            )
        
        # Extreme imbalance warning
        if profile.imbalance_level == ImbalanceLevel.EXTREME:
            profile.warnings.append(
                f"⚠️ EXTREME imbalance ({profile.imbalance_ratio:.1f}:1). "
                "Generating many synthetic samples with strict duplicate detection."
            )
        
        # Catastrophic fairness warning
        if profile.fairness_level == FairnessViolationLevel.CATASTROPHIC:
            max_violation = max(profile.fairness_violations.values())
            profile.warnings.append(
                f"🚨 CATASTROPHIC fairness violations (max: {max_violation:.4f}, "
                f"{max_violation/self.fairness_threshold:.1f}x over threshold). "
                "Using fairness-first generation with counterfactuals."
            )
        elif profile.fairness_level == FairnessViolationLevel.SEVERE:
            max_violation = max(profile.fairness_violations.values())
            profile.warnings.append(
                f"⚠️ SEVERE fairness violations (max: {max_violation:.4f}, "
                f"{max_violation/self.fairness_threshold:.1f}x over threshold). "
                "Using fairness-aware generation."
            )
        
        # Underrepresented groups warning
        if profile.underrepresented_groups:
            profile.warnings.append(
                f"⚠️ Underrepresented minority subgroups detected: "
                f"{', '.join(profile.underrepresented_groups[:3])}{'...' if len(profile.underrepresented_groups) > 3 else ''}"
            )
    
    def print_profile(self, profile: DatasetProfile, dataset_name: str = "Dataset") -> None:
        """Print comprehensive dataset profile."""
        print(f"\n{'='*70}")
        print(f"DATASET PROFILE: {dataset_name}")
        print(f"{'='*70}")
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"  Total samples:      {profile.n_samples:,}")
        print(f"  Features:           {profile.n_features} ({profile.n_categorical} categorical, {profile.n_numerical} numerical)")
        print(f"  Minority samples:   {profile.n_minority:,}")
        print(f"  Majority samples:   {profile.n_majority:,}")
        print(f"  Imbalance ratio:    {profile.imbalance_ratio:.2f}:1")
        
        print(f"\n📏 CATEGORIZATIONS:")
        print(f"  Size:               {profile.size_category.value.upper()}")
        print(f"  Imbalance level:    {profile.imbalance_level.value.upper()}")
        print(f"  Feature complexity: {profile.feature_complexity.upper()}")
        
        if profile.fairness_violations:
            print(f"\n⚖️  FAIRNESS ANALYSIS:")
            print(f"  Violation level:    {profile.fairness_level.value.upper()}")
            for metric, value in profile.fairness_violations.items():
                violation_factor = value / self.fairness_threshold
                status = "✓" if value < self.fairness_threshold else "✗"
                print(f"  {metric:20s}: {value:.4f} ({violation_factor:.1f}x) {status}")
            
            if profile.underrepresented_groups:
                print(f"\n  Underrepresented groups:")
                for group in profile.underrepresented_groups[:5]:
                    print(f"    • {group}")
                if len(profile.underrepresented_groups) > 5:
                    print(f"    ... and {len(profile.underrepresented_groups) - 5} more")
        
        print(f"\n🎯 RECOMMENDED STRATEGY: {profile.recommended_strategy.value.upper()}")
        print(f"\n  Configuration:")
        for key, value in profile.strategy_config.items():
            print(f"    {key:25s}: {value}")
        
        if profile.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in profile.warnings:
                print(f"  {warning}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test the Dataset Profiler
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    
    print("="*70)
    print("Testing Dataset Profiler")
    print("="*70)
    
    profiler = DatasetProfiler(fairness_threshold=0.05)
    
    # Test on all 3 datasets
    datasets = [
        ('german_credit', ['personal_status']),
        ('german_credit', ['personal_status', 'age'])
    ]
    
    for dataset_name, sensitive_cols in datasets:
        print(f"\n\n{'='*70}")
        print(f"PROFILING: {dataset_name.upper()}")
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
        
        sensitive_features = X_train[available_sensitive_cols] if available_sensitive_cols else None
        
        # Profile dataset
        profile = profiler.profile(X_train, y_train, sensitive_features, dataset_name)
        
        # Print profile
        profiler.print_profile(profile, dataset_name)
    
    print("\n✅ Dataset Profiler Test Complete")

