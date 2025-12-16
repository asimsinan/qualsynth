"""
Enhanced Schema Profiler for Qualsynth Framework

This component extracts comprehensive schema information from datasets:
1. Automatic type detection (categorical, numerical, ordinal, binary)
2. Range extraction (min/max, valid categories)
3. Fairness constraints (sensitive attributes, balance requirements)
4. Logical constraints (feature dependencies, mutual exclusions)
5. Statistical constraints (correlations, distributions)

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings


class FeatureType(Enum):
    """Feature type classification."""
    BINARY = "binary"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    NUMERICAL_CONTINUOUS = "numerical_continuous"
    NUMERICAL_DISCRETE = "numerical_discrete"


@dataclass
class FeatureSchema:
    """Schema information for a single feature."""
    name: str
    type: FeatureType
    
    # Range information
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_categories: Optional[List[Any]] = None
    
    # Statistical properties
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    skewness: Optional[float] = None
    
    # Constraints
    is_sensitive: bool = False
    nullable: bool = False
    outlier_threshold: Optional[Tuple[float, float]] = None
    
    # Relationships
    correlated_features: List[Tuple[str, float]] = field(default_factory=list)
    dependent_on: List[str] = field(default_factory=list)


@dataclass
class LogicalConstraint:
    """A logical constraint between features."""
    constraint_type: str  # 'implication', 'mutual_exclusion', 'dependency'
    description: str
    features_involved: List[str]
    rule: str  # Human-readable rule
    
    # For implication: if condition then consequence
    condition: Optional[str] = None
    consequence: Optional[str] = None


@dataclass
class FairnessConstraint:
    """A fairness constraint for generation."""
    sensitive_attribute: str
    constraint_type: str  # 'balance', 'target_proportion', 'counterfactual'
    description: str
    
    # For balance constraints
    target_proportions: Optional[Dict[Any, float]] = None
    
    # For counterfactual constraints
    counterfactual_pairs: Optional[List[Tuple[Any, Any]]] = None
    
    # Priority
    priority: str = "medium"  # 'low', 'medium', 'high'


@dataclass
class DatasetSchema:
    """
    Comprehensive schema for a dataset.
    
    This is the output of the Schema Profiler and will be used by:
    - Generator module to create valid samples
    - Validator module to check sample validity
    - Diversity planner to identify constraint-aware sparse regions
    """
    dataset_name: str
    n_samples: int
    n_features: int
    
    # Feature schemas
    features: Dict[str, FeatureSchema] = field(default_factory=dict)
    
    # Constraints
    logical_constraints: List[LogicalConstraint] = field(default_factory=list)
    fairness_constraints: List[FairnessConstraint] = field(default_factory=list)
    
    # Feature categorization
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    sensitive_features: List[str] = field(default_factory=list)
    
    # Global statistics
    feature_correlations: Optional[pd.DataFrame] = None
    
    # Summary
    summary: str = ""


class SchemaProfiler:
    """
    Enhanced Schema Profiler - extracts comprehensive schema from datasets.
    
    This is a TOOL (not an agent) that performs deterministic analysis.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        outlier_iqr_multiplier: float = 3.0,
        categorical_threshold: int = 20,
        ordinal_detection: bool = True
    ):
        """
        Initialize Schema Profiler.
        
        Args:
            correlation_threshold: Threshold for detecting correlated features
            outlier_iqr_multiplier: IQR multiplier for outlier detection
            categorical_threshold: Max unique values to consider categorical
            ordinal_detection: Whether to attempt ordinal detection
        """
        self.correlation_threshold = correlation_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.categorical_threshold = categorical_threshold
        self.ordinal_detection = ordinal_detection
    
    def profile(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: Optional[List[str]] = None,
        fairness_targets: Optional[List[Any]] = None,
        dataset_name: str = "Dataset"
    ) -> DatasetSchema:
        """
        Extract comprehensive schema from dataset.
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_features: List of sensitive feature names
            fairness_targets: Fairness targets from FairnessAuditor
            dataset_name: Name of the dataset
        
        Returns:
            DatasetSchema with all extracted information
        """
        schema = DatasetSchema(
            dataset_name=dataset_name,
            n_samples=len(X),
            n_features=len(X.columns)
        )
        
        # 1. Detect feature types and extract ranges
        for col in X.columns:
            feature_schema = self._analyze_feature(X[col], col, sensitive_features)
            schema.features[col] = feature_schema
            
            # Categorize
            if feature_schema.type in [FeatureType.BINARY, FeatureType.CATEGORICAL_NOMINAL, 
                                       FeatureType.CATEGORICAL_ORDINAL]:
                schema.categorical_features.append(col)
            else:
                schema.numerical_features.append(col)
            
            if feature_schema.is_sensitive:
                schema.sensitive_features.append(col)
        
        # 2. Extract statistical constraints (correlations)
        schema.feature_correlations = self._compute_correlations(X, schema)
        
        # 3. Detect logical constraints
        schema.logical_constraints = self._detect_logical_constraints(X, y, schema)
        
        # 4. Extract fairness constraints
        has_sensitive = (sensitive_features is not None and not sensitive_features.empty) or schema.sensitive_features
        if has_sensitive:
            schema.fairness_constraints = self._extract_fairness_constraints(
                X, y, schema, fairness_targets
            )
        
        # 5. Generate summary
        schema.summary = self._generate_summary(schema)
        
        return schema
    
    def _analyze_feature(
        self,
        series: pd.Series,
        name: str,
        sensitive_features: Optional[List[str]] = None
    ) -> FeatureSchema:
        """Analyze a single feature and extract its schema."""
        feature = FeatureSchema(name=name, type=FeatureType.NUMERICAL_CONTINUOUS)
        
        # Check if sensitive
        if sensitive_features is not None and not sensitive_features.empty and name in sensitive_features.columns:
            feature.is_sensitive = True
        
        # Check for nulls
        feature.nullable = series.isnull().any()
        
        # Detect type
        unique_values = series.nunique()
        
        if unique_values == 2:
            # Binary
            feature.type = FeatureType.BINARY
            feature.valid_categories = sorted(series.unique().tolist())
            feature.mode = series.mode()[0] if len(series.mode()) > 0 else None
        
        elif unique_values <= self.categorical_threshold:
            # Categorical
            feature.valid_categories = sorted(series.unique().tolist())
            feature.mode = series.mode()[0] if len(series.mode()) > 0 else None
            
            # Try to detect ordinal
            if self.ordinal_detection and self._is_ordinal(series):
                feature.type = FeatureType.CATEGORICAL_ORDINAL
            else:
                feature.type = FeatureType.CATEGORICAL_NOMINAL
        
        else:
            # Check if it's actually a string/object column (high-cardinality categorical)
            if series.dtype == 'object' or not pd.api.types.is_numeric_dtype(series):
                # High-cardinality categorical (e.g., country names, IDs)
                feature.type = FeatureType.CATEGORICAL_NOMINAL
                feature.valid_categories = sorted(series.dropna().unique().tolist())[:100]  # Limit to 100 categories
                feature.mode = series.mode()[0] if len(series.mode()) > 0 else None
            else:
                # Numerical
                feature.min_value = float(series.min())
                feature.max_value = float(series.max())
                feature.mean = float(series.mean())
                feature.std = float(series.std())
                feature.median = float(series.median())
                
                # Check if discrete (all integers)
                try:
                    if series.apply(lambda x: float(x).is_integer()).all():
                        feature.type = FeatureType.NUMERICAL_DISCRETE
                    else:
                        feature.type = FeatureType.NUMERICAL_CONTINUOUS
                except (TypeError, ValueError):
                    feature.type = FeatureType.NUMERICAL_CONTINUOUS
                
                # Compute skewness (only for numerical)
                try:
                    feature.skewness = float(series.skew())
                except:
                    feature.skewness = None
                
                # Detect outliers using IQR (only for numerical)
                try:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.outlier_iqr_multiplier * IQR
                    upper_bound = Q3 + self.outlier_iqr_multiplier * IQR
                    feature.outlier_threshold = (float(lower_bound), float(upper_bound))
                except (TypeError, ValueError):
                    feature.outlier_threshold = None
        
        return feature
    
    def _is_ordinal(self, series: pd.Series) -> bool:
        """
        Heuristic to detect if categorical variable is ordinal.
        
        Checks for patterns like: low/medium/high, small/large, etc.
        """
        unique_vals = [str(v).lower() for v in series.unique()]
        
        # Common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['none', 'some', 'many'],
            ['bad', 'neutral', 'good'],
        ]
        
        for pattern in ordinal_patterns:
            if all(p in unique_vals for p in pattern):
                return True
        
        # Check if all values are numeric strings (e.g., '1', '2', '3')
        try:
            numeric_vals = [float(v) for v in unique_vals]
            return True
        except:
            pass
        
        return False
    
    def _compute_correlations(
        self,
        X: pd.DataFrame,
        schema: DatasetSchema
    ) -> pd.DataFrame:
        """Compute feature correlations and update schema."""
        # Only compute for numerical features
        if not schema.numerical_features:
            return pd.DataFrame()
        
        numerical_df = X[schema.numerical_features]
        corr_matrix = numerical_df.corr()
        
        # Update each feature's correlated_features
        for feat in schema.numerical_features:
            if feat not in corr_matrix.columns:
                continue
            
            correlations = corr_matrix[feat].drop(feat)
            strong_corrs = correlations[abs(correlations) >= self.correlation_threshold]
            
            schema.features[feat].correlated_features = [
                (other_feat, float(corr_val))
                for other_feat, corr_val in strong_corrs.items()
            ]
        
        return corr_matrix
    
    def _detect_logical_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        schema: DatasetSchema
    ) -> List[LogicalConstraint]:
        """
        Detect logical constraints between features.
        
        This is a simplified version - in practice, you might use
        more sophisticated constraint mining algorithms.
        """
        constraints = []
        
        # 1. Detect mutual exclusions (for categorical features)
        for i, feat1 in enumerate(schema.categorical_features):
            for feat2 in schema.categorical_features[i+1:]:
                if self._are_mutually_exclusive(X[feat1], X[feat2]):
                    constraints.append(LogicalConstraint(
                        constraint_type='mutual_exclusion',
                        description=f"{feat1} and {feat2} are mutually exclusive",
                        features_involved=[feat1, feat2],
                        rule=f"If {feat1} is set, {feat2} cannot be set (and vice versa)"
                    ))
        
        # 2. Detect implications (simple version)
        # Example: if age < 18, then certain employment statuses are invalid
        # This is dataset-specific, so we'll add a few common patterns
        
        if 'age' in schema.features and 'workclass' in schema.features:
            constraints.append(LogicalConstraint(
                constraint_type='implication',
                description="Age constraints on employment",
                features_involved=['age', 'workclass'],
                rule="If age < 18, then workclass should not be 'full-time'",
                condition="age < 18",
                consequence="workclass != 'full-time'"
            ))
        
        # 3. Detect dependencies (using correlations)
        for feat_name, feat_schema in schema.features.items():
            if feat_schema.correlated_features:
                for corr_feat, corr_val in feat_schema.correlated_features:
                    if abs(corr_val) > 0.8:  # Very strong correlation
                        constraints.append(LogicalConstraint(
                            constraint_type='dependency',
                            description=f"{feat_name} strongly correlated with {corr_feat}",
                            features_involved=[feat_name, corr_feat],
                            rule=f"{feat_name} and {corr_feat} have correlation {corr_val:.3f}"
                        ))
        
        return constraints
    
    def _are_mutually_exclusive(
        self,
        series1: pd.Series,
        series2: pd.Series,
        threshold: float = 0.95
    ) -> bool:
        """
        Check if two categorical features are mutually exclusive.
        
        Two features are mutually exclusive if when one is set to a non-null
        value, the other is almost always null.
        """
        # Check if both have nulls
        if not series1.isnull().any() or not series2.isnull().any():
            return False
        
        # Check overlap
        both_non_null = (~series1.isnull()) & (~series2.isnull())
        overlap_ratio = both_non_null.sum() / len(series1)
        
        return overlap_ratio < (1 - threshold)
    
    def _extract_fairness_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        schema: DatasetSchema,
        fairness_targets: Optional[List[Any]] = None
    ) -> List[FairnessConstraint]:
        """Extract fairness constraints from sensitive features and targets."""
        constraints = []
        
        sensitive_cols = schema.sensitive_features or []
        
        for col in sensitive_cols:
            if col not in X.columns:
                continue
            
            # Get current distribution in minority class
            minority_mask = y == 1
            minority_dist = X[minority_mask][col].value_counts(normalize=True).to_dict()
            
            # Check if we have fairness targets for this attribute
            target_proportions = None
            priority = "medium"
            
            if fairness_targets:
                # fairness_targets could be a FairnessAuditReport or a list
                targets_list = fairness_targets.fairness_targets if hasattr(fairness_targets, 'fairness_targets') else fairness_targets
                for target in targets_list:
                    if hasattr(target, 'attribute') and target.attribute == col:
                        priority = target.priority
                        # Create target proportions
                        target_proportions = {
                            target.target_group: target.target_proportion
                        }
            
            # Create balance constraint
            constraints.append(FairnessConstraint(
                sensitive_attribute=col,
                constraint_type='balance',
                description=f"Balance {col} distribution in generated samples",
                target_proportions=target_proportions or minority_dist,
                priority=priority
            ))
            
            # Create counterfactual constraint for binary attributes
            if schema.features[col].type == FeatureType.BINARY:
                categories = schema.features[col].valid_categories
                if len(categories) == 2:
                    constraints.append(FairnessConstraint(
                        sensitive_attribute=col,
                        constraint_type='counterfactual',
                        description=f"Generate counterfactual pairs for {col}",
                        counterfactual_pairs=[(categories[0], categories[1])],
                        priority=priority
                    ))
        
        return constraints
    
    def _generate_summary(self, schema: DatasetSchema) -> str:
        """Generate human-readable summary of schema."""
        lines = []
        
        lines.append(f"Dataset: {schema.dataset_name}")
        lines.append(f"Samples: {schema.n_samples:,}, Features: {schema.n_features}")
        lines.append(f"Categorical: {len(schema.categorical_features)}, "
                    f"Numerical: {len(schema.numerical_features)}, "
                    f"Sensitive: {len(schema.sensitive_features)}")
        
        if schema.logical_constraints:
            lines.append(f"Logical constraints: {len(schema.logical_constraints)}")
        
        if schema.fairness_constraints:
            lines.append(f"Fairness constraints: {len(schema.fairness_constraints)}")
        
        return " | ".join(lines)
    
    def print_schema(self, schema: DatasetSchema, verbose: bool = True) -> None:
        """Print comprehensive schema report."""
        print(f"\n{'='*70}")
        print(f"SCHEMA PROFILE: {schema.dataset_name}")
        print(f"{'='*70}")
        
        # Basic info
        print(f"\n📊 BASIC INFORMATION:")
        print(f"   Samples: {schema.n_samples:,}")
        print(f"   Features: {schema.n_features}")
        print(f"   Categorical: {len(schema.categorical_features)}")
        print(f"   Numerical: {len(schema.numerical_features)}")
        print(f"   Sensitive: {len(schema.sensitive_features)}")
        
        # Feature details
        if verbose:
            print(f"\n📋 FEATURE DETAILS:")
            
            # Categorical features
            if schema.categorical_features:
                print(f"\n   Categorical Features ({len(schema.categorical_features)}):")
                for feat_name in schema.categorical_features[:10]:
                    feat = schema.features[feat_name]
                    sensitive_mark = " 🔒" if feat.is_sensitive else ""
                    print(f"      • {feat_name}{sensitive_mark}: {feat.type.value}")
                    if feat.valid_categories:
                        cats_str = ", ".join(str(c) for c in feat.valid_categories[:5])
                        if len(feat.valid_categories) > 5:
                            cats_str += f", ... ({len(feat.valid_categories)} total)"
                        print(f"        Categories: {cats_str}")
                
                if len(schema.categorical_features) > 10:
                    print(f"      ... and {len(schema.categorical_features) - 10} more")
            
            # Numerical features
            if schema.numerical_features:
                print(f"\n   Numerical Features ({len(schema.numerical_features)}):")
                for feat_name in schema.numerical_features[:10]:
                    feat = schema.features[feat_name]
                    sensitive_mark = " 🔒" if feat.is_sensitive else ""
                    print(f"      • {feat_name}{sensitive_mark}: {feat.type.value}")
                    print(f"        Range: [{feat.min_value:.2f}, {feat.max_value:.2f}]")
                    print(f"        Mean: {feat.mean:.2f}, Std: {feat.std:.2f}")
                    if feat.skewness is not None:
                        print(f"        Skewness: {feat.skewness:.2f}")
                
                if len(schema.numerical_features) > 10:
                    print(f"      ... and {len(schema.numerical_features) - 10} more")
        
        # Logical constraints
        if schema.logical_constraints:
            print(f"\n🔗 LOGICAL CONSTRAINTS ({len(schema.logical_constraints)}):")
            for i, constraint in enumerate(schema.logical_constraints[:5], 1):
                print(f"\n   {i}. {constraint.constraint_type.upper()}:")
                print(f"      {constraint.description}")
                print(f"      Rule: {constraint.rule}")
            
            if len(schema.logical_constraints) > 5:
                print(f"\n   ... and {len(schema.logical_constraints) - 5} more")
        
        # Fairness constraints
        if schema.fairness_constraints:
            print(f"\n⚖️  FAIRNESS CONSTRAINTS ({len(schema.fairness_constraints)}):")
            for i, constraint in enumerate(schema.fairness_constraints, 1):
                print(f"\n   {i}. {constraint.constraint_type.upper()} "
                      f"({constraint.sensitive_attribute}):")
                print(f"      {constraint.description}")
                print(f"      Priority: {constraint.priority.upper()}")
                
                if constraint.target_proportions:
                    print(f"      Target proportions:")
                    for group, prop in constraint.target_proportions.items():
                        print(f"         {group}: {prop:.4f}")
                
                if constraint.counterfactual_pairs:
                    print(f"      Counterfactual pairs:")
                    for pair in constraint.counterfactual_pairs:
                        print(f"         {pair[0]} ↔ {pair[1]}")
        
        # Summary
        print(f"\n📝 SUMMARY:")
        print(f"   {schema.summary}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test the Schema Profiler
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    
    print("="*70)
    print("Testing Enhanced Schema Profiler")
    print("="*70)
    
    profiler = SchemaProfiler()
    auditor = FairnessAuditor(fairness_threshold=0.05)
    
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
            categorical_cols = X_train.select_dtypes(include=['int', 'float']).columns.tolist()
            if categorical_cols:
                available_sensitive_cols = [categorical_cols[0]]
        
        # Run fairness audit first (to get targets)
        fairness_targets = None
        if available_sensitive_cols:
            sensitive_features = X_train[available_sensitive_cols]
            audit_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
            fairness_targets = audit_report.fairness_targets
        
        # Profile schema
        schema = profiler.profile(
            X_train,
            y_train,
            sensitive_features=available_sensitive_cols,
            fairness_targets=fairness_targets,
            dataset_name=dataset_name
        )
        
        # Print schema
        profiler.print_schema(schema, verbose=True)
    
    print("\n✅ Schema Profiler Test Complete")

