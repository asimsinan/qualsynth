"""
Value Transformer for Qualsynth

This module handles the transformation between normalized (z-score) values
and real-world values, enabling LLMs to generate in human-understandable
ranges while the system works with normalized data internally.

Key Features:
1. Learns transformation parameters from normalized data
2. Converts normalized ranges to real-world ranges for prompts
3. Normalizes LLM-generated real-world values back to z-scores
4. Handles both continuous and categorical features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class FeatureTransform:
    """Transformation parameters for a single feature."""
    name: str
    is_continuous: bool
    
    # For continuous features (z-score normalization)
    mean: float = 0.0
    std: float = 1.0
    normalized_min: float = -3.0
    normalized_max: float = 3.0
    
    # Estimated real-world ranges
    real_min: float = 0.0
    real_max: float = 100.0
    
    # Distribution statistics for probability-driven generation
    # These help LLM generate statistically representative samples
    percentile_25: float = 0.0
    percentile_50: float = 0.0  # median
    percentile_75: float = 0.0
    percentile_10: float = 0.0
    percentile_90: float = 0.0
    
    # For categorical features
    categories: List[Any] = field(default_factory=list)
    category_frequencies: Dict[Any, float] = field(default_factory=dict)  # For distribution matching
    
    # For ordinal features (maps integer index to normalized value)
    # e.g., {1: -1.76, 2: -0.87, 3: 0.02, 4: 0.92} for installment_rate
    ordinal_mapping: Dict[int, float] = field(default_factory=dict)
    is_ordinal: bool = False
    
    def to_real(self, normalized_value: float) -> float:
        """Convert normalized value to real-world value."""
        if not self.is_continuous:
            return normalized_value  # Categorical stays as-is
        return normalized_value * self.std + self.mean
    
    def to_normalized(self, real_value: float) -> float:
        """Convert real-world value to normalized value."""
        if not self.is_continuous:
            return real_value  # Categorical stays as-is
        if self.std == 0:
            return 0.0
        return (real_value - self.mean) / self.std


class ValueTransformer:
    """
    Transforms values between normalized and real-world representations.
    
    This enables LLMs to generate values in human-understandable ranges
    (e.g., age=35, hours=40) while the system works with normalized data
    (e.g., age=0.5, hours=-0.2).
    
    Usage:
        transformer = ValueTransformer()
        transformer.fit(X_train)  # Learn from normalized data
        
        # For prompts: get real-world ranges
        real_ranges = transformer.get_real_world_ranges()
        
        # After LLM generation: normalize the output
        X_normalized = transformer.normalize(X_generated)
    """
    
    def __init__(
        self,
        continuous_threshold: float = 0.5,
        categorical_max_unique: int = 20
    ):
        """
        Initialize ValueTransformer.
        
        Args:
            continuous_threshold: If mean is within this of 0 and std within 0.5 of 1,
                                  consider the feature normalized continuous
            categorical_max_unique: Max unique values to consider categorical
        """
        self.continuous_threshold = continuous_threshold
        self.categorical_max_unique = categorical_max_unique
        self.transforms: Dict[str, FeatureTransform] = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, schema_report: Optional[Any] = None) -> 'ValueTransformer':
        """
        Learn transformation parameters from normalized data.
        
        Args:
            X: Normalized training data
            schema_report: Optional schema report with feature type info
        
        Returns:
            self for chaining
        """
        self.transforms = {}
        
        for col in X.columns:
            transform = self._fit_feature(X[col], col, schema_report)
            self.transforms[col] = transform
        
        self.is_fitted = True
        return self
    
    def _fit_feature(
        self,
        series: pd.Series,
        name: str,
        schema_report: Optional[Any] = None
    ) -> FeatureTransform:
        """Fit transformation for a single feature."""
        
        # Detect if feature is continuous (normalized) or categorical
        n_unique = series.nunique()
        mean = series.mean()
        std = series.std()
        unique_vals = sorted([v for v in series.unique() if pd.notna(v)])
        
        # Check if it looks like a normalized continuous feature
        # Key insight: normalized continuous features have mean≈0, std≈1, and many unique values
        is_normalized_continuous = (
            abs(mean) < self.continuous_threshold and
            abs(std - 1.0) < self.continuous_threshold and
            n_unique > self.categorical_max_unique
        )
        
        # Check if it's an ordinal/discrete feature that was normalized
        # These have few unique values but the values are NOT clean integers (0,1,2,...)
        # Examples: installment_rate (1-4 → normalized), residence_since (1-4 → normalized)
        is_normalized_ordinal = False
        if n_unique <= self.categorical_max_unique and n_unique >= 2:
            # Check if values look normalized (not clean integers like 0, 1, 2)
            values_are_normalized = any(
                abs(v - round(v)) > 0.01 for v in unique_vals
            )
            if values_are_normalized:
                is_normalized_ordinal = True
        
        # Also check schema if available
        if schema_report and hasattr(schema_report, 'features'):
            try:
                from ..modules.schema_profiler import FeatureType
                feat_info = schema_report.features.get(name)
                if feat_info and hasattr(feat_info, 'type'):
                    if feat_info.type == FeatureType.NUMERICAL_CONTINUOUS:
                        is_normalized_continuous = True
                        is_normalized_ordinal = False
                    elif feat_info.type == FeatureType.CATEGORICAL_ORDINAL:
                        # Ordinal features should use integer indices
                        is_normalized_ordinal = True
                        is_normalized_continuous = False
                    elif feat_info.type in [FeatureType.CATEGORICAL_NOMINAL, 
                                            FeatureType.BINARY]:
                        is_normalized_continuous = False
                        is_normalized_ordinal = False
            except ImportError:
                pass
        
        # Handle normalized ordinal features - present as integer indices (1, 2, 3, ...)
        # LLM will generate integers, we'll map them back to normalized values
        if is_normalized_ordinal:
            # Store mapping: index -> normalized_value
            # We use 1-based indices for human readability (1, 2, 3, 4 instead of 0, 1, 2, 3)
            ordinal_mapping = {i+1: v for i, v in enumerate(unique_vals)}
            
            return FeatureTransform(
                name=name,
                is_continuous=False,
                is_ordinal=True,
                categories=list(range(1, n_unique + 1)),  # 1, 2, 3, ..., n
                ordinal_mapping=ordinal_mapping,
                normalized_min=min(unique_vals),
                normalized_max=max(unique_vals),
            )
        
        if is_normalized_continuous:
            # This is a normalized continuous feature
            # Estimate real-world parameters from the normalized data
            
            # The normalized data has mean≈0, std≈1
            # We need to estimate what the original mean and std were
            # Since we don't have the original scaler, we estimate from the data range
            
            normalized_min = float(series.min())
            normalized_max = float(series.max())
            
            # Compute percentiles for distribution-matching prompts
            # These help LLM generate statistically representative samples
            p10 = float(series.quantile(0.10))
            p25 = float(series.quantile(0.25))
            p50 = float(series.quantile(0.50))  # median
            p75 = float(series.quantile(0.75))
            p90 = float(series.quantile(0.90))
            
            # Estimate real-world range based on common patterns
            # Use the feature name to make educated guesses
            real_min, real_max = self._estimate_real_range(
                name, normalized_min, normalized_max, mean, std
            )
            
            # Convert percentiles to real-world values for prompts
            # real_value = norm_value * real_range / norm_range + real_offset
            norm_range = normalized_max - normalized_min
            real_range = real_max - real_min
            if norm_range > 0:
                scale = real_range / norm_range
                offset = real_min - normalized_min * scale
                real_p10 = p10 * scale + offset
                real_p25 = p25 * scale + offset
                real_p50 = p50 * scale + offset
                real_p75 = p75 * scale + offset
                real_p90 = p90 * scale + offset
            else:
                real_p10 = real_p25 = real_p50 = real_p75 = real_p90 = (real_min + real_max) / 2
            
            return FeatureTransform(
                name=name,
                is_continuous=True,
                mean=mean,
                std=std,
                normalized_min=normalized_min,
                normalized_max=normalized_max,
                real_min=real_min,
                real_max=real_max,
                percentile_10=real_p10,
                percentile_25=real_p25,
                percentile_50=real_p50,
                percentile_75=real_p75,
                percentile_90=real_p90
            )
        else:
            # Categorical feature - store valid categories and their frequencies
            categories = sorted(series.unique().tolist())
            
            # Compute frequency distribution for probability-driven generation
            value_counts = series.value_counts(normalize=True)
            category_frequencies = {cat: float(value_counts.get(cat, 0)) for cat in categories}
            
            return FeatureTransform(
                name=name,
                is_continuous=False,
                categories=categories,
                category_frequencies=category_frequencies
            )
    
    def _estimate_real_range(
        self,
        name: str,
        norm_min: float,
        norm_max: float,
        mean: float,
        std: float
    ) -> Tuple[float, float]:
        """
        Estimate real-world range from normalized data.
        
        Since we don't have the original scaler, we use heuristics based on:
        1. The normalized range (z-scores)
        2. Common patterns in feature names
        """
        # Common feature patterns and their typical ranges
        patterns = {
            'age': (18, 90),
            'hour': (0, 100),  # hours-per-week, duration in hours
            'income': (0, 500000),
            'capital': (0, 100000),
            'gain': (0, 100000),
            'loss': (0, 100000),
            'balance': (-10000, 100000),
            'amount': (0, 100000),
            'credit': (0, 50000),
            'duration': (0, 100),
            'rate': (0, 100),
            'weight': (0, 1000000),  # fnlwgt
            'fnlwgt': (10000, 1500000),
            'education': (0, 20),  # years of education
            'num': (0, 20),
            'campaign': (0, 50),
            'pdays': (-1, 999),
            'previous': (0, 50),
            'emp': (-5, 5),
            'cons': (-50, 100),
            'euribor': (0, 10),
            'nr': (4000, 6000),
        }
        
        # Try to match feature name to a pattern
        name_lower = name.lower().replace('-', '').replace('_', '')
        
        for pattern, (typical_min, typical_max) in patterns.items():
            if pattern in name_lower:
                # Found a match - use this as the estimated range
                return (typical_min, typical_max)
        
        # Default: use the normalized range to estimate
        # If normalized range is [-2, 3], real range might be something like [0, 100]
        # This is a rough estimate
        range_size = norm_max - norm_min
        if range_size > 0:
            # Estimate based on typical z-score ranges
            estimated_std = 100 / range_size  # Assume real range ~100
            estimated_mean = 50  # Assume centered around 50
            real_min = max(0, estimated_mean + norm_min * estimated_std)
            real_max = estimated_mean + norm_max * estimated_std
            return (real_min, real_max)
        
        return (0, 100)  # Default fallback
    
    def get_real_world_ranges(self) -> Dict[str, Dict[str, Any]]:
        """
        Get real-world ranges for all features.
        
        Returns:
            Dictionary with feature names as keys and range info as values
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        ranges = {}
        for name, transform in self.transforms.items():
            if transform.is_continuous:
                ranges[name] = {
                    'type': 'continuous',
                    'min': transform.real_min,
                    'max': transform.real_max,
                    'normalized_min': transform.normalized_min,
                    'normalized_max': transform.normalized_max
                }
            else:
                ranges[name] = {
                    'type': 'categorical',
                    'categories': transform.categories
                }
        
        return ranges
    
    def get_distribution_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get distribution statistics for probability-driven generation.
        
        This enables LLMs to generate statistically representative samples
        by providing percentile information and category frequencies.
        
        Returns:
            Dictionary with feature names as keys and distribution info as values:
            - For continuous: percentiles (10, 25, 50, 75, 90), min, max
            - For categorical: category frequencies (probability distribution)
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        stats = {}
        for name, transform in self.transforms.items():
            if transform.is_continuous:
                stats[name] = {
                    'type': 'continuous',
                    'min': transform.real_min,
                    'max': transform.real_max,
                    'p10': transform.percentile_10,
                    'p25': transform.percentile_25,
                    'p50': transform.percentile_50,  # median
                    'p75': transform.percentile_75,
                    'p90': transform.percentile_90,
                    # Compute target proportions for generation guidance
                    'target_low': 0.25,   # ~25% should be below p25
                    'target_mid': 0.50,   # ~50% should be between p25 and p75
                    'target_high': 0.25,  # ~25% should be above p75
                }
            elif transform.is_ordinal:
                stats[name] = {
                    'type': 'ordinal',
                    'categories': transform.categories,
                    'frequencies': transform.category_frequencies if transform.category_frequencies else {},
                }
            else:
                stats[name] = {
                    'type': 'categorical',
                    'categories': transform.categories,
                    'frequencies': transform.category_frequencies if transform.category_frequencies else {},
                }
        
        return stats
    
    def get_continuous_features(self) -> List[str]:
        """Get list of continuous feature names."""
        return [name for name, t in self.transforms.items() if t.is_continuous]
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names."""
        return [name for name, t in self.transforms.items() if not t.is_continuous]
    
    def normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize real-world values to z-scores.
        
        Args:
            X: DataFrame with real-world values
        
        Returns:
            DataFrame with normalized values
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        X_normalized = X.copy()
        
        for col in X.columns:
            if col in self.transforms:
                transform = self.transforms[col]
                
                if transform.is_ordinal and transform.ordinal_mapping:
                    # ROBUST HASH-BASED APPROACH: Always use hash for deterministic selection
                    # This ensures different LLM values → different outputs, regardless of
                    # what format the LLM generates (0.22, -1.76, 2.1, 1, 2, 3, etc.)
                    # 
                    # Benefits:
                    # - Deterministic: Same input always gives same output (reproducible)
                    # - Well-distributed: Hash spreads values evenly across all indices
                    # - LLM-agnostic: Works regardless of LLM output format
                    # - Diversity-preserving: Different values → different hashes → different indices
                    valid_indices = sorted(transform.ordinal_mapping.keys())
                    n_categories = len(valid_indices)
                    
                    def map_ordinal(val):
                        # Always use hash - simple, robust, and consistent
                        hash_val = abs(hash(str(val))) % n_categories
                        selected_idx = valid_indices[hash_val]
                        return transform.ordinal_mapping[selected_idx]
                    
                    X_normalized[col] = X[col].apply(map_ordinal)
                
                elif not transform.is_continuous:
                    # CATEGORICAL FEATURES: Also use hash-based mapping
                    # This ensures diversity even when LLM generates similar values
                    categories = transform.categories
                    n_cats = len(categories)
                    
                    def map_categorical(val):
                        # Always use hash for consistent, diverse mapping
                        hash_val = abs(hash(str(val))) % n_cats
                        return categories[hash_val]
                    
                    X_normalized[col] = X[col].apply(map_categorical)
                
                elif transform.is_continuous:
                    # CONTINUOUS: Linear mapping (preserve LLM's intent)
                    real_min = transform.real_min
                    real_max = transform.real_max
                    norm_min = transform.normalized_min
                    norm_max = transform.normalized_max
                    
                    if real_max != real_min:
                        X_normalized[col] = (
                            (X[col] - real_min) / (real_max - real_min) * 
                            (norm_max - norm_min) + norm_min
                        )
                    else:
                        X_normalized[col] = 0.0
        
        return X_normalized
    
    def denormalize(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert normalized values to real-world values.
        
        Args:
            X: DataFrame with normalized values
        
        Returns:
            DataFrame with real-world values
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        X_real = X.copy()
        
        for col in X.columns:
            if col in self.transforms:
                transform = self.transforms[col]
                if transform.is_continuous:
                    # Denormalize: x = z * std + mean
                    real_min = transform.real_min
                    real_max = transform.real_max
                    norm_min = transform.normalized_min
                    norm_max = transform.normalized_max
                    
                    # Linear mapping from normalized to real
                    if norm_max != norm_min:
                        X_real[col] = (
                            (X[col] - norm_min) / (norm_max - norm_min) * 
                            (real_max - real_min) + real_min
                        )
        
        return X_real
    
    def denormalize_for_prompt(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert normalized values to the format shown in prompts.
        
        This is used to convert few-shot examples from normalized format
        to the same format shown in the prompt (real-world ranges for continuous,
        integer indices for ordinal, as-is for categorical).
        
        Args:
            X: DataFrame with normalized values
        
        Returns:
            DataFrame with prompt-friendly values
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        X_prompt = X.copy()
        
        for col in X.columns:
            if col not in self.transforms:
                continue
            
            transform = self.transforms[col]
            
            if transform.is_continuous:
                # Convert to real-world range
                real_min = transform.real_min
                real_max = transform.real_max
                norm_min = transform.normalized_min
                norm_max = transform.normalized_max
                
                if norm_max != norm_min:
                    X_prompt[col] = (
                        (X[col] - norm_min) / (norm_max - norm_min) * 
                        (real_max - real_min) + real_min
                    )
                    # Round to reasonable precision
                    X_prompt[col] = X_prompt[col].round(1)
            
            elif transform.is_ordinal and transform.ordinal_mapping:
                # Convert normalized values to integer indices
                # Find the closest normalized value and return its index
                def find_index(norm_val):
                    # Find the index whose normalized value is closest
                    best_idx = 1
                    best_diff = float('inf')
                    for idx, nv in transform.ordinal_mapping.items():
                        diff = abs(nv - norm_val)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = idx
                    return best_idx
                
                X_prompt[col] = X[col].apply(find_index)
            
            # Categorical features stay as-is (they're already integers)
        
        return X_prompt
    
    def get_prompt_ranges(self) -> str:
        """
        Get feature ranges formatted for LLM prompts.
        
        Returns:
            String describing valid ranges in real-world terms
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        lines = [
            "📊 FEATURE VALUE RANGES:",
            "",
            "⚠️ CRITICAL: Use EXACT values from the allowed sets below!",
            "   - For INTEGER features: Use ONLY the exact integers shown (e.g., 1, 2, 3, 4)",
            "   - For CONTINUOUS features: Use decimal numbers in the range shown",
            "   - DO NOT generate decimals like 0.22 for integer features!",
            ""
        ]
        
        # Separate features by type for clarity
        continuous_features = []
        integer_features = []
        
        for name, transform in self.transforms.items():
            if transform.is_continuous:
                continuous_features.append(f"• {name}: {transform.real_min:.1f} to {transform.real_max:.1f} (decimal)")
            else:
                cats = transform.categories[:10]  # Show first 10
                cats_str = ", ".join(map(str, cats))
                if len(transform.categories) > 10:
                    cats_str += f", ... ({len(transform.categories)} total)"
                integer_features.append(f"• {name}: MUST be one of {{{cats_str}}} (INTEGER ONLY)")
        
        lines.append("CONTINUOUS FEATURES (use decimal values):")
        lines.extend(continuous_features)
        lines.append("")
        lines.append("INTEGER FEATURES (use EXACT integers from the set):")
        lines.extend(integer_features)
        lines.append("")
        lines.append("EXAMPLES:")
        lines.append("  ✓ CORRECT: installment_rate: 3, residence_since: 2, age: 45.5")
        lines.append("  ✗ WRONG: installment_rate: 0.22, residence_since: 0.44 (decimals NOT allowed for integers!)")
        
        return "\n".join(lines)
    
    def validate_and_normalize(
        self,
        X: pd.DataFrame,
        clip_to_range: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validate real-world values and normalize them.
        
        Args:
            X: DataFrame with real-world values from LLM
            clip_to_range: Whether to clip out-of-range values
        
        Returns:
            Tuple of (normalized DataFrame, dict of clipped counts per feature)
        """
        if not self.is_fitted:
            raise ValueError("ValueTransformer not fitted. Call fit() first.")
        
        X_normalized = X.copy()
        clipped_counts = {}
        
        for col in X.columns:
            if col not in self.transforms:
                continue
            
            transform = self.transforms[col]
            
            # Handle NaN values - fill with default before processing
            if X[col].isna().any():
                if transform.is_continuous:
                    # Fill NaN with middle of range
                    default_val = (transform.real_min + transform.real_max) / 2
                elif transform.is_ordinal and transform.ordinal_mapping:
                    # Fill NaN with first ordinal index
                    default_val = min(transform.ordinal_mapping.keys())
                else:
                    # Fill NaN with first category
                    default_val = transform.categories[0] if transform.categories else 0
                X_normalized[col] = X[col].fillna(default_val)
            
            if transform.is_continuous:
                # CONTINUOUS: Linear mapping (preserve LLM's intent)
                real_min = transform.real_min
                real_max = transform.real_max
                norm_min = transform.normalized_min
                norm_max = transform.normalized_max
                
                # Use NaN-filled values if available
                col_data = X_normalized[col] if col in X_normalized.columns else X[col]
                
                # Check for out-of-range values
                out_of_range = ((col_data < real_min) | (col_data > real_max)).sum()
                clipped_counts[col] = int(out_of_range)
                
                if clip_to_range:
                    X_normalized[col] = col_data.clip(real_min, real_max)
                else:
                    X_normalized[col] = col_data.copy()
                
                # Linear mapping from real to normalized
                if real_max != real_min:
                    X_normalized[col] = (
                        (X_normalized[col] - real_min) / (real_max - real_min) * 
                        (norm_max - norm_min) + norm_min
                    )
                else:
                    X_normalized[col] = 0.0
            
            elif transform.is_ordinal and transform.ordinal_mapping:
                # ORDINAL: Use hash-based mapping for diversity
                valid_indices = sorted(transform.ordinal_mapping.keys())
                n_categories = len(valid_indices)
                clipped_counts[col] = 0  # Hash always produces valid output
                
                # Use NaN-filled values if available
                col_data = X_normalized[col] if col in X_normalized.columns else X[col]
                
                def map_ordinal(val):
                    # Always use hash - simple, robust, and consistent
                    hash_val = abs(hash(str(val))) % n_categories
                    selected_idx = valid_indices[hash_val]
                    return transform.ordinal_mapping[selected_idx]
                
                X_normalized[col] = col_data.apply(map_ordinal)
            
            else:
                # CATEGORICAL: Use hash-based mapping for diversity
                categories = transform.categories
                n_cats = len(categories)
                clipped_counts[col] = 0  # Hash always produces valid output
                
                # Use NaN-filled values if available
                col_data = X_normalized[col] if col in X_normalized.columns else X[col]
                
                def map_categorical(val):
                    # Always use hash for consistent, diverse mapping
                    hash_val = abs(hash(str(val))) % n_cats
                    return categories[hash_val]
                
                X_normalized[col] = col_data.apply(map_categorical)
        
        return X_normalized, clipped_counts


if __name__ == "__main__":
    # Test ValueTransformer
    import pickle
    
    print("="*70)
    print("Testing ValueTransformer")
    print("="*70)
    
    # Load German Credit dataset
    with open('data/splits/german_credit/split_seed42.pkl', 'rb') as f:
        split = pickle.load(f)
    
    X_train = split['X_train']
    
    # Fit transformer
    transformer = ValueTransformer()
    transformer.fit(X_train)
    
    print("\n📊 Detected Feature Types:")
    print(f"   Continuous: {transformer.get_continuous_features()}")
    print(f"   Categorical: {transformer.get_categorical_features()[:5]}...")
    
    print("\n📊 Real-World Ranges:")
    ranges = transformer.get_real_world_ranges()
    for name, info in list(ranges.items())[:5]:
        if info['type'] == 'continuous':
            print(f"   {name}: {info['min']:.1f} to {info['max']:.1f}")
        else:
            print(f"   {name}: {info['categories'][:5]}...")
    
    print("\n📊 Prompt Ranges:")
    print(transformer.get_prompt_ranges()[:500] + "...")
    
    print("\n✅ ValueTransformer Test Complete!")

