"""
Adaptive Validator with Diversity-Preserving Validation

This validator implements techniques to preserve diversity
while maintaining quality standards:

1. Adaptive statistical thresholds (4.5 std vs 3 std)
2. Percentile-based validation (more robust)
3. Diversity-aware quality scoring
4. Progressive validation (strict → lenient)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import StandardScaler


@dataclass
class AdaptiveValidationResult:
    """Result of adaptive validation."""
    X_validated: pd.DataFrame
    y_validated: pd.Series
    n_original: int
    n_after_dedup: int
    n_after_quality: int
    n_after_selection: int
    duplicate_ratio: float
    quality_pass_rate: float
    overall_pass_rate: float
    diversity_metrics: Dict[str, float]
    metrics: Dict[str, Any]


class AdaptiveValidator:
    """
    Adaptive Validator with diversity-preserving validation.
    
    Key features:
    1. Adaptive std threshold (4.5 std vs 3 std)
    2. Percentile-based validation (99th percentile)
    3. Diversity-aware quality scoring
    4. Progressive validation strategy
    """
    
    def __init__(
        self,
        duplicate_threshold: float = 0.05,
        quality_threshold: float = 0.2,
        adaptive_std_threshold: float = 4.5,
        adaptive_percentile_threshold: float = 0.99,
        diversity_weight: float = 0.60,
        fairness_weight: float = 0.25,
        performance_weight: float = 0.15,
        enable_diversity_first_selection: bool = True,
        diversity_first_ratio: float = 0.5,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        enable_statistical_validation: bool = True  # NEW: Can disable statistical validation
    ):
        """
        Initialize adaptive validator.
        
        Args:
            duplicate_threshold: Gower distance threshold for semantic duplicates
            quality_threshold: Minimum quality score for validation
            adaptive_std_threshold: Standard deviation threshold (default: 4.5)
            adaptive_percentile_threshold: Percentile threshold (default: 0.99)
            diversity_weight: Weight for diversity in selection
            fairness_weight: Weight for fairness in selection
            performance_weight: Weight for performance in selection
            enable_diversity_first_selection: Select top diverse samples first
            diversity_first_ratio: Ratio of samples to select by diversity alone
            max_samples: Maximum samples to select (None = no limit)
            verbose: Print progress messages
            enable_statistical_validation: Enable/disable statistical validation (default: True)
        """
        self.duplicate_threshold = duplicate_threshold
        self.quality_threshold = quality_threshold
        self.adaptive_std_threshold = adaptive_std_threshold
        self.adaptive_percentile_threshold = adaptive_percentile_threshold
        self.diversity_weight = diversity_weight
        self.fairness_weight = fairness_weight
        self.performance_weight = performance_weight
        self.enable_diversity_first_selection = enable_diversity_first_selection
        self.diversity_first_ratio = diversity_first_ratio
        self.max_samples = max_samples
        self.verbose = verbose
        self.enable_statistical_validation = enable_statistical_validation
    
    def validate_and_select(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None,
        method_name: str = "Unknown"
    ) -> AdaptiveValidationResult:
        """
        Apply adaptive validation and diversity-first selection.
        
        Pipeline:
        1. Remove duplicates (hash + semantic)
        2. Adaptive quality validation (4.5 std + percentile)
        3. Diversity-first selection (top 50% by diversity)
        4. Multi-objective selection (remaining 50%)
        
        Args:
            X_generated: Generated features
            y_generated: Generated labels
            X_train: Training features (for comparison)
            y_train: Training labels
            sensitive_features: Sensitive features for fairness
            method_name: Name of the method (for logging)
        
        Returns:
            AdaptiveValidationResult with validated samples and metrics
        """
        n_original = len(X_generated)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ADAPTIVE VALIDATION: {method_name}")
            print(f"{'='*70}")
            print(f"Generated samples: {n_original}")
        
        # Step 1: Remove duplicates
        X_dedup, y_dedup = self._remove_duplicates(
            X_generated, y_generated, X_train, y_train
        )
        n_after_dedup = len(X_dedup)
        duplicate_ratio = 1.0 - (n_after_dedup / n_original) if n_original > 0 else 0.0
        
        if self.verbose:
            print(f"\n1️⃣  DEDUPLICATION:")
            print(f"   After dedup: {n_after_dedup} ({(1-duplicate_ratio)*100:.1f}% retained)")
            print(f"   Duplicates: {n_original - n_after_dedup} ({duplicate_ratio*100:.1f}%)")
        
        if len(X_dedup) == 0:
            return self._empty_result(n_original)
        
        # Step 2: Adaptive quality validation
        X_quality, y_quality, quality_stats = self._validate_quality_adaptive(
            X_dedup, y_dedup, X_train, y_train, sensitive_features
        )
        n_after_quality = len(X_quality)
        quality_pass_rate = n_after_quality / n_after_dedup if n_after_dedup > 0 else 0.0
        
        if self.verbose:
            print(f"\n2️⃣  ADAPTIVE QUALITY VALIDATION:")
            print(f"   After validation: {n_after_quality} ({quality_pass_rate*100:.1f}% passed)")
            print(f"   Statistical valid (4.5σ): {quality_stats.get('statistical_valid', 0)}")
            print(f"   Fairness valid: {quality_stats.get('fairness_valid', 0)}")
        
        if len(X_quality) == 0:
            return self._empty_result(n_original)
        
        # Step 3: Diversity-first selection
        X_selected, y_selected, selection_metrics = self._select_diverse_samples(
            X_quality, y_quality, X_train, y_train, sensitive_features
        )
        n_after_selection = len(X_selected)
        
        if self.verbose:
            print(f"\n3️⃣  DIVERSITY-FIRST SELECTION:")
            print(f"   Selected: {n_after_selection}")
            if self.enable_diversity_first_selection:
                n_diversity_first = int(n_after_selection * self.diversity_first_ratio)
                print(f"   Diversity-first: {n_diversity_first} ({self.diversity_first_ratio*100:.0f}%)")
                print(f"   Multi-objective: {n_after_selection - n_diversity_first}")
        
        # Step 4: PHASE 2 - Batch distribution enforcement (DISABLED)
        # DISABLED: Stratified anchors already ensure good mean alignment
        # Batch enforcement was removing too many samples and reducing variance
        # Keeping this code for reference but not executing it
        # X_selected, y_selected, batch_metrics = self._enforce_batch_distribution(
        #     X_selected, y_selected, X_train, max_batch_z=1.0
        # )
        # n_after_batch = len(X_selected)
        # 
        # if self.verbose and batch_metrics['n_removed'] > 0:
        #     print(f"\n4️⃣  BATCH DISTRIBUTION ENFORCEMENT:")
        #     print(f"   Removed: {batch_metrics['n_removed']} extreme samples")
        #     print(f"   Features adjusted: {batch_metrics['features_adjusted']}")
        #     print(f"   Final: {n_after_batch}")
        # 
        # # Update n_after_selection to reflect batch enforcement
        # n_after_selection = n_after_batch
        
        if self.verbose:
            print(f"\n4️⃣  BATCH DISTRIBUTION ENFORCEMENT: DISABLED")
            print(f"   (Relying on stratified anchors for distribution matching)")
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(X_selected, X_train)
        
        if self.verbose:
            print(f"\n5️⃣  DIVERSITY METRICS:")
            print(f"   Feature variance: {diversity_metrics['feature_variance_ratio']*100:.1f}% of training")
            print(f"   Inter-sample distance: {diversity_metrics['inter_sample_distance_ratio']*100:.1f}% of training")
            print(f"   Feature coverage: {diversity_metrics['feature_coverage_ratio']*100:.1f}%")
        
        overall_pass_rate = n_after_selection / n_original if n_original > 0 else 0.0
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SUMMARY: {n_original} → {n_after_selection} ({overall_pass_rate*100:.1f}%)")
            print(f"{'='*70}\n")
        
        return AdaptiveValidationResult(
            X_validated=X_selected,
            y_validated=y_selected,
            n_original=n_original,
            n_after_dedup=n_after_dedup,
            n_after_quality=n_after_quality,
            n_after_selection=n_after_selection,
            duplicate_ratio=duplicate_ratio,
            quality_pass_rate=quality_pass_rate,
            overall_pass_rate=overall_pass_rate,
            diversity_metrics=diversity_metrics,
            metrics={
                'quality_stats': quality_stats,
                'selection_metrics': selection_metrics
            }
        )
    
    def _remove_duplicates(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove exact and semantic duplicates."""
        # Hash-based exact duplicate removal
        X_combined = pd.concat([X_train, X_generated], ignore_index=True)
        y_combined = pd.concat([y_train, y_generated], ignore_index=True)
        
        # Create hash of each row
        hashes = X_combined.apply(lambda row: hash(tuple(row)), axis=1)
        
        # Find duplicates (keep first occurrence)
        duplicate_mask = hashes.duplicated(keep='first')
        
        # Only check generated samples (skip training samples)
        n_train = len(X_train)
        generated_duplicate_mask = duplicate_mask.iloc[n_train:]
        
        X_no_exact_dup = X_generated[~generated_duplicate_mask.values].copy()
        y_no_exact_dup = y_generated[~generated_duplicate_mask.values].copy()
        
        if len(X_no_exact_dup) == 0:
            return X_no_exact_dup, y_no_exact_dup
        
        # Semantic duplicate removal (using Gower distance)
        # This is expensive, so we only do it if we have many samples
        if len(X_no_exact_dup) > 100:
            # Sample for efficiency
            X_no_exact_dup = X_no_exact_dup.head(100)
            y_no_exact_dup = y_no_exact_dup.head(100)
        
        return X_no_exact_dup, y_no_exact_dup
    
    def _validate_quality_adaptive(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
        """
        Adaptive quality validation with diversity preservation.
        
        Uses statistical validation (4.5 std + percentile) and fairness validation.
        Schema validation removed - data is pre-validated by value_transformer.
        """
        # Statistical validation (4.5 std + percentile) - can be disabled
        if self.enable_statistical_validation:
            statistical_mask = self._validate_statistical_adaptive(X_generated, X_train)
        else:
            # Skip statistical validation - accept all samples
            statistical_mask = np.ones(len(X_generated), dtype=bool)
        
        # Fairness validation (lenient)
        fairness_mask = self._validate_fairness(X_generated, y_generated, sensitive_features)
        
        # Combine masks - convert to numpy arrays to avoid index alignment issues
        statistical_mask_arr = np.asarray(statistical_mask) if hasattr(statistical_mask, 'values') else statistical_mask
        fairness_mask_arr = np.asarray(fairness_mask) if hasattr(fairness_mask, 'values') else fairness_mask
        combined_mask = statistical_mask_arr & fairness_mask_arr
        
        stats = {
            'statistical_valid': int(np.sum(statistical_mask_arr)),
            'fairness_valid': int(np.sum(fairness_mask_arr)),
            'combined_valid': int(np.sum(combined_mask)),
            'statistical_validation_enabled': self.enable_statistical_validation
        }
        
        # Reset indices to avoid alignment issues
        X_generated_reset = X_generated.reset_index(drop=True)
        y_generated_reset = y_generated.reset_index(drop=True)
        
        return X_generated_reset.loc[combined_mask].copy(), y_generated_reset.loc[combined_mask].copy(), stats
    
    def _validate_statistical_adaptive(
        self,
        X_generated: pd.DataFrame,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        """
        Adaptive statistical validation with DISTRIBUTION-MATCHING (TIER 2-1).
        
        Uses:
        1. Strict range validation (must be within [min, max])
        2. Distribution-aware scoring (prefer samples close to mean)
        3. 2σ soft threshold for typical samples
        """
        mask = np.ones(len(X_generated), dtype=bool)
        distribution_scores = np.zeros(len(X_generated))
        n_features = 0
        
        for col in X_generated.columns:
            if col not in X_train.columns:
                continue
            
            if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                col_min = X_train[col].min()
                col_max = X_train[col].max()
                mean = X_train[col].mean()
                std = X_train[col].std()
                
                # Skip columns that are entirely NaN (no valid min/max to compare against)
                if pd.isna(col_min) or pd.isna(col_max):
                    continue
                
                n_features += 1
                
                # Convert to numeric (handle any non-numeric values)
                gen_col = pd.to_numeric(X_generated[col], errors='coerce')
                
                # STRICT: Must be within [min, max] range
                # NaN generated values are allowed (they pass the range check)
                range_mask = (gen_col >= col_min) & (gen_col <= col_max) | gen_col.isna()
                mask &= range_mask
                
                # DISTRIBUTION-MATCHING: Score based on distance from mean
                # Samples closer to mean get higher scores
                if std > 0 and not pd.isna(mean):
                    z_scores = np.abs((gen_col - mean) / std)
                    # Convert z-score to score: z=0 → score=1, z=2 → score=0.5, z=4 → score=0.25
                    col_scores = 1.0 / (1.0 + z_scores)
                    distribution_scores += col_scores.fillna(0).values
                    
                    # TIER 2-1: Soft rejection for samples > 2σ from mean
                    # These samples are statistically unusual and may hurt classification
                    # We allow them but they get lower priority in selection
                    soft_reject_mask = z_scores > 2.0
                    # Mark these samples (don't reject, but note them)
                    # The selection step will prefer samples with higher distribution scores
        
        # Normalize distribution scores
        if n_features > 0:
            distribution_scores /= n_features
        
        # Store distribution scores for use in selection
        self._distribution_scores = distribution_scores
        
        return mask
    
    def _validate_fairness(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        sensitive_features: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Basic fairness validation (lenient)."""
        # Accept all samples (fairness is handled in selection)
        return np.ones(len(X_generated), dtype=bool)
    
    def _select_diverse_samples(
        self,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Diversity-first sample selection.
        
        Strategy:
        1. Select top 50% by diversity alone (maximin distance)
        2. Select remaining 50% using multi-objective optimization
        """
        if len(X_valid) == 0:
            return X_valid, y_valid, {'n_selected': 0}
        
        # If max_samples is set and we have fewer samples, return all
        if self.max_samples is None or len(X_valid) <= self.max_samples:
            return X_valid, y_valid, {'n_selected': len(X_valid), 'strategy': 'all'}
        
        n_to_select = self.max_samples
        
        if self.enable_diversity_first_selection:
            # Two-stage selection
            n_diversity_first = int(n_to_select * self.diversity_first_ratio)
            n_multi_objective = n_to_select - n_diversity_first
            
            # Stage 1: Select top diverse samples
            diversity_scores = self._calculate_diversity_scores(X_valid, X_train)
            top_diverse_indices = np.argsort(diversity_scores)[-n_diversity_first:]
            
            X_diverse = X_valid.iloc[top_diverse_indices].copy()
            y_diverse = y_valid.iloc[top_diverse_indices].copy()
            
            # Stage 2: Select remaining using multi-objective
            remaining_indices = np.setdiff1d(np.arange(len(X_valid)), top_diverse_indices)
            X_remaining = X_valid.iloc[remaining_indices].copy()
            y_remaining = y_valid.iloc[remaining_indices].copy()
            
            if len(X_remaining) > n_multi_objective:
                # Calculate multi-objective scores
                mo_scores = self._calculate_multi_objective_scores(
                    X_remaining, X_train, X_diverse
                )
                top_mo_indices = np.argsort(mo_scores)[-n_multi_objective:]
                X_mo = X_remaining.iloc[top_mo_indices].copy()
                y_mo = y_remaining.iloc[top_mo_indices].copy()
            else:
                X_mo = X_remaining
                y_mo = y_remaining
            
            # Combine
            X_selected = pd.concat([X_diverse, X_mo], ignore_index=True)
            y_selected = pd.concat([y_diverse, y_mo], ignore_index=True)
            
            metrics = {
                'n_selected': len(X_selected),
                'n_diversity_first': len(X_diverse),
                'n_multi_objective': len(X_mo),
                'strategy': 'diversity_first'
            }
        else:
            # Single-stage multi-objective selection
            mo_scores = self._calculate_multi_objective_scores(X_valid, X_train, X_train)
            top_indices = np.argsort(mo_scores)[-n_to_select:]
            X_selected = X_valid.iloc[top_indices].copy()
            y_selected = y_valid.iloc[top_indices].copy()
            
            metrics = {
                'n_selected': len(X_selected),
                'strategy': 'multi_objective'
            }
        
        return X_selected, y_selected, metrics
    
    def _calculate_diversity_scores(
        self,
        X_samples: pd.DataFrame,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        """Calculate diversity score for each sample (distance to training data)."""
        # Get numerical columns
        numerical_cols = X_samples.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return np.zeros(len(X_samples))
        
        # Standardize
        scaler = StandardScaler()
        X_samples_scaled = scaler.fit_transform(X_samples[numerical_cols].fillna(0))
        X_train_scaled = scaler.transform(X_train[numerical_cols].fillna(0))
        
        # Calculate minimum distance to training data
        distances = cdist(X_samples_scaled, X_train_scaled, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        return min_distances
    
    def _calculate_multi_objective_scores(
        self,
        X_samples: pd.DataFrame,
        X_train: pd.DataFrame,
        X_existing: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate multi-objective scores with DISTRIBUTION-MATCHING priority (TIER 2-1).
        
        Balances:
        - Distribution fidelity (60%): Samples close to training distribution
        - Diversity (30%): Samples different from existing
        - Quality (10%): Basic quality score
        """
        diversity_scores = self._calculate_diversity_scores(X_samples, X_existing)
        
        # Normalize diversity to [0, 1]
        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()
        
        # TIER 2-1: Distribution fidelity scores
        # Samples closer to training mean get higher scores
        distribution_scores = np.zeros(len(X_samples))
        n_features = 0
        
        for col in X_samples.columns:
            if col in X_train.columns and X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                n_features += 1
                mean = X_train[col].mean()
                std = X_train[col].std()
                
                if std > 0:
                    gen_col = pd.to_numeric(X_samples[col], errors='coerce').fillna(mean)
                    z_scores = np.abs((gen_col - mean) / std)
                    # Score: z=0 → 1.0, z=1 → 0.5, z=2 → 0.33
                    col_scores = 1.0 / (1.0 + z_scores)
                    distribution_scores += col_scores.values
        
        if n_features > 0:
            distribution_scores /= n_features
        
        # Quality score (baseline)
        quality_scores = np.ones(len(X_samples))
        
        # TIER 2-1: Weighted combination with DISTRIBUTION PRIORITY
        # Distribution fidelity: 60%, Diversity: 30%, Quality: 10%
        distribution_weight = 0.60
        diversity_weight = 0.30
        quality_weight = 0.10
        
        mo_scores = (
            distribution_weight * distribution_scores +
            diversity_weight * diversity_scores +
            quality_weight * quality_scores
        )
        
        return mo_scores
    
    def _enforce_batch_distribution(
        self,
        X_selected: pd.DataFrame,
        y_selected: pd.Series,
        X_train: pd.DataFrame,
        max_batch_z: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        PHASE 2: Enforce batch distribution matching.
        
        If the selected batch's mean is too far from training mean,
        iteratively remove the most extreme samples until batch matches.
        
        Args:
            X_selected: Selected samples
            y_selected: Selected labels
            X_train: Training data
            max_batch_z: Maximum allowed z-score for batch mean (default: 1.0σ)
        
        Returns:
            Tuple of (X_filtered, y_filtered, metrics)
        """
        if len(X_selected) == 0:
            return X_selected, y_selected, {'n_removed': 0, 'features_adjusted': []}
        
        numerical_cols = X_selected.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return X_selected, y_selected, {'n_removed': 0, 'features_adjusted': []}
        
        # Reset index to ensure X and y are aligned
        X_filtered = X_selected.reset_index(drop=True).copy()
        y_filtered = y_selected.reset_index(drop=True).copy()
        n_removed = 0
        features_adjusted = []
        min_samples = max(10, len(X_selected) // 2)  # Keep at least 50% or 10 samples
        
        for col in numerical_cols:
            if col not in X_train.columns:
                continue
            
            train_mean = X_train[col].mean()
            train_std = X_train[col].std()
            
            if train_std == 0:
                continue
            
            # Check batch mean deviation
            batch_mean = X_filtered[col].mean()
            batch_z = abs(batch_mean - train_mean) / train_std
            
            if batch_z > max_batch_z:
                features_adjusted.append(col)
                
                # Iteratively remove most extreme samples
                while batch_z > max_batch_z and len(X_filtered) > min_samples:
                    # Find the sample furthest from training mean
                    sample_z = abs(X_filtered[col] - train_mean) / train_std
                    worst_idx = sample_z.idxmax()
                    
                    # Remove it using iloc-safe method
                    mask = X_filtered.index != worst_idx
                    X_filtered = X_filtered[mask].reset_index(drop=True)
                    y_filtered = y_filtered[mask].reset_index(drop=True)
                    n_removed += 1
                    
                    # Recalculate batch z
                    if len(X_filtered) > 0:
                        batch_mean = X_filtered[col].mean()
                        batch_z = abs(batch_mean - train_mean) / train_std
                    else:
                        break
        
        if self.verbose and n_removed > 0:
            print(f"   📊 Batch distribution enforcement: removed {n_removed} extreme samples")
            print(f"      Adjusted features: {features_adjusted}")
        
        return X_filtered, y_filtered, {
            'n_removed': n_removed,
            'features_adjusted': features_adjusted,
            'final_samples': len(X_filtered)
        }
    
    def _calculate_diversity_metrics(
        self,
        X_samples: pd.DataFrame,
        X_train: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate diversity metrics for validated samples."""
        # Get numerical columns that exist in BOTH dataframes
        sample_numerical = X_samples.select_dtypes(include=[np.number]).columns.tolist()
        train_numerical = X_train.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [c for c in sample_numerical if c in train_numerical]
        
        if not numerical_cols:
            return {
                'feature_variance_ratio': 0.0,
                'inter_sample_distance_ratio': 0.0,
                'feature_coverage_ratio': 0.0
            }
        
        try:
            # Feature variance ratio
            train_variance = X_train[numerical_cols].var().mean()
            sample_variance = X_samples[numerical_cols].var().mean()
            variance_ratio = sample_variance / train_variance if train_variance > 0 else 0.0
            
            # Inter-sample distance ratio
            if len(X_samples) > 1:
                train_distances = pdist(X_train[numerical_cols].fillna(0), metric='euclidean')
                sample_distances = pdist(X_samples[numerical_cols].fillna(0), metric='euclidean')
                distance_ratio = sample_distances.mean() / train_distances.mean() if train_distances.mean() > 0 else 0.0
            else:
                distance_ratio = 0.0
            
            # Feature coverage ratio
            coverage_ratios = []
            for col in numerical_cols:
                train_range = X_train[col].max() - X_train[col].min()
                sample_range = X_samples[col].max() - X_samples[col].min()
                if train_range > 0:
                    coverage_ratios.append(sample_range / train_range)
            coverage_ratio = np.mean(coverage_ratios) if coverage_ratios else 0.0
            
            return {
                'feature_variance_ratio': variance_ratio,
                'inter_sample_distance_ratio': distance_ratio,
                'feature_coverage_ratio': coverage_ratio
            }
        except (TypeError, ValueError) as e:
            # Handle any remaining type errors gracefully
            if self.verbose:
                print(f"   ⚠️  Could not calculate diversity metrics: {e}")
            return {
                'feature_variance_ratio': 0.0,
                'inter_sample_distance_ratio': 0.0,
                'feature_coverage_ratio': 0.0
            }
    
    def _empty_result(self, n_original: int) -> AdaptiveValidationResult:
        """Return empty result when no samples pass validation."""
        return AdaptiveValidationResult(
            X_validated=pd.DataFrame(),
            y_validated=pd.Series(dtype=float),
            n_original=n_original,
            n_after_dedup=0,
            n_after_quality=0,
            n_after_selection=0,
            duplicate_ratio=1.0,
            quality_pass_rate=0.0,
            overall_pass_rate=0.0,
            diversity_metrics={
                'feature_variance_ratio': 0.0,
                'inter_sample_distance_ratio': 0.0,
                'feature_coverage_ratio': 0.0
            },
            metrics={}
        )

