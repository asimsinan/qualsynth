"""
Universal Validation Pipeline for Fair Comparison Across All Methods

This module provides a consistent validation and selection pipeline that can be
applied to any oversampling method (Qualsynth, CTGAN, SMOTE, TabFairGDT).

The pipeline includes:
1. Duplicate detection (hash-based exact dedup)
2. Quality validation (schema + statistical)
3. Multi-objective selection (diversity + novelty)

Uses adaptive threshold calculation for fair comparison.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class ValidationResult:
    """Result of validation pipeline."""
    X_validated: pd.DataFrame
    y_validated: pd.Series
    n_original: int
    n_after_dedup: int
    n_after_quality: int
    n_after_selection: int
    duplicate_ratio: float
    quality_pass_rate: float
    selection_rate: float
    overall_pass_rate: float
    metrics: Dict[str, Any]


class UniversalValidator:
    """
    Universal validation pipeline for fair comparison across all methods.
    
    Applies the same validation logic used in Qualsynth to all baseline methods.
    """
    
    def __init__(
        self,
        duplicate_threshold: float = 0.10,
        quality_threshold: float = 0.3,
        diversity_weight: float = 0.30,
        fairness_weight: float = 0.40,
        performance_weight: float = 0.30,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        use_adaptive_threshold: bool = True,
        statistical_std_threshold: float = 4.5,
        enable_semantic_dedup: bool = False,
        validation_mode: str = "standard",
        # Legacy parameters (kept for API compatibility, but not used)
        enable_classifier_filter: bool = False,
        classifier_threshold: float = 0.3,
        enable_boundary_filter: bool = False,
        boundary_percentile: float = 80.0
    ):
        """
        Initialize universal validator.
        
        Args:
            duplicate_threshold: Base Euclidean distance threshold for semantic duplicates
            quality_threshold: Minimum quality score for validation
            diversity_weight: Weight for diversity in selection
            fairness_weight: Weight for fairness in selection
            performance_weight: Weight for performance in selection
            max_samples: Maximum samples to select (None = no limit)
            verbose: Print progress messages
            use_adaptive_threshold: Use adaptive threshold calculation
            statistical_std_threshold: Number of standard deviations for statistical validation
            enable_semantic_dedup: Enable semantic duplicate detection (default: False - exact only)
        """
        self.duplicate_threshold = duplicate_threshold
        self.quality_threshold = quality_threshold
        self.diversity_weight = diversity_weight
        self.fairness_weight = fairness_weight
        self.performance_weight = performance_weight
        self.max_samples = max_samples
        self.verbose = verbose
        self.use_adaptive_threshold = use_adaptive_threshold
        self.statistical_std_threshold = statistical_std_threshold
        self.enable_semantic_dedup = enable_semantic_dedup
        self.validation_mode = validation_mode
        self._adaptive_threshold = None
        self._threshold_cache = {}
        self._active_validation_mode = validation_mode
    
    def validate_and_select(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None,
        method_name: str = "Unknown"
    ) -> ValidationResult:
        """
        Apply full validation and selection pipeline.
        
        Args:
            X_generated: Generated samples
            y_generated: Generated labels
            X_train: Original training data (for duplicate checking)
            y_train: Original training labels
            sensitive_features: Sensitive features for fairness validation (unused)
            method_name: Name of the generation method
        
        Returns:
            ValidationResult with validated samples and statistics
        """
        n_original = len(X_generated)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Universal Validation Pipeline: {method_name}")
            print(f"{'='*80}")
            print(f"Input: {n_original} generated samples")

        self._active_validation_mode = self._resolve_validation_mode(X_generated, X_train)
        if self.verbose:
            print(f"Validation mode: {self._active_validation_mode}")
        
        # Calculate adaptive threshold if enabled
        if self.use_adaptive_threshold:
            class_counts = y_train.value_counts()
            imbalance_ratio = round(class_counts.max() / class_counts.min(), 2) if class_counts.min() > 0 else 1.0
            cache_key = (len(X_train), X_train.shape[1], imbalance_ratio)
            
            if cache_key in self._threshold_cache:
                self._adaptive_threshold = self._threshold_cache[cache_key]
                if self.verbose:
                    print(f"\n📊 Adaptive Threshold: {self._adaptive_threshold:.3f} (cached)")
            else:
                self._adaptive_threshold = self._calculate_adaptive_threshold(X_train, y_train)
                self._threshold_cache[cache_key] = self._adaptive_threshold
                if self.verbose:
                    print(f"\n📊 Adaptive Threshold: {self._adaptive_threshold:.3f} (calculated)")
        else:
            self._adaptive_threshold = self.duplicate_threshold
        
        # Step 1: Remove duplicates
        X_dedup, y_dedup, dup_stats = self._remove_duplicates(
            X_generated, y_generated, X_train
        )
        n_after_dedup = len(X_dedup)
        
        if self.verbose:
            print(f"\nStep 1: Duplicate Removal")
            print(f"  Exact duplicates: {dup_stats['exact_duplicates']}")
            print(f"  Semantic duplicates: {dup_stats['semantic_duplicates']}")
            print(f"  Remaining: {n_after_dedup} ({n_after_dedup/n_original*100:.1f}%)")
        
        # Step 2: Quality validation
        X_valid, y_valid, quality_stats = self._validate_quality(
            X_dedup, y_dedup, X_train
        )
        n_after_quality = len(X_valid)
        
        if self.verbose:
            print(f"\nStep 2: Quality Validation")
            print(f"  Schema valid: {quality_stats['schema_valid']}")
            print(f"  Statistical valid: {quality_stats['statistical_valid']}")
            pct = (n_after_quality/n_after_dedup*100) if n_after_dedup > 0 else 0.0
            print(f"  Remaining: {n_after_quality} ({pct:.1f}%)")
        
        # Step 3: Multi-objective selection
        X_selected, y_selected, selection_stats = self._select_best_samples(
            X_valid, y_valid, X_train
        )
        n_after_selection = len(X_selected)
        
        if self.verbose:
            print(f"\nStep 3: Multi-Objective Selection")
            print(f"  Selected: {n_after_selection} samples")
            selection_pct = (n_after_selection/n_after_quality*100) if n_after_quality > 0 else 0.0
            print(f"  Selection rate: {selection_pct:.1f}%")
            print(f"\nFinal Result:")
            print(f"  Original: {n_original} samples")
            print(f"  Final: {n_after_selection} samples")
            overall_pct = (n_after_selection/n_original*100) if n_original > 0 else 0.0
            print(f"  Overall pass rate: {overall_pct:.1f}%")
            print(f"{'='*80}\n")
        
        metrics = {
            'duplicate_stats': dup_stats,
            'quality_stats': quality_stats,
            'selection_stats': selection_stats
        }
        
        return ValidationResult(
            X_validated=X_selected,
            y_validated=y_selected,
            n_original=n_original,
            n_after_dedup=n_after_dedup,
            n_after_quality=n_after_quality,
            n_after_selection=n_after_selection,
            duplicate_ratio=(n_original - n_after_dedup) / n_original if n_original > 0 else 0,
            quality_pass_rate=n_after_quality / n_after_dedup if n_after_dedup > 0 else 0,
            selection_rate=n_after_selection / n_after_quality if n_after_quality > 0 else 0,
            overall_pass_rate=n_after_selection / n_original if n_original > 0 else 0,
            metrics=metrics
        )
    
    def _remove_duplicates(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
        """Remove exact and semantic duplicates."""
        # Step 1: Remove exact duplicates after deterministic numeric
        # canonicalisation. Equivalent values can otherwise differ only in
        # floating-point rendering and evade a string-based hash while becoming
        # identical in a CSV artifact.
        X_with_hash = X_generated.copy()
        X_with_hash['_hash'] = self._canonical_row_hashes(X_generated)
        
        # Check against training data
        X_train_hash = X_train.copy()
        X_train_hash['_hash'] = self._canonical_row_hashes(X_train)
        train_hashes = set(X_train_hash['_hash'])
        
        # Remove duplicates within generated samples and against training
        mask_exact = ~X_with_hash.duplicated(subset=['_hash'], keep='first')
        mask_train = ~X_with_hash['_hash'].isin(train_hashes)
        mask_combined = mask_exact & mask_train
        
        # Reset indices to avoid alignment issues
        X_generated_reset = X_generated.reset_index(drop=True)
        y_generated_reset = y_generated.reset_index(drop=True)
        mask_combined_values = mask_combined.values if hasattr(mask_combined, 'values') else mask_combined
        
        X_no_exact = X_generated_reset[mask_combined_values].copy()
        y_no_exact = y_generated_reset[mask_combined_values].copy()
        
        exact_duplicates = len(X_generated) - len(X_no_exact)
        
        # Step 2: Remove semantic duplicates (only if enabled)
        if self.enable_semantic_dedup and len(X_no_exact) > 0:
            X_semantic, y_semantic, semantic_dups = self._remove_semantic_duplicates(
                X_no_exact, y_no_exact, X_train
            )
        else:
            X_semantic, y_semantic = X_no_exact, y_no_exact
            semantic_dups = 0
        
        stats = {
            'exact_duplicates': exact_duplicates,
            'semantic_duplicates': semantic_dups,
            'total_duplicates': exact_duplicates + semantic_dups
        }
        
        return X_semantic, y_semantic, stats

    @staticmethod
    def _canonical_row_hashes(frame: pd.DataFrame) -> pd.Series:
        """Hash rows after normalising insignificant floating-point noise."""
        canonical = frame.copy()
        numeric_columns = canonical.select_dtypes(include=[np.number]).columns
        if len(numeric_columns):
            canonical.loc[:, numeric_columns] = canonical.loc[:, numeric_columns].round(12)
        return pd.util.hash_pandas_object(canonical, index=False).astype(str)
    
    def _remove_semantic_duplicates(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, int]:
        """Remove semantic duplicates using adaptive Euclidean distance threshold."""
        if len(X_generated) == 0:
            return X_generated, y_generated, 0
        
        threshold = self._adaptive_threshold if self._adaptive_threshold else self.duplicate_threshold
        
        # Normalize features for distance calculation
        scaler = StandardScaler()
        X_gen_scaled = scaler.fit_transform(X_generated)
        X_train_scaled = scaler.transform(X_train)
        
        keep_mask = np.ones(len(X_generated), dtype=bool)
        
        for i in range(len(X_generated)):
            if not keep_mask[i]:
                continue
            
            distances = np.sqrt(np.sum((X_train_scaled - X_gen_scaled[i])**2, axis=1))
            min_distance = np.min(distances)
            
            if min_distance < threshold:
                keep_mask[i] = False
        
        # Also check within generated samples
        if np.sum(keep_mask) > 1:
            X_remaining_scaled = X_gen_scaled[keep_mask]
            distances = pdist(X_remaining_scaled, metric='euclidean')
            dist_matrix = squareform(distances)
            
            indices = np.where(keep_mask)[0]
            for i in range(len(dist_matrix)):
                if not keep_mask[indices[i]]:
                    continue
                for j in range(i + 1, len(dist_matrix)):
                    if dist_matrix[i, j] < threshold:
                        keep_mask[indices[j]] = False
        
        semantic_dups = len(X_generated) - np.sum(keep_mask)
        
        X_generated_reset = X_generated.reset_index(drop=True)
        y_generated_reset = y_generated.reset_index(drop=True)
        
        return X_generated_reset[keep_mask].copy(), y_generated_reset[keep_mask].copy(), semantic_dups
    
    def _validate_quality(
        self,
        X_generated: pd.DataFrame,
        y_generated: pd.Series,
        X_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
        """Validate sample quality (schema, statistical)."""
        if len(X_generated) == 0:
            return X_generated, y_generated, {
                'schema_valid': 0,
                'statistical_valid': 0,
                'total_valid': 0
            }
        
        # Schema validation: check data types and ranges
        if self._active_validation_mode == "high_dimensional":
            schema_mask = self._validate_high_dimensional_schema(X_generated, X_train)
            statistical_mask, statistical_details = self._validate_high_dimensional_statistical(
                X_generated,
                X_train,
            )
        else:
            schema_mask = self._validate_schema(X_generated, X_train)
            statistical_mask = self._validate_statistical(X_generated, X_train)
            statistical_details = {}
        
        # Combine all validation masks
        combined_mask = schema_mask & statistical_mask
        
        stats = {
            'schema_valid': np.sum(schema_mask),
            'statistical_valid': np.sum(statistical_mask),
            'total_valid': np.sum(combined_mask),
            'validation_mode': self._active_validation_mode,
            **statistical_details,
        }
        
        # Reset indices and convert mask to numpy array
        X_generated_reset = X_generated.reset_index(drop=True)
        y_generated_reset = y_generated.reset_index(drop=True)
        
        if hasattr(combined_mask, 'values'):
            combined_mask = combined_mask.values
        
        return X_generated_reset[combined_mask].copy(), y_generated_reset[combined_mask].copy(), stats

    def _resolve_validation_mode(
        self,
        X_generated: pd.DataFrame,
        X_train: pd.DataFrame,
    ) -> str:
        """Resolve validation mode for the current dataset."""
        mode = (self.validation_mode or "standard").lower()
        if mode != "auto":
            return mode

        n_features = max(X_generated.shape[1], X_train.shape[1])
        n_train = len(X_train)
        if n_features >= 30 and n_train < n_features:
            return "high_dimensional"
        return "standard"
    
    def _validate_schema(
        self,
        X_generated: pd.DataFrame,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        """Validate schema (data types, ranges)."""
        mask = np.ones(len(X_generated), dtype=bool)
        
        for col in X_generated.columns:
            if col not in X_train.columns:
                continue
            
            # Check for NaN/inf
            mask &= ~X_generated[col].isna()
            mask &= ~np.isinf(X_generated[col].replace([np.inf, -np.inf], np.nan))
            
            # Check ranges (within training data range + 20% margin)
            if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                train_min = X_train[col].min()
                train_max = X_train[col].max()
                train_range = train_max - train_min
                margin = max(0.2 * train_range, 0.5)
                
                min_val = train_min - margin
                max_val = train_max + margin
                mask &= (X_generated[col] >= min_val) & (X_generated[col] <= max_val)
        
        return mask

    def _validate_high_dimensional_schema(
        self,
        X_generated: pd.DataFrame,
        X_train: pd.DataFrame,
    ) -> np.ndarray:
        """Schema checks for small-n, high-p data: finite values and aligned columns."""
        mask = np.ones(len(X_generated), dtype=bool)
        common_cols = [col for col in X_generated.columns if col in X_train.columns]
        if len(common_cols) != len(X_train.columns):
            return np.zeros(len(X_generated), dtype=bool)

        numeric = X_generated[common_cols].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if finite.ndim == 2:
            mask &= finite.all(axis=1)
        return mask
    
    def _validate_statistical(
        self,
        X_generated: pd.DataFrame,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        """Validate statistical properties (z-score bounds check)."""
        mask = np.ones(len(X_generated), dtype=bool)
        
        for col in X_generated.columns:
            if col not in X_train.columns:
                continue
            
            if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean = X_train[col].mean()
                std = X_train[col].std()
                if std > 0:
                    z_scores = np.abs((X_generated[col] - mean) / std)
                    mask &= z_scores < self.statistical_std_threshold
        
        return mask

    def _validate_high_dimensional_statistical(
        self,
        X_generated: pd.DataFrame,
        X_train: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Validate high-dimensional samples in PCA space instead of per-feature z-scores."""
        if len(X_generated) == 0:
            return np.zeros(0, dtype=bool), {}

        numeric_cols = [
            col for col in X_train.columns
            if col in X_generated.columns and pd.api.types.is_numeric_dtype(X_train[col])
        ]
        if not numeric_cols:
            return np.ones(len(X_generated), dtype=bool), {
                "high_dimensional_reason": "no_numeric_columns",
            }

        X_train_num = X_train[numeric_cols].apply(pd.to_numeric, errors="coerce")
        X_gen_num = X_generated[numeric_cols].apply(pd.to_numeric, errors="coerce")
        finite_mask = np.isfinite(X_gen_num.to_numpy(dtype=float)).all(axis=1)
        if len(X_train_num) < 3 or len(numeric_cols) < 2:
            return finite_mask, {
                "high_dimensional_reason": "insufficient_reference_samples",
            }

        train_means = X_train_num.mean(axis=0)
        X_train_num = X_train_num.fillna(train_means)
        X_gen_num = X_gen_num.fillna(train_means)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train_num)
        gen_scaled = scaler.transform(X_gen_num)

        n_components = min(10, len(X_train_num) - 1, len(numeric_cols))
        if n_components < 1:
            return finite_mask, {
                "high_dimensional_reason": "no_pca_components",
            }

        pca = PCA(n_components=n_components, random_state=0)
        train_latent = pca.fit_transform(train_scaled)
        gen_latent = pca.transform(gen_scaled)

        nn = NearestNeighbors(n_neighbors=min(2, len(train_latent)))
        nn.fit(train_latent)
        train_distances, _ = nn.kneighbors(train_latent)
        train_nn = train_distances[:, 1] if train_distances.shape[1] > 1 else train_distances[:, 0]

        gen_distances, _ = nn.kneighbors(gen_latent, n_neighbors=1)
        gen_nn = gen_distances[:, 0]
        nn_threshold = max(float(np.quantile(train_nn, 0.95)) * 3.0, 1e-8)

        train_reconstructed = pca.inverse_transform(train_latent)
        gen_reconstructed = pca.inverse_transform(gen_latent)
        train_reconstruction_error = np.linalg.norm(
            train_scaled - train_reconstructed,
            axis=1,
        ) / np.sqrt(len(numeric_cols))
        gen_reconstruction_error = np.linalg.norm(
            gen_scaled - gen_reconstructed,
            axis=1,
        ) / np.sqrt(len(numeric_cols))
        reconstruction_threshold = max(
            float(np.quantile(train_reconstruction_error, 0.995)) * 5.0,
            float(train_reconstruction_error.mean() + 5.0 * train_reconstruction_error.std()),
            1e-8,
        )

        latent_mask = gen_nn <= nn_threshold
        reconstruction_mask = gen_reconstruction_error <= reconstruction_threshold
        mask = finite_mask & latent_mask & reconstruction_mask

        return mask, {
            "high_dimensional_components": int(n_components),
            "high_dimensional_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "high_dimensional_nn_threshold": nn_threshold,
            "high_dimensional_reconstruction_threshold": reconstruction_threshold,
            "high_dimensional_nn_valid": int(np.sum(latent_mask)),
            "high_dimensional_reconstruction_valid": int(np.sum(reconstruction_mask)),
        }
    
    def _select_best_samples(
        self,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Select best samples using multi-objective optimization."""
        if len(X_valid) == 0:
            return X_valid, y_valid, {'n_selected': 0}
        
        # If max_samples is set and we have more samples, select best ones
        if self.max_samples is not None and len(X_valid) > self.max_samples:
            diversity_scores = self._calculate_diversity_scores(X_valid)
            novelty_scores = self._calculate_novelty_scores(X_valid, X_train)
            
            combined_scores = (
                self.diversity_weight * diversity_scores +
                (1 - self.diversity_weight) * novelty_scores
            )
            
            top_indices = np.argsort(combined_scores)[-self.max_samples:]
            X_selected = X_valid.iloc[top_indices].copy()
            y_selected = y_valid.iloc[top_indices].copy()
        else:
            X_selected = X_valid.copy()
            y_selected = y_valid.copy()
        
        stats = {
            'n_selected': len(X_selected),
            'n_candidates': len(X_valid)
        }
        
        return X_selected, y_selected, stats
    
    def _calculate_diversity_scores(
        self,
        X_valid: pd.DataFrame
    ) -> np.ndarray:
        """Calculate diversity scores for samples."""
        scaler = StandardScaler()
        X_valid_scaled = scaler.fit_transform(X_valid)
        
        if len(X_valid) > 1:
            distances = pdist(X_valid_scaled, metric='euclidean')
            dist_matrix = squareform(distances)
            diversity_scores = np.mean(dist_matrix, axis=1)
        else:
            diversity_scores = np.ones(len(X_valid))
        
        if np.max(diversity_scores) > 0:
            diversity_scores = diversity_scores / np.max(diversity_scores)
        
        return diversity_scores
    
    def _calculate_novelty_scores(
        self,
        X_valid: pd.DataFrame,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        """Calculate novelty scores (distance from training data)."""
        scaler = StandardScaler()
        X_valid_scaled = scaler.fit_transform(X_valid)
        X_train_scaled = scaler.transform(X_train)
        
        novelty_scores = np.zeros(len(X_valid))
        for i in range(len(X_valid)):
            distances = np.sqrt(np.sum((X_train_scaled - X_valid_scaled[i])**2, axis=1))
            novelty_scores[i] = np.min(distances)
        
        if np.max(novelty_scores) > 0:
            novelty_scores = novelty_scores / np.max(novelty_scores)
        
        return novelty_scores
    
    def _calculate_adaptive_threshold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> float:
        """
        Calculate adaptive duplicate detection threshold.
        
        Based on:
        1. Dataset size (logarithmic scaling)
        2. Intrinsic dimensionality (feature space complexity)
        3. Class imbalance ratio
        4. Feature space density
        
        Returns:
            Adaptive distance threshold (0.05 - 0.50)
        """
        base_threshold = 0.10
        
        # 1. Dataset size factor
        n_samples = len(X_train)
        size_adjustment = 0.02 * np.log10(max(n_samples, 10))
        
        # 2. Intrinsic dimensionality factor
        try:
            n_features = X_train.shape[1]
            n_components = min(n_features, n_samples, 50)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = np.searchsorted(cumsum, 0.95) + 1
            
            dim_ratio = intrinsic_dim / n_features
            dim_adjustment = 0.05 * dim_ratio
        except Exception:
            dim_adjustment = 0.02
        
        # 3. Class imbalance factor
        class_counts = y_train.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else 1.0
        imbalance_adjustment = -0.01 * np.log2(max(imbalance_ratio, 1.0))
        
        # 4. Feature space density factor
        try:
            sample_size = min(500, len(X_train))
            sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_scaled[sample_idx]
            
            nn = NearestNeighbors(n_neighbors=min(6, sample_size))
            nn.fit(X_sample)
            distances, _ = nn.kneighbors(X_sample)
            avg_nn_distance = np.mean(distances[:, 1:])
            
            density_factor = 1.0 / (1.0 + avg_nn_distance)
            density_adjustment = 0.05 * density_factor
        except Exception:
            density_adjustment = 0.02
        
        adaptive_threshold = base_threshold + size_adjustment + dim_adjustment + imbalance_adjustment + density_adjustment
        adaptive_threshold = np.clip(adaptive_threshold, 0.05, 0.50)
        
        if self.verbose:
            print(f"   📊 Adaptive Threshold Calculation:")
            print(f"      Base: {base_threshold:.3f}")
            print(f"      + Size ({n_samples} samples): {size_adjustment:+.3f}")
            print(f"      + Dimensionality: {dim_adjustment:+.3f}")
            print(f"      + Imbalance ({imbalance_ratio:.1f}:1): {imbalance_adjustment:+.3f}")
            print(f"      + Density: {density_adjustment:+.3f}")
            print(f"      = Final: {adaptive_threshold:.3f}")
        
        return float(adaptive_threshold)
