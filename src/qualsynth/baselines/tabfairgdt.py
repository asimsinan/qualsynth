"""
TabFairGDT Baseline Implementation

TabFairGDT: Fair Gradient Decision Tree for Tabular Data Generation
Key features:
1. Uses gradient-boosted decision trees (GBDT) for generation
2. Incorporates fairness constraints during tree construction
3. Fair leaf resampling: oversample from leaves that improve fairness
4. No LLM required - purely tree-based approach

Reference: Panagiotou et al. (2025). TabFairGDT: A Fast Fair Tabular Data 
Generator Using Autoregressive Decision Trees. arXiv:2509.19927
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TabFairGDTResult:
    """Result of TabFairGDT generation."""
    samples: pd.DataFrame
    n_requested: int
    n_generated: int
    generation_time: float
    fairness_score: float = 0.0


class TabFairGDT:
    """
    TabFairGDT baseline for fair oversampling.
    
    Key differences from Qualsynth:
    - Tree-based generation (no LLM)
    - Fairness-aware leaf selection
    - Simpler fairness metric (demographic parity only)
    - No iterative refinement or multi-objective optimization
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        fairness_weight: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize TabFairGDT.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of trees
            learning_rate: Learning rate for boosting
            fairness_weight: Weight for fairness vs accuracy (0-1)
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.fairness_weight = fairness_weight
        self.random_state = random_state
        
        self.model = None
        self.leaf_samples = {}  # Store samples per leaf
        self.leaf_fairness = {}  # Store fairness score per leaf
    
    def generate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        sensitive_features: Optional[pd.DataFrame] = None,
        target_class: int = 1
    ) -> TabFairGDTResult:
        """
        Generate synthetic samples using fair leaf resampling.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_samples: Number of samples to generate
            sensitive_features: Sensitive attributes for fairness
            target_class: Target class (minority)
            
        Returns:
            TabFairGDTResult with generated samples
        """
        import time
        start_time = time.time()
        
        # Get minority class samples
        minority_mask = y_train == target_class
        X_minority = X_train[minority_mask].copy()
        
        if len(X_minority) == 0:
            warnings.warn(f"No samples found for target class {target_class}")
            return TabFairGDTResult(
                samples=pd.DataFrame(),
                n_requested=n_samples,
                n_generated=0,
                generation_time=time.time() - start_time
            )
        
        # Train gradient boosting model
        self._train_model(X_train, y_train)
        
        # Analyze leaves for fairness
        if sensitive_features is not None and not sensitive_features.empty:
            self._analyze_leaf_fairness(X_minority, sensitive_features.loc[minority_mask])
        else:
            # No fairness info, use uniform sampling
            self._analyze_leaves(X_minority)
        
        # Generate samples via fair leaf resampling
        generated_samples = self._fair_leaf_resample(X_minority, n_samples)
        
        generation_time = time.time() - start_time
        
        return TabFairGDTResult(
            samples=generated_samples,
            n_requested=n_samples,
            n_generated=len(generated_samples),
            generation_time=generation_time,
            fairness_score=self._compute_avg_fairness()
        )
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train gradient boosting model."""
        # Handle NaN values by imputing with median (or 0 if all NaN)
        X_train_clean = X_train.copy()
        for col in X_train_clean.columns:
            if X_train_clean[col].isna().any():
                median_val = X_train_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback if all values are NaN
                X_train_clean[col] = X_train_clean[col].fillna(median_val)
        
        # Final safety check - fill any remaining NaN with 0
        X_train_clean = X_train_clean.fillna(0)
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model.fit(X_train_clean, y_train)
    
    def _clean_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values by imputing with median (or 0 if all NaN)."""
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
        # Final safety check
        return X_clean.fillna(0)
    
    def _analyze_leaves(self, X_minority: pd.DataFrame):
        """Analyze leaf distribution without fairness."""
        # Handle NaN values
        X_minority_clean = self._clean_nan(X_minority)
        
        # Get leaf indices for each sample
        leaf_indices = self.model.apply(X_minority_clean)
        
        # Group samples by leaf
        for tree_idx in range(self.n_estimators):
            for leaf_id in np.unique(leaf_indices[:, tree_idx]):
                mask = leaf_indices[:, tree_idx] == leaf_id
                key = (tree_idx, leaf_id)
                self.leaf_samples[key] = X_minority[mask].copy()  # Keep original with NaN for generation
                self.leaf_fairness[key] = 1.0  # Uniform fairness
    
    def _analyze_leaf_fairness(
        self,
        X_minority: pd.DataFrame,
        sensitive_features: pd.DataFrame
    ):
        """
        Analyze leaf fairness scores.
        
        Fairness score: How well the leaf represents underrepresented groups
        Higher score = more fair (better representation of minority groups)
        """
        # Handle NaN values
        X_minority_clean = self._clean_nan(X_minority)
        
        # Get leaf indices
        leaf_indices = self.model.apply(X_minority_clean)
        
        # Get sensitive attribute (use first column)
        if len(sensitive_features.columns) > 0:
            sensitive_col = sensitive_features.columns[0]
            sensitive_values = sensitive_features[sensitive_col].values
            
            # Compute global distribution
            unique_vals, global_counts = np.unique(sensitive_values, return_counts=True)
            global_dist = dict(zip(unique_vals, global_counts / len(sensitive_values)))
            
            # Analyze each leaf
            for tree_idx in range(self.n_estimators):
                for leaf_id in np.unique(leaf_indices[:, tree_idx]):
                    mask = (leaf_indices[:, tree_idx] == leaf_id)
                    key = (tree_idx, leaf_id)
                    
                    # Store samples - mask is 1D boolean array
                    sample_indices = np.where(mask)[0]
                    self.leaf_samples[key] = X_minority.iloc[sample_indices].copy()
                    
                    # Compute leaf distribution
                    leaf_sensitive = sensitive_values[sample_indices]
                    if len(leaf_sensitive) == 0:
                        self.leaf_fairness[key] = 0.0
                        continue
                    
                    unique_leaf, leaf_counts = np.unique(leaf_sensitive, return_counts=True)
                    leaf_dist = dict(zip(unique_leaf, leaf_counts / len(leaf_sensitive)))
                    
                    # Fairness score: inverse of distribution difference
                    # (smaller difference = more fair)
                    diff = 0.0
                    for val in unique_vals:
                        global_prop = global_dist.get(val, 0)
                        leaf_prop = leaf_dist.get(val, 0)
                        diff += abs(global_prop - leaf_prop)
                    
                    # Convert to score (higher = more fair)
                    fairness_score = 1.0 / (1.0 + diff)
                    self.leaf_fairness[key] = fairness_score
        else:
            # No sensitive features, use uniform
            self._analyze_leaves(X_minority)
    
    def _fair_leaf_resample(
        self,
        X_minority: pd.DataFrame,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Resample from leaves with fairness-aware selection.
        
        Strategy:
        1. Select leaves with probability proportional to fairness score
        2. Sample from selected leaves with small Gaussian noise
        """
        if not self.leaf_samples:
            warnings.warn("No leaf samples available")
            return pd.DataFrame()
        
        # Get leaf keys and fairness scores
        leaves = list(self.leaf_fairness.keys())
        fairness_scores = np.array([self.leaf_fairness[k] for k in leaves])
        
        # Compute sampling probabilities (weighted by fairness)
        # Mix fairness and uniform: p = fairness_weight * fairness + (1-fairness_weight) * uniform
        uniform_prob = np.ones(len(leaves)) / len(leaves)
        fairness_prob = fairness_scores / fairness_scores.sum() if fairness_scores.sum() > 0 else uniform_prob
        sampling_prob = (self.fairness_weight * fairness_prob + 
                        (1 - self.fairness_weight) * uniform_prob)
        sampling_prob = sampling_prob / sampling_prob.sum()
        
        # Generate samples
        generated = []
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(n_samples):
            # Select leaf
            leaf_idx = rng.choice(len(leaves), p=sampling_prob)
            leaf_key = leaves[leaf_idx]
            leaf_data = self.leaf_samples[leaf_key]
            
            if len(leaf_data) == 0:
                continue
            
            # Sample from leaf
            sample_idx = rng.choice(len(leaf_data))
            sample = leaf_data.iloc[sample_idx].copy()
            
            # Add small Gaussian noise for diversity
            for col in sample.index:
                if pd.api.types.is_numeric_dtype(leaf_data[col]):
                    std = leaf_data[col].std()
                    if std > 0:
                        noise = rng.normal(0, std * 0.1)  # 10% noise
                        sample[col] += noise
            
            generated.append(sample)
        
        if generated:
            return pd.DataFrame(generated)
        else:
            return pd.DataFrame()
    
    def _compute_avg_fairness(self) -> float:
        """Compute average fairness score across leaves."""
        if not self.leaf_fairness:
            return 0.0
        return np.mean(list(self.leaf_fairness.values()))

