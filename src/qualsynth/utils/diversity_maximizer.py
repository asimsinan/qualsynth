"""
Diversity Maximizer for Qualsynth

Based on research from:
- GReaT: Column permutation for diverse generation
- G2 (EMNLP 2025): Entropy-based selective intervention
- TabDDPM: Feature-wise learnable diffusion
- DPP: Determinantal Point Process for diverse subset selection

Key techniques:
1. COLUMN PERMUTATION: Shuffle column order each batch (GReaT)
2. TEMPERATURE SCHEDULING: Higher temp for diversity, lower for quality
3. ANCHOR ROTATION: Use different anchor subsets each iteration
4. DPP SELECTION: Select maximally diverse samples from candidates
5. ANTI-SIMILARITY FILTER: Reject samples too similar to each other
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


@dataclass
class DiversityConfig:
    """Configuration for diversity maximization."""
    # Column permutation
    enable_column_permutation: bool = True
    permutation_seed_offset: int = 0  # Changes per iteration
    
    # Temperature scheduling
    base_temperature: float = 0.7
    max_temperature: float = 1.2
    temperature_schedule: str = "cosine"  # "constant", "linear", "cosine"
    
    # Anchor rotation
    n_anchors: int = 12
    anchor_rotation_strategy: str = "kmeans_diverse"  # "random", "kmeans", "kmeans_diverse"
    
    # DPP selection
    enable_dpp_selection: bool = True
    dpp_kernel_bandwidth: float = 0.5
    
    # Anti-similarity filter
    enable_anti_similarity: bool = True
    min_distance_threshold: float = 0.1  # Minimum normalized distance between samples
    
    # Diversity scoring weights
    numerical_weight: float = 0.6
    categorical_weight: float = 0.4


class DiversityMaximizer:
    """
    Diversity Maximization System.
    
    Ensures generated samples are maximally diverse while staying in-distribution.
    """
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        self.config = config or DiversityConfig()
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_ranges = {}
        self.categorical_features = []
        self.numerical_features = []
        
    def fit(self, X_train: pd.DataFrame, categorical_features: Optional[List[str]] = None):
        """Fit on training data to learn feature statistics."""
        self.categorical_features = categorical_features or []
        self.numerical_features = [c for c in X_train.columns if c not in self.categorical_features]
        
        # Learn feature ranges
        for col in X_train.columns:
            if col in self.numerical_features:
                col_data = pd.to_numeric(X_train[col], errors='coerce')
                self.feature_ranges[col] = (col_data.min(), col_data.max())
            else:
                self.feature_ranges[col] = list(X_train[col].dropna().unique())
        
        # Fit scaler on numerical features
        if self.numerical_features:
            X_num = X_train[self.numerical_features].fillna(0)
            self.scaler.fit(X_num)
        
        self.fitted = True
        return self
    
    def get_permuted_columns(self, columns: List[str], iteration: int) -> List[str]:
        """
        Get permuted column order for this iteration.
        
        GReaT paper shows that randomizing column order during generation
        leads to more diverse outputs because the LLM generates each feature
        conditioned on different context.
        """
        if not self.config.enable_column_permutation:
            return columns
        
        # Use iteration-specific seed for reproducibility
        rng = np.random.RandomState(42 + iteration * 17 + self.config.permutation_seed_offset)
        permuted = columns.copy()
        rng.shuffle(permuted)
        
        return permuted
    
    def get_scheduled_temperature(self, iteration: int, max_iterations: int) -> float:
        """
        Get temperature for this iteration using a schedule.
        
        Higher temperature early = more exploration
        Lower temperature later = more refinement
        """
        progress = iteration / max(1, max_iterations)
        
        if self.config.temperature_schedule == "constant":
            return self.config.base_temperature
        
        elif self.config.temperature_schedule == "linear":
            # Linear decrease from max to base
            return self.config.max_temperature - progress * (self.config.max_temperature - self.config.base_temperature)
        
        elif self.config.temperature_schedule == "cosine":
            # Cosine annealing: starts high, decreases smoothly
            cosine_factor = (1 + np.cos(np.pi * progress)) / 2
            return self.config.base_temperature + cosine_factor * (self.config.max_temperature - self.config.base_temperature)
        
        else:
            return self.config.base_temperature
    
    def select_diverse_anchors(
        self, 
        X_minority: pd.DataFrame, 
        n_anchors: int,
        iteration: int,
        previous_anchors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Select anchors using configured strategy.
        
        Strategies:
        - "typical": Select samples closest to distribution center (RECOMMENDED for distribution matching)
        - "stratified": Select proportionally from quantile strata (balanced coverage)
        - "kmeans_diverse": K-means clustering with rotation (maximum diversity)
        - "kmeans": Simple K-means clustering
        - "random": Random sampling
        """
        if len(X_minority) <= n_anchors:
            return X_minority
        
        strategy = self.config.anchor_rotation_strategy
        
        if strategy == "boundary":
            # BEST: Select minority samples CLOSEST to majority class (decision boundary)
            # These are the most informative samples for classification
            # Inspired by Borderline SMOTE which outperforms all other methods
            return self._select_boundary_anchors(X_minority, n_anchors, iteration)
        
        elif strategy == "discriminative":
            # Select samples with highest P(minority) from a guide classifier
            # These are "prototypical" minority samples that are most different from majority
            return self._select_discriminative_anchors(X_minority, n_anchors, iteration)
        
        elif strategy == "typical":
            # Select samples closest to distribution center
            # This ensures LLM sees "typical" examples, generating typical samples
            return self._select_typical_anchors(X_minority, n_anchors, iteration)
        
        elif strategy == "stratified":
            # NEW: Select proportionally from quantile strata
            # More from center, fewer from tails
            return self._select_stratified_anchors(X_minority, n_anchors, iteration)
        
        elif strategy == "random":
            # Simple random sampling with iteration-based seed
            return X_minority.sample(n=n_anchors, random_state=42 + iteration * 7)
        
        elif strategy == "kmeans":
            # K-means clustering
            from sklearn.cluster import KMeans
            
            num_cols = [c for c in X_minority.columns if c in self.numerical_features]
            if not num_cols:
                return X_minority.sample(n=n_anchors, random_state=42 + iteration)
            
            X_num = X_minority[num_cols].fillna(0)
            
            kmeans = KMeans(n_clusters=n_anchors, random_state=42 + iteration * 13, n_init=10)
            kmeans.fit(X_num)
            
            anchor_indices = []
            for i in range(n_anchors):
                cluster_mask = kmeans.labels_ == i
                if cluster_mask.sum() > 0:
                    cluster_samples = X_num[cluster_mask]
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_samples.values - center, axis=1)
                    closest_idx = cluster_samples.index[np.argmin(distances)]
                    anchor_indices.append(closest_idx)
            
            return X_minority.loc[anchor_indices]
        
        elif strategy == "kmeans_diverse":
            # K-means + avoid previous anchors
            from sklearn.cluster import KMeans
            
            num_cols = [c for c in X_minority.columns if c in self.numerical_features]
            if not num_cols:
                return X_minority.sample(n=n_anchors, random_state=42 + iteration)
            
            X_num = X_minority[num_cols].fillna(0)
            
            # Use more clusters than needed, then select diverse subset
            n_clusters = min(n_anchors * 2, len(X_minority))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42 + iteration * 13, n_init=10)
            kmeans.fit(X_num)
            
            # Get one sample from each cluster
            candidates = []
            for i in range(n_clusters):
                cluster_mask = kmeans.labels_ == i
                if cluster_mask.sum() > 0:
                    cluster_samples = X_num[cluster_mask]
                    # Select randomly from cluster (not always closest to center)
                    rng = np.random.RandomState(42 + iteration + i)
                    selected_idx = rng.choice(cluster_samples.index)
                    candidates.append(selected_idx)
            
            # If we have previous anchors, prefer samples far from them
            if previous_anchors is not None and len(previous_anchors) > 0:
                prev_num = previous_anchors[num_cols].fillna(0)
                candidate_samples = X_num.loc[candidates]
                
                # Compute distances to previous anchors
                distances_to_prev = []
                for idx in candidates:
                    sample = X_num.loc[idx].values.reshape(1, -1)
                    dists = np.linalg.norm(prev_num.values - sample, axis=1)
                    distances_to_prev.append(dists.min())
                
                # Sort by distance to previous (prefer far samples)
                sorted_indices = np.argsort(distances_to_prev)[::-1]
                selected = [candidates[i] for i in sorted_indices[:n_anchors]]
            else:
                # Just take first n_anchors
                selected = candidates[:n_anchors]
            
            return X_minority.loc[selected]
        
        else:
            return X_minority.sample(n=n_anchors, random_state=42 + iteration)
    
    def _select_boundary_anchors(
        self,
        X_minority: pd.DataFrame,
        n_anchors: int,
        iteration: int
    ) -> pd.DataFrame:
        """
        Select minority samples CLOSEST to majority class (decision boundary).
        
        This is inspired by Borderline SMOTE, which outperforms other methods
        because boundary samples are most informative for classification.
        
        The LLM will generate variations of these boundary samples, creating
        synthetic samples that help define the decision boundary.
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        # We need access to training data (including majority class)
        if not hasattr(self, '_X_train') or self._X_train is None:
            # Fall back to typical selection if no training data available
            return self._select_typical_anchors(X_minority, n_anchors, iteration)
        
        try:
            # Get majority class samples
            X_majority = self._X_train[self._y_train == 0]
            
            if len(X_majority) == 0:
                return self._select_typical_anchors(X_minority, n_anchors, iteration)
            
            # Scale features for distance calculation
            scaler = StandardScaler()
            X_all = pd.concat([X_minority, X_majority], ignore_index=True)
            scaler.fit(X_all)
            
            X_minority_scaled = scaler.transform(X_minority)
            X_majority_scaled = scaler.transform(X_majority)
            
            # Find k nearest majority neighbors for each minority sample
            k_neighbors = min(5, len(X_majority))
            nn_majority = NearestNeighbors(n_neighbors=k_neighbors).fit(X_majority_scaled)
            distances, _ = nn_majority.kneighbors(X_minority_scaled)
            
            # Average distance to k nearest majority neighbors
            avg_dist_to_majority = distances.mean(axis=1)
            
            # Sort by distance (CLOSEST to majority = boundary samples)
            sorted_indices = np.argsort(avg_dist_to_majority)
            
            # Take top 2*n_anchors boundary samples, then sample for diversity
            top_boundary = sorted_indices[:min(n_anchors * 2, len(sorted_indices))]
            
            rng = np.random.RandomState(42 + iteration * 7)
            selected_positions = rng.choice(
                len(top_boundary),
                size=min(n_anchors, len(top_boundary)),
                replace=False
            )
            selected_indices = X_minority.index[top_boundary[selected_positions]]
            
            if hasattr(self, 'verbose') and self.verbose:
                print(f"    🎯 Boundary anchor selection: {len(selected_indices)} samples")
                print(f"       Avg distance to majority: {avg_dist_to_majority[top_boundary[selected_positions]].mean():.3f}")
            
            return X_minority.loc[selected_indices]
            
        except Exception as e:
            print(f"    ⚠️ Boundary anchor selection failed: {e}")
            return self._select_typical_anchors(X_minority, n_anchors, iteration)
    
    def _select_discriminative_anchors(
        self,
        X_minority: pd.DataFrame,
        n_anchors: int,
        iteration: int
    ) -> pd.DataFrame:
        """
        Select anchors with highest P(minority) from a guide classifier.
        
        These are "prototypical" minority samples that are most DIFFERENT from
        the majority class. Using these as anchors helps the LLM generate samples
        that are clearly in the minority decision region.
        
        This is the KEY innovation for beating baselines.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # We need access to training data and labels
        # Check if we have them stored
        if not hasattr(self, '_X_train') or self._X_train is None:
            # Fall back to typical selection if no training data available
            return self._select_typical_anchors(X_minority, n_anchors, iteration)
        
        try:
            # Train guide classifier
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self._X_train)
            
            guide_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            guide_clf.fit(X_train_scaled, self._y_train)
            
            # Get P(minority) for each minority sample
            X_minority_scaled = scaler.transform(X_minority)
            proba = guide_clf.predict_proba(X_minority_scaled)[:, 1]
            
            # Sort by P(minority) (highest first = most discriminative)
            sorted_indices = np.argsort(proba)[::-1]
            
            # Take top 2*n_anchors most discriminative, then sample for diversity
            top_discriminative = sorted_indices[:min(n_anchors * 2, len(sorted_indices))]
            
            rng = np.random.RandomState(42 + iteration * 7)
            selected_positions = rng.choice(
                len(top_discriminative), 
                size=min(n_anchors, len(top_discriminative)), 
                replace=False
            )
            selected_indices = X_minority.index[top_discriminative[selected_positions]]
            
            return X_minority.loc[selected_indices]
            
        except Exception as e:
            # Fall back to typical selection on error
            print(f"    ⚠️ Discriminative anchor selection failed: {e}")
            return self._select_typical_anchors(X_minority, n_anchors, iteration)
    
    def set_training_data(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Set training data for discriminative anchor selection.
        
        Must be called before using 'discriminative' anchor strategy.
        """
        self._X_train = X_train
        self._y_train = y_train
    
    def _select_typical_anchors(
        self,
        X_minority: pd.DataFrame,
        n_anchors: int,
        iteration: int
    ) -> pd.DataFrame:
        """
        Select anchors that are TYPICAL of the minority class distribution.
        
        Samples with lower average z-score across all features are more "typical".
        This ensures the LLM sees representative examples and generates similar samples.
        """
        num_cols = [c for c in X_minority.columns if c in self.numerical_features]
        
        if not num_cols:
            return X_minority.sample(n=n_anchors, random_state=42 + iteration)
        
        X_num = X_minority[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Compute typicality score for each sample
        # Lower average z-score = more typical
        typicality_scores = []
        
        for idx in X_minority.index:
            z_scores = []
            for col in num_cols:
                col_mean = X_num[col].mean()
                col_std = X_num[col].std()
                if col_std > 0:
                    z = abs(X_num.loc[idx, col] - col_mean) / col_std
                    z_scores.append(z)
            
            # Average z-score across all features
            avg_z = np.mean(z_scores) if z_scores else 0
            # Typicality = inverse of average z-score (higher = more typical)
            typicality = 1.0 / (1.0 + avg_z)
            typicality_scores.append((idx, typicality))
        
        # Sort by typicality (highest first = most typical)
        typicality_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add some variation across iterations to avoid always selecting same anchors
        # Take top 2*n_anchors most typical, then sample n_anchors from them
        top_typical = [idx for idx, _ in typicality_scores[:min(n_anchors * 2, len(typicality_scores))]]
        
        rng = np.random.RandomState(42 + iteration * 7)
        selected = rng.choice(top_typical, size=min(n_anchors, len(top_typical)), replace=False)
        
        return X_minority.loc[selected]
    
    def _select_stratified_anchors(
        self,
        X_minority: pd.DataFrame,
        n_anchors: int,
        iteration: int
    ) -> pd.DataFrame:
        """
        Select anchors from different STRATA of the distribution.
        
        Uses the first numerical feature to create strata, then selects:
        - More anchors from the center (median region)
        - Fewer anchors from the tails
        
        This ensures the LLM sees the full distribution but weighted toward typical values.
        """
        num_cols = [c for c in X_minority.columns if c in self.numerical_features]
        
        if not num_cols:
            return X_minority.sample(n=n_anchors, random_state=42 + iteration)
        
        # Use the first numerical column for stratification
        # (could be enhanced to use PCA or a weighted combination)
        key_col = num_cols[0]
        X_minority = X_minority.copy()
        
        try:
            # Create 5 quantile strata
            X_minority['_stratum'] = pd.qcut(
                X_minority[key_col].apply(pd.to_numeric, errors='coerce'),
                q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
            )
        except ValueError:
            # If qcut fails (e.g., too few unique values), fall back to typical selection
            return self._select_typical_anchors(X_minority.drop('_stratum', axis=1, errors='ignore'), 
                                                 n_anchors, iteration)
        
        # Distribution: more from center, fewer from tails
        # Stratum 1 (lowest 20%): 10% of anchors
        # Stratum 2: 15% of anchors
        # Stratum 3 (median): 50% of anchors  <- Most from center!
        # Stratum 4: 15% of anchors
        # Stratum 5 (highest 20%): 10% of anchors
        stratum_weights = {1: 0.10, 2: 0.15, 3: 0.50, 4: 0.15, 5: 0.10}
        
        anchors = []
        rng = np.random.RandomState(42 + iteration * 7)
        
        for stratum, weight in stratum_weights.items():
            n_select = max(1, int(n_anchors * weight))
            stratum_samples = X_minority[X_minority['_stratum'] == stratum]
            
            if len(stratum_samples) > 0:
                n_actual = min(n_select, len(stratum_samples))
                selected = stratum_samples.sample(n=n_actual, random_state=rng.randint(0, 10000))
                anchors.append(selected.drop('_stratum', axis=1))
        
        if anchors:
            result = pd.concat(anchors, ignore_index=False)
            # Ensure we have exactly n_anchors (may have rounding differences)
            if len(result) > n_anchors:
                result = result.sample(n=n_anchors, random_state=42 + iteration)
            return result
        else:
            return X_minority.drop('_stratum', axis=1).sample(n=n_anchors, random_state=42 + iteration)
    
    def select_diverse_subset_dpp(
        self, 
        samples: pd.DataFrame, 
        n_select: int
    ) -> pd.DataFrame:
        """
        Select diverse subset using Determinantal Point Process (DPP).
        
        DPP naturally favors diverse subsets by penalizing similar items.
        """
        if len(samples) <= n_select:
            return samples
        
        if not self.config.enable_dpp_selection:
            return samples.head(n_select)
        
        # Get numerical columns - must be actually numeric
        num_cols = []
        for c in samples.columns:
            if c in self.numerical_features:
                try:
                    pd.to_numeric(samples[c], errors='raise')
                    num_cols.append(c)
                except (ValueError, TypeError):
                    pass
        
        if not num_cols:
            return samples.head(n_select)
        
        X_num = samples[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        # Standardize
        if self.fitted and len(num_cols) == len(self.numerical_features):
            try:
                X_scaled = self.scaler.transform(X_num)
            except ValueError:
                X_scaled = (X_num - X_num.mean(axis=0)) / (X_num.std(axis=0) + 1e-8)
        else:
            X_scaled = (X_num - X_num.mean(axis=0)) / (X_num.std(axis=0) + 1e-8)
        
        # Build similarity kernel (RBF)
        pairwise_sq_dists = squareform(pdist(X_scaled, 'sqeuclidean'))
        bandwidth = self.config.dpp_kernel_bandwidth
        L = np.exp(-pairwise_sq_dists / (2 * bandwidth ** 2))
        
        # Greedy DPP selection (approximate)
        selected_indices = []
        remaining = list(range(len(samples)))
        
        for _ in range(n_select):
            if not remaining:
                break
            
            if not selected_indices:
                # First selection: highest quality (diagonal of L)
                scores = np.diag(L)
                best_idx = remaining[np.argmax([scores[i] for i in remaining])]
            else:
                # Subsequent: maximize determinant gain
                best_score = -np.inf
                best_idx = remaining[0]
                
                for idx in remaining:
                    # Compute log-det gain
                    subset = selected_indices + [idx]
                    L_subset = L[np.ix_(subset, subset)]
                    try:
                        score = np.linalg.slogdet(L_subset)[1]
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    except np.linalg.LinAlgError:
                        continue
            
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        
        return samples.iloc[selected_indices]
    
    def filter_by_anti_similarity(
        self, 
        samples: pd.DataFrame,
        existing_samples: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Remove samples that are too similar to each other or existing samples.
        
        This is a post-generation filter to ensure diversity.
        """
        if not self.config.enable_anti_similarity:
            return samples
        
        if len(samples) <= 1:
            return samples
        
        # Get numerical columns - must be actually numeric dtype
        num_cols = []
        for c in samples.columns:
            if c in self.numerical_features:
                # Double-check it's actually numeric
                try:
                    pd.to_numeric(samples[c], errors='raise')
                    num_cols.append(c)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric columns
        
        if not num_cols:
            return samples
        
        # Combine with existing samples if provided
        if existing_samples is not None and len(existing_samples) > 0:
            all_samples = pd.concat([existing_samples, samples], ignore_index=True)
            n_existing = len(existing_samples)
        else:
            all_samples = samples
            n_existing = 0
        
        # Convert to numeric and normalize
        X_num = all_samples[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        X_scaled = (X_num - X_num.mean(axis=0)) / (X_num.std(axis=0) + 1e-8)
        
        # Compute pairwise distances
        distances = squareform(pdist(X_scaled, 'euclidean'))
        
        # Greedy selection: keep samples that are far enough from all kept samples
        kept_indices = list(range(n_existing))  # Keep all existing samples
        
        for i in range(n_existing, len(all_samples)):
            if not kept_indices:
                kept_indices.append(i)
                continue
            
            # Check minimum distance to kept samples
            min_dist = distances[i, kept_indices].min()
            
            if min_dist >= self.config.min_distance_threshold:
                kept_indices.append(i)
        
        # Return only the new samples that were kept
        new_kept = [i - n_existing for i in kept_indices if i >= n_existing]
        
        return samples.iloc[new_kept]
    
    def compute_diversity_score(self, samples: pd.DataFrame) -> Dict[str, float]:
        """
        Compute comprehensive diversity metrics for a set of samples.
        """
        metrics = {}
        
        # 1. Numerical diversity: Coefficient of Variation
        # Get actually numeric columns
        num_cols = []
        for c in samples.columns:
            if c in self.numerical_features:
                try:
                    pd.to_numeric(samples[c], errors='raise')
                    num_cols.append(c)
                except (ValueError, TypeError):
                    pass
        
        if num_cols:
            cvs = []
            for col in num_cols:
                col_data = pd.to_numeric(samples[col], errors='coerce').dropna()
                if len(col_data) > 1 and col_data.mean() != 0:
                    cv = col_data.std() / abs(col_data.mean())
                    cvs.append(cv)
            metrics['numerical_cv'] = np.mean(cvs) * 100 if cvs else 0
        else:
            metrics['numerical_cv'] = 0
        
        # 2. Categorical diversity: Entropy
        cat_cols = [c for c in samples.columns if c in self.categorical_features]
        if cat_cols:
            entropies = []
            for col in cat_cols:
                value_counts = samples[col].value_counts(normalize=True)
                if len(value_counts) > 1:
                    entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
                    max_entropy = np.log(len(value_counts))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    entropies.append(normalized_entropy)
            metrics['categorical_entropy'] = np.mean(entropies) * 100 if entropies else 0
        else:
            metrics['categorical_entropy'] = 0
        
        # 3. Inter-sample distance
        if num_cols and len(samples) > 1:
            X_num = samples[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            std_vals = X_num.std(axis=0)
            std_vals[std_vals == 0] = 1  # Avoid division by zero
            X_scaled = (X_num - X_num.mean(axis=0)) / (std_vals + 1e-8)
            distances = pdist(X_scaled, 'euclidean')
            metrics['mean_distance'] = float(np.mean(distances))
            metrics['min_distance'] = float(np.min(distances)) if len(distances) > 0 else 0
        else:
            metrics['mean_distance'] = 0
            metrics['min_distance'] = 0
        
        # 4. Duplicate rate
        n_duplicates = samples.duplicated().sum()
        metrics['duplicate_rate'] = n_duplicates / len(samples) * 100 if len(samples) > 0 else 0
        
        # 5. Overall diversity score (weighted)
        metrics['overall_diversity'] = (
            self.config.numerical_weight * metrics['numerical_cv'] +
            self.config.categorical_weight * metrics['categorical_entropy']
        )
        
        return metrics


def create_diversity_maximizer(
    enable_column_permutation: bool = True,
    temperature_schedule: str = "cosine",
    enable_dpp: bool = True,
    enable_anti_similarity: bool = True
) -> DiversityMaximizer:
    """Factory function to create configured diversity maximizer."""
    config = DiversityConfig(
        enable_column_permutation=enable_column_permutation,
        temperature_schedule=temperature_schedule,
        enable_dpp_selection=enable_dpp,
        enable_anti_similarity=enable_anti_similarity
    )
    return DiversityMaximizer(config)

