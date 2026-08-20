"""
Duplicate Prevention System for LLM-Generated Synthetic Data

Combines multiple approaches:
1. Semantic Deduplication (SemDeDup) - embedding-based similarity
2. Memory-Augmented Generation - track generated samples
3. Real-time Filtering - prevent duplicates during generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json


@dataclass
class DuplicatePreventionResult:
    """Result of duplicate prevention."""
    original_count: int
    filtered_count: int
    exact_duplicates_removed: int
    semantic_duplicates_removed: int
    hash_duplicates_removed: int
    final_samples: List[Dict[str, Any]]
    duplicate_rate: float
    diversity_score: float


class SOTADuplicatePrevention:
    """
    Duplicate prevention system combining multiple approaches.
    
    Methods:
    1. Hash-based exact duplicate detection (O(1) lookup)
    2. Semantic embedding similarity (SemDeDup approach)
    3. Feature-level diversity enforcement
    4. Memory-augmented tracking across batches
    5. Sliding window memory management
    """
    
    def __init__(
        self,
        exact_match_threshold: float = 1.0,
        semantic_similarity_threshold: float = 0.95,
        feature_diversity_threshold: float = 0.15,
        enable_semantic_dedup: bool = True,
        enable_hash_dedup: bool = True,
        enable_feature_diversity: bool = True,
        verbose: bool = True,
        max_memory_size: int = 200,
        memory_strategy: str = 'sliding_window'
    ):
        """
        Initialize duplicate prevention system.
        
        Args:
            exact_match_threshold: Threshold for exact matches (1.0 = identical)
            semantic_similarity_threshold: Cosine similarity threshold (0.95 = 95% similar)
            feature_diversity_threshold: Minimum Gower distance required
            enable_semantic_dedup: Use embedding-based semantic deduplication
            enable_hash_dedup: Use hash-based exact duplicate detection
            enable_feature_diversity: Use feature-level diversity checks
            verbose: Print detailed logs
            max_memory_size: Maximum number of samples to keep in memory
            memory_strategy: Strategy for memory management:
                - 'sliding_window': FIFO - remove oldest samples
                - 'diversity_preserving': Remove most redundant samples
                - 'cluster_based': Keep representatives from each cluster
        """
        self.exact_match_threshold = exact_match_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.feature_diversity_threshold = feature_diversity_threshold
        self.enable_semantic_dedup = enable_semantic_dedup
        self.enable_hash_dedup = enable_hash_dedup
        self.enable_feature_diversity = enable_feature_diversity
        self.verbose = verbose
        
        # Bounded memory management
        self.max_memory_size = max_memory_size
        self.memory_strategy = memory_strategy
        
        # Memory: track seen samples across batches
        self.seen_hashes: Set[str] = set()
        self.seen_embeddings: List[np.ndarray] = []
        self.seen_samples: List[Dict[str, Any]] = []
        
        # Feature ranges for proper Gower distance normalization (RAW data support)
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.categorical_features: List[str] = []
        
        # Track memory statistics
        self._prune_count = 0
        
        if self.verbose:
            print("🔬 Duplicate Prevention System Initialized")
            print(f"   Methods enabled:")
            print(f"   - Hash-based exact dedup: {enable_hash_dedup}")
            print(f"   - Semantic embedding dedup: {enable_semantic_dedup}")
            print(f"   - Feature diversity check: {enable_feature_diversity}")
            print(f"   - Memory management: {memory_strategy} (max={max_memory_size})")
    
    def fit(self, X_train: pd.DataFrame, categorical_features: Optional[List[str]] = None, 
            add_to_memory: bool = True) -> 'SOTADuplicatePrevention':
        """
        Fit the duplicate prevention system on training data.
        
        Computes feature ranges for proper Gower distance normalization.
        Optionally adds training samples to memory so generated samples
        are compared against BOTH original data AND other generated samples.
        
        Args:
            X_train: Training data (RAW or ENCODED)
            categorical_features: List of categorical feature names
            add_to_memory: If True, add training samples to seen_hashes/embeddings
                          so generated samples are compared against training data.
                          This ensures generated samples are DIFFERENT from originals.
        
        Returns:
            self for chaining
        """
        self.categorical_features = categorical_features or []
        
        # Compute feature ranges for numerical features
        for col in X_train.columns:
            if col not in self.categorical_features:
                try:
                    col_data = pd.to_numeric(X_train[col], errors='coerce')
                    min_val = col_data.min()
                    max_val = col_data.max()
                    if pd.notna(min_val) and pd.notna(max_val):
                        self.feature_ranges[col] = (float(min_val), float(max_val))
                except (ValueError, TypeError):
                    # Not numerical, treat as categorical
                    if col not in self.categorical_features:
                        self.categorical_features.append(col)
        
        if self.verbose:
            print(f"   📊 Fitted on {len(X_train)} samples")
            print(f"   📈 Numerical features: {len(self.feature_ranges)}")
            print(f"   📝 Categorical features: {len(self.categorical_features)}")
        
        # Add training samples to memory for comparison
        if add_to_memory:
            if self.verbose:
                print(f"   🧠 Adding {len(X_train)} training samples to memory...")
            
            train_samples = X_train.to_dict('records')
            added_count = 0
            
            for sample in train_samples:
                # Add hash for exact duplicate detection
                if self.enable_hash_dedup:
                    sample_hash = self._compute_hash(sample)
                    self.seen_hashes.add(sample_hash)
                
                # Add embedding for semantic dedup
                if self.enable_semantic_dedup:
                    sample_embedding = self._compute_embedding(sample)
                    self.seen_embeddings.append(sample_embedding)
                
                # Add sample for feature diversity check
                if self.enable_feature_diversity:
                    self.seen_samples.append(sample)
                
                added_count += 1
            
            if self.verbose:
                print(f"   ✅ Added {added_count} training samples to memory")
                print(f"      - Hashes: {len(self.seen_hashes)}")
                print(f"      - Embeddings: {len(self.seen_embeddings)}")
                print(f"      - Samples: {len(self.seen_samples)}")
        
        return self
    
    def _compute_hash(self, sample: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of a sample.
        
        Uses sorted JSON representation for consistent hashing.
        """
        # Sort keys and convert to JSON string
        sorted_sample = {k: sample[k] for k in sorted(sample.keys())}
        # Round floats to 6 decimals to handle floating point precision
        rounded_sample = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in sorted_sample.items()
        }
        sample_str = json.dumps(rounded_sample, sort_keys=True)
        return hashlib.sha256(sample_str.encode()).hexdigest()
    
    def _compute_embedding(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Compute semantic embedding of a sample.
        
        Uses feature values as a simple embedding, normalized to [0,1] range.
        For more advanced use, could integrate BERT/sentence transformers.
        """
        # Extract numeric values in sorted key order, normalized
        values = []
        for k in sorted(sample.keys()):
            v = sample[k]
            if isinstance(v, (int, float)):
                # Normalize using feature ranges if available
                if k in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[k]
                    range_val = max_val - min_val
                    if range_val > 0:
                        values.append((float(v) - min_val) / range_val)
                    else:
                        values.append(0.5)
                else:
                    # Fallback: use value directly (assumes [0,1] range)
                    values.append(float(v))
            elif isinstance(v, str):
                # Simple hash for categorical values (in [0,1] range)
                values.append(float(hash(v) % 10000) / 10000.0)
            else:
                values.append(0.0)
        
        return np.array(values, dtype=np.float32)
    
    def _compute_gower_distance(
        self,
        sample1: Dict[str, Any],
        sample2: Dict[str, Any],
        categorical_features: Optional[List[str]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> float:
        """
        Compute Gower distance between two samples.
        
        Gower distance handles mixed data types (categorical + numerical).
        Works with both RAW and ENCODED data by using feature ranges for normalization.
        
        Args:
            sample1: First sample
            sample2: Second sample
            categorical_features: List of categorical feature names
            feature_ranges: Dict of {feature: (min, max)} for normalization
        """
        if categorical_features is None:
            categorical_features = []
        if feature_ranges is None:
            feature_ranges = {}
        
        distances = []
        for key in sample1.keys():
            if key in categorical_features:
                # Categorical: 0 if same, 1 if different
                distances.append(0.0 if sample1[key] == sample2[key] else 1.0)
            else:
                # Numerical: normalized absolute difference
                try:
                    v1 = float(sample1[key])
                    v2 = float(sample2[key])
                    
                    # Normalize by feature range if available
                    if key in feature_ranges:
                        min_val, max_val = feature_ranges[key]
                        range_val = max_val - min_val
                        if range_val > 0:
                            distances.append(abs(v1 - v2) / range_val)
                        else:
                            distances.append(0.0)
                    else:
                        # Fallback: use max of the two values for normalization
                        # This handles both [0,1] normalized and RAW data
                        max_val = max(abs(v1), abs(v2), 1.0)
                        distances.append(abs(v1 - v2) / max_val)
                except (ValueError, TypeError):
                    # Treat as categorical if can't convert to float
                    distances.append(0.0 if sample1[key] == sample2[key] else 1.0)
        
        return np.mean(distances) if distances else 0.0
    
    def filter_duplicates(
        self,
        samples: List[Dict[str, Any]],
        categorical_features: Optional[List[str]] = None
    ) -> DuplicatePreventionResult:
        """
        Filter duplicates using multiple methods.
        
        Args:
            samples: List of sample dictionaries
            categorical_features: List of categorical feature names
        
        Returns:
            DuplicatePreventionResult with filtered samples and statistics
        """
        if not samples:
            return DuplicatePreventionResult(
                original_count=0,
                filtered_count=0,
                exact_duplicates_removed=0,
                semantic_duplicates_removed=0,
                hash_duplicates_removed=0,
                final_samples=[],
                duplicate_rate=0.0,
                diversity_score=0.0
            )
        
        original_count = len(samples)
        filtered_samples = []
        
        exact_dupes = 0
        semantic_dupes = 0
        hash_dupes = 0
        
        if self.verbose:
            print(f"\n🔬 Duplicate Prevention: Processing {original_count} samples...")
        
        for i, sample in enumerate(samples):
            is_duplicate = False
            
            # Method 1: Hash-based exact duplicate detection (O(1))
            if self.enable_hash_dedup:
                sample_hash = self._compute_hash(sample)
                if sample_hash in self.seen_hashes:
                    is_duplicate = True
                    hash_dupes += 1
                    exact_dupes += 1
                    if self.verbose and i < 5:  # Show first few
                        print(f"   ❌ Sample {i}: Exact duplicate (hash match)")
                    continue
                
            # Method 2: Semantic embedding similarity (SemDeDup)
            if self.enable_semantic_dedup and not is_duplicate:
                sample_embedding = self._compute_embedding(sample)
                
                if len(self.seen_embeddings) > 0:
                    # Compute cosine similarity with all seen embeddings
                    similarities = cosine_similarity(
                        [sample_embedding],
                        self.seen_embeddings
                    )[0]
                    
                    max_similarity = np.max(similarities)
                    if max_similarity >= self.semantic_similarity_threshold:
                        is_duplicate = True
                        semantic_dupes += 1
                        if self.verbose and i < 5:
                            print(f"   ❌ Sample {i}: Semantic duplicate (similarity={max_similarity:.3f})")
                        continue
            
            # Method 3: Feature-level diversity check (Gower distance)
            if self.enable_feature_diversity and not is_duplicate:
                if len(self.seen_samples) > 0:
                    # Check against recent samples (last 100 for efficiency)
                    recent_samples = self.seen_samples[-100:]
                    min_distance = float('inf')
                    
                    # Use stored categorical features and feature ranges
                    cat_feats = categorical_features or self.categorical_features
                    
                    for seen_sample in recent_samples:
                        distance = self._compute_gower_distance(
                            sample,
                            seen_sample,
                            cat_feats,
                            self.feature_ranges  # Use stored ranges for RAW data
                        )
                        min_distance = min(min_distance, distance)
                    
                    if min_distance < self.feature_diversity_threshold:
                        is_duplicate = True
                        if self.verbose and i < 5:
                            print(f"   ❌ Sample {i}: Low diversity (distance={min_distance:.3f})")
                        continue
            
            # Sample passed all checks - add to filtered set
            if not is_duplicate:
                filtered_samples.append(sample)
                
                # Update memory (reuse computed hash and embedding if available)
                self._add_to_memory(
                    sample,
                    sample_hash=sample_hash if self.enable_hash_dedup else None,
                    sample_embedding=sample_embedding if self.enable_semantic_dedup else None
                )
                
                if self.verbose and i < 5:
                    print(f"   ✅ Sample {i}: Unique (passed all checks)")
        
        filtered_count = len(filtered_samples)
        duplicate_rate = (original_count - filtered_count) / original_count if original_count > 0 else 0.0
        
        # Calculate diversity score (average pairwise distance)
        diversity_score = 0.0
        if filtered_count > 1:
            distances = []
            cat_feats = categorical_features or self.categorical_features
            for i in range(min(50, filtered_count)):  # Sample 50 pairs
                for j in range(i + 1, min(50, filtered_count)):
                    dist = self._compute_gower_distance(
                        filtered_samples[i],
                        filtered_samples[j],
                        cat_feats,
                        self.feature_ranges  # Use stored ranges for RAW data
                    )
                    distances.append(dist)
            diversity_score = np.mean(distances) if distances else 0.0
        
        if self.verbose:
            print(f"\n   📊 Results:")
            print(f"      Original: {original_count} samples")
            print(f"      Filtered: {filtered_count} samples")
            print(f"      Removed:  {original_count - filtered_count} duplicates ({duplicate_rate*100:.1f}%)")
            print(f"        - Exact duplicates: {exact_dupes}")
            print(f"        - Semantic duplicates: {semantic_dupes}")
            print(f"        - Low diversity: {original_count - filtered_count - exact_dupes - semantic_dupes}")
            print(f"      Diversity score: {diversity_score:.3f}")
        
        return DuplicatePreventionResult(
            original_count=original_count,
            filtered_count=filtered_count,
            exact_duplicates_removed=exact_dupes,
            semantic_duplicates_removed=semantic_dupes,
            hash_duplicates_removed=hash_dupes,
            final_samples=filtered_samples,
            duplicate_rate=duplicate_rate,
            diversity_score=diversity_score
        )
    
    def _add_to_memory(
        self,
        sample: Dict[str, Any],
        sample_hash: Optional[str] = None,
        sample_embedding: Optional[np.ndarray] = None
    ):
        """
        Add sample to memory with bounded management.
        
        Automatically prunes memory when it exceeds max_memory_size.
        """
        # Add to hash set
        if self.enable_hash_dedup:
            if sample_hash is None:
                sample_hash = self._compute_hash(sample)
            self.seen_hashes.add(sample_hash)
        
        # Add to embeddings list
        if self.enable_semantic_dedup:
            if sample_embedding is None:
                sample_embedding = self._compute_embedding(sample)
            self.seen_embeddings.append(sample_embedding)
        
        # Add to samples list
        if self.enable_feature_diversity:
            self.seen_samples.append(sample)
        
        # Prune memory if it exceeds max size
        if len(self.seen_samples) > self.max_memory_size:
            self._prune_memory()
    
    def _prune_memory(self):
        """
        Prune memory based on configured strategy.
        
        Strategies:
        - sliding_window: FIFO - remove oldest samples
        - diversity_preserving: Remove most redundant (highest similarity) samples
        - cluster_based: Keep representatives from each cluster
        """
        if len(self.seen_samples) <= self.max_memory_size:
            return
        
        self._prune_count += 1
        
        if self.verbose and self._prune_count % 10 == 1:
            print(f"   🔄 Memory pruning #{self._prune_count}: {len(self.seen_samples)} → {self.max_memory_size} samples")
        
        if self.memory_strategy == 'sliding_window':
            self._prune_sliding_window()
        elif self.memory_strategy == 'diversity_preserving':
            self._prune_diversity_preserving()
        elif self.memory_strategy == 'cluster_based':
            self._prune_cluster_based()
        else:
            # Default to sliding window
            self._prune_sliding_window()
    
    def _prune_sliding_window(self):
        """
        FIFO pruning - remove oldest samples.
        
        Simple and effective: newer samples are more relevant for
        detecting duplicates of recent LLM outputs.
        """
        n_to_remove = len(self.seen_samples) - self.max_memory_size
        
        # Remove oldest samples
        self.seen_samples = self.seen_samples[n_to_remove:]
        
        # Also prune embeddings
        if len(self.seen_embeddings) > self.max_memory_size:
            self.seen_embeddings = self.seen_embeddings[n_to_remove:]
        
        # Rebuild hash set from remaining samples
        if self.enable_hash_dedup:
            self.seen_hashes = {self._compute_hash(s) for s in self.seen_samples}
    
    def _prune_diversity_preserving(self):
        """
        Remove most redundant samples (highest average similarity to others).
        
        Keeps a diverse set of samples in memory to maximize coverage
        of the feature space for duplicate detection.
        """
        if len(self.seen_embeddings) < 2:
            self._prune_sliding_window()
            return
        
        n_to_remove = len(self.seen_samples) - self.max_memory_size
        
        # Compute pairwise similarities
        embeddings_array = np.array(self.seen_embeddings)
        similarities = cosine_similarity(embeddings_array)
        np.fill_diagonal(similarities, 0)  # Ignore self-similarity
        
        # Iteratively remove most redundant samples
        indices_to_remove = []
        for _ in range(n_to_remove):
            # Find sample with highest average similarity (most redundant)
            avg_similarities = similarities.mean(axis=1)
            
            # Mask already-removed indices
            for idx in indices_to_remove:
                avg_similarities[idx] = -1
            
            most_redundant_idx = np.argmax(avg_similarities)
            indices_to_remove.append(most_redundant_idx)
            
            # Zero out this sample's similarities to not affect future selections
            similarities[most_redundant_idx, :] = 0
            similarities[:, most_redundant_idx] = 0
        
        # Remove samples at identified indices (in reverse order to preserve indices)
        indices_to_keep = [i for i in range(len(self.seen_samples)) if i not in indices_to_remove]
        
        self.seen_samples = [self.seen_samples[i] for i in indices_to_keep]
        self.seen_embeddings = [self.seen_embeddings[i] for i in indices_to_keep]
        
        # Rebuild hash set
        if self.enable_hash_dedup:
            self.seen_hashes = {self._compute_hash(s) for s in self.seen_samples}
    
    def _prune_cluster_based(self):
        """
        Keep one representative per cluster.
        
        Uses K-Means to cluster samples and keeps the sample closest
        to each cluster centroid.
        """
        from sklearn.cluster import KMeans
        
        if len(self.seen_embeddings) < self.max_memory_size:
            return
        
        embeddings_array = np.array(self.seen_embeddings)
        
        # Cluster into max_memory_size clusters
        n_clusters = min(self.max_memory_size, len(embeddings_array))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        
        # Keep sample closest to each centroid
        indices_to_keep = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                cluster_embeddings = embeddings_array[cluster_mask]
                distances = np.linalg.norm(
                    cluster_embeddings - kmeans.cluster_centers_[i], axis=1
                )
                closest_local_idx = np.argmin(distances)
                closest_global_idx = cluster_indices[closest_local_idx]
                indices_to_keep.append(closest_global_idx)
        
        # Keep only selected samples
        self.seen_samples = [self.seen_samples[i] for i in indices_to_keep]
        self.seen_embeddings = [self.seen_embeddings[i] for i in indices_to_keep]
        
        # Rebuild hash set
        if self.enable_hash_dedup:
            self.seen_hashes = {self._compute_hash(s) for s in self.seen_samples}
    
    def reset_memory(self):
        """Reset memory of seen samples (for new generation batch)."""
        self.seen_hashes.clear()
        self.seen_embeddings.clear()
        self.seen_samples.clear()
        self._prune_count = 0
        if self.verbose:
            print("🔄 Memory reset - starting fresh batch")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        return {
            'seen_hashes': len(self.seen_hashes),
            'seen_embeddings': len(self.seen_embeddings),
            'seen_samples': len(self.seen_samples),
            'max_memory_size': self.max_memory_size,
            'memory_strategy': self.memory_strategy,
            'prune_count': self._prune_count,
            'memory_utilization': len(self.seen_samples) / self.max_memory_size if self.max_memory_size > 0 else 0
        }

