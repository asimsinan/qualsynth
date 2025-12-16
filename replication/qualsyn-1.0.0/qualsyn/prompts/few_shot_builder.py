"""
Few-Shot Example Builder for Qualsynth

This module selects and formats few-shot examples for LLM prompts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans


class FewShotBuilder:
    """
    Selects and formats few-shot examples for LLM prompts.
    
    Provides:
    - Representative sample selection
    - Diverse example selection
    - Sparse region examples
    - Annotated examples with fairness context
    """
    
    @staticmethod
    def get_optimal_few_shot_count(n_samples: int) -> int:
        """
        Determine optimal number of few-shot examples based on dataset size.
        
        Uses tiered approach:
        - Small datasets (<5K samples): 10 examples
        - Medium datasets (5K-20K): 15 examples
        - Large datasets (>20K): 20 examples
        
        Args:
            n_samples: Number of samples in training set
        
        Returns:
            Optimal number of few-shot examples
        """
        if n_samples < 5000:
            return 10  # Small datasets
        elif n_samples < 20000:
            return 15  # Medium datasets
        else:
            return 20  # Large datasets
    
    @staticmethod
    def select_representative_samples(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_class: int = 1,
        n_samples: int = 5,
        sensitive_features: Optional[pd.DataFrame] = None,
        target_group: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
        selection_strategy: str = 'mixed'
    ) -> pd.DataFrame:
        """
        Select representative samples from training data with iteration-aware diversity.
        
        IMPROVEMENT #2: Dynamic few-shot selection that varies examples each iteration
        to prevent LLM mode collapse from seeing the same examples repeatedly.
        
        Args:
            X_train: Training features
            y_train: Training labels
            target_class: Target class (usually minority)
            n_samples: Number of samples to select
            sensitive_features: Sensitive features (for fairness)
            target_group: Target group specification (e.g., {'sex': 0})
            iteration: Current iteration number (for dynamic selection)
            selection_strategy: Strategy for selection:
                - 'rotate': Different K-Means seed each iteration
                - 'edge_cases': Select from distribution edges
                - 'stratified': Ensure coverage of different feature regions
                - 'mixed': Rotate through all strategies based on iteration
        
        Returns:
            DataFrame of representative samples
        """
        # Filter by target class
        mask = y_train == target_class
        X_target = X_train[mask]
        
        if len(X_target) == 0:
            return pd.DataFrame()
        
        # Further filter by target group (if specified)
        if target_group and sensitive_features is not None:
            for attr, value in target_group.items():
                if attr in sensitive_features.columns:
                    group_mask = sensitive_features[attr] == value
                    X_target = X_target[group_mask[mask]]
        
        if len(X_target) == 0:
            return pd.DataFrame()
        
        # Select diverse samples using K-Means clustering
        n_samples = min(n_samples, len(X_target))
        
        if n_samples == len(X_target):
            return X_target
        
        # Get numerical columns for clustering
        numerical_cols = X_target.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            # Random selection with iteration-based seed
            return X_target.sample(n=n_samples, random_state=42 + iteration * 7)
        
        X_numerical = X_target[numerical_cols].fillna(0)
        
        # DYNAMIC SELECTION BASED ON STRATEGY
        if selection_strategy == 'mixed':
            # Rotate through strategies based on iteration
            strategies = ['rotate', 'edge_cases', 'stratified']
            actual_strategy = strategies[iteration % len(strategies)]
        else:
            actual_strategy = selection_strategy
        
        if actual_strategy == 'rotate':
            # Different K-Means seed each iteration for different clusters
            random_state = 42 + (iteration * 7)  # Prime multiplier for better spread
            return FewShotBuilder._select_via_kmeans(
                X_target, X_numerical, n_samples, random_state
            )
        
        elif actual_strategy == 'edge_cases':
            # Select samples far from centroid (edge cases)
            return FewShotBuilder._select_edge_cases(
                X_target, X_numerical, n_samples, iteration
            )
        
        elif actual_strategy == 'stratified':
            # Stratified sampling across feature ranges
            return FewShotBuilder._select_stratified(
                X_target, X_numerical, n_samples, iteration
            )
        
        else:
            # Default: standard K-Means with iteration-based seed
            random_state = 42 + (iteration * 7)
            return FewShotBuilder._select_via_kmeans(
                X_target, X_numerical, n_samples, random_state
            )
    
    @staticmethod
    def _select_via_kmeans(
        X_target: pd.DataFrame,
        X_numerical: pd.DataFrame,
        n_samples: int,
        random_state: int
    ) -> pd.DataFrame:
        """
        Select samples using K-Means clustering.
        
        Args:
            X_target: Target class samples (all features)
            X_numerical: Numerical features for clustering
            n_samples: Number of samples to select
            random_state: Random state for K-Means
        
        Returns:
            Selected samples DataFrame
        """
        kmeans = KMeans(n_clusters=n_samples, random_state=random_state, n_init=10)
        kmeans.fit(X_numerical)
        
        # Select samples closest to cluster centers
        selected_indices = []
        for i in range(n_samples):
            cluster_mask = kmeans.labels_ == i
            cluster_samples = X_numerical[cluster_mask]
            
            if len(cluster_samples) > 0:
                # Find sample closest to center
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_samples.values - center, axis=1)
                closest_idx = cluster_samples.index[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return X_target.loc[selected_indices]
    
    @staticmethod
    def _select_edge_cases(
        X_target: pd.DataFrame,
        X_numerical: pd.DataFrame,
        n_samples: int,
        iteration: int
    ) -> pd.DataFrame:
        """
        Select samples from distribution edges (far from centroid).
        
        These are "unusual but valid" samples that help the LLM explore
        the full feature space rather than just typical cases.
        
        Args:
            X_target: Target class samples (all features)
            X_numerical: Numerical features for distance calculation
            n_samples: Number of samples to select
            iteration: Current iteration (for variety)
        
        Returns:
            Selected edge case samples DataFrame
        """
        # Compute centroid of all samples
        centroid = X_numerical.mean().values
        
        # Compute distance of each sample from centroid
        distances = np.linalg.norm(X_numerical.values - centroid, axis=1)
        
        # Sort by distance (farthest first = edge cases)
        sorted_indices = np.argsort(distances)[::-1]
        
        # Select from different "distance bands" based on iteration
        # This ensures variety across iterations
        n_total = len(sorted_indices)
        
        if iteration % 2 == 0:
            # Even iterations: select from farthest samples
            start_pct = 0.0
            end_pct = 0.3
        else:
            # Odd iterations: select from moderately far samples
            start_pct = 0.2
            end_pct = 0.5
        
        start_idx = int(n_total * start_pct)
        end_idx = int(n_total * end_pct)
        
        # Ensure we have enough samples in the range
        candidate_indices = sorted_indices[start_idx:max(end_idx, start_idx + n_samples * 2)]
        
        # Add some randomness based on iteration
        np.random.seed(42 + iteration * 13)
        selected_positions = np.random.choice(
            len(candidate_indices), 
            size=min(n_samples, len(candidate_indices)), 
            replace=False
        )
        
        selected_original_indices = X_numerical.index[candidate_indices[selected_positions]]
        return X_target.loc[selected_original_indices]
    
    @staticmethod
    def _select_stratified(
        X_target: pd.DataFrame,
        X_numerical: pd.DataFrame,
        n_samples: int,
        iteration: int
    ) -> pd.DataFrame:
        """
        Select samples using stratified sampling across feature ranges.
        
        Ensures coverage of low, medium, and high values for key features.
        
        Args:
            X_target: Target class samples (all features)
            X_numerical: Numerical features for stratification
            n_samples: Number of samples to select
            iteration: Current iteration (for feature rotation)
        
        Returns:
            Selected stratified samples DataFrame
        """
        n_features = X_numerical.shape[1]
        
        if n_features == 0:
            return X_target.sample(n=min(n_samples, len(X_target)), random_state=42 + iteration)
        
        # Rotate which feature to stratify on based on iteration
        feature_idx = iteration % n_features
        feature_name = X_numerical.columns[feature_idx]
        feature_values = X_numerical[feature_name]
        
        # Divide into terciles (low, medium, high)
        tercile_1 = feature_values.quantile(0.33)
        tercile_2 = feature_values.quantile(0.67)
        
        low_mask = feature_values <= tercile_1
        mid_mask = (feature_values > tercile_1) & (feature_values <= tercile_2)
        high_mask = feature_values > tercile_2
        
        # Allocate samples across terciles
        n_per_tercile = n_samples // 3
        n_remainder = n_samples % 3
        
        selected_indices = []
        
        for i, mask in enumerate([low_mask, mid_mask, high_mask]):
            tercile_samples = X_numerical[mask]
            n_select = n_per_tercile + (1 if i < n_remainder else 0)
            
            if len(tercile_samples) > 0:
                # Random selection within tercile with iteration-based seed
                np.random.seed(42 + iteration * 17 + i * 3)
                select_count = min(n_select, len(tercile_samples))
                selected = np.random.choice(
                    tercile_samples.index, 
                    size=select_count, 
                    replace=False
                )
                selected_indices.extend(selected)
        
        # If we didn't get enough, fill with random samples
        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            available = X_numerical.index.difference(selected_indices)
            if len(available) > 0:
                np.random.seed(42 + iteration * 19)
                extra = np.random.choice(
                    available, 
                    size=min(remaining, len(available)), 
                    replace=False
                )
                selected_indices.extend(extra)
        
        return X_target.loc[selected_indices[:n_samples]]
    
    @staticmethod
    def format_examples(
        samples: pd.DataFrame,
        include_annotations: bool = True,
        target_group: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format samples as few-shot examples.
        
        Args:
            samples: Samples to format
            include_annotations: Whether to include annotations
            target_group: Target group specification (for annotations)
        
        Returns:
            Formatted examples string
        """
        if len(samples) == 0:
            return "No examples available."
        
        lines = ["FEW-SHOT EXAMPLES:", ""]
        lines.append("Here are some representative samples from the training data:")
        lines.append("")
        
        for i, (idx, row) in enumerate(samples.iterrows(), 1):
            lines.append(f"Example {i}:")
            
            # Format as JSON
            sample_dict = row.to_dict()
            lines.append("{")
            for key, value in sample_dict.items():
                if isinstance(value, float):
                    lines.append(f'  "{key}": {value:.2f},')
                else:
                    lines.append(f'  "{key}": {value},')
            lines.append("}")
            
            # Annotations
            if include_annotations and target_group:
                annotations = []
                for attr, target_val in target_group.items():
                    if attr in sample_dict:
                        if sample_dict[attr] == target_val:
                            annotations.append(f"{attr}={target_val} ✓ (target group)")
                        else:
                            annotations.append(f"{attr}={sample_dict[attr]} (non-target)")
                
                if annotations:
                    lines.append(f"→ {', '.join(annotations)}")
            
            lines.append("")
        
        lines.append("INSTRUCTION: Generate samples similar to these examples.")
        lines.append("Maintain the same format, types, and value ranges.")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_sparse_region_examples(
        diversity_plan: Any,
        X_train: pd.DataFrame,
        n_examples_per_region: int = 2
    ) -> str:
        """
        Format examples from sparse regions.
        
        Args:
            diversity_plan: Diversity plan from DiversityPlanner
            X_train: Training features
            n_examples_per_region: Number of examples per sparse region
        
        Returns:
            Formatted sparse region examples string
        """
        if not diversity_plan or not hasattr(diversity_plan, 'sparse_regions'):
            return ""
        
        sparse_regions = diversity_plan.sparse_regions
        if not sparse_regions:
            return ""
        
        lines = ["SPARSE REGION EXAMPLES:", ""]
        lines.append("These samples are from SPARSE (underrepresented) regions.")
        lines.append("Generate MORE samples like these to improve coverage:")
        lines.append("")
        
        for region_id, region_info in enumerate(sparse_regions[:3], 1):  # Top 3 sparse regions
            if 'samples' not in region_info or len(region_info['samples']) == 0:
                continue
            
            density = region_info.get('density', 0.0)
            group = region_info.get('group', 'unknown')
            
            lines.append(f"Sparse Region {region_id} ({group}, density={density:.3f}):")
            
            # Get sample indices
            sample_indices = region_info['samples'][:n_examples_per_region]
            
            for i, idx in enumerate(sample_indices, 1):
                if idx in X_train.index:
                    sample = X_train.loc[idx]
                    lines.append(f"  Example {i}:")
                    lines.append("  {")
                    for key, value in sample.items():
                        if isinstance(value, float):
                            lines.append(f'    "{key}": {value:.2f},')
                        else:
                            lines.append(f'    "{key}": {value},')
                    lines.append("  }")
            
            lines.append("")
        
        lines.append("PRIORITY: Generate samples in these sparse regions to improve diversity.")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_counterfactual_examples(
        base_samples: pd.DataFrame,
        protected_attr: str,
        original_value: Any,
        target_value: Any,
        n_examples: int = 2
    ) -> str:
        """
        Format counterfactual examples.
        
        Args:
            base_samples: Base samples to use
            protected_attr: Protected attribute name
            original_value: Original value of protected attribute
            target_value: Target value of protected attribute
            n_examples: Number of examples to show
        
        Returns:
            Formatted counterfactual examples string
        """
        if len(base_samples) == 0:
            return ""
        
        lines = ["COUNTERFACTUAL EXAMPLES:", ""]
        lines.append(f"Generate counterfactuals by changing {protected_attr} from {original_value} to {target_value}:")
        lines.append("")
        
        for i, (idx, row) in enumerate(base_samples.head(n_examples).iterrows(), 1):
            lines.append(f"Example {i}:")
            lines.append("")
            
            # Base sample
            lines.append(f"  Base ({protected_attr}={original_value}):")
            lines.append("  {")
            for key, value in row.items():
                if isinstance(value, float):
                    lines.append(f'    "{key}": {value:.2f},')
                else:
                    lines.append(f'    "{key}": {value},')
            lines.append("  }")
            lines.append("")
            
            # Counterfactual
            lines.append(f"  Counterfactual ({protected_attr}={target_value}):")
            lines.append("  {")
            for key, value in row.items():
                if key == protected_attr:
                    lines.append(f'    "{key}": {target_value},  ← CHANGED')
                else:
                    if isinstance(value, float):
                        lines.append(f'    "{key}": {value:.2f},  ← SAME')
                    else:
                        lines.append(f'    "{key}": {value},  ← SAME')
            lines.append("  }")
            lines.append("")
        
        lines.append("INSTRUCTION: Generate counterfactuals following this pattern.")
        lines.append(f"Change ONLY {protected_attr} (and make minimal adjustments if needed).")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test few-shot builder
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    from src.qualsynth.modules.diversity_planner import DiversityPlanner
    
    print("="*70)
    print("Testing Few-Shot Builder")
    print("="*70)
    
    # Load German Credit dataset
    dataset_name = 'german_credit'
    sensitive_cols = ['race', 'sex']
    
    split_data = load_split(dataset_name, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
    if available_sensitive_cols:
        sensitive_features = X_train[available_sensitive_cols]
    
    # Run fairness audit
    auditor = FairnessAuditor(fairness_threshold=0.05)
    audit_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
    
    # Create diversity plan
    planner = DiversityPlanner()
    diversity_plan = planner.plan(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=audit_report.fairness_targets,
        dataset_name=dataset_name
    )
    
    # Test 1: Select representative samples
    print("\n\nTEST 1: Select Representative Samples")
    print("-"*70)
    
    target_group = {'sex': 0}
    representative_samples = FewShotBuilder.select_representative_samples(
        X_train, y_train,
        target_class=1,
        n_samples=5,
        sensitive_features=sensitive_features,
        target_group=target_group
    )
    
    print(f"Selected {len(representative_samples)} representative samples")
    print(representative_samples.head())
    
    # Test 2: Format examples
    print("\n\nTEST 2: Format Examples")
    print("-"*70)
    formatted = FewShotBuilder.format_examples(
        representative_samples,
        include_annotations=True,
        target_group=target_group
    )
    print(formatted)
    
    # Test 3: Sparse region examples
    print("\n\nTEST 3: Sparse Region Examples")
    print("-"*70)
    sparse_examples = FewShotBuilder.format_sparse_region_examples(
        diversity_plan,
        X_train,
        n_examples_per_region=2
    )
    if sparse_examples:
        print(sparse_examples[:1000], "...")  # Print first 1000 chars
    else:
        print("No sparse region examples")
    
    # Test 4: Counterfactual examples
    print("\n\nTEST 4: Counterfactual Examples")
    print("-"*70)
    
    # Get base samples (sex=1)
    base_samples = X_train[(y_train == 1) & (sensitive_features['sex'] == 1)].head(3)
    
    cf_examples = FewShotBuilder.format_counterfactual_examples(
        base_samples,
        protected_attr='sex',
        original_value=1,
        target_value=0,
        n_examples=2
    )
    print(cf_examples)
    
    print("\n\n✅ Few-Shot Builder Test Complete")

