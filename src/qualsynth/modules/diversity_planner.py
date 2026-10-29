"""
Fairness-Aware Diversity Planner for Qualsynth Framework

This component identifies sparse regions in the feature space PER protected group
and prioritizes underrepresented subgroups for generation.

Key innovation: Instead of finding sparse regions globally, we find them
WITHIN each protected group to ensure fairness-aware diversity.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings


@dataclass
class SparseRegion:
    """A sparse region in the feature space."""
    region_id: str
    center: np.ndarray  # Center point of the region
    density: float  # Density score (lower = more sparse)
    n_samples: int  # Number of samples in this region
    radius: float  # Radius of the region
    
    # Fairness-aware attributes
    protected_group: Optional[str] = None  # e.g., "sex=0"
    protected_attribute: Optional[str] = None  # e.g., "sex"
    group_value: Optional[Any] = None  # e.g., 0
    
    # Priority
    priority: str = "medium"  # 'low', 'medium', 'high'
    
    # Representative sample (for LLM prompting)
    representative_sample: Optional[Dict[str, Any]] = None


@dataclass
class DiversityTarget:
    """Target for diversity-aware generation."""
    target_id: str
    description: str
    
    # Sparse regions to target
    sparse_regions: List[SparseRegion] = field(default_factory=list)
    
    # Protected group
    protected_group: Optional[str] = None
    
    # Generation guidance
    n_samples_needed: int = 0
    priority: str = "medium"
    
    # Feature constraints for this target
    feature_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiversityPlan:
    """
    Comprehensive diversity plan for generation.
    
    This is the output of the Diversity Planner and will be used by:
    - Generator module to create diverse samples
    - Optimizer to evaluate diversity
    """
    dataset_name: str
    
    # Sparse regions (global and per-group)
    global_sparse_regions: List[SparseRegion] = field(default_factory=list)
    group_sparse_regions: Dict[str, List[SparseRegion]] = field(default_factory=dict)
    
    # Diversity targets
    diversity_targets: List[DiversityTarget] = field(default_factory=list)
    
    # Statistics
    overall_density: float = 0.0
    density_per_group: Dict[str, float] = field(default_factory=dict)
    
    # Summary
    summary: str = ""


class DiversityPlanner:
    """
    Fairness-Aware Diversity Planner - identifies sparse regions per protected group.
    
    This is a TOOL (not an agent) that performs deterministic analysis.
    
    Key innovation: Traditional diversity planners find sparse regions globally.
    This planner finds sparse regions WITHIN each protected group to ensure
    fairness-aware diversity.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        density_threshold: float = 0.5,
        min_samples_per_region: int = 5,
        k_neighbors: int = 10
    ):
        """
        Initialize Diversity Planner.
        
        Args:
            n_clusters: Number of clusters for region identification
            density_threshold: Threshold for sparse region (lower = more sparse)
            min_samples_per_region: Minimum samples to consider a region
            k_neighbors: Number of neighbors for density estimation
        """
        self.n_clusters = n_clusters
        self.density_threshold = density_threshold
        self.min_samples_per_region = min_samples_per_region
        self.k_neighbors = k_neighbors
    
    def plan(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: Optional[List[str]] = None,
        fairness_targets: Optional[List[Any]] = None,
        dataset_name: str = "Dataset"
    ) -> DiversityPlan:
        """
        Create fairness-aware diversity plan.
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_features: List of sensitive feature names
            fairness_targets: Fairness targets from FairnessAuditor
            dataset_name: Name of the dataset
        
        Returns:
            DiversityPlan with sparse regions and targets
        """
        plan = DiversityPlan(dataset_name=dataset_name)
        
        # Focus on minority class
        minority_mask = y == 1
        X_minority = X[minority_mask].copy()
        
        if len(X_minority) < self.min_samples_per_region:
            warnings.warn(f"Too few minority samples ({len(X_minority)}) for diversity planning")
            return plan
        
        # Get numerical features for distance calculations
        numerical_cols = X_minority.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            warnings.warn("No numerical features found for diversity planning")
            return plan
        
        X_minority_numerical = X_minority[numerical_cols].values
        
        # 1. Identify global sparse regions
        plan.global_sparse_regions = self._identify_sparse_regions(
            X_minority_numerical,
            X_minority,
            numerical_cols
        )
        
        # 2. Calculate overall density
        plan.overall_density = self._calculate_density(X_minority_numerical)
        
        # 3. Identify sparse regions PER protected group (fairness-aware)
        if sensitive_features is not None and not sensitive_features.empty:
            for sensitive_attr in sensitive_features.columns:
                if sensitive_attr not in X_minority.columns:
                    continue
                
                # Get unique groups
                groups = X_minority[sensitive_attr].unique()
                
                for group_val in groups:
                    group_mask = X_minority[sensitive_attr] == group_val
                    X_group = X_minority[group_mask]
                    
                    if len(X_group) < self.min_samples_per_region:
                        continue
                    
                    X_group_numerical = X_group[numerical_cols].values
                    
                    # Identify sparse regions for this group
                    group_regions = self._identify_sparse_regions(
                        X_group_numerical,
                        X_group,
                        numerical_cols,
                        protected_group=f"{sensitive_attr}={group_val}",
                        protected_attribute=sensitive_attr,
                        group_value=group_val
                    )
                    
                    group_key = f"{sensitive_attr}={group_val}"
                    plan.group_sparse_regions[group_key] = group_regions
                    
                    # Calculate density for this group
                    plan.density_per_group[group_key] = self._calculate_density(X_group_numerical)
        
        # 4. Create diversity targets
        plan.diversity_targets = self._create_diversity_targets(
            plan,
            fairness_targets
        )
        
        # 5. Generate summary
        plan.summary = self._generate_summary(plan)
        
        return plan
    
    def _identify_sparse_regions(
        self,
        X_numerical: np.ndarray,
        X_full: pd.DataFrame,
        numerical_cols: List[str],
        protected_group: Optional[str] = None,
        protected_attribute: Optional[str] = None,
        group_value: Optional[Any] = None
    ) -> List[SparseRegion]:
        """
        Identify sparse regions using clustering and density estimation.
        
        Strategy:
        1. Cluster the data into n_clusters regions
        2. For each cluster, estimate density using k-NN
        3. Mark low-density clusters as sparse regions
        """
        regions = []
        
        if len(X_numerical) < self.n_clusters:
            # Too few samples, treat as single region
            n_clusters = 1
        else:
            n_clusters = min(self.n_clusters, len(X_numerical))
        
        # Cluster the data
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_numerical)
            cluster_centers = kmeans.cluster_centers_
        except Exception as e:
            warnings.warn(f"Clustering failed: {e}")
            return regions
        
        # For each cluster, estimate density
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = X_numerical[cluster_mask]
            
            if len(cluster_samples) < self.min_samples_per_region:
                continue
            
            # Estimate density using k-NN
            k = min(self.k_neighbors, len(cluster_samples) - 1)
            if k < 1:
                continue
            
            try:
                nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
                nbrs.fit(cluster_samples)
                distances, _ = nbrs.kneighbors(cluster_samples)
                
                # Average distance to k-th neighbor (lower = denser)
                avg_distance = np.mean(distances[:, -1])
                
                # Normalize to density score (higher = denser)
                # Use inverse distance as density
                density = 1.0 / (1.0 + avg_distance)
                
                # Calculate radius (max distance from center)
                center = cluster_centers[cluster_id]
                distances_from_center = cdist([center], cluster_samples)[0]
                radius = float(np.max(distances_from_center))
                
                # Get representative sample (closest to center)
                closest_idx = np.argmin(distances_from_center)
                representative_idx = np.where(cluster_mask)[0][closest_idx]
                representative_sample = X_full.iloc[representative_idx].to_dict()
                
                # Determine priority (lower density = higher priority)
                if density < self.density_threshold * 0.5:
                    priority = "high"
                elif density < self.density_threshold:
                    priority = "medium"
                else:
                    priority = "low"
                
                # Create sparse region
                region_id = f"{protected_group or 'global'}_cluster_{cluster_id}"
                
                regions.append(SparseRegion(
                    region_id=region_id,
                    center=center,
                    density=density,
                    n_samples=len(cluster_samples),
                    radius=radius,
                    protected_group=protected_group,
                    protected_attribute=protected_attribute,
                    group_value=group_value,
                    priority=priority,
                    representative_sample=representative_sample
                ))
            
            except Exception as e:
                warnings.warn(f"Density estimation failed for cluster {cluster_id}: {e}")
                continue
        
        # Sort by density (sparsest first)
        regions.sort(key=lambda r: r.density)
        
        return regions
    
    def _calculate_density(self, X_numerical: np.ndarray) -> float:
        """Calculate overall density of the data."""
        if len(X_numerical) < 2:
            return 0.0
        
        k = min(self.k_neighbors, len(X_numerical) - 1)
        if k < 1:
            return 0.0
        
        try:
            nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
            nbrs.fit(X_numerical)
            distances, _ = nbrs.kneighbors(X_numerical)
            avg_distance = np.mean(distances[:, -1])
            density = 1.0 / (1.0 + avg_distance)
            return float(density)
        except:
            return 0.0
    
    def _create_diversity_targets(
        self,
        plan: DiversityPlan,
        fairness_targets: Optional[List[Any]] = None
    ) -> List[DiversityTarget]:
        """
        Create diversity targets by combining sparse regions with fairness targets.
        
        Strategy:
        - For each fairness target, identify relevant sparse regions
        - Prioritize sparse regions in underrepresented groups
        """
        targets = []
        
        # If we have fairness targets, create diversity targets for each
        if fairness_targets:
            # Extract targets list from audit report if needed
            targets_list = fairness_targets.fairness_targets if hasattr(fairness_targets, 'fairness_targets') else fairness_targets
            
            for fairness_target in targets_list:
                if not hasattr(fairness_target, 'attribute'):
                    continue
                
                attr = fairness_target.attribute
                target_group = fairness_target.target_group
                n_samples = fairness_target.n_samples_needed
                priority = fairness_target.priority
                
                # Find sparse regions for this group
                group_key = f"{attr}={target_group}"
                sparse_regions = plan.group_sparse_regions.get(group_key, [])
                
                # Filter to only sparse regions (below threshold)
                sparse_regions = [r for r in sparse_regions if r.density < self.density_threshold]
                
                if sparse_regions:
                    target = DiversityTarget(
                        target_id=f"fairness_diversity_{attr}_{target_group}",
                        description=f"Generate diverse samples for {attr}={target_group} in sparse regions",
                        sparse_regions=sparse_regions,
                        protected_group=group_key,
                        n_samples_needed=n_samples,
                        priority=priority,
                        feature_constraints={attr: target_group}
                    )
                    targets.append(target)
        
        # Also create targets for global sparse regions (if no fairness targets)
        if not targets and plan.global_sparse_regions:
            sparse_regions = [r for r in plan.global_sparse_regions if r.density < self.density_threshold]
            
            if sparse_regions:
                target = DiversityTarget(
                    target_id="global_diversity",
                    description="Generate diverse samples in global sparse regions",
                    sparse_regions=sparse_regions,
                    n_samples_needed=100,  # Default
                    priority="medium"
                )
                targets.append(target)
        
        return targets
    
    def _generate_summary(self, plan: DiversityPlan) -> str:
        """Generate human-readable summary."""
        lines = []
        
        lines.append(f"Dataset: {plan.dataset_name}")
        lines.append(f"Overall density: {plan.overall_density:.4f}")
        lines.append(f"Global sparse regions: {len(plan.global_sparse_regions)}")
        lines.append(f"Protected groups: {len(plan.group_sparse_regions)}")
        lines.append(f"Diversity targets: {len(plan.diversity_targets)}")
        
        return " | ".join(lines)
    
    def print_plan(self, plan: DiversityPlan, verbose: bool = True) -> None:
        """Print comprehensive diversity plan."""
        print(f"\n{'='*70}")
        print(f"DIVERSITY PLAN: {plan.dataset_name}")
        print(f"{'='*70}")
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Overall density: {plan.overall_density:.4f}")
        print(f"   Global sparse regions: {len(plan.global_sparse_regions)}")
        print(f"   Protected groups analyzed: {len(plan.group_sparse_regions)}")
        
        # Global sparse regions
        if plan.global_sparse_regions and verbose:
            print(f"\n🌍 GLOBAL SPARSE REGIONS ({len(plan.global_sparse_regions)}):")
            for i, region in enumerate(plan.global_sparse_regions[:5], 1):
                print(f"\n   {i}. {region.region_id}:")
                print(f"      Density: {region.density:.4f} (priority: {region.priority.upper()})")
                print(f"      Samples: {region.n_samples}")
                print(f"      Radius: {region.radius:.4f}")
            
            if len(plan.global_sparse_regions) > 5:
                print(f"\n   ... and {len(plan.global_sparse_regions) - 5} more")
        
        # Per-group sparse regions (fairness-aware)
        if plan.group_sparse_regions:
            print(f"\n⚖️  FAIRNESS-AWARE SPARSE REGIONS:")
            for group_key, regions in plan.group_sparse_regions.items():
                density = plan.density_per_group.get(group_key, 0.0)
                print(f"\n   {group_key} (density: {density:.4f}):")
                print(f"      Sparse regions: {len(regions)}")
                
                if verbose and regions:
                    for region in regions[:3]:
                        print(f"         • {region.region_id}: density={region.density:.4f}, "
                              f"n={region.n_samples}, priority={region.priority.upper()}")
                    
                    if len(regions) > 3:
                        print(f"         ... and {len(regions) - 3} more")
        
        # Diversity targets
        if plan.diversity_targets:
            print(f"\n🎯 DIVERSITY TARGETS ({len(plan.diversity_targets)}):")
            for i, target in enumerate(plan.diversity_targets, 1):
                print(f"\n   {i}. {target.target_id}:")
                print(f"      {target.description}")
                print(f"      Priority: {target.priority.upper()}")
                print(f"      Samples needed: {target.n_samples_needed}")
                print(f"      Sparse regions: {len(target.sparse_regions)}")
                
                if target.protected_group:
                    print(f"      Protected group: {target.protected_group}")
                
                if target.feature_constraints:
                    print(f"      Constraints: {target.feature_constraints}")
                
                if verbose and target.sparse_regions:
                    print(f"      Top sparse regions:")
                    for region in target.sparse_regions[:3]:
                        print(f"         • {region.region_id}: density={region.density:.4f}")
        
        # Summary
        print(f"\n📝 SUMMARY:")
        print(f"   {plan.summary}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test the Diversity Planner
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    
    print("="*70)
    print("Testing Fairness-Aware Diversity Planner")
    print("="*70)
    
    planner = DiversityPlanner(
        n_clusters=5,
        density_threshold=0.5,
        min_samples_per_region=10,
        k_neighbors=10
    )
    auditor = FairnessAuditor(fairness_threshold=0.05)
    
    # Test on all 3 datasets
    datasets = [
        ('german_credit', ['personal_status']),
        ('german_credit', ['personal_status', 'age'])
    ]
    
    for dataset_name, sensitive_cols in datasets:
        print(f"\n\n{'='*70}")
        print(f"PLANNING: {dataset_name.upper()}")
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
        
        # Create diversity plan
        diversity_plan = planner.plan(
            X_train,
            y_train,
            sensitive_features=available_sensitive_cols,
            fairness_targets=fairness_targets,
            dataset_name=dataset_name
        )
        
        # Print plan
        planner.print_plan(diversity_plan, verbose=True)
    
    print("\n✅ Diversity Planner Test Complete")

