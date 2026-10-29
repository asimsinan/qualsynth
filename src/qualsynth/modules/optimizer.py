"""
Multi-Objective Optimizer for Qualsynth Framework

This component selects the best samples from validated candidates using:
1. Pareto-optimal selection (fairness + performance + diversity)
2. Weighted scoring based on strategy
3. Fairness-first prioritization

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import warnings
from sklearn.neighbors import NearestNeighbors


@dataclass
class ObjectiveScores:
    """Scores for a single sample across multiple objectives."""
    sample_id: str
    
    # Objective scores (higher is better)
    fairness_score: float = 0.0
    diversity_score: float = 0.0
    quality_score: float = 0.0  # Schema compliance, constraint satisfaction
    
    # Combined score
    weighted_score: float = 0.0
    
    # Pareto dominance
    is_pareto_optimal: bool = False
    dominated_by: List[str] = field(default_factory=list)
    
    # Sample data
    sample: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """
    Result of multi-objective optimization.
    
    This is the output of the Optimizer and will be used by:
    - Re-Auditor to check final fairness
    - Generator module to understand what worked
    """
    total_candidates: int
    selected_samples: int
    
    # Selected samples
    selected_df: Optional[pd.DataFrame] = None
    selected_scores: List[ObjectiveScores] = field(default_factory=list)
    
    # Pareto frontier
    pareto_frontier: List[ObjectiveScores] = field(default_factory=list)
    
    # Statistics
    avg_fairness_score: float = 0.0
    avg_diversity_score: float = 0.0
    avg_quality_score: float = 0.0
    
    # Summary
    summary: str = ""


class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimizer - selects best samples using Pareto optimization.
    
    This is a TOOL (not an agent) that performs deterministic optimization.
    
    Key features:
    1. Pareto-optimal selection (non-dominated samples)
    2. Weighted scoring (fairness + diversity + quality)
    3. Fairness-first prioritization
    4. Diversity maximization within fairness constraints
    """
    
    def __init__(
        self,
        fairness_weight: float = 0.5,
        diversity_weight: float = 0.3,
        quality_weight: float = 0.2,
        use_pareto: bool = True
    ):
        """
        Initialize Multi-Objective Optimizer.
        
        Args:
            fairness_weight: Weight for fairness objective (0-1)
            diversity_weight: Weight for diversity objective (0-1)
            quality_weight: Weight for quality objective (0-1)
            use_pareto: Whether to use Pareto-optimal selection
        """
        # Normalize weights
        total = fairness_weight + diversity_weight + quality_weight
        self.fairness_weight = fairness_weight / total
        self.diversity_weight = diversity_weight / total
        self.quality_weight = quality_weight / total
        self.use_pareto = use_pareto
    
    def optimize(
        self,
        candidates_df: pd.DataFrame,
        n_samples: int,
        existing_data: Optional[pd.DataFrame] = None,
        fairness_targets: Optional[List[Any]] = None,
        diversity_plan: Optional[Any] = None,
        schema: Optional[Any] = None
    ) -> OptimizationResult:
        """
        Select best samples using multi-objective optimization.
        
        Args:
            candidates_df: Valid candidate samples (from Validator)
            n_samples: Number of samples to select
            existing_data: Existing training data
            fairness_targets: Fairness targets from FairnessAuditor
            diversity_plan: Diversity plan from DiversityPlanner
            schema: Schema from SchemaProfiler
        
        Returns:
            OptimizationResult with selected samples
        """
        result = OptimizationResult(
            total_candidates=len(candidates_df),
            selected_samples=min(n_samples, len(candidates_df))
        )
        
        if len(candidates_df) == 0:
            result.summary = "No candidates to optimize"
            return result
        
        # 1. Calculate objective scores for each candidate
        scores = []
        for idx, row in candidates_df.iterrows():
            sample_id = f"candidate_{idx}"
            sample_dict = row.to_dict()
            
            # Calculate fairness score
            fairness_score = self._calculate_fairness_score(
                sample_dict, fairness_targets
            )
            
            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(
                sample_dict, existing_data, diversity_plan, schema
            )
            
            # Calculate quality score (always 1.0 for valid samples)
            quality_score = 1.0
            
            # Calculate weighted score
            weighted_score = (
                self.fairness_weight * fairness_score +
                self.diversity_weight * diversity_score +
                self.quality_weight * quality_score
            )
            
            scores.append(ObjectiveScores(
                sample_id=sample_id,
                fairness_score=fairness_score,
                diversity_score=diversity_score,
                quality_score=quality_score,
                weighted_score=weighted_score,
                sample=sample_dict
            ))
        
        # 2. Pareto-optimal selection (if enabled)
        if self.use_pareto:
            self._identify_pareto_frontier(scores)
            result.pareto_frontier = [s for s in scores if s.is_pareto_optimal]
        
        # 3. Select top n_samples by weighted score
        scores.sort(key=lambda s: s.weighted_score, reverse=True)
        selected_scores = scores[:result.selected_samples]
        
        # 4. Create selected DataFrame
        selected_samples = [s.sample for s in selected_scores]
        result.selected_df = pd.DataFrame(selected_samples)
        result.selected_scores = selected_scores
        
        # 5. Calculate statistics
        if selected_scores:
            result.avg_fairness_score = np.mean([s.fairness_score for s in selected_scores])
            result.avg_diversity_score = np.mean([s.diversity_score for s in selected_scores])
            result.avg_quality_score = np.mean([s.quality_score for s in selected_scores])
        
        # 6. Generate summary
        result.summary = self._generate_summary(result)
        
        return result
    
    def _calculate_fairness_score(
        self,
        sample: Dict[str, Any],
        fairness_targets: Optional[List[Any]]
    ) -> float:
        """
        Calculate fairness score for a sample.
        
        Higher score = better fairness (sample belongs to target group).
        """
        if not fairness_targets:
            return 1.0
        
        score = 0.0
        n_targets = len(fairness_targets)
        
        for target in fairness_targets:
            if not hasattr(target, 'attribute'):
                continue
            
            attr = target.attribute
            target_group = target.target_group
            
            if attr in sample:
                # Check if sample belongs to target group
                if sample[attr] == target_group:
                    # Higher priority = higher score
                    if target.priority == "high":
                        score += 1.0
                    elif target.priority == "medium":
                        score += 0.7
                    else:
                        score += 0.5
                else:
                    # Sample doesn't belong to target group
                    score += 0.1
        
        return score / n_targets if n_targets > 0 else 1.0
    
    def _calculate_diversity_score(
        self,
        sample: Dict[str, Any],
        existing_data: Optional[pd.DataFrame],
        diversity_plan: Optional[Any],
        schema: Optional[Any]
    ) -> float:
        """
        Calculate diversity score for a sample.
        
        Higher score = more diverse (farther from existing samples).
        """
        if existing_data is None or len(existing_data) == 0:
            return 1.0
        
        # Convert sample to DataFrame
        sample_df = pd.DataFrame([sample])
        
        # Get common columns
        common_cols = [col for col in sample_df.columns if col in existing_data.columns]
        if not common_cols:
            return 1.0
        
        sample_df = sample_df[common_cols]
        existing_df = existing_data[common_cols]
        
        # Get numerical columns for distance calculation
        # Convert to numeric, coercing errors to NaN
        sample_df_numeric = sample_df.copy()
        existing_df_numeric = existing_df.copy()
        
        for col in common_cols:
            sample_df_numeric[col] = pd.to_numeric(sample_df_numeric[col], errors='coerce')
            existing_df_numeric[col] = pd.to_numeric(existing_df_numeric[col], errors='coerce')
        
        # Select only numeric columns (non-NaN)
        numerical_cols = sample_df_numeric.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out columns with all NaN
        numerical_cols = [col for col in numerical_cols if not sample_df_numeric[col].isna().all()]
        
        if not numerical_cols:
            return 0.5
        
        # Fill NaN with 0 for distance calculation
        sample_numerical = sample_df_numeric[numerical_cols].fillna(0).values
        existing_numerical = existing_df_numeric[numerical_cols].fillna(0).values
        
        # Calculate minimum distance to existing samples
        try:
            from scipy.spatial.distance import cdist
            distances = cdist(sample_numerical, existing_numerical, metric='euclidean')
            min_distance = np.min(distances)
            
            # Normalize to [0, 1] (higher = more diverse)
            # Enhanced diversity bonus: more aggressive reward for diverse samples
            # Changed from sigmoid to linear scaling with bonus
            # min_distance typically ranges from 0-10
            diversity_score = min(1.0, min_distance / 5.0)  # Linear scaling
            
            # Add diversity bonus for very diverse samples (distance > 3)
            if min_distance > 3.0:
                diversity_bonus = min(0.2, (min_distance - 3.0) / 10.0)
                diversity_score = min(1.0, diversity_score + diversity_bonus)
            
            return float(diversity_score)
        
        except Exception as e:
            warnings.warn(f"Diversity calculation failed: {e}")
            return 0.5
    
    def _identify_pareto_frontier(self, scores: List[ObjectiveScores]) -> None:
        """
        Identify Pareto-optimal samples (non-dominated).
        
        A sample is Pareto-optimal if no other sample is better in all objectives.
        """
        n = len(scores)
        
        for i in range(n):
            is_dominated = False
            
            for j in range(n):
                if i == j:
                    continue
                
                # Check if sample j dominates sample i
                # (j is better or equal in all objectives, and strictly better in at least one)
                # Ensure scores are floats to avoid type comparison errors
                fairness_i = float(scores[i].fairness_score) if scores[i].fairness_score is not None else 0.0
                fairness_j = float(scores[j].fairness_score) if scores[j].fairness_score is not None else 0.0
                diversity_i = float(scores[i].diversity_score) if scores[i].diversity_score is not None else 0.0
                diversity_j = float(scores[j].diversity_score) if scores[j].diversity_score is not None else 0.0
                quality_i = float(scores[i].quality_score) if scores[i].quality_score is not None else 0.0
                quality_j = float(scores[j].quality_score) if scores[j].quality_score is not None else 0.0
                
                fairness_better = fairness_j >= fairness_i
                diversity_better = diversity_j >= diversity_i
                quality_better = quality_j >= quality_i
                
                fairness_strictly_better = fairness_j > fairness_i
                diversity_strictly_better = diversity_j > diversity_i
                quality_strictly_better = quality_j > quality_i
                
                if (fairness_better and diversity_better and quality_better and
                    (fairness_strictly_better or diversity_strictly_better or quality_strictly_better)):
                    # Sample j dominates sample i
                    is_dominated = True
                    scores[i].dominated_by.append(scores[j].sample_id)
            
            scores[i].is_pareto_optimal = not is_dominated
    
    def _generate_summary(self, result: OptimizationResult) -> str:
        """Generate human-readable summary."""
        lines = []
        
        lines.append(f"Candidates: {result.total_candidates}")
        lines.append(f"Selected: {result.selected_samples}")
        
        if result.selected_samples > 0:
            lines.append(f"Avg fairness: {result.avg_fairness_score:.3f}")
            lines.append(f"Avg diversity: {result.avg_diversity_score:.3f}")
            lines.append(f"Avg quality: {result.avg_quality_score:.3f}")
        
        if result.pareto_frontier:
            lines.append(f"Pareto optimal: {len(result.pareto_frontier)}")
        
        return " | ".join(lines)
    
    def print_result(self, result: OptimizationResult, verbose: bool = True) -> None:
        """Print comprehensive optimization result."""
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULT")
        print(f"{'='*70}")
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total candidates: {result.total_candidates}")
        print(f"   Selected samples: {result.selected_samples}")
        
        if result.selected_samples > 0:
            print(f"\n📈 AVERAGE SCORES:")
            print(f"   Fairness: {result.avg_fairness_score:.3f}")
            print(f"   Diversity: {result.avg_diversity_score:.3f}")
            print(f"   Quality: {result.avg_quality_score:.3f}")
        
        # Pareto frontier
        if result.pareto_frontier:
            print(f"\n⭐ PARETO FRONTIER:")
            print(f"   Pareto-optimal samples: {len(result.pareto_frontier)}")
            
            if verbose:
                print(f"\n   Top 5 Pareto-optimal samples:")
                for i, score in enumerate(result.pareto_frontier[:5], 1):
                    print(f"      {i}. {score.sample_id}:")
                    print(f"         Fairness: {score.fairness_score:.3f}")
                    print(f"         Diversity: {score.diversity_score:.3f}")
                    print(f"         Quality: {score.quality_score:.3f}")
                    print(f"         Weighted: {score.weighted_score:.3f}")
        
        # Selected samples
        if verbose and result.selected_scores:
            print(f"\n✅ SELECTED SAMPLES (Top 5):")
            for i, score in enumerate(result.selected_scores[:5], 1):
                pareto_mark = "⭐" if score.is_pareto_optimal else ""
                print(f"      {i}. {score.sample_id} {pareto_mark}:")
                print(f"         Fairness: {score.fairness_score:.3f}")
                print(f"         Diversity: {score.diversity_score:.3f}")
                print(f"         Weighted: {score.weighted_score:.3f}")
        
        # Weights
        print(f"\n⚖️  OPTIMIZATION WEIGHTS:")
        print(f"   Fairness: {self.fairness_weight:.3f}")
        print(f"   Diversity: {self.diversity_weight:.3f}")
        print(f"   Quality: {self.quality_weight:.3f}")
        
        # Summary
        print(f"\n📝 SUMMARY:")
        print(f"   {result.summary}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test the Multi-Objective Optimizer
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.modules.fairness_auditor import FairnessAuditor
    from src.qualsynth.modules.schema_profiler import SchemaProfiler
    from src.qualsynth.modules.diversity_planner import DiversityPlanner
    from src.qualsynth.modules.validator import Validator
    
    print("="*70)
    print("Testing Multi-Objective Optimizer")
    print("="*70)
    
    # Test on German Credit dataset
    dataset_name = 'german_credit'
    sensitive_cols = ['race', 'sex']
    
    print(f"\n\n{'='*70}")
    print(f"OPTIMIZING: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    split_data = load_split(dataset_name, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    # Get sensitive features
    available_sensitive_cols = [col for col in sensitive_cols if col in X_train.columns]
    if available_sensitive_cols:
        sensitive_features = X_train[available_sensitive_cols]
    
    # Run fairness audit
    auditor = FairnessAuditor(fairness_threshold=0.05)
    audit_report = auditor.audit(X_train, y_train, sensitive_features, dataset_name)
    fairness_targets = audit_report.fairness_targets
    
    # Profile schema
    schema_profiler = SchemaProfiler()
    schema = schema_profiler.profile(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=fairness_targets,
        dataset_name=dataset_name
    )
    
    # Create diversity plan
    planner = DiversityPlanner()
    diversity_plan = planner.plan(
        X_train, y_train,
        sensitive_features=available_sensitive_cols,
        fairness_targets=fairness_targets,
        dataset_name=dataset_name
    )
    
    # Get candidate samples (use some minority samples as candidates)
    minority_samples = X_train[y_train == 1].head(100)
    
    # Validate candidates
    validator = Validator(duplicate_threshold=0.05)
    validation_report = validator.validate(
        minority_samples.to_dict('records'),
        schema,
        existing_data=X_train,
        fairness_constraints=schema.fairness_constraints
    )
    
    print(f"\n📊 Validation Results:")
    print(f"   Total candidates: {validation_report.total_samples}")
    print(f"   Valid candidates: {validation_report.valid_samples}")
    print(f"   Validation rate: {validation_report.validation_rate:.1%}")
    
    if validation_report.valid_samples_df is not None and len(validation_report.valid_samples_df) > 0:
        # Test 1: Standard optimization (fairness-aware)
        print(f"\n\n{'='*70}")
        print(f"TEST 1: Standard Optimization (fairness_weight=0.5)")
        print(f"{'='*70}")
        
        optimizer1 = MultiObjectiveOptimizer(
            fairness_weight=0.5,
            diversity_weight=0.3,
            quality_weight=0.2,
            use_pareto=True
        )
        
        result1 = optimizer1.optimize(
            validation_report.valid_samples_df,
            n_samples=20,
            existing_data=X_train,
            fairness_targets=fairness_targets,
            diversity_plan=diversity_plan,
            schema=schema
        )
        
        optimizer1.print_result(result1, verbose=True)
        
        # Test 2: Fairness-first optimization (high fairness weight)
        print(f"\n\n{'='*70}")
        print(f"TEST 2: Fairness-First Optimization (fairness_weight=0.8)")
        print(f"{'='*70}")
        
        optimizer2 = MultiObjectiveOptimizer(
            fairness_weight=0.8,
            diversity_weight=0.15,
            quality_weight=0.05,
            use_pareto=True
        )
        
        result2 = optimizer2.optimize(
            validation_report.valid_samples_df,
            n_samples=20,
            existing_data=X_train,
            fairness_targets=fairness_targets,
            diversity_plan=diversity_plan,
            schema=schema
        )
        
        optimizer2.print_result(result2, verbose=True)
        
        # Test 3: Diversity-focused optimization
        print(f"\n\n{'='*70}")
        print(f"TEST 3: Diversity-Focused Optimization (diversity_weight=0.7)")
        print(f"{'='*70}")
        
        optimizer3 = MultiObjectiveOptimizer(
            fairness_weight=0.2,
            diversity_weight=0.7,
            quality_weight=0.1,
            use_pareto=True
        )
        
        result3 = optimizer3.optimize(
            validation_report.valid_samples_df,
            n_samples=20,
            existing_data=X_train,
            fairness_targets=fairness_targets,
            diversity_plan=diversity_plan,
            schema=schema
        )
        
        optimizer3.print_result(result3, verbose=True)
    
    else:
        print("\n⚠️  No valid candidates for optimization")
    
    print("\n✅ Multi-Objective Optimizer Test Complete")

