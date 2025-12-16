"""
Adaptive Iteration Predictor for Qualsynth

Predicts optimal iteration count before running experiments using:
1. Dataset complexity analysis
2. Sample size estimation (PAC learning bounds)
3. Validation rate prediction
4. Historical performance modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class IterationPrediction:
    """Result of iteration prediction."""
    predicted_iterations: int
    min_iterations: int
    max_iterations: int
    confidence: float
    reasoning: str
    estimated_time_minutes: float
    expected_samples: int
    expected_validation_rate: float


class AdaptiveIterationPredictor:
    """
    Adaptive Iteration Predictor using multiple signals:
    
    Methods:
    1. Dataset Complexity Analysis (features, samples, imbalance)
    2. Sample Size Estimation (statistical power analysis)
    3. Validation Rate Prediction (based on complexity)
    4. Historical Performance Modeling (if available)
    """
    
    def __init__(
        self,
        target_samples: Optional[int] = None,
        max_samples: int = 1000,
        max_time_hours: float = 2.0,
        batch_size: int = 1000,
        duplicate_threshold: float = 0.10,
        quality_threshold: float = 0.3,
        time_per_iteration_minutes: float = 3.5,
        verbose: bool = True
    ):
        """
        Initialize adaptive iteration predictor with time-aware scaling.
        
        Args:
            target_samples: Target number of samples to generate (None = auto-calculate from imbalance)
            max_samples: Maximum samples to generate (cap for very large datasets)
            max_time_hours: Maximum allowed runtime in hours (default: 2 hours)
            batch_size: Samples generated per iteration
            duplicate_threshold: Gower distance threshold for duplicates
            quality_threshold: Quality threshold for validation
            time_per_iteration_minutes: Average time per iteration
            verbose: Print detailed reasoning
        """
        self.target_samples = target_samples  # Can be None for adaptive
        self.max_samples = max_samples
        self.max_time_hours = max_time_hours
        self.batch_size = batch_size
        self.duplicate_threshold = duplicate_threshold
        self.quality_threshold = quality_threshold
        self.time_per_iteration = time_per_iteration_minutes
        self.verbose = verbose
    
    def predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None,
        dataset_name: str = "unknown"
    ) -> IterationPrediction:
        """
        Predict optimal iteration count before running experiment.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_features: Sensitive features (optional)
            dataset_name: Dataset name for historical lookup
        
        Returns:
            IterationPrediction with optimal iteration count and reasoning
        """
        if self.verbose:
            print("🔮 Adaptive Iteration Predictor")
            print("="*80)
        
        # Step 0: Calculate adaptive target samples if not provided
        target_samples = self._calculate_target_samples(y_train)
        
        # Step 1: Analyze dataset complexity
        complexity = self._analyze_complexity(X_train, y_train, sensitive_features)
        
        # Step 2: Predict validation rate
        validation_rate = self._predict_validation_rate(complexity)
        
        # Step 3: Calculate required iterations
        iterations = self._calculate_iterations(
            target_samples=target_samples,
            validation_rate=validation_rate,
            complexity=complexity
        )
        
        # Step 4: Apply safety margins
        final_iterations = self._apply_safety_margins(iterations, complexity)
        
        # Step 5: Generate prediction
        prediction = self._generate_prediction(
            iterations=final_iterations,
            validation_rate=validation_rate,
            complexity=complexity,
            target_samples=target_samples
        )
        
        if self.verbose:
            self._print_prediction(prediction, complexity, target_samples)
        
        return prediction
    
    def _calculate_target_samples(self, y_train: pd.Series) -> int:
        """
        Calculate adaptive target samples with time-aware scaling.
        
        Strategy:
        1. If target_samples is provided explicitly, use it
        2. Calculate full balance gap
        3. Apply smart scaling based on dataset size and time constraints
        
        Tiered approach:
        - Small gap (<500): Full balance
        - Medium gap (500-2000): Partial balance (70%)
        - Large gap (>2000): Time-constrained balance
        
        Args:
            y_train: Training labels
        
        Returns:
            Target number of samples to generate
        """
        # If explicitly provided, use it
        if self.target_samples is not None:
            return self.target_samples
        
        # Calculate adaptive target (ALWAYS FULL 1:1 BALANCE)
        class_counts = y_train.value_counts()
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        gap = majority_count - minority_count
        
        # ALWAYS target full balance (1:1 ratio)
        # Apply practical constraints: max_samples cap
        target = min(gap, self.max_samples)
        strategy = f"Full 1:1 balance (target: {target}/{gap} samples)"
        
        if self.verbose:
            print(f"📊 Adaptive Target Calculation:")
            print(f"   Majority class: {majority_count} samples")
            print(f"   Minority class: {minority_count} samples")
            print(f"   Gap (full balance): {gap} samples")
            print(f"   Strategy: {strategy}")
            print(f"   ✅ Target: {target} samples")
            
            # Show impact
            final_minority = minority_count + target
            final_ratio = majority_count / final_minority
            print(f"   📈 Impact: {majority_count/minority_count:.2f}:1 → {final_ratio:.2f}:1")
            print()
        
        return target
    
    def _analyze_complexity(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze dataset complexity using multiple signals."""
        n_samples = len(X_train)
        n_features = len(X_train.columns)
        
        # Class imbalance
        class_counts = y_train.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        # Feature complexity
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        n_categorical = len(categorical_features)
        n_numerical = len(numerical_features)
        
        # Categorical cardinality (avg unique values per categorical feature)
        avg_cardinality = 0
        if n_categorical > 0:
            cardinalities = [X_train[col].nunique() for col in categorical_features]
            avg_cardinality = np.mean(cardinalities)
        
        # Feature correlation (measure of redundancy)
        if n_numerical > 0:
            corr_matrix = X_train[numerical_features].corr().abs()
            # Exclude diagonal
            np.fill_diagonal(corr_matrix.values, 0)
            avg_correlation = corr_matrix.values.mean()
        else:
            avg_correlation = 0.0
        
        # Sensitive feature complexity
        n_sensitive = len(sensitive_features.columns) if sensitive_features is not None else 0
        
        # Overall complexity score (0-1, higher = more complex)
        complexity_score = self._calculate_complexity_score(
            n_samples=n_samples,
            n_features=n_features,
            imbalance_ratio=imbalance_ratio,
            avg_cardinality=avg_cardinality,
            avg_correlation=avg_correlation,
            n_sensitive=n_sensitive
        )
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_categorical': n_categorical,
            'n_numerical': n_numerical,
            'imbalance_ratio': imbalance_ratio,
            'avg_cardinality': avg_cardinality,
            'avg_correlation': avg_correlation,
            'n_sensitive': n_sensitive,
            'complexity_score': complexity_score
        }
    
    def _calculate_complexity_score(
        self,
        n_samples: int,
        n_features: int,
        imbalance_ratio: float,
        avg_cardinality: float,
        avg_correlation: float,
        n_sensitive: int
    ) -> float:
        """
        Calculate overall complexity score (0-1).
        
        Higher score = more complex = needs more iterations
        """
        # Normalize each factor to 0-1
        
        # 1. Sample size (smaller = more complex)
        sample_score = max(0, 1 - (n_samples / 10000))  # 0 at 10k+, 1 at 0
        
        # 2. Feature count (more features = more complex)
        feature_score = min(1, n_features / 50)  # 0 at 0, 1 at 50+
        
        # 3. Imbalance (higher ratio = more complex)
        imbalance_score = min(1, (imbalance_ratio - 1) / 4)  # 0 at 1:1, 1 at 5:1+
        
        # 4. Categorical cardinality (higher = more complex)
        cardinality_score = min(1, avg_cardinality / 20)  # 0 at 0, 1 at 20+
        
        # 5. Feature correlation (lower = more complex, less redundancy)
        correlation_score = 1 - avg_correlation  # 0 at high corr, 1 at low corr
        
        # 6. Sensitive features (more = more complex fairness constraints)
        sensitive_score = min(1, n_sensitive / 3)  # 0 at 0, 1 at 3+
        
        # Weighted average
        weights = {
            'sample': 0.15,
            'feature': 0.20,
            'imbalance': 0.25,
            'cardinality': 0.15,
            'correlation': 0.10,
            'sensitive': 0.15
        }
        
        complexity = (
            weights['sample'] * sample_score +
            weights['feature'] * feature_score +
            weights['imbalance'] * imbalance_score +
            weights['cardinality'] * cardinality_score +
            weights['correlation'] * correlation_score +
            weights['sensitive'] * sensitive_score
        )
        
        return complexity
    
    def _predict_validation_rate(self, complexity: Dict[str, Any]) -> float:
        """
        Predict validation pass rate based on complexity.
        
        Uses empirical model:
        - Simple datasets: 15-25% pass rate
        - Medium datasets: 10-15% pass rate
        - Complex datasets: 5-10% pass rate
        """
        complexity_score = complexity['complexity_score']
        
        # Base rate (for medium complexity)
        base_rate = 0.12  # Empirical: ~12% pass rate
        
        # Adjust based on complexity
        # Low complexity (0-0.3): +10% pass rate
        # Medium complexity (0.3-0.7): base rate
        # High complexity (0.7-1.0): -10% pass rate
        
        if complexity_score < 0.3:
            # Simple dataset
            adjustment = (0.3 - complexity_score) * 0.33  # Up to +10%
            predicted_rate = base_rate + adjustment
        elif complexity_score > 0.7:
            # Complex dataset
            adjustment = (complexity_score - 0.7) * 0.33  # Up to -10%
            predicted_rate = base_rate - adjustment
        else:
            # Medium complexity
            predicted_rate = base_rate
        
        # Additional adjustments
        
        # High imbalance reduces pass rate
        if complexity['imbalance_ratio'] > 3:
            predicted_rate *= 0.9
        
        # High categorical cardinality reduces pass rate
        if complexity['avg_cardinality'] > 15:
            predicted_rate *= 0.9
        
        # Many sensitive features reduce pass rate (stricter fairness)
        if complexity['n_sensitive'] > 2:
            predicted_rate *= 0.85
        
        # Clamp to reasonable range
        predicted_rate = max(0.05, min(0.50, predicted_rate))
        
        return predicted_rate
    
    def _calculate_iterations(
        self,
        target_samples: int,
        validation_rate: float,
        complexity: Dict[str, Any]
    ) -> int:
        """
        Calculate required iterations using sample size estimation.
        
        Formula:
        iterations = ceil(target_samples / (batch_size * validation_rate * efficiency))
        
        Efficiency factor accounts for:
        - Duplicate generation across iterations
        - Quality degradation over time
        - Convergence slowdown
        """

        efficiency_factor = 0.15  # Realistic: 15% efficiency with aggressive filtering
        
        # Base calculation with efficiency
        samples_per_iteration = self.batch_size * validation_rate * efficiency_factor
        base_iterations = np.ceil(target_samples / samples_per_iteration)
        
        # Adjust for convergence time
        # Complex datasets need more iterations to converge
        complexity_multiplier = 1 + (complexity['complexity_score'] * 0.5)  # 1.0 to 1.5x
        
        adjusted_iterations = base_iterations * complexity_multiplier
        
        return int(adjusted_iterations)
    
    def _apply_safety_margins(
        self,
        iterations: int,
        complexity: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Apply safety margins with dynamic time-based caps.
        
        Returns dict with min, recommended, max iterations
        """
        # Calculate time-based maximum
        max_time_minutes = self.max_time_hours * 60
        time_based_max = int(max_time_minutes / self.time_per_iteration)
        
        # Safety margin: +20% for uncertainty
        recommended = max(5, int(iterations * 1.2))  # Absolute min: 5
        
        # Cap recommended at time-based maximum
        recommended = min(recommended, time_based_max)
        
        # Minimum: 80% of recommended (absolute min: 5)
        minimum = max(5, int(recommended * 0.8))
        
        # Maximum: 120% of recommended (capped by time)
        maximum = max(recommended + 2, min(time_based_max, int(recommended * 1.2)))
        
        # Adjust based on complexity (within time constraints)
        if complexity['complexity_score'] > 0.7:
            # Very complex: increase max slightly
            maximum = min(time_based_max, maximum + 3)
        elif complexity['complexity_score'] < 0.3:
            # Simple: decrease max slightly
            maximum = max(minimum + 2, maximum - 3)
        
        return {
            'min': minimum,
            'recommended': recommended,
            'max': maximum
        }
    
    def _generate_prediction(
        self,
        iterations: Dict[str, int],
        validation_rate: float,
        complexity: Dict[str, Any],
        target_samples: int
    ) -> IterationPrediction:
        """Generate final prediction with reasoning."""
        recommended = iterations['recommended']
        
        # Calculate confidence (0-1)
        # Higher complexity = lower confidence
        confidence = 1 - (complexity['complexity_score'] * 0.3)  # 0.7 to 1.0
        
        # Generate reasoning
        reasoning_parts = []
        
        # Complexity assessment
        if complexity['complexity_score'] < 0.3:
            reasoning_parts.append("✅ Low complexity dataset")
        elif complexity['complexity_score'] < 0.7:
            reasoning_parts.append("⚠️  Medium complexity dataset")
        else:
            reasoning_parts.append("🔴 High complexity dataset")
        
        # Validation rate
        reasoning_parts.append(f"📊 Predicted validation rate: {validation_rate*100:.1f}%")
        
        # Imbalance
        if complexity['imbalance_ratio'] > 3:
            reasoning_parts.append(f"⚖️  High imbalance ({complexity['imbalance_ratio']:.1f}:1)")
        
        # Features
        reasoning_parts.append(f"📐 {complexity['n_features']} features ({complexity['n_categorical']} cat, {complexity['n_numerical']} num)")
        
        # Sensitive features
        if complexity['n_sensitive'] > 0:
            reasoning_parts.append(f"🔒 {complexity['n_sensitive']} sensitive features (stricter fairness)")
        
        reasoning = " | ".join(reasoning_parts)
        
        # Estimated time
        estimated_time = recommended * self.time_per_iteration
        
        # Expected samples (accounting for efficiency factor)
        efficiency_factor = 0.15  # Match calculation method
        expected_samples = int(recommended * self.batch_size * validation_rate * efficiency_factor)
        
        return IterationPrediction(
            predicted_iterations=recommended,
            min_iterations=iterations['min'],
            max_iterations=iterations['max'],
            confidence=confidence,
            reasoning=reasoning,
            estimated_time_minutes=estimated_time,
            expected_samples=expected_samples,
            expected_validation_rate=validation_rate
        )
    
    def _print_prediction(
        self,
        prediction: IterationPrediction,
        complexity: Dict[str, Any],
        target_samples: int
    ):
        """Print prediction details."""
        print()
        print("📊 Dataset Analysis:")
        print(f"   Samples: {complexity['n_samples']}")
        print(f"   Features: {complexity['n_features']} ({complexity['n_categorical']} categorical, {complexity['n_numerical']} numerical)")
        print(f"   Imbalance: {complexity['imbalance_ratio']:.2f}:1")
        print(f"   Complexity Score: {complexity['complexity_score']:.2f} (0=simple, 1=complex)")
        print()
        print("🔮 Prediction:")
        print(f"   Recommended Iterations: {prediction.predicted_iterations}")
        print(f"   Range: {prediction.min_iterations}-{prediction.max_iterations} iterations")
        print(f"   Confidence: {prediction.confidence*100:.0f}%")
        print()
        print("📈 Expected Results:")
        print(f"   Target Samples: {target_samples}")
        print(f"   Validation Rate: {prediction.expected_validation_rate*100:.1f}%")
        print(f"   Expected Samples: {prediction.expected_samples}")
        print(f"   Estimated Time: {prediction.estimated_time_minutes:.1f} minutes")
        print()
        print("💡 Reasoning:")
        print(f"   {prediction.reasoning}")
        print()
        print("="*80)

