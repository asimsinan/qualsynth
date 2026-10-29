"""
Fairness Metrics Evaluation

This module implements comprehensive fairness metrics using Fairlearn and AIF360
for evaluating bias in classification models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

# Fairlearn imports
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    MetricFrame
)

# AIF360 imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("Warning: AIF360 not fully available. Some metrics may be limited.")


class FairnessEvaluator:
    """
    Comprehensive fairness evaluator for classification tasks.
    
    Computes fairness metrics using both Fairlearn and AIF360.
    """
    
    def __init__(self):
        """Initialize fairness evaluator."""
        self.metrics = {}
        self.aif360_available = AIF360_AVAILABLE
    
    def evaluate_fairlearn(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute fairness metrics using Fairlearn.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: DataFrame of sensitive attributes (e.g., sex, race)
            y_proba: Optional prediction probabilities
        
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        # Ensure sensitive_features is a DataFrame
        if isinstance(sensitive_features, pd.Series):
            sensitive_features = sensitive_features.to_frame()
        
        # Compute metrics for each sensitive attribute
        for col in sensitive_features.columns:
            sf = sensitive_features[col]
            
            # Demographic Parity Difference (DPD)
            try:
                dpd = demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sf
                )
                metrics[f'{col}_demographic_parity_difference'] = dpd
            except Exception as e:
                metrics[f'{col}_demographic_parity_difference'] = np.nan
            
            # Demographic Parity Ratio (DPR)
            try:
                dpr = demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=sf
                )
                metrics[f'{col}_demographic_parity_ratio'] = dpr
            except Exception as e:
                metrics[f'{col}_demographic_parity_ratio'] = np.nan
            
            # Equalized Odds Difference (EOD)
            try:
                eod = equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sf
                )
                metrics[f'{col}_equalized_odds_difference'] = eod
            except Exception as e:
                metrics[f'{col}_equalized_odds_difference'] = np.nan
            
            # Equalized Odds Ratio (EOR)
            try:
                eor = equalized_odds_ratio(
                    y_true, y_pred, sensitive_features=sf
                )
                metrics[f'{col}_equalized_odds_ratio'] = eor
            except Exception as e:
                metrics[f'{col}_equalized_odds_ratio'] = np.nan
        
        # Overall fairness score (average absolute DPD and EOD)
        dpd_values = [v for k, v in metrics.items() if 'demographic_parity_difference' in k and not np.isnan(v)]
        eod_values = [v for k, v in metrics.items() if 'equalized_odds_difference' in k and not np.isnan(v)]
        
        if dpd_values:
            metrics['avg_demographic_parity_difference'] = np.mean(np.abs(dpd_values))
        if eod_values:
            metrics['avg_equalized_odds_difference'] = np.mean(np.abs(eod_values))
        
        # Overall fairness violation (max absolute value)
        if dpd_values or eod_values:
            all_values = dpd_values + eod_values
            metrics['max_fairness_violation'] = np.max(np.abs(all_values))
        
        return metrics
    
    def evaluate_aif360(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: pd.DataFrame,
        sensitive_features: List[str],
        favorable_label: int = 1,
        unfavorable_label: int = 0
    ) -> Dict[str, Any]:
        """
        Compute fairness metrics using AIF360.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Feature DataFrame (needed for AIF360 dataset creation)
            sensitive_features: List of sensitive attribute column names
            favorable_label: Favorable class label
            unfavorable_label: Unfavorable class label
        
        Returns:
            Dictionary of AIF360 fairness metrics
        """
        if not self.aif360_available:
            return {'aif360_available': False}
        
        metrics = {}
        
        try:
            # Create AIF360 datasets
            # Combine features with labels
            df_true = X.copy()
            df_true['label'] = y_true
            
            df_pred = X.copy()
            df_pred['label'] = y_pred
            
            # Create BinaryLabelDataset objects
            dataset_true = BinaryLabelDataset(
                df=df_true,
                label_names=['label'],
                protected_attribute_names=sensitive_features,
                favorable_label=favorable_label,
                unfavorable_label=unfavorable_label
            )
            
            dataset_pred = BinaryLabelDataset(
                df=df_pred,
                label_names=['label'],
                protected_attribute_names=sensitive_features,
                favorable_label=favorable_label,
                unfavorable_label=unfavorable_label
            )
            
            # Compute metrics for each sensitive attribute
            for sf in sensitive_features:
                # Get privileged and unprivileged groups
                unique_values = X[sf].unique()
                if len(unique_values) == 2:
                    # Binary sensitive attribute
                    privileged_groups = [{sf: unique_values[0]}]
                    unprivileged_groups = [{sf: unique_values[1]}]
                    
                    # Create ClassificationMetric
                    cm = ClassificationMetric(
                        dataset_true,
                        dataset_pred,
                        unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups
                    )
                    
                    # Compute various fairness metrics
                    metrics[f'{sf}_aif360_statistical_parity_difference'] = cm.statistical_parity_difference()
                    metrics[f'{sf}_aif360_disparate_impact'] = cm.disparate_impact()
                    metrics[f'{sf}_aif360_equal_opportunity_difference'] = cm.equal_opportunity_difference()
                    metrics[f'{sf}_aif360_average_odds_difference'] = cm.average_odds_difference()
                    metrics[f'{sf}_aif360_theil_index'] = cm.theil_index()
                else:
                    # Multi-valued sensitive attribute - skip for now
                    metrics[f'{sf}_aif360_note'] = 'Multi-valued attribute - metrics not computed'
        
        except Exception as e:
            metrics['aif360_error'] = str(e)
        
        return metrics
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: Union[pd.DataFrame, pd.Series],
        X: Optional[pd.DataFrame] = None,
        y_proba: Optional[np.ndarray] = None,
        use_aif360: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive fairness metrics using both Fairlearn and AIF360.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: DataFrame or Series of sensitive attributes
            X: Optional feature DataFrame (required for AIF360)
            y_proba: Optional prediction probabilities
            use_aif360: Whether to compute AIF360 metrics
        
        Returns:
            Dictionary of all fairness metrics
        """
        # Ensure sensitive_features is a DataFrame
        if isinstance(sensitive_features, pd.Series):
            sensitive_features = sensitive_features.to_frame()
        
        # Compute Fairlearn metrics
        fairlearn_metrics = self.evaluate_fairlearn(
            y_true, y_pred, sensitive_features, y_proba
        )
        
        # Compute AIF360 metrics if requested and available
        aif360_metrics = {}
        if use_aif360 and self.aif360_available and X is not None:
            aif360_metrics = self.evaluate_aif360(
                y_true, y_pred, X, list(sensitive_features.columns)
            )
        
        # Combine metrics
        all_metrics = {**fairlearn_metrics, **aif360_metrics}
        
        self.metrics = all_metrics
        return all_metrics
    
    def check_fairness_thresholds(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        dpd_threshold: float = 0.05,
        eod_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check if fairness metrics meet specified thresholds.
        
        Args:
            metrics: Optional metrics dict (uses self.metrics if None)
            dpd_threshold: Threshold for Demographic Parity Difference
            eod_threshold: Threshold for Equalized Odds Difference
        
        Returns:
            Dictionary with pass/fail status for each metric
        """
        if metrics is None:
            metrics = self.metrics
        
        results = {}
        
        # Check DPD thresholds
        for key, value in metrics.items():
            if 'demographic_parity_difference' in key and not np.isnan(value):
                results[f'{key}_pass'] = abs(value) <= dpd_threshold
        
        # Check EOD thresholds
        for key, value in metrics.items():
            if 'equalized_odds_difference' in key and not np.isnan(value):
                results[f'{key}_pass'] = abs(value) <= eod_threshold
        
        # Overall fairness pass
        if results:
            results['overall_fairness_pass'] = all(results.values())
        
        return results
    
    def print_metrics(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        show_thresholds: bool = True,
        dpd_threshold: float = 0.05,
        eod_threshold: float = 0.05
    ) -> None:
        """
        Print fairness metrics in a formatted way.
        
        Args:
            metrics: Optional metrics dict (uses self.metrics if None)
            show_thresholds: Whether to show threshold pass/fail
            dpd_threshold: Threshold for DPD
            eod_threshold: Threshold for EOD
        """
        if metrics is None:
            metrics = self.metrics
        
        if not metrics:
            print("No metrics available")
            return
        
        print("\n" + "="*70)
        print("Fairness Metrics")
        print("="*70)
        
        # Group metrics by sensitive attribute
        sensitive_attrs = set()
        for key in metrics.keys():
            if '_' in key:
                attr = key.split('_')[0]
                if attr not in ['avg', 'max', 'overall', 'aif360']:
                    sensitive_attrs.add(attr)
        
        # Print metrics for each sensitive attribute
        for attr in sorted(sensitive_attrs):
            print(f"\n{attr.upper()} Fairness Metrics:")
            
            # Fairlearn metrics
            dpd_key = f'{attr}_demographic_parity_difference'
            if dpd_key in metrics and not np.isnan(metrics[dpd_key]):
                value = metrics[dpd_key]
                status = "✓" if abs(value) <= dpd_threshold else "✗"
                print(f"  Demographic Parity Diff:  {value:>7.4f} {status if show_thresholds else ''}")
            
            eod_key = f'{attr}_equalized_odds_difference'
            if eod_key in metrics and not np.isnan(metrics[eod_key]):
                value = metrics[eod_key]
                status = "✓" if abs(value) <= eod_threshold else "✗"
                print(f"  Equalized Odds Diff:      {value:>7.4f} {status if show_thresholds else ''}")
            
            # AIF360 metrics (if available)
            spd_key = f'{attr}_aif360_statistical_parity_difference'
            if spd_key in metrics:
                print(f"  Statistical Parity Diff:  {metrics[spd_key]:>7.4f} (AIF360)")
            
            eo_key = f'{attr}_aif360_equal_opportunity_difference'
            if eo_key in metrics:
                print(f"  Equal Opportunity Diff:   {metrics[eo_key]:>7.4f} (AIF360)")
        
        # Overall metrics
        if 'avg_demographic_parity_difference' in metrics:
            print(f"\nOverall Metrics:")
            print(f"  Avg DPD:  {metrics['avg_demographic_parity_difference']:.4f}")
        if 'avg_equalized_odds_difference' in metrics:
            print(f"  Avg EOD:  {metrics['avg_equalized_odds_difference']:.4f}")
        if 'max_fairness_violation' in metrics:
            print(f"  Max Violation: {metrics['max_fairness_violation']:.4f}")
        
        if show_thresholds:
            threshold_results = self.check_fairness_thresholds(
                metrics, dpd_threshold, eod_threshold
            )
            if 'overall_fairness_pass' in threshold_results:
                status = "✓ PASS" if threshold_results['overall_fairness_pass'] else "✗ FAIL"
                print(f"\nOverall Fairness: {status}")
        
        print("="*70)


if __name__ == "__main__":
    # Test fairness evaluator
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.evaluation.classifiers import ClassifierPipeline
    
    print("="*70)
    print("Testing Fairness Evaluator")
    print("="*70)
    
    # Load German Credit dataset with sensitive features
    split_data = load_split('german_credit', seed=42, split_dir=str(project_root / "data" / "splits"))
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    # Identify sensitive features (sex and race)
    sensitive_cols = [col for col in X_test.columns if 'sex' in col.lower() or 'race' in col.lower()]
    print(f"\nSensitive features found: {sensitive_cols}")
    
    if not sensitive_cols:
        print("Warning: No sensitive features found. Creating dummy sensitive feature for testing.")
        # Create dummy binary sensitive feature
        X_test['sex_Male'] = np.random.randint(0, 2, size=len(X_test))
        sensitive_cols = ['sex_Male']
    
    sensitive_features = X_test[sensitive_cols]
    
    print(f"\nDataset loaded:")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Sensitive features: {sensitive_cols}")
    print(f"  Class distribution: {y_test.value_counts().to_dict()}")
    
    # Train a classifier
    print("\n" + "-"*70)
    print("Training XGBoost classifier")
    print("-"*70)
    
    pipeline = ClassifierPipeline(random_state=42)
    pipeline.train(X_train, y_train, verbose=False)
    
    # Make predictions
    predictions = pipeline.predict(X_test, classifier_name='XGBoost')
    probabilities = pipeline.predict_proba(X_test, classifier_name='XGBoost')
    
    y_pred = predictions['XGBoost']
    y_proba = probabilities['XGBoost']
    
    # Test 1: Fairlearn metrics
    print("\n" + "-"*70)
    print("Test 1: Fairlearn fairness metrics")
    print("-"*70)
    
    evaluator = FairnessEvaluator()
    fairlearn_metrics = evaluator.evaluate_fairlearn(
        y_test, y_pred, sensitive_features, y_proba
    )
    
    print("\nFairlearn Metrics:")
    for key, value in fairlearn_metrics.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.4f}")
    
    # Test 2: AIF360 metrics (if available)
    print("\n" + "-"*70)
    print("Test 2: AIF360 fairness metrics")
    print("-"*70)
    
    if evaluator.aif360_available:
        aif360_metrics = evaluator.evaluate_aif360(
            y_test, y_pred, X_test, sensitive_cols
        )
        
        print("\nAIF360 Metrics:")
        for key, value in aif360_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("AIF360 not available - skipping AIF360 metrics")
    
    # Test 3: Combined evaluation
    print("\n" + "-"*70)
    print("Test 3: Combined fairness evaluation")
    print("-"*70)
    
    all_metrics = evaluator.evaluate(
        y_test, y_pred, sensitive_features, X_test, y_proba
    )
    evaluator.print_metrics()
    
    # Test 4: Threshold checking
    print("\n" + "-"*70)
    print("Test 4: Fairness threshold checking")
    print("-"*70)
    
    threshold_results = evaluator.check_fairness_thresholds(
        dpd_threshold=0.05,
        eod_threshold=0.05
    )
    
    print("\nThreshold Results:")
    for key, value in threshold_results.items():
        status = "✓ PASS" if value else "✗ FAIL"
        print(f"  {key}: {status}")
    
    print("\n" + "="*70)
    print("✅ Fairness Evaluator Tests Passed")
    print("="*70)

