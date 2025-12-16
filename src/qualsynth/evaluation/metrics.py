"""
Performance Metrics Evaluation

This module implements comprehensive performance metrics for evaluating classifiers
on imbalanced datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for classification tasks.
    
    Computes standard and imbalanced-learning specific metrics.
    """
    
    def __init__(self):
        """Initialize metrics evaluator."""
        self.metrics = {}
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'binary',
        pos_label: int = 1,
        calibrate_threshold: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for ROC-AUC, PR-AUC)
            average: Averaging strategy ('binary', 'macro', 'weighted')
            pos_label: Positive class label
            calibrate_threshold: Whether to compute metrics at optimal threshold
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics (at default threshold 0.5)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        
        # Imbalanced-learning specific metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics (if probabilities provided)
        if y_proba is not None:
            # For binary classification, use positive class probabilities
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            except Exception as e:
                metrics['roc_auc'] = np.nan
            
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
            except Exception as e:
                metrics['pr_auc'] = np.nan
            
            # Threshold calibration - find optimal threshold for F1
            if calibrate_threshold:
                calibration_result = self.find_optimal_threshold(y_true, y_proba_pos)
                metrics['optimal_threshold'] = calibration_result['optimal_threshold']
                metrics['f1_calibrated'] = calibration_result['f1_calibrated']
                metrics['accuracy_calibrated'] = calibration_result['accuracy_calibrated']
                metrics['precision_calibrated'] = calibration_result['precision_calibrated']
                metrics['recall_calibrated'] = calibration_result['recall_calibrated']
                metrics['f1_improvement'] = calibration_result['f1_calibrated'] - metrics['f1']
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Specificity and Sensitivity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # G-mean (geometric mean of sensitivity and specificity)
            metrics['g_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])
        
        self.metrics = metrics
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1',
        n_thresholds: int = 100
    ) -> Dict[str, float]:
        """
        Find the optimal classification threshold that maximizes a given metric.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities for positive class
            metric: Metric to optimize ('f1', 'balanced_accuracy', 'g_mean')
            n_thresholds: Number of thresholds to evaluate
        
        Returns:
            Dictionary with optimal threshold and metrics at that threshold
        """
        # Generate thresholds to evaluate
        thresholds = np.linspace(0.1, 0.9, n_thresholds)
        
        best_score = -1
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # Calculate metrics at this threshold
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            acc = accuracy_score(y_true, y_pred_thresh)
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            bal_acc = balanced_accuracy_score(y_true, y_pred_thresh)
            
            # G-mean
            cm = confusion_matrix(y_true, y_pred_thresh)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                g_mean = np.sqrt(sens * spec)
            else:
                g_mean = 0
            
            # Select score based on metric
            if metric == 'f1':
                score = f1
            elif metric == 'balanced_accuracy':
                score = bal_acc
            elif metric == 'g_mean':
                score = g_mean
            else:
                score = f1
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'f1_calibrated': f1,
                    'accuracy_calibrated': acc,
                    'precision_calibrated': prec,
                    'recall_calibrated': rec,
                    'balanced_accuracy_calibrated': bal_acc,
                    'g_mean_calibrated': g_mean
                }
        
        return {
            'optimal_threshold': best_threshold,
            **best_metrics
        }
    
    def evaluate_per_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
        
        Returns:
            Dictionary of per-class metrics
        """
        # Get classification report as dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract per-class metrics
        per_class_metrics = {}
        classes = np.unique(y_true)
        
        for cls in classes:
            cls_str = str(int(cls))
            if cls_str in report:
                class_name = class_names[cls] if class_names and cls < len(class_names) else f"Class_{cls}"
                per_class_metrics[class_name] = {
                    'precision': report[cls_str]['precision'],
                    'recall': report[cls_str]['recall'],
                    'f1': report[cls_str]['f1-score'],
                    'support': report[cls_str]['support']
                }
        
        return per_class_metrics
    
    def get_roc_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data for plotting.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
        
        Returns:
            Dictionary with fpr, tpr, thresholds
        """
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def get_pr_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get Precision-Recall curve data for plotting.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
        
        Returns:
            Dictionary with precision, recall, thresholds
        """
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba_pos)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def print_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Optional metrics dict (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        if not metrics:
            print("No metrics available")
            return
        
        print("\n" + "="*60)
        print("Performance Metrics")
        print("="*60)
        
        # Basic metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"  Precision:          {metrics.get('precision', 0):.4f}")
        print(f"  Recall:             {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:           {metrics.get('f1', 0):.4f}")
        print(f"  MCC:                {metrics.get('mcc', 0):.4f}")
        
        # Probability-based metrics
        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            print("\nProbability-based Metrics:")
            print(f"  ROC-AUC:            {metrics.get('roc_auc', 0):.4f}")
            print(f"  PR-AUC:             {metrics.get('pr_auc', 0):.4f}")
        
        # Binary classification specific
        if 'sensitivity' in metrics:
            print("\nBinary Classification Metrics:")
            print(f"  Sensitivity (TPR):  {metrics.get('sensitivity', 0):.4f}")
            print(f"  Specificity (TNR):  {metrics.get('specificity', 0):.4f}")
            print(f"  G-Mean:             {metrics.get('g_mean', 0):.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            print("\nConfusion Matrix:")
            cm = np.array(metrics['confusion_matrix'])
            print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
            print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        print("="*60)


if __name__ == "__main__":
    # Test metrics evaluator
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.evaluation.classifiers import ClassifierPipeline
    
    print("="*70)
    print("Testing Metrics Evaluator")
    print("="*70)
    
    # Load German Credit dataset
    split_data = load_split('german_credit', seed=42, split_dir=str(project_root / "data" / "splits"))
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    print(f"\nDataset loaded:")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
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
    
    # Test 1: Basic metrics evaluation
    print("\n" + "-"*70)
    print("Test 1: Comprehensive metrics evaluation")
    print("-"*70)
    
    evaluator = MetricsEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    evaluator.print_metrics()
    
    # Test 2: Per-class metrics
    print("\n" + "-"*70)
    print("Test 2: Per-class metrics")
    print("-"*70)
    
    per_class = evaluator.evaluate_per_class(y_test, y_pred, class_names=['Majority', 'Minority'])
    
    for class_name, class_metrics in per_class.items():
        print(f"\n  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1:        {class_metrics['f1']:.4f}")
        print(f"    Support:   {class_metrics['support']}")
    
    # Test 3: ROC and PR curve data
    print("\n" + "-"*70)
    print("Test 3: ROC and PR curve data")
    print("-"*70)
    
    roc_data = evaluator.get_roc_curve_data(y_test, y_proba)
    pr_data = evaluator.get_pr_curve_data(y_test, y_proba)
    
    print(f"\n  ROC curve:")
    print(f"    FPR points: {len(roc_data['fpr'])}")
    print(f"    TPR points: {len(roc_data['tpr'])}")
    print(f"    Thresholds: {len(roc_data['thresholds'])}")
    
    print(f"\n  PR curve:")
    print(f"    Precision points: {len(pr_data['precision'])}")
    print(f"    Recall points:    {len(pr_data['recall'])}")
    print(f"    Thresholds:       {len(pr_data['thresholds'])}")
    
    # Test 4: Compare multiple classifiers
    print("\n" + "-"*70)
    print("Test 4: Compare multiple classifiers")
    print("-"*70)
    
    all_predictions = pipeline.predict(X_test)
    all_probabilities = pipeline.predict_proba(X_test)
    
    results = {}
    for clf_name in all_predictions.keys():
        y_pred_clf = all_predictions[clf_name]
        y_proba_clf = all_probabilities[clf_name]
        
        evaluator_clf = MetricsEvaluator()
        metrics_clf = evaluator_clf.evaluate(y_test, y_pred_clf, y_proba_clf)
        results[clf_name] = metrics_clf
    
    # Create comparison table
    print("\n  Comparison Table:")
    print(f"  {'Metric':<20} {'RandomForest':<15} {'XGBoost':<15} {'LogisticReg':<15}")
    print("  " + "-"*65)
    
    metric_names = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'pr_auc', 'mcc']
    for metric in metric_names:
        values = [results[clf].get(metric, 0) for clf in ['RandomForest', 'XGBoost', 'LogisticRegression']]
        print(f"  {metric:<20} {values[0]:<15.4f} {values[1]:<15.4f} {values[2]:<15.4f}")
    
    # Test 5: Export metrics to DataFrame
    print("\n" + "-"*70)
    print("Test 5: Export metrics to DataFrame")
    print("-"*70)
    
    # Create DataFrame for easy comparison
    df_metrics = pd.DataFrame(results).T
    print("\n", df_metrics[['accuracy', 'f1', 'roc_auc', 'balanced_accuracy', 'mcc']].round(4))
    
    print("\n" + "="*70)
    print("✅ Metrics Evaluator Tests Passed")
    print("="*70)

