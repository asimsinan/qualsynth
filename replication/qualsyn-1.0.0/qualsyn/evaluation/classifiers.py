"""
Classifier Training Pipeline

This module implements a unified interface for training and evaluating multiple classifiers
on imbalanced datasets with synthetic data augmentation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

try:
    from .metrics import MetricsEvaluator
    from .fairness import FairnessEvaluator
except ImportError:
    from metrics import MetricsEvaluator
    from fairness import FairnessEvaluator


class ClassifierPipeline:
    """
    Unified pipeline for training and evaluating multiple classifiers.
    
    Supports: Random Forest, XGBoost, Logistic Regression
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classifier pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.classifiers = {}
        self.trained_models = {}
        
    def get_default_classifiers(self) -> Dict[str, BaseEstimator]:
        """
        Get default classifier configurations.
        
        Returns:
            Dictionary of classifier name to initialized model
        """
        # Set n_jobs=1 to prevent joblib from spawning worker processes
        # This prevents process explosion (21+ processes)
        # Sequential processing is slower but more stable
        n_jobs = 1
        
        classifiers = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=n_jobs  # Limited to 4 workers
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Will be adjusted based on class imbalance
                random_state=self.random_state,
                n_jobs=n_jobs,  # Limited to 4 workers
                eval_metric='logloss'
            ),
            'LogisticRegression': LogisticRegression(
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=n_jobs  # Limited to 4 workers
            )
        }
        return classifiers
    
    def set_classifiers(self, classifiers: Dict[str, BaseEstimator]) -> None:
        """
        Set custom classifiers.
        
        Args:
            classifiers: Dictionary of classifier name to initialized model
        """
        self.classifiers = classifiers
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        classifiers: Optional[Dict[str, BaseEstimator]] = None,
        verbose: bool = True
    ) -> Dict[str, BaseEstimator]:
        """
        Train multiple classifiers on the training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            classifiers: Optional custom classifiers (uses defaults if None)
            verbose: Whether to print training progress
        
        Returns:
            Dictionary of trained models
        """
        if classifiers is None:
            classifiers = self.get_default_classifiers()
        
        self.classifiers = classifiers
        self.trained_models = {}
        
        # Adjust XGBoost scale_pos_weight based on class imbalance
        if 'XGBoost' in classifiers:
            class_counts = y_train.value_counts()
            if len(class_counts) == 2:
                majority_count = class_counts.max()
                minority_count = class_counts.min()
                scale_pos_weight = majority_count / minority_count
                classifiers['XGBoost'].set_params(scale_pos_weight=scale_pos_weight)
                if verbose:
                    print(f"XGBoost scale_pos_weight set to {scale_pos_weight:.2f}")
        
        if verbose:
            print(f"\nTraining {len(classifiers)} classifiers...")
            print(f"Training set: {len(X_train)} samples")
            print(f"Class distribution: {y_train.value_counts().to_dict()}")
        
        for name, clf in classifiers.items():
            if verbose:
                print(f"\n  Training {name}...")
            
            try:
                clf.fit(X_train, y_train)
                self.trained_models[name] = clf
                if verbose:
                    print(f"    ✓ {name} trained successfully")
            except Exception as e:
                if verbose:
                    print(f"    ✗ {name} training failed: {str(e)}")
                continue
        
        if verbose:
            print(f"\n✓ Training completed: {len(self.trained_models)}/{len(classifiers)} classifiers trained")
        
        return self.trained_models
    
    def predict(
        self,
        X_test: pd.DataFrame,
        classifier_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using trained classifiers.
        
        Args:
            X_test: Test features
            classifier_name: Optional specific classifier to use (uses all if None)
        
        Returns:
            Dictionary of classifier name to predictions
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Call train() first.")
        
        predictions = {}
        
        if classifier_name is not None:
            if classifier_name not in self.trained_models:
                raise ValueError(f"Classifier '{classifier_name}' not found in trained models")
            predictions[classifier_name] = self.trained_models[classifier_name].predict(X_test)
        else:
            for name, clf in self.trained_models.items():
                predictions[name] = clf.predict(X_test)
        
        return predictions
    
    def predict_proba(
        self,
        X_test: pd.DataFrame,
        classifier_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get prediction probabilities using trained classifiers.
        
        Args:
            X_test: Test features
            classifier_name: Optional specific classifier to use (uses all if None)
        
        Returns:
            Dictionary of classifier name to prediction probabilities
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Call train() first.")
        
        probabilities = {}
        
        if classifier_name is not None:
            if classifier_name not in self.trained_models:
                raise ValueError(f"Classifier '{classifier_name}' not found in trained models")
            probabilities[classifier_name] = self.trained_models[classifier_name].predict_proba(X_test)
        else:
            for name, clf in self.trained_models.items():
                probabilities[name] = clf.predict_proba(X_test)
        
        return probabilities
    
    def get_model(self, classifier_name: str) -> BaseEstimator:
        """
        Get a specific trained model.
        
        Args:
            classifier_name: Name of the classifier
        
        Returns:
            Trained model
        """
        if classifier_name not in self.trained_models:
            raise ValueError(f"Classifier '{classifier_name}' not found in trained models")
        return self.trained_models[classifier_name]
    
    def get_all_models(self) -> Dict[str, BaseEstimator]:
        """
        Get all trained models.
        
        Returns:
            Dictionary of trained models
        """
        return self.trained_models
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        classifier_names: Optional[List[str]] = None,
        sensitive_features: Optional[pd.DataFrame] = None,
        compute_fairness: bool = True,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate trained classifiers on test data with performance and fairness metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            classifier_names: Optional list of classifier names to evaluate (default: all)
            sensitive_features: Optional DataFrame of sensitive attributes for fairness
            compute_fairness: Whether to compute fairness metrics
            verbose: Whether to print evaluation progress
        
        Returns:
            Dictionary mapping classifier names to their metrics (performance + fairness)
        """
        if classifier_names is None:
            classifier_names = list(self.trained_models.keys())
        
        if verbose:
            print(f"\nEvaluating {len(classifier_names)} classifiers...")
            print(f"Test set: {len(X_test)} samples")
            print(f"Class distribution: {y_test.value_counts().to_dict()}")
            if sensitive_features is not None and compute_fairness:
                print(f"Sensitive features: {list(sensitive_features.columns)}")
        
        results = {}
        metrics_evaluator = MetricsEvaluator()
        fairness_evaluator = FairnessEvaluator() if compute_fairness else None
        
        for name in classifier_names:
            if name not in self.trained_models:
                if verbose:
                    print(f"  ⚠ Skipping {name} (not trained)")
                continue
            
            if verbose:
                print(f"\n  Evaluating {name}...")
            
            # Get predictions
            y_pred = self.predict(X_test, classifier_name=name)[name]
            y_proba = self.predict_proba(X_test, classifier_name=name)[name]
            
            # Compute performance metrics
            metrics = metrics_evaluator.evaluate(y_test, y_pred, y_proba)
            
            # Compute fairness metrics if requested
            if compute_fairness and sensitive_features is not None and fairness_evaluator is not None:
                fairness_metrics = fairness_evaluator.evaluate(
                    y_test, y_pred, sensitive_features, X_test, y_proba
                )
                metrics.update(fairness_metrics)
            
            results[name] = metrics
            
            # Print summary
            if verbose:
                print(f"    F1:         {metrics['f1']:.4f}")
                print(f"    ROC-AUC:    {metrics['roc_auc']:.4f}")
                print(f"    Bal. Acc:   {metrics['balanced_accuracy']:.4f}")
                if compute_fairness and 'avg_demographic_parity_difference' in metrics:
                    print(f"    Avg DPD:    {metrics['avg_demographic_parity_difference']:.4f}")
                    print(f"    Avg EOD:    {metrics['avg_equalized_odds_difference']:.4f}")
        
        if verbose:
            print(f"\n✓ Evaluation completed: {len(results)} classifiers")
        return results


if __name__ == "__main__":
    # Test classifier pipeline
    import sys
    from pathlib import Path
    import time
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    
    print("="*70)
    print("Testing Classifier Pipeline")
    print("="*70)
    
    # Load German Credit dataset
    split_data = load_split('german_credit', seed=42, split_dir=str(project_root / "data" / "splits"))
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    print(f"\nDataset loaded:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Initialize pipeline
    pipeline = ClassifierPipeline(random_state=42)
    
    # Test 1: Train with default classifiers
    print("\n" + "-"*70)
    print("Test 1: Training with default classifiers")
    print("-"*70)
    
    start_time = time.time()
    trained_models = pipeline.train(X_train, y_train, verbose=True)
    training_time = time.time() - start_time
    
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print(f"Trained models: {list(trained_models.keys())}")
    
    # Test 2: Make predictions
    print("\n" + "-"*70)
    print("Test 2: Making predictions")
    print("-"*70)
    
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    
    for name in predictions:
        y_pred = predictions[name]
        y_proba = probabilities[name]
        
        # Calculate basic accuracy
        accuracy = (y_pred == y_test).sum() / len(y_test)
        
        print(f"\n  {name}:")
        print(f"    Predictions shape: {y_pred.shape}")
        print(f"    Probabilities shape: {y_proba.shape}")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Predicted class distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    
    # Test 3: Get specific model
    print("\n" + "-"*70)
    print("Test 3: Retrieving specific models")
    print("-"*70)
    
    rf_model = pipeline.get_model('RandomForest')
    print(f"\n  RandomForest model: {type(rf_model).__name__}")
    print(f"  Number of estimators: {rf_model.n_estimators}")
    print(f"  Feature importances shape: {rf_model.feature_importances_.shape}")
    
    xgb_model = pipeline.get_model('XGBoost')
    print(f"\n  XGBoost model: {type(xgb_model).__name__}")
    print(f"  Number of estimators: {xgb_model.n_estimators}")
    print(f"  Feature importances shape: {xgb_model.feature_importances_.shape}")
    
    lr_model = pipeline.get_model('LogisticRegression')
    print(f"\n  LogisticRegression model: {type(lr_model).__name__}")
    print(f"  Coefficients shape: {lr_model.coef_.shape}")
    
    # Test 4: Custom classifier configuration
    print("\n" + "-"*70)
    print("Test 4: Custom classifier configuration")
    print("-"*70)
    
    custom_classifiers = {
        'RF_Small': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ),
        'XGB_Fast': XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    }
    
    pipeline_custom = ClassifierPipeline(random_state=42)
    trained_custom = pipeline_custom.train(X_train, y_train, classifiers=custom_classifiers, verbose=True)
    
    predictions_custom = pipeline_custom.predict(X_test)
    for name, y_pred in predictions_custom.items():
        accuracy = (y_pred == y_test).sum() / len(y_test)
        print(f"\n  {name} accuracy: {accuracy:.4f}")
    
    print("\n" + "="*70)
    print("✅ Classifier Pipeline Tests Passed")
    print("="*70)

