"""
SMOTE (Synthetic Minority Over-sampling Technique) baseline.

Uses imbalanced-learn's SMOTE implementation with consistent interface.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from imblearn.over_sampling import SMOTE as IMBLEARN_SMOTE


class SMOTEBaseline:
    """
    SMOTE baseline for comparison with Qualsynth.
    
    SMOTE generates synthetic samples by interpolating between minority class
    samples and their k-nearest neighbors.
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = 'auto',
        random_state: int = 42
    ):
        """
        Initialize SMOTE baseline.
        
        Args:
            k_neighbors: Number of nearest neighbors for interpolation
            sampling_strategy: Sampling strategy ('auto', 'minority', or ratio)
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = None
        
    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using SMOTE.
        
        Args:
            X: Training features
            y: Training labels
            n_samples: Number of synthetic samples to generate (optional)
                      If None, uses sampling_strategy
            
        Returns:
            Tuple of (resampled features, resampled labels)
        """
        # Determine sampling strategy
        if n_samples is not None:
            # Calculate ratio to achieve desired number of samples
            minority_count = (y == 1).sum()
            majority_count = (y == 0).sum()
            target_minority = minority_count + n_samples
            sampling_ratio = target_minority / majority_count
            strategy = {1: target_minority}
        else:
            strategy = self.sampling_strategy
        
        # Initialize SMOTE
        self.smote = IMBLEARN_SMOTE(
            k_neighbors=self.k_neighbors,
            sampling_strategy=strategy,
            random_state=self.random_state
        )
        
        # Store original size for tracking synthetic samples
        self.n_original = len(X)
        
        # Fit and resample
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        # Store resampled data
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        
        return X_resampled, y_resampled
    
    def get_synthetic_samples(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate only synthetic samples (without original data).
        
        Args:
            X: Training features (optional if fit_resample was already called)
            y: Training labels (optional if fit_resample was already called)
            n_samples: Number of synthetic samples to generate (optional if fit_resample was already called)
            
        Returns:
            DataFrame of synthetic samples
        """
        # If X and y are provided, call fit_resample first
        if X is not None and y is not None:
            self.fit_resample(X, y, n_samples)
        
        # Check if fit_resample was called
        if not hasattr(self, 'X_resampled') or self.X_resampled is None:
            raise ValueError("Must call fit_resample() first or provide X and y")
        
        # Extract only synthetic samples (those beyond original data)
        X_synthetic = self.X_resampled.iloc[self.n_original:]
        
        return X_synthetic
    
    def get_params(self) -> dict:
        """Get hyperparameters."""
        return {
            'method': 'SMOTE',
            'k_neighbors': self.k_neighbors,
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state
        }


if __name__ == "__main__":
    # Test SMOTE baseline
    import sys
    from pathlib import Path
    import json
    from datetime import datetime
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split
    
    # Create results directory
    results_dir = project_root / "results" / "smote"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Testing SMOTE Baseline")
    print("="*70)
    
    # Load German Credit dataset
    split_data = load_split('german_credit', seed=42, split_dir=str(project_root / "data" / "splits"))
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    print(f"\nOriginal training data:")
    print(f"  Total samples: {len(X_train)}")
    print(f"  Class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")
    
    # Log original data stats
    original_stats = {
        'dataset': 'german_credit',
        'split': 'train',
        'total_samples': len(X_train),
        'class_0_count': int((y_train == 0).sum()),
        'class_1_count': int((y_train == 1).sum()),
        'imbalance_ratio': float((y_train == 0).sum() / (y_train == 1).sum())
    }
    
    # Test 1: Auto balancing
    print("\n" + "-"*70)
    print("Test 1: Auto balancing (1:1 ratio)")
    print("-"*70)
    
    smote = SMOTEBaseline(k_neighbors=5, sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  Resampled data:")
    print(f"    Total samples: {len(X_resampled)}")
    print(f"    Class distribution: {y_resampled.value_counts().to_dict()}")
    print(f"    Imbalance ratio: {(y_resampled == 0).sum() / (y_resampled == 1).sum():.2f}:1")
    print(f"    Synthetic samples generated: {len(X_resampled) - len(X_train)}")
    
    # Save Test 1 data
    X_synthetic_test1 = smote.get_synthetic_samples()
    y_synthetic_test1 = y_resampled.loc[X_synthetic_test1.index]
    
    # Save synthetic samples
    synthetic_file_test1 = results_dir / f"synthetic_samples_test1_auto_balancing.csv"
    X_synthetic_test1.to_csv(synthetic_file_test1, index=False)
    
    # Save full resampled dataset
    resampled_file_test1 = results_dir / f"resampled_full_test1_auto_balancing.csv"
    X_resampled_with_labels = X_resampled.copy()
    X_resampled_with_labels['target'] = y_resampled
    X_resampled_with_labels.to_csv(resampled_file_test1, index=False)
    
    # Log test 1 results
    test1_results = {
        'test': 'auto_balancing',
        'k_neighbors': 5,
        'total_samples': len(X_resampled),
        'class_0_count': int((y_resampled == 0).sum()),
        'class_1_count': int((y_resampled == 1).sum()),
        'imbalance_ratio': float((y_resampled == 0).sum() / (y_resampled == 1).sum()),
        'synthetic_generated': len(X_resampled) - len(X_train),
        'synthetic_file': str(synthetic_file_test1),
        'resampled_file': str(resampled_file_test1)
    }
    
    # Test 2: Generate specific number of samples
    print("\n" + "-"*70)
    print("Test 2: Generate 1000 synthetic samples")
    print("-"*70)
    
    smote_test2 = SMOTEBaseline(k_neighbors=5, random_state=42)
    X_synthetic_test2 = smote_test2.get_synthetic_samples(X_train, y_train, n_samples=1000)
    
    print(f"  Synthetic samples: {len(X_synthetic_test2)}")
    print(f"  Features: {X_synthetic_test2.shape[1]}")
    print(f"  Sample values (first row):")
    print(f"    {X_synthetic_test2.iloc[0].to_dict()}")
    
    # Save Test 2 synthetic samples
    synthetic_file_test2 = results_dir / f"synthetic_samples_test2_1000samples.csv"
    X_synthetic_test2.to_csv(synthetic_file_test2, index=False)
    
    # Log test 2 results
    test2_results = {
        'test': 'specific_count',
        'k_neighbors': 5,
        'target_samples': 1000,
        'generated_samples': len(X_synthetic_test2),
        'n_features': X_synthetic_test2.shape[1],
        'synthetic_file': str(synthetic_file_test2)
    }
    
    # Test 3: Different k values
    print("\n" + "-"*70)
    print("Test 3: Different k_neighbors values")
    print("-"*70)
    
    test3_results = []
    for k in [3, 5, 10]:
        smote_test3 = SMOTEBaseline(k_neighbors=k, random_state=42)
        X_synthetic_test3 = smote_test3.get_synthetic_samples(X_train, y_train, n_samples=100)
        print(f"  k={k}: Generated {len(X_synthetic_test3)} samples")
        
        # Save Test 3 synthetic samples for each k
        synthetic_file_test3 = results_dir / f"synthetic_samples_test3_k{k}.csv"
        X_synthetic_test3.to_csv(synthetic_file_test3, index=False)
        
        test3_results.append({
            'k_neighbors': k,
            'target_samples': 100,
            'generated_samples': len(X_synthetic_test3),
            'synthetic_file': str(synthetic_file_test3)
        })
    
    
    # Save all results to file
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'original_data': original_stats,
        'test1_auto_balancing': test1_results,
        'test2_specific_count': test2_results,
        'test3_different_k': test3_results
    }
    
    results_file = results_dir / f"smote_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ SMOTE Baseline Tests Passed")
    print("="*70)
    print(f"\nResults saved to: {results_file}")
    print(f"\nGenerated data files:")
    print(f"  Test 1 (Auto-balancing):")
    print(f"    - Synthetic samples: {synthetic_file_test1}")
    print(f"    - Full resampled: {resampled_file_test1}")
    print(f"  Test 2 (1000 samples):")
    print(f"    - Synthetic samples: {synthetic_file_test2}")
    print(f"  Test 3 (Different k values):")
    for result in test3_results:
        print(f"    - k={result['k_neighbors']}: {result['synthetic_file']}")
    

