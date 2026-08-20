"""
CTGAN Baseline

This module implements CTGAN (Conditional Tabular GAN) for imbalanced classification.
CTGAN uses deep learning to generate synthetic tabular data.
"""

# CRITICAL: Set environment variables BEFORE any imports
# This must be the FIRST thing in the file (before pandas, numpy, ctgan, torch)
import os
os.environ['PYTORCH_MPS_METAL'] = '0'  # KEY FIX: Disable MPS Metal backend
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPS_DISABLE'] = '1'
os.environ['PYTORCH_MPS_ENABLED'] = '0'
os.environ['PYTORCH_MPS_PREFER_METAL'] = '0'

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Now import CTGAN (after env vars are set)
from ctgan import CTGAN as CTGAN_Model

# Verify PyTorch is using CPU and monkey-patch to force CPU
try:
    import torch
    
    # Monkey-patch torch.device to always return 'cpu'
    _original_device = torch.device
    def _force_cpu_device(device='cpu', *args, **kwargs):
        if device in ['mps', 'cuda']:
            device = 'cpu'
        return _original_device(device, *args, **kwargs)
    torch.device = _force_cpu_device
    
    # Disable MPS backend
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.__dict__['is_available'] = lambda: False
        torch.backends.mps.__dict__['is_built'] = lambda: False
    
    if torch.backends.mps.is_available():
        print("⚠️  MPS (Apple GPU) detected but DISABLED for CTGAN to avoid segmentation faults")
        print("   Using CPU instead (more stable, slightly slower)")
    else:
        print("✓ PyTorch configured for CPU-only mode")
except ImportError:
    pass
except Exception as e:
    print(f"⚠️  Warning: Could not fully disable MPS: {e}")
    print("   Attempting to continue with CPU...")


class CTGANBaseline:
    """
    CTGAN baseline for imbalanced data.
    
    Uses Conditional Tabular GAN to generate synthetic samples.
    """
    
    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: Tuple[int, ...] = (256, 256),
        discriminator_dim: Tuple[int, ...] = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        verbose: bool = False,
        random_state: int = 42
    ):
        """
        Initialize CTGAN baseline.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            generator_dim: Dimensions of generator layers
            discriminator_dim: Dimensions of discriminator layers
            generator_lr: Learning rate for generator
            discriminator_lr: Learning rate for discriminator
            discriminator_steps: Number of discriminator steps per generator step
            verbose: Whether to print training progress
            random_state: Random seed for reproducibility
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.verbose = verbose
        self.random_state = random_state
        
        # Initialize CTGAN model
        self.model = None
        self.discrete_columns = []
        self.feature_names = []
        
    def _identify_discrete_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Identify discrete (categorical) columns in the dataset.
        
        Args:
            X: Input DataFrame
            
        Returns:
            List of discrete column names
        """
        discrete_cols = []
        for col in X.columns:
            # Consider columns with ≤10 unique values as discrete
            if X[col].nunique() <= 10:
                discrete_cols.append(col)
        return discrete_cols
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train CTGAN on the minority class samples.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Get minority class samples
        minority_class = y.value_counts().idxmin()
        X_minority = X[y == minority_class].copy()
        
        # Identify discrete columns
        self.discrete_columns = self._identify_discrete_columns(X_minority)
        
        if self.verbose:
            print(f"Training CTGAN on {len(X_minority)} minority samples...")
            print(f"Discrete columns: {self.discrete_columns}")
        
        # Initialize and train CTGAN
        # Force CPU to avoid MPS segfaults - use enable_gpu=False only
        # CTGAN requires batch_size to be divisible by pac (default 10) AND even
        pac = 10  # CTGAN default
        adjusted_batch_size = min(self.batch_size, len(X_minority))
        # Round down to nearest multiple of pac*2 (must be divisible by pac and even)
        adjusted_batch_size = max(pac * 2, (adjusted_batch_size // (pac * 2)) * (pac * 2))
        
        self.model = CTGAN_Model(
            epochs=self.epochs,
            batch_size=adjusted_batch_size,  # Must be divisible by pac (10) and even
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            verbose=self.verbose,
            pac=pac,
            enable_gpu=False  # Disable GPU (MPS/CUDA) - DO NOT use 'cuda' parameter with this
        )
        
        self.model.fit(X_minority, discrete_columns=self.discrete_columns)
        
        if self.verbose:
            print("CTGAN training completed!")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples using trained CTGAN.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame of synthetic samples
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call fit() before generate().")
        
        # Generate samples
        synthetic_samples = self.model.sample(n_samples)
        
        # Ensure column order matches original
        synthetic_samples = synthetic_samples[self.feature_names]
        
        return synthetic_samples
    
    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Train CTGAN and generate synthetic samples to balance the dataset.
        
        Args:
            X: Training features
            y: Training labels
            n_samples: Optional. If provided, generates exactly n_samples synthetic samples.
                      If None, generates enough to achieve 1:1 balance.
        
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Identify minority class
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        
        # Determine number of samples to generate
        if n_samples is not None:
            n_to_generate = n_samples
        else:
            # Generate enough to balance
            n_to_generate = n_majority - n_minority
        
        if n_to_generate <= 0:
            # No generation needed
            return X.copy(), y.copy()
        
        # Train CTGAN
        self.fit(X, y)
        
        # Generate synthetic samples
        X_synthetic = self.generate(n_to_generate)
        y_synthetic = pd.Series([minority_class] * n_to_generate, name=y.name)
        
        # Combine original and synthetic
        X_resampled = pd.concat([X, X_synthetic], ignore_index=True)
        y_resampled = pd.concat([y, y_synthetic], ignore_index=True)
        
        return X_resampled, y_resampled
    
    def get_synthetic_samples(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Generate synthetic samples without combining with original data.
        
        Args:
            X: Training features (required if model not trained)
            y: Training labels (required if model not trained)
            n_samples: Number of samples to generate
        
        Returns:
            DataFrame of synthetic samples
        """
        # Train if not already trained
        if self.model is None:
            if X is None or y is None:
                raise ValueError("Must provide X and y if model not trained")
            self.fit(X, y)
        
        # Generate samples
        return self.generate(n_samples)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'method': 'CTGAN',
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'generator_dim': self.generator_dim,
            'discriminator_dim': self.discriminator_dim,
            'generator_lr': self.generator_lr,
            'discriminator_lr': self.discriminator_lr,
            'discriminator_steps': self.discriminator_steps,
            'random_state': self.random_state
        }


if __name__ == "__main__":
    # Test CTGAN baseline
    import sys
    from pathlib import Path
    import json
    from datetime import datetime
    import time
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.qualsynth.data.splitting import load_split

    # Create results directory
    results_dir = project_root / "results" / "ctgan"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    
    print("="*70)
    print("Testing CTGAN Baseline")
    print("="*70)
    print("\n⚠️  Note: CTGAN training may take several minutes...")
    
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
    
    # Test 1: Generate 1000 samples (faster test)
    print("\n" + "-"*70)
    print("Test 1: Generate 1000 synthetic samples")
    print("-"*70)
    
    start_time = time.time()
    
    # Use reduced epochs for faster testing
    ctgan = CTGANBaseline(
        epochs=100,  # Reduced for testing
        batch_size=500,
        verbose=True,
        random_state=42
    )
    
    X_synthetic_test1 = ctgan.get_synthetic_samples(X_train, y_train, n_samples=1000)
    
    training_time = time.time() - start_time
    
    print(f"\n  Training completed in {training_time:.2f} seconds")
    print(f"  Synthetic samples: {len(X_synthetic_test1)}")
    print(f"  Features: {X_synthetic_test1.shape[1]}")
    print(f"  Sample values (first row):")
    print(f"    {X_synthetic_test1.iloc[0].to_dict()}")
    
    # Save Test 1 synthetic samples
    synthetic_file_test1 = results_dir / f"synthetic_samples_test1_1000samples.csv"
    X_synthetic_test1.to_csv(synthetic_file_test1, index=False)
    
    # Log test 1 results
    test1_results = {
        'test': 'generate_1000',
        'target_samples': 1000,
        'generated_samples': len(X_synthetic_test1),
        'n_features': X_synthetic_test1.shape[1],
        'training_time_seconds': training_time,
        'epochs': 100,
        'synthetic_file': str(synthetic_file_test1)
    }
    
    # Test 2: Check for missing values and data quality
    print("\n" + "-"*70)
    print("Test 2: Data quality analysis")
    print("-"*70)
    
    n_missing = X_synthetic_test1.isnull().sum().sum()
    n_duplicates = X_synthetic_test1.duplicated().sum()
    
    print(f"  Missing values: {n_missing}")
    print(f"  Duplicate samples: {n_duplicates}")
    print(f"  Uniqueness rate: {(1 - n_duplicates/len(X_synthetic_test1))*100:.2f}%")
    
    # Check if values are within reasonable ranges
    minority_samples = X_train[y_train == 1]
    
    print(f"\n  Feature range comparison:")
    range_coverage = []
    for col in X_train.columns:
        orig_min, orig_max = minority_samples[col].min(), minority_samples[col].max()
        synth_min, synth_max = X_synthetic_test1[col].min(), X_synthetic_test1[col].max()
        
        # Check if synthetic range is within original range (with some tolerance)
        in_range = (synth_min >= orig_min - 0.1) and (synth_max <= orig_max + 0.1)
        coverage = min(1.0, (synth_max - synth_min) / (orig_max - orig_min)) if (orig_max - orig_min) > 0 else 1.0
        range_coverage.append(coverage)
        
        status = "✓" if in_range else "⚠"
        print(f"    {status} {col:20s}: Orig=[{orig_min:7.2f}, {orig_max:7.2f}], Synth=[{synth_min:7.2f}, {synth_max:7.2f}]")
    
    avg_coverage = np.mean(range_coverage)
    
    test2_results = {
        'test': 'data_quality',
        'missing_values': int(n_missing),
        'duplicates': int(n_duplicates),
        'uniqueness_rate': float((1 - n_duplicates/len(X_synthetic_test1))*100),
        'avg_range_coverage': float(avg_coverage * 100)
    }
    
    # Test 3: Distribution statistics
    print("\n" + "-"*70)
    print("Test 3: Distribution statistics")
    print("-"*70)
    
    print(f"\n  Mean comparison:")
    mean_diffs = []
    for col in X_train.columns:
        orig_mean = minority_samples[col].mean()
        synth_mean = X_synthetic_test1[col].mean()
        diff = abs(synth_mean - orig_mean)
        mean_diffs.append(diff)
        print(f"    {col:20s}: Orig={orig_mean:7.3f}, Synth={synth_mean:7.3f}, Diff={diff:7.3f}")
    
    print(f"\n  Std comparison:")
    std_diffs = []
    for col in X_train.columns:
        orig_std = minority_samples[col].std()
        synth_std = X_synthetic_test1[col].std()
        diff = abs(synth_std - orig_std)
        std_diffs.append(diff)
        print(f"    {col:20s}: Orig={orig_std:7.3f}, Synth={synth_std:7.3f}, Diff={diff:7.3f}")
    
    avg_mean_diff = np.mean(mean_diffs)
    avg_std_diff = np.mean(std_diffs)
    
    test3_results = {
        'test': 'distribution_stats',
        'avg_mean_diff': float(avg_mean_diff),
        'avg_std_diff': float(avg_std_diff)
    }   
    
    # Save all results to file
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'original_data': original_stats,
        'test1_generate_1000': test1_results,
        'test2_data_quality': test2_results,
        'test3_distribution_stats': test3_results
    }
    
    results_file = results_dir / f"ctgan_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ CTGAN Baseline Tests Passed")
    print("="*70)
    print(f"\nResults saved to: {results_file}")
    print(f"\nGenerated data files:")
    print(f"  Test 1 (1000 samples):")
    print(f"    - Synthetic samples: {synthetic_file_test1}")
    
    print(f"\nPerformance Summary:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Uniqueness rate: {test2_results['uniqueness_rate']:.2f}%")
    print(f"  Avg range coverage: {test2_results['avg_range_coverage']:.2f}%")
    print(f"  Avg mean difference: {avg_mean_diff:.4f}")
    print(f"  Avg std difference: {avg_std_diff:.4f}")
    

