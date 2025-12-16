"""
Data preprocessing utilities for Qualsynth experiments.

Handles:
- Missing value imputation
- Categorical encoding
- Numerical normalization
- Protected attribute identification
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional


class DatasetPreprocessor:
    """Preprocess datasets for Qualsynth experiments."""
    
    def __init__(self, dataset_name: str):
        """
        Initialize preprocessor for a specific dataset.
        
        Args:
            dataset_name: Name of dataset ('german_credit', etc.)
        """
        self.dataset_name = dataset_name
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.protected_attributes = []
        self.target_column = None
        
    def load_and_preprocess(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess dataset.
        
        Args:
            filepath: Path to raw CSV file
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Load data
        df = pd.read_csv(filepath)
        
        # Dataset-specific preprocessing
        if self.dataset_name == 'german_credit':
            df, target = self._preprocess_german_credit(df)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return df, target
    
    def _preprocess_german_credit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess German Credit dataset."""
        target = (df['class'] == 2).astype(int)
        df = df.drop('class', axis=1)
        
        # Protected attributes
        self.protected_attributes = ['age', 'personal_status']
        
        # Identify feature types
        self.categorical_features = [
            'checking_status', 'credit_history', 'purpose', 'savings_status',
            'employment', 'personal_status', 'other_parties', 'property_magnitude',
            'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
        ]
        self.numerical_features = [
            'duration', 'credit_amount', 'installment_rate', 'residence_since',
            'age', 'existing_credits', 'num_dependents'
        ]
        
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Normalize numerical features
        if self.numerical_features:
            numerical_cols = [col for col in self.numerical_features if col in df.columns]
            if numerical_cols:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        self.feature_names = df.columns.tolist()
        self.target_column = 'credit_risk'
        
        return df, target
    
    def get_feature_info(self) -> Dict:
        """Get information about features."""
        return {
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'protected_attributes': self.protected_attributes,
            'target_column': self.target_column,
            'n_features': len(self.feature_names)
        }


def load_dataset(dataset_name: str, data_dir: str = "data/raw", return_preprocessor: bool = False) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Load and preprocess a dataset.
    
    Args:
        dataset_name: Name of dataset ('german_credit', etc.)
        data_dir: Directory containing raw data files
        return_preprocessor: If True, also return the preprocessor object
        
    Returns:
        Tuple of (features DataFrame, target Series, feature info dict)
        If return_preprocessor=True: (features, target, feature_info, preprocessor)
    """
    # Map dataset names to files
    file_mapping = {
        'german_credit': 'german_credit.csv'
    }
    
    if dataset_name not in file_mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    filepath = f"{data_dir}/{file_mapping[dataset_name]}"
    
    # Preprocess
    preprocessor = DatasetPreprocessor(dataset_name)
    X, y = preprocessor.load_and_preprocess(filepath)
    feature_info = preprocessor.get_feature_info()
    
    if return_preprocessor:
        return X, y, feature_info, preprocessor
    return X, y, feature_info


if __name__ == "__main__":
    # Test preprocessing
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    data_dir = project_root / "data" / "raw"
    
    for dataset in ['german_credit']:
        print(f"\n{'='*70}")
        print(f"Testing {dataset} preprocessing")
        print('='*70)
        
        X, y, info = load_dataset(dataset, str(data_dir))
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}")
        print(f"\nFeature info:")
        print(f"  - Total features: {info['n_features']}")
        print(f"  - Categorical: {len(info['categorical_features'])}")
        print(f"  - Numerical: {len(info['numerical_features'])}")
        print(f"  - Protected attributes: {info['protected_attributes']}")

