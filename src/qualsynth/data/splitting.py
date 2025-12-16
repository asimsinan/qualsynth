"""
Data splitting utilities for Qualsynth experiments.

Creates stratified train/validation/test splits with multiple random seeds.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List
import pickle

try:
    from .preprocessing import load_dataset
except ImportError:
    from preprocessing import load_dataset


def create_splits(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    seeds: List[int] = [42, 123, 456],
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    output_dir: str = "data/splits",
    preprocessor = None
) -> None:
    """
    Create stratified train/validation/test splits.
    
    Args:
        X: Features DataFrame
        y: Target Series
        dataset_name: Name of dataset
        seeds: List of random seeds for multiple splits
        train_size: Proportion for training set (default 0.6)
        val_size: Proportion for validation set (default 0.2)
        test_size: Proportion for test set (default 0.2)
        output_dir: Directory to save splits
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        print(f"\nCreating split with seed={seed}...")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_size,
            stratify=y,
            random_state=seed
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            stratify=y_temp,
            random_state=seed
        )
        
        # Save splits (include preprocessor for encoding/decoding)
        split_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'seed': seed,
            'preprocessor': preprocessor
        }
        
        output_file = output_path / f"split_seed{seed}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        print(f"  Train: {len(X_train)} samples (class dist: {y_train.value_counts().to_dict()})")
        print(f"  Val:   {len(X_val)} samples (class dist: {y_val.value_counts().to_dict()})")
        print(f"  Test:  {len(X_test)} samples (class dist: {y_test.value_counts().to_dict()})")
        print(f"  Saved to: {output_file}")


def load_split(dataset_name: str, seed: int, split_dir: str = "data/splits", return_raw: bool = False, 
               include_sensitive_indicators: bool = False, dataset_config=None) -> dict:
    """
    Load a specific data split.
    
    Args:
        dataset_name: Name of dataset
        seed: Random seed used for split
        split_dir: Directory containing splits
        return_raw: If True, return raw (unencoded) data for LLM generation
        include_sensitive_indicators: If True, include binary protected group indicators
        dataset_config: Dataset configuration (required if include_sensitive_indicators=True)
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        If return_raw=True, also includes preprocessor for encoding/decoding
        If include_sensitive_indicators=True, also includes sensitive_train, sensitive_val, sensitive_test
    """
    split_file = Path(split_dir) / dataset_name / f"split_seed{seed}.pkl"
    
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'rb') as f:
        split_data = pickle.load(f)
    
    if return_raw and 'preprocessor' in split_data:
        # Decode the data back to original format for LLM
        preprocessor = split_data['preprocessor']
        split_data_raw = {}
        
        for key in ['X_train', 'X_val', 'X_test']:
            if key in split_data:
                split_data_raw[key] = decode_features(split_data[key].copy(), preprocessor)
        
        for key in ['y_train', 'y_val', 'y_test']:
            if key in split_data:
                split_data_raw[key] = split_data[key]
        
        split_data_raw['preprocessor'] = preprocessor
        
        # Add sensitive indicators if requested
        if include_sensitive_indicators and dataset_config is not None:
            preprocessor = split_data.get('preprocessor')
            for key in ['X_train', 'X_val', 'X_test']:
                if key in split_data:
                    sensitive_key = key.replace('X_', 'sensitive_')
                    split_data_raw[sensitive_key] = binarize_sensitive_features(
                        split_data[key].copy(), 
                        dataset_config,
                        preprocessor
                    )
        
        return split_data_raw
    
    # Add sensitive indicators if requested (for encoded data)
    if include_sensitive_indicators and dataset_config is not None:
        preprocessor = split_data.get('preprocessor')
        for key in ['X_train', 'X_val', 'X_test']:
            if key in split_data:
                sensitive_key = key.replace('X_', 'sensitive_')
                split_data[sensitive_key] = binarize_sensitive_features(
                    split_data[key].copy(),
                    dataset_config,
                    preprocessor
                )
    
    return split_data


def decode_features(X: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """
    Decode features back to original format.
    
    Args:
        X: Encoded features DataFrame (or already RAW - will detect and skip)
        preprocessor: DatasetPreprocessor instance
        
    Returns:
        DataFrame with original (unencoded) features
    """
    X_decoded = X.copy()
    
    # Check if data is already in RAW format
    if hasattr(preprocessor, 'label_encoders') and preprocessor.label_encoders:
        first_cat_col = list(preprocessor.label_encoders.keys())[0]
        if first_cat_col in X.columns:
            # If the column contains strings (not numeric), data is already RAW
            if X[first_cat_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[first_cat_col]):
                # Data is already in RAW format, return as-is
                return X_decoded
    
    # Inverse transform numerical features
    if hasattr(preprocessor, 'numerical_features') and preprocessor.numerical_features:
        num_cols = [c for c in preprocessor.numerical_features if c in X.columns]
        if num_cols:
            sample_col = num_cols[0]
            col_range = X[sample_col].max() - X[sample_col].min()
            col_mean = X[sample_col].mean()
            if col_range < 20 and abs(col_mean) < 5:
                X_decoded[num_cols] = preprocessor.scaler.inverse_transform(X[num_cols])
    
    # Inverse transform categorical features
    if hasattr(preprocessor, 'label_encoders'):
        for col, encoder in preprocessor.label_encoders.items():
            if col in X_decoded.columns:
                if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                    continue
                
                try:
                    encoded_vals = X[col].astype(int).values.copy()
                except (ValueError, TypeError):
                    continue
                    
                n_classes = len(encoder.classes_)
                invalid_mask = (encoded_vals < 0) | (encoded_vals >= n_classes)
                if invalid_mask.any():
                    encoded_vals[invalid_mask] = 0
                
                X_decoded[col] = encoder.inverse_transform(encoded_vals)
    
    return X_decoded


def encode_features(X: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """
    Encode features to match training data format.
    
    Args:
        X: Raw features DataFrame
        preprocessor: DatasetPreprocessor instance
        
    Returns:
        DataFrame with encoded features
    """
    X_encoded = X.copy()
    
    # Encode categorical features
    if hasattr(preprocessor, 'label_encoders'):
        for col, encoder in preprocessor.label_encoders.items():
            if col in X_encoded.columns:
                # Handle unseen categories by mapping to most common category
                def safe_encode(x, enc=encoder):
                    if pd.isna(x):
                        return enc.transform([enc.classes_[0]])[0]
                    x_str = str(x).strip()
                    if x_str in enc.classes_:
                        return enc.transform([x_str])[0]
                    else:
                        return enc.transform([enc.classes_[0]])[0]
                
                X_encoded[col] = X_encoded[col].apply(safe_encode).astype(int)
    
    # Normalize numerical features
    if hasattr(preprocessor, 'numerical_features') and preprocessor.numerical_features:
        # Ensure numerical features are numeric before scaling
        for num_col in preprocessor.numerical_features:
            if num_col in X_encoded.columns:
                X_encoded[num_col] = pd.to_numeric(X_encoded[num_col], errors='coerce')
        
        # Fill any NaN values with 0 before scaling
        X_encoded[preprocessor.numerical_features] = X_encoded[preprocessor.numerical_features].fillna(0)
        X_encoded[preprocessor.numerical_features] = preprocessor.scaler.transform(
            X_encoded[preprocessor.numerical_features]
        )
    
    return X_encoded


def binarize_sensitive_features(
    X: pd.DataFrame,
    dataset_config,
    preprocessor=None
) -> pd.DataFrame:
    """
    Create binary protected group indicators for sensitive features.
    
    Args:
        X: Features DataFrame (can be normalized or raw)
        dataset_config: Dataset configuration with sensitive_attributes
        preprocessor: Optional preprocessor to denormalize features
        
    Returns:
        DataFrame with binary protected group indicators
    """
    X_indicators = pd.DataFrame(index=X.index)
    
    # Get sensitive attributes from config
    if not hasattr(dataset_config, 'sensitive_attributes'):
        return X_indicators
    
    sensitive_attrs = dataset_config.sensitive_attributes
    if not sensitive_attrs:
        return X_indicators
    
    # Handle both list of dicts and list of strings
    if isinstance(sensitive_attrs[0], dict):
        attrs_list = sensitive_attrs
    else:
        # If it's a list of strings, we can't determine protected groups
        # Return empty DataFrame
        return X_indicators
    
    for attr in attrs_list:
        attr_name = attr['name']
        attr_type = attr.get('type', 'categorical')
        protected_group_def = attr.get('protected_group', None)
        
        if not protected_group_def or attr_name not in X.columns:
            continue
        
        # Denormalize if needed (for continuous features)
        if attr_type == 'continuous' and preprocessor is not None:
            # Check if feature is normalized (mean ~0, std ~1)
            if abs(X[attr_name].mean()) < 0.1 and abs(X[attr_name].std() - 1.0) < 0.1:
                # Feature is normalized, denormalize it
                if hasattr(preprocessor, 'scaler') and hasattr(preprocessor, 'numerical_features'):
                    if attr_name in preprocessor.numerical_features:
                        # Get the index of this feature in numerical_features
                        idx = preprocessor.numerical_features.index(attr_name)
                        # Denormalize: x_original = x_normalized * std + mean
                        mean = preprocessor.scaler.mean_[idx]
                        std = np.sqrt(preprocessor.scaler.var_[idx])
                        X_denorm = X[attr_name] * std + mean
                    else:
                        X_denorm = X[attr_name]
                else:
                    X_denorm = X[attr_name]
            else:
                X_denorm = X[attr_name]
        else:
            X_denorm = X[attr_name]
        
        # Create binary indicator based on protected group definition
        if attr_type == 'continuous':
            # Parse condition like "age >= 60"
            try:
                # Extract operator and threshold
                if '>=' in protected_group_def:
                    threshold = float(protected_group_def.split('>=')[1].strip())
                    X_indicators[f'{attr_name}_protected'] = (X_denorm >= threshold).astype(int)
                elif '<=' in protected_group_def:
                    threshold = float(protected_group_def.split('<=')[1].strip())
                    X_indicators[f'{attr_name}_protected'] = (X_denorm <= threshold).astype(int)
                elif '>' in protected_group_def:
                    threshold = float(protected_group_def.split('>')[1].strip())
                    X_indicators[f'{attr_name}_protected'] = (X_denorm > threshold).astype(int)
                elif '<' in protected_group_def:
                    threshold = float(protected_group_def.split('<')[1].strip())
                    X_indicators[f'{attr_name}_protected'] = (X_denorm < threshold).astype(int)
                else:
                    print(f"Warning: Could not parse protected group condition: {protected_group_def}")
                    continue
            except Exception as e:
                print(f"Warning: Error parsing protected group for {attr_name}: {e}")
                continue
                
        elif attr_type in ['categorical', 'binary']:
            # For categorical, check if value matches protected group
            # Handle both encoded and raw values
            protected_value = protected_group_def
            
            # If feature is encoded, we need to check the encoded value
            if preprocessor is not None and hasattr(preprocessor, 'label_encoders'):
                if attr_name in preprocessor.label_encoders:
                    encoder = preprocessor.label_encoders[attr_name]
                    # Check if protected_value is in the encoder's classes
                    if protected_value in encoder.classes_:
                        protected_encoded = encoder.transform([protected_value])[0]
                        X_indicators[f'{attr_name}_protected'] = (X[attr_name] == protected_encoded).astype(int)
                    else:
                        # Try direct comparison (might be already encoded)
                        X_indicators[f'{attr_name}_protected'] = (X[attr_name] == protected_value).astype(int)
                else:
                    # Not encoded, direct comparison
                    X_indicators[f'{attr_name}_protected'] = (X[attr_name] == protected_value).astype(int)
            else:
                # No preprocessor, direct comparison
                X_indicators[f'{attr_name}_protected'] = (X[attr_name] == protected_value).astype(int)
    
    return X_indicators


def create_splits_with_preprocessor(
    dataset_name: str,
    seeds: List[int] = [42, 123, 456],
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    data_dir: str = "data/raw",
    output_dir: str = "data/splits"
) -> None:
    """
    Create stratified train/validation/test splits with preprocessor saved.
    """
    from .preprocessing import load_dataset
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    X, y, info, preprocessor = load_dataset(dataset_name, data_dir, return_preprocessor=True)
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Numerical: {len(preprocessor.numerical_features)}")
    print(f"  Categorical: {len(preprocessor.categorical_features)}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        print(f"\n  Creating split with seed={seed}...")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_size,
            stratify=y,
            random_state=seed
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            stratify=y_temp,
            random_state=seed
        )
        
        split_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'seed': seed,
            'preprocessor': preprocessor
        }
        
        output_file = output_path / f"split_seed{seed}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        print(f"    Train: {len(X_train)} samples")
        print(f"    Val:   {len(X_val)} samples")
        print(f"    Test:  {len(X_test)} samples")
        print(f"    ✓ Saved with preprocessor")


if __name__ == "__main__":
    # Create splits for all datasets WITH preprocessor
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "splits"
    
    datasets = ['german_credit', 'thyroid']
    seeds = [42, 123, 456]
    
    print("="*70)
    print("Creating Data Splits for Qualsynth (WITH PREPROCESSOR)")
    print("="*70)
    print(f"Seeds: {seeds}")
    print(f"Split ratio: 60% train / 20% val / 20% test")
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        create_splits_with_preprocessor(
            dataset_name=dataset,
            seeds=seeds,
            data_dir=str(data_dir),
            output_dir=str(output_dir)
        )
    
    print("\n" + "="*70)
    print("✅ All splits created successfully!")
    print("="*70)
    
    # Verify splits can decode to raw
    print("\nVerifying raw data decoding...")
    for dataset in datasets:
        for seed in seeds:
            split = load_split(dataset, seed, str(output_dir), return_raw=True)
            if 'preprocessor' in split and split['preprocessor'] is not None:
                # Check a sample value - pick a numerical column
                preprocessor = split['preprocessor']
                if preprocessor.numerical_features:
                    sample_col = preprocessor.numerical_features[0]
                    sample_val = split['X_train'][sample_col].iloc[0]
                    print(f"  {dataset} (seed={seed}): {sample_col}={sample_val:.2f} ✓")
                else:
                    print(f"  {dataset} (seed={seed}): Has preprocessor ✓")
            else:
                print(f"  {dataset} (seed={seed}): No preprocessor ✗")

