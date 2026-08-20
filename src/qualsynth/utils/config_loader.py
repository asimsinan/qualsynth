"""
Configuration Loader for Qualsynth Experiments

This module provides utilities for loading and validating YAML configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    description: str
    properties: Dict[str, Any]
    sensitive_attributes: List[Dict[str, Any]]
    splits: Dict[str, float]
    seeds: List[int]
    fairness: Dict[str, float]
    metrics: Dict[str, List[str]]
    notes: str = ""


@dataclass
class MethodConfig:
    """Method configuration."""
    name: str
    type: str
    description: str
    category: str
    hyperparameters: Dict[str, Any] = None
    tuning_grid: Optional[Dict[str, List[Any]]] = None
    settings: Optional[Dict[str, bool]] = None
    expected: Optional[Dict[str, Any]] = None
    references: Optional[List[str]] = None
    notes: str = ""
    components: Optional[List[str]] = None
    strategy: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str
    datasets: List[str]
    methods: Optional[Dict[str, List[str]]] = None  # For main experiments
    seeds: List[int] = None
    total_experiments: int = 0
    evaluation: Dict[str, Any] = None
    output: Dict[str, Any] = None
    execution: Dict[str, Any] = None
    resources: Optional[Dict[str, Any]] = None
    failure_handling: Optional[Dict[str, Any]] = None
    notifications: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    variants: Optional[List[Dict[str, Any]]] = None  # For ablation studies
    notes: str = ""


class ConfigLoader:
    """
    Configuration loader for Qualsynth experiments.
    
    Loads and validates YAML configuration files for datasets, methods, and experiments.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Base directory for configuration files (defaults to project root/configs)
        """
        if config_dir is None:
            # Default to project root/configs
            project_root = Path(__file__).parent.parent.parent.parent
            config_dir = project_root / "configs"
        
        self.config_dir = Path(config_dir)
        
        # Subdirectories
        self.datasets_dir = self.config_dir / "datasets"
        self.methods_dir = self.config_dir / "methods"
        self.experiments_dir = self.config_dir / "experiments"
    
    def load_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """
        Load dataset configuration.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            DatasetConfig object
        """
        config_path = self.datasets_dir / f"{dataset_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return DatasetConfig(**config_dict)
    
    def load_method_config(self, method_name: str) -> MethodConfig:
        """
        Load method configuration.
        
        Args:
            method_name: Name of the method
        
        Returns:
            MethodConfig object
        """
        config_path = self.methods_dir / f"{method_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Method config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return MethodConfig(**config_dict)
    
    def load_experiment_config(self, experiment_name: str) -> ExperimentConfig:
        """
        Load experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
        
        Returns:
            ExperimentConfig object
        """
        config_path = self.experiments_dir / f"{experiment_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ExperimentConfig(**config_dict)
    
    def list_datasets(self) -> List[str]:
        """List all available dataset configurations."""
        return [f.stem for f in self.datasets_dir.glob("*.yaml")]
    
    def list_methods(self) -> List[str]:
        """List all available method configurations."""
        return [f.stem for f in self.methods_dir.glob("*.yaml")]
    
    def list_experiments(self) -> List[str]:
        """List all available experiment configurations."""
        return [f.stem for f in self.experiments_dir.glob("*.yaml")]
    
    def validate_experiment_config(self, experiment_config: ExperimentConfig) -> bool:
        """
        Validate experiment configuration.
        
        Checks:
        - All referenced datasets exist
        - All referenced methods exist
        - Seeds are valid
        
        Args:
            experiment_config: Experiment configuration to validate
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check datasets
        available_datasets = self.list_datasets()
        for dataset in experiment_config.datasets:
            if dataset not in available_datasets:
                raise ValueError(f"Dataset '{dataset}' not found in configs")
        
        # Check methods
        available_methods = self.list_methods()
        all_methods = []
        for method_list in experiment_config.methods.values():
            all_methods.extend(method_list)
        
        for method in all_methods:
            if method not in available_methods:
                raise ValueError(f"Method '{method}' not found in configs")
        
        # Check seeds
        if not experiment_config.seeds:
            raise ValueError("At least one seed must be specified")
        
        return True
    
    def get_experiment_matrix(self, experiment_config: ExperimentConfig) -> List[Dict[str, Any]]:
        """
        Generate experiment matrix (all combinations of dataset, method, seed).
        
        Handles both main experiments (with 'methods') and ablation studies (with 'variants').
        
        Args:
            experiment_config: Experiment configuration
        
        Returns:
            List of experiment specifications
        """
        experiments = []
        
        # Get all methods (either from 'methods' or 'variants')
        all_methods = []
        
        if experiment_config.methods:
            # Main experiments: methods grouped by type
            for method_list in experiment_config.methods.values():
                all_methods.extend(method_list)
        elif experiment_config.variants:
            # Ablation study: variants with different configurations
            all_methods = [variant['name'] for variant in experiment_config.variants]
        else:
            raise ValueError("Experiment config must have either 'methods' or 'variants'")
        
        # Generate all combinations
        for dataset in experiment_config.datasets:
            for method in all_methods:
                for seed in experiment_config.seeds:
                    experiments.append({
                        'dataset': dataset,
                        'method': method,
                        'seed': seed,
                        'experiment_id': f"{dataset}_{method}_seed{seed}"
                    })
        
        return experiments


def load_config(config_type: str, config_name: str, config_dir: Optional[str] = None) -> Any:
    """
    Convenience function to load a configuration.
    
    Args:
        config_type: Type of config ('dataset', 'method', 'experiment')
        config_name: Name of the configuration
        config_dir: Base directory for configuration files
    
    Returns:
        Configuration object
    """
    loader = ConfigLoader(config_dir)
    
    if config_type == 'dataset':
        return loader.load_dataset_config(config_name)
    elif config_type == 'method':
        return loader.load_method_config(config_name)
    elif config_type == 'experiment':
        return loader.load_experiment_config(config_name)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # Test configuration loader
    print("="*70)
    print("Testing Configuration Loader")
    print("="*70)
    
    loader = ConfigLoader()
    
    # Test 1: List available configs
    print("\n\nTEST 1: List Available Configurations")
    print("-"*70)
    print(f"Datasets: {loader.list_datasets()}")
    print(f"Methods: {loader.list_methods()}")
    print(f"Experiments: {loader.list_experiments()}")
    
    # Test 2: Load dataset config
    print("\n\nTEST 2: Load Dataset Configuration")
    print("-"*70)
    try:
        german_config = loader.load_dataset_config('german_credit')
        print(f"✓ Loaded: {german_config.name}")
        print(f"  Description: {german_config.description}")
        print(f"  Samples: {german_config.properties['n_samples']}")
        print(f"  Features: {german_config.properties['n_features']}")
        print(f"  Sensitive attributes: {[attr['name'] for attr in german_config.sensitive_attributes]}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Load method config
    print("\n\nTEST 3: Load Method Configuration")
    print("-"*70)
    try:
        qualsynth_config = loader.load_method_config('qualsynth')
        print(f"✓ Loaded: {qualsynth_config.name}")
        print(f"  Type: {qualsynth_config.type}")
        print(f"  Category: {qualsynth_config.category}")
        print(f"  Components: {len(qualsynth_config.components or [])}")
        print(f"  Hyperparameters: {list(qualsynth_config.hyperparameters.keys())[:5]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Load experiment config
    print("\n\nTEST 4: Load Experiment Configuration")
    print("-"*70)
    try:
        exp_config = loader.load_experiment_config('main_experiments')
        print(f"✓ Loaded: {exp_config.name}")
        print(f"  Description: {exp_config.description}")
        print(f"  Datasets: {exp_config.datasets}")
        print(f"  Methods: {sum(len(v) for v in exp_config.methods.values())}")
        print(f"  Seeds: {exp_config.seeds}")
        print(f"  Total experiments: {exp_config.total_experiments}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Validate experiment
    print("\n\nTEST 5: Validate Experiment Configuration")
    print("-"*70)
    try:
        exp_config = loader.load_experiment_config('main_experiments')
        loader.validate_experiment_config(exp_config)
        print("✓ Experiment configuration is valid")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    
    # Test 6: Generate experiment matrix
    print("\n\nTEST 6: Generate Experiment Matrix")
    print("-"*70)
    try:
        exp_config = loader.load_experiment_config('main_experiments')
        matrix = loader.get_experiment_matrix(exp_config)
        print(f"✓ Generated {len(matrix)} experiments")
        print(f"  First experiment: {matrix[0]}")
        print(f"  Last experiment: {matrix[-1]}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n\n✅ Configuration Loader Test Complete")

