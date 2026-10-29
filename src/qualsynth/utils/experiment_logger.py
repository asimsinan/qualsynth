"""
Experiment Logger with Real-time Progress Tracking

Provides detailed logging for experiment execution with progress updates.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class ExperimentLogger:
    """
    Logger for experiment execution with real-time progress tracking.
    
    Logs to both file and console with structured progress updates.
    """
    
    def __init__(
        self,
        experiment_id: str,
        log_dir: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_id: Unique experiment identifier
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
        """
        self.experiment_id = experiment_id
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path("results/logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"experiment.{experiment_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level)
        log_file = self.log_dir / f"{experiment_id}.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Progress file (JSON for easy parsing)
        self.progress_file = self.log_dir / f"{experiment_id}_progress.json"
        self.progress_data = {
            'experiment_id': experiment_id,
            'start_time': datetime.now().isoformat(),
            'status': 'starting',
            'current_step': None,
            'progress': 0.0,
            'steps': []
        }
        self._write_progress()
    
    def _write_progress(self):
        """Write progress data to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)
    
    def start_experiment(self, dataset: str, method: str, seed: int):
        """Log experiment start."""
        self.logger.info("="*80)
        self.logger.info(f"STARTING EXPERIMENT: {self.experiment_id}")
        self.logger.info(f"Dataset: {dataset} | Method: {method} | Seed: {seed}")
        self.logger.info("="*80)
        
        self.progress_data['status'] = 'running'
        self.progress_data['dataset'] = dataset
        self.progress_data['method'] = method
        self.progress_data['seed'] = seed
        self._write_progress()
    
    def start_step(self, step_name: str, description: str = ""):
        """Log step start."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"STEP: {step_name}")
        if description:
            self.logger.info(f"Description: {description}")
        self.logger.info(f"{'='*80}")
        
        step_data = {
            'name': step_name,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        self.progress_data['current_step'] = step_name
        self.progress_data['steps'].append(step_data)
        self._write_progress()
    
    def update_step(self, message: str, level: str = 'info'):
        """Log step update."""
        log_func = getattr(self.logger, level.lower())
        log_func(f"  → {message}")
    
    def complete_step(self, step_name: str, result: Optional[dict] = None):
        """Log step completion."""
        self.logger.info(f"✓ Completed: {step_name}")
        
        # Update progress data
        for step in self.progress_data['steps']:
            if step['name'] == step_name and step['status'] == 'running':
                step['status'] = 'completed'
                step['end_time'] = datetime.now().isoformat()
                if result:
                    step['result'] = result
                break
        
        self._write_progress()
    
    def fail_step(self, step_name: str, error: str):
        """Log step failure."""
        self.logger.error(f"✗ Failed: {step_name}")
        self.logger.error(f"Error: {error}")
        
        # Update progress data
        for step in self.progress_data['steps']:
            if step['name'] == step_name and step['status'] == 'running':
                step['status'] = 'failed'
                step['end_time'] = datetime.now().isoformat()
                step['error'] = error
                break
        
        self.progress_data['status'] = 'failed'
        self._write_progress()
    
    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log metrics in a formatted way."""
        self.logger.info(f"\n{prefix}Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def complete_experiment(self, success: bool, result: Optional[dict] = None):
        """Log experiment completion."""
        self.logger.info("\n" + "="*80)
        if success:
            self.logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        else:
            self.logger.error("❌ EXPERIMENT FAILED")
        self.logger.info("="*80)
        
        self.progress_data['status'] = 'completed' if success else 'failed'
        self.progress_data['end_time'] = datetime.now().isoformat()
        if result:
            self.progress_data['result'] = result
        self._write_progress()
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


def get_experiment_logger(experiment_id: str) -> ExperimentLogger:
    """
    Get or create experiment logger.
    
    Args:
        experiment_id: Unique experiment identifier
    
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_id)


if __name__ == "__main__":
    # Test logger
    logger = ExperimentLogger("test_german_credit_smote_seed42")
    
    logger.start_experiment("german_credit", "smote", 42)
    
    logger.start_step("load_data", "Loading and preprocessing data")
    logger.update_step("Loading raw data from CSV")
    logger.update_step("Preprocessing features")
    logger.complete_step("load_data", {"n_samples": 27133})
    
    logger.start_step("apply_method", "Applying SMOTE oversampling")
    logger.update_step("Fitting SMOTE model")
    logger.update_step("Generating synthetic samples")
    logger.complete_step("apply_method", {"n_generated": 18123})
    
    logger.start_step("train_classifiers", "Training classifiers")
    logger.update_step("Training Random Forest")
    logger.update_step("Training XGBoost")
    logger.update_step("Training Logistic Regression")
    logger.complete_step("train_classifiers")
    
    logger.log_metrics({
        'f1': 0.5150,
        'roc_auc': 0.8416,
        'dpd': 0.3031
    }, prefix="Performance ")
    
    logger.complete_experiment(True, {'execution_time': 3.42})
    
    print("\n✅ Logger test complete!")
    print(f"Log file: {logger.log_dir / f'{logger.experiment_id}.log'}")
    print(f"Progress file: {logger.progress_file}")

