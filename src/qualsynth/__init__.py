"""
QualSynth: Quality-Driven Synthetic Data Generation via LLM-Guided Oversampling

A Python package for generating high-quality synthetic samples for imbalanced 
classification using Large Language Models (LLMs) with iterative refinement 
and multi-objective optimization.

Quick Start:
    from qualsynth import QualSynthGenerator
    
    generator = QualSynthGenerator(
        model_name="gemma3:12b",
        temperature=0.7,
        max_iterations=20
    )
    
    X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)

Author: Asım Sinan Yüksel
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Asım Sinan Yüksel"
__email__ = "asimyuksel@sdu.edu.tr"

# Main API - simple interface
from .generator import QualSynthGenerator

# Advanced API - full control
from .core.iterative_workflow import IterativeRefinementWorkflow as IterativeWorkflow
from .core.iterative_workflow import WorkflowConfig
from .generators.counterfactual_generator import CounterfactualGenerator
from .validation.adaptive_validator import AdaptiveValidator
from .data.splitting import load_split, encode_features, decode_features

__all__ = [
    # Simple API
    "QualSynthGenerator",
    
    # Advanced API
    "IterativeWorkflow",
    "WorkflowConfig",
    "CounterfactualGenerator", 
    "AdaptiveValidator",
    
    # Data utilities
    "load_split",
    "encode_features",
    "decode_features",
]
