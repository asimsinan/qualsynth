"""
Qualsynth Prompts Module

This module contains prompt engineering components for fairness-aware
synthetic data generation with LLMs.
"""

from .base_templates import BaseTemplates
from .fairness_templates import FairnessTemplates
from .counterfactual_templates import CounterfactualTemplates
from .few_shot_builder import FewShotBuilder
from .constraint_encoder import ConstraintEncoder
from .prompt_builder import PromptBuilder

__all__ = [
    'BaseTemplates',
    'FairnessTemplates',
    'CounterfactualTemplates',
    'FewShotBuilder',
    'ConstraintEncoder',
    'PromptBuilder'
]

