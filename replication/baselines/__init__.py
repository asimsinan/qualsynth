"""
Baseline methods for comparison with QualSynth.

These are NOT part of the QualSynth package - they are provided here
for replication purposes only. Each baseline wraps an existing library.

Baselines:
- SMOTE: Synthetic Minority Over-sampling Technique (imbalanced-learn)
- CTGAN: Conditional Tabular GAN (ctgan)
- TabFairGDT: Fair Gradient Decision Tree (custom implementation)
"""

from .smote import SMOTEBaseline
from .ctgan_baseline import CTGANBaseline
from .tabfairgdt import TabFairGDTBaseline

__all__ = [
    "SMOTEBaseline",
    "CTGANBaseline",
    "TabFairGDTBaseline",
]
