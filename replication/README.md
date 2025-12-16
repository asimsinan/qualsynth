# Replication Materials for QualSynth Paper

**Paper Title:** QualSynth: A Python Package for Quality-Driven Synthetic Data Generation via LLM-Guided Oversampling

**Authors:** Asım Sinan Yüksel  
**Affiliation:** Süleyman Demirel University  
**Journal:** Journal of Statistical Software

## Contents

```
replication/
├── README.md                    # This file
├── replication.py               # Main replication script
├── setup_ubuntu.sh              # Automated Ubuntu/Linux setup script
├── requirements.txt             # Python dependencies
├── qualsyn-1.0.0/              # QualSynth software package
│   ├── LICENSE                  # GPL-3.0 license
│   ├── README.md               # Package documentation
│   ├── pyproject.toml          # Package configuration
│   └── qualsyn/                # Source code
│       ├── __init__.py         # Package exports (QualSynthGenerator)
│       ├── generator.py        # Simple API (QualSynthGenerator class)
│       ├── core/               # Iterative workflow engine
│       ├── generators/         # LLM-based sample generators
│       ├── validation/         # Quality validation pipeline
│       ├── data/               # Data loading and preprocessing
│       ├── evaluation/         # Performance and fairness metrics
│       ├── experiments/        # Experiment runner
│       ├── prompts/            # Prompt templates
│       ├── modules/            # Profiling and optimization
│       └── utils/              # Utilities and configuration
├── baselines/                   # Baseline methods (for replication only)
│   ├── smote.py                # SMOTE wrapper (imbalanced-learn)
│   ├── ctgan_baseline.py       # CTGAN wrapper
│   └── tabfairgdt.py           # TabFairGDT wrapper
└── tables/                     # Pre-computed results
    ├── german_credit_*.csv     # German Credit results
    ├── breast_cancer_*.csv     # Breast Cancer results
```

**Note:** Baseline methods are NOT part of the QualSynth package. They are provided 
separately in the `baselines/` folder for replication purposes only.

## Quick Start

### Option 1: Quick Replication (Recommended for Reviewers)

Uses pre-computed results to generate all tables and figures from the paper.
**No LLM required - runs in ~5 minutes:**

```bash
cd replication
pip install pandas numpy scipy
python replication.py --quick
```

This will reproduce all tables from the paper using the pre-computed results in the `tables/` directory.

### Option 2: Ubuntu/Linux Automated Setup

For Ubuntu 20.04/22.04/24.04 (including WSL2), use the automated setup script:

```bash
cd replication
chmod +x setup_ubuntu.sh

# Quick replication only (5 minutes)
./setup_ubuntu.sh --quick

# Full setup without running experiments
./setup_ubuntu.sh --setup-only

# Full setup and run all experiments
./setup_ubuntu.sh
```

The script automatically:
- Installs Python 3.10 and dependencies
- Creates a virtual environment
- Installs Ollama and downloads the LLM model
- Runs experiments with checkpointing (can be interrupted and resumed)

### Option 3: Manual Full Replication (macOS/Windows/Linux)

Runs all experiments from scratch (requires ~80 hours of computation):

```bash
# Install the package
cd qualsyn-1.0.0
pip install -e .
cd ..

# Install Ollama (https://ollama.ai)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: Download from https://ollama.ai

# Pull the model (~8GB download)
ollama serve  # Start server (in separate terminal)
ollama pull gemma:12b

# Run full replication
python replication.py --full
```

### Using QualSynth Directly

You can also use QualSynth directly in your own code:

```python
import pandas as pd
from qualsyn import QualSynthGenerator

# Load your imbalanced dataset
X_train = pd.read_csv("train_features.csv")
y_train = pd.read_csv("train_labels.csv")["target"]

# Initialize generator
generator = QualSynthGenerator(
    model_name="ollama/gemma:12b",  # or "gpt-4" for OpenAI
    temperature=0.7,
    max_iterations=20
)

# Generate synthetic samples
X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)

# Combine with original data for training
X_balanced = pd.concat([X_train, X_synthetic])
y_balanced = pd.concat([y_train, y_synthetic])
```

## Requirements

### For Quick Replication
- Python 3.10+
- pandas
- numpy
- scipy

### For Full Replication
- All of the above, plus:
- Ollama 0.13.0+ with Gemma 3 12B model
- scikit-learn 1.6.1+
- imbalanced-learn 0.14.0+
- xgboost 3.1.1+
- httpx 0.27.0+
- ~80 hours of computation time
- ~16GB RAM (for local LLM inference)

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install QualSynth package
cd qualsyn-1.0.0
pip install -e .
```

## Datasets

The experiments use 8 benchmark datasets from the UCI Machine Learning Repository across 5 domains:

| Dataset | Domain | Samples | Features | Imbalance Ratio |
|---------|--------|---------|----------|-----------------|
| German Credit | Finance | 1,000 | 20 | 2.33 |
| Breast Cancer | Medical | 569 | 30 | 1.68 |
| Pima Diabetes | Medical | 768 | 8 | 1.87 |
| Wine Quality | Food Science | 6,497 | 12 | 4.09 |
| Yeast | Biology | 1,484 | 8 | 28.10 |
| Haberman | Medical | 306 | 3 | 2.78 |
| Thyroid | Medical | 3,772 | 29 | 15.33 |
| HTRU2 | Astronomy | 17,898 | 8 | 9.92 |

Datasets are automatically downloaded when running the full replication.

## Reproducing Specific Results

### Run specific dataset:
```bash
python replication.py --dataset german_credit --seeds 42 123 456
```

### Run with specific seeds:
```bash
python replication.py --full --seeds 42 123
```

## Pre-computed Results

The `tables/` directory contains CSV files with results for each dataset-method combination:

- `{dataset}_{method}_results.csv`: Per-seed results including F1, ROC-AUC, accuracy, recall, precision

These files are used by `replication.py --quick` to generate the paper's tables.

## License

This software is released under the GNU General Public License v3.0 (GPL-3.0).
See `qualsyn-1.0.0/LICENSE` for details.

## Citation

If you use this software or these results, please cite:

```bibtex
@article{yuksel2025qualsynth,
  title={{QualSynth}: A {Python} Package for Quality-Driven Synthetic Data 
         Generation via {LLM}-Guided Oversampling},
  author={Y{\"u}ksel, As{\i}m Sinan},
  journal={Journal of Statistical Software},
  year={2025}
}
```

## Contact

For questions or issues, please contact:
- Asım Sinan Yüksel: asimyuksel@sdu.edu.tr
