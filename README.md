# QualSynth: Quality-Driven Synthetic Data Generation via LLM-Guided Oversampling

[PyPI version](https://pypi.org/project/qualsynth/)
[Python 3.8+](https://www.python.org/downloads/)
[License: GPL v3](https://www.gnu.org/licenses/gpl-3.0)

**QualSynth** is a Python package that leverages Large Language Models (LLMs) with iterative refinement to generate quality-validated synthetic samples for imbalanced classification tasks.



## Key Features

- **LLM-Guided Generation**: Uses LLMs to generate contextually aware synthetic samples that respect domain constraints
- **Multi-Stage Validation**: Every sample passes schema validation, statistical checks, and duplicate detection
- **Anchor-Centric Approach**: Generates variations of real minority samples, preserving natural feature correlations
- **Zero Duplicates**: Achieves 0% duplicate ratio across all datasets (vs 29.7% for TabFairGDT)
- **Fairness-Aware**: Reduces demographic parity difference without explicit fairness constraints
- **Multiple LLM Backends**: Supports OpenAI, Ollama (local), OpenRouter, and custom endpoints

## Performance Highlights

Evaluated on **8 benchmark datasets** across **320 experiments** (8 datasets × 10 seeds × 4 methods):


| Metric              | QualSynth | SMOTE | CTGAN | TabFairGDT |
| ------------------- | --------- | ----- | ----- | ---------- |
| **F1 Rank**         | **2.12**  | 2.25  | 2.50  | 3.12       |
| **ROC-AUC Rank**    | **1.63**  | 2.50  | 3.63  | 2.25       |
| **Duplicate Ratio** | **0%**    | 0%    | 0%    | 29.7%      |
| **DPD (Fairness)**  | **0.062** | 0.089 | 0.139 | 0.095      |




## Quick Start

### Installation

**From PyPI (Recommended):**

```bash
pip install qualsynth
```

**From Source:**

```bash
# Clone the repository
git clone https://github.com/asimsinan/qualsynth.git
cd qualsynth

# Install in development mode
pip install -e .
```

**Verify Installation:**

```python
from qualsynth import QualSynthGenerator
print("QualSynth installed successfully!")
```

### Basic Usage

```python
from qualsynth import QualSynthGenerator

# Initialize generator
generator = QualSynthGenerator(
    model_name="gpt-4",
    api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
    temperature=0.7,
    max_iterations=20
)

# Generate synthetic samples
X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)

# Combine with original data for training
X_augmented = pd.concat([X_train, X_synthetic])
y_augmented = pd.concat([y_train, y_synthetic])
```

### Using Local LLMs (Ollama)

```python
from qualsynth import QualSynthGenerator

# First, start Ollama server: ollama serve
# Pull a model: ollama pull gemma3:12b

generator = QualSynthGenerator(
    model_name="gemma3:12b",  # Model name from 'ollama list'
    api_base="http://localhost:11434/v1"
)

X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)
```

### Using OpenRouter (Cloud)

```python
from qualsynth import QualSynthGenerator

generator = QualSynthGenerator(
    model_name="google/gemma-2-9b-it",
    api_key="your-openrouter-api-key",
    api_base="https://openrouter.ai/api/v1"
)

X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)
```

## Project Structure

```
qualsynth/
├── src/qualsynth/           # Main package source code
│   ├── core/                # Core workflow logic
│   │   └── iterative_workflow.py
│   ├── generators/          # Sample generation
│   │   └── counterfactual_generator.py
│   ├── validation/          # Multi-stage validation
│   │   ├── adaptive_validator.py
│   │   └── universal_validator.py
│   ├── modules/             # LLM-powered modules
│   │   ├── dataset_profiler.py
│   │   ├── schema_profiler.py
│   │   ├── diversity_planner.py
│   │   ├── fairness_auditor.py
│   │   └── validator.py
│   ├── baselines/           # Baseline implementations
│   │   ├── smote.py
│   │   ├── ctgan_baseline.py
│   │   └── tabfairgdt.py
│   ├── evaluation/          # Metrics and classifiers
│   └── prompts/             # LLM prompt templates
├── configs/                 # Dataset and method configurations
├── scripts/                 # Experiment scripts
├── replication/             # Replication package
│   ├── qualsyn-1.0.0/       # Standalone package
│   └── tables/              # Pre-computed results
└── data/splits/             # Pre-computed dataset splits
```

## 🔧 Configuration Options


| Parameter              | Default        | Description                                    |
| ---------------------- | -------------- | ---------------------------------------------- |
| `model_name`           | `"gemma3:12b"` | LLM model to use                               |
| `api_key`              | `None`         | API key for cloud providers                    |
| `api_base`             | `None`         | Custom API endpoint URL                        |
| `temperature`          | `0.7`          | Generation diversity (lower = more consistent) |
| `batch_size`           | `20`           | Samples per LLM call                           |
| `max_iterations`       | `20`           | Maximum refinement iterations                  |
| `target_ratio`         | `1.0`          | Target class ratio (1.0 = balanced)            |
| `validation_threshold` | `4.5`          | Statistical validation threshold (σ)           |
| `sensitive_attributes` | `None`         | Columns for fairness-aware generation          |


## Datasets

The package has been evaluated on 8 benchmark datasets:


| Dataset       | Domain       | Samples | Features | Imbalance Ratio |
| ------------- | ------------ | ------- | -------- | --------------- |
| German Credit | Finance      | 1,000   | 20       | 2.33:1          |
| Breast Cancer | Medical      | 569     | 30       | 1.68:1          |
| Pima Diabetes | Medical      | 768     | 8        | 1.87:1          |
| Haberman      | Medical      | 306     | 3        | 2.78:1          |
| Wine Quality  | Food Science | 4,898   | 11       | 3.39:1          |
| Yeast         | Biology      | 1,484   | 8        | 28.10:1         |
| Thyroid       | Medical      | 3,772   | 25       | 15.09:1         |
| HTRU2         | Astronomy    | 17,898  | 8        | 9.16:1          |


## Reproducing Experiments

### Running All Experiments

```bash
# Using OpenRouter (recommended)
python scripts/run_openrouter_experiments.py --all --seeds 42 123 456 789 1234 2024 3141 4242 5555 6789

# Using local Ollama
./scripts/run_with_ollama_m4.sh

# Using Ubuntu setup script (automated)
./scripts/setup_ubuntu.sh
```

### Running Single Experiment

```bash
python scripts/run_experiments.py \
    --dataset german_credit \
    --method qualsynth \
    --seed 42
```

### Running Baselines

```bash
# SMOTE
python scripts/run_experiments.py --dataset german_credit --method smote --seed 42

# CTGAN
python scripts/run_experiments.py --dataset german_credit --method ctgan --seed 42

# TabFairGDT
python scripts/run_experiments.py --dataset german_credit --method tabfairgdt --seed 42
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

**Asım Sinan Yüksel**  
Department of Computer Engineering  
Süleyman Demirel University  
Email: [asimyuksel@sdu.edu.tr](mailto:asimyuksel@sdu.edu.tr)