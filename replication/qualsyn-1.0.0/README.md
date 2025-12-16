# QualSynth

**Quality-Driven Synthetic Data Generation via LLM-Guided Oversampling**

QualSynth is a Python package for generating high-quality synthetic samples for imbalanced classification using Large Language Models (LLMs) with iterative refinement and multi-objective optimization.

## Key Features

- **Anchor-Centric Generation**: LLM generates variations of real samples, preserving feature correlations
- **Multi-Stage Validation**: Hash-based deduplication, schema validation, statistical validation
- **Multi-Objective Optimization**: Balances fairness (60%), diversity (20%), and quality (20%)
- **100% Validation Pass Rate**: All generated samples pass quality validation
- **Local LLM Support**: Works with Ollama for privacy-preserving local deployment

## Installation

```bash
pip install qualsyn
```

Or install from source:

```bash
cd qualsyn-1.0.0
pip install -e .
```

## Quick Start

```python
import pandas as pd
from qualsyn import QualSynthGenerator

# Load imbalanced dataset
X_train = pd.read_csv("german_credit_train.csv")
y_train = pd.read_csv("german_credit_labels.csv")["target"]

# Initialize generator (using OpenAI)
generator = QualSynthGenerator(
    model_name="gpt-4",
    api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
    temperature=0.7,
    max_iterations=20
)

# Generate synthetic samples
X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)

# Combine with original data
X_balanced = pd.concat([X_train, X_synthetic])
y_balanced = pd.concat([y_train, y_synthetic])
```

## Using Cloud APIs (OpenRouter, OpenAI, etc.)

For cloud-based LLM providers:

```python
# OpenRouter example
generator = QualSynthGenerator(
    model_name="google/gemma-2-9b-it",
    api_base="https://openrouter.ai/api/v1",
    api_key="your-openrouter-api-key"
)

# OpenAI example
generator = QualSynthGenerator(
    model_name="gpt-4",
    api_key="your-openai-api-key"
)

X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)
```

## Using Local Models (Ollama)

For privacy-sensitive applications, QualSynth supports fully local deployment:

```python
generator = QualSynthGenerator(
    model_name="gemma2:9b",  # Model name as shown by 'ollama list'
    api_base="http://localhost:11434/v1"
)

X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)
```

## Fairness-Aware Generation

When sensitive attributes are present, QualSynth prioritizes samples that reduce demographic disparity:

```python
generator = QualSynthGenerator(
    model_name="gpt-4",
    api_key="your-api-key",
    sensitive_attributes=["gender", "race"]
)

X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"ollama/gemma:12b"` | LLM model name |
| `api_base` | `None` | API endpoint URL (e.g., `"https://openrouter.ai/api/v1"`) |
| `api_key` | `None` | API key for cloud providers (or set `OPENAI_API_KEY` env var) |
| `temperature` | `0.7` | Generation consistency (lower = more consistent) |
| `max_iterations` | `20` | Maximum refinement iterations |
| `batch_size` | `20` | Samples per LLM call |
| `target_ratio` | `1.0` | Target class ratio (1.0 = balanced) |
| `validation_threshold` | `4.5` | Statistical validation threshold (σ) |
| `sensitive_attributes` | `None` | List of sensitive attribute column names |

## Advanced Usage

For more control, use the `IterativeWorkflow` class directly:

```python
from qualsyn import IterativeWorkflow, WorkflowConfig

config = WorkflowConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_iterations=20,
    batch_size=20,
    target_samples=100,
    fairness_weight=0.6,
    diversity_weight=0.2,
    performance_weight=0.2
)

workflow = IterativeWorkflow(config, verbose=True)
result = workflow.run(X_train, y_train, sensitive_features)
```

## Requirements

- Python >= 3.10
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- imbalanced-learn >= 0.11.0
- xgboost >= 1.7.0
- openai >= 1.0.0

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use QualSynth in your research, please cite:

```bibtex
@article{yuksel2025qualsynth,
  title={{QualSynth}: A {Python} Package for Quality-Driven Synthetic Data Generation via {LLM}-Guided Oversampling},
  author={Y{\"u}ksel, As{\i}m Sinan},
  journal={Journal of Statistical Software},
  year={2025}
}
```

## Author

Asım Sinan Yüksel - Süleyman Demirel University
