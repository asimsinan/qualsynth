# QualSynth

Quality-controlled synthetic oversampling for imbalanced tabular classification using large language models.

QualSynth generates new minority-class samples as small variations of observed minority rows. A language model proposes candidates, but candidates enter the augmented training set only after deterministic checks for structure, valid values, statistical plausibility, and exact duplication. The method requires neither a separately trained tabular generator nor language-model fine-tuning for each dataset.

## How it works

1. **Profile the training fold.** QualSynth identifies feature types, valid categories, numerical ranges, and minority-class statistics without using the test fold.
2. **Select minority anchors.** Real minority rows provide the local context for generation.
3. **Generate sparse edits.** The prompt asks the language model to change only one or two features while preserving the rest of the anchor.
4. **Validate every candidate.** Deterministic checks reject malformed, unsupported, implausible, or duplicate rows.
5. **Admit and record samples.** Validated rows are retained until the requested class balance is reached, with candidate-level outcomes available for inspection.

This separation allows users to change the language-model backend while retaining the same validation and admission workflow.

## Installation

Install from PyPI:

```bash
pip install qualsynth
```

Or install the repository in development mode:

```bash
git clone https://github.com/asimyuksel/qualsynth.git
cd qualsynth
pip install -e .
```

QualSynth requires Python 3.8 or later.

## Quick start

```python
import pandas as pd
from qualsynth import QualSynthGenerator

# X_train is a pandas DataFrame; y_train is a pandas Series.
generator = QualSynthGenerator(
    model_name="gpt-4",
    api_key="your-api-key",
    target_ratio=1.0,
    temperature=0.7,
)

X_synthetic, y_synthetic = generator.fit_generate(X_train, y_train)

X_augmented = pd.concat([X_train, X_synthetic], ignore_index=True)
y_augmented = pd.concat([y_train, y_synthetic], ignore_index=True)
```

`target_ratio=1.0` requests equal minority- and majority-class counts. Generation stops when the target is reached, the configured iteration limit is reached, or repeated iterations make no progress.

## Backends

QualSynth uses an OpenAI-compatible interface and can work with hosted services or compatible local servers.

### Local model with Ollama

Start Ollama and pull a model:

```bash
ollama serve
ollama pull gemma3:12b
```

Then configure QualSynth:

```python
generator = QualSynthGenerator(
    model_name="gemma3:12b",
    api_base="http://localhost:11434/v1",
)
```

### Hosted or custom endpoint

```python
generator = QualSynthGenerator(
    model_name="provider/model-name",
    api_base="https://provider.example/v1",
    api_key="your-api-key",
)
```

The framework is backend-flexible, not backend-invariant. Different models can produce different candidate yields, validation rates, distributional characteristics, runtimes, and costs. Local serving can reduce disclosure to a third-party API, but privacy also depends on server logging, retention, access controls, and host security.

## Main configuration options

| Parameter | Default | Purpose |
| --- | --- | --- |
| `model_name` | `gemma3:12b` | Model identifier exposed by the selected endpoint |
| `api_base` | `None` | OpenAI-compatible endpoint URL |
| `temperature` | `0.7` | Sampling variability |
| `batch_size` | `20` | Candidate rows requested per model call |
| `target_ratio` | `1.0` | Desired minority-to-majority ratio |
| `max_iterations` | `0` | Iteration cap; `0` disables the cap |
| `stall_iterations` | `10` | Consecutive no-progress iterations before stopping |
| `validation_threshold` | `4.5` | Feature-wise statistical admission threshold |
| `selection_policy` | `generation_order` | Policy for selecting among validated candidates |
| `seed` | `None` | Random seed for reproducible workflow components |

Multi-objective candidate selection, adaptive validation, minority-range clipping, high-dimensional validation, and additional request controls are available through `QualSynthGenerator` and the YAML configurations under `configs/`.

## Evaluation summary

The study accompanying QualSynth evaluates eight imbalanced binary-classification datasets with ten random seeds and three downstream classifiers.

- Among five synthetic oversamplers, QualSynth achieved the best average ranks for both F1 and ROC-AUC.
- QualSynth obtained higher ROC-AUC than CTGAN and TabDDPM after Holm correction.
- A class-weighted model trained without augmentation achieved slightly higher aggregate mean F1 and ROC-AUC than QualSynth, although the pairwise differences were not significant after correction.
- Relative to unweighted training, the F1 results show that the benefit arises from imbalance-aware learning more broadly rather than from synthetic oversampling alone.
- QualSynth retained zero exact duplicates after canonicalized numeric screening. This exact-row result is not a semantic-novelty or privacy guarantee.
- Gemma 3 27B and GPT-5.6 Luna Pro produced comparable predictive utility under the same workflow, while differing in candidate acceptance, distributional quality, request volume, runtime, and provider charge.

These results position QualSynth as a quality-controlled option when synthetic minority samples, inspectable admission decisions, reproducibility, or rapid adaptation are required. It is not intended as a universal replacement for class weighting or established oversamplers.

## Reproducing the experiments

Experiment definitions are stored under `configs/experiments/`, with method settings under `configs/methods/`.

Prerequisites:

- benchmark CSV files under `data/raw/`;
- predefined splits under `data/splits/<dataset>/`;
- project dependencies installed in the active environment; and
- backend credentials or a running local endpoint for QualSynth experiments.

Run a small baseline check:

```bash
python scripts/run_experiments.py main_experiments \
  --datasets german_credit \
  --methods smote \
  --seeds 42
```

Run the configured benchmark matrix:

```bash
python scripts/run_experiments.py main_experiments
```

Sensitivity and baseline analyses have dedicated entry points:

```bash
python scripts/run_reviewer3_threshold_sensitivity.py --help
python scripts/run_reviewer3_backend_sensitivity.py --help
python scripts/analyze_reviewer3_no_augmentation.py --help
```

## Project structure

```text
qualsynth/
├── configs/                    # Dataset, experiment, and method settings
├── data/                       # Benchmark data and predefined splits
├── scripts/                    # Experiment, analysis, and verification tools
└── src/qualsynth/
    ├── baselines/              # Synthetic-data comparison methods
    ├── core/                   # Iterative generation workflow
    ├── evaluation/             # Classifiers and evaluation utilities
    ├── generators/             # Anchor-centric candidate generation
    ├── prompts/                # Prompt construction
    └── validation/             # Validation and threshold calibration
```

## License

QualSynth is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

## Contact

Asım Sinan Yüksel  
Department of Computer Engineering, Süleyman Demirel University  
[asimyuksel@sdu.edu.tr](mailto:asimyuksel@sdu.edu.tr)
