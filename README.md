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

- Among five synthetic oversamplers, QualSynth obtained the highest F1 on the most datasets (four of eight), and its predictive utility was equivalent to that of SMOTE within pre-specified margins. SMOTE had slightly better average ranks because its performance was more uniform across datasets.
- QualSynth obtained higher ROC-AUC than TabDDPM after Holm correction.
- A class-weighted model trained without augmentation had the best overall ranks and slightly higher aggregate mean F1 and ROC-AUC than QualSynth, although the pairwise differences were not significant after correction.
- Relative to unweighted training, the F1 results show that the benefit arises from imbalance-aware learning more broadly rather than from synthetic oversampling alone.
- QualSynth was the only evaluated oversampler that produced no exact duplicate rows. This exact-row result is not a semantic-novelty or privacy guarantee.
- Gemma 3 27B and GPT-5.6 Luna Pro produced comparable predictive utility under the same workflow, while differing in candidate acceptance, distributional quality, request volume, runtime, and provider charge.

These results position QualSynth as a quality-controlled option when synthetic minority samples, inspectable admission decisions, reproducibility, or rapid adaptation are required. It is not intended as a universal replacement for class weighting or established oversamplers.

## Reproducing the experiments

Experiment definitions are stored under `configs/experiments/`, with method settings under `configs/methods/`.

Prerequisites:

- benchmark CSV files under `data/raw/`;
- predefined splits under `data/splits/<dataset>/`;
- project dependencies installed in the active environment;
- backend credentials or a running local endpoint for QualSynth experiments; and
- for TabDDPM, the official implementation (https://github.com/yandex-research/tab-ddpm, commit `b476257`) cloned into `third_party/tab-ddpm`. With scikit-learn 1.6 or later, line 221 of `lib/data.py` must pass `subsample=int(1e9)` instead of `subsample=1e9`; the value, and therefore the result, is unchanged.

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
python scripts/run_threshold_sensitivity.py --help
python scripts/run_backend_sensitivity.py --help
python scripts/analyze_benchmark.py --help
```

## Scripts

| Script | Purpose |
| --- | --- |
| `run_experiments.py` | Runs the benchmark baselines (SMOTE, CTGAN, TabFairGDT, TabDDPM) and the no-augmentation references from an experiment configuration |
| `run_openrouter_experiments.py` | Runs QualSynth on the benchmark datasets through an OpenAI-compatible endpoint (OpenRouter by default) |
| `evaluate_stored_runs.py` | Retrains and evaluates the three classifiers for every run from its archived synthetic rows, scoring all methods in one pipeline (the reported benchmark values) |
| `analyze_benchmark.py` | Computes dataset-level means, ranks, Friedman tests, Holm-corrected Wilcoxon comparisons, the classifier summary, and per-dataset seed-paired tests for the benchmark |
| `equivalence_tests.py` | Runs paired TOST equivalence tests of QualSynth against each comparator |
| `quality_audit.py` | Computes acceptance, exact-duplicate rate, and correlation difference for the synthetic rows of each method |
| `run_threshold_sensitivity.py` | Applies the nine admission policies to fixed candidate pools (validation-threshold sensitivity) |
| `reapply_ablation_final_pass.py` | Re-applies the final validation pass to the component-ablation runs with the canonicalised duplicate check |
| `fold_safe_preprocessing_check.py` | Re-evaluates the benchmark with features standardised on each training fold (`--mode fold`), regenerating SMOTE; `--mode stored` reproduces the archived evaluation |
| `fixed_reference_sensitivity.py` | Removes stored QualSynth rows that fail a fixed z-score reference (`--reference fold` or `minority_numeric`) and re-evaluates utility |
| `anchor_coverage_analysis.py` | Compares anchor coverage of the implemented stratification with first-principal-component strata and random anchors (no generation) |
| `threshold_protection_metrics.py` | Reports multivariate outliers and class overlap among the rows admitted under each threshold condition |
| `run_backend_sensitivity.py`, `analyze_backend_sensitivity.py` | Run and analyse the paired backend comparison (Gemma 3 27B and GPT-5.6 Luna Pro) |
| `run_component_ablation.py`, `build_component_ablation_report.py` | Run the component ablation and compute its quality diagnostics and paired tests |
| `prepare_high_dimensional_benchmark.py`, `prepare_high_dim_extension.py` | Prepare fold-safe high-dimensional splits for the Alon and Golub datasets |
| `run_high_dim_extension.py`, `build_high_dim_extension_report.py` | Run and analyse the high-dimensional extension |
| `build_high_dim_table.py` | Computes the high-dimensional table (all five methods at every setting) from the archived runs |
| `analyze_cost_runtime.py` | Aggregates runtime and request-volume measurements |

The analysis scripts read the outputs written by the corresponding run scripts under `results/`.

### Recomputing the reported results from the companion dataset

The companion dataset archive (`qualsynth-1.1.3-data.zip`) contains the predefined splits and the per-run outputs, including the generated synthetic samples, on which the reported results are based. Extract it into the root of this archive so that `data/` and `results/` sit next to `scripts/`, then run:

```bash
python scripts/evaluate_stored_runs.py
python scripts/analyze_benchmark.py --per-run-dir results/analysis/evaluation
python scripts/equivalence_tests.py
python scripts/quality_audit.py
```

The outputs are written to `results/analysis/`. Because hosted LLM decoding is not deterministic, new QualSynth runs produce comparable but not identical samples; the stored samples allow the reported numbers to be recomputed exactly.


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

Asim Sinan Yuksel  
Department of Computer Engineering, Süleyman Demirel University  
[asimyuksel@sdu.edu.tr](mailto:asimyuksel@sdu.edu.tr)
