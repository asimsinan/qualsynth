#!/usr/bin/env python3
"""Build a submission-grade reproducibility bundle for the revised benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.qualsynth.data.splitting import load_split
    from src.qualsynth.prompts.base_templates import BaseTemplates
except ModuleNotFoundError:
    from qualsynth.data.splitting import load_split
    from qualsynth.prompts.base_templates import BaseTemplates

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "breast_cancer",
    "german_credit",
    "pima_diabetes",
    "wine_quality",
    "yeast",
    "haberman",
    "thyroid",
    "htru2",
]

SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]

CANONICAL_PATHS = [
    PROJECT_ROOT / ".env.example",
    PROJECT_ROOT / "docs" / "step-by-step-experiments.md",
    PROJECT_ROOT / "results" / "openrouter1",
    PROJECT_ROOT / "results" / "experiments1",
    PROJECT_ROOT
    / "results"
    / "reviewer_revision"
    / "canonical_dedup_correction"
    / "openrouter1",
    PROJECT_ROOT / "results" / "reviewer_revision" / "tabddpm_main",
    PROJECT_ROOT
    / "results"
    / "reviewer_revision"
    / "claim_verification_refreshed_tabddpm",
    PROJECT_ROOT
    / "results"
    / "reviewer_revision"
    / "reviewer3_round2"
    / "claim_freeze",
    PROJECT_ROOT / "scripts" / "verify_result_claims.py",
    PROJECT_ROOT / "scripts" / "generate_cd_diagram.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        yield file_path


def build_environment_metadata() -> dict:
    versions = {}
    for module_name in ["numpy", "pandas", "sklearn", "xgboost", "torch"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "unavailable"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "project_root": str(PROJECT_ROOT),
        "package_versions": versions,
    }


def write_split_manifest(output_dir: Path) -> None:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            split = load_split(dataset, seed=seed, return_raw=True)
            y_train = split["y_train"]
            y_val = split["y_val"]
            y_test = split["y_test"]
            minority_label = int(y_train.value_counts().idxmin())
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "minority_label": minority_label,
                    "train_rows": len(split["X_train"]),
                    "val_rows": len(split["X_val"]),
                    "test_rows": len(split["X_test"]),
                    "train_minority": int((y_train == minority_label).sum()),
                    "val_minority": int((y_val == minority_label).sum()),
                    "test_minority": int((y_test == minority_label).sum()),
                    "train_majority": int((y_train != minority_label).sum()),
                    "val_majority": int((y_val != minority_label).sum()),
                    "test_majority": int((y_test != minority_label).sum()),
                    "split_file": str(
                        Path("data") / "splits" / dataset / f"split_seed{seed}.pkl"
                    ),
                }
            )

    fieldnames = list(rows[0].keys())
    with (output_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dataset_summary = {}
    for row in rows:
        dataset_summary.setdefault(row["dataset"], {"seeds": 0, "train_rows": row["train_rows"]})
        dataset_summary[row["dataset"]]["seeds"] += 1
    with (output_dir / "split_manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2)


def write_prompt_bundle(output_dir: Path) -> None:
    system_prompt = BaseTemplates.get_system_prompt(strategy="STANDARD")
    task_prompt = BaseTemplates.get_task_description(
        dataset_name="<dataset_name>",
        n_samples=20,
        target_class=1,
        minority_class_name="minority class",
    )
    representative_prompt = """SYSTEM INSTRUCTION
Objective: Generate local variations of real minority-class anchors while
preserving relationships among features.

Sparse-edit rule:
- Begin with an exact copy of the assigned anchor.
- Modify only one or two features.
- Keep every other value unchanged.
- Do not construct rows by generating features independently.

Output contract:
- Return exactly <batch_size> header-free CSV rows.
- Return <number_of_features> comma-separated values per row.
- Use this column order: <feature_1>, <feature_2>, ..., <feature_d>.
- Do not return explanations or additional text.

USER PROMPT
Task: Generate <batch_size> small variations of the anchors below.

Column order:
<feature_1>, <feature_2>, ..., <feature_d>

Valid feature values:
<feature-specific numerical ranges and summaries>
<observed categorical values>

Anchor assignments:
ANCHOR 1: <feature_1=value_1, feature_2=value_2, ...>
Generate <n_1> variations; modify only 1-2 features and retain all others.
...

Permitted variation:
- Numerical: approximately +/-10-15% of the anchor value.
- Categorical: retain with 80% probability; otherwise use another valid value.

Runtime example:
<one- and two-feature edits derived from the first anchor, plus an invalid edit>

Final instruction:
Generate exactly <batch_size> anchor-variation rows in the stated column order,
without a header.
"""
    content = f"""# Prompt Bundle

This bundle records the fixed prompt contract used by the revised manuscript benchmark.

## Source Files

- `src/qualsynth/prompts/base_templates.py`
- `src/qualsynth/prompts/prompt_builder.py`
- `src/qualsynth/prompts/few_shot_builder.py`

## Fixed System Prompt Template

```text
{system_prompt}
```

## Fixed Task Template

```text
{task_prompt}
```

## Representative Assembled Prompt Structure

```text
{representative_prompt}
```

## Anchor Notes

- Anchors are selected dynamically from the minority-class training split.
- The benchmarked manuscript path uses anchor-centric sparse edits, structured output, and deterministic post-generation validation.
- The public manuscript documents the operative prompt contract rather than claiming token-level replay of historical API calls.
"""
    (output_dir / "prompt_bundle.md").write_text(content, encoding="utf-8")


def write_artifact_manifest(output_dir: Path) -> None:
    rows = []
    for canonical_path in CANONICAL_PATHS:
        exists = canonical_path.exists()
        if not exists:
            rows.append(
                {
                    "root": str(canonical_path.relative_to(PROJECT_ROOT)),
                    "path": str(canonical_path.relative_to(PROJECT_ROOT)),
                    "exists": False,
                    "size_bytes": 0,
                    "sha256": "",
                }
            )
            continue

        for file_path in iter_files(canonical_path):
            rows.append(
                {
                    "root": str(canonical_path.relative_to(PROJECT_ROOT)),
                    "path": str(file_path.relative_to(PROJECT_ROOT)),
                    "exists": True,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),
                }
            )

    with (output_dir / "artifact_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["root", "path", "exists", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "roots": [str(path.relative_to(PROJECT_ROOT)) for path in CANONICAL_PATHS],
        "file_count": len([row for row in rows if row["exists"]]),
        "missing_roots": [row["root"] for row in rows if not row["exists"]],
    }
    with (output_dir / "artifact_manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def write_bundle_readme(output_dir: Path) -> None:
    content = """# Reproducibility Bundle

This directory collects the concrete artifacts used to support the revised Scientific Reports submission.

## Contents

- `environment.json`: execution environment metadata for the bundle-generation step.
- `split_manifest.csv`: canonical 60/20/20 split inventory for every dataset and seed.
- `prompt_bundle.md`: fixed prompt-contract documentation for the benchmarked QualSynth path.
- `artifact_manifest.csv`: file-level inventory with SHA-256 digests for the canonical benchmark roots.
- `bundle_manifest.json`: machine-readable summary of the bundle contents.

## Canonical Benchmark Roots

- `results/openrouter1`: archived QualSynth benchmark outputs.
- `results/experiments1`: baseline benchmark outputs for SMOTE, CTGAN, and TabFairGDT.
- `results/reviewer_revision/canonical_dedup_correction/openrouter1`: provenance-preserving correction overlays, including the retained-sample Thyroid representation correction.
- `results/reviewer_revision/tabddpm_main`: regenerated TabDDPM outputs for the matched five-method comparison.
- `results/reviewer_revision/claim_verification_refreshed_tabddpm`: rebuilt five-method statistics and candidate-quality audit.
- `results/reviewer_revision/reviewer3_round2/claim_freeze`: frozen six-condition manuscript claims and generated tables.

## Reproduction Entry Points

- `.env.example`
- `docs/step-by-step-experiments.md`
- `scripts/verify_result_claims.py`
- `scripts/generate_cd_diagram.py`
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/reviewer_revision/reproducibility_bundle",
        help="Output directory for the generated bundle.",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_bundle_readme(output_dir)
    write_split_manifest(output_dir)
    write_prompt_bundle(output_dir)
    write_artifact_manifest(output_dir)

    environment = build_environment_metadata()
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    bundle_manifest = {
        "generated_at_utc": environment["generated_at_utc"],
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote reproducibility bundle to: {output_dir}")


if __name__ == "__main__":
    main()
