#!/usr/bin/env python3
"""Verify that the revised manuscript is controlled by the round-2 claim freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE_ROOT = PROJECT_ROOT / "results/reviewer_revision/reviewer3_round2/claim_freeze"
DEFAULT_MANUSCRIPT = PROJECT_ROOT / "sreport/main.tex"
AUDIT_PATH = FREEZE_ROOT / "manuscript_claim_audit.json"

REQUIRED_TABLE_INPUTS = [
    "generated/round2_performance_f1.tex",
    "generated/round2_performance_roc_auc.tex",
    "generated/round2_statistical_tests.tex",
    "generated/round2_threshold_sensitivity.tex",
    "generated/round2_backend_sensitivity.tex",
    "generated/round2_classifier_summary.tex",
    "generated/round2_quality_audit.tex",
    "generated/round2_component_ablation.tex",
    "generated/round2_high_dimensional.tex",
    "generated/round2_historical_cost.tex",
]

REQUIRED_TABLE_LABELS = [
    "tab:round2_f1",
    "tab:round2_auc",
    "tab:round2_stats",
    "tab:round2_threshold",
    "tab:round2_backend",
    "tab:round2_classifier",
    "tab:round2_quality",
    "tab:round2_component",
    "tab:round2_high_dim",
    "tab:round2_cost",
]

FORBIDDEN_STALE_TEXT = [
    "1200 total experiments",
    "best average ranks for F1 score and ROC-AUC",
    "best average F1 ranking across all methods",
    "zero false positives",
    "truly novel",
    "percentile guard at the 99.5th percentile",
    "all feature values must fall within the minority-class range",
    "The reported monetary cost is zero",
    "Monetary cost in the runs is zero",
    "free, open-source LLM",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--no-write-audit", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_manifest_path(relative: str) -> Path:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_hashes(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for group in ["inputs", "outputs"]:
        for relative, expected in manifest[group].items():
            try:
                path = resolve_manifest_path(relative)
            except FileNotFoundError:
                errors.append(f"Missing frozen {group[:-1]}: {relative}")
                continue
            observed = sha256_file(path)
            if observed != expected:
                errors.append(
                    f"Hash drift for {relative}: expected {expected}, observed {observed}"
                )
    return errors


def parse_macro_file(text: str) -> dict[str, str]:
    pattern = re.compile(r"\\providecommand\{\\(?P<name>RThree[A-Za-z]+)\}\{(?P<value>[^{}]*)\}")
    return {match.group("name"): match.group("value") for match in pattern.finditer(text)}


def controlled_blocks(text: str) -> list[str]:
    return re.findall(
        r"% CLAIM-CONTROLLED-BEGIN\s*(.*?)% CLAIM-CONTROLLED-END",
        text,
        flags=re.DOTALL,
    )


def raw_numbers_in_controlled_block(block: str) -> list[str]:
    cleaned = re.sub(r"(?<!\\)%.*", "", block)
    cleaned = re.sub(r"\\(?:cite|ref|pageref|label)\{[^{}]*\}", "", cleaned)
    cleaned = re.sub(r"\\RThree[A-Za-z]+", "", cleaned)
    cleaned = re.sub(r"\\(?:textit|textbf|emph)\{([^{}]*)\}", r"\1", cleaned)
    return re.findall(r"(?<![A-Za-z\\])[-+]?\d+(?:\.\d+)?", cleaned)


def manuscript_errors(
    manuscript: str,
    expected_macros: dict[str, str],
    observed_macros: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if "\\input{" in manuscript or "\\include{" in manuscript:
        errors.append("Manuscript must be self-contained and may not load auxiliary TeX content")
    for label in REQUIRED_TABLE_LABELS:
        if f"\\label{{{label}}}" not in manuscript:
            errors.append(f"Self-contained manuscript is missing frozen table label: {label}")

    if observed_macros != expected_macros:
        missing = sorted(set(expected_macros) - set(observed_macros))
        extra = sorted(set(observed_macros) - set(expected_macros))
        changed = sorted(
            name
            for name in set(expected_macros) & set(observed_macros)
            if expected_macros[name] != observed_macros[name]
        )
        errors.append(
            f"Generated macro mismatch; missing={missing}, extra={extra}, changed={changed}"
        )

    used_macros = set(re.findall(r"\\(RThree[A-Za-z]+)", manuscript))
    undefined = sorted(used_macros - set(observed_macros))
    if undefined:
        errors.append(f"Undefined round-2 macros used by manuscript: {undefined}")

    for phrase in FORBIDDEN_STALE_TEXT:
        if phrase.lower() in manuscript.lower():
            errors.append(f"Stale or overbroad claim remains: {phrase!r}")

    blocks = controlled_blocks(manuscript)
    if not blocks:
        errors.append("No CLAIM-CONTROLLED manuscript blocks were found")
    for index, block in enumerate(blocks, start=1):
        raw_numbers = raw_numbers_in_controlled_block(block)
        if raw_numbers:
            errors.append(
                f"Raw numerical literals in CLAIM-CONTROLLED block {index}: {raw_numbers}"
            )

    if "\\DIFadd" in manuscript or "\\DIFdel" in manuscript:
        errors.append("Clean manuscript contains latexdiff markup")
    if re.search(r"(?:sk-or-|OPENROUTER_API_KEY\s*=)", manuscript, flags=re.I):
        errors.append("Possible API secret or secret variable assignment in manuscript")

    author_positions = [
        manuscript.find("\\author[1,2,*]{Tunc Asuroglu}"),
        manuscript.find('\\author[3]{As{\\i}m Sinan Y\\"uksel}'),
        manuscript.find("\\author[4]{Muhammed Abdulhamid Karabiyik}"),
        manuscript.find("\\author[5]{Bahaeddin Turkoglu}"),
    ]
    if any(position < 0 for position in author_positions) or author_positions != sorted(
        author_positions
    ):
        errors.append("Author order or affiliation numbering differs from the requested order")
    return errors


def main() -> None:
    args = parse_args()
    manuscript_path = args.manuscript
    if not manuscript_path.is_absolute():
        manuscript_path = PROJECT_ROOT / manuscript_path
    freeze_manifest_path = FREEZE_ROOT / "freeze_manifest.json"
    claim_path = FREEZE_ROOT / "round2_claims.json"
    manifest = read_json(freeze_manifest_path)
    bundle = read_json(claim_path)
    expected_macros = {str(name): str(value) for name, value in bundle["manuscript_macros"].items()}
    manuscript = manuscript_path.read_text(encoding="utf-8")
    observed_macros = parse_macro_file(manuscript)
    errors = verify_hashes(manifest)
    errors.extend(manuscript_errors(manuscript, expected_macros, observed_macros))
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "manuscript": str(manuscript_path.relative_to(PROJECT_ROOT)),
        "manuscript_sha256": sha256_file(manuscript_path),
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "claim_bundle_sha256": sha256_file(claim_path),
        "controlled_blocks": len(controlled_blocks(manuscript)),
        "round2_macros_defined": len(observed_macros),
        "errors": errors,
    }
    if not args.no_write_audit:
        AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("Manuscript claim audit failed:\n- " + "\n- ".join(errors))
    print(
        f"PASS: {len(observed_macros)} round-2 macros, "
        f"{audit['controlled_blocks']} controlled blocks, all frozen hashes verified"
    )


if __name__ == "__main__":
    main()
