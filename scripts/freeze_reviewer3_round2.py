#!/usr/bin/env python3
"""Freeze the Reviewer 3 round-2 manuscript baseline and provenance manifest.

The current clean manuscript already contains the requested author reorder.  The
reviewer-received baseline is reconstructed by reverting only that known author
and affiliation block.  No scientific content is removed or rewritten.

The script is intentionally conservative:

* an existing, matching baseline is left untouched;
* an existing, conflicting baseline aborts the run;
* the provenance manifest is immutable once written;
* secrets and environment-variable values are never recorded.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_MANUSCRIPT = PROJECT_ROOT / "sreport" / "main.tex"
BASELINE_MANUSCRIPT = PROJECT_ROOT / "sreport" / "main_round1_received.tex"
ROUND2_ROOT = PROJECT_ROOT / "results" / "reviewer_revision" / "reviewer3_round2"
MANIFEST_PATH = ROUND2_ROOT / "provenance_manifest.json"
PLAN_PATH = PROJECT_ROOT / "docs" / "REVIEWER3_REVISION_PLAN.md"


CURRENT_AUTHOR_BLOCK = r'''\author[1,2,*]{Tunc Asuroglu}
\author[3]{As{\i}m Sinan Y\"uksel}
\author[4]{Muhammed Abdulhamid Karabiyik}
\author[5]{Bahaeddin Turkoglu}

\affil[1]{Faculty of Medicine and Health Technology, Tampere University, Tampere, Finland}
\affil[2]{VTT Technical Research Centre of Finland, Tampere, Finland}
\affil[3]{Department of Computer Engineering, S\"uleyman Demirel University, Isparta 32200, Turkey}
\affil[4]{Department of Software Engineering, Konya Technical University, Konya, T\"urkiye}
\affil[5]{Department of Artificial Intelligence and Data Engineering, Ankara University, Ankara, T\"urkiye}
\affil[*]{tunc.asuroglu@tuni.fi}'''


REVIEWER_RECEIVED_AUTHOR_BLOCK = r'''\author[1]{As{\i}m Sinan Y\"uksel}
\author[2]{Muhammed Abdulhamid Karabiyik}
\author[3]{Bahaeddin Turkoglu}
\author[4,5,*]{Tunc Asuroglu}

\affil[1]{Department of Computer Engineering, S\"uleyman Demirel University, Isparta 32200, Turkey}
\affil[2]{Department of Software Engineering, Konya Technical University, Konya, T\"urkiye}
\affil[3]{Department of Artificial Intelligence and Data Engineering, Ankara University, Ankara, T\"urkiye}
\affil[4]{Faculty of Medicine and Health Technology, Tampere University, Tampere, Finland}
\affil[5]{VTT Technical Research Centre of Finland, Tampere, Finland}
\affil[*]{tunc.asuroglu@tuni.fi}'''


EVIDENCE_PATHS = [
    Path("reviewer_comments.md"),
    Path("docs/QUALSYNTH_KNOWLEDGE_GRAPH.md"),
    Path("docs/REVIEWER3_REVISION_PLAN.md"),
    Path("results/reviewer_revision/claim_verification_refreshed_tabddpm/bundle_manifest.json"),
    Path("results/reviewer_revision/claim_verification_refreshed_tabddpm/five_method/global_stats_f1.json"),
    Path("results/reviewer_revision/claim_verification_refreshed_tabddpm/five_method/global_stats_roc_auc.json"),
    Path("results/reviewer_revision/claim_verification_refreshed_tabddpm/quality_audit_summary.csv"),
    Path("results/reviewer_revision/ablations/component_3seed/analysis/REPORT.md"),
    Path("results/reviewer_revision/ablations/component_3seed/analysis/component_paired_tests.csv"),
    Path("results/reviewer_revision/high_dim_extension/analysis/REPORT.md"),
    Path("results/reviewer_revision/cost_runtime_manuscript_table.csv"),
]


PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "openai",
    "PyYAML",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_expected_baseline(current_text: str) -> str:
    count = current_text.count(CURRENT_AUTHOR_BLOCK)
    if count != 1:
        raise RuntimeError(
            "Expected the current reordered author block exactly once in sreport/main.tex; "
            f"found {count}. Refusing to infer a reviewer baseline."
        )
    return current_text.replace(
        CURRENT_AUTHOR_BLOCK,
        REVIEWER_RECEIVED_AUTHOR_BLOCK,
        1,
    )


def write_baseline(expected_text: str) -> str:
    if BASELINE_MANUSCRIPT.exists():
        existing = BASELINE_MANUSCRIPT.read_text(encoding="utf-8")
        if existing != expected_text:
            raise RuntimeError(
                f"Existing baseline conflicts with the expected reconstruction: {BASELINE_MANUSCRIPT}"
            )
        return "existing-matching"

    BASELINE_MANUSCRIPT.write_text(expected_text, encoding="utf-8")
    return "created"


def evidence_hashes() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for relative_path in EVIDENCE_PATHS:
        absolute_path = PROJECT_ROOT / relative_path
        rows[str(relative_path)] = {
            "exists": absolute_path.is_file(),
            "size_bytes": absolute_path.stat().st_size if absolute_path.is_file() else 0,
            "sha256": sha256_file(absolute_path) if absolute_path.is_file() else None,
        }
    return rows


def build_manifest(
    current_text: str,
    baseline_text: str,
    baseline_status: str,
) -> dict[str, Any]:
    return {
        "manifest_name": "qualsynth_reviewer3_round2_provenance_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_status": baseline_status,
        "baseline_reconstruction": {
            "source": "sreport/main.tex",
            "operation": "reverted only the known author and affiliation reorder",
            "scientific_content_changed": False,
            "author_reorder_expected_in_latexdiff": True,
        },
        "git": {
            "branch": git_output("branch", "--show-current"),
            "commit": git_output("rev-parse", "HEAD"),
            "status_short": git_output("status", "--short").splitlines(),
        },
        "manuscripts": {
            "reviewer_received_baseline": {
                "path": str(BASELINE_MANUSCRIPT.relative_to(PROJECT_ROOT)),
                "size_bytes": len(baseline_text.encode("utf-8")),
                "sha256": sha256_bytes(baseline_text.encode("utf-8")),
            },
            "clean_at_freeze": {
                "path": str(CURRENT_MANUSCRIPT.relative_to(PROJECT_ROOT)),
                "size_bytes": len(current_text.encode("utf-8")),
                "sha256": sha256_bytes(current_text.encode("utf-8")),
            },
        },
        "plan": {
            "path": str(PLAN_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PLAN_PATH),
        },
        "evidence": evidence_hashes(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
        "secret_policy": "No environment-variable values or API keys are stored in this manifest.",
    }


def main() -> None:
    if not CURRENT_MANUSCRIPT.is_file():
        raise FileNotFoundError(CURRENT_MANUSCRIPT)
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(PLAN_PATH)

    current_text = CURRENT_MANUSCRIPT.read_text(encoding="utf-8")
    baseline_text = build_expected_baseline(current_text)
    baseline_status = write_baseline(baseline_text)

    ROUND2_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(current_text, baseline_text, baseline_status)

    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        existing_sha = existing.get("manuscripts", {}).get("reviewer_received_baseline", {}).get("sha256")
        expected_sha = manifest["manuscripts"]["reviewer_received_baseline"]["sha256"]
        if existing_sha != expected_sha:
            raise RuntimeError(
                f"Existing provenance manifest references a different baseline: {MANIFEST_PATH}"
            )
        print(f"Baseline already frozen: {BASELINE_MANUSCRIPT.relative_to(PROJECT_ROOT)}")
        print(f"Manifest already exists: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
        return

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline {baseline_status}: {BASELINE_MANUSCRIPT.relative_to(PROJECT_ROOT)}")
    print(f"Manifest created: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Baseline SHA-256: {manifest['manuscripts']['reviewer_received_baseline']['sha256']}")


if __name__ == "__main__":
    main()
