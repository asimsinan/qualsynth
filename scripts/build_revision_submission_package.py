#!/usr/bin/env python3
"""Build the local Scientific Reports revision submission package.

The package is intentionally local: DOI reservation/deposition must be completed
through an external archive service such as Zenodo after reviewing the zip.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
PACKAGE_DIR = SUBMISSION_DIR / "files"
ARCHIVE_NAME = "qualsynth_reviewer_revision_artifacts.zip"

UPLOAD_FILES = [
    ("manuscript", PROJECT_ROOT / "sreport" / "main.pdf", "clean-manuscript.pdf"),
    ("manuscript-source", PROJECT_ROOT / "sreport" / "main.tex", "main.tex"),
    (
        "manuscript-source",
        PROJECT_ROOT / "sreport" / "architecture_diagram.png",
        "architecture_diagram.png",
    ),
    ("manuscript-source", PROJECT_ROOT / "sreport" / "wlscirep.cls", "wlscirep.cls"),
    ("manuscript-source", PROJECT_ROOT / "sreport" / "jabbrv.sty", "jabbrv.sty"),
    ("manuscript-source", PROJECT_ROOT / "sreport" / "jabbrv-ltwa-all.ldf", "jabbrv-ltwa-all.ldf"),
    ("manuscript-source", PROJECT_ROOT / "sreport" / "jabbrv-ltwa-en.ldf", "jabbrv-ltwa-en.ldf"),
    (
        "response",
        PROJECT_ROOT / "sreport" / "response-to-reviewers.pdf",
        "response-to-reviewers.pdf",
    ),
    (
        "response-source",
        PROJECT_ROOT / "sreport" / "response-to-reviewers.md",
        "response-to-reviewers.md",
    ),
    ("cover-letter", PROJECT_ROOT / "sreport" / "cover-letter.pdf", "cover-letter.pdf"),
    ("cover-letter-source", PROJECT_ROOT / "sreport" / "cover-letter.md", "cover-letter.md"),
    ("related-file", PROJECT_ROOT / "README.md", "README.md"),
    ("related-file", PROJECT_ROOT / "sreport" / "main_marked.pdf", "marked-up-manuscript.pdf"),
    ("related-file-source", PROJECT_ROOT / "sreport" / "main_marked.tex", "main_marked.tex"),
    (
        "related-file-source",
        PROJECT_ROOT / "sreport" / "main_round1_received.tex",
        "main_round1_received.tex",
    ),
]

ARCHIVE_PATHS = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "configs",
    "data/splits",
    "docs",
    "reviewer_comments.md",
    "scripts",
    "src",
    "tests",
    "sreport/main.tex",
    "sreport/main.pdf",
    "sreport/main_round1_received.tex",
    "sreport/main_marked.tex",
    "sreport/main_marked.pdf",
    "sreport/response-to-reviewers.md",
    "sreport/response-to-reviewers.pdf",
    "sreport/cover-letter.md",
    "sreport/cover-letter.pdf",
    "sreport/architecture_diagram.png",
    "sreport/wlscirep.cls",
    "sreport/jabbrv.sty",
    "sreport/jabbrv-ltwa-all.ldf",
    "sreport/jabbrv-ltwa-en.ldf",
    "results/reviewer_revision",
]

EXCLUDE_NAMES = {
    ".DS_Store",
}

EXCLUDE_SUFFIXES = {
    ".aux",
    ".log",
    ".out",
    ".toc",
}

SECRET_PATTERNS = {
    "OpenRouter key": re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "OpenAI-style key": re.compile(rb"\bsk-[A-Za-z0-9_-]{32,}\b"),
    "API-key assignment": re.compile(
        rb"(?im)^\s*(?:OPENROUTER_API_KEY|OPENAI_API_KEY)\s*=\s*[^\s\"']{16,}\s*$"
    ),
    "JSON API-key field": re.compile(rb'(?i)"api_key"\s*:\s*"[^"\r\n]{16,}"'),
}

SECRET_PLACEHOLDER_MARKERS = (
    b"your-openai-api-key",
    b"your-openrouter-api-key",
    b"replace-me",
    b"example",
    b"placeholder",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_archive_files() -> list[Path]:
    files: list[Path] = []
    for rel in ARCHIVE_PATHS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            if candidate.name in EXCLUDE_NAMES or candidate.suffix in EXCLUDE_SUFFIXES:
                continue
            files.append(candidate)
    return sorted(set(files), key=lambda p: str(p.relative_to(PROJECT_ROOT)))


def scan_for_secrets(files: list[Path]) -> None:
    """Fail closed without printing any matched credential value."""
    findings: list[str] = []
    for path in files:
        content = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(content):
                candidate = match.group(0).lower()
                if any(marker in candidate for marker in SECRET_PLACEHOLDER_MARKERS):
                    continue
                findings.append(f"{path.relative_to(PROJECT_ROOT)} ({label})")
                break
    if findings:
        raise RuntimeError(
            "Submission package refused: possible credentials detected in:\n- "
            + "\n- ".join(sorted(set(findings)))
        )


def ensure_clean_submission_dir() -> None:
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def copy_upload_files() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot, src, name in UPLOAD_FILES:
        dst_dir = PACKAGE_DIR / slot
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        exists = src.exists()
        if exists:
            shutil.copy2(src, dst)
        rows.append(
            {
                "slot": slot,
                "source": str(src.relative_to(PROJECT_ROOT)),
                "package_path": str(dst.relative_to(PROJECT_ROOT)) if exists else "",
                "exists": exists,
                "size_bytes": dst.stat().st_size if exists else 0,
                "sha256": sha256_file(dst) if exists else "",
            }
        )
    return rows


def build_archive() -> dict[str, object]:
    archive_path = PACKAGE_DIR / "supplementary" / ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_files = iter_archive_files()
    scan_for_secrets(archive_files)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in archive_files:
            zf.write(file_path, file_path.relative_to(PROJECT_ROOT))
    return {
        "slot": "supplementary-archive",
        "source": "local revision artifacts",
        "package_path": str(archive_path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "file_count": len(archive_files),
    }


def write_files_to_upload(rows: list[dict[str, object]], archive_row: dict[str, object]) -> None:
    all_rows = rows + [archive_row]
    lines = [
        "# Files to Upload",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Upload Manifest",
        "",
        "| Slot | File | Source | SHA-256 | Notes |",
        "|---|---|---|---|---|",
    ]
    notes_by_slot = {
        "manuscript": "Clean manuscript PDF; no tracked changes.",
        "manuscript-source": "LaTeX source/support file for clean manuscript.",
        "response": "Point-by-point response PDF.",
        "response-source": "Editable response source.",
        "cover-letter": "Cover letter PDF with summary of changes.",
        "cover-letter-source": "Editable cover letter source.",
        "related-file": "Updated project/reproducibility overview.",
        "related-file-source": "Editable source for optional related file.",
        "supplementary-archive": "Local archive ready for external DOI deposition.",
    }
    for row in all_rows:
        if not row["exists"]:
            notes = "Missing; verify before upload."
        elif str(row["package_path"]).endswith("marked-up-manuscript.pdf"):
            notes = "Optional marked-up manuscript; upload only as a related file."
        else:
            notes = notes_by_slot.get(str(row["slot"]), "")
        lines.append(
            f"| {row['slot']} | `{row['package_path']}` | `{row['source']}` | "
            f"`{row['sha256']}` | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Marked-Up Manuscript",
            "",
            "`submission/files/related-file/marked-up-manuscript.pdf` is a separate marked-up "
            "manuscript for optional upload as a related file only. It was generated automatically "
            "with `latexdiff` against the frozen reviewer-received source "
            "`sreport/main_round1_received.tex`. Do not upload this file as the clean manuscript.",
            "",
            "## DOI Status",
            "",
            "No DOI is embedded by this local build step. Deposit "
            "`submission/files/supplementary/qualsynth_reviewer_revision_artifacts.zip` in the "
            "chosen repository (for example Zenodo), reserve/publish the DOI there, then update "
            "the manuscript, README, response letter, and cover letter with the final DOI before "
            "submission.",
        ]
    )
    (SUBMISSION_DIR / "files_to_upload.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_machine_manifest(rows: list[dict[str, object]], archive_row: dict[str, object]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "doi_status": "pending_external_deposition",
        "marked_up_manuscript_status": "produced_from_frozen_reviewer_received_baseline",
        "files": rows + [archive_row],
    }
    (SUBMISSION_DIR / "submission_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_clean_submission_dir()
    rows = copy_upload_files()
    archive_row = build_archive()
    write_files_to_upload(rows, archive_row)
    write_machine_manifest(rows, archive_row)
    print(f"Wrote revision submission package to {SUBMISSION_DIR}")
    print(f"Archive file count: {archive_row['file_count']}")
    print(f"Archive: {archive_row['package_path']}")


if __name__ == "__main__":
    main()
