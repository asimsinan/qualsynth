#!/usr/bin/env python3
"""Render the Reviewer 3 response and cover letter from their Markdown sources."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SREPORT = PROJECT_ROOT / "sreport"
DOCUMENTS = [
    (
        SREPORT / "response-to-reviewers.md",
        SREPORT / "response-to-reviewers.pdf",
        "0.75in",
        "10pt",
        "1.0",
    ),
    (SREPORT / "cover-letter.md", SREPORT / "cover-letter.pdf", "0.75in", "10pt", "1.0"),
]


def render(
    source: Path,
    output: Path,
    margin: str,
    fontsize: str,
    linestretch: str,
    pandoc: str,
    xelatex: str,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    subprocess.run(
        [
            pandoc,
            str(source),
            "--from=markdown",
            "--standalone",
            f"--pdf-engine={xelatex}",
            f"--variable=geometry:margin={margin}",
            f"--variable=fontsize:{fontsize}",
            f"--variable=linestretch:{linestretch}",
            "--variable=colorlinks:true",
            "--variable=linkcolor:blue",
            "--variable=urlcolor:blue",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"PDF was not created: {output}")


def main() -> None:
    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    if pandoc is None or xelatex is None:
        raise RuntimeError("pandoc and xelatex are required")
    for source, output, margin, fontsize, linestretch in DOCUMENTS:
        render(source, output, margin, fontsize, linestretch, pandoc, xelatex)
        print(f"Wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
