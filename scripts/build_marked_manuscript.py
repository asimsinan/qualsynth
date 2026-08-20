#!/usr/bin/env python3
"""Generate the marked manuscript from the frozen reviewer-received baseline."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SREPORT = PROJECT_ROOT / "sreport"
PRIMARY_BASELINE_TEX = SREPORT / "main_round1_received.tex"
PACKAGED_BASELINE_TEX = (
    PROJECT_ROOT / "submission/files/related-file-source/main_round1_received.tex"
)
BASELINE_TEX = (
    PRIMARY_BASELINE_TEX if PRIMARY_BASELINE_TEX.is_file() else PACKAGED_BASELINE_TEX
)
CURRENT_TEX = SREPORT / "main.tex"
DIFF_TEX = SREPORT / "main_marked.tex"
DIFF_PDF = SREPORT / "main_marked.pdf"


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def generate_diff(latexdiff: str) -> str:
    """Return an automatically generated, flattened LaTeX difference."""
    return subprocess.run(
        [
            latexdiff,
            "--type=CFONT",
            "--flatten",
            "--math-markup=off",
            "--disable-citation-markup",
            str(BASELINE_TEX),
            str(CURRENT_TEX),
        ],
        cwd=SREPORT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def generate_flattened_identity(latexdiff: str, source: Path) -> str:
    """Flatten one source without introducing textual changes."""
    return subprocess.run(
        [
            latexdiff,
            "--flatten",
            "--math-markup=off",
            "--disable-citation-markup",
            str(source),
            str(source),
        ],
        cwd=SREPORT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def baseline_float_reference_numbers(source: str) -> dict[str, str]:
    """Return the table/figure numbers assigned by the frozen baseline.

    The reviewer-received manuscript numbers floats in source order.  Reading
    those numbers directly from the frozen source keeps deleted prose legible
    even when latexdiff comments out the corresponding old float and label.
    """
    counters = {"table": 0, "figure": 0}
    references: dict[str, str] = {}
    float_pattern = re.compile(
        r"\\begin\{(table|figure)\*?\}(.*?)\\end\{\1\*?\}",
        flags=re.DOTALL,
    )
    for match in float_pattern.finditer(source):
        float_type, body = match.groups()
        counters[float_type] += 1
        for label in re.findall(r"\\label\{([^{}]+)\}", body):
            references[label] = str(counters[float_type])
    return references


def neutralize_deleted_only_references(diff: str) -> str:
    """Make references in visible deleted prose explicit and non-executable.

    Latexdiff comments structural commands from removed tables, including their
    labels, while references in visible deleted prose still execute. Replace only
    references whose targets exist solely in deleted structure with their
    round-one float number as inert text. This preserves readable redline prose
    without creating misleading links or unresolved-reference warnings.
    """
    deleted = set(
        re.findall(r"%DIFDELCMD\s*<\s*\\label\{([^{}]+)\}", diff)
    )
    uncommented_lines = []
    for line in diff.splitlines():
        uncommented_lines.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
    active = set(re.findall(r"\\label\{([^{}]+)\}", "\n".join(uncommented_lines)))
    missing = sorted(deleted - active)
    baseline_numbers = baseline_float_reference_numbers(
        BASELINE_TEX.read_text(encoding="utf-8")
    )
    for label in missing:
        replacement = baseline_numbers.get(label, "reference removed")
        diff = diff.replace(f"\\ref{{{label}}}", rf"\textnormal{{{replacement}}}")
        diff = diff.replace(
            f"\\pageref{{{label}}}",
            r"\textnormal{[page reference removed]}",
        )
        diff = diff.replace(
            f"\\autoref{{{label}}}",
            rf"\textnormal{{{replacement}}}",
        )
    return diff


def make_float_wrappers_alignment_safe(diff: str) -> str:
    r"""Keep float text markup without alignment-unsafe graphics assignments.

    ``latexdiff --type=CFONT`` redefines its float begin/end sentinels to switch
    ``\includegraphics`` implementations.  When a whole generated table is new,
    an end sentinel can land immediately after ``\begin{tabular}``; its ``\let``
    assignment then precedes ``\toprule`` inside the alignment and triggers a
    ``Misplaced \noalign`` error.  The manuscript does not add or remove figure
    files in this revision, so retaining the original no-op float sentinels is
    both sufficient and deterministic.  Cell-level ``\DIFaddFL``/``\DIFdelFL``
    markup remains intact.
    """
    replacements = {
        r"\DeclareRobustCommand{\DIFaddbeginFL}{\DIFOaddbeginFL \let\includegraphics\DIFaddincludegraphics}":
            r"\DeclareRobustCommand{\DIFaddbeginFL}{\DIFOaddbeginFL}",
        r"\DeclareRobustCommand{\DIFaddendFL}{\DIFOaddendFL \let\includegraphics\DIFOincludegraphics}":
            r"\DeclareRobustCommand{\DIFaddendFL}{\DIFOaddendFL}",
        r"\DeclareRobustCommand{\DIFdelbeginFL}{\DIFOdelbeginFL \let\includegraphics\DIFdelincludegraphics}":
            r"\DeclareRobustCommand{\DIFdelbeginFL}{\DIFOdelbeginFL}",
        r"\DeclareRobustCommand{\DIFdelendFL}{\DIFOaddendFL \let\includegraphics\DIFOincludegraphics}":
            r"\DeclareRobustCommand{\DIFdelendFL}{\DIFOaddendFL}",
    }
    for old, new in replacements.items():
        if old not in diff:
            raise RuntimeError(f"Expected latexdiff float wrapper not found: {old}")
        diff = diff.replace(old, new, 1)

    marker = r"\begin{document}"
    if marker not in diff:
        raise RuntimeError("Generated latexdiff source has no document body")
    preamble, body = diff.split(marker, 1)
    # Even an otherwise empty robust sentinel is non-expandable while TeX scans
    # an alignment, so it cannot sit between ``\begin{tabular}`` and
    # ``\toprule``.  The FL text macros retain every visible cell change.
    body = re.sub(r"\\DIF(?:add|del)(?:begin|end)FL\b\s*", "", body)
    diff = preamble + marker + body
    return diff


def remove_balanced_macro_calls(text: str, macro: str) -> str:
    """Remove every ``macro{...}`` call while respecting nested braces."""
    needle = macro + "{"
    chunks: list[str] = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            chunks.append(text[cursor:])
            break
        chunks.append(text[cursor:start])
        depth = 1
        pos = start + len(needle)
        while pos < len(text) and depth:
            char = text[pos]
            backslashes = 0
            probe = pos - 1
            while probe >= 0 and text[probe] == "\\":
                backslashes += 1
                probe -= 1
            escaped = backslashes % 2 == 1
            if not escaped:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            pos += 1
        if depth:
            raise RuntimeError(f"Unbalanced {macro} call in latexdiff output")
        cursor = pos
    return "".join(chunks)


def suppress_deleted_table_cells(diff: str) -> str:
    r"""Retain clean table alignment and added-cell markup in schema rewrites.

    When an old and new table have different column counts, ``latexdiff``
    comments the deleted ``&`` and row terminators but leaves the corresponding
    ``\DIFdelFL`` cell text active.  Those fragments form an unterminated row in
    the new alignment.  Removing only deleted cell payloads yields the exact
    clean-table structure, while every added or changed replacement remains
    visibly marked by ``\DIFaddFL``.
    """
    begin = r"\begin{tabular}"
    end = r"\end{tabular}"
    chunks: list[str] = []
    cursor = 0
    while True:
        start = diff.find(begin, cursor)
        if start < 0:
            chunks.append(diff[cursor:])
            break
        stop = diff.find(end, start)
        if stop < 0:
            raise RuntimeError("Unclosed tabular environment in latexdiff output")
        stop += len(end)
        chunks.append(diff[cursor:start])
        table = remove_balanced_macro_calls(diff[start:stop], r"\DIFdelFL")
        table = remove_balanced_macro_calls(table, r"\DIFdel")
        chunks.append(table)
        cursor = stop
    return "".join(chunks)


def use_clean_bibliography(diff: str, current: str) -> str:
    """Use the clean bibliography to avoid invalid diffs inside TeX accents."""
    begin = r"\begin{thebibliography}"
    end = r"\end{thebibliography}"

    def bounds(text: str) -> tuple[int, int]:
        start = text.find(begin)
        if start < 0:
            raise RuntimeError("Bibliography start not found")
        stop = text.find(end, start)
        if stop < 0:
            raise RuntimeError("Bibliography end not found")
        return start, stop + len(end)

    diff_start, diff_stop = bounds(diff)
    current_start, current_stop = bounds(current)
    return diff[:diff_start] + current[current_start:current_stop] + diff[diff_stop:]


def active_table_spans(text: str) -> list[tuple[int, int]]:
    """Return active, non-commented table-environment character spans."""
    begin = r"\begin{table}"
    end = r"\end{table}"
    spans: list[tuple[int, int]] = []
    open_start: int | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        begin_at = code.find(begin)
        end_at = code.find(end)
        if begin_at >= 0:
            if open_start is not None:
                raise RuntimeError("Nested active table environments are unsupported")
            open_start = offset + begin_at
        if end_at >= 0:
            if open_start is None:
                raise RuntimeError("Active table end has no matching start")
            spans.append((open_start, offset + end_at + len(end)))
            open_start = None
        offset += len(line)
    if open_start is not None:
        raise RuntimeError("Active table environment is unclosed")
    return spans


def table_label(block: str) -> str:
    """Return a table label, or a stable empty marker when absent."""
    match = re.search(r"\\label\{([^{}]+)\}", block)
    return match.group(1) if match else ""


def visibly_mark_table(block: str) -> str:
    """Colour a changed clean table without altering its alignment structure."""
    first_newline = block.find("\n")
    if first_newline < 0:
        raise RuntimeError("Unexpected single-line table block")
    return block[: first_newline + 1] + r"\color{blue}\sffamily" + "\n" + block[first_newline + 1 :]


def replace_tables_with_clean_marked_blocks(
    diff: str,
    current_flat: str,
    baseline_flat: str,
) -> str:
    r"""Replace schema-mangled table diffs with clean, visibly marked tables.

    ``latexdiff`` cannot preserve TeX alignment when a replacement table has a
    different number of columns.  We still generate the document from the
    frozen source diff, but substitute each active table by its flattened clean
    counterpart.  A table is coloured blue when its labelled block differs from
    the same labelled block in the reviewer-received baseline; byte-identical
    tables remain black.
    """
    diff_spans = active_table_spans(diff)
    current_spans = active_table_spans(current_flat)
    baseline_spans = active_table_spans(baseline_flat)
    if len(diff_spans) != len(current_spans):
        raise RuntimeError(
            "Current table count does not match latexdiff output: "
            f"{len(current_spans)} != {len(diff_spans)}"
        )

    current_blocks = [current_flat[start:stop] for start, stop in current_spans]
    baseline_blocks = [baseline_flat[start:stop] for start, stop in baseline_spans]
    baseline_by_label = {
        label: block
        for block in baseline_blocks
        if (label := table_label(block))
    }

    replacements: list[str] = []
    for block in current_blocks:
        label = table_label(block)
        baseline = baseline_by_label.get(label) if label else None
        replacements.append(block if baseline == block else visibly_mark_table(block))

    for (start, stop), replacement in reversed(list(zip(diff_spans, replacements))):
        diff = diff[:start] + replacement + diff[stop:]
    return diff


def replace_related_work_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Show a wholesale Related Work rewrite as one readable blue block.

    Paragraph-level latexdiff markup becomes illegible when an entire section is
    reorganised: long deleted and inserted passages can occupy the same lines.
    Retain the current flattened section and colour its body blue, matching the
    established treatment of structurally replaced tables. The unchanged
    section heading remains black and the colour is reset before Methods.
    """
    section_start = r"\section*{Related Work}"
    next_section = r"\section*{Methods}"

    def bounds(text: str) -> tuple[int, int]:
        start = text.find(section_start)
        if start < 0:
            raise RuntimeError("Related Work section not found")
        stop = text.find(next_section, start)
        if stop < 0:
            raise RuntimeError("Methods section not found after Related Work")
        return start, stop

    diff_start, diff_stop = bounds(diff)
    current_start, current_stop = bounds(current_flat)
    block = current_flat[current_start:current_stop]
    table_begin = r"\begin{table}[H]"
    if table_begin not in block:
        raise RuntimeError("Architectural comparison table not found in Related Work")
    block = block.replace(
        table_begin,
        table_begin + "\n" + r"\color{blue}\sffamily",
        1,
    )
    label = r"\label{sec:related}"
    label_end = block.find(label)
    if label_end < 0:
        raise RuntimeError("Related Work label not found")
    label_end += len(label)
    marked = (
        block[:label_end]
        + "\n\n\\color{blue}\n"
        + block[label_end:]
        + "\n\\color{black}\n\n"
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def replace_introduction_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Show the substantially rewritten Introduction as a readable blue block."""
    section_start = r"\section*{Introduction}"
    next_section = r"\section*{Related Work}"

    def bounds(text: str) -> tuple[int, int]:
        start = text.find(section_start)
        if start < 0:
            raise RuntimeError("Introduction section not found")
        stop = text.find(next_section, start)
        if stop < 0:
            raise RuntimeError("Related Work section not found after Introduction")
        return start, stop

    diff_start, diff_stop = bounds(diff)
    current_start, current_stop = bounds(current_flat)
    block = current_flat[current_start:current_stop]
    label = r"\label{sec:intro}"
    label_end = block.find(label)
    if label_end < 0:
        raise RuntimeError("Introduction label not found")
    label_end += len(label)
    marked = (
        block[:label_end]
        + "\n\n\\color{blue}\n"
        + block[label_end:]
        + "\n\\color{black}\n\n"
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def replace_abstract_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Mark the revised abstract visibly even though it precedes the document body.

    The Scientific Reports class places the abstract before ``\begin{document}``,
    which causes latexdiff to treat its text as preamble material and emit only
    non-visible diff comments. Replacing that block with the current abstract in
    blue preserves a readable marked manuscript and makes the revision explicit.
    """
    begin = r"\begin{abstract}"
    end = r"\end{abstract}"

    def bounds(text: str) -> tuple[int, int]:
        start = text.find(begin)
        if start < 0:
            raise RuntimeError("Abstract start not found")
        stop = text.find(end, start)
        if stop < 0:
            raise RuntimeError("Abstract end not found")
        return start, stop + len(end)

    diff_start, diff_stop = bounds(diff)
    current_start, current_stop = bounds(current_flat)
    block = current_flat[current_start:current_stop]
    body_start = block.find("\n", len(begin))
    body_stop = block.rfind(end)
    if body_start < 0 or body_stop < 0:
        raise RuntimeError("Could not isolate abstract text")
    marked = (
        block[: body_start + 1]
        + "{\\color{blue}\\sffamily\n"
        + block[body_start + 1 : body_stop]
        + "}\n"
        + block[body_stop:]
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def replace_problem_formulation_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Show the rewritten Problem Formulation as one readable blue block."""
    subsection_start = r"\subsection*{Problem Formulation}"
    next_subsection = r"\subsection*{System Architecture}"

    def bounds(text: str) -> tuple[int, int]:
        start = text.find(subsection_start)
        if start < 0:
            raise RuntimeError("Problem Formulation subsection not found")
        stop = text.find(next_subsection, start)
        if stop < 0:
            raise RuntimeError(
                "System Architecture subsection not found after Problem Formulation"
            )
        return start, stop

    diff_start, diff_stop = bounds(diff)
    current_start, current_stop = bounds(current_flat)
    block = current_flat[current_start:current_stop]
    heading_end = block.find("\n", len(subsection_start))
    if heading_end < 0:
        raise RuntimeError("Problem Formulation heading terminator not found")
    marked = (
        block[: heading_end + 1]
        + "\n\\color{blue}\n"
        + block[heading_end + 1 :]
        + "\n\\color{black}\n\n"
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def replace_early_methods_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Show the substantially rewritten early Methods subsections in blue."""
    subsection_start = r"\subsection*{System Architecture}"
    next_subsection = r"\subsection*{Generation settings and reproducibility}"

    def current_bounds(text: str) -> tuple[int, int]:
        start = text.find(subsection_start)
        if start < 0:
            raise RuntimeError("System Architecture subsection not found")
        stop = text.find(next_subsection, start)
        if stop < 0:
            raise RuntimeError(
                "Generation settings subsection not found after System Architecture"
            )
        return start, stop

    current_start, current_stop = current_bounds(current_flat)
    diff_start = diff.find(subsection_start)
    reproducibility_label = r"\label{sec:reproducibility}"
    label_position = diff.find(reproducibility_label, diff_start)
    if diff_start < 0 or label_position < 0:
        raise RuntimeError("Could not locate early Methods block in marked text")
    diff_stop = diff.rfind(r"\subsection*", diff_start, label_position)
    if diff_stop < 0:
        raise RuntimeError("Could not locate generation settings heading in marked text")
    block = current_flat[current_start:current_stop]
    heading_end = block.find("\n", len(subsection_start))
    if heading_end < 0:
        raise RuntimeError("System Architecture heading terminator not found")
    marked = (
        block[: heading_end + 1]
        + "\n\\color{blue}\n"
        + block[heading_end + 1 :]
        + "\n\\color{black}\n\n"
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def replace_implementation_details_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Show the substantially rewritten reproducibility subsection in blue."""
    subsection_start = r"\subsection*{Generation settings and reproducibility}"
    next_subsection = r"\subsection*{Sensitivity and robustness analyses}"

    def current_bounds(text: str) -> tuple[int, int]:
        start = text.find(subsection_start)
        if start < 0:
            raise RuntimeError("Generation settings subsection not found")
        stop = text.find(next_subsection, start)
        if stop < 0:
            raise RuntimeError(
                "Sensitivity analyses subsection not found after generation settings"
            )
        return start, stop

    current_start, current_stop = current_bounds(current_flat)
    reproducibility_label = r"\label{sec:reproducibility}"
    reproducibility_position = diff.find(reproducibility_label)
    diff_start = diff.rfind(r"\subsection*", 0, reproducibility_position)
    sensitivity_label = r"\label{sec:sensitivity_protocols}"
    label_position = diff.find(sensitivity_label, diff_start)
    if diff_start < 0 or label_position < 0:
        raise RuntimeError("Could not locate generation settings in marked text")
    diff_stop = diff.rfind(r"\DIFaddbegin", diff_start, label_position)
    if diff_stop < 0:
        diff_stop = diff.rfind(r"\subsection*", diff_start, label_position)
    if diff_stop < 0:
        raise RuntimeError("Could not locate sensitivity heading in marked text")
    block = current_flat[current_start:current_stop]
    for table_position in ("H", "ht!"):
        block = block.replace(
            rf"\begin{{table}}[{table_position}]",
            rf"\begin{{table}}[{table_position}]" + "\n" + r"\color{blue}\sffamily",
        )
    heading_end = block.find("\n", len(subsection_start))
    if heading_end < 0:
        raise RuntimeError("Generation settings heading terminator not found")
    marked = (
        block[: heading_end + 1]
        + "\n\\color{blue}\n"
        + block[heading_end + 1 :]
        + "\n\\color{black}\n\n"
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def replace_results_with_clean_marked_block(
    diff: str,
    current_flat: str,
) -> str:
    r"""Show the reorganised Results section as one readable blue block."""
    section_start = r"\section*{Results}"
    next_section = r"\section*{Discussion}"

    def bounds(text: str) -> tuple[int, int]:
        start = text.find(section_start)
        if start < 0:
            raise RuntimeError("Results section not found")
        stop = text.find(next_section, start)
        if stop < 0:
            raise RuntimeError("Discussion section not found after Results")
        return start, stop

    diff_start, diff_stop = bounds(diff)
    current_start, current_stop = bounds(current_flat)
    block = current_flat[current_start:current_stop]
    for table_position in ("H", "ht!", "ht", "t"):
        block = block.replace(
            rf"\begin{{table}}[{table_position}]",
            rf"\begin{{table}}[{table_position}]"
            + "\n"
            + r"\color{blue}\sffamily",
        )
    label = r"\label{sec:experiments}"
    label_end = block.find(label)
    if label_end < 0:
        raise RuntimeError("Results label not found")
    label_end += len(label)
    marked = (
        block[:label_end]
        + "\n\n\\color{blue}\n"
        + block[label_end:]
        + "\n\\color{black}\n\n"
    )
    return diff[:diff_start] + marked + diff[diff_stop:]


def main() -> None:
    for path in [BASELINE_TEX, CURRENT_TEX]:
        if not path.is_file():
            raise FileNotFoundError(path)
    latexdiff = shutil.which("latexdiff")
    if latexdiff is None:
        raise RuntimeError("latexdiff is not installed or not on PATH")

    diff = neutralize_deleted_only_references(generate_diff(latexdiff))
    diff = make_float_wrappers_alignment_safe(diff)
    diff = suppress_deleted_table_cells(diff)
    diff = use_clean_bibliography(diff, CURRENT_TEX.read_text(encoding="utf-8"))
    current_flat = generate_flattened_identity(latexdiff, CURRENT_TEX)
    baseline_flat = generate_flattened_identity(latexdiff, BASELINE_TEX)
    diff = replace_tables_with_clean_marked_blocks(
        diff,
        current_flat,
        baseline_flat,
    )
    diff = replace_abstract_with_clean_marked_block(diff, current_flat)
    diff = replace_introduction_with_clean_marked_block(diff, current_flat)
    diff = replace_related_work_with_clean_marked_block(diff, current_flat)
    diff = replace_problem_formulation_with_clean_marked_block(diff, current_flat)
    diff = replace_early_methods_with_clean_marked_block(diff, current_flat)
    diff = replace_implementation_details_with_clean_marked_block(diff, current_flat)
    diff = replace_results_with_clean_marked_block(diff, current_flat)
    # Some TeX distributions duplicate this class-level declaration. Removing
    # the duplicate is a deterministic build normalization, not a manual edit.
    diff = diff.replace("\\RequirePackage[utf8]{inputenc}\n", "")
    DIFF_TEX.write_text(diff, encoding="utf-8")

    for suffix in [".aux", ".log", ".out"]:
        artifact = SREPORT / f"main_marked{suffix}"
        if artifact.exists():
            artifact.unlink()
    run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", DIFF_TEX.name],
        SREPORT,
    )
    run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", DIFF_TEX.name],
        SREPORT,
    )
    if not DIFF_PDF.is_file():
        raise RuntimeError(f"Marked PDF was not created: {DIFF_PDF}")
    print(f"Baseline: {BASELINE_TEX.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {DIFF_TEX.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {DIFF_PDF.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
