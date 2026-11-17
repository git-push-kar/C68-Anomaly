"""Build the manuscript's LaTeX source and PDF from the Markdown original.

The Markdown file stays the single source of truth: the title becomes document metadata, the body is converted by
pandoc with natbib citations against ``references.bib``, and the generated ``.tex`` is sanitised to pure ASCII so it
compiles under both XeTeX (tectonic's engine) and pdfLaTeX. Figures resolve relative to this directory, so the
build runs with ``docs/reports`` as its working directory.

    python scripts/build_paper_latex.py            # .tex and .pdf
    python scripts/build_paper_latex.py --no-pdf   # .tex only

The toolchain is not vendored in git: pandoc and tectonic are expected under ``tools/`` (see the message printed
when either is missing).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
MD = REPORTS / "PAPER_MANUSCRIPT_2026-11-16.md"
TEX = MD.with_suffix(".tex")
PDF = MD.with_suffix(".pdf")
PREAMBLE = REPORTS / "paper_preamble.tex"
BIB = "references.bib"

PANDOC = ROOT / "tools" / "pandoc" / "pandoc-3.6.4" / "pandoc.exe"
TECTONIC = ROOT / "tools" / "tectonic" / "tectonic.exe"

#: Characters the manuscript uses that cannot appear in a portable ASCII .tex file.
CHAR_MAP = {
    "—": "---",
    "–": "--",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "×": r"$\times$",
    "·": r"$\cdot$",
    "≈": r"$\approx$",
    "→": r"$\rightarrow$",
    "κ": r"$\kappa$",
    "§": r"\S",
    "’": "'",
    "‘": "`",
    "“": "``",
    "”": "''",
    "…": r"\dots{}",
    "\u00a0": " ",
}

SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
}


def split_title(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise SystemExit("expected the manuscript to start with a level-1 title")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).lstrip("\n")
    return title, body


def sanitise(tex: str) -> tuple[str, list[str]]:
    """Replace non-ASCII typography with LaTeX equivalents, returning what stayed behind."""
    def superscript(match: re.Match) -> str:
        base, run = match.group(1), match.group(2)
        exponent = "".join(SUPERSCRIPTS[ch] for ch in run)
        return f"{base}$^{{{exponent}}}$"

    tex = re.sub(r"([0-9A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", superscript, tex)
    for char, replacement in CHAR_MAP.items():
        tex = tex.replace(char, replacement)
    # Repository paths are longer than the measure and cannot hyphenate. Keep the typewriter face but allow a break
    # after each slash, so long paths wrap at directory boundaries instead of running into the margin.
    def breakable_path(match: re.Match) -> str:
        body = match.group(1)
        if "/" not in body:
            return match.group(0)
        body = body.replace("/", "/\\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        return r"\texttt{" + body + "}"

    tex = re.sub(r"\\texttt\{([^{}]*)\}", breakable_path, tex)
    leftovers = sorted({ch for ch in tex if ord(ch) > 127})
    return tex, leftovers


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true", help="write the .tex but skip compilation")
    args = parser.parse_args()

    if not PANDOC.exists():
        print(f"pandoc not found at {PANDOC}", file=sys.stderr)
        return 2
    title, body = split_title(MD.read_text(encoding="utf-8"))

    tmp = REPORTS / ".paper_body_tmp.md"
    tmp.write_text(body, encoding="utf-8")
    try:
        result = run([
            str(PANDOC), str(tmp), "-o", str(TEX),
            "--standalone",
            "--natbib",
            f"--bibliography={BIB}",
            f"--include-in-header={PREAMBLE.name}",
            "--metadata", f"title={title}",
            "--metadata", "date=16 November 2026",
            "--variable", "fontsize=10pt",
            "--variable", "geometry:margin=1in",
            "--variable", "colorlinks=true",
            "--variable", "linkcolor=blue",
            "--variable", "urlcolor=blue",
        ], cwd=REPORTS)
    finally:
        tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    tex, leftovers = sanitise(TEX.read_text(encoding="utf-8"))
    TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {TEX.relative_to(ROOT)} ({len(tex.splitlines())} lines)")
    if leftovers:
        print(f"non-ASCII characters left in the .tex: {leftovers}", file=sys.stderr)

    if args.no_pdf:
        return 0
    if not TECTONIC.exists():
        print(f"tectonic not found at {TECTONIC}", file=sys.stderr)
        return 2

    build = run([str(TECTONIC), "-X", "compile", TEX.name], cwd=REPORTS)
    tail = "\n".join((build.stdout + build.stderr).splitlines()[-40:])
    print(tail)
    if build.returncode != 0 or not PDF.exists():
        return build.returncode or 1
    print(f"wrote {PDF.relative_to(ROOT)} ({PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
