"""Check every number, figure and PDF artefact the manuscript depends on.

Each row prints PASS/FAIL against the artefact that produced it, so a mismatch is traceable to one file rather than
to "the paper". Run after ``scripts/build_paper_latex.py``:

    python scripts/verify_paper_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
MD = REPORTS / "PAPER_MANUSCRIPT_2026-11-16.md"
TEX = MD.with_suffix(".tex")
PDF = MD.with_suffix(".pdf")
FIGDIR = REPORTS / "figures"

FAILS: list[str] = []
CHECKS = 0


def check(label: str, actual, expected) -> None:
    global CHECKS
    CHECKS += 1
    ok = actual == expected
    if not ok:
        FAILS.append(f"{label}: actual {actual!r} != expected {expected!r}")
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}")


def close(label: str, actual: float, expected: float, tol: float = 5e-4) -> None:
    global CHECKS
    CHECKS += 1
    ok = abs(float(actual) - float(expected)) <= tol
    if not ok:
        FAILS.append(f"{label}: actual {actual} != expected {expected}")
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual}")


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    cells = load("outputs/overnight_2026-11-05/n4_cells.json")
    summary = cells["summary"]
    test = load("outputs/analysis/phase3a/d33/test_contrast.json")
    figdata = json.loads((FIGDIR / "figure_data.json").read_text(encoding="utf-8"))
    contrasts = load("outputs/analysis/phase3a/hybrid/vocabfix/classifier_contrasts.json")
    kappa = load("docs/judging/confirm/preference_2026-11-16_kappa.json")
    text = MD.read_text(encoding="utf-8")

    print("\n-- 6.1 fit-set cells (n4_cells.json) --")
    for cell, numeric, anchor in (("A", 0, 520), ("B", 30, 188), ("C", 0, 520), ("M", 30, 191), ("D", 0, 262)):
        check(f"{cell} numeric defects", summary[cell]["numeric"], numeric)
        check(f"{cell} numeric rate", summary[cell]["numeric_rate"], round(numeric / 520, 4))
        check(f"{cell} anchor exact", summary[cell]["anchor_exact"], anchor)
    check("B grounding", summary["B"]["grounding"], 33)
    check("B instrument role", summary["B"]["tag_role"], 57)
    check("C grounding", summary["C"]["grounding"], 1)
    check("M grounding", summary["M"]["grounding"], 32)
    check("M instrument role", summary["M"]["tag_role"], 43)
    check("seam: C leaks B numeric", cells["seam_certification"]["C_b_numeric_leaked"], 0)
    check("seam: M leaks A numeric", cells["seam_certification"]["M_a_numeric_leaked"], 0)
    check("D fallback rows", cells["seam_certification"]["D_fallback_rows"], 114)
    close("D fallback rate", 114 / 520, 0.219)
    check("McNemar b-only", cells["mcnemar"]["b_only"], 30)
    check("McNemar a-only", cells["mcnemar"]["a_only"], 0)

    print("\n-- 6.2 confirmatory pass (test_contrast.json) --")
    counts, rates, anchor = test["counts"], test["rates"], test["anchor_status"]
    check("rows", test["rows"], 3168)
    check("A numeric", counts["A"]["numeric"], 0)
    check("B numeric", counts["B"]["numeric"], 40)
    close("B numeric rate", rates["B"]["numeric"], 0.0126)
    check("A grounding", counts["A"]["grounding"], 250)
    check("B grounding", counts["B"]["grounding"], 325)
    close("B grounding rate", rates["B"]["grounding"], 0.1026)
    check("B contradiction", counts["B"]["contradiction"], 5)
    check("B instrument role", counts["B"]["tag_role"], 458)
    close("B role rate", rates["B"]["tag_role"], 0.1446)
    check("A anchor exact", anchor["A"]["exact"], 3168)
    check("B anchor exact", anchor["B"]["exact"], 1170)
    close("B anchor rate", 1170 / 3168, 0.3693)
    close("anchor: z omitted share", anchor["B"]["z_omitted"] / 3168, 0.461)
    close("anchor: contradicting share", anchor["B"]["contradicting_value"] / 3168, 0.064)
    close("anchor: absent sensor share", anchor["B"]["sensor_absent"] / 3168, 0.106)
    close("abstract: 63 % anchor failure", 1 - 1170 / 3168, 0.6307)

    print("\n-- 4 and 6.3 composition and strata (figure_data.json) --")
    val = figdata["stratified"]["validation"]
    tst = figdata["stratified"]["test"]
    check("fit normal-transient", (val["normal-transient"]["defects"], val["normal-transient"]["rows"]), (29, 47))
    close("fit normal-transient rate", val["normal-transient"]["rate"], 0.617)
    check("fit open-set", (val["open-set"]["defects"], val["open-set"]["rows"]), (1, 454))
    check("fit classified", (val["classified"]["defects"], val["classified"]["rows"]), (0, 19))
    check("test normal-transient", (tst["normal-transient"]["defects"], tst["normal-transient"]["rows"]), (29, 40))
    close("test normal-transient rate", tst["normal-transient"]["rate"], 0.725)
    check("test open-set", (tst["open-set"]["defects"], tst["open-set"]["rows"]), (6, 1814))
    check("test classified", (tst["classified"]["defects"], tst["classified"]["rows"]), (5, 1314))
    check("fit composition sums to 520", 454 + 47 + 19, 520)
    check("test composition sums to 3168", 1814 + 40 + 1314, 3168)
    close("fit open-set share", 454 / 520, 0.873)
    close("fit normal share", 47 / 520, 0.090)
    close("fit classified share", 19 / 520, 0.037)
    close("test open-set share", 1814 / 3168, 0.573)
    close("test normal share", 40 / 3168, 0.013)
    close("test classified share", 1314 / 3168, 0.415)
    check("fig1 cells match n4", figdata["cells"], {c: summary[c]["numeric_rate"] for c in "ABCDM"})

    print("\n-- 6.4 diagnosis channel (classifier_contrasts.json) --")
    systems = contrasts["per_system"]
    expect = {"menu_rule": 1.0, "hybrid": 0.9960, "gbdt": 0.9980, "mlp": 0.9799, "adapter": 0.8330}
    for name, value in expect.items():
        close(f"{name} accuracy", systems[name]["accuracy"], value)
        low, high = systems[name]["accuracy_ci95"]
        print(f"      {name} CI: {low:.4f}-{high:.4f}")
    check("adapter CI matches manuscript", tuple(round(x, 4) for x in systems["adapter"]["accuracy_ci95"]),
          (0.7977, 0.8632))
    hg = contrasts["contrasts"]["hybrid_vs_gbdt"]
    check("hybrid vs LightGBM discordants", (hg["b"], hg["c"]), (2, 1))
    ha = contrasts["contrasts"]["hybrid_vs_adapter"]
    check("hybrid vs adapter discordants", (ha["b"], ha["c"]), (1, 82))

    print("\n-- 6.5 preference (kappa record) --")
    check("kappa pairs", kappa["pairs"], 360)
    close("observed agreement", kappa["observed_agreement"], 0.7778)
    close("Cohen kappa", kappa["cohen_kappa"], 0.5776)
    check("kappa below floor", kappa["meets_0_60"], False)

    print("\n-- manuscript text carries the verified values --")
    for needle in ("0.9980", "0.9960", "0.8330", "0.7977", "0.8632", "0.5776", "1.26 %", "5.77 %",
                   "36.93 %", "14.46 %", "10.26 %", "61.7 %", "72.5 %", "21.9 %", "114 of 520", "3,168"):
        check(f"text contains {needle!r}", needle in text, True)

    print("\n-- figures and build artefacts --")
    for name in ("fig1_cells", "fig2_stratified", "fig3_classifier", "fig4_anchor"):
        path = FIGDIR / f"{name}.png"
        size = path.stat().st_size if path.exists() else 0
        check(f"{name}.png present and non-trivial", size > 20_000, True)
    check("tex references four figures", len(re.findall(r"\\includegraphics", TEX.read_text(encoding="utf-8"))), 4)
    check("tex is pure ASCII", all(ord(ch) < 128 for ch in TEX.read_text(encoding="utf-8")), True)

    if PDF.exists():
        sys.path.insert(0, str(ROOT / "tools" / "pylibs"))
        from pypdf import PdfReader

        reader = PdfReader(str(PDF))
        body = "\n".join((page.extract_text() or "") for page in reader.pages)
        images = sum(len(page.images) for page in reader.pages)
        print(f"\n-- PDF ({PDF.name}, {PDF.stat().st_size} bytes) --")
        check("PDF header", PDF.read_bytes()[:5], b"%PDF-")
        check("PDF has pages", len(reader.pages) > 6, True)
        check("PDF embeds the four figures", images >= 4, True)
        for needle in ("Deterministic Diagnosis", "Diagnosis channel", "Non-claims", "0.8330", "0.5776"):
            check(f"PDF text contains {needle!r}", needle in body, True)
        check("no unresolved reference marker", "??" not in body, True)
        check("no unresolved citation marker", "[?]" not in body, True)
        print(f"      pages: {len(reader.pages)}, images: {images}")
    else:
        FAILS.append("PDF missing")

    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks passed")
    if FAILS:
        print("\nFAILURES")
        for item in FAILS:
            print(" -", item)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
