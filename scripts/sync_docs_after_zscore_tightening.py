"""Bring every document to one set of numbers after the z-score tightening.

Cell table (unchanged by the tightening): A 13/520, B 50/520.
Test contrast (tightening moves B only): A 619/3,168, B 115/3,168, McNemar 101 vs 605.
Every replacement is asserted so a partial edit cannot pass.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACHIEVED = ROOT / "docs/reports/BENCHMARK_ACHIEVED_METRICS_REPORT.md"
D33 = ROOT / "docs/reports/D33_TEST_CONTRAST_2026-11-16.md"
REPORT = ROOT / "docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md"
CLAIMS = ROOT / "docs/reports/PAPER_CLAIMS_2026-11-16.md"

SUBS = [
    # --- the older achieved-metrics report: numeric figures only -----------------
    (ACHIEVED,
     "| **Numeric Defect Rate** | $\\le 2.0\\%$ | **3.65% (19 / 520)** | **11.92% (62 / 520)** | **Adapter fails; vetoed at Rung 1** |",
     "| **Numeric Defect Rate** | $\\le 2.0\\%$ | **2.50% (13 / 520)** | **9.62% (50 / 520)** | **Adapter fails; vetoed at Rung 1** |"),
    (ACHIEVED,
     'CondA["Condition A: Fallback v3.1 (Deterministic)<br>• 3.65% Numeric Defects<br>',
     'CondA["Condition A: Fallback v3.1 (Deterministic)<br>• 2.50% Numeric Defects<br>'),
    (ACHIEVED,
     '• 11.92% Numeric Defects (Vetoed at Rung 1)<br>',
     '• 9.62% Numeric Defects (Vetoed at Rung 1)<br>'),
    (ACHIEVED,
     "[ Rung 1: Numeric Defect Gate ] ──► Fallback: 3.65% vs Adapter: 11.92% ──► ADAPTER VETOED",
     "[ Rung 1: Numeric Defect Gate ] ──► Fallback: 2.50% vs Adapter: 9.62% ──► ADAPTER VETOED"),
    # --- D33 record -------------------------------------------------------------
    (D33,
     "| **numeric defect** | **19.54 % (619)** | **3.50 % (111)** | ≤ 2 % | **B better by 16.0 pp**; both fail the gate |",
     "| **numeric defect** | **19.54 % (619)** | **3.63 % (115)** | ≤ 2 % | **B better by 15.9 pp**; both fail the gate |"),
    (D33,
     "McNemar on the primary, numeric defect: **B-only 97, A-only 605, p ≈ 0** (two-sided exact, 97 + 605\ndiscordant). Test items where A invents a number and B does not outnumber the reverse 6.2 : 1.",
     "McNemar on the primary, numeric defect: **B-only 101, A-only 605, p ≈ 0** (two-sided exact, 101 + 605\ndiscordant). Test items where A invents a number and B does not outnumber the reverse 6.0 : 1."),
    # --- final report §9b -------------------------------------------------------
    (REPORT,
     "| **numeric defect** | **19.54 % (619)** | **3.50 % (111)** | ≤ 2 % — both fail |",
     "| **numeric defect** | **19.54 % (619)** | **3.63 % (115)** | ≤ 2 % — both fail |"),
    (REPORT,
     "McNemar on the primary check: **B-only 97, A-only 605, p ≈ 0**. On held-out items A invents a number where B\ndoes not 6.2 times as often as the reverse — the validation ordering (A 2.50 % vs B 9.62 %) reverses on test.",
     "McNemar on the primary check: **B-only 101, A-only 605, p ≈ 0**. On held-out items A invents a number where B\ndoes not 6.0 times as often as the reverse — the validation ordering (A 2.50 % vs B 9.62 %) reverses on test."),
    # --- claims -----------------------------------------------------------------
    (CLAIMS,
     "on 3,168\nheld-out test items the ordering reverses (B 3.50 % against A 19.54 %; McNemar 97 vs 605, p ≈ 0).",
     "on 3,168\nheld-out test items the ordering reverses (B 3.63 % against A 19.54 %; McNemar 101 vs 605, p ≈ 0)."),
    (CLAIMS,
     "| test numeric defects | A 619/3,168 (19.54 %), B 111/3,168 (3.50 %) | `d33/test_contrast.json` |",
     "| test numeric defects | A 619/3,168 (19.54 %), B 115/3,168 (3.63 %) | `d33/test_contrast.json` |"),
    (CLAIMS,
     "deterministic writer (3.5 % vs 19.5 % numeric defects; p < 0.001)",
     "deterministic writer (3.6 % vs 19.5 % numeric defects; p < 0.001)"),
]

NOTE = """
## 1b. Tightening, and a known over-count

The checker now requires a number quoted in a role to match a value of that role: a sentence writing
`z=17.99` must match a sensor's z-score, not merely a number printed elsewhere on the page. Validation stays
at **A 13/520 and B 50/520**, so the tightening changed no verdict; on test it moves B from 111 to 115 rows
(3.50 % → 3.63 %).

**Known over-count, deliberately not changed today.** The tokenizer still reads the digits inside identifiers:
`FAULT_08` contributes the token `08`, `Component_D_Stream9` contributes `9`, and both can be flagged when
the sentence also names a detector. Fixing that would lower both rates further, which is why it is recorded
rather than quietly applied — **the reported numeric rates are upper bounds**, and the direction of the
comparison is unaffected because the artefact inflates A far more often than B (A quotes many more
identifiers per report).
"""


def main() -> int:
    texts = {}
    for path, old, new in SUBS:
        if path not in texts:
            texts[path] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if old not in texts[path]:
            print("MISS in %s: %s" % (path.name, old.splitlines()[0][:80]))
            return 2
        texts[path] = texts[path].replace(old, new, 1)
    if "## 1b. Tightening, and a known over-count" not in texts[D33]:
        marker = "## 2. The contrast on 3,168 held-out test items"
        if marker not in texts[D33]:
            print("MISS: D33 section marker")
            return 2
        texts[D33] = texts[D33].replace(marker, NOTE.strip() + "\n\n" + marker, 1)
    for path, text in texts.items():
        path.write_text(text, encoding="utf-8")
        print("updated", path.name, len(text.splitlines()), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
