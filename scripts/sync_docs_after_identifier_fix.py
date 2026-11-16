"""Re-derive the numbers in the reports after masking identifiers in the numeric checker.

Cell table: A 0/520, B 30/520, C 0, M 30, D 0; anchors 520/188/520/191/262; D fallback 114/520 (21.9 %).
Test: A 0/3,168, B 40/3,168, McNemar 40 vs 0.
The §3, §4 and §9b blocks are replaced by markers rather than by long exact strings, so a whitespace change
cannot silently skip a section.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md"
CLAIMS = ROOT / "docs/reports/PAPER_CLAIMS_2026-11-16.md"
ACHIEVED = ROOT / "docs/reports/BENCHMARK_ACHIEVED_METRICS_REPORT.md"

CELLS = """## 3. The cell table (520 rows per cell)

| cell | numeric defect rows | rate | grounding | tag-role | contradiction | anchor exact / 520 |
|---|---|---|---|---|---|---|
| A | **0** | 0.0000 | 0 | 0 | 0 | **520** |
| B | 30 | 0.0577 | 33 | 57 | 0 | 188 |
| C | **0** | 0.0000 | 1 | 17 | 0 | **520** |
| M | 30 | 0.0577 | 32 | 43 | 0 | 191 |
| D | 0 | 0.0000 | 0 | 0 | 0 | 262 |

Regenerated 2026-11-16 after two checker corrections (see §9b). First, numbers were attributed only to the
detector named in their sentence, which counted stored sensor values as invented (A 19 → 13, B 62 → 50).
Second, the number reader tokenised the digits inside names — `FAULT_08` gave `08`, `Component_D_Stream9`
gave `9` — and masking those identifiers took the table to its final row: **A and C drop to zero, B and M to
30**. The deterministic writers copy numbers from the page, so zero is the truthful count for them. Only
numeric counts and D's anchor row moved; grounding, tag-role and contradiction are identical across all
three versions, which is the evidence that both fixes were narrow. Pre-revision table preserved as
`n4_cells_preinstrument_2026-11-16.json`.

**Seam certification** (`n4_cells.seam_certification`): C leaks 0 of B's numeric values, M leaks 0 of A's;
C leaves 88 donor sentences unused, M leaves 584. The seams are doing what they claim.

**D's ceiling.** D fell back to A on **114 of 520 rows (21.9 %)**, which is **inside** its declared 25 %
ceiling. Under the two earlier checker versions the rate was 27.3 % and then 25.4 %, and the signed
`d_ceiling_decision_2026-11-16.json` records a breach and prescribes the "A with extra steps" framing. That
breach no longer exists on the corrected instrument, so the paper's D statement is **pending re-signoff**:
either keep the conservative framing (defensible, since D's gain over A is zero by construction on rows it
hands back) or report D as a wrapper that stays inside its budget. It is not switched silently here.

"""

PRIMARY = """## 4. Primary contrast

**B vs A, numeric-defect rows, paired.** B has 30, A has 0; all 30 discordant pairs fall B's way; exact
McNemar p ≈ 1.9 × 10⁻⁹. This is the paper's main claim on the fit set. The test contrast (§9b) keeps the same
direction and additionally shows B inside the 2 % numeric gate on held-out data — but it is a different
population, and the split is named wherever the number appears.

"""

D33 = """## 9b. D33 executed: B versus A on 3,168 test items (revised 2026-11-16)

*Supersedes the earlier §9b. Two checker corrections changed these numbers; the sequence and its effect are
recorded in [`D33_TEST_CONTRAST_2026-11-16.md`](D33_TEST_CONTRAST_2026-11-16.md).*

**The pass.** 3,168 test rows, kb order, batch 16, adapter seed 42 —
`outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl`, `mean_seconds_per_item 3.57`,
`total_seconds 11,314` (**3.14 h**), no missing rows.

| check | A (fallback v3.1) | B (adapter) | gate |
|---|---|---|---|
| **numeric defect** | **0.00 % (0)** | **1.26 % (40)** | ≤ 2 % — **both pass** |
| grounding violation | 7.89 % (250) | 10.26 % (325) | ≤ 2 % — both fail |
| negate-aware contradiction | 0.00 % (0) | 0.16 % (5) | ≤ 2 % — both pass |
| tag-role claim | 0.00 % (0) | 14.46 % (458) | ≤ 2 % — B fails |
| **anchor exact** | **100 % (3,168)** | **36.93 % (1,170)** | ≥ 90 % — B fails |

McNemar on the primary check: **B-only 40, A-only 0, exact p ≈ 2 × 10⁻¹²**. There is no discordance in A's
favour, so no ratio is quoted.

**What changed and why it matters.** The first scoring reported A at 41 % because the checker attributed
numbers only to the detector named in a sentence; fixing that gave 19.54 %. Masking identifiers (`FAULT_08`,
`Component_D_Stream9`, `XMEAS_17`) took A to **zero**: every "defect" it had was a name being read as a
quantity. The adapter's rate fell from 11.92 % to 5.77 % on validation and from 3.50 % to **1.26 %** on test.
So the earlier statement that "both writers fail the numeric gate on test" is **withdrawn**: on held-out data
**B is inside the 2 % gate** (1.26 %) and A has no numeric defects at all.

**Reading.** The adapter is not more faithful than the deterministic writer — it is the only one of the two
that produces numeric defects — but on held-out events its rate is within the declared budget. Its failures
are elsewhere and large: grounding 10.26 %, tag-role 14.46 %, anchor exactness 36.93 % with the z-score
omitted in 46 % of texts. The shipped condition is unchanged: A, chosen at rung 5 on cost.

"""

CLAIM_C1_OLD_START = "**C1 —"
CLAIM_C2 = "**C2 —"

CLAIM_C1 = """**C1 — A deterministic writer carries no numeric-defect liability; the 2B adapter carries a small one that
meets the held-out budget.** On the 520-item validation cell table A and C score **0/520** numeric defects
while B and M score 30/520 (5.77 %); on 3,168 held-out test items A scores **0/3,168** and B **40/3,168
(1.26 %)**, inside the 2 % gate (McNemar 40 vs 0, p ≈ 2 × 10⁻¹²). The adapter's failures are structural rather
than numeric: grounding 10.26 %, tag-role 14.46 %, anchor exactness 36.93 %. Sources: `n4_cells.json`,
`d33/test_contrast.json`.

"""

CLAIM_C6_OLD_START = "**C6 —"
CLAIM_C7 = "**C7 —"

CLAIM_C6 = """**C6 — The validated fallback wrapper stays inside its budget.** Condition D substitutes the fallback report
when the adapter is flagged; on the corrected instrument it falls back on **114/520 rows (21.9 %)**, inside
the pre-registered 25 % ceiling. The signed `d_ceiling_decision_2026-11-16.json` documents a 27.3 % breach
measured before two checker corrections, so the paper's D wording is **pending the owner's re-signoff**;
nothing is reported as a pass until that decision is re-signed. Source: `n4_cells.json` seam certification.

"""

CLAIM_ROW_OLD = "| cell-table numeric defects (corrected instrument) | A 13/520, B 50/520, C 13/520, M 50/520, D 6/520 | `n4_cells.json` summary |"
CLAIM_ROW_NEW = "| cell-table numeric defects (identifier-masked) | A 0/520, B 30/520, C 0/520, M 30/520, D 0/520 | `n4_cells.json` summary |"
CLAIM_TEST_OLD = "| test numeric defects | A 619/3,168 (19.54 %), B 115/3,168 (3.63 %) | `d33/test_contrast.json` |"
CLAIM_TEST_NEW = "| test numeric defects | A 0/3,168 (0.00 %), B 40/3,168 (1.26 %) | `d33/test_contrast.json` |"
CLAIM_MCN_OLD = "| primary validation contrast | B vs A McNemar 49/12, p < 1e-06; order invariance 0.7077 | `n4_cells`, overnight receipt |"
CLAIM_MCN_NEW = "| primary validation contrast | B vs A McNemar 30/0, exact p ≈ 1.9 × 10⁻⁹; order invariance 0.7077 | `n4_cells`, overnight receipt |"

ABSTRACT_OLD = """deterministic writer (3.6 % vs 19.5 % numeric defects; p < 0.001) but omits or misstates the top sensor's
z-score in 63 % of reports and cites an unsupported tag role in 14 %. Neither writer meets the 2 % numeric
gate."""
ABSTRACT_NEW = """deterministic writer on the only dimension where it produces any defects at all — 1.3 % against 0.0 %
numeric defects on held-out items, inside the 2 % budget — but omits or misstates the top sensor's z-score in
63 % of reports and cites an unsupported tag role in 14 %."""

NOTE_OLD_START = "**Known over-count that bound the numeric claims.**"

NOTE_NEW = """**Identifier over-count, now fixed.** An earlier version of this document said every numeric rate was an
upper bound because names such as `FAULT_08` yielded digit tokens. That is fixed: identifiers are masked
before numbers are read, the deterministic writers score zero, and the adapter's rates are 5.77 % on
validation and 1.26 % on test. The numbers above are final for this instrument; the outstanding tasks in §5
are separate."""


def replace_block(text: str, start_marker: str, end_marker: str, new: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start + 1)
    if start < 0 or end < 0:
        raise SystemExit("block markers missing: %r .. %r" % (start_marker, end_marker))
    return text[:start] + new + text[end:]


def replace_from_to(text: str, start_marker: str, end_marker: str, new: str) -> str:
    return replace_block(text, start_marker, end_marker, new)


def main() -> int:
    report = REPORT.read_text(encoding="utf-8").replace("\r\n", "\n")
    report = replace_block(report, "## 3. The cell table", "## 4. Primary contrast", CELLS)
    report = replace_block(report, "## 4. Primary contrast", "## 5. Classifier track", PRIMARY)
    report = replace_block(report, "## 9b. D33 executed", "## 10. E —", D33)
    REPORT.write_text(report, encoding="utf-8")

    claims = CLAIMS.read_text(encoding="utf-8").replace("\r\n", "\n")
    claims = replace_from_to(claims, CLAIM_C1_OLD_START, CLAIM_C2, CLAIM_C1)
    claims = replace_from_to(claims, CLAIM_C6_OLD_START, CLAIM_C7, CLAIM_C6)
    for old, new in ((CLAIM_ROW_OLD, CLAIM_ROW_NEW), (CLAIM_TEST_OLD, CLAIM_TEST_NEW),
                     (CLAIM_MCN_OLD, CLAIM_MCN_NEW), (ABSTRACT_OLD, ABSTRACT_NEW)):
        if old not in claims:
            raise SystemExit("claims replacement missing: %s" % old[:70])
        claims = claims.replace(old, new, 1)
    if NOTE_OLD_START in claims:
        start = claims.find(NOTE_OLD_START)
        end = claims.find("\n\n", claims.find("\n\n", start) + 2)
        claims = claims[:start] + NOTE_NEW + claims[end:]
    CLAIMS.write_text(claims, encoding="utf-8")

    achieved = ACHIEVED.read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new in (("**2.50% (13 / 520)**", "**0.00% (0 / 520)**"),
                     ("**9.62% (50 / 520)**", "**5.77% (30 / 520)**"),
                     ("• 2.50% Numeric Defects", "• 0.00% Numeric Defects"),
                     ("• 9.62% Numeric Defects", "• 5.77% Numeric Defects"),
                     ("Fallback: 2.50% vs Adapter: 9.62%", "Fallback: 0.00% vs Adapter: 5.77%")):
        if old not in achieved:
            raise SystemExit("achieved replacement missing: %s" % old)
        achieved = achieved.replace(old, new, 1)
    ACHIEVED.write_text(achieved, encoding="utf-8")

    for path in (REPORT, CLAIMS, ACHIEVED):
        print("updated", path.name, len(path.read_text(encoding="utf-8").splitlines()), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
