"""Update the report and claims to the regenerated cell table (corrected numeric_flags).

Every replacement is asserted so a partial edit cannot pass silently. The pre-revision files are kept as
`n4_cells_preinstrument_2026-11-16.json`, and the corrected ones are the live `n4_cells.json`.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md"
CLAIMS = ROOT / "docs/reports/PAPER_CLAIMS_2026-11-16.md"

TABLE_OLD = """| cell | numeric defect rows | rate | grounding | tag-role | contradiction | anchor exact / 520 |
|---|---|---|---|---|---|---|
| A | **19** | 0.0365 | 0 | 0 | 0 | **520** |
| B | 62 | 0.1192 | 33 | 57 | 0 | 188 |
| C | **19** | 0.0365 | 1 | 17 | 0 | **520** |
| M | 62 | 0.1192 | 32 | 43 | 0 | 191 |
| D | 7 | 0.0135 | 0 | 0 | 0 | 286 |

Source: `outputs/overnight_2026-11-05/n4_cells.json` (`summary`), reproduced flag-for-flag in §25."""

TABLE_NEW = """| cell | numeric defect rows | rate | grounding | tag-role | contradiction | anchor exact / 520 |
|---|---|---|---|---|---|---|
| A | **13** | 0.0250 | 0 | 0 | 0 | **520** |
| B | 50 | 0.0962 | 33 | 57 | 0 | 188 |
| C | **13** | 0.0250 | 1 | 17 | 0 | **520** |
| M | 50 | 0.0962 | 32 | 43 | 0 | 191 |
| D | 6 | 0.0115 | 0 | 0 | 0 | 279 |

Regenerated 2026-11-16 with the corrected `numeric_flags` (see §9b): the checker had counted stored sensor
and lag values as invented whenever a sentence also named a detector, which inflated A from 13 to 19 and B
from 50 to 62. Only the numeric count and D's anchor row moved; grounding, tag-role and contradiction are
unchanged, which is itself evidence that the fix was narrow. The pre-revision table is preserved as
`n4_cells_preinstrument_2026-11-16.json`. Source: `outputs/overnight_2026-11-05/n4_cells.json`."""

CEIL_OLD = """**D's ceiling.** D fell back to A on **142 of 520 rows (27.3 %)** against a declared 25 % ceiling. It is
therefore reported as **"A with extra steps"**: its numeric count of 7 is published with the fallback rate
beside it, and no claim is made that D reduces defects on the rows it wrote itself. Signed acceptance:"""

CEIL_NEW = """**D's ceiling.** D fell back to A on **132 of 520 rows (25.4 %)** against a declared 25 % ceiling. It is
therefore reported as **"A with extra steps"**: its numeric count of 6 is published with the fallback rate
beside it, and no claim is made that D reduces defects on the rows it wrote itself. The corrected instrument
moves the rate from 27.3 % to 25.4 % and the verdict does not change. Signed acceptance:"""

CONTRAST_OLD = """**B vs A, numeric-defect rows, paired.** B has 62, A has 19; discordant pairs 60 (B only) and 17 (A only);
exact McNemar p < 1e-06. This is the paper's main claim and it is measured on the fit set only — the test
confirmation is prepared and unspent (§9)."""

CONTRAST_NEW = """**B vs A, numeric-defect rows, paired.** B has 50, A has 13; discordant pairs 49 (B only) and 12 (A only);
exact McNemar p < 1e-06. This is the paper's main claim and it is measured on the fit set; the test contrast
is reported separately in §9b and points the other way, so the split must be named wherever this is quoted."""

CLAIM_CEIL_OLD = """report when the adapter is flagged; it falls back on 142/520 rows, 27.3 % against a pre-registered 25 %"""

CLAIM_CEIL_NEW = """report when the adapter is flagged; it falls back on 132/520 rows, 25.4 % against a pre-registered 25 %"""

CLAIM_ROW_OLD = """| cell-table numeric defects | A 19/520, B 62/520, C 19/520, M 62/520, D 7/520 | `n4_cells.summary.json` |"""

CLAIM_ROW_NEW = """| cell-table numeric defects (corrected instrument) | A 13/520, B 50/520, C 13/520, M 50/520, D 6/520 | `n4_cells.json` summary |"""

CLAIM_MCN_OLD = """| primary validation contrast | B vs A McNemar 60/17, p < 1e-06; order invariance 0.7077 | `n4_cells`, overnight receipt |"""

CLAIM_MCN_NEW = """| primary validation contrast | B vs A McNemar 49/12, p < 1e-06; order invariance 0.7077 | `n4_cells`, overnight receipt |"""

CLAIM_C1_OLD = """On the 520-item validation cell table the adapter carries 9.62 % numeric defects against A's 2.50 %; on 3,168"""

CLAIM_C1_NEW = """On the 520-item validation cell table the adapter carries 9.62 % numeric defects (50/520) against A's
2.50 % (13/520); on 3,168"""

SUBS = [(REPORT, TABLE_OLD, TABLE_NEW), (REPORT, CEIL_OLD, CEIL_NEW), (REPORT, CONTRAST_OLD, CONTRAST_NEW),
        (CLAIMS, CLAIM_CEIL_OLD, CLAIM_CEIL_NEW), (CLAIMS, CLAIM_ROW_OLD, CLAIM_ROW_NEW),
        (CLAIMS, CLAIM_MCN_OLD, CLAIM_MCN_NEW), (CLAIMS, CLAIM_C1_OLD, CLAIM_C1_NEW)]


def main() -> int:
    texts = {}
    for path, old, new in SUBS:
        if path not in texts:
            texts[path] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if old not in texts[path]:
            print("MISS in %s: %s" % (path.name, old.splitlines()[0][:70]))
            return 2
        texts[path] = texts[path].replace(old, new, 1)
    for path, text in texts.items():
        path.write_text(text, encoding="utf-8")
        print("updated", path.name, len(text.splitlines()), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
