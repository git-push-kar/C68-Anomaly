"""Replace the provisional §9b with the scored D33 contrast, and record it in the handover."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SECTION = """## 9b. D33 executed: B versus A on 3,168 test items (revised 2026-11-16)

*Supersedes the earlier §9b, which reported the A column as incomparable. The cause was found and fixed, and
the fix changes the validation numeric counts as well — recorded honestly below rather than hidden.*

**The pass.** 3,168 test rows, kb order, batch 16, adapter seed 42 —
`outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl`, `mean_seconds_per_item 3.57`,
`total_seconds 11,314` (**3.14 h**), no missing rows.

| check | A (fallback v3.1) | B (adapter) | gate |
|---|---|---|---|
| **numeric defect** | **19.54 % (619)** | **3.50 % (111)** | ≤ 2 % — both fail |
| grounding violation | 7.89 % (250) | 10.26 % (325) | ≤ 2 % — both fail |
| negate-aware contradiction | 0.00 % (0) | 0.16 % (5) | ≤ 2 % — both pass |
| tag-role claim | 0.00 % (0) | 14.46 % (458) | ≤ 2 % — B fails |
| **anchor exact** | **100 % (3,168)** | **36.93 % (1,170)** | ≥ 90 % — B fails |

McNemar on the primary check: **B-only 97, A-only 605, p ≈ 0**. On held-out items A invents a number where B
does not 6.2 times as often as the reverse — the validation ordering (A 2.50 % vs B 9.62 %) reverses on test.

**Instrument revision, declared.** `numeric_flags` attributed a number only to the detectors named in its
sentence, so a sentence naming LSTM and CVA while quoting a stored sensor z-score counted that value as
invented. The fix allows values from the non-detector evidence sections to satisfy the stored test while
detector numbers are still checked against the named detector. Consequence: validation becomes **A 13/520
(2.50 %) and B 50/520 (9.62 %)**, against 19/520 and 62/520 as stored — six A items and twelve B items were
false positives, so a check that reproduced the old numbers would have reproduced the bug. Both conditions
still exceed the 2 % gate, so the rung-1 veto direction is unchanged, and anchor, grounding, contradiction
and tag-role counts are untouched.

**Reading.** The adapter is the more numerically faithful writer on held-out events and the less faithful
one on validation; the paper must name the split. Anchor exactness is the structural failure (36.9 % against
100 %, with the z-score omitted in 46 % of texts). Nothing here changes the shipped condition: A is chosen at
rung 5 on cost, and both writers fail the numeric gate on test.

Full record and reproduction: [`D33_TEST_CONTRAST_2026-11-16.md`](D33_TEST_CONTRAST_2026-11-16.md).

"""

HANDOVER = """
## 33. D33 contrast scored and the numeric instrument revised, 2026-11-16

`numeric_flags` attributed quoted numbers only to detectors named in the same sentence, so stored sensor z,
lag and residual values read as invented whenever a sentence named a detector too. Fixed in
`scripts/pilot_rechecks.py`; the fix necessarily moves validation from 19/520 and 62/520 to **13/520 and
50/520** (six and twelve false positives), and the request to preserve the old numbers was therefore not
satisfiable — a check that reproduced them would have reproduced the bug.

With the corrected instrument, D33 on 3,168 held-out test rows is: numeric **A 19.54 % vs B 3.50 %**
(McNemar 97 vs 605, p ≈ 0), grounding A 7.89 % vs B 10.26 %, contradiction 0 vs 0.16 %, tag-role 0 vs
14.46 %, anchor exact **100 % vs 36.93 %**. The numeric ordering reverses relative to validation, where A is
the better writer; the shipped condition is unchanged because it was decided on cost at rung 5, and both
writers fail the 2 % numeric gate on test.

Paper-facing claims, the quotable numbers and the ten explicit non-claims are in
`docs/reports/PAPER_CLAIMS_2026-11-16.md`. Still open: test order runs (shuffled-1/2), E scoring and its
seed-43/44 integrity check, the detector-frontier table, the follow-up T condition, parity (1,390), release
packaging, and the owner's Version A/B signature.
"""


def main() -> int:
    report = ROOT / "docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md"
    text = report.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.find("## 9b. D33 executed")
    end = text.find("## 10. E —")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit("could not locate the §9b block")
    text = text[:start] + SECTION + text[end:]
    report.write_text(text, encoding="utf-8")
    doc = ROOT / "docs/HANDOVER_BENCHMARK_PLAN_V2.md"
    body = doc.read_text(encoding="utf-8").replace("\r\n", "\n")
    if "## 33. D33 contrast scored" not in body:
        doc.write_text(body.rstrip("\n") + "\n" + HANDOVER, encoding="utf-8")
    print("report lines", len(report.read_text(encoding="utf-8").splitlines()))
    print("handover lines", len(doc.read_text(encoding="utf-8").splitlines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
