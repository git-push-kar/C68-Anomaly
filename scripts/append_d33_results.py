"""Append the D33 execution result to the final report and the handover, with no gate reinterpreted."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "outputs/analysis/phase3a/d33/test_contrast.json").read_text(encoding="utf-8"))

SECTION = """
## 9b. D33 executed: B measured on the test split, A not comparable

**The pass ran.** 3,168 test rows, kb order, batch 16, adapter seed 42:
`outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl`, 3,168 rows,
`mean_seconds_per_item 3.57`, `total_seconds 11,314` (**3.14 h**), no missing rows.

| condition B (adapter), 3,168 test rows | count | rate | validation B (520) |
|---|---|---|---|
| numeric defect | 182 | **5.74 %** | 11.92 % |
| grounding violation | 325 | **10.26 %** | 6.35 % |
| negate-aware contradiction | 5 | **0.16 %** | 0.00 % |
| tag-role claim | 458 | **14.46 %** | 11.0 % |
| anchor exact | 1,170 | **36.93 %** | 36.2 % |
| anchor z omitted | 1,459 | 46.06 % | — |
| anchor contradicting value | 203 | 6.41 % | — |
| anchor sensor absent | 336 | 10.61 % | — |

On the test split the adapter's numeric defect rate **halves** (11.92 % → 5.74 %) while its grounding
violations **rise** (6.35 % → 10.26 %) and contradictions appear for the first time (5 rows, 0.16 %). Anchor
exactness is flat at 36.9 %, still far under the 90 % gate.

**The A column is not a measurement and is not used.** Rebuilding A on the test rows yields
1,314/3,168 = 41.48 % numeric defects, against 3.65 % on validation. That is not the same writer failing
more often; it is an instrument artefact. The same rebuild code reproduces validation exactly — A 19/520 and
B 62/520, both matching the stored cell table — and the flagged test sentences quote values that *are* in the
evidence store (17.99, 0.52, 4 all present), so `numeric_flags` is attributing them to the wrong detector on
the much longer test prompts (3,270 tokens against 1,030 on validation). Until that attribution is fixed,
**D33 cannot be scored as a B-versus-A gate**, and no contrast claim is made. What is reported is B's
absolute rate on held-out test items, which needs no A column.

Supporting artefacts: `scripts/crossfit_test_rows.py` (test probabilities; `--check-val` reproduces all 834
stored validation probability vectors exactly), `scripts/score_test_pass.py`,
`outputs/analysis/phase3a/d33/test_contrast.json`.
"""

HANDOVER = """
## 32. D33 executed on the test split, 2026-11-16

The one-shot pass ran at batch 16 on 3,168 test rows: 3.14 h, 3.57 s/item, no missing rows. B's test rates are
numeric 5.74 % (182), grounding 10.26 % (325), contradiction 0.16 % (5), tag-role 14.46 % (458), anchor exact
36.93 % (1,170). Against validation B the numeric rate halves and grounding rises; anchor exactness is flat.

A is not comparable on the test split: rebuilt A scores 41.48 % numeric defects against 3.65 % on validation,
although the identical code reproduces validation exactly (A 19/520, B 62/520). The flagged sentences quote
values that are in the store (17.99, 0.52, 4), so `numeric_flags` mis-attributes numbers to detectors on the
longer test prompts. **No B-vs-A contrast is claimed**; the A column must not be quoted until that
attribution is fixed. Test probabilities come from `scripts/crossfit_test_rows.py`, whose `--check-val`
reproduces all 834 stored validation probability vectors.
"""


def main() -> int:
    assert SUMMARY["rows"] == 3168, SUMMARY["rows"]
    report = ROOT / "docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md"
    text = report.read_text(encoding="utf-8").replace("\r\n", "\n")
    if "## 9b. D33 executed" not in text:
        anchor = "## 10. E — trained, unscored, and one wording correction"
        if anchor not in text:
            raise SystemExit("anchor missing in final report")
        text = text.replace(anchor, SECTION.strip() + "\n\n" + anchor, 1)
        report.write_text(text, encoding="utf-8")
    doc = ROOT / "docs/HANDOVER_BENCHMARK_PLAN_V2.md"
    body = doc.read_text(encoding="utf-8").replace("\r\n", "\n")
    if "## 32. D33 executed" not in body:
        body = body.rstrip("\n") + "\n" + HANDOVER
        doc.write_text(body, encoding="utf-8")
    print("report lines", len(report.read_text(encoding='utf-8').splitlines()))
    print("handover lines", len(doc.read_text(encoding='utf-8').splitlines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
