"""Finish the identifier-fix sync: claims document, achieved-metrics report, handover entry."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "docs/reports/PAPER_CLAIMS_2026-11-16.md"
ACHIEVED = ROOT / "docs/reports/BENCHMARK_ACHIEVED_METRICS_REPORT.md"
DOC = ROOT / "docs/HANDOVER_BENCHMARK_PLAN_V2.md"

C1 = """**C1 — The deterministic writer carries no numeric-defect liability; the 2B adapter carries a small one
that meets the held-out budget.** On the 520-item validation cell table A and C score **0/520** numeric
defects while B and M score 30/520 (5.77 %); on 3,168 held-out test items A scores **0/3,168** and B
**40/3,168 (1.26 %)**, inside the 2 % gate (McNemar 40 vs 0, p ≈ 2 × 10⁻¹²). The adapter's failures are
structural rather than numeric: grounding 10.26 %, tag-role 14.46 %, anchor exactness 36.93 %. Sources:
`n4_cells.json`, `d33/test_contrast.json`.

"""

C6 = """**C6 — The fallback wrapper stays inside its budget, and the paper's D wording awaits re-signoff.**
Condition D substitutes the fallback report when the adapter is flagged; on the corrected instrument it falls
back on **114/520 rows (21.9 %)**, inside the pre-registered 25 % ceiling. The signed
`d_ceiling_decision_2026-11-16.json` records a 27.3 % breach measured before two checker corrections, so D is
**not** claimed as a pass here: the owner must re-sign the decision, and the conservative "A with extra
steps" framing remains available because D's own contribution is zero on the rows it hands back. Source:
`n4_cells.json` seam certification.

"""

ABSTRACT = """> We evaluate a 2B parameter language model as a report writer for an industrial anomaly-detection system,
> with diagnosis kept deterministic. On 3,168 held-out events the model meets the numeric-fidelity budget
> (1.3 % defective reports against 2 %) while the deterministic writer produces none at all, so no numerical
> advantage is claimed for the model. Its failures are elsewhere: it omits or misstates the top sensor's
> z-score in 63 % of reports, cites an unsupported tag role in 14 %, and violates grounding in 10 %.
> Pairwise preference between writers fails its reliability floor (κ 0.58), and the shipped configuration is
> chosen on cost, not quality. A gradient-boosted classifier matches our calibrated hybrid exactly (0.9940 vs
> 0.9940, p = 1), so the knowledge-base contribution is explanatory rather than predictive."""

NOT_CLAIMED = """2. **A numerical advantage for the adapter** — on test it meets the gate (1.26 %) but the deterministic
   writer has no numeric defects at all (0/3,168); the adapter's claim is budget compliance, not superiority."""

NOTE = """**Identifier over-count, now fixed.** An earlier version of this document said every numeric rate was an
upper bound because names such as `FAULT_08` yielded digit tokens. That is fixed: identifiers are masked
before numbers are read, the deterministic writers score zero, and the adapter's rates are 5.77 % on
validation and 1.26 % on test. These numbers are final for this instrument."""

HANDOVER = """
## 34. Identifier over-count fixed, and every rate re-derived, 2026-11-16

The numeric reader tokenised digits inside names (`FAULT_08` → `08`, `Component_D_Stream9` → `9`,
`XMEAS_17` → `17`). Any token starting with a letter and containing a digit is now masked. Result:

| | before identifiers masked | after |
|---|---|---|
| validation A / B numeric | 13/520 · 50/520 | **0/520 · 30/520** |
| test A / B numeric | 619/3,168 · 115/3,168 | **0/3,168 · 40/3,168** |
| D fallback rows | 132 (25.4 %) | **114 (21.9 %)** |
| D anchor exact | 279 | **262** |
| McNemar, validation | 49 / 12 | **30 / 0** (p ≈ 1.9e-9) |

Two claims move. **C1 is rewritten**: the adapter no longer "beats" A on numeric fidelity — A has no numeric
defects at all, and B's claim is that it meets the 2 % gate on held-out data (1.26 %). **C6 is suspended
pending re-signoff**: the signed D-ceiling decision records a 27.3 % breach that no longer exists at 21.9 %,
and the conservative framing stays available. The selection is untouched: chosen A, `decided_by_rung 5`,
passers [A, D].
"""


def splice(text: str, start: str, end: str, new: str) -> str:
    a, b = text.find(start), text.find(end)
    if a < 0 or b < 0 or b <= a:
        raise SystemExit("markers missing: %r %r" % (start, end))
    return text[:a] + new + text[b:]


def main() -> int:
    claims = CLAIMS.read_text(encoding="utf-8").replace("\r\n", "\n")
    claims = splice(claims, "**C1 —", "**C2 —", C1)
    claims = splice(claims, "**C6 —", "**C7 —", C6)
    for old, new in (
        ("| cell-table numeric defects (corrected instrument) | A 13/520, B 50/520, C 13/520, M 50/520, D 6/520 | `n4_cells.json` summary |",
         "| cell-table numeric defects (identifier-masked) | A 0/520, B 30/520, C 0/520, M 30/520, D 0/520 | `n4_cells.json` summary |"),
        ("| corrected-instrument numeric defects (validation) | A 13/520 (2.50 %), B 50/520 (9.62 %) | this revision, §1 of `D33_TEST_CONTRAST_2026-11-16.md` |",
         "| validation numeric defects (final instrument) | A 0/520 (0.00 %), B 30/520 (5.77 %) | `n4_cells.json` |"),
        ("| test numeric defects | A 619/3,168 (19.54 %), B 115/3,168 (3.63 %) | `d33/test_contrast.json` |",
         "| test numeric defects | A 0/3,168 (0.00 %), B 40/3,168 (1.26 %) | `d33/test_contrast.json` |"),
        ("| primary validation contrast | B vs A McNemar 49/12, p < 1e-06; order invariance 0.7077 | `n4_cells`, overnight receipt |",
         "| primary validation contrast | B vs A McNemar 30/0, exact p ≈ 1.9 × 10⁻⁹; order invariance 0.7077 | `n4_cells`, overnight receipt |"),
        ("2. **A numeric-gate pass by either writer** — on test B is 3.50 % and A 19.54 % against a 2 % bar.", NOT_CLAIMED),
        ("> predictive.", "> predictive."),
    ):
        if old not in claims:
            raise SystemExit("claims missing: %s" % old[:70])
        claims = claims.replace(old, new, 1)
    start = claims.find("> We evaluate")
    end = claims.find("## 5. Remaining work")
    if start < 0 or end < 0:
        raise SystemExit("abstract block not found")
    claims = claims[:start] + ABSTRACT + "\n\n" + claims[end:]
    start = claims.find("**Known over-count that bound the numeric claims.**")
    if start >= 0:
        claims = claims[:start] + NOTE + "\n"
    claims = claims.replace("| **Test order runs (shuffled-1/2)** | whether B's 3.63 % numeric rate is order-stable",
                            "| **Test order runs (shuffled-1/2)** | whether B's 1.26 % numeric rate is order-stable")
    claims = claims.replace("today the reversal rests on kb alone. If B's rate swings with order, C1 must say \"on the kb order\" and the reversal weakens to a single-order finding",
                            "today C1's test half rests on the kb order alone. If B's rate swings with order, the 1.26 % figure must be quoted as \"on the kb order\"")
    claims = claims.replace("whether the adapter's numeric advantage is seed-stable", "whether the adapter's gate compliance is seed-stable")
    claims = claims.replace("a seed that jumps above A's rate turns C1 from \"B is more faithful\" into \"B is more faithful at seed 42\"",
                            "a seed above the 2 % gate turns C1 from \"meets the budget\" into \"meets it at seed 42\"")
    CLAIMS.write_text(claims, encoding="utf-8")

    achieved = ACHIEVED.read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new in (("**2.50% (13 / 520)**", "**0.00% (0 / 520)**"),
                     ("**9.62% (50 / 520)**", "**5.77% (30 / 520)**"),
                     ("• 2.50% Numeric Defects", "• 0.00% Numeric Defects"),
                     ("• 9.62% Numeric Defects", "• 5.77% Numeric Defects"),
                     ("Fallback: 2.50% vs Adapter: 9.62%", "Fallback: 0.00% vs Adapter: 5.77%")):
        if old not in achieved:
            raise SystemExit("achieved missing: %s" % old)
        achieved = achieved.replace(old, new, 1)
    ACHIEVED.write_text(achieved, encoding="utf-8")

    body = DOC.read_text(encoding="utf-8").replace("\r\n", "\n")
    if "## 34. Identifier over-count fixed" not in body:
        DOC.write_text(body.rstrip("\n") + "\n" + HANDOVER, encoding="utf-8")
    print("claims", len(CLAIMS.read_text(encoding="utf-8").splitlines()),
          "achieved", len(ACHIEVED.read_text(encoding="utf-8").splitlines()),
          "handover", len(DOC.read_text(encoding="utf-8").splitlines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
