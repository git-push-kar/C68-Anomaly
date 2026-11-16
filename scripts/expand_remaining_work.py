"""Replace §5 of the claims document with an explicit task-to-claim impact table."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "docs/reports/PAPER_CLAIMS_2026-11-16.md"

OLD = """## 5. Remaining work before submission

Order runs on test (shuffled-1/2), E scoring plus its integrity check, the detector-frontier table, the
follow-up T condition, parity resolution, and release packaging. The Version A/B paper choice is still
unsigned by the owner; nothing here presupposes it."""

NEW = """## 5. Remaining work, and what each item can change in the claims

Nothing below blocks submission of the claims as written, but each one bounds how strongly a claim can be
phrased. "Impact" names the claim that would move if the task came back badly.

| task | what it settles | impact if it comes back badly | cost |
|---|---|---|---|
| **Test order runs (shuffled-1/2)** | whether B's 3.63 % numeric rate is order-stable, or an artefact of the one menu order measured | **C1**: today the reversal rests on kb alone. If B's rate swings with order, C1 must say "on the kb order" and the reversal weakens to a single-order finding | ≈6.3 h GPU |
| **E scoring (seeds 43/44 on 520 rows)** | whether the adapter's numeric advantage is seed-stable | **C1, C3**: a seed that behaves like seed 42 keeps C1; a seed that jumps above A's rate turns C1 from "B is more faithful" into "B is more faithful at seed 42" | ≈0.9 h GPU per seed plus scoring |
| **Seed-43/44 integrity check** | whether those adapters are genuinely distinct runs after the 2026-11-05 checkpoint incident | **E claims only**: until cleared, E cannot be reported at all. Clearing it changes no existing claim | minutes |
| **Detector-frontier table (7)** | how the prose layer compares with detector-only and classifier-only front-ends | **none of C1–C8**: it is a new table. Its absence costs a comparison, not a claim | a build, plus P1 arm table |
| **Follow-up condition T** | whether store-backed answers beat router-only answers on follow-ups | **no claim**: dialogue is not claimed in the paper. T would only fill the report's §5 | ≈2 h |
| **Parity (1,390 disagreements)** | whether every number was produced by one instrument | **C1–C6 provenance**: unresolved, each figure must keep the scoring path named beside it. Resolving it either confirms the numbers or replaces them with one path's | instrument work, unknown |
| **Release packaging (D10)** | whether a reader can rebuild the artefacts | **no numeric claim**: affects reproducibility, not values | ≈1 h |
| **Version A/B signature** | which framing the paper uses (with or without the preference and mechanism sections) | **C4, C5**: Version A reports them descriptively, Version B omits them. Neither changes C1–C3 | owner decision |

**Known over-count that bound the numeric claims.** The checker still reads digits inside identifiers
(`FAULT_08` → `08`, `Component_D_Stream9` → `9`), so **every numeric-defect rate in this paper is an upper
bound**. Fixing it lowers A more than B, so the direction of C1 is safe; the absolute rates would move and
the 2 % gate would still fail for both writers on test."""


def main() -> int:
    text = CLAIMS.read_text(encoding="utf-8").replace("\r\n", "\n")
    if OLD not in text:
        print("MISS: remaining-work section not found")
        return 2
    CLAIMS.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("claims lines", len(CLAIMS.read_text(encoding="utf-8").splitlines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
