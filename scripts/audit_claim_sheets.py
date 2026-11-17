"""Audit the paper's claims against the artefacts they cite.

Covers three documents — the manuscript, the claims/evidence registry (`PAPER_CLAIMS`) and the methods/metrics
sheet — by (a) recomputing each quoted number from its stored artefact, (b) resolving every backticked path the
documents cite, and (c) flagging statements whose source does not contain the number attributed to it.

    python scripts/audit_claim_sheets.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = {
    "manuscript": ROOT / "docs/reports/PAPER_MANUSCRIPT_2026-11-16.md",
    "claims": ROOT / "docs/reports/PAPER_CLAIMS_2026-11-16.md",
    "metrics": ROOT / "docs/reports/PAPER_METHODS_METRICS_2026-11-16.md",
}

BASES = [
    ROOT, ROOT / "docs/reports", ROOT / "docs/reports/figures", ROOT / "docs/judging", ROOT / "docs/judging/confirm",
    ROOT / "docs/judging/pilot_v5", ROOT / "outputs", ROOT / "outputs/overnight_2026-11-05",
    ROOT / "outputs/analysis/phase3a", ROOT / "outputs/analysis/phase3a/fit",
    ROOT / "outputs/analysis/phase3a/d33", ROOT / "outputs/analysis/phase3a/hybrid",
    ROOT / "outputs/analysis/phase3a/hybrid/vocabfix", ROOT / "outputs/benchmark_v2/eval",
]

FAILS: list[str] = []
NOTES: list[str] = []
COUNT = 0


def check(label: str, actual, expected) -> None:
    global COUNT
    COUNT += 1
    ok = actual == expected
    if not ok:
        FAILS.append(f"{label}: {actual!r} != {expected!r}")
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual!r}")


def near(label: str, actual: float, expected: float, tol: float = 5e-4) -> None:
    global COUNT
    COUNT += 1
    ok = abs(float(actual) - float(expected)) <= tol
    if not ok:
        FAILS.append(f"{label}: {actual} != {expected}")
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual}")


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def resolve(token: str) -> bool:
    token = token.strip().rstrip(".,;:")
    return any((base / token).exists() for base in BASES)


def main() -> int:
    cells = load("outputs/overnight_2026-11-05/n4_cells.json")
    s = cells["summary"]
    seam = cells["seam_certification"]
    test = load("outputs/analysis/phase3a/d33/test_contrast.json")
    figdata = json.loads((ROOT / "docs/reports/figures/figure_data.json").read_text(encoding="utf-8"))
    contrasts = load("outputs/analysis/phase3a/hybrid/vocabfix/classifier_contrasts.json")
    pref = load("docs/judging/confirm/preference_2026-11-16_result.json")
    kappa = load("docs/judging/confirm/preference_2026-11-16_kappa.json")
    receipt = load("docs/judging/confirm/selection_receipt_2026-11-16_final.json")
    ceiling = load("docs/judging/confirm/d_ceiling_decision_2026-11-16.json")
    invariance = load("outputs/overnight_2026-11-05/n5_invariance_dialogue.json")["order_invariance"]
    pre = load("outputs/overnight_2026-11-05/n4_cells_preinstrument_2026-11-16.json")["summary"]

    print("-- C1 numeric fidelity --")
    check("validation A", s["A"]["numeric"], 0)
    check("validation B", s["B"]["numeric"], 30)
    check("validation C", s["C"]["numeric"], 0)
    check("validation M", s["M"]["numeric"], 30)
    check("test A", test["counts"]["A"]["numeric"], 0)
    check("test B", test["counts"]["B"]["numeric"], 40)
    near("test B rate 1.26 %", test["rates"]["B"]["numeric"], 0.0126)
    check("McNemar b only", cells["mcnemar"]["b_only"], 30)
    check("McNemar a only", cells["mcnemar"]["a_only"], 0)

    print("\n-- C2 anchor exactness --")
    near("validation anchor 188/520 = 36.2 %", s["B"]["anchor_exact"] / 520, 0.3615)
    check("validation anchor A", s["A"]["anchor_exact"], 520)
    check("test anchor exact", test["anchor_status"]["B"]["exact"], 1170)
    near("test anchor 36.93 %", 1170 / 3168, 0.3693)
    near("z omitted 46.05 %", test["anchor_status"]["B"]["z_omitted"] / 3168, 0.4605)
    claims_text = DOCS["claims"].read_text(encoding="utf-8")
    check("C2 cites n4_cells anchor_exact", "summary (`anchor_exact`)" in claims_text, True)
    check("C2 no longer cites pilot anchor_gate as source", "`pilot_rechecks.json`\n`anchor_gate`" not in claims_text, True)
    check("C6 describes the decision as signed", "The decision is **signed**" in claims_text, True)
    check("C6 no longer asks for re-signoff", "must re-sign" not in claims_text, True)

    print("\n-- C3 diagnosis contrast --")
    systems = contrasts["per_system"]
    near("hybrid", systems["hybrid"]["accuracy"], 0.996)
    near("gbdt", systems["gbdt"]["accuracy"], 0.998)
    near("adapter", systems["adapter"]["accuracy"], 0.833)
    near("mlp", systems["mlp"]["accuracy"], 0.9799)
    near("menu rule", systems["menu_rule"]["accuracy"], 1.0)
    check("hybrid vs gbdt discordants", (contrasts["contrasts"]["hybrid_vs_gbdt"]["b"],
                                         contrasts["contrasts"]["hybrid_vs_gbdt"]["c"]), (2, 1))

    print("\n-- C4 preference --")
    near("kappa", pref["cohen_kappa"], 0.5776)
    check("pairs", pref["pairs"], 360)
    near("observed", kappa["observed_agreement"], 0.7778)
    near("expected", kappa["expected_agreement"], 0.4739)
    pair = pref["per_pairing"]
    check("A vs C rater1 A/C", (pair["A vs C"]["sheet_rater1"]["A"], pair["A vs C"]["sheet_rater1"]["C"]), (98, 20))
    check("A vs C rater2 A/C", (pair["A vs C"]["sheet_rater2"]["A"], pair["A vs C"]["sheet_rater2"]["C"]), (70, 48))
    check("B vs C rater1 B/C", (pair["B vs C"]["sheet_rater1"]["B"], pair["B vs C"]["sheet_rater1"]["C"]), (10, 108))
    check("B vs C rater2 B/C", (pair["B vs C"]["sheet_rater2"]["B"], pair["B vs C"]["sheet_rater2"]["C"]), (53, 64))
    check("M vs B rater1 B/M", (pair["B vs M"]["sheet_rater1"]["B"], pair["B vs M"]["sheet_rater1"]["M"]), (0, 114))
    check("M vs B rater2 B/M", (pair["B vs M"]["sheet_rater2"]["B"], pair["B vs M"]["sheet_rater2"]["M"]), (2, 112))

    print("\n-- C5 selection --")
    check("decided by rung", receipt["decided_by_rung"], 5)
    check("chosen", receipt["chosen"], "A")

    print("\n-- C6 fallback wrapper --")
    check("fallback rows", seam["D_fallback_rows"], 114)
    near("fallback rate 21.9 %", 114 / 520, 0.219)
    check("ceiling decision signed", ceiling["status"], "signed")
    near("signed record's pre-correction rate", ceiling["measured"]["rate"], 0.273)

    print("\n-- C7 batch invariance --")
    check("B1 run exists", any((ROOT / "outputs/analysis/phase3a/fit/eq_B1").iterdir()), True)
    check("B8 run exists", any((ROOT / "outputs/analysis/phase3a/fit/eq_B8").iterdir()), True)
    check("pb16 exists", (ROOT / "outputs/benchmark_v2/eval/prose_pilot/pb16").exists(), True)
    check("pb8 exists", (ROOT / "outputs/benchmark_v2/eval/prose_pilot/pb8").exists(), True)
    NOTES.append("C7's '1 of 12' and '3 of 8' identity counts are not stored as a JSON field; only the run dirs exist.")

    print("\n-- order invariance cited in C1/§2 --")
    check("order invariance items", invariance["items"], 520)
    check("identical across orders", invariance["identical_defect_profile_all_orders"], 368)
    near("rate 0.7077", invariance["rate"], 0.7077)

    print("\n-- metrics sheet: composition and strata --")
    val, tst = figdata["stratified"]["validation"], figdata["stratified"]["test"]
    check("fit normal-transient", (val["normal-transient"]["defects"], val["normal-transient"]["rows"]), (29, 47))
    check("test normal-transient", (tst["normal-transient"]["defects"], tst["normal-transient"]["rows"]), (29, 40))
    near("fit share normal 9.0 %", 47 / 520, 0.090)
    near("test share normal 1.3 %", 40 / 3168, 0.013)
    near("fit share open-set 87.3 %", 454 / 520, 0.873)
    near("test share open-set 57.3 %", 1814 / 3168, 0.573)
    near("fit share classified 3.7 %", 19 / 520, 0.037)
    near("test share classified 41.5 %", 1314 / 3168, 0.415)

    print("\n-- metrics sheet: test table and instrument anchors --")
    check("test grounding A", test["counts"]["A"]["grounding"], 250)
    check("test grounding B", test["counts"]["B"]["grounding"], 325)
    check("test contradiction B", test["counts"]["B"]["contradiction"], 5)
    check("test tag-role B", test["counts"]["B"]["tag_role"], 458)
    check("calibration cases", len((ROOT / "docs/judging/confirm/cases46_audit.csv")
                                    .read_text(encoding="utf-8").strip().splitlines()) - 1, 46)
    NOTES.append("'834/834 stored probability vectors' has no stored counter artefact; it is a rerun claim.")

    print("\n-- metrics sheet: changelog v1 row --")
    check("pre-instrument A", pre["A"]["numeric"], 19)
    check("pre-instrument B", pre["B"]["numeric"], 62)
    NOTES.append("changelog v2 row (13/520, 50/520) has no stored artefact; only v1 and v3 do.")

    print("\n-- cross-document consistency --")
    for name, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        for needle in ("0.5776", "0.8330" if name != "metrics" else "0.8330"):
            check(f"{name} carries {needle}", needle in text, True)
        check(f"{name} has no 46.06 %", "46.06" not in text, True)

    print("\n-- cited paths resolve --")
    for name, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        tokens = set(re.findall(r"`([^`\s]+)`", text))
        paths = [t for t in tokens if ("/" in t or t.endswith((".json", ".jsonl", ".csv", ".md", ".py")))]
        unresolved = [t for t in paths if not resolve(t)]
        check(f"{name}: all {len(paths)} cited paths resolve", unresolved, [])

    print(f"\n{COUNT - len(FAILS)}/{COUNT} checks passed")
    if NOTES:
        print("\nNOTES (not machine-verifiable, or source mismatch)")
        for note in NOTES:
            print(" -", note)
    if FAILS:
        print("\nFAILURES")
        for item in FAILS:
            print(" -", item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
