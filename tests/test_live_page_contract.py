"""Contract test: a live event must render to the benchmark page the classifier was measured on.

The strongest available form of the parity gate. Each stored benchmark prompt is parsed back into an evidence
dict, run through the live mapping layer and the benchmark renderer, and the output must equal the stored page
byte for byte (with the opaque event id normalised, since it is a hash of the internal id).

If this passes, the classifier sees the same page live that it saw in the benchmark. If it fails, no accuracy
number from the live path means anything.
"""
from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark.live_page import (  # noqa: E402
    CANONICAL_ORDER,
    ONSET_LIMIT,
    IncompleteEvidence,
    derive_lag_attribution,
    detector_evidence_ready,
    normalise_evidence,
    render_classifier_page,
    render_live_page,
    require_detector_evidence,
)
from llm.benchmark.prompt_v2 import render_prompt  # noqa: E402
from scripts.evidence_classifier import prompt_features  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
DB = ROOT / "outputs/events.db"

EVENT_LINE = re.compile(r"^Event: .*$", re.M)
SENSOR_LINE = re.compile(r"^- (.+?) \[(X(?:MEAS|MV)_\d{1,2}), [^\]]+\]: z=(-?[\d.]+), "
                         r"deviation (-?[\d.]+)%, trend (\w+)")
RESIDUAL_LINE = re.compile(r"^- (X(?:MEAS|MV)_\d{1,2}) \[[^\]]+\]: residual share ([\d.]+)")
ONSET_LINE = re.compile(r"^\d+\. (.+?) \[(X(?:MEAS|MV)_\d{1,2}), [^\]]+\] at \+([\d.]+) min")
MENU_LINE = re.compile(r"^- (FAULT_\d{1,2}|UNCLASSIFIED_OPEN_SET): (.+)$")


def parse_page(page: str) -> tuple[dict, list[dict]]:
    """Recover an evidence dict and menu from a rendered benchmark page."""
    evidence: dict = {"detector_evidence": {}, "top_anomalous_sensors": [], "temporal_sequence": []}
    menu: list[dict] = []
    residual: list[dict] = []
    in_menu = False

    for line in page.splitlines():
        if line.startswith("Event: "):
            evidence["event_id"] = line.split("Event: ", 1)[1].strip()
            continue
        if "Overall anomaly score:" in line:
            evidence["anomaly_score"] = float(line.split(":", 1)[1])
            continue
        if line.startswith("CANDIDATE MENU"):
            in_menu = True
            continue
        if line.startswith("Return one JSON object"):
            in_menu = False
            continue
        if in_menu:
            match = MENU_LINE.match(line)
            if match:
                menu.append({"fault_id": match.group(1), "signature": match.group(2)})
            continue

        match = SENSOR_LINE.match(line)
        if match:
            evidence["top_anomalous_sensors"].append({
                "display_name": match.group(1), "name": match.group(2),
                "z_score": float(match.group(3)), "deviation_percent": float(match.group(4)),
                "trend": match.group(5)})
            continue
        match = ONSET_LINE.match(line)
        if match:
            evidence["temporal_sequence"].append({
                "display_name": match.group(1), "sensor": match.group(2),
                "relative_time_minutes": float(match.group(3))})
            continue
        match = RESIDUAL_LINE.match(line)
        if match:
            residual.append({"variable": match.group(1), "share": float(match.group(2))})
            continue
        if line.startswith("- LSTM autoencoder reconstruction error"):
            numbers = re.search(r"error ([\d.]+) \(threshold ([\d.]+), ratio ([\d.]+)x, alarmed (\w+)\)", line)
            evidence["detector_evidence"]["lstm_ae"] = {
                "score": float(numbers.group(1)), "threshold": float(numbers.group(2)),
                "alarmed": numbers.group(4) == "True"}
            continue
        if line.startswith("- CVA Q-statistic"):
            numbers = re.search(r"statistic ([\d.]+) \(threshold ([\d.]+), ratio ([\d.]+)x, alarmed (\w+)\)", line)
            evidence["detector_evidence"]["dynamic"] = {
                "score": float(numbers.group(1)), "threshold": float(numbers.group(2)),
                "alarmed": numbers.group(4) == "True"}
            continue
        if line.startswith("- Fused trigger:"):
            trigger = line.split("Fused trigger:", 1)[1].strip()
            trig, _, rest = trigger.partition(" (")
            evidence["detector_evidence"]["triggered_by"] = trig.strip()
            evidence["detector_evidence"]["change_type"] = rest.replace(" change)", "").strip()
            continue
        if line.startswith("Dominant lag"):
            numbers = re.search(r"Dominant lag (\d+) window\(s\), slow-drift ratio ([\d.]+)", line)
            evidence["detector_evidence"]["lag_attribution"] = {
                "dominant_lag": int(numbers.group(1)), "slow_drift_ratio": float(numbers.group(2)),
                "top_variables": residual}

    if residual and "lag_attribution" not in evidence["detector_evidence"]:
        evidence["detector_evidence"]["lag_attribution"] = {"top_variables": residual}
    return evidence, menu


RATION = re.compile(r"ratio [\d.]+x")


def normalise_event_line(page: str) -> str:
    """Drop the two things a page cannot reproduce from its own text.

    The event id is a hash of the internal id, and the CVA ratio is computed from *unrounded* score and
    threshold, so dividing the rounded operands gives a different last digit. Both are normalised here; the
    ratio's consistency is asserted separately in ``test_cva_ratio_is_consistent_with_its_operands``.
    """
    page = EVENT_LINE.sub("Event: <id>", page)
    return RATION.sub("ratio <r>x", page).rstrip()


def stored_pages(limit: int = 20) -> list[str]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    step = max(1, len(rows) // limit)
    return [row["prompt"] for row in rows[::step][:limit]]


@pytest.mark.parametrize("page", stored_pages(), ids=range(20))
def test_round_trip_reproduces_the_stored_page(page: str) -> None:
    evidence, menu = parse_page(page)
    rendered = render_live_page(evidence, menu, event_id=evidence["event_id"])
    assert normalise_event_line(rendered) == normalise_event_line(page)


@pytest.mark.parametrize("page", stored_pages(5), ids=range(5))
def test_cva_ratio_is_consistent_with_its_operands(page: str) -> None:
    evidence, menu = parse_page(page)
    rendered = render_live_page(evidence, menu, event_id=evidence["event_id"])
    dynamic = evidence["detector_evidence"]["dynamic"]
    expected = dynamic["score"] / dynamic["threshold"]
    shown = float(re.search(r"CVA Q-statistic .*?ratio ([\d.]+)x", rendered).group(1))
    assert abs(shown - expected) / expected < 0.01


#: The two ratio features (LSTM and CVA) cannot survive a text round trip: the stored page prints score and
#: threshold rounded and divides the *unrounded* pair, so recomputing from the printed operands differs in the
#: last digit. Every other feature must match exactly, and both ratios must agree to 1 %.
RATIO_FEATURES = (2, 4)


def test_round_trip_reproduces_the_feature_vector() -> None:
    for page in stored_pages(10):
        evidence, menu = parse_page(page)
        rendered = render_live_page(evidence, menu, event_id=evidence["event_id"])
        stored = np.asarray(prompt_features(page))
        produced = np.asarray(prompt_features(rendered))
        for index in RATIO_FEATURES:
            assert abs(produced[index] - stored[index]) / stored[index] < 0.01
        keep = [i for i in range(stored.size) if i not in RATIO_FEATURES]
        assert np.array_equal(stored[keep], produced[keep])


def live_shaped_evidence(onsets: int = 6) -> dict:
    """An evidence dict shaped like ``events.db``: no lag_attribution, CVA nested, six onsets."""
    contributions = [0.0] * 52
    for index, value in ((15, 9.0), (43, 7.0), (17, 5.0), (22, 3.0), (50, 2.0), (4, 1.0), (9, 0.5)):
        contributions[index] = value
    return {
        "event_id": "ANOM-TEST",
        "anomaly_score": 118.627,
        "detector_evidence": {
            "triggered_by": "both",
            "change_type": "mixed",
            "lstm_ae": {"score": 17.5333, "threshold": 0.687, "alarmed": True},
            "dynamic": {"cva": {"score": 10960.5, "threshold": 194.6, "alarmed": True,
                                "sensor_contributions": contributions,
                                "dominant_lag": 4, "slow_drift_ratio": 0.51}},
        },
        "top_anomalous_sensors": [
            {"display_name": "A_Feed_Stream1", "name": "XMEAS_1", "z_score": 18.12,
             "deviation_percent": 223.37, "trend": "decreasing"},
            {"display_name": "A_Feed_Flow", "name": "XMV_44", "z_score": 18.10,
             "deviation_percent": 223.21, "trend": "decreasing"},
        ],
        "temporal_sequence": [{"display_name": f"Sensor_{i}", "sensor": f"XMEAS_{i + 1}",
                               "relative_time_minutes": 0.1 * i} for i in range(onsets)],
    }


def test_lag_attribution_is_derived_from_cva_contributions() -> None:
    evidence = live_shaped_evidence()
    lag = derive_lag_attribution(evidence)
    assert lag is not None
    assert len(lag["top_variables"]) == 5
    shares = [entry["share"] for entry in lag["top_variables"]]
    assert shares == sorted(shares, reverse=True)
    # The top five of 27.5 units of contribution carry 26 of them; the rest is deliberately not listed.
    assert abs(sum(shares) - 26.0 / 27.5) < 1e-6
    assert lag["top_variables"][0]["variable"] == CANONICAL_ORDER[15]
    assert lag["dominant_lag"] == 4
    assert lag["slow_drift_ratio"] == 0.51


def test_existing_lag_attribution_is_kept() -> None:
    evidence = live_shaped_evidence()
    evidence["detector_evidence"]["lag_attribution"] = {
        "top_variables": [{"variable": "XMEAS_16", "share": 0.1}], "dominant_lag": 3,
        "slow_drift_ratio": 0.4}
    lag = derive_lag_attribution(evidence)
    assert lag["dominant_lag"] == 3
    assert lag["top_variables"][0]["variable"] == "XMEAS_16"


def test_onset_list_is_preserved_not_truncated() -> None:
    """A five-entry cap was the working hypothesis and it was wrong: 12 of 20 stored pages carry six."""
    evidence = live_shaped_evidence(onsets=6)
    mapped = normalise_evidence(evidence)
    assert len(mapped["temporal_sequence"]) == 6
    page = render_live_page(evidence)
    assert len(re.findall(r"^\d+\. .+ at \+[\d.]+ min", page, re.M)) == 6
    assert ONSET_LIMIT is None


def test_cva_score_is_hoisted_so_the_page_carries_the_cva_line() -> None:
    page = render_live_page(live_shaped_evidence())
    assert "- CVA Q-statistic 10960.5 (threshold 194.6, ratio 56.32x, alarmed True)" in page


def test_rendered_live_page_carries_the_residual_block() -> None:
    page = render_live_page(live_shaped_evidence())
    assert page.count("residual share") == 5
    assert "Dominant lag 4 window(s), slow-drift ratio 0.51" in page


def test_rendered_live_page_carries_the_teacher_instruction_before_the_schema() -> None:
    page = render_live_page(live_shaped_evidence())
    instruction = ("Legal answers: " + ", ".join(
        ["FAULT_%02d" % i for i in range(1, 16)] + ["UNCLASSIFIED_OPEN_SET"]) + ".")
    assert instruction + "\n\nReturn one JSON object" in page


def legacy_evidence() -> dict:
    """A pre-instrumentation record: schema v1, no detector block at all."""
    return {
        "event_id": "ANOM-0333",
        "anomaly_score": 118.627,
        "severity": "high",
        "candidate_subsystem": "Stripper",
        "top_anomalous_sensors": [
            {"display_name": "Separator_Level", "name": "XMEAS_12", "z_score": 0.08,
             "deviation_percent": 0.15, "trend": "stable"}],
        "temporal_sequence": [],
    }


def test_legacy_record_is_not_classifier_ready() -> None:
    evidence = legacy_evidence()
    assert detector_evidence_ready(evidence) is False
    with pytest.raises(IncompleteEvidence) as raised:
        require_detector_evidence(evidence)
    assert "ANOM-0333" in str(raised.value)
    assert "detector_evidence" in str(raised.value)


def test_legacy_record_still_renders_for_display() -> None:
    page = render_live_page(legacy_evidence())
    assert "Separator_Level [XMEAS_12, measurement]: z=0.08" in page
    assert page.count("residual share") == 0


def test_legacy_record_is_refused_by_the_classifier_page() -> None:
    with pytest.raises(IncompleteEvidence):
        render_classifier_page(legacy_evidence())


def test_empty_detector_block_is_refused() -> None:
    evidence = live_shaped_evidence()
    evidence["detector_evidence"] = {}
    assert detector_evidence_ready(evidence) is False
    with pytest.raises(IncompleteEvidence):
        render_classifier_page(evidence)


def test_one_detector_score_is_enough() -> None:
    """An LSTM-only event was instrumented; refusing it would discard real evidence."""
    evidence = live_shaped_evidence()
    del evidence["detector_evidence"]["dynamic"]
    assert detector_evidence_ready(evidence) is True
    assert "LSTM autoencoder reconstruction error" in render_classifier_page(evidence)


def test_normalise_is_idempotent() -> None:
    once = normalise_evidence(live_shaped_evidence())
    twice = normalise_evidence(once)
    assert once == twice


@pytest.mark.skipif(not DB.exists(), reason="no stored live events")
def test_real_live_events_populate_the_expected_features() -> None:
    con = sqlite3.connect(DB)
    evidence_rows = [json.loads(value) for (value,) in
                     con.execute("select evidence_json from events order by rowid desc limit 10")]
    con.close()
    populated = []
    for evidence in evidence_rows:
        page = render_live_page(evidence)
        assert page.count("residual share") in (0, 5)
        populated.append(int(np.count_nonzero(prompt_features(page))))
    assert len(populated) == len(evidence_rows)
    assert min(populated) >= 20, populated
