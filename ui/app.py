"""Streamlit UI prototype for the TEP RCA system.

Live sensor status, anomaly alerts, automatic reports, and follow-up chat.
Runs the TEPApp directly in-process (no separate API server required), which
keeps the prototype simple; the FastAPI backend (api/server.py) exposes the
same functionality over HTTP.

Run with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import TEPApp  # noqa: E402
from preprocessing.tep_loader import load_single_csv  # noqa: E402
from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="TEP RCA System", layout="wide")

CONFIG = load_config()


@st.cache_resource
def get_app() -> TEPApp:
    return TEPApp(config=CONFIG, enable_llm=True)


app = get_app()

st.title("Tennessee Eastman Process — Anomaly Detection & Root-Cause Analysis")
st.caption("Unsupervised LSTM autoencoder + InternVL2-2B (tep_rca adapter)")

detector_status = app.detector.describe()
st.sidebar.header("System status")
st.sidebar.write(f"Detector threshold: {detector_status['threshold']:.5f}")
st.sidebar.write(f"LLM adapter loaded: {'YES' if app.rca is not None else 'NO (fallback reports)'}")
st.sidebar.write(f"Event store: {CONFIG['events']['db_path']}")

tab_live, tab_events, tab_chat = st.tabs(["Live stream", "Anomaly events", "Follow-up chat"])

with tab_live:
    st.subheader("Live sensor status")
    col1, col2 = st.columns([2, 1])
    with col1:
        normal_source = st.text_input("Normal CSV source", value=CONFIG["streaming"]["normal_source"])
        use_fault = st.checkbox("Inject a fault scenario")
        fault_source = st.text_input(
            "Fault CSV source",
            value=CONFIG["streaming"]["fault_source"] if use_fault else "",
            disabled=not use_fault,
        )
        inject_at = st.number_input("Inject fault at sample index", value=800, min_value=0)
    with col2:
        n_records = st.number_input("Records to replay", value=1600, min_value=100)
        run_stream = st.button("Run live simulation")

    status_placeholder = st.empty()
    score_placeholder = st.empty()
    if run_stream:
        try:
            df = load_single_csv(normal_source, CONFIG)
            fault_df = None
            if use_fault and fault_source:
                fault_df = load_single_csv(fault_source, CONFIG)

            progress = st.progress(0.0)
            events_detected = []
            for i in range(int(n_records)):
                values = df.iloc[i % len(df)].to_numpy(dtype=np.float32)
                fault_label = None
                if fault_df is not None and i >= inject_at:
                    values = fault_df.iloc[(i - inject_at) % len(fault_df)].to_numpy(dtype=np.float32)
                    fault_label = _fault_id(fault_source)
                result = app.process_sensor_stream(
                    {"values": values, "sample_index": i, "fault_label": fault_label}
                )
                if result["anomaly_detected"]:
                    events_detected.append(result)
                progress.progress(min(1.0, (i + 1) / n_records))
                status_placeholder.write(
                    f"Window {result['window_id']} | score {result['window_score']} | "
                    f"anomalous: {result['is_anomalous']} | open event: {result['open_event']}"
                )
                score_placeholder.metric("Last window score", f"{result['window_score'] or 0:.5f}")

            st.success(f"Simulation finished. {len(events_detected)} anomaly event(s) closed.")
            for ev in events_detected:
                st.warning(f"Anomaly event {ev['event_id']} (severity {ev['report']['severity']})")
        except Exception as exc:
            st.error(f"Stream failed: {exc}")

with tab_events:
    st.subheader("Anomaly events")
    events = app.event_store.list_events(limit=50)
    if not events:
        st.info("No events yet. Run the live stream first.")
    for ev in events:
        evidence = ev.get("evidence", {})
        with st.expander(
            f"{ev['event_id']} | {evidence.get('severity', '?').upper()} | "
            f"score {ev.get('max_anomaly_score', 0):.3f}"
        ):
            st.json(ev)

with tab_chat:
    st.subheader("Ask a question about a detected anomaly")
    events = app.event_store.list_events(limit=50)
    if not events:
        st.info("No anomaly events available.")
    else:
        options = {ev["event_id"]: ev for ev in events}
        selected = st.selectbox("Select anomaly event", list(options.keys()))
        question = st.text_input("Ask a question about this anomaly...",
                                 placeholder="Why do you think the cooling system was the cause?")
        if st.button("Ask") and question:
            try:
                answer = app.answer_followup(selected, question)
                st.markdown(f"**Answer:**\n\n{answer['answer']}")
                st.caption("Answer stored in the event store.")
            except Exception as exc:
                st.error(f"Failed to answer: {exc}")


def _fault_id(path: str) -> int:
    import re

    match = re.search(r"fault[_-]?(\d+)", Path(path).stem, re.I)
    return int(match.group(1)) if match else None