import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import time
import numpy as np

from main import TEPApp, _infer_fault_label
from preprocessing.tep_loader import load_single_csv
from utils import load_config
from ui.state import init_app_state, get_state, set_state
from ui.components import render_metric_card, render_live_chart, show_event_details, severity_badge

st.set_page_config(page_title="TEP RCA System", layout="wide")

init_app_state()
CONFIG = load_config()

@st.cache_resource
def get_app() -> TEPApp:
    return TEPApp(config=CONFIG, enable_llm=True)

@st.cache_data
def load_data(normal_src: str, fault_src: str, use_fault: bool, fault_num: int):
    df = load_single_csv(normal_src, CONFIG)
    fault_df = None
    if use_fault and fault_src:
        fault_df = load_single_csv(fault_src, CONFIG, fault_number=fault_num, simulation_run=1)
    return df, fault_df

app = get_app()

# Load Events for UI
events_list = app.event_store.list_events(limit=100)

# 1. Page and status bar
st.title("TEP Anomaly Detection & RCA")

# Calculate metrics
plant_status = "NORMAL"
status_color = "success"
active_events = 0
for ev in events_list:
    if ev.get("status", "open") == "open":
        active_events += 1
        plant_status = "ALARM"
        status_color = "danger"

if not events_list and get_state("sim_history") and get_state("sim_history")[-1].get("is_anomalous"):
    plant_status = "WARNING"
    status_color = "warning"

latest_score = "0.000"
if get_state("sim_history"):
    latest_score = f"{get_state('sim_history')[-1].get('lstm_score', 0.0):.3f}"

llm_status = "Online" if app.rca is not None else "Fallback"

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Plant Status", plant_status, status_color)
with col2:
    render_metric_card("Latest LSTM Score", latest_score, "primary")
with col3:
    render_metric_card("Active Events", str(active_events), "warning" if active_events > 0 else "secondary")
with col4:
    render_metric_card("LLM Status", llm_status, "success" if llm_status == "Online" else "warning")

# 6. Sidebar
detector_status = app.detector.describe()
st.sidebar.header("System Status")
st.sidebar.markdown(f"**LSTM Threshold:** {detector_status['threshold']:.3f}", help="Reconstruction error limit. Higher means less sensitive.")
try:
    _q = (app.dynamic_runner.thresholds.get("cva", {}) or {}).get("Q", 0.0)
    st.sidebar.markdown(f"**CVA Threshold:** {_q:.1f}", help="Canonical Variate Analysis limit. Detects slow drifts and sticky valves.")
except Exception:
    pass
st.sidebar.markdown(f"**LLM Adapter:** {llm_status}")

with st.sidebar.expander("Known Limitations"):
    st.markdown("- Faults 3 and 9 are physically undetectable due to closed-loop controller compensation.")
    st.markdown("- Fault 15 requires CVA for reliable detection due to hysteresis.")

clear_hist = st.sidebar.checkbox("Confirm clear history")
if st.sidebar.button("Clear History", disabled=not clear_hist):
    app.event_store.db.execute("DELETE FROM events")
    app.event_store.db.commit()
    st.sidebar.success("Cleared!")
    time.sleep(1)
    st.rerun()

# 2. Navigation
tab_monitor, tab_history, tab_chat = st.tabs(["Monitor", "Incident History", "Event Assistant"])

# 3. Monitor
with tab_monitor:
    st.subheader("Live Plant Monitor")
    
    # Controls
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6 = st.columns([2, 1, 1, 2, 1, 2])
    with ctrl1:
        normal_source = st.text_input("Normal Data", value=CONFIG["streaming"]["normal_source"])
    with ctrl2:
        use_fault = st.checkbox("Inject Fault", value=False)
    with ctrl3:
        fault_id_sel = st.number_input("Fault ID", value=CONFIG["streaming"].get("fault_number", 6), min_value=1, max_value=22, disabled=not use_fault)
    with ctrl4:
        fault_source = st.text_input("Fault Data", value=CONFIG["streaming"]["fault_source"], disabled=not use_fault)
    with ctrl5:
        inject_at = st.number_input("Start Step", value=800, min_value=0, disabled=not use_fault)
    with ctrl6:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        speed = 10 # Hardcoded speed or we can put it back. Let's put speed somewhere.
        # Actually I can replace ctrl6 with speed and add ctrl7 for buttons.
        
    ctrl_speed, ctrl_btn = st.columns([1, 2])
    with ctrl_speed:
        speed = st.number_input("Steps/sec", value=10, min_value=1)
    with ctrl_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Start / Resume", use_container_width=True):
            set_state("sim_running", True)
        if b2.button("Pause / Stop", use_container_width=True):
            set_state("sim_running", False)
            
    # Non-blocking run using st.fragment if available, otherwise fallback
    # For compatibility, we'll run a small loop inside a fragment.
    @st.fragment(run_every=1)
    def run_simulation():
        if get_state("sim_running"):
            df, fault_df = load_data(normal_source, fault_source, use_fault, int(fault_id_sel))
            
            step = get_state("sim_step")
            history = get_state("sim_history")
            
            # Run chunks of steps based on speed
            for _ in range(int(speed)):
                if not get_state("sim_running"):
                    break
                
                values = df.iloc[step % len(df)].to_numpy(dtype=np.float32)
                fault_label = None
                fault_injected = False
                
                if use_fault and fault_df is not None and step >= inject_at:
                    values = fault_df.iloc[(step - inject_at) % len(fault_df)].to_numpy(dtype=np.float32)
                    fault_label = _infer_fault_label(fault_source)
                    fault_injected = True
                    
                result = app.process_sensor_stream(
                    {"values": values, "sample_index": step, "fault_label": fault_label}
                )
                
                # Extract CVA from detector_evidence if present
                det_ev = result.get("detector_evidence", {}) or {}
                lstm_score = result.get("window_score", 0.0)
                lstm_thr = app.detector.threshold.threshold
                
                cva_score = 0.0
                cva_thr = 0.0
                if det_ev:
                    cva_s = (det_ev.get("dynamic", {}) or {}).get("cva", {}) or {}
                    cva_score = cva_s.get("score", 0.0)
                    cva_thr = cva_s.get("threshold", 0.0)
                
                history.append({
                    "step": step,
                    "lstm_score": lstm_score,
                    "lstm_threshold": lstm_thr,
                    "cva_score": cva_score,
                    "cva_threshold": cva_thr,
                    "is_anomalous": result.get("is_anomalous", False),
                    "fault_injected": fault_injected
                })
                
                # Keep last 1000 steps to avoid memory issues
                if len(history) > 1000:
                    history.pop(0)
                
                step += 1
                
                if result.get("anomaly_detected") and result.get("event_id"):
                    set_state("new_anomaly_event", result.get("event_id"))
            
            set_state("sim_step", step)
            set_state("sim_history", history)
            
        new_ev = get_state("new_anomaly_event")
        if new_ev:
            st.error(f"🚨 CRITICAL: New Anomaly Event {new_ev} detected!")
            if st.button(f"View Incident {new_ev}"):
                set_state("selected_event_id", new_ev)
                set_state("new_anomaly_event", None)
                # Note: Streamlit cannot switch tabs programmatically, but we set the selected event for the Event Assistant.
                st.rerun()

        render_live_chart(get_state("sim_history"))
        
        if get_state("sim_running"):
            # Sleep slightly to prevent high CPU if speed is low, but fragment run_every=1 handles it mostly
            time.sleep(0.1)
            
    run_simulation()

# 4. Incident History
with tab_history:
    st.subheader("Incident History")
    if not events_list:
        st.info("No incidents recorded. Run the monitor and inject a fault to see incidents here.")
    else:
        # Table data
        table_data = []
        for ev in events_list:
            rep = ev.get("report", {})
            evidence = ev.get("evidence", {})
            det = evidence.get("detector_evidence", {}) or {}
            
            table_data.append({
                "Event ID": ev.get("event_id"),
                "Time": ev.get("detection_time"),
                "Severity": rep.get("severity", "unknown").upper(),
                "Detector": det.get("triggered_by", "-"),
                "Score": round(ev.get("max_anomaly_score", 0), 3),
                "Suspected Fault": rep.get("root_cause", "-"),
                "Status": ev.get("status", "open")
            })
            
        df_events = pd.DataFrame(table_data)
        
        # Filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            f_sev = st.multiselect("Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        with f_col2:
            f_det = st.multiselect("Detector", ["lstm_ae", "dynamic", "both"])
        with f_col3:
            f_search = st.text_input("Search Fault")
            
        if f_sev:
            df_events = df_events[df_events["Severity"].isin(f_sev)]
        if f_det:
            df_events = df_events[df_events["Detector"].isin(f_det)]
        if f_search:
            df_events = df_events[df_events["Suspected Fault"].str.contains(f_search, case=False, na=False)]
            
        with f_col4:
            st.markdown("<br>", unsafe_allow_html=True)
            csv = df_events.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV", data=csv, file_name="incidents.csv", mime="text/csv")
            
        st.markdown("Click a row to view incident details.")
        event_selection = st.dataframe(
            df_events, 
            use_container_width=True, 
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        sel_rows = event_selection.selection.rows
        if sel_rows:
            sel_ev_id = df_events.iloc[sel_rows[0]]["Event ID"]
            sel_ev_data = next((e for e in events_list if e.get("event_id") == sel_ev_id), None)
            if sel_ev_data:
                st.markdown("---")
                history = app.event_store.conversation_history(sel_ev_id)
                show_event_details(sel_ev_data, history)
                
                st.info("To ask the LLM about this event, go to the Event Assistant tab.")

# 5. Event Assistant
with tab_chat:
    st.subheader("Event Assistant")
    if not events_list:
        st.info("No events available to assist with.")
    else:
        options = {ev["event_id"]: ev for ev in events_list}
        event_keys = list(options.keys())
        
        sel_idx = 0
        state_selected = get_state("selected_event_id")
        if state_selected in event_keys:
            sel_idx = event_keys.index(state_selected)
            
        selected = st.selectbox("Select Anomaly Event", event_keys, index=sel_idx)
        if selected != state_selected:
            set_state("selected_event_id", selected)
        
        if selected:
            ev_data = options[selected]
            history = app.event_store.conversation_history(selected)
            show_event_details(ev_data, history)
            
            st.markdown("---")
            q = st.chat_input("Ask the LLM about this event...")
            if q:
                if llm_status == "Fallback":
                    st.warning("LLM is currently offline. Responses will be deterministic rule-based fallbacks.")
                
                # Show immediate user message
                st.chat_message("user").write(q)
                
                with st.spinner("Generating answer..."):
                    try:
                        ans = app.answer_followup(selected, q)
                        st.chat_message("assistant").write(ans["answer"])
                        st.rerun() # Refresh to show in history correctly
                    except Exception as e:
                        st.error(f"Error answering: {e}")