import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Any

def render_metric_card(title: str, value: str, color: str = "primary"):
    # Simple metric card using markdown
    color_map = {
        "primary": "blue",
        "success": "green",
        "warning": "orange",
        "danger": "red",
        "secondary": "gray"
    }
    c = color_map.get(color, "gray")
    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:15px; text-align:center; background-color: rgba(255, 255, 255, 0.05);">
            <h4 style="margin:0; font-size:14px; color:#888;">{title}</h4>
            <h2 style="margin:5px 0 0 0; font-size:24px; color:{c};">{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def severity_badge(severity: str) -> str:
    color_map = {
        "low": "blue",
        "medium": "orange",
        "high": "red",
        "critical": "darkred",
        "unknown": "gray"
    }
    s = str(severity).lower()
    c = color_map.get(s, "gray")
    return f'<span style="background-color:{c}; color:white; padding:3px 8px; border-radius:12px; font-size:12px; font-weight:bold;">{s.upper()}</span>'

def render_live_chart(history: List[Dict[str, Any]]):
    if not history:
        st.info("No data to display. Start the simulation.", icon="ℹ️")
        return

    df = pd.DataFrame(history)
    fig = go.Figure()

    # Determine thresholds. Using mean of threshold in history to draw a straight line if it varies,
    # or just the max threshold. Usually it's constant.
    if "lstm_score" in df.columns:
        fig.add_trace(go.Scatter(x=df["step"], y=df["lstm_score"], mode="lines", name="LSTM Score", line=dict(color="blue")))
    
    if "cva_score" in df.columns:
        # Normalize CVA score or put on secondary axis? The requirement says "score vs time, dashed threshold line per detector".
        # Let's put them on the same axis for now or secondary y.
        fig.add_trace(go.Scatter(x=df["step"], y=df["cva_score"], mode="lines", name="CVA Q Score", line=dict(color="purple"), yaxis="y2"))
    elif "score" in df.columns:
        fig.add_trace(go.Scatter(x=df["step"], y=df["score"], mode="lines", name="Fusion Score", line=dict(color="blue")))

    if "lstm_threshold" in df.columns and df["lstm_threshold"].notnull().any():
        lstm_thr = df["lstm_threshold"].iloc[0]
        fig.add_hline(y=lstm_thr, line_dash="dash", line_color="blue", annotation_text="LSTM Thr")
    
    if "cva_threshold" in df.columns and df["cva_threshold"].notnull().any():
        cva_thr = df["cva_threshold"].iloc[0]
        fig.add_hline(y=cva_thr, line_dash="dash", line_color="purple", annotation_text="CVA Thr", yref="y2")
    elif "threshold" in df.columns and df["threshold"].notnull().any():
        thr = df["threshold"].iloc[0]
        fig.add_hline(y=thr, line_dash="dash", line_color="red", annotation_text="Threshold")

    # Add shaded regions for anomalies
    anomalous_regions = df[df["is_anomalous"] == True]
    if not anomalous_regions.empty:
        # Group adjacent steps
        anomalies_grouped = []
        current_group = []
        for step in anomalous_regions["step"]:
            if not current_group:
                current_group.append(step)
            elif step == current_group[-1] + 1:
                current_group.append(step)
            else:
                anomalies_grouped.append(current_group)
                current_group = [step]
        if current_group:
            anomalies_grouped.append(current_group)
            
        for g in anomalies_grouped:
            fig.add_vrect(
                x0=g[0], x1=g[-1],
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
            )

    # Add vertical line for injected fault if present
    fault_steps = df[df["fault_injected"] == True]
    if not fault_steps.empty:
        first_fault_step = fault_steps["step"].iloc[0]
        fig.add_vline(x=first_fault_step, line_dash="dot", line_color="orange", annotation_text="Fault Injected")

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Simulation Step",
        yaxis_title="LSTM Score",
        yaxis2=dict(
            title="CVA Q Score",
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def show_event_details(event: Dict[str, Any], chat_history: List[Dict[str, str]] = None):
    report = event.get("report", {})
    evidence = event.get("evidence", {})
    
    st.markdown(f"### {event['event_id']} Details")
    st.markdown(f"**Severity:** {severity_badge(report.get('severity', 'unknown'))}", unsafe_allow_html=True)
    st.markdown(f"**Suspected Fault:** {report.get('root_cause', 'Unknown')}", help="The AI's best guess of what failed. Verify sensors in this subsystem.")
    st.markdown(f"**Subsystem:** {report.get('affected_subsystem', 'Unknown')}")
    
    # Check if CVA was involved for additional hints
    det_ev = evidence.get("detector_evidence", {}) or {}
    if det_ev.get("triggered_by") in ["dynamic", "both"]:
        st.info("💡 **Dynamic shift detected:** The CVA (Canonical Variate Analysis) flagged unusual lag energy or drift. This often indicates a mechanical issue like a sticky valve (stiction) rather than a sudden sensor spike.")

    st.info(f"**Summary:** {report.get('summary', 'No summary available.')}")
    st.warning(f"**Recommended Action:** {report.get('recommended_action', 'None')}")
    
    # Bar chart of top contributing variables
    if isinstance(evidence, dict) and "top_anomalous_sensors" in evidence:
        sensors = evidence["top_anomalous_sensors"]
        if sensors:
            names = [s.get("display_name", "Unknown") for s in sensors]
            contribs = [s.get("contribution", 0) for s in sensors]
            bar_fig = go.Figure(go.Bar(x=contribs, y=names, orientation='h'))
            bar_fig.update_layout(title="Top Contributing Variables", height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(bar_fig, use_container_width=True)
            
    with st.expander("🛠 Engineer Details (Technical Evidence)"):
        st.json(event)
        
    st.markdown("### Operator Notes & Chat Thread")
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user")
            st.chat_message(role).write(msg.get("content", ""))
    else:
        st.write("No follow-up questions asked yet.")
