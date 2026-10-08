from typing import Any

import streamlit as st

def init_state(key: str, default: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default

def get_state(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)

def set_state(key: str, value: Any) -> None:
    st.session_state[key] = value

def init_app_state() -> None:
    init_state("sim_running", False)
    init_state("sim_step", 0)
    init_state("sim_history", [])
    init_state("sim_fault_injected", False)
    init_state("sim_fault_step", -1)
    init_state("selected_event_id", None)
    init_state("event_chat_history", {})
    init_state("app_loaded", False)
