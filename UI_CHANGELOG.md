# UI Redesign Changelog

## 1. Concrete Usability Flaws Found & Fixed
- **Monolithic code structure:** `ui/app.py` contained all UI and logic; split into `ui/app.py` (layout), `ui/components.py` (visuals), and `ui/state.py` (state handling).
- **Blocking Live Stream Loop:** The previous simulation used a blocking `for` loop that froze the Streamlit frontend. It could not be paused. This was fixed by using `@st.fragment(run_every=1)` and processing chunks non-blockingly with session state.
- **State reset on rerun:** Rerunning the app reset simulation progress. Fixed by implementing persistent `ui/state.py` functions to keep simulation history, steps, and selected events across reruns.
- **Jargon-heavy, unformatted logs:** Anomaly events just dumped raw JSON without clear hierarchy. Fixed by creating a detailed card with a severity badge, bar chart of top contributors, plain-language summaries, and hiding raw JSON in an "Engineer details" expander.
- **Unfriendly navigation for incident details:** Required manual selection in the chat tab to view details. Fixed by replacing the hard-capped table with a selection-enabled `st.dataframe` that expands into a rich detail view on click.
- **Missing fault selector:** `load_single_csv` would fail or load all 5 million rows when using Rieth consolidated formats. Added a dedicated `fault_id` numeric input to correctly isolate a fault run.
- **Missing empty/error states:** App lacked friendly messages when no incidents were logged. Added empty states (e.g. `st.info("No incidents recorded.")`).
- **Layout and styling:** Did not use a wide layout consistently and relied entirely on color without text in some places (now fixed via descriptive metric cards and badges).

## 2. Structural Improvements Implemented
- **Top Metric Cards:** 4 clear cards showing real-time Plant Status, Last LSTM Score, Active Events, and LLM Status.
- **Plotly Monitor Chart:** Real-time dual-axis chart showing LSTM score and CVA Q score, annotated with dashed threshold lines, a vertical marker for fault injection, and shaded anomaly regions.
- **Event Selection DataFrame:** A filterable and selectable dataframe allowing engineers to quickly search events by severity, detector type, and suspected fault text. CSV export is included.
- **In-context Chat:** The Event Assistant chat is integrated directly inside the incident detail view or bound tightly to the selected event.
- **Tooltips:** Added explanatory `help=` tooltips on the sidebar thresholds and contextual hints when CVA detects dynamic shifts (explaining "stiction" and "lag energy" in plain English).

## 3. Extra Small & Safe Improvements
1. **CRITICAL Event Banner:** A prominent `st.error` banner appears live when a new anomaly is detected, featuring a clickable button that instantly reruns the app and jumps to the incident detail view.
2. **Adaptive Plotly Theme:** Used Streamlit's native `theme="streamlit"` for Plotly charts, ensuring it perfectly respects the user's light/dark mode preference without manual intervention.
3. **Keyboard & Operator Friendly Controls:** Tightly grouped simulation inputs using responsive column ratios, and fully disabled fault-specific inputs (Fault ID, Data, Start Step) unless "Inject Fault" is explicitly ticked, reducing operator error.

## 4. Verification Details
- **Command Run:** `streamlit run ui/app.py`
- **Fault IDV(6) Check:** Confirmed that clicking "Inject Fault", setting Fault ID to 6 and Start Step to 800, then hitting "Start" streams the data smoothly. A vertical line correctly drops at step 800, and soon after, the LSTM score breaches the threshold, shading the region red. The Start/Pause buttons toggle the loop correctly.
- **Limitations:** Note that switching tabs in Streamlit natively via a Python button click is officially unsupported without hacks; clicking the banner's "View Incident" button correctly selects the event in memory and displays a hint to navigate to the Event Assistant tab to chat.
