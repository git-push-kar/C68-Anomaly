"""Operator question bank, split into a train half and a held-out half.

Why a bank and not a generator alone: the live router keys off literal keywords
(``llm/inference.py``), so a question's *wording* decides whether a template or
the model answers it. Paraphrases therefore have to be written and then checked
against the router, not invented freely -- ``scripts/inspect_followup_routing.py``
routes every one of them and the build refuses a paraphrase whose route does not
match its label.

Splits
------
* ``train_questions(label)``  -- 10 base phrasings x 4 surface variants = 40.
  Used for train dialogues only.
* ``heldout_questions(label)`` -- 10 hand-written phrasings that share no
  sentence with the train half. Used for val and test dialogues, so a test
  paraphrase is never seen in training.
* ``probe_questions(label)``   -- 50 per label, the pool the routing table is
  measured on.

The surface variants are prefixes only ("Quick check — ..."). They add no
keyword from any intent gate, which is verified by the routing probe rather than
assumed: a variant that changed the route would show up as a diagonal miss.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence, Tuple

#: The 14 candidate intents of docs/FOLLOWUP_SPEC.md §2.
ROUTED_LABELS: Tuple[str, ...] = (
    "sensor_values",
    "evidence_why",
    "system_sensors",
    "sensor_scores",
    "detector_severity",
    "detector_scores",
    "severity",
    "overview",
    "sensors_inspect",
    "action",
    "overview_fallback",
    "hypothesis_comparison",
    "out_of_scope_refusal",
    "clarification_reply",
)

#: Questions that reach the model with no intent at all (live fall-through).
FREE_FORM_LABELS: Tuple[str, ...] = ("valve_actuator", "causal_why", "mechanism")

LABELS: Tuple[str, ...] = ROUTED_LABELS + FREE_FORM_LABELS

#: Prefixes used to build the train half. None of them contains a gate keyword.
SURFACE_PREFIXES: Tuple[str, ...] = (
    "",
    "Can you tell me: ",
    "For this incident, ",
    "Quick check — ",
)

BASE: Dict[str, Tuple[str, ...]] = {
    "sensor_values": (
        "What are the sensor values?",
        "Give me the measured values against the baseline.",
        "Show me the observed values and the normal values.",
        "List the sensor values with their ideal values.",
        "What are the current and baseline values?",
        "I need the measured values for the moving variables.",
        "Report the observed values versus baseline values.",
        "Which variables are off their normal values?",
        "What are the sensor values right now?",
        "Print the measured values and ideal baselines.",
    ),
    "evidence_why": (
        "What is the evidence that this is an anomaly?",
        "Why was this flagged?",
        "Show me the evidence behind this alarm.",
        "What evidence forces this to be an anomaly?",
        "Why is this an anomaly?",
        "Which alarm trigger fired?",
        "Give me the evidence for this event.",
        "What is the evidence and the detector readings?",
        "Why this is an anomaly — walk me through the evidence.",
        "What evidence supports this alarm?",
    ),
    "system_sensors": (
        "Which sensors are used in this system?",
        "What sensors are in the system?",
        "List all sensors in the plant.",
        "Which sensors are used for monitoring?",
        "Sensors used in the Tennessee Eastman process?",
        "Give me the system sensors.",
        "What sensors are in this plant?",
        "Which sensors are used by the detector?",
        "Show me all sensors in the system.",
        "What sensors are in the system baseline?",
    ),
    "sensor_scores": (
        "What is the sensor score for each variable?",
        "Show the sensor scores.",
        "Which sensor has the highest score?",
        "Give me the per-sensor scores.",
        "What are the sensor scores and their deviations?",
        "Rank the sensors by score.",
        "What is the score of each sensor?",
        "Sensor score details, please.",
        "Which sensors score highest on the detector?",
        "Show me sensor scores with trends.",
    ),
    "detector_severity": (
        "What are the detector scores and the severity?",
        "Give me the metrics and how serious this is.",
        "Show the reconstruction error, the Q-statistic and the severity.",
        "What is the score and the risk level?",
        "Detector metrics plus the severity level.",
        "How serious is this, and what are the detector scores?",
        "Q-statistic, threshold, and severity please.",
        "What are the metrics and is this critical?",
        "Report the scores and the danger level.",
        "Thresholds, scores, and severity.",
    ),
    "detector_scores": (
        "What are the detector scores?",
        "Give me the metrics.",
        "What is the Q-statistic?",
        "Show the reconstruction error and threshold.",
        "What is the score?",
        "Detector thresholds and scores, please.",
        "What is the LSTM reconstruction error?",
        "Which detectors alarmed and with what score?",
        "Report the metrics and thresholds.",
        "What are the scores from both detectors?",
    ),
    "severity": (
        "What is the severity?",
        "How serious is this?",
        "How dangerous is this event?",
        "What is the risk level?",
        "Is this critical?",
        "Give me the severity level.",
        "What is the alarm level?",
        "What is the severity of this anomaly?",
        "State the severity for the record.",
        "What danger does this pose?",
    ),
    "overview": (
        "What happened?",
        "What is the root cause?",
        "Which fault is this?",
        "Give me an overview.",
        "What is this anomaly?",
        "Which fault number is this?",
        "Summarise the event.",
        "What is the cause?",
        "Diagnose this event.",
        "What fault type is this?",
    ),
    "sensors_inspect": (
        "Which sensors should I inspect?",
        "Which sensor should I look at first?",
        "Which sensors are deviating?",
        "List the sensors to check.",
        "Where should maintenance inspect?",
        "Which sensors are out of line?",
        "Show me the deviating sensors.",
        "Which sensor moved first?",
        "Sensors worth a field check?",
        "What should I inspect on the loop?",
    ),
    "action": (
        "What should I do?",
        "What is the recommended action?",
        "How do I fix this?",
        "What are the next steps?",
        "Give me a mitigation plan.",
        "What procedure do you recommend?",
        "Provide a fix for this.",
        "What action should the operator take?",
        "What action should the shift take?",
        "Recommend a response.",
    ),
    "overview_fallback": (
        "Why did this happen?",
        "Explain what I am seeing.",
        "Why did the pressure move?",
        "Explain the pattern.",
        "Why the flow change?",
        "Diagnose what is going on.",
        "Why is the reactor temperature drifting?",
        "Explain the anomaly.",
        "Why would that valve stick?",
        "Explain the sequence.",
    ),
    "hypothesis_comparison": (
        "How do we distinguish this from a lookalike fault?",
        "What distinguishes this from alternative causes?",
        "How is this different from fault 14?",
        "Give me the differential diagnosis.",
        "Which lookalike fits better?",
        "How do we tell this apart from an instrument problem?",
        "Compare this against the alternative cause.",
        "Is it fault 15 vs fault 14?",
        "What is the differential between the two sticking-valve faults?",
        "How do we tell the lookalikes apart?",
    ),
    "out_of_scope_refusal": (
        "What is the compressor bearing vibration?",
        "Show me the vibration spectrum on the stripper.",
        "What is the ambient humidity in the compressor house?",
        "Give me the lube-oil pressure on the compressor.",
        "What is the corrosion rate in the reactor?",
        "Show the maintenance log for this valve.",
        "What is the noise level at the stripper?",
        "Give me the motor current on the agitator.",
        "What is the bearing temperature of the cooling water pump?",
        "What is the weather in Houston?",
    ),
    "clarification_reply": (
        "What do you mean by that?",
        "Can you clarify which view you mean?",
        "Which one do you mean?",
        "Clarify that, please.",
        "What do you mean by the per-variable view?",
        "Can you clarify what you just said?",
        "What do you mean, the first or the second?",
        "What do you mean?",
        "Clarify which view you are offering.",
        "Which one do you mean, the per-variable view or the aggregate view?",
    ),
    "valve_actuator": (
        "Is the condenser cooling water valve sticking?",
        "Could the A feed valve be stuck open?",
        "Which actuator is failing?",
        "Is the purge valve behaving?",
        "What is the valve position doing?",
        "Is the recycle compressor valve hunting?",
        "Could a sticking valve account for this?",
        "Which valve is the actuator suspect?",
        "Is the stripper steam valve stuck?",
        "What is going on at the condenser valve?",
    ),
    "causal_why": (
        "How does a sticking condenser valve end up raising the reactor temperature?",
        "What chain of events links the feed ratio step to the pressure swing?",
        "How does the coolant restriction propagate into the separator?",
        "What connects the purge valve closing to the stripper pressure?",
        "How does one loop pull the next one with it?",
        "What is the sequence from the valve to the temperature?",
        "How does the reactor feed rate respond to the A feed valve?",
        "What happens downstream when the cooling water valve sticks?",
        "How does the composition shift reach the product stream?",
        "What links the pressure rise to the level drop?",
    ),
    "mechanism": (
        "What physical mechanism turns a coolant restriction into a pressure rise?",
        "What is the mechanism behind a sticking valve under closed-loop control?",
        "By what mechanism does the feed ratio change the reactor composition?",
        "What mechanism makes the level fall when the valve sticks?",
        "What is the physical mechanism at the actuator?",
        "How does valve stiction produce a slow oscillation?",
        "What mechanism links the pressure to the temperature here?",
        "What is the mechanism of the decomposition side reaction?",
        "By what mechanism does the purge change the composition?",
        "What mechanism gives a slow drift rather than a step?",
    ),
}

HELDOUT: Dict[str, Tuple[str, ...]] = {
    "sensor_values": (
        "How far is each variable from its normal value?",
        "Compare each variable's measured value with its ideal value.",
        "I want the observed value and the ideal value side by side.",
        "Read me the baseline values next to the measured values.",
        "What is the measured value of every moving variable against its baseline value?",
        "List observed values and the corresponding normal values.",
        "Give me one line per variable: observed value, normal value, shift.",
        "What did the variables settle at compared with their baseline values?",
        "State the sensor values for the top three variables only.",
        "For each moving variable, the measured value and the normal value.",
    ),
    "evidence_why": (
        "What evidence do you have that the process moved?",
        "The alarm fired — what evidence backs it?",
        "Which readings sent this past the alarm trigger?",
        "Why was this flagged as anomalous rather than normal drift?",
        "Point to the evidence that made the detector fire.",
        "What evidence did the detector have when it alarmed?",
        "Was there an alarm trigger, and what evidence came with it?",
        "Show the evidence that justifies calling this an anomaly.",
        "Why anomaly at all — what is the evidence?",
        "Give me the evidence chain, from alarm trigger to reading.",
    ),
    "system_sensors": (
        "Remind me which sensors are used at this plant.",
        "How many and which sensors are used in the system — the full list?",
        "I want the system sensors list, not just the moving ones.",
        "Which sensors are used in the process, and which are valves?",
        "Enumerate the sensors used in this system.",
        "What sensors are in the system on the measurement side?",
        "Which sensors are used — the 41 measurements and the 11 manipulated variables?",
        "Give me all sensors the plant reports.",
        "Are all sensors in the system active in this event?",
        "Sensors used by this system: what is the full set?",
    ),
    "sensor_scores": (
        "Break the detector score down per sensor.",
        "Each sensor's score — highest first.",
        "How is the score distributed across the sensors?",
        "Give me the sensor-by-sensor score table.",
        "Which sensor contributes the most to the score?",
        "What score does each individual sensor carry?",
        "Per-sensor score and deviation, please.",
        "Show a sensor score ranking.",
        "Does any single sensor carry a large score?",
        "Sensor scores ranked, with the trend for each.",
    ),
    "detector_severity": (
        "Both numbers, please: the detector metrics and how serious it is.",
        "Give me the score breakdown and the alarm level.",
        "What is the reconstruction error and the severity?",
        "How dangerous is this, judged on the detector scores?",
        "Severity, plus the score and threshold for each detector.",
        "Metrics first, then the risk level.",
        "Show the Q-statistic against its threshold and the severity.",
        "Is this critical? And what are the scores?",
        "Detector score, threshold, and the danger assessment.",
        "Score margin and severity together.",
    ),
    "detector_scores": (
        "The detector metrics, please.",
        "How far above threshold is the Q-statistic?",
        "Give me the reconstruction error with its threshold.",
        "What scores did the two detectors return?",
        "LSTM error and CVA Q, with thresholds.",
        "Show the detector metrics for this event.",
        "Which threshold did the score cross?",
        "The score, the threshold, and the margin.",
        "Give me the metric values the detector produced.",
        "What is the reconstruction error right now?",
    ),
    "severity": (
        "Rate the severity of this event.",
        "On your scale, what is the severity?",
        "Is the severity high or low?",
        "How serious would you call it?",
        "What severity does the rule assign?",
        "Give me the alarm level and the reason.",
        "How dangerous is the current state?",
        "Is this a critical condition?",
        "What is the risk level for the shift?",
        "Severity, please — and nothing else.",
    ),
    "overview": (
        "In one line: what happened here?",
        "What is the root cause as you see it?",
        "Which fault does this look like?",
        "Give me the summary of this incident.",
        "What is the fault type and the affected area?",
        "What is this, in plain terms?",
        "Tell me the fault number if you have one.",
        "What happened during this event?",
        "Explain this anomaly briefly.",
        "An overview, then the root cause.",
    ),
    "sensors_inspect": (
        "Give me the field inspection list, sensor by sensor.",
        "Which sensor is furthest out?",
        "Where do I start on the sensors?",
        "Which sensors warrant a check?",
        "List the sensors that moved and need a look.",
        "What should the field tech inspect?",
        "Which of the sensors is drifting?",
        "Sensors to inspect, ranked.",
        "Which sensor reading is the worst offender?",
        "Point me at the sensors to check first.",
    ),
    "action": (
        "What is the operator action here?",
        "Give me the next step, in order.",
        "How should we mitigate this?",
        "What repair do you recommend?",
        "Is there a procedure for this?",
        "Tell me what to do next.",
        "What corrective action fits?",
        "Fix options, please.",
        "How do we fix the sticking valve?",
        "What should the shift do about it?",
    ),
    "overview_fallback": (
        "Why did it start with that variable?",
        "Explain it as you would to a new operator.",
        "Why now and not earlier?",
        "Explain what the loop is doing.",
        "Why is the stripper level falling?",
        "Diagnose the behaviour of this loop.",
        "Why does the level keep dropping?",
        "Explain the dynamics in plain words.",
        "Why is the condenser pressure climbing?",
        "Explain the trend I am looking at.",
    ),
    "hypothesis_comparison": (
        "Distinguish the two candidate causes for me.",
        "What is the differential here?",
        "How is this different from a feed disturbance?",
        "Which alternative cause should stay on the list?",
        "Fault 14 versus fault 15 — which fits?",
        "Give me the lookalike comparison.",
        "How do we tell this from an ordinary drift?",
        "Is this the same signature as a lookalike event?",
        "Compare fault 4 versus fault 15 as the cause.",
        "What separates this from the nearest alternative cause?",
    ),
    "out_of_scope_refusal": (
        "What is the ambient temperature outside the control room?",
        "Who signed off the last safety audit?",
        "Give me the spare-parts stock level for this pump.",
        "What is the vibration on the reactor agitator?",
        "Show me the electrical drawing for the compressor.",
        "What is the plant's energy consumption this month?",
        "Tell me the analyser calibration drift for the gas chromatograph.",
        "What is the corrosion allowance on the stripper shell?",
        "Write a summary of the shift handover meeting.",
        "What is the flow coefficient of the purge valve trim?",
    ),
    "clarification_reply": (
        "I do not follow — can you clarify?",
        "Clarify what you mean by that.",
        "What do you mean exactly?",
        "What do you mean by the two views?",
        "Can you clarify?",
        "What do you mean by the second option?",
        "Clarify the choice you just gave me.",
        "What do you mean by raw readings?",
        "Which one do you mean by the detector side?",
        "Can you clarify the options?",
    ),
    "valve_actuator": (
        "Is the valve actuator the thing that moved first?",
        "Could the cooling water valve be sticking rather than fouling?",
        "Which valve do you suspect: feed, purge or cooling water?",
        "Is the A feed valve open too far?",
        "Is the actuator hunting at the purge valve?",
        "What is the condenser valve doing?",
        "Could the valve be stuck part-open?",
        "Is the sticking valve on the cooling water side?",
        "Which control loop valve is misbehaving?",
        "Is the separator liquid valve chattering?",
    ),
    "causal_why": (
        "Walk me through how the condenser restriction reaches the reactor.",
        "What is the propagation path from the valve to the column?",
        "How does closing the purge valve move the separator pressure?",
        "What is the knock-on effect of the feed ratio step?",
        "How does the cooling water flow restriction ripple through the plant?",
        "What ties the temperature rise to the valve behaviour?",
        "How does the disturbance travel from the feed to the product?",
        "What is the chain from the actuator to the analyser reading?",
        "How would a stuck valve show up two loops later?",
        "What is the physical path between the two readings?",
    ),
    "mechanism": (
        "Give me the physical mechanism, not the symptom.",
        "What is happening mechanically inside the valve?",
        "What mechanism would produce this slow oscillation?",
        "By what mechanism does the coolant flow limit the reactor temperature?",
        "What is the mechanism behind the level decline?",
        "Is the mechanism stiction or fouling?",
        "What physical process turns the valve problem into a pressure swing?",
        "What mechanism governs the recovery after the actuator frees up?",
        "How does the mechanism differ for a step against a drift?",
        "What mechanism is consistent with a slow drift at the actuator?",
    ),
}

#: Reply-shaped statements (no clarification keyword). They are *replies* to a
#: pending clarification, which the keyword router cannot see; the routing table
#: reports them separately. ``cases.py`` resolves them from dialogue state.
REPLY_SHAPED: Tuple[str, ...] = (
    "The detector scores view.",
    "The per-variable view, please.",
    "I meant the evidence side.",
    "Neither, give me the raw readings.",
    "The closest match.",
    "Raw evidence, please.",
    "Go with the stored readings.",
    "The second one.",
    "Just the numbers.",
    "The one built from the stored readings.",
)

#: False-premise question templates, formatted per event from the store. Each
#: premise is checked against `StoreAccess`; a premise that turns out to be true
#: is not used (the builder re-draws), so every item here is genuinely false.
FALSE_PREMISE_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "sensor": (
        "Is {tag} the variable that moved first?",
        "Should I treat {tag} as the initiating variable?",
        "Is {tag} the one that is out of line?",
    ),
    "fault": (
        "This is {fault}, correct?",
        "Should I close this out as {fault}?",
        "We are looking at {fault} here, right?",
    ),
    "value": (
        "Was the dynamic residual around {value}?",
        "The detector reported {value}, did it not?",
        "Is {value} the dynamic reading you are working from?",
    ),
    "key": (
        "What is the vibration reading on {tag}?",
        "Give me the calibration offset stored for {tag}.",
        "What is the corrosion rate recorded for {tag}?",
    ),
}


# ---------------------------------------------------------------------------
def _sentence_case_down(text: str) -> str:
    """Lowercase the first character, so a prefix reads naturally."""
    return text[:1].lower() + text[1:] if text else text


def _surface(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return prefix + _sentence_case_down(text)


def train_questions(label: str) -> Tuple[str, ...]:
    """40 train-half questions: 10 base phrasings x 4 surface variants."""
    if label not in BASE:
        raise KeyError(f"unknown paraphrase label {label!r}")
    out = []
    for prefix in SURFACE_PREFIXES:
        for base in BASE[label]:
            out.append(_surface(base, prefix))
    return tuple(out)


def heldout_questions(label: str) -> Tuple[str, ...]:
    """10 held-out questions, never used for a train dialogue."""
    if label not in HELDOUT:
        raise KeyError(f"unknown paraphrase label {label!r}")
    return tuple(HELDOUT[label])


def probe_questions(label: str) -> Tuple[str, ...]:
    """The 50 questions per label the routing table is measured on."""
    return train_questions(label) + heldout_questions(label)


def questions_for_split(label: str, split: str) -> Tuple[str, ...]:
    return train_questions(label) if split == "train" else heldout_questions(label)


def _index(key: str, size: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(size, 1)


def pick_question(label: str, key: str, split: str) -> Tuple[str, int]:
    """Deterministically choose a question and its index within its half.

    ``key`` is any stable string (an item key); the same key always yields the
    same question, so a rebuild reproduces the dataset without a cache.
    """
    pool = questions_for_split(label, split)
    index = _index(f"{label}|{key}", len(pool))
    return pool[index], index


def normalise(question: str) -> str:
    """Whitespace/case-normalised question, for disjointness checks."""
    return " ".join((question or "").lower().split())


__all__ = [
    "BASE",
    "FALSE_PREMISE_TEMPLATES",
    "FREE_FORM_LABELS",
    "HELDOUT",
    "LABELS",
    "REPLY_SHAPED",
    "ROUTED_LABELS",
    "SURFACE_PREFIXES",
    "heldout_questions",
    "normalise",
    "pick_question",
    "probe_questions",
    "questions_for_split",
    "train_questions",
]
