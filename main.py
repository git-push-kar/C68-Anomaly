"""TEPApp: the single high-level interface for the whole TEP RCA system.

    app = TEPApp(config, enable_llm=True)
    app.process_sensor_stream(sensor_record)     # one record at a time
    app.answer_followup("ANOM-1042", "Why the cooling system?")

The pipeline behind these two calls:

    sensor_stream -> preprocessor -> anomaly_detector -> event_aggregator
      -> evidence_generator -> tep_rca_adapter -> automatic_report
      -> event_store -> user_follow_up -> tep_rca_adapter -> answer

The TEP adapter stays an independent artifact; everything here is orchestration.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from anomaly_detection import AnomalyDetector
from events.event_store import EventStore
from evidence.event_builder import AnomalyEvent, build_event, make_event_id
from preprocessing.scaler import load_scaler
from utils import ensure_dir, load_config

logger = logging.getLogger(__name__)


def _fallback_report(event: AnomalyEvent) -> Dict:
    """Deterministic report used when the InternVL adapter is unavailable.

    The LSTM pipeline is fully functional without InternVL; the adapter only
    adds LLM reasoning. This report stays grounded in the evidence.
    """
    ev = event.evidence
    evidence_lines = [
        f"{s['display_name']} changed by {s['deviation_percent']:+.1f}% "
        f"(trend {s['trend']}, contribution {s['contribution']:.2f})"
        for s in ev.top_anomalous_sensors[:3]
    ]
    temporal = ev.temporal_sequence[:3]
    reasoning = (
        "Temporal evidence (not proof of causation): "
        + "; ".join(
            f"{t['display_name']} at {t['relative_time_minutes']:.1f} min"
            for t in temporal
        )
        or "No clear onset ordering detected."
    )
    return {
        "summary": f"Anomaly detected with score {ev.anomaly_score:.2f}; "
                   f"candidate subsystem '{ev.candidate_subsystem}'.",
        "root_cause": f"candidate: {ev.candidate_subsystem}",
        "affected_subsystem": ev.candidate_subsystem,
        "evidence": evidence_lines,
        "reasoning": reasoning + " " + " ".join(ev.reasoning_notes),
        "severity": ev.severity,
        "confidence": round(ev.candidate_subsystem_score, 2),
        "recommended_action": (
            f"Inspect the {ev.candidate_subsystem} components and verify "
            "the suspect sensors on site."
        ),
        "uncertainty": ev.uncertainty,
    }


class TEPApp:
    """End-to-end TEP anomaly detection + automatic RCA + follow-up QA."""

    def __init__(self, config: Optional[dict] = None, enable_llm: bool = True) -> None:
        self.config = config or load_config()
        model_dir = self.config["anomaly_detector"]["model_dir"]
        scaler_dir = self.config["preprocessing"]["scaler_dir"]
        ensure_dir(Path(model_dir))

        self.detector = AnomalyDetector.from_artifacts(
            model_dir=model_dir,
            scaler_dir=scaler_dir,
            threshold_dir=model_dir,
        )
        self.scaler = load_scaler(scaler_dir)
        self.event_store = EventStore(self.config["events"]["db_path"])

        # --- LLM (tep_rca adapter) is optional at runtime -------------------
        self.rca = None
        adapter_dir = Path(self.config["llm"]["adapter_dir"])
        if enable_llm and (adapter_dir / "adapter_config.json").exists():
            try:
                from llm.inference import RCAInference

                self.rca = RCAInference.from_adapter(config=self.config)
                logger.info("InternVL2-2B + tep_rca adapter loaded.")
            except Exception as exc:  # pragma: no cover - depends on hardware
                logger.error("Failed to load LLM adapter (will use fallback reports): %s", exc)
        elif enable_llm:
            logger.info("No tep_rca adapter found at %s; using deterministic reports.",
                        adapter_dir)

        # --- streaming state ------------------------------------------------
        self.window_size = int(self.config["streaming"]["window_size"])
        self.stride = int(self.config["streaming"]["stride"])
        self._buffer: deque = deque(maxlen=self.window_size)
        self._records_since_emit = 0
        self._window_id = 0

        ev_cfg = self.config["events"]
        self._confirm = int(ev_cfg.get("consecutive_windows_to_confirm", 3))
        self._separate = int(ev_cfg.get("min_separation_windows", 20))
        self._max_event_windows = int(ev_cfg.get("max_event_windows", 200))

        self._recent_flags: deque = deque(maxlen=max(self._confirm, self._separate) + 5)
        self._open: Optional[Dict] = None

    # ==================================================================
    # continuous stream entry point
    # ==================================================================
    def process_sensor_stream(self, sensor_record) -> Dict:
        """Process one incoming sensor record.

        Args:
            sensor_record: dict with {"values": np.ndarray[F]} and optional
                {"sample_index", "is_fault", "fault_label"}, or a raw
                numpy array / list of F values.

        Returns:
            dict (see module docstring for the pipeline semantics):
            {"anomaly_detected", "event_id", "report", "window_id",
             "window_score", "is_anomalous", "open_event"}
        """
        if isinstance(sensor_record, dict):
            values = np.asarray(sensor_record["values"], dtype=np.float32)
            fault_label = sensor_record.get("fault_label")
        else:
            values = np.asarray(sensor_record, dtype=np.float32)
            fault_label = None

        if values.ndim != 1:
            raise ValueError(f"Expected a 1D sensor record, got {values.shape}")

        self._buffer.append(values)
        self._records_since_emit += 1
        if len(self._buffer) < self.window_size:
            return {"anomaly_detected": False, "event_id": None, "report": None,
                    "window_id": None, "window_score": None, "is_anomalous": False,
                    "open_event": self._open is not None}

        emit = self._records_since_emit % self.stride == 0
        self._window_id += 1
        if not emit:
            return {"anomaly_detected": False, "event_id": None, "report": None,
                    "window_id": self._window_id, "window_score": None,
                    "is_anomalous": False, "open_event": self._open is not None}

        window = np.stack(list(self._buffer)[-self.window_size:], axis=0)
        score, per_sensor_error = self.detector.score_window(window)
        is_anomalous = self.detector.is_anomalous(score)
        self._recent_flags.append(int(is_anomalous))

        event_payload = self._feed_aggregator(
            score, per_sensor_error, window, fault_label
        )
        result = {
            "anomaly_detected": event_payload is not None,
            "event_id": event_payload.get("event_id") if event_payload else None,
            "report": event_payload.get("report") if event_payload else None,
            "event": event_payload if event_payload else None,
            "window_id": self._window_id,
            "window_score": round(float(score), 5),
            "is_anomalous": bool(is_anomalous),
            "open_event": self._open is not None,
        }
        return result

    # ==================================================================
    # event aggregation (no LLM call per anomalous window)
    # ==================================================================
    def _feed_aggregator(self, score: float, per_sensor_error, window, fault_label):
        """Group consecutive anomalous windows into a single event.

        Returns the event payload (dict) when an event is closed, else None.
        """
        if self._open is None:
            if self._is_open_candidate():
                self._open = {
                    "scores": [score],
                    "errors": [per_sensor_error],
                    "windows": [window],
                    "start_sample": max(0, self._window_id * self.stride),
                    "start_time": datetime.now(),
                    "fault_label": fault_label,
                    "normal_since": 0,
                }
            return None

        if score > self.detector.threshold.threshold:
            self._open["normal_since"] = 0
            self._open["scores"].append(score)
            self._open["errors"].append(per_sensor_error)
            self._open["windows"].append(window)
            if fault_label is not None:
                self._open["fault_label"] = fault_label
            if len(self._open["scores"]) >= self._max_event_windows:
                return self._close_event()
            return None

        self._open["normal_since"] += 1
        if self._open["normal_since"] >= self._separate:
            return self._close_event()
        return None

    def _is_open_candidate(self) -> bool:
        flags = list(self._recent_flags)
        return len(flags) >= self._confirm and all(f == 1 for f in flags[-self._confirm:])

    def _close_event(self):
        state = self._open
        self._open = None
        event = build_event(
            event_id=make_event_id(self._next_event_counter()),
            anomaly_scores=state["scores"],
            per_sensor_errors=np.asarray(state["errors"], dtype=np.float32),
            event_windows=np.asarray(state["windows"], dtype=np.float32),
            baseline=self.scaler.baseline,
            feature_names=list(self.scaler.baseline.feature_names),
            threshold=self.detector.threshold.threshold,
            start_sample=state["start_sample"],
            config=self.config,
            start_time=state["start_time"],
            detection_time=datetime.now(),
            fault_label=state.get("fault_label"),
        )
        event.report = self._generate_report(event)
        self.event_store.store_event(event)
        logger.info("Closed anomaly event %s (score %.3f, severity %s).",
                    event.event_id, event.max_anomaly_score, event.evidence.severity)
        return event.to_dict()

    def _next_event_counter(self) -> int:
        # derive next counter from the store so ids stay unique across runs
        return int(self.event_store.next_event_id().split("-")[1])

    def _generate_report(self, event: AnomalyEvent) -> Dict:
        if self.rca is not None:
            report, raw = self.rca.generate_report(event.evidence.to_dict())
            logger.debug("LLM report raw: %.200s", raw)
            return report
        return _fallback_report(event)

    # ==================================================================
    # follow-up questions
    # ==================================================================
    def answer_followup(self, event_id: str, user_question: str) -> Dict:
        """Answer a follow-up question about a stored anomaly event."""
        event = self.event_store.get_event(event_id)
        if event is None:
            raise KeyError(f"Unknown event: {event_id}")
        report = event.get("report") or {}
        history = self.event_store.conversation_history(event_id)

        if self.rca is not None:
            answer = self.rca.answer_followup(event, report, history, user_question)
        else:
            answer = (
                f"[Deterministic fallback] The event {event_id} shows "
                f"{len(event.get('evidence', {}).get('top_anomalous_sensors', []))} "
                "deviating sensors. The candidate subsystem is "
                f"'{event.get('evidence', {}).get('candidate_subsystem', 'unknown')}'. "
                "Load the tep_rca adapter for a conversational answer."
            )
        self.event_store.add_followup(event_id, user_question, answer)
        return {"event_id": event_id, "question": user_question, "answer": answer}

    # ==================================================================
    # batch / file helpers
    # ==================================================================
    def run_stream_from_file(
        self,
        source_file: Optional[str] = None,
        inject_fault_file: Optional[str] = None,
        inject_fault_at: Optional[int] = None,
        replay_rate: Optional[float] = None,
        normal_simulation_run: Optional[int] = None,
        fault_number: Optional[int] = None,
        fault_simulation_run: Optional[int] = None,
    ) -> List[Dict]:
        """Replay a CSV through the pipeline; returns closed event payloads."""
        from streaming.simulator import SensorStream

        # Use streaming config defaults if not provided via CLI
        normal_run = normal_simulation_run if normal_simulation_run is not None else int(self.config["streaming"].get("normal_simulation_run", 1))
        fnum = fault_number if fault_number is not None else self.config["streaming"].get("fault_number")
        frun = fault_simulation_run if fault_simulation_run is not None else int(self.config["streaming"].get("fault_simulation_run", 1))
        stream = SensorStream.from_csv(
            source_file or self.config["streaming"]["normal_source"],
            config=self.config,
            replay_rate=replay_rate,
            fault_number=0,
            simulation_run=normal_run,
        )
        fault_frame = None
        fault_label = None
        if inject_fault_file:
            from preprocessing.tep_loader import load_single_csv

            if fnum is not None:
                fault_label = int(fnum)
            else:
                fault_label = _infer_fault_label(inject_fault_file)
            fault_frame = load_single_csv(
                inject_fault_file,
                self.config,
                fault_number=fault_label,
                simulation_run=frun,
            )
        fault_onset = int(self.config["dataset"].get("fault_onset_index", 160))
        events = []

        # simulate fault injection by switching frames
        all_records = []
        for rec in stream.iter_records():
            if (inject_fault_at is not None and fault_frame is not None
                    and rec["sample_index"] >= inject_fault_at):
                off = rec["sample_index"] - inject_fault_at
                fault_row = (fault_onset + off) % len(fault_frame)
                rec = {
                    "sample_index": rec["sample_index"],
                    "values": fault_frame.iloc[fault_row].to_numpy(dtype=np.float32),
                    "is_fault": True,
                    "fault_label": fault_label,
                }
            all_records.append(rec)
        for rec in all_records:
            result = self.process_sensor_stream(rec)
            if result.get("event") is not None:
                events.append(result["event"])
        final_event = self.finalize_stream()
        if final_event is not None:
            events.append(final_event)
        return events

    def finalize_stream(self) -> Optional[Dict]:
        """Close an active event when a finite replay/input stream ends."""
        if self._open is None:
            return None
        return self._close_event()


def _infer_fault_label(path: str) -> int:
    import re

    match = re.search(r"fault[_-]?(\d+)", Path(path).stem, re.I)
    return int(match.group(1)) if match else None