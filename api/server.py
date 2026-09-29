"""FastAPI backend for the TEP RCA system."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import TEPApp
from utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI(title="TEP RCA System", version="2.0.0")

_CONFIG = load_config()
_TEP_APP: Optional[TEPApp] = None

import time as _time
_LAT = {"n": 0, "total_ms": 0.0, "max_ms": 0.0}


def get_app() -> TEPApp:
    global _TEP_APP
    if _TEP_APP is None:
        _TEP_APP = TEPApp(config=_CONFIG, enable_llm=True)
    return _TEP_APP


class SensorRecord(BaseModel):
    values: List[float]
    sample_index: Optional[int] = None
    is_fault: bool = False
    fault_label: Optional[int] = None


class StreamRunRequest(BaseModel):
    source_file: Optional[str] = None
    fault_source: Optional[str] = None
    inject_fault_at: Optional[int] = None
    replay_rate: Optional[float] = None


class FollowUpRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/api/status")
def status() -> Dict:
    tep = get_app()
    return {
        "status": "ok",
        "detector": tep.detector.describe(),
        "llm_loaded": tep.rca is not None,
        "events": len(tep.event_store.list_events(limit=5)),
    }


@app.post("/api/stream/sample")
def stream_sample(record: SensorRecord) -> Dict:
    tep = get_app()
    t0 = _time.perf_counter()
    out = tep.process_sensor_stream(record.dict())
    dt = (_time.perf_counter() - t0) * 1000.0
    _LAT["n"] += 1
    _LAT["total_ms"] += dt
    _LAT["max_ms"] = max(_LAT["max_ms"], dt)
    out["latency_ms"] = round(dt, 3)
    return out


@app.get("/api/health")
def health() -> Dict:
    tep = get_app()
    n = _LAT["n"]
    return {
        "status": "ok",
        "threshold": tep.detector.threshold.threshold,
        "cva_q_threshold": ((tep.dynamic_runner.thresholds.get("cva", {}) or {}).get("Q")),
        "llm_loaded": tep.rca is not None,
        "accepted_scope": "18/20 (faults 3,9 documented limitation)",
        "latency": {
            "n_samples": n,
            "mean_ms": round(_LAT["total_ms"] / max(n, 1), 3),
            "max_ms": round(_LAT["max_ms"], 3),
        },
    }


@app.get("/api/model-info")
def model_info() -> Dict:
    import json as _json
    from pathlib import Path as _P
    tep = get_app()
    info = {
        "detector": tep.detector.describe(),
        "dynamic_thresholds": tep.dynamic_runner.thresholds,
        "fusion_mode": tep.dynamic_runner.fusion_mode,
        "llm_loaded": tep.rca is not None,
    }
    for p in ["outputs/tep_rca_adapter/training_summary.json",
              "outputs/llm_dataset_v2/llm_20x1_metrics.json",
              "data/processed/c68_full_20_validation_1.85.json"]:
        fp = _P(p)
        if fp.exists():
            try:
                info[p] = _json.loads(fp.read_text()[:2000])
            except Exception:
                pass
    return info


@app.post("/api/stream/run")
def stream_run(req: StreamRunRequest) -> Dict:
    tep = get_app()
    config = _CONFIG
    source = req.source_file or config["streaming"]["normal_source"]
    fault = req.fault_source or (config["streaming"]["fault_source"] if req.inject_fault_at is not None else None)
    events = tep.run_stream_from_file(
        source_file=source,
        inject_fault_file=fault,
        inject_fault_at=req.inject_fault_at,
        replay_rate=req.replay_rate,
    )
    return {"n_events": len(events), "events": events}


@app.get("/api/events")
def list_events(limit: int = 50) -> List[Dict]:
    return get_app().event_store.list_events(limit=limit)


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> Dict:
    event = get_app().event_store.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    event["followups"] = [f.to_dict() for f in get_app().event_store.get_followups(event_id)]
    return event


@app.post("/api/events/{event_id}/followup")
def followup(event_id: str, body: FollowUpRequest) -> Dict:
    try:
        return get_app().answer_followup(event_id, body.question)
    except KeyError:
        raise HTTPException(status_code=404, detail="Event not found")