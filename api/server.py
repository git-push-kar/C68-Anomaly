"""FastAPI backend for the TEP RCA system."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import TEPApp
from utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI(title="TEP RCA System", version="1.0.0")

_CONFIG = load_config()
_TEP_APP: Optional[TEPApp] = None


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
    return tep.process_sensor_stream(record.dict())


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