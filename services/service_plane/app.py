"""Service plane (L6): FastAPI application.

Endpoints:
  GET  /health                     liveness
  POST /process_video              run the pipeline over a video path
  GET  /incidents                  filtered incident list
  GET  /incidents/{id}             full incident detail
  POST /incidents/{id}/feedback    operator confirm/dismiss + KB write
  POST /incidents/{id}/replay      re-run evidence clip (Slice 7.5)
  GET  /metrics                    live operational stats
  GET  /architecture               system layer status
  WS   /ws/alerts                  real-time alert stream

WebSocket pub/sub: in-memory ConnectionManager. Every alert-state incident
created via /process_video is broadcast to connected clients. In production
this should be replaced with a Redis pub/sub channel so multiple service
instances share the same feed.
"""
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from services.persistence.db import Incident as IncidentRow
from services.persistence.db import SessionLocal
from services.pipeline.perception_pipeline import process_video as run_pipeline
from services.pipeline.replay import replay_incident
from shared.schemas.incident import AuditEntry, Incident, IncidentDetail

app = FastAPI(title="Room Safety Monitor - Service Plane", version="0.1.0")

_DASHBOARD_ORIGIN = os.environ.get("DASHBOARD_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_DASHBOARD_ORIGIN],
    # Also allow any localhost port so the dashboard works when Next.js starts
    # on 3001 or another port because 3000 is occupied.
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

# Serve local evidence clips at /clips/{id}.mp4.
# check_dir=False lets the service start before the clips directory exists.
# Production: replace with CDN or MinIO presigned URLs.
app.mount("/clips", StaticFiles(directory="clips", check_dir=False), name="clips")


# ---------------------------------------------------------------------------
# In-memory metrics counters (reset on restart; see /metrics docstring)
# ---------------------------------------------------------------------------

class _Counters:
    def __init__(self):
        self._lock = threading.Lock()
        self.frames_processed: int = 0
        self.frames_escalated: int = 0
        self.process_video_calls: int = 0

    def update(self, result: dict) -> None:
        with self._lock:
            self.frames_processed += result.get("frames_processed", 0)
            self.frames_escalated += result.get("frames_escalated", 0)
            self.process_video_calls += 1

_counters = _Counters()


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._connections: set = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, data: dict) -> None:
        dead = set()
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._connections -= dead

manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_kb():
    """Return a KB instance backed by the default Postgres session.

    Returns None when Postgres is unavailable (e.g., in unit tests using
    SQLite). The feedback endpoint degrades gracefully when kb is None:
    operator_decision is still written to the incident row, but no KB entry
    is created.
    """
    try:
        from services.kb.kb import KB
        return KB(SessionLocal())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ProcessVideoRequest(BaseModel):
    video_path: str
    camera_id: str = "cam0"
    room_id: str = "room0"


class FeedbackRequest(BaseModel):
    decision: Literal["confirm", "dismiss"]
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process_video")
async def process_video(req: ProcessVideoRequest, db: Session = Depends(get_db)):
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=400, detail=f"video not found: {req.video_path}")

    import asyncio
    result = await asyncio.to_thread(
        run_pipeline, req.video_path, db,
        camera_id=req.camera_id, room_id=req.room_id,
    )
    _counters.update(result)

    # Broadcast all newly created incidents to WebSocket subscribers.
    # Previously filtered to state=="alert" only, which meant dismissed
    # incidents (the common case with stub VLM + stub-caution) were never
    # broadcast and the home feed required a hard refresh to show them.
    if result.get("incidents_created", 0) > 0:
        rows = (
            db.execute(
                select(IncidentRow)
                .order_by(IncidentRow.created_at.desc())
                .limit(result["incidents_created"])
            )
            .scalars()
            .all()
        )
        for row in rows:
            await manager.broadcast({
                "type": "alert",
                "incident": {
                    "id": str(row.id),
                    "camera_id": row.camera_id,
                    "room_id": row.room_id,
                    "event_type": row.event_type,
                    "severity": row.severity,
                    "confidence": row.confidence,
                    "rationale": row.rationale,
                    "state": row.state,
                    "created_at": row.created_at.isoformat(),
                    "evidence_clip_url": row.evidence_clip_url,
                    "operator_decision": row.operator_decision,
                },
            })

    return {"status": "ok", "incidents_created": result["incidents_created"]}


@app.get("/incidents", response_model=List[Incident])
def list_incidents(
    camera_id: Optional[str] = None,
    room_id: Optional[str] = None,
    state: Optional[str] = None,
    severity: Optional[str] = None,
    from_time: Optional[datetime] = Query(None, alias="from"),
    to_time: Optional[datetime] = Query(None, alias="to"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    stmt = select(IncidentRow).order_by(IncidentRow.created_at.desc())
    if camera_id:
        stmt = stmt.where(IncidentRow.camera_id == camera_id)
    if room_id:
        stmt = stmt.where(IncidentRow.room_id == room_id)
    if state:
        stmt = stmt.where(IncidentRow.state == state)
    if severity:
        stmt = stmt.where(IncidentRow.severity == severity)
    if from_time:
        stmt = stmt.where(IncidentRow.created_at >= from_time)
    if to_time:
        stmt = stmt.where(IncidentRow.created_at <= to_time)
    stmt = stmt.limit(limit).offset(offset)
    return db.execute(stmt).scalars().all()


@app.get("/incidents/{incident_id}", response_model=IncidentDetail)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    try:
        import uuid
        parsed = uuid.UUID(incident_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")
    row = db.get(IncidentRow, parsed)
    if row is None:
        raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")

    from services.persistence.db import IncidentAudit
    audit_rows = (
        db.execute(
            select(IncidentAudit)
            .where(IncidentAudit.incident_id == parsed)
            .order_by(IncidentAudit.created_at.asc())
        )
        .scalars()
        .all()
    )
    detail = IncidentDetail.model_validate(row)
    detail.audit_entries = [AuditEntry.model_validate(a) for a in audit_rows]
    return detail


@app.post("/incidents/{incident_id}/feedback")
def feedback(
    incident_id: str,
    body: FeedbackRequest,
    db: Session = Depends(get_db),
    kb=Depends(get_kb),
):
    """Accept operator confirmation or dismissal for an incident.

    Updates incident.operator_decision and writes a KB entry so future
    incidents can benefit from the operator's judgement. KB write happens
    regardless of whether the original VLM was real or stub; the metadata
    records which path was taken so downstream analysis can filter on it.
    """
    try:
        import uuid
        parsed = uuid.UUID(incident_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")

    row = db.get(IncidentRow, parsed)
    if row is None:
        raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")

    decision_value = "confirmed" if body.decision == "confirm" else "dismissed"
    row.operator_decision = decision_value
    db.commit()

    if kb is not None:
        vlm_source = "stub" if row.vlm_prompt is None else "real"
        prompt_text = row.vlm_prompt or row.rationale
        kb.write(
            prompt=prompt_text,
            rationale=row.rationale,
            label=row.event_type,
            operator_decision=decision_value,
            metadata={
                "incident_id": str(row.id),
                "camera_id": row.camera_id,
                "room_id": row.room_id,
                "vlm_source": vlm_source,
                "note": body.note,
            },
        )

    return {"status": "ok", "operator_decision": decision_value}


@app.post("/incidents/{incident_id}/replay")
def replay_incident_endpoint(incident_id: str, db: Session = Depends(get_db)):
    """Re-run the evidence clip through the current pipeline and compare outcomes."""
    try:
        return replay_incident(incident_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    """Live operational statistics.

    Runtime counters (frames, gate rate) are in-memory and reset on restart.
    Historical stats (incident counts, operator decisions, alert rate) are
    computed from the incidents table. In production these would be served from
    a time-series store (Prometheus + Grafana is the intended path).

    Fields added in Slice 8c:
      kb_entry_count          — total KB entries; null when kb_entries table is
                                absent (SQLite test environment or KB disabled).
      incidents_by_severity_24h — severity breakdown for the last 24 hours.

    Fields not yet available (require pipeline timing instrumentation):
      per_layer_latency_p50/p95, operator_decision_latency_median.
      These are planned for the Prometheus/Grafana path in Slice 9.
    """
    from datetime import timedelta
    from sqlalchemy import text as sa_text

    fp = _counters.frames_processed
    fe = _counters.frames_escalated
    gate_filter_rate = 1.0 - (fe / fp) if fp > 0 else 0.0

    # Incidents by state
    state_counts = {
        row.state: row.count
        for row in db.execute(
            select(IncidentRow.state, func.count().label("count"))
            .group_by(IncidentRow.state)
        ).all()
    }

    # Incidents by severity (all-time)
    severity_counts = {
        row.severity: row.count
        for row in db.execute(
            select(IncidentRow.severity, func.count().label("count"))
            .group_by(IncidentRow.severity)
        ).all()
    }

    # Incidents by severity in the last 24 hours
    one_day_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    try:
        severity_24h: dict = {
            row.severity: row.count
            for row in db.execute(
                select(IncidentRow.severity, func.count().label("count"))
                .where(IncidentRow.created_at >= one_day_ago)
                .group_by(IncidentRow.severity)
            ).all()
        }
    except Exception:
        severity_24h = {}

    # Operator decision distribution
    decision_counts = {
        row.operator_decision: row.count
        for row in db.execute(
            select(IncidentRow.operator_decision, func.count().label("count"))
            .group_by(IncidentRow.operator_decision)
        ).all()
    }

    # Alerts in the last hour
    try:
        alerts_last_hour = db.execute(
            select(func.count()).select_from(IncidentRow).where(
                IncidentRow.state == "alert",
                IncidentRow.created_at >= func.datetime("now", "-1 hour"),
            )
        ).scalar_one()
    except Exception:
        alerts_last_hour = db.execute(
            select(func.count()).select_from(IncidentRow).where(
                IncidentRow.state == "alert"
            )
        ).scalar_one()

    # KB entry count — gracefully absent in SQLite test env (KBBase not in Base)
    try:
        kb_entry_count: Optional[int] = db.execute(
            sa_text("SELECT COUNT(*) FROM kb_entries")
        ).scalar_one()
    except Exception:
        kb_entry_count = None

    return {
        "frames_processed_total": fp,
        "frames_escalated_total": fe,
        "gate_filter_rate": gate_filter_rate,
        "process_video_calls": _counters.process_video_calls,
        "incidents_by_state": state_counts,
        "incidents_by_severity": severity_counts,
        "incidents_by_severity_24h": severity_24h,
        "operator_decisions": decision_counts,
        "alerts_last_hour": alerts_last_hour,
        "kb_entry_count": kb_entry_count,
    }


@app.get("/architecture")
def get_architecture():
    """System architecture: six layers with real/simulated status.

    Mirrors the README real/simulated table. The Architecture dashboard page
    reads this endpoint to render the layer diagram.
    """
    return {
        "layers": [
            {
                "id": "L1",
                "name": "Stream Ingest",
                "status": "not_built",
                "description": (
                    "Ingest is a file path passed to process_video. "
                    "Full RTSP ingest (per-camera workers, NVDEC decode, auto-reconnect) "
                    "is deferred to Slice 9."
                ),
                "components": [
                    {"name": "Video source", "real": "cv2.VideoCapture(file_path)", "production": "Multicast RTSP receiver"},
                    {"name": "Decode", "real": "OpenCV CPU decode", "production": "NVDEC H.264"},
                    {"name": "ROI crop", "real": "Not implemented", "production": "Per-camera config"},
                ],
            },
            {
                "id": "L2",
                "name": "Fast Classical CV",
                "status": "approximated",
                "description": (
                    "Real models running in-process on CPU. "
                    "No Triton/TensorRT; latency target (<=50 ms/frame) not met on this hardware."
                ),
                "components": [
                    {"name": "Detection", "real": "YOLOv8n via ultralytics", "production": "YOLOv8 on TensorRT"},
                    {"name": "Pose", "real": "RTMPose via rtmlib/ONNX", "production": "RTMPose on TensorRT"},
                    {"name": "Action", "real": "SlowFast R50 (hand-rolled preprocessing)", "production": "SlowFast on TensorRT"},
                    {"name": "Tracker", "real": "supervision.ByteTrack", "production": "ByteTrack"},
                ],
            },
            {
                "id": "gate",
                "name": "Event Gate",
                "status": "real",
                "description": "Seven deterministic rules over L2 outputs. Room policies in config/rooms.yaml.",
                "components": [
                    {"name": "Rules engine", "real": "services/event_gate/engine.py", "production": "Same"},
                    {"name": "Room policies", "real": "config/rooms.yaml", "production": "Same"},
                ],
            },
            {
                "id": "L3",
                "name": "VLM Deep Analysis",
                "status": "real",
                "description": (
                    "Real Qwen 2.5 VL 72B via HF Inference Providers. "
                    "Stub fallback via VLM_MODE env var."
                ),
                "components": [
                    {"name": "Model", "real": "Qwen2.5-VL-72B-Instruct (featherless-ai/ovhcloud)", "production": "Same or self-hosted"},
                    {"name": "KB context injection", "real": "Top-3 pgvector matches prepended to prompt", "production": "Same"},
                ],
            },
            {
                "id": "L4",
                "name": "AI Agent",
                "status": "real",
                "description": "LangGraph 6-node StateGraph with policy engine, confidence fusion, incident FSM.",
                "components": [
                    {"name": "Agent framework", "real": "LangGraph StateGraph", "production": "Same"},
                    {"name": "Policy", "real": "YAML rules in config/policies/", "production": "Same"},
                    {"name": "Confidence fusion", "real": "Weighted sum with redistribution", "production": "Same"},
                ],
            },
            {
                "id": "L5",
                "name": "Persistence + KB",
                "status": "partial",
                "description": (
                    "Postgres incidents + pgvector KB are real. "
                    "Evidence clips save to local filesystem; MinIO upload is deferred to Slice 9."
                ),
                "components": [
                    {"name": "Incidents DB", "real": "Postgres + SQLAlchemy", "production": "Same"},
                    {"name": "KB", "real": "pgvector + sentence-transformers", "production": "Same"},
                    {"name": "Clips", "real": "Local filesystem (clips/)", "production": "MinIO"},
                    {"name": "Pre-incident buffer", "real": "90-frame ring buffer (~3s)", "production": "900-frame ring buffer (30s)"},
                ],
            },
            {
                "id": "L6",
                "name": "Service Plane",
                "status": "partial",
                "description": (
                    "FastAPI REST + WebSocket are real. "
                    "Next.js dashboard is Slice 8b/8c. Redis pub/sub is documented but not wired."
                ),
                "components": [
                    {"name": "REST API", "real": "FastAPI", "production": "Same"},
                    {"name": "WebSocket", "real": "In-memory ConnectionManager", "production": "Redis pub/sub"},
                    {"name": "Dashboard", "real": "Built (Slice 8b/8c) — Next.js 14", "production": "Next.js 14"},
                    {"name": "Feedback loop", "real": "POST /incidents/{id}/feedback -> KB write", "production": "Same"},
                ],
            },
        ]
    }


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
