"""Service plane (L6): FastAPI application.

Slice 1 exposes the minimum surface to prove the loop:
  - GET  /health         liveness
  - POST /process_video  run the stub pipeline over a video path
  - GET  /incidents      list stored incidents

WebSocket alerts, search, and the feedback loop arrive in Slice 8.

Run it with:
  uvicorn services.service_plane.app:app --reload
"""
import os

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from services.persistence.db import Incident as IncidentRow
from services.persistence.db import SessionLocal
from services.pipeline.perception_pipeline import process_video as run_pipeline
from services.pipeline.replay import replay_incident
from shared.schemas.incident import Incident

app = FastAPI(title="Room Safety Monitor - Service Plane", version="0.1.0")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ProcessVideoRequest(BaseModel):
    video_path: str
    camera_id: str = "cam0"
    room_id: str = "room0"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process_video")
def process_video(req: ProcessVideoRequest, db: Session = Depends(get_db)):
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=400, detail=f"video not found: {req.video_path}")
    created = run_pipeline(
        req.video_path, db, camera_id=req.camera_id, room_id=req.room_id
    )
    return {"status": "ok", "incidents_created": created}


@app.get("/incidents", response_model=list[Incident])
def list_incidents(db: Session = Depends(get_db)):
    rows = (
        db.execute(select(IncidentRow).order_by(IncidentRow.created_at.desc()))
        .scalars()
        .all()
    )
    return rows


@app.post("/incidents/{incident_id}/replay")
def replay_incident_endpoint(incident_id: str, db: Session = Depends(get_db)):
    """Re-run the evidence clip for incident_id through the current pipeline.

    Returns the original outcome alongside the replayed outcome and a
    structured diff. The replay never persists new incidents.
    """
    try:
        return replay_incident(incident_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
