"""Incident schema shared across layers.

This is the canonical wire/API representation of an incident. The SQLAlchemy
row model in services/persistence/db.py stores the same fields; this Pydantic
model is what the service plane (L6) serializes to clients.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

IncidentState = Literal["new", "alert", "resolved", "dismissed"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Incident(BaseModel):
    """Wire/API representation for the incident list endpoint."""

    id: UUID = Field(default_factory=uuid4)
    camera_id: str
    room_id: str
    event_type: str
    severity: str
    confidence: float
    rationale: str
    state: IncidentState = "new"
    created_at: datetime = Field(default_factory=_utcnow)
    evidence_clip_url: Optional[str] = None

    model_config = {"from_attributes": True}


class IncidentDetail(Incident):
    """Full incident representation for the detail endpoint and Inspector page."""

    vlm_prompt: Optional[str] = None
    fired_rules: Optional[List[str]] = None
    confidence_breakdown: Optional[Dict[str, float]] = None
    kb_matches: Optional[List[Dict[str, Any]]] = None
    operator_decision: Optional[str] = None
