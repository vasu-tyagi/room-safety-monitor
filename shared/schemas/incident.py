"""Incident schema shared across layers.

This is the canonical wire/API representation of an incident. The SQLAlchemy
row model in services/persistence/db.py stores the same fields; this Pydantic
model is what the service plane (L6) serializes to clients.
"""
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

IncidentState = Literal["new", "alert", "resolved", "dismissed"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Incident(BaseModel):
    """A single safety incident as it flows through and out of the system."""

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

    # Allow construction directly from the SQLAlchemy row object.
    model_config = {"from_attributes": True}
