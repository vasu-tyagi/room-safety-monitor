"""Persistence layer (L5/L7): SQLAlchemy models and session factory.

incidents — one row per safety incident, written by the agent persist node.
incident_audit — FSM transition log written by the agent on every state change.

Both models use GUID (platform-independent UUID) so they work against Postgres
in deployment and SQLite in unit tests.
"""
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import CHAR, TypeDecorator

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/room_safety",
)


class GUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses Postgres' native UUID in deployment and a 32-char hex string
    elsewhere, so the same model runs against Postgres and SQLite (tests).
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Incident(Base):
    """Durable row form of shared.schemas.incident.Incident."""

    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    room_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="new")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    evidence_clip_url: Mapped[str | None] = mapped_column(String, nullable=True)


class IncidentAudit(Base):
    """FSM transition log: one row per incident state change.

    Written by the agent's persist node (Slice 7) whenever an incident moves
    from one state to another (new -> alert, new -> dismissed, etc.).
    incident_id is not a FK so audit rows can be written in the same transaction
    as the incident without ordering constraints.
    """

    __tablename__ = "incident_audit"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    incident_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    from_state: Mapped[str] = mapped_column(String(16), nullable=False)
    to_state: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_node: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


# create_engine is lazy; it does not connect until a session is used, so this
# is safe to import even when Postgres is not running.
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
