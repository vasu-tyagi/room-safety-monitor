"""SQLAlchemy model for the KB entries table (Slice 6).

Kept in a separate DeclarativeBase (KBBase) so that the incidents-only Base in
services/persistence/db.py can still be used against SQLite in unit tests
without hitting pgvector's Vector type, which has no SQLite backend.
"""
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class KBBase(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KBEntryRow(KBBase):
    __tablename__ = "kb_entries"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    embedding: Mapped[list] = mapped_column(Vector(768), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    rationale_text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    operator_decision: Mapped[str] = mapped_column(String(64), nullable=False, default="pending")
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
