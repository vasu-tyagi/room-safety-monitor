"""Tests for the shared Incident Pydantic schema."""
from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from shared.schemas.incident import Incident


def _valid_kwargs():
    return dict(
        camera_id="cam0",
        room_id="room0",
        event_type="person_detected",
        severity="low",
        confidence=0.5,
        rationale="stub",
    )


def test_defaults_are_populated():
    inc = Incident(**_valid_kwargs())
    assert isinstance(inc.id, UUID)
    assert isinstance(inc.created_at, datetime)
    assert inc.state == "new"
    assert inc.evidence_clip_url is None


def test_invalid_state_rejected():
    with pytest.raises(ValidationError):
        Incident(**_valid_kwargs(), state="bogus")


def test_confidence_must_be_float():
    with pytest.raises(ValidationError):
        Incident(**{**_valid_kwargs(), "confidence": "not-a-number"})


def test_from_attributes_roundtrip():
    class Row:
        pass

    row = Row()
    for k, v in _valid_kwargs().items():
        setattr(row, k, v)
    row.id = UUID("12345678-1234-5678-1234-567812345678")
    row.state = "alert"
    row.created_at = datetime(2026, 6, 17, 12, 0, 0)
    row.evidence_clip_url = None

    inc = Incident.model_validate(row)
    assert inc.state == "alert"
    assert inc.camera_id == "cam0"
