"""Slice 8a backend API tests.

TDD: written before the production code they cover.

Tests:
  - GET /incidents with each supported filter
  - GET /incidents/{id} full detail
  - POST /incidents/{id}/feedback (operator decision + KB write)
  - GET /metrics shape
  - GET /architecture shape
  - WebSocket /ws/alerts connection
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.persistence.db import Base
from services.persistence.db import Incident as IncidentRow
from services.service_plane.app import app, get_db, get_kb


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    mock_kb = MagicMock()
    mock_kb.write.return_value = uuid.uuid4()

    def override_get_kb():
        return mock_kb

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_kb] = override_get_kb
    yield TestClient(app), Session, mock_kb
    app.dependency_overrides.clear()


def _incident(**kwargs) -> IncidentRow:
    defaults = dict(
        camera_id="cam0", room_id="room0",
        event_type="fall", severity="high",
        confidence=0.85, rationale="Person fell.",
        state="alert",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return IncidentRow(**defaults)


# ---------------------------------------------------------------------------
# GET /incidents filtering
# ---------------------------------------------------------------------------

def test_filter_by_state(client):
    api, Session, _ = client
    s = Session()
    s.add(_incident(state="alert"))
    s.add(_incident(state="dismissed"))
    s.commit()

    resp = api.get("/incidents?state=alert")
    assert resp.status_code == 200
    body = resp.json()
    assert all(i["state"] == "alert" for i in body)
    assert len(body) == 1


def test_filter_by_camera_id(client):
    api, Session, _ = client
    s = Session()
    s.add(_incident(camera_id="cam1"))
    s.add(_incident(camera_id="cam2"))
    s.commit()

    resp = api.get("/incidents?camera_id=cam1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["camera_id"] == "cam1"


def test_filter_by_severity(client):
    api, Session, _ = client
    s = Session()
    s.add(_incident(severity="high"))
    s.add(_incident(severity="low"))
    s.commit()

    resp = api.get("/incidents?severity=high")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["severity"] == "high"


def test_filter_limit_and_offset(client):
    api, Session, _ = client
    s = Session()
    for i in range(5):
        s.add(_incident(camera_id=f"cam{i}"))
    s.commit()

    resp = api.get("/incidents?limit=2&offset=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_filter_by_date_range(client):
    api, Session, _ = client
    now = datetime.now(timezone.utc)
    s = Session()
    s.add(_incident(created_at=now - timedelta(hours=2)))
    s.add(_incident(created_at=now - timedelta(minutes=10)))
    s.commit()

    # Only the recent one falls within the last hour.
    # Use params= so TestClient handles URL-encoding of the timezone +00:00 offset.
    from_dt = (now - timedelta(hours=1)).isoformat()
    resp = api.get("/incidents", params={"from": from_dt})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


# ---------------------------------------------------------------------------
# GET /incidents/{id} full detail
# ---------------------------------------------------------------------------

def test_incident_detail_returns_all_fields(client):
    api, Session, _ = client
    s = Session()
    incident_id = uuid.uuid4()
    s.add(_incident(
        id=incident_id,
        vlm_prompt="Describe what you see.",
        fired_rules=["fall_pose_detected"],
        confidence_breakdown={"yolo": 0.9, "pose": 0.85, "vlm": 0.0},
        kb_matches=[{"id": str(uuid.uuid4()), "label": "fall", "similarity": 0.92}],
        operator_decision=None,
    ))
    s.commit()

    resp = api.get(f"/incidents/{incident_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(incident_id)
    assert body["vlm_prompt"] == "Describe what you see."
    assert body["fired_rules"] == ["fall_pose_detected"]
    assert body["confidence_breakdown"]["yolo"] == pytest.approx(0.9)
    assert len(body["kb_matches"]) == 1
    assert body["operator_decision"] is None


def test_incident_detail_404_for_unknown(client):
    api, _, _ = client
    resp = api.get(f"/incidents/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /incidents/{id}/feedback
# ---------------------------------------------------------------------------

def test_feedback_confirm_updates_operator_decision(client):
    api, Session, _ = client
    s = Session()
    incident_id = uuid.uuid4()
    s.add(_incident(id=incident_id, state="alert"))
    s.commit()

    resp = api.post(f"/incidents/{incident_id}/feedback", json={"decision": "confirm"})
    assert resp.status_code == 200

    s.expire_all()
    row = s.get(IncidentRow, incident_id)
    assert row.operator_decision == "confirmed"


def test_feedback_dismiss_updates_operator_decision(client):
    api, Session, _ = client
    s = Session()
    incident_id = uuid.uuid4()
    s.add(_incident(id=incident_id, state="alert"))
    s.commit()

    resp = api.post(f"/incidents/{incident_id}/feedback", json={"decision": "dismiss"})
    assert resp.status_code == 200

    s.expire_all()
    row = s.get(IncidentRow, incident_id)
    assert row.operator_decision == "dismissed"


def test_feedback_writes_kb_entry(client):
    api, Session, mock_kb = client
    s = Session()
    incident_id = uuid.uuid4()
    s.add(_incident(
        id=incident_id,
        vlm_prompt="some VLM prompt",
        rationale="Person fell.",
        event_type="fall",
    ))
    s.commit()

    resp = api.post(f"/incidents/{incident_id}/feedback", json={"decision": "confirm"})
    assert resp.status_code == 200
    mock_kb.write.assert_called_once()
    call_kwargs = mock_kb.write.call_args.kwargs
    assert call_kwargs["operator_decision"] == "confirmed"
    assert call_kwargs["label"] == "fall"


def test_feedback_stub_vlm_tags_kb_metadata(client):
    """When vlm_prompt is None (stub path), metadata includes vlm_source='stub'."""
    api, Session, mock_kb = client
    s = Session()
    incident_id = uuid.uuid4()
    s.add(_incident(
        id=incident_id,
        vlm_prompt=None,  # stub path
        rationale="No VLM analysis available.",
        event_type="fall",
    ))
    s.commit()

    api.post(f"/incidents/{incident_id}/feedback", json={"decision": "dismiss"})
    mock_kb.write.assert_called_once()
    meta = mock_kb.write.call_args.kwargs["metadata"]
    assert meta.get("vlm_source") == "stub"


def test_feedback_real_vlm_tags_kb_metadata_as_real(client):
    """When vlm_prompt is present, metadata marks vlm_source='real'."""
    api, Session, mock_kb = client
    s = Session()
    incident_id = uuid.uuid4()
    s.add(_incident(
        id=incident_id,
        vlm_prompt="Describe this scene.",
        rationale="Person is lying prone.",
    ))
    s.commit()

    api.post(f"/incidents/{incident_id}/feedback", json={"decision": "confirm"})
    meta = mock_kb.write.call_args.kwargs["metadata"]
    assert meta.get("vlm_source") == "real"


def test_feedback_404_for_unknown_incident(client):
    api, _, _ = client
    resp = api.post(f"/incidents/{uuid.uuid4()}/feedback", json={"decision": "confirm"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_expected_shape(client):
    api, _, _ = client
    resp = api.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "frames_processed_total" in body
    assert "frames_escalated_total" in body
    assert "gate_filter_rate" in body
    assert "incidents_by_state" in body
    assert "incidents_by_severity" in body
    assert "operator_decisions" in body
    assert "alerts_last_hour" in body


# ---------------------------------------------------------------------------
# GET /architecture
# ---------------------------------------------------------------------------

def test_architecture_returns_six_layers(client):
    api, _, _ = client
    resp = api.get("/architecture")
    assert resp.status_code == 200
    body = resp.json()
    assert "layers" in body
    layer_ids = [l["id"] for l in body["layers"]]
    for expected in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        assert expected in layer_ids, f"{expected} missing from architecture"


def test_architecture_layer_has_required_fields(client):
    api, _, _ = client
    body = api.get("/architecture").json()
    for layer in body["layers"]:
        assert "id" in layer
        assert "name" in layer
        assert "status" in layer
        assert layer["status"] in ("real", "approximated", "substituted", "not_built", "partial")


# ---------------------------------------------------------------------------
# WebSocket /ws/alerts
# ---------------------------------------------------------------------------

def test_websocket_connection_accepted(client):
    api, _, _ = client
    with api.websocket_connect("/ws/alerts") as ws:
        # Connection established — server must send nothing on connect.
        # Just verify the handshake succeeds (no exception = connected).
        pass
