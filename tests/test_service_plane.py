"""Tests for the L6 service plane API, backed by an in-memory SQLite DB."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.persistence.db import Base
from services.persistence.db import Incident as IncidentRow
from services.persistence.db import IncidentAudit
from services.service_plane.app import app, get_db

try:
    from testcontainers.postgres import PostgresContainer
    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False


@pytest.fixture
def client():
    # StaticPool + shared connection so the in-memory DB survives across the
    # request thread the TestClient uses.
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), TestingSession
    app.dependency_overrides.clear()


def test_health(client):
    api, _ = client
    resp = api.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_incidents_empty(client):
    api, _ = client
    resp = api.get("/incidents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_incidents_returns_stored_row(client):
    api, TestingSession = client
    session = TestingSession()
    session.add(
        IncidentRow(
            camera_id="cam7",
            room_id="ward-3",
            event_type="person_detected",
            severity="low",
            confidence=0.5,
            rationale="stub",
            state="new",
            created_at=datetime.now(timezone.utc),
        )
    )
    session.commit()
    session.close()

    resp = api.get("/incidents")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["camera_id"] == "cam7"
    assert body[0]["room_id"] == "ward-3"
    assert body[0]["state"] == "new"


def test_process_video_missing_path_returns_400(client):
    api, _ = client
    resp = api.post("/process_video", json={"video_path": "/no/such/file.mp4"})
    assert resp.status_code == 400


def test_cors_allowed_origin_returns_header(client):
    api, _ = client
    resp = api.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_localhost_alternate_port_returns_header(client):
    # Next.js may start on 3001 if 3000 is occupied; allow_origin_regex covers this.
    api, _ = client
    resp = api.get("/health", headers={"Origin": "http://localhost:3001"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3001"


def test_cors_disallowed_origin_omits_header(client):
    api, _ = client
    resp = api.get("/health", headers={"Origin": "http://evil.example.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_process_video_passes_camera_and_room_ids(client, tmp_path, monkeypatch):
    """process_video endpoint forwards camera_id and room_id from the request body."""
    import cv2, numpy as np
    from shared.schemas.pipeline import ProcessVideoResult

    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    captured: dict = {}

    def fake_pipeline(video_path, session, **kwargs):
        captured.update(kwargs)
        return ProcessVideoResult(
            incidents_created=0, frames_processed=3,
            frames_escalated=0, escalation_ratio=0.0,
            last_incident_state="", last_fused_confidence=0.0, last_rationale="",
        )

    monkeypatch.setattr("services.service_plane.app.run_pipeline", fake_pipeline)

    api, _ = client
    resp = api.post(
        "/process_video",
        json={"video_path": str(video), "camera_id": "cam-X", "room_id": "room-Y"},
    )
    assert resp.status_code == 200
    assert captured.get("camera_id") == "cam-X"
    assert captured.get("room_id") == "room-Y"


# ---------------------------------------------------------------------------
# Slice 8c additions
# ---------------------------------------------------------------------------

def _make_incident(session, **kwargs):
    defaults = dict(
        camera_id="cam0", room_id="room0", event_type="fall_detected",
        severity="high", confidence=0.9, rationale="Test rationale",
        state="alert", created_at=datetime.now(timezone.utc),
    )
    row = IncidentRow(**{**defaults, **kwargs})
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def test_incident_detail_returns_empty_audit_entries(client):
    """GET /incidents/{id} includes audit_entries list even when empty."""
    api, TestingSession = client
    session = TestingSession()
    row = _make_incident(session)
    session.close()

    resp = api.get(f"/incidents/{row.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "audit_entries" in body
    assert body["audit_entries"] == []


def test_incident_detail_returns_audit_entries(client):
    """GET /incidents/{id} includes FSM transitions from incident_audit."""
    api, TestingSession = client
    session = TestingSession()
    row = _make_incident(session, state="alert")
    audit = IncidentAudit(
        incident_id=row.id,
        from_state="new",
        to_state="alert",
        reason="confidence above threshold",
        agent_node="decide",
        created_at=datetime.now(timezone.utc),
    )
    session.add(audit)
    session.commit()
    session.close()

    resp = api.get(f"/incidents/{row.id}")
    assert resp.status_code == 200
    entries = resp.json()["audit_entries"]
    assert len(entries) == 1
    assert entries[0]["from_state"] == "new"
    assert entries[0]["to_state"] == "alert"
    assert entries[0]["agent_node"] == "decide"


def test_metrics_includes_slice8c_fields(client):
    """GET /metrics returns kb_entry_count and incidents_by_severity_24h."""
    api, _ = client
    resp = api.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    # kb_entry_count is None in SQLite test env (kb_entries table absent)
    assert "kb_entry_count" in body
    # incidents_by_severity_24h is always present (may be empty)
    assert "incidents_by_severity_24h" in body
    assert isinstance(body["incidents_by_severity_24h"], dict)


def test_metrics_severity_24h_counts_recent_incidents(client):
    """incidents_by_severity_24h reflects incidents created in the last 24h."""
    api, TestingSession = client
    session = TestingSession()
    _make_incident(session, severity="high")
    _make_incident(session, severity="high")
    _make_incident(session, severity="low")
    session.close()

    resp = api.get("/metrics")
    body = resp.json()
    sev_24h = body["incidents_by_severity_24h"]
    assert sev_24h.get("high", 0) == 2
    assert sev_24h.get("low", 0) == 1


def test_metrics_operator_decisions_has_explicit_keys(client):
    """operator_decisions uses confirmed/dismissed/pending keys, not None/null."""
    api, TestingSession = client
    session = TestingSession()
    _make_incident(session, state="alert")                              # pending
    _make_incident(session, state="alert", operator_decision="confirmed")
    _make_incident(session, state="dismissed", operator_decision="dismissed")
    session.close()

    resp = api.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    od = body["operator_decisions"]
    assert od["confirmed"] == 1
    assert od["dismissed"] == 1
    assert od["pending"] == 1
    assert "null" not in od


def test_metrics_includes_total_incidents(client):
    """GET /metrics response includes incidents_total reflecting the DB row count."""
    api, TestingSession = client
    session = TestingSession()
    _make_incident(session, severity="high")
    _make_incident(session, severity="low")
    session.close()

    resp = api.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "incidents_total" in body
    assert body["incidents_total"] == 2


def test_metrics_includes_service_start_time(client):
    """GET /metrics response includes service_start_time as an ISO 8601 string."""
    api, _ = client
    resp = api.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "service_start_time" in body
    assert body["service_start_time"] is not None


def test_metrics_alerts_last_hour_counts_only_recent(client):
    """alerts_last_hour counts alert-state incidents from the last hour only."""
    api, TestingSession = client
    session = TestingSession()
    # Recent alert (now, inside the window)
    _make_incident(session, state="alert")
    # Old alert (2 hours ago, outside the window)
    _make_incident(
        session, state="alert",
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    # Dismissed incident (inside window but wrong state)
    _make_incident(session, state="dismissed")
    session.close()

    resp = api.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["alerts_last_hour"] == 1


# ---------------------------------------------------------------------------
# Postgres tests — skipped when Docker / testcontainers unavailable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pg_url():
    """Start pgvector/pgvector:pg16 and run Alembic migrations."""
    if not _HAS_TESTCONTAINERS:
        pytest.skip("testcontainers[postgres] not installed")
    try:
        with PostgresContainer("pgvector/pgvector:pg16") as pg:
            url = pg.get_connection_url()
            from alembic.config import Config
            from alembic import command
            cfg = Config("alembic.ini")
            cfg.set_main_option("sqlalchemy.url", url)
            command.upgrade(cfg, "head")
            yield url
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture
def pg_client(pg_url):
    """FastAPI TestClient wired to the Postgres container."""
    from sqlalchemy import create_engine
    engine = create_engine(pg_url)
    PGSession = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        db = PGSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), PGSession
    app.dependency_overrides.clear()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM incidents"))
        conn.commit()


def test_metrics_alerts_last_hour_postgres(pg_client):
    """alerts_last_hour uses portable Python datetime, not SQLite func.datetime.

    Regression test for the bug where func.datetime('now', '-1 hour') aborted
    the Postgres transaction and caused /metrics to return 500.
    """
    api, PGSession = pg_client
    session = PGSession()
    # Alert inside the window (30 minutes ago)
    session.add(IncidentRow(
        camera_id="cam0", room_id="room0", event_type="fall",
        severity="high", confidence=0.9, rationale="recent",
        state="alert",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    ))
    # Alert outside the window (2 hours ago)
    session.add(IncidentRow(
        camera_id="cam0", room_id="room0", event_type="fall",
        severity="high", confidence=0.9, rationale="old",
        state="alert",
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    ))
    session.commit()
    session.close()

    resp = api.get("/metrics")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["alerts_last_hour"] == 1
