"""Tests for the L6 service plane API, backed by an in-memory SQLite DB."""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.persistence.db import Base
from services.persistence.db import Incident as IncidentRow
from services.service_plane.app import app, get_db


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
