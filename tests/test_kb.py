"""Tests for Slice 6 KB: embedder + pgvector knowledge base.

KB tests require a running Docker daemon (testcontainers spins up
pgvector/pgvector:pg16). They are skipped automatically when Docker is not
available or testcontainers is not installed.
"""
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from testcontainers.postgres import PostgresContainer
    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_frame(w=64, h=48):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Embedder tests (no Postgres needed)
# ---------------------------------------------------------------------------

def test_embed_output_is_768_dim():
    from services.kb.embedder import embed
    vec = embed("person fell on the floor")
    assert len(vec) == 768


def test_embed_same_input_same_vector():
    from services.kb.embedder import embed
    v1 = embed("person fell on the floor")
    v2 = embed("person fell on the floor")
    assert v1 == v2


def test_embed_different_inputs_different_vectors():
    from services.kb.embedder import embed
    v1 = embed("person fell on the floor")
    v2 = embed("fire detected in kitchen")
    assert v1 != v2


# ---------------------------------------------------------------------------
# Postgres fixture (module-scoped: one container, one schema, per test file)
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
def kb_session(pg_url):
    """Fresh SQLAlchemy session; deletes all KB rows after each test."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(pg_url)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.execute(text("DELETE FROM kb_entries"))
    session.commit()
    session.close()
    engine.dispose()


@pytest.fixture
def kb(kb_session):
    from services.kb.kb import KB
    return KB(kb_session)


# ---------------------------------------------------------------------------
# KB unit tests
# ---------------------------------------------------------------------------

def test_kb_write_returns_uuid(kb):
    uid = kb.write(
        prompt="test prompt",
        rationale="person lying on floor",
        label="fall",
        operator_decision="pending",
        metadata={"camera_id": "c1", "room_id": "r1", "frame_idx": 42, "fired_rules": ["fall_pose_detected"]},
    )
    assert isinstance(uid, uuid.UUID)


def test_kb_find_similar_empty_kb_returns_empty(kb):
    results = kb.find_similar("person fell")
    assert results == []


def test_kb_write_then_find_similar_returns_entry(kb):
    kb.write(
        prompt="person fell on floor",
        rationale="lying motionless",
        label="fall",
        operator_decision="confirmed",
        metadata={"camera_id": "c1", "room_id": "r1", "frame_idx": 10, "fired_rules": []},
    )
    results = kb.find_similar("person fell on floor", threshold=0.5)
    assert len(results) >= 1
    assert results[0].label == "fall"
    assert results[0].similarity_score >= 0.5


def test_kb_entry_has_all_fields(kb):
    kb.write(
        prompt="test prompt text",
        rationale="test rationale",
        label="fall",
        operator_decision="pending",
        metadata={"camera_id": "cam1", "room_id": "room1", "frame_idx": 42, "fired_rules": ["r1"]},
    )
    results = kb.find_similar("test prompt text", threshold=0.5)
    assert len(results) >= 1
    entry = results[0]
    assert entry.label == "fall"
    assert entry.operator_decision == "pending"
    assert entry.rationale == "test rationale"
    assert entry.metadata["camera_id"] == "cam1"
    assert entry.metadata["frame_idx"] == 42
    assert isinstance(entry.id, uuid.UUID)


def test_kb_find_similar_high_threshold_filters_dissimilar(kb):
    kb.write(
        prompt="person fell on floor",
        rationale="lying down",
        label="fall",
        operator_decision="confirmed",
        metadata={},
    )
    # Near-identical query at threshold=0.9 → the entry matches (cosine similarity near 1)
    close = kb.find_similar("person fell on floor", threshold=0.9)
    assert len(close) >= 1

    # Unrelated query at same high threshold → filtered out
    unrelated = kb.find_similar(
        "boiling temperature of chemical compounds in laboratory",
        threshold=0.9,
    )
    assert len(unrelated) == 0


def test_kb_find_similar_returns_top_k(kb):
    for i in range(5):
        kb.write(
            prompt=f"person fell frame {i}",
            rationale="lying on floor",
            label="fall",
            operator_decision="pending",
            metadata={},
        )
    results = kb.find_similar("person fell", top_k=3, threshold=0.0)
    assert len(results) <= 3


def test_kb_find_similar_orders_by_similarity(kb):
    kb.write(
        prompt="person fell hard on the floor",
        rationale="lying on floor",
        label="fall",
        operator_decision="confirmed",
        metadata={},
    )
    kb.write(
        prompt="cooking fire in kitchen smoke alarm",
        rationale="smoke visible through camera",
        label="fight",
        operator_decision="dismissed",
        metadata={},
    )
    results = kb.find_similar("person fell on the ground", top_k=5, threshold=0.0)
    assert len(results) >= 2
    # Fall entry should rank higher for a fall query
    assert results[0].label == "fall"
    # Results are ordered most similar first
    for i in range(len(results) - 1):
        assert results[i].similarity_score >= results[i + 1].similarity_score


# ---------------------------------------------------------------------------
# Pipeline integration: KB entry written when VLM produces non-stub result
# ---------------------------------------------------------------------------

def _make_video(path, n_frames, size=(64, 48)):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_pipeline_writes_kb_entry_on_fall_incident(tmp_path, monkeypatch, kb_session):
    """process_video with a real KB writes an entry when a fall incident is created
    from a non-stub VLM result."""
    from sqlalchemy import text as sa_text
    from services.perception.detection import Detection
    from services.perception.fall import FallAssessment
    from services.perception.pose import PersonPose
    from services.pipeline.perception_pipeline import process_video
    from services.kb.kb import KB
    from services.vlm.stub import VLMResult

    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    fake_vlm = VLMResult(
        label="fall", rationale="Person lying on floor.", confidence=0.9, is_stub=False
    )

    class FakeDetector:
        def persons(self, frame, conf=None):
            return [Detection((0, 0, 50, 80), 0.9, 0, "person", "person")]

    kp = np.zeros((17, 2), dtype=float)
    kp[5] = (100, 150); kp[6] = (100, 170)
    kp[11] = (220, 150); kp[12] = (220, 170)

    class FakePose:
        def estimate(self, frame, bboxes):
            return [PersonPose(keypoints=kp, scores=np.ones(17)) for _ in bboxes]

    video = tmp_path / "fall.mp4"
    _make_video(video, 15)

    # Use a separate SQLite session for incidents (KB session is PG)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from services.persistence.db import Base
    inc_engine = create_engine(
        "sqlite://", future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(inc_engine)
    inc_session = sessionmaker(bind=inc_engine, expire_on_commit=False)()

    kb = KB(kb_session)

    with patch("services.vlm.dispatch.analyze_clip", return_value=fake_vlm):
        result = process_video(
            video, inc_session,
            detector=FakeDetector(),
            pose_estimator=FakePose(),
            persist=3,
            room_policy={},
            kb=kb,
        )

    assert result["incidents_created"] >= 1
    count = kb_session.execute(sa_text("SELECT COUNT(*) FROM kb_entries")).scalar()
    assert count >= 1
