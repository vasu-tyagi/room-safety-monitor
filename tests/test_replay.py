"""Tests for Slice 7.5: incident clip storage and replay.

TDD: these tests are written before the production code they cover.

Structure:
  - Tests 1-2: clip is saved by process_video and linked to the incident row.
  - Tests 3-4: replay_incident returns the right structure and does not persist.
  - Tests 5-6: replay comparison logic (KB plumbed through, confidence_delta).
"""
import os
import uuid
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.perception.detection import Detection
from services.perception.pose import PersonPose
from services.persistence.db import Base
from services.persistence.db import Incident as IncidentRow
from services.pipeline.perception_pipeline import process_video

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _lying_pose():
    kp = np.zeros((17, 2), dtype=float)
    kp[L_SHOULDER], kp[R_SHOULDER] = (100, 150), (100, 170)
    kp[L_HIP], kp[R_HIP] = (220, 150), (220, 170)
    return PersonPose(keypoints=kp, scores=np.ones(17))


class FakeDetector:
    def persons(self, frame, conf=None):
        return [Detection((0, 0, 50, 80), 0.9, 0, "person", "person")]


class FakePose:
    def estimate(self, frame, bboxes):
        return [_lying_pose() for _ in bboxes]


def _make_video(path, n_frames=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def _session():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


# ---------------------------------------------------------------------------
# Tests 1-2: clip storage
# ---------------------------------------------------------------------------

def test_clip_saved_at_expected_path(tmp_path):
    """process_video saves an MP4 clip file when an incident is created."""
    video = tmp_path / "fall.mp4"
    _make_video(video)
    clips_dir = tmp_path / "clips"
    session = _session()

    result = process_video(
        video, session,
        detector=FakeDetector(),
        pose_estimator=FakePose(),
        persist=3,
        clips_dir=str(clips_dir),
    )

    assert result["incidents_created"] == 1
    mp4s = list(clips_dir.glob("*.mp4"))
    assert len(mp4s) == 1, f"expected 1 clip, found {mp4s}"


def test_evidence_clip_url_stored_in_incident_row(tmp_path):
    """Incident row evidence_clip_url points to the saved clip and the file exists."""
    video = tmp_path / "fall.mp4"
    _make_video(video)
    clips_dir = tmp_path / "clips"
    session = _session()

    process_video(
        video, session,
        detector=FakeDetector(),
        pose_estimator=FakePose(),
        persist=3,
        clips_dir=str(clips_dir),
    )

    rows = session.execute(select(IncidentRow)).scalars().all()
    assert len(rows) == 1
    url = rows[0].evidence_clip_url
    assert url is not None, "evidence_clip_url should not be None"
    assert url.endswith(".mp4")
    assert os.path.exists(url), f"clip file missing: {url}"


# ---------------------------------------------------------------------------
# Tests 3-4: basic replay
# ---------------------------------------------------------------------------

def test_replay_returns_structured_comparison(tmp_path):
    """replay_incident returns original_incident, replayed_outcome, and changed dict."""
    from services.pipeline.replay import replay_incident

    video = tmp_path / "fall.mp4"
    _make_video(video)
    clips_dir = tmp_path / "clips"
    session = _session()

    process_video(
        video, session,
        detector=FakeDetector(),
        pose_estimator=FakePose(),
        persist=3,
        clips_dir=str(clips_dir),
    )
    row = session.execute(select(IncidentRow)).scalars().first()

    result = replay_incident(str(row.id), session)

    assert "original_incident" in result
    assert "replayed_outcome" in result
    assert "changed" in result

    changed = result["changed"]
    assert "state_changed" in changed
    assert "confidence_delta" in changed
    assert "rationale_changed" in changed
    assert "any_change" in changed
    assert isinstance(changed["confidence_delta"], float)

    orig = result["original_incident"]
    assert orig["id"] == str(row.id)
    assert orig["state"] == row.state


def test_replay_does_not_persist_additional_incident(tmp_path):
    """Replay runs the pipeline without writing to the incidents table."""
    from services.pipeline.replay import replay_incident

    video = tmp_path / "fall.mp4"
    _make_video(video)
    clips_dir = tmp_path / "clips"
    session = _session()

    process_video(
        video, session,
        detector=FakeDetector(),
        pose_estimator=FakePose(),
        persist=3,
        clips_dir=str(clips_dir),
    )
    row = session.execute(select(IncidentRow)).scalars().first()
    count_before = len(session.execute(select(IncidentRow)).scalars().all())

    # Patch process_video inside replay to return a controlled result that
    # would include incidents_created=1, verifying dry_run prevents DB writes.
    from shared.schemas.pipeline import ProcessVideoResult
    with patch("services.pipeline.replay.process_video") as mock_pv:
        mock_pv.return_value = ProcessVideoResult(
            incidents_created=1,
            frames_processed=10,
            frames_escalated=3,
            escalation_ratio=0.3,
            last_incident_state="dismissed",
            last_fused_confidence=0.3,
            last_rationale="replayed rationale",
        )
        replay_incident(str(row.id), session)

    count_after = len(session.execute(select(IncidentRow)).scalars().all())
    assert count_after == count_before, (
        f"replay created extra incident rows: {count_before} -> {count_after}"
    )


# ---------------------------------------------------------------------------
# Tests 5-6: comparison logic and KB plumbing
# ---------------------------------------------------------------------------

def test_replay_state_changed_false_when_outcome_matches(tmp_path):
    """When replay returns the same state as the original, state_changed is False."""
    from services.pipeline.replay import replay_incident

    session = _session()
    clip_path = str(tmp_path / "fake.mp4")
    _make_video(clip_path)

    incident_id = uuid.uuid4()
    session.add(IncidentRow(
        id=incident_id,
        camera_id="cam0", room_id="room0",
        event_type="fall", severity="low",
        confidence=0.3, rationale="original rationale",
        state="dismissed", evidence_clip_url=clip_path,
    ))
    session.commit()

    from shared.schemas.pipeline import ProcessVideoResult
    with patch("services.pipeline.replay.process_video") as mock_pv:
        mock_pv.return_value = ProcessVideoResult(
            incidents_created=1,
            frames_processed=10,
            frames_escalated=3,
            escalation_ratio=0.3,
            last_incident_state="dismissed",
            last_fused_confidence=0.3,
            last_rationale="original rationale",
        )
        result = replay_incident(str(incident_id), session)

    assert result["changed"]["state_changed"] is False
    assert result["changed"]["any_change"] is False


def test_replay_kb_passed_through_and_confidence_delta_computed(tmp_path):
    """KB arg is forwarded to process_video; confidence_delta reflects the shift."""
    from services.pipeline.replay import replay_incident

    session = _session()
    clip_path = str(tmp_path / "fake.mp4")
    _make_video(clip_path)

    incident_id = uuid.uuid4()
    session.add(IncidentRow(
        id=incident_id,
        camera_id="cam0", room_id="room0",
        event_type="fall", severity="low",
        confidence=0.27, rationale="original rationale",
        state="dismissed", evidence_clip_url=clip_path,
    ))
    session.commit()

    mock_kb = MagicMock()
    from shared.schemas.pipeline import ProcessVideoResult
    with patch("services.pipeline.replay.process_video") as mock_pv:
        mock_pv.return_value = ProcessVideoResult(
            incidents_created=1,
            frames_processed=10,
            frames_escalated=3,
            escalation_ratio=0.3,
            last_incident_state="dismissed",
            last_fused_confidence=0.36,  # higher due to KB context
            last_rationale="Updated rationale with KB context.",
        )
        result = replay_incident(str(incident_id), session, kb=mock_kb)

        # KB was forwarded to process_video
        _, call_kwargs = mock_pv.call_args
        assert call_kwargs.get("kb") is mock_kb

    assert result["replayed_outcome"]["fused_confidence"] == pytest.approx(0.36)
    assert result["changed"]["confidence_delta"] == pytest.approx(0.36 - 0.27, abs=1e-6)
    assert result["changed"]["rationale_changed"] is True
