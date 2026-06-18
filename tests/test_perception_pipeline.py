"""Tests for the L2 perception pipeline (video -> sustained fall -> incident).

Uses injected fake detector/pose/action so no model weights are needed; a tiny
real video exercises the OpenCV read loop and the DB write path.
"""
import cv2
import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.perception.action import ActionResult
from services.perception.detection import Detection
from services.perception.pose import PersonPose
from services.persistence.db import Base
from services.persistence.db import Incident as IncidentRow
from services.pipeline.perception_pipeline import process_video

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _pose(lying):
    kp = np.zeros((17, 2), dtype=float)
    if lying:
        kp[L_SHOULDER], kp[R_SHOULDER] = (100, 150), (100, 170)
        kp[L_HIP], kp[R_HIP] = (220, 150), (220, 170)
    else:
        kp[L_SHOULDER], kp[R_SHOULDER] = (140, 100), (160, 100)
        kp[L_HIP], kp[R_HIP] = (142, 210), (158, 210)
    return PersonPose(keypoints=kp, scores=np.ones(17))


class FakeDetector:
    def persons(self, frame, conf=None):
        return [Detection((0, 0, 50, 80), 0.9, 0, "person", "person")]


class FakePose:
    def __init__(self, lying):
        self.p = _pose(lying)

    def estimate(self, frame, bboxes):
        return [self.p for _ in bboxes]


class FakeAction:
    def recognize(self, frames):
        return ActionResult(label="falling off chair", target_action="falling", confidence=0.8)


def _make_video(path, n_frames, size=(64, 48)):
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


def test_sustained_fall_creates_single_incident(tmp_path):
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    result = process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
    )

    assert result["incidents_created"] == 1  # confirmed once, no duplicates while still down
    assert result["frames_processed"] == 10
    rows = session.execute(select(IncidentRow)).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_type == "fall"
    # No gate, no VLM — stub caution dismisses with severity="low" (Slice 7 agent).
    # Severity "high" required a real VLM fall classification; stub defaults to "low".
    assert rows[0].severity == "low"
    assert rows[0].state == "dismissed"


def test_standing_person_creates_no_incident(tmp_path):
    video = tmp_path / "ok.mp4"
    _make_video(video, 10)
    session = _session()

    result = process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=False), persist=3,
    )

    assert result["incidents_created"] == 0
    total = session.execute(select(func.count()).select_from(IncidentRow)).scalar_one()
    assert total == 0


def test_action_label_included_in_rationale(tmp_path):
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), action_recognizer=FakeAction(),
        persist=3,
    )

    row = session.execute(select(IncidentRow)).scalars().first()
    assert "falling" in row.rationale
