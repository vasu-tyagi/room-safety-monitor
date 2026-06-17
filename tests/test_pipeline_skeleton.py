"""Tests for the Slice 1 stub pipeline.

A fake detector is injected so the test needs neither YOLO weights nor torch.
A tiny real video file exercises the OpenCV read loop and DB write path.
"""
import cv2
import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.persistence.db import Base
from services.persistence.db import Incident as IncidentRow
from services.pipeline.skeleton import SAMPLE_EVERY, process_video


class _FakeResult:
    def __init__(self, n_people):
        self.boxes = [object()] * n_people


class _FakeModel:
    """Returns a fixed person count for every frame."""

    def __init__(self, n_people=1):
        self.n_people = n_people

    def __call__(self, frame, **kwargs):
        return [_FakeResult(self.n_people)]


def _make_video(path, n_frames, size=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
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


def test_creates_incident_every_nth_person_frame(tmp_path):
    n_frames = SAMPLE_EVERY * 2 + 1  # frames 0..60 -> hits at 0, 30, 60
    video = tmp_path / "clip.mp4"
    _make_video(video, n_frames)

    session = _session()
    created = process_video(video, session, model=_FakeModel(n_people=1))

    expected = len(range(0, n_frames, SAMPLE_EVERY))
    assert created == expected
    total = session.execute(select(func.count()).select_from(IncidentRow)).scalar_one()
    assert total == expected


def test_no_incident_when_no_person(tmp_path):
    video = tmp_path / "empty.mp4"
    _make_video(video, SAMPLE_EVERY * 2)

    session = _session()
    created = process_video(video, session, model=_FakeModel(n_people=0))

    assert created == 0


def test_missing_file_raises(tmp_path):
    session = _session()
    try:
        process_video(tmp_path / "nope.mp4", session, model=_FakeModel())
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
