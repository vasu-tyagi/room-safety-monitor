"""Tests for the L2 perception pipeline (video -> sustained fall -> incident).

Uses injected fake detector/pose/action so no model weights are needed; a tiny
real video exercises the OpenCV read loop and the DB write path.
"""
import unittest.mock
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


def test_sustained_fall_creates_single_incident(tmp_path, monkeypatch):
    # Force stub VLM so the test is deterministic regardless of HF_TOKEN in env.
    # Gate fires fall_pose_detected; stub VLM returns confidence=0.0; stub-caution
    # requires fused >= threshold+0.1 (≈0.8). With only pose+yolo active the fused
    # score is ~0.4, so the agent dismisses.
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    result = process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
    )

    assert result["incidents_created"] == 1
    assert result["frames_processed"] == 10
    rows = session.execute(select(IncidentRow)).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_type == "fall"
    assert rows[0].severity == "low"
    assert rows[0].state == "dismissed"


def test_camera_and_room_ids_stored_in_incident(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
        camera_id="cam-X", room_id="room-Y",
    )

    rows = session.execute(select(IncidentRow)).scalars().all()
    assert len(rows) == 1
    assert rows[0].camera_id == "cam-X"
    assert rows[0].room_id == "room-Y"


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


# ---------------------------------------------------------------------------
# Slice 8d: progress_callback tests
# ---------------------------------------------------------------------------

def _events(cb_calls, layer=None, status=None):
    """Filter callback positional args by layer and/or status."""
    return [
        call.args[0] for call in cb_calls
        if (layer is None or call.args[0]["layer"] == layer)
        and (status is None or call.args[0]["status"] == status)
    ]


def test_progress_callback_emits_l1_and_l2(tmp_path, monkeypatch):
    """L1 complete and L2 processing are always emitted, even with no fall."""
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "ok.mp4"
    _make_video(video, 5)
    session = _session()

    cb = unittest.mock.Mock()
    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=False), persist=3,
        progress_callback=cb,
    )

    assert _events(cb.call_args_list, layer="L1", status="complete")
    assert _events(cb.call_args_list, layer="L2", status="processing")
    assert _events(cb.call_args_list, layer="L2", status="complete")


def test_progress_callback_emits_l3_on_escalation(tmp_path, monkeypatch):
    """L3 processing/complete fire when the gate escalates a frame to the VLM."""
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    cb = unittest.mock.Mock()
    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
        progress_callback=cb,
    )

    layers_seen = [call.args[0]["layer"] for call in cb.call_args_list]
    # Gate escalates because pose is lying → fall_pose_detected fires
    assert "L3" in layers_seen
    assert _events(cb.call_args_list, layer="L3", status="processing")
    assert _events(cb.call_args_list, layer="L3", status="complete")


def test_progress_callback_emits_l4_l5_on_agent_run(tmp_path, monkeypatch):
    """L4 and L5 events fire when fall persistence triggers the agent graph."""
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    cb = unittest.mock.Mock()
    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
        progress_callback=cb,
    )

    assert _events(cb.call_args_list, layer="L4", status="processing")
    assert _events(cb.call_args_list, layer="L4", status="complete")
    assert _events(cb.call_args_list, layer="L5", status="complete")


def test_progress_callback_not_called_when_none(tmp_path, monkeypatch):
    """process_video runs cleanly with progress_callback=None (default)."""
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    result = process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
        # no progress_callback argument
    )
    assert result["incidents_created"] == 1


def test_progress_callback_forward_order(tmp_path, monkeypatch):
    """Layer cascade order in emitted events is always L1→L2→gate→L3→L4→L5."""
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _session()

    cb = unittest.mock.Mock()
    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=True), persist=3,
        progress_callback=cb,
    )

    cascade = ["L1", "L2", "gate", "L3", "L4", "L5", "L6"]
    layers_emitted = [call.args[0]["layer"] for call in cb.call_args_list]
    # Map each emitted layer to its cascade index; verify non-decreasing order.
    indices = [cascade.index(l) for l in layers_emitted if l in cascade]
    assert indices == sorted(indices), f"Events went backward: {layers_emitted}"


# ---------------------------------------------------------------------------
# Regression tests: dedup + L1 singleton (added after overlay feature)
# ---------------------------------------------------------------------------

def test_l1_progress_event_fires_exactly_once(tmp_path, monkeypatch):
    """L1 complete is emitted exactly once per process_video call, never zero or twice."""
    monkeypatch.setenv("VLM_MODE", "stub")
    video = tmp_path / "ok.mp4"
    _make_video(video, 5)
    session = _session()

    cb = unittest.mock.Mock()
    process_video(
        video, session, detector=FakeDetector(),
        pose_estimator=FakePose(lying=False), persist=100,
        progress_callback=cb,
    )

    l1_events = _events(cb.call_args_list, layer="L1", status="complete")
    assert len(l1_events) == 1, f"expected exactly 1 L1 complete event, got {len(l1_events)}"


def test_two_falling_people_create_two_incidents(tmp_path, monkeypatch):
    """Two distinct persons each triggering fall persistence should each get an incident."""
    monkeypatch.setenv("VLM_MODE", "stub")

    class TwoPersonDetector:
        def persons(self, frame, conf=None):
            return [
                Detection((0, 0, 25, 40), 0.9, 0, "person", "person"),
                Detection((35, 0, 60, 40), 0.9, 0, "person", "person"),
            ]

    video = tmp_path / "fall.mp4"
    _make_video(video, 20)
    session = _session()

    result = process_video(
        video, session,
        detector=TwoPersonDetector(),
        pose_estimator=FakePose(lying=True),
        persist=3,
    )

    assert result["incidents_created"] == 2, (
        f"expected 2 incidents for 2 falling people, got {result['incidents_created']}"
    )


def test_vlm_fires_at_persistence_threshold_not_at_brief_gate_trigger(tmp_path, monkeypatch):
    """VLM fires when the fall is confirmed by FallPersistenceTracker, not on a brief gate trigger.

    Sequence with persist=5:
      Frames 1-3:  lying  (gate hits N=3, fires; tracker count=3, no fire yet)
      Frames 4-8:  standing (gate + tracker both reset; 5 upright frames clear velocity history)
      Frames 9-20: lying  (tracker hits persist=5 at frame 13; VLM fires here)

    The velocity check requires history[-6] < 30 deg. With 5 upright frames (4-8) the
    six-frame lookback at frame 13 still shows 0 deg, so detection is allowed.
    If VLM were wired to the gate trigger it would fire at frame 3; here it fires at 13.
    """
    monkeypatch.setenv("VLM_MODE", "stub")

    from services.vlm.stub import VLMResult
    stub_result = VLMResult(label="fall", rationale="fall detected", confidence=0.8, is_stub=False)

    frame_counter = [0]
    vlm_called_at_frame = []

    class BriefThenSustainedPose:
        def estimate(self, frame, bboxes):
            frame_counter[0] += 1
            n = frame_counter[0]
            lying = (n <= 3) or (n >= 9)  # brief bending 1-3, upright 4-8, real fall 9+
            return [_pose(lying=lying) for _ in bboxes]

    def fake_vlm(frames, fired_rules, kb=None):
        vlm_called_at_frame.append(frame_counter[0])
        return stub_result, "test", None

    video = tmp_path / "fall.mp4"
    _make_video(video, n_frames=25)
    session = _session()

    with unittest.mock.patch("services.vlm.dispatch.analyze_escalated", side_effect=fake_vlm):
        result = process_video(
            video, session,
            detector=FakeDetector(),
            pose_estimator=BriefThenSustainedPose(),
            persist=5,
        )

    assert result["incidents_created"] == 1, f"expected 1 incident, got {result['incidents_created']}"
    assert vlm_called_at_frame, "VLM was never called"
    # Fall starts at frame 9; with persist=5 and velocity check, tracker fires at frame 13.
    # If VLM is called before frame 9 it was triggered by the brief gate firing, not the fall.
    assert vlm_called_at_frame[0] >= 9, (
        f"VLM called at frame {vlm_called_at_frame[0]}: should be >= 9 (confirmed fall), "
        "not at frame 3 (spurious gate trigger from brief bending)"
    )


def _pose_at_deg(angle_deg):
    """Return a PersonPose whose torso angle (from vertical) equals angle_deg."""
    kp = np.zeros((17, 2), dtype=float)
    dx = 100.0 * np.tan(np.radians(angle_deg))
    kp[L_SHOULDER], kp[R_SHOULDER] = (140, 100), (160, 100)
    kp[L_HIP]      = (140 + dx, 200)
    kp[R_HIP]      = (160 + dx, 200)
    return PersonPose(keypoints=kp, scores=np.ones(17))


# ---------------------------------------------------------------------------
# Velocity-based fall detection: mopping / false-positive regression tests
# ---------------------------------------------------------------------------

def test_mopping_does_not_trigger_fall(tmp_path, monkeypatch):
    """Gradual bending (mopping) must not create an incident.

    Sequence (persist=5):
      Frames 1-5:  upright at ~5 deg
      Frames 6-8:  gradual bend at ~40 deg (transition, not a fall)
      Frames 9-20: sustained mop at ~74 deg

    The velocity check sees angle >= 30 deg in the 5-frame-ago slot by the
    time the tracker would reach count=5, so effective_is_fall resets first.
    """
    monkeypatch.setenv("VLM_MODE", "stub")

    frame_counter = [0]

    class MoppingPose:
        def estimate(self, frame, bboxes):
            frame_counter[0] += 1
            n = frame_counter[0]
            if n <= 5:
                return [_pose_at_deg(5) for _ in bboxes]
            elif n <= 8:
                return [_pose_at_deg(40) for _ in bboxes]
            else:
                return [_pose_at_deg(74) for _ in bboxes]

    video = tmp_path / "mop.mp4"
    _make_video(video, n_frames=20)
    session = _session()

    result = process_video(
        video, session,
        detector=FakeDetector(),
        pose_estimator=MoppingPose(),
        persist=5,
    )

    assert result["incidents_created"] == 0, (
        f"mopping should not trigger an incident, got {result['incidents_created']}"
    )


def test_fall_triggers_one_incident_per_person(tmp_path, monkeypatch):
    """Instant transition from upright to fallen fires exactly one incident.

    Sequence (persist=5):
      Frames 1-5:  upright at ~5 deg
      Frames 6-20: instant fall at ~74 deg

    The velocity check allows detection (history[-6] = 5 deg < 30 deg).
    Per-call dedup prevents more than one VLM call for the same track.
    """
    monkeypatch.setenv("VLM_MODE", "stub")

    frame_counter = [0]

    class InstantFallPose:
        def estimate(self, frame, bboxes):
            frame_counter[0] += 1
            n = frame_counter[0]
            lying = n > 5
            return [_pose_at_deg(74 if lying else 5) for _ in bboxes]

    video = tmp_path / "fall.mp4"
    _make_video(video, n_frames=20)
    session = _session()

    result = process_video(
        video, session,
        detector=FakeDetector(),
        pose_estimator=InstantFallPose(),
        persist=5,
    )

    assert result["incidents_created"] == 1, (
        f"instant fall should create exactly 1 incident, got {result['incidents_created']}"
    )
