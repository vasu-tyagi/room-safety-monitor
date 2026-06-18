"""L2 perception pipeline (Slice 2). Supersedes the Slice 1 stub skeleton.

Runs detection -> pose -> pose-geometry fall over each frame of a video. When a
person stays fallen for the persistence window, it writes a real incident whose
rationale comes from the pose geometry. If an action recognizer is supplied, the
SlowFast action label over the recent frame buffer is attached to the rationale.

The formal Event Gate (Slice 4) and VLM confirmation (Slice 5) are not here yet;
this pipeline turns the pose-fall signal directly into incidents.
"""
import os
from collections import deque

import cv2

from services.perception.l2 import (
    DEFAULT_PERSIST_FRAMES,
    FallPersistenceTracker,
    L2Perception,
)
from services.persistence.db import Incident as IncidentRow

CLIP_LEN = 32  # frames buffered for the action recognizer


def _build_l2(detector, pose_estimator):
    if detector is None:
        from services.perception.detection import Detector

        detector = Detector()
    if pose_estimator is None:
        from services.perception.pose import PoseEstimator

        pose_estimator = PoseEstimator()
    return L2Perception(detector, pose_estimator)


def _fall_incident(camera_id, room_id, frame_idx, fall, action_result):
    rationale = (
        f"Pose-geometry fall (Slice 2). Frame {frame_idx}: torso angle "
        f"{fall.torso_angle_deg:.0f} deg from vertical, sustained past the "
        f"persistence window."
    )
    if action_result is not None:
        rationale += (
            f" Action: {action_result.label} "
            f"(target={action_result.target_action}, "
            f"conf={action_result.confidence:.2f})."
        )
    return IncidentRow(
        camera_id=camera_id,
        room_id=room_id,
        event_type="fall",
        severity="high",
        confidence=float(fall.confidence),
        rationale=rationale,
        state="new",
        evidence_clip_url=None,
    )


def process_video(
    video_path,
    session,
    detector=None,
    pose_estimator=None,
    action_recognizer=None,
    camera_id="cam0",
    room_id="room0",
    persist=DEFAULT_PERSIST_FRAMES,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    l2 = _build_l2(detector, pose_estimator)
    tracker = FallPersistenceTracker(persist=persist)
    buffer = deque(maxlen=CLIP_LEN)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    created = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buffer.append(frame)
            result = l2.process_frame(frame)
            confirmed = tracker.update(result.any_fall)
            if confirmed:
                fall = _strongest_fall(result)
                action_result = None
                if action_recognizer is not None and len(buffer) > 0:
                    action_result = action_recognizer.recognize(list(buffer))
                session.add(
                    _fall_incident(camera_id, room_id, frame_idx, fall, action_result)
                )
                created += 1
            frame_idx += 1
    finally:
        cap.release()

    session.commit()
    return created


def _strongest_fall(result):
    """Pick the fall assessment of the most-horizontal person in the frame."""
    falls = [p.fall for p in result.persons if p.fall.is_fall]
    return max(falls, key=lambda f: f.confidence)
