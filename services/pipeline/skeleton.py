"""Slice 1 stub pipeline.

Proves the end-to-end path: video file -> YOLOv8n person detection -> stored
incident. It is deliberately dumb. For every Nth frame that contains at least
one person it writes a STUB incident. There is no pose, action, gating, VLM,
or fusion yet; those arrive in Slices 2-7.

YOLO is the only real perception here, reused from src/detect.py.
"""
import os

import cv2

from services.persistence.db import Incident as IncidentRow

PERSON = 0  # COCO class 0 is "person"; everything else is ignored
SAMPLE_EVERY = 30  # emit a stub incident on every Nth person-bearing frame

_model = None


def _get_model():
    """Lazily load YOLOv8n so importing this module does not require torch."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO("yolov8n.pt")
    return _model


def _stub_incident(camera_id, room_id, frame_idx, count) -> IncidentRow:
    return IncidentRow(
        camera_id=camera_id,
        room_id=room_id,
        event_type="person_detected",
        severity="low",
        confidence=0.5,
        rationale=(
            f"STUB incident (Slice 1 skeleton). Frame {frame_idx}: YOLOv8n "
            f"detected {count} person(s). No fall/action analysis yet."
        ),
        state="new",
        evidence_clip_url=None,
    )


def process_video(video_path, session, camera_id="cam0", room_id="room0", model=None):
    """Run the stub pipeline over a video and persist stub incidents.

    Args:
        video_path: path to a readable video file.
        session: an open SQLAlchemy session to write incidents into.
        camera_id, room_id: identifiers stamped onto each incident.
        model: optional detector override (tests inject a fake one).

    Returns:
        The number of stub incidents created.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    detector = model or _get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    created = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector(frame, classes=[PERSON], conf=0.4, verbose=False)[0]
            count = len(result.boxes)
            if count >= 1 and frame_idx % SAMPLE_EVERY == 0:
                session.add(_stub_incident(camera_id, room_id, frame_idx, count))
                created += 1
            frame_idx += 1
    finally:
        cap.release()

    session.commit()
    return created
