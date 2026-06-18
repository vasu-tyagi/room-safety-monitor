"""L2 perception pipeline (Slice 3). Supersedes the Slice 2 version.

Runs detection -> ByteTrack -> pose -> pose-geometry fall over each frame of a
video. Fall persistence and pose history are tracked per-track ID so that two
people in the same frame don't share state. When a track's fall signal is
sustained for the persistence window, one incident is written.

If an action recognizer is supplied, the SlowFast label from the global frame
buffer is attached to the rationale (per-track frame buffers come in Slice 4).

The formal Event Gate (Slice 4) and VLM confirmation (Slice 5) are not here yet.
"""
import os
from collections import deque

import cv2

from services.perception.l2 import (
    DEFAULT_PERSIST_FRAMES,
    FallPersistenceTracker,
    L2Perception,
)
from services.perception.tracker import ByteTracker
from services.persistence.db import Incident as IncidentRow

CLIP_LEN = 32  # frames buffered for the action recognizer and per-track pose history


def _build_l2(detector, pose_estimator, tracker):
    if detector is None:
        from services.perception.detection import Detector

        detector = Detector()
    if pose_estimator is None:
        from services.perception.pose import PoseEstimator

        pose_estimator = PoseEstimator()
    return L2Perception(detector, pose_estimator, tracker=tracker)


def _fall_incident(camera_id, room_id, frame_idx, fall, action_result, track_id=-1):
    tid_str = f" track_id={track_id}" if track_id != -1 else ""
    rationale = (
        f"Pose-geometry fall (Slice 3).{tid_str} Frame {frame_idx}: torso angle "
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
    tracker=None,
    camera_id="cam0",
    room_id="room0",
    persist=DEFAULT_PERSIST_FRAMES,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    if tracker is None:
        tracker = ByteTracker()

    l2 = _build_l2(detector, pose_estimator, tracker)

    fall_trackers = {}   # track_id -> FallPersistenceTracker
    track_history = {}   # track_id -> deque[PersonPose], maxlen=CLIP_LEN
    # TODO: cleanup of stale tracker entries when a track is lost for N seconds.
    # Memory-bounded eviction deferred to Slice 9 polish.

    buffer = deque(maxlen=CLIP_LEN)  # global frame buffer for action recognizer

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

            for person in result.persons:
                tid = person.track_id
                if tid not in fall_trackers:
                    fall_trackers[tid] = FallPersistenceTracker(persist=persist)
                    track_history[tid] = deque(maxlen=CLIP_LEN)
                track_history[tid].append(person.pose)

                if fall_trackers[tid].update(person.fall.is_fall):
                    action_result = None
                    if action_recognizer is not None and len(buffer) > 0:
                        action_result = action_recognizer.recognize(list(buffer))
                    session.add(
                        _fall_incident(
                            camera_id, room_id, frame_idx,
                            person.fall, action_result, track_id=tid,
                        )
                    )
                    created += 1

            frame_idx += 1
    finally:
        cap.release()

    session.commit()
    return created


def _strongest_fall(result):
    """Pick the single track with highest fall confidence.

    Slice 3 rule: one incident per confirmed frame, from the track with the most
    confident fall geometry. Multi-track simultaneous falls are captured in
    subsequent frames via per-track fall trackers.
    """
    falls = [p.fall for p in result.persons if p.fall.is_fall]
    return max(falls, key=lambda f: f.confidence)
