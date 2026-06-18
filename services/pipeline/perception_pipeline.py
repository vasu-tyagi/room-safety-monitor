"""L2+Gate perception pipeline (Slice 4). Supersedes the Slice 3 version.

Data flow per frame:
  video frame -> ByteTrack -> L2 (detection, pose, fall) -> Event Gate -> L3 stub

The Event Gate filters ~99% of frames; only frames where a rule fires are
escalated to the L3 stub. Real VLM analysis replaces the stub in Slice 5.

Incident creation still uses the Slice 3 per-track persistence path.
Slice 5 will move incident creation to the L3/L4 pipeline path.
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
from shared.schemas.pipeline import ProcessVideoResult

CLIP_LEN = 32  # frames buffered for action recognizer and per-track pose history


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
        f"Pose-geometry fall (Slice 3/4).{tid_str} Frame {frame_idx}: torso angle "
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
    room_policy=None,
    camera_id="cam0",
    room_id="room0",
    persist=DEFAULT_PERSIST_FRAMES,
) -> ProcessVideoResult:
    """Process a video file through the L2+Gate pipeline.

    Args:
        room_policy: dict loaded from config/rooms.yaml for this room, or None
            to skip the event gate (all frames processed, no escalation metrics).

    Returns:
        ProcessVideoResult with incidents_created, frames_processed,
        frames_escalated, and escalation_ratio.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    if tracker is None:
        tracker = ByteTracker()

    l2 = _build_l2(detector, pose_estimator, tracker)

    engine = None
    if room_policy is not None:
        from services.event_gate.engine import RulesEngine
        engine = RulesEngine(room_policy, fps=30)

    fall_trackers: dict = {}   # track_id -> FallPersistenceTracker
    track_history: dict = {}   # track_id -> deque[PersonPose], maxlen=CLIP_LEN
    # TODO: cleanup of stale tracker entries when a track is lost for N seconds.
    # Memory-bounded eviction deferred to Slice 9 polish.

    buffer = deque(maxlen=CLIP_LEN)
    last_action_label = None  # cached from most recent action recognizer run

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    incidents_created = 0
    frames_processed = 0
    frames_escalated = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            buffer.append(frame)
            result = l2.process_frame(frame)
            frames_processed += 1

            # --- Event Gate ---
            if engine is not None:
                fired_rules = engine.evaluate(result, action_label=last_action_label)
                if fired_rules:
                    frames_escalated += 1
                    from services.vlm.stub import analyze_frame
                    analyze_frame(result, fired_rules)  # stub: logged but no incident yet

            # --- Per-track fall persistence (incident creation path) ---
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
                        if action_result is not None:
                            last_action_label = action_result.target_action
                    session.add(
                        _fall_incident(
                            camera_id, room_id, frames_processed - 1,
                            person.fall, action_result, track_id=tid,
                        )
                    )
                    incidents_created += 1

    finally:
        cap.release()

    session.commit()

    escalation_ratio = frames_escalated / frames_processed if frames_processed else 0.0
    return ProcessVideoResult(
        incidents_created=incidents_created,
        frames_processed=frames_processed,
        frames_escalated=frames_escalated,
        escalation_ratio=escalation_ratio,
    )


def _strongest_fall(result):
    """Pick the single track with highest fall confidence.

    Slice 3 rule: one incident per confirmed frame, from the track with the most
    confident fall geometry. Multi-track simultaneous falls are captured in
    subsequent frames via per-track fall trackers.
    """
    falls = [p.fall for p in result.persons if p.fall.is_fall]
    return max(falls, key=lambda f: f.confidence)
