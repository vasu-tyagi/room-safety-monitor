"""L2+Gate+VLM+KB+Agent perception pipeline (Slice 7). Supersedes Slice 6.

Data flow per frame:
  video frame -> ByteTrack -> L2 (detection, pose, fall) -> Event Gate -> L3 VLM
    -> Agent graph (policy, fusion, decide, KB write-back, persist)

The Event Gate filters ~99% of frames. Escalated frames go to the real Qwen 2.5 VL
client (or stub fallback) via services.vlm.dispatch. When fall persistence triggers,
the agent graph runs to decide ALERT or DISMISS, write a KB entry, and persist the
incident + audit row. The pipeline no longer writes incidents or KB entries directly.
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
from shared.schemas.pipeline import ProcessVideoResult

CLIP_LEN = 32           # SlowFast input window (frames fed to action recognizer)
CLIP_BUFFER_LEN = 90   # pre-incident ring buffer saved as evidence clip (~3 s at 30 fps)
                       # Production target is 30 s (900 frames, ~600 MB/camera); 90 is
                       # the demo-safe default.  Raise CLIP_BUFFER_LEN without code changes.


def _build_l2(detector, pose_estimator, tracker):
    if detector is None:
        from services.perception.detection import Detector
        detector = Detector()
    if pose_estimator is None:
        from services.perception.pose import PoseEstimator
        pose_estimator = PoseEstimator()
    return L2Perception(detector, pose_estimator, tracker=tracker)


def _detection_notes(person, frame_idx, action_result=None):
    """Build the L2 fall-geometry rationale string for the agent state."""
    tid = person.track_id
    notes = (
        f"Pose-geometry fall. track_id={tid}. Frame {frame_idx}: "
        f"torso angle {person.fall.torso_angle_deg:.0f} deg from vertical, "
        f"sustained past persistence window."
    )
    if action_result is not None:
        notes += (
            f" Action: {action_result.label} "
            f"(target={action_result.target_action}, conf={action_result.confidence:.2f})."
        )
    return notes


def _stub_vlm_result():
    from services.vlm.stub import VLMResult
    return VLMResult(label="unknown", rationale="No VLM analysis.", confidence=0.0, is_stub=True)


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
    kb=None,
    policy_dir="config/policies",
    clips_dir="clips",
    dry_run=False,
) -> ProcessVideoResult:
    """Process a video file through the L2+Gate+VLM+Agent pipeline.

    Args:
        room_policy: dict loaded from config/rooms.yaml for this room, or None
            to skip the event gate (all frames processed, no escalation metrics).
        kb: optional KB instance. Used by the agent for similarity retrieval
            and write-back.
        policy_dir: directory containing per-facility policy YAML files.

    Returns:
        ProcessVideoResult with incidents_created, frames_processed,
        frames_escalated, escalation_ratio, last_incident_state,
        last_fused_confidence, and last_rationale.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    if tracker is None:
        tracker = ByteTracker()

    l2 = _build_l2(detector, pose_estimator, tracker)

    gate_engine = None
    if room_policy is not None:
        from services.event_gate.engine import RulesEngine
        gate_engine = RulesEngine(room_policy, fps=30)

    from services.agent.graph import make_agent_graph
    agent_graph = make_agent_graph(policy_dir=policy_dir)

    facility_id = (room_policy or {}).get("facility_id", "default")
    room_tags = (room_policy or {}).get("tags", [])

    fall_trackers: dict = {}
    track_history: dict = {}

    buffer = deque(maxlen=CLIP_LEN)           # SlowFast / VLM input window
    clip_buffer = deque(maxlen=CLIP_BUFFER_LEN)  # pre-incident ring buffer for saved clips
    last_action_label = None
    last_vlm_result = None
    last_vlm_prompt = None
    last_fired_rules: list = []

    last_incident_state = ""
    last_fused_confidence = 0.0
    last_rationale = ""

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
            clip_buffer.append(frame)
            result = l2.process_frame(frame)
            frames_processed += 1

            # --- Event Gate ---
            if gate_engine is not None:
                fired_rules = gate_engine.evaluate(result, action_label=last_action_label)
                if fired_rules:
                    frames_escalated += 1
                    from services.vlm.dispatch import analyze_escalated
                    vlm_result, _reason, vlm_prompt = analyze_escalated(
                        list(buffer), fired_rules, kb=kb
                    )
                    last_vlm_result = vlm_result
                    last_vlm_prompt = vlm_prompt
                    last_fired_rules = list(fired_rules)

            # --- Per-track fall persistence -> agent graph ---
            for person in result.persons:
                tid = person.track_id
                if tid not in fall_trackers:
                    fall_trackers[tid] = FallPersistenceTracker(persist=persist)
                    track_history[tid] = deque(maxlen=CLIP_LEN)
                track_history[tid].append(person.pose)

                if fall_trackers[tid].update(person.fall.is_fall):
                    action_result = None
                    action_conf = None
                    if action_recognizer is not None and len(buffer) > 0:
                        action_result = action_recognizer.recognize(list(buffer))
                        if action_result is not None:
                            last_action_label = action_result.target_action
                            action_conf = action_result.confidence

                    vlm = last_vlm_result if last_vlm_result is not None else _stub_vlm_result()

                    from services.agent.state import AgentState
                    initial_state: AgentState = {
                        "session": session,
                        "kb": kb,
                        "camera_id": camera_id,
                        "room_id": room_id,
                        "facility_id": facility_id,
                        "room_tags": room_tags,
                        "fired_rules": last_fired_rules,
                        "frame_idx": frames_processed - 1,
                        "detection_notes": _detection_notes(
                            person, frames_processed - 1, action_result
                        ),
                        "yolo_confidence": person.detection.confidence,
                        "pose_confidence": float(person.fall.confidence),
                        "action_confidence": action_conf,
                        "vlm_result": vlm,
                        "vlm_prompt": last_vlm_prompt,
                        # Placeholders filled by graph nodes
                        "label": "",
                        "severity": "",
                        "rationale": "",
                        "policy_suppress": False,
                        "policy_threshold": 0.7,
                        "policy_severity": None,
                        "policy_notes": [],
                        "fusion_weights": {},
                        "fused_confidence": 0.0,
                        "decision": "",
                        "incident_state": "",
                        "decide_reason": "",
                        # Slice 7.5: clip storage
                        "clip_frames": list(clip_buffer),
                        "clips_dir": clips_dir,
                        "dry_run": dry_run,
                    }

                    final_state = agent_graph.invoke(initial_state)
                    incidents_created += 1
                    last_incident_state = final_state.get("incident_state", "")
                    last_fused_confidence = final_state.get("fused_confidence", 0.0)
                    last_rationale = final_state.get("rationale", "")

    finally:
        cap.release()

    # session.commit() is called inside the agent's persist node; this is a
    # no-op if the session was already committed.
    session.commit()

    escalation_ratio = frames_escalated / frames_processed if frames_processed else 0.0
    return ProcessVideoResult(
        incidents_created=incidents_created,
        frames_processed=frames_processed,
        frames_escalated=frames_escalated,
        escalation_ratio=escalation_ratio,
        last_incident_state=last_incident_state,
        last_fused_confidence=last_fused_confidence,
        last_rationale=last_rationale,
    )


def _strongest_fall(result):
    """Pick the single track with highest fall confidence."""
    falls = [p.fall for p in result.persons if p.fall.is_fall]
    return max(falls, key=lambda f: f.confidence)
