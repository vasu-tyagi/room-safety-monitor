"""AgentState TypedDict for the Slice 7 LangGraph agent (services/agent/graph.py).

Fields are populated incrementally as the graph runs through its nodes.
session and kb are runtime dependencies (not serializable; no checkpointing used).
"""
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class FramePerception(TypedDict):
    """Per-frame perception data carried alongside clip_frames for overlay rendering.

    detections: list of Detection dataclasses (bbox, confidence, class_id, …)
    poses:      list of PersonPose dataclasses (keypoints ndarray, scores ndarray)
    Indices correspond 1-to-1: detections[i] and poses[i] describe the same person.
    """
    detections: List  # list[Detection] from services.perception.detection
    poses: List       # list[PersonPose] from services.perception.pose


class AgentState(TypedDict):
    # Runtime dependencies — captured at graph invocation, not checkpointed.
    session: Any         # SQLAlchemy Session for incident + audit writes
    kb: Any              # Optional KB instance for similarity write-back

    # Input context (supplied by the pipeline at invocation time).
    camera_id: str
    room_id: str
    facility_id: str     # maps to config/policies/<facility_id>.yaml
    room_tags: List[str] # from room_policy.get("tags", [])
    fired_rules: List[str]
    frame_idx: int
    detection_notes: str  # L2 fall-geometry rationale built by the pipeline

    # Per-source confidence scores from L2.
    # action_confidence is None when SlowFast was not run (weight redistributed).
    yolo_confidence: float
    pose_confidence: float
    action_confidence: Optional[float]

    # VLM output from dispatch.analyze_escalated.
    vlm_result: Any           # VLMResult dataclass
    vlm_prompt: Optional[str]

    # Filled by parse_vlm_output.
    label: str
    severity: str
    rationale: str

    # Filled by policy_check.
    policy_suppress: bool
    policy_threshold: float
    policy_severity: Optional[str]
    policy_notes: List[str]
    fusion_weights: Dict[str, float]

    # Filled by confidence_fusion.
    fused_confidence: float
    confidence_breakdown: Dict   # raw per-source scores before fusion weighting
    kb_matches: List             # serialised KB entries retrieved during fusion

    # Filled by decide.
    decision: str       # "alert" | "dismiss"
    incident_state: str  # "alert" | "dismissed"
    decide_reason: str

    # Slice 7.5: clip storage and replay control.
    # clip_frames: BGR frames captured by the pre-incident ring buffer.
    # clips_dir: destination directory for the saved MP4 (default "clips").
    # dry_run: when True the persist node skips all DB and filesystem writes.
    clip_frames: List
    clips_dir: str
    dry_run: bool

    # Slice 9: per-frame perception data for clip overlay rendering.
    # Parallel to clip_frames; index i holds detections+poses for clip_frames[i].
    frame_perception_data: List  # list[FramePerception]
