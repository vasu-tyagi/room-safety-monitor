"""AgentState TypedDict for the Slice 7 LangGraph agent (services/agent/graph.py).

Fields are populated incrementally as the graph runs through its nodes.
session and kb are runtime dependencies (not serializable; no checkpointing used).
"""
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


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

    # Filled by decide.
    decision: str       # "alert" | "dismiss"
    incident_state: str  # "alert" | "dismissed"
    decide_reason: str
