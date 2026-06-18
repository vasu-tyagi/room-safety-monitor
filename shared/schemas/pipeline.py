"""Pipeline result schemas."""
from typing import TypedDict


class ProcessVideoResult(TypedDict):
    incidents_created: int
    frames_processed: int
    frames_escalated: int
    escalation_ratio: float
    last_incident_state: str    # "alert" | "dismissed" | "" when no incidents ran
    last_fused_confidence: float  # 0.0 when no incidents ran
    last_rationale: str         # "" when no incidents ran
