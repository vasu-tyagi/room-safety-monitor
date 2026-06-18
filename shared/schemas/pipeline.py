"""Pipeline result schemas."""
from typing import TypedDict


class ProcessVideoResult(TypedDict):
    incidents_created: int
    frames_processed: int
    frames_escalated: int
    escalation_ratio: float
