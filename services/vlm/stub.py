"""L3 VLM stub (Slice 4).

Slice 5 adds the real Qwen 2.5 VL client alongside this stub. The stub is the
documented fallback path for when the model is unavailable or for offline runs.
"""
from dataclasses import dataclass


@dataclass
class VLMResult:
    label: str
    rationale: str
    confidence: float
    is_stub: bool = True


def analyze_frame(frame_result, fired_rules: list) -> VLMResult:
    """Return a stub L3 analysis response.

    In Slice 5 this is replaced by a real Qwen 2.5 VL call. The stub always
    returns confidence=0.0 so downstream code can distinguish real from stub.
    """
    rule_summary = ", ".join(fired_rules) if fired_rules else "none"
    return VLMResult(
        label="unknown",
        rationale=f"Stub L3 analysis. Rules that fired: {rule_summary}.",
        confidence=0.0,
        is_stub=True,
    )
