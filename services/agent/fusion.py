"""Confidence fusion for the agent layer (Slice 7).

Combines per-source confidence scores into a single fused score using a
weighted average. Weights live in config/policies/<facility_id>.yaml under the
fusion_weights key so different facilities can tune them without code changes.

Default weights (also in config/policies/default.yaml):
  yolo=0.1  — detection confidence; present on every frame but coarse
  pose=0.2  — fall geometry; directly measures torso angle
  action=0.2 — SlowFast action label; confirms dynamic movement
  vlm=0.4   — VLM classification; highest discriminative value
  kb=0.1    — prior KB similarity; weak signal from historical context

Missing sources (key absent from scores dict) had their process interrupted
before producing a result (e.g. SlowFast not loaded). Their weight is
redistributed proportionally to the sources that did run.

Present sources with score=0.0 had their process run but returned zero
confidence (e.g. action model ran, found no confident label). Weight is kept
and they contribute 0 to the numerator — this is a meaningful signal.
"""
from typing import Dict, Optional

DEFAULT_WEIGHTS: Dict[str, float] = {
    "yolo": 0.1,
    "pose": 0.2,
    "action": 0.2,
    "vlm": 0.4,
    "kb": 0.1,
}


def fuse(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted confidence fusion with proportional weight redistribution.

    Args:
        scores: map of source name -> confidence in [0, 1]. Sources absent
                from this dict are treated as "did not run" and their weight
                is redistributed. Sources present with 0.0 keep their weight.
        weights: per-source weights summing to 1.0. Defaults to DEFAULT_WEIGHTS.

    Returns:
        Fused confidence in [0.0, 1.0].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    present = {k for k in weights if k in scores}
    absent = {k for k in weights if k not in scores}

    if not present:
        return 0.0

    present_weight = sum(weights[k] for k in present)
    if present_weight <= 0.0:
        return 0.0

    absent_weight = sum(weights[k] for k in absent)
    # Scale each present-source weight up so they still sum to 1.0.
    redistribution_factor = (present_weight + absent_weight) / present_weight

    total = sum(weights[k] * redistribution_factor * scores[k] for k in present)
    return min(1.0, max(0.0, total))
