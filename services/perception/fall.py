"""Pose-geometry fall rule (L2).

Replaces the v0.5 aspect-ratio rule. Given COCO-17 keypoints for one person,
read the torso as the vector from the shoulder midpoint to the hip midpoint
and measure its angle from vertical. A standing person's torso is near
vertical; a fallen person's is near horizontal.

This is a pure function over keypoints so it is fully unit-testable without
any model. The persistence logic ("stayed down for N frames") lives in the
pipeline, not here.
"""
from dataclasses import dataclass

import numpy as np

# COCO-17 keypoint indices used by this rule.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# Torso angle (degrees from vertical) at or above which we call it a fall.
# Vertical torso ~ 0 deg, horizontal torso ~ 90 deg.
FALL_ANGLE_DEG = 50.0


@dataclass
class FallAssessment:
    is_fall: bool
    confidence: float  # how horizontal the torso is, 0 (vertical) .. 1 (flat)
    torso_angle_deg: float  # angle of the torso from vertical


def assess_fall(keypoints, scores=None, conf_thr=0.3) -> FallAssessment:
    """Assess whether a single person's pose looks like a fall.

    Args:
        keypoints: array-like of shape (17, 2) with (x, y) pixel coordinates
            in image convention (y grows downward).
        scores: optional array-like of shape (17,) per-keypoint confidences.
            Torso keypoints below conf_thr make the assessment undetermined.
        conf_thr: minimum confidence for a torso keypoint to be trusted.

    Returns:
        FallAssessment.
    """
    kp = np.asarray(keypoints, dtype=float)

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if np.any(scores[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] < conf_thr):
            return FallAssessment(is_fall=False, confidence=0.0, torso_angle_deg=0.0)

    shoulder_mid = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
    hip_mid = (kp[L_HIP] + kp[R_HIP]) / 2.0
    dx, dy = hip_mid - shoulder_mid

    # Angle of the torso from vertical: 0 when purely vertical (dy dominates),
    # 90 when purely horizontal (dx dominates).
    torso_angle_deg = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
    confidence = torso_angle_deg / 90.0
    is_fall = torso_angle_deg >= FALL_ANGLE_DEG

    return FallAssessment(
        is_fall=is_fall, confidence=confidence, torso_angle_deg=torso_angle_deg
    )
