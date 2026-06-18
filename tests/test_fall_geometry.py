"""Tests for the pose-geometry fall rule (replaces the v0.5 aspect-ratio rule).

The rule works on COCO-17 keypoints. It reads the torso as the vector from the
shoulder midpoint to the hip midpoint and measures its angle from vertical:
a standing person's torso is near-vertical, a fallen person's is near-horizontal.
"""
import numpy as np

from services.perception.fall import FALL_ANGLE_DEG, assess_fall

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def make_kp(l_sh, r_sh, l_hip, r_hip):
    kp = np.zeros((17, 2), dtype=float)
    kp[L_SHOULDER] = l_sh
    kp[R_SHOULDER] = r_sh
    kp[L_HIP] = l_hip
    kp[R_HIP] = r_hip
    return kp


def test_standing_is_not_fall():
    # hips well below shoulders (image y grows downward) -> vertical torso
    a = assess_fall(make_kp((140, 100), (160, 100), (142, 210), (158, 210)))
    assert a.is_fall is False
    assert a.torso_angle_deg < 20


def test_lying_horizontal_is_fall():
    # shoulders and hips at the same height, far apart in x -> horizontal torso
    a = assess_fall(make_kp((100, 150), (100, 170), (220, 150), (220, 170)))
    assert a.is_fall is True
    assert a.torso_angle_deg > 70


def test_sitting_upright_is_not_fall():
    # hips below shoulders but closer; torso still mostly vertical
    a = assess_fall(make_kp((140, 120), (160, 120), (141, 180), (159, 180)))
    assert a.is_fall is False


def test_fall_confidence_higher_for_more_horizontal_torso():
    standing = assess_fall(make_kp((140, 100), (160, 100), (142, 210), (158, 210)))
    lying = assess_fall(make_kp((100, 150), (100, 170), (220, 150), (220, 170)))
    assert lying.confidence > standing.confidence


def test_low_confidence_torso_keypoints_return_undetermined():
    kp = make_kp((100, 150), (100, 170), (220, 150), (220, 170))
    scores = np.ones(17)
    scores[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] = 0.05
    a = assess_fall(kp, scores=scores, conf_thr=0.3)
    assert a.is_fall is False
    assert a.confidence == 0.0


def test_accepts_plain_python_lists():
    kp = [[0, 0] for _ in range(17)]
    kp[L_SHOULDER], kp[R_SHOULDER] = [100, 150], [100, 170]
    kp[L_HIP], kp[R_HIP] = [220, 150], [220, 170]
    a = assess_fall(kp)
    assert a.is_fall is True


def test_threshold_constant_is_between_vertical_and_horizontal():
    assert 0 < FALL_ANGLE_DEG < 90
