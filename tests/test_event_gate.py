"""Tests for the Event Gate rules and engine (Slice 4).

All tests use synthetic Detection/PersonResult data — no model weights needed.
Each rule has at least one positive case (fires) and one negative case (doesn't fire).
"""
import numpy as np
import pytest

from services.event_gate.engine import RulesEngine
from services.event_gate.rules import (
    action_in_target_set,
    fall_pose_detected,
    person_count_exceeds_policy,
    prolonged_inactivity_in_private_zone,
    rapid_motion_in_restricted_zone,
    two_persons_close_proximity_sustained,
    unattended_minor_in_high_risk_zone,
)
from services.perception.detection import Detection
from services.perception.fall import FallAssessment
from services.perception.l2 import L2FrameResult, PersonResult
from services.perception.pose import PersonPose

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _det(bbox, track_id=1):
    return Detection(
        bbox=tuple(float(x) for x in bbox),
        confidence=0.9,
        class_id=0,
        class_name="person",
        group="person",
        track_id=track_id,
    )


def _kp_standing():
    kp = np.zeros((17, 2), dtype=float)
    kp[L_SHOULDER] = (140, 100)
    kp[R_SHOULDER] = (160, 100)
    kp[L_HIP] = (142, 210)
    kp[R_HIP] = (158, 210)
    return kp


def _person(det, is_fall=False, keypoints=None):
    if keypoints is None:
        keypoints = _kp_standing()
    pose = PersonPose(keypoints=keypoints, scores=np.ones(17))
    fall = FallAssessment(
        is_fall=is_fall,
        confidence=0.9 if is_fall else 0.0,
        torso_angle_deg=80.0 if is_fall else 5.0,
    )
    return PersonResult(detection=det, pose=pose, fall=fall, track_id=det.track_id)


def _frame(persons):
    return L2FrameResult(persons=persons, any_fall=any(p.fall.is_fall for p in persons))


# Zone fixtures
_FULL_PRIVATE = {"tag": "private",     "polygon": [[0, 0], [640, 0], [640, 480], [0, 480]]}
_SMALL_PRIVATE = {"tag": "private",    "polygon": [[0, 0], [50, 0],  [50, 50],   [0, 50]]}
_HIGH_RISK     = {"tag": "high_risk",  "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]]}
_RESTRICTED    = {"tag": "restricted", "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]]}


# ---------------------------------------------------------------------------
# fall_pose_detected
# ---------------------------------------------------------------------------

def test_fall_pose_detected_fires_when_person_is_falling():
    assert fall_pose_detected([_person(_det((0, 0, 100, 200)), is_fall=True)]) is True


def test_fall_pose_detected_silent_when_no_fall():
    assert fall_pose_detected([_person(_det((0, 0, 100, 200)), is_fall=False)]) is False


def test_fall_pose_detected_silent_for_empty_frame():
    assert fall_pose_detected([]) is False


# ---------------------------------------------------------------------------
# action_in_target_set
# ---------------------------------------------------------------------------

def test_action_in_target_set_fires_for_target_actions():
    ts = {"falling", "fighting", "running"}
    assert action_in_target_set("falling",  ts) is True
    assert action_in_target_set("fighting", ts) is True
    assert action_in_target_set("running",  ts) is True


def test_action_in_target_set_silent_for_non_target():
    ts = {"falling", "fighting", "running"}
    assert action_in_target_set("walking", ts) is False
    assert action_in_target_set("other",   ts) is False


# ---------------------------------------------------------------------------
# two_persons_close_proximity_sustained
# ---------------------------------------------------------------------------
# det_a center=(50,50), det_b center=(100,50) — distance=50, threshold=60 → close

def test_proximity_fires_after_duration_met():
    det_a = _det((0, 0, 100, 100), track_id=1)
    det_b = _det((50, 0, 150, 100), track_id=2)
    state = {}
    r1 = two_persons_close_proximity_sustained([det_a, det_b], state, 60, 2.0, fps=1)
    assert r1 is False  # 1 frame — threshold is 2
    r2 = two_persons_close_proximity_sustained([det_a, det_b], state, 60, 2.0, fps=1)
    assert r2 is True   # 2 frames — threshold reached


def test_proximity_silent_when_duration_not_yet_met():
    det_a = _det((0, 0, 100, 100), track_id=1)
    det_b = _det((50, 0, 150, 100), track_id=2)
    state = {}
    result = two_persons_close_proximity_sustained([det_a, det_b], state, 60, 5.0, fps=1)
    assert result is False  # 1 frame, need 5


def test_proximity_silent_when_persons_far_apart():
    # Centers: (50,50) and (350,50) — distance=300 > threshold=60
    det_a = _det((0, 0, 100, 100), track_id=1)
    det_b = _det((300, 0, 400, 100), track_id=2)
    state = {}
    for _ in range(10):
        result = two_persons_close_proximity_sustained([det_a, det_b], state, 60, 1.0, fps=1)
    assert result is False


def test_proximity_state_clears_when_persons_separate():
    det_a      = _det((0, 0, 100, 100), track_id=1)
    det_b_near = _det((50, 0, 150, 100), track_id=2)
    det_b_far  = _det((300, 0, 400, 100), track_id=2)
    state = {}
    two_persons_close_proximity_sustained([det_a, det_b_near], state, 60, 2.0, fps=1)
    assert len(state) == 1
    two_persons_close_proximity_sustained([det_a, det_b_far], state, 60, 2.0, fps=1)
    assert len(state) == 0  # pair cleared because they're now far apart


# ---------------------------------------------------------------------------
# person_count_exceeds_policy
# ---------------------------------------------------------------------------

def test_count_fires_when_over_limit():
    dets = [_det((i * 10, 0, i * 10 + 10, 20), track_id=i) for i in range(3)]
    assert person_count_exceeds_policy(dets, {"max_occupancy": 2}) is True


def test_count_silent_when_at_limit():
    dets = [_det((i * 10, 0, i * 10 + 10, 20), track_id=i) for i in range(2)]
    assert person_count_exceeds_policy(dets, {"max_occupancy": 2}) is False


def test_count_silent_when_under_limit():
    dets = [_det((0, 0, 100, 200), track_id=1)]
    assert person_count_exceeds_policy(dets, {"max_occupancy": 2}) is False


# ---------------------------------------------------------------------------
# rapid_motion_in_restricted_zone
# ---------------------------------------------------------------------------
# Restricted zone polygon: [[0,0],[200,0],[200,200],[0,200]] (square)

def test_rapid_motion_fires_when_fast_inside_zone():
    state = {}
    # Frame 1: center (100,100) — inside zone, stores position
    rapid_motion_in_restricted_zone([_det((50, 50, 150, 150), track_id=1)], state, [_RESTRICTED], 50)
    # Frame 2: center (180,100) — speed=80 > 50, inside zone → fires
    result = rapid_motion_in_restricted_zone([_det((130, 50, 230, 150), track_id=1)], state, [_RESTRICTED], 50)
    assert result is True


def test_rapid_motion_silent_when_slow_inside_zone():
    state = {}
    rapid_motion_in_restricted_zone([_det((50, 50, 150, 150), track_id=1)], state, [_RESTRICTED], 50)
    # Frame 2: center (110,100) — speed=10 < 50 → silent
    result = rapid_motion_in_restricted_zone([_det((60, 50, 160, 150), track_id=1)], state, [_RESTRICTED], 50)
    assert result is False


def test_rapid_motion_silent_when_fast_outside_zone():
    state = {}
    # Person at center (300,100) — outside zone (x=300 > 200)
    rapid_motion_in_restricted_zone([_det((250, 50, 350, 150), track_id=1)], state, [_RESTRICTED], 50)
    # Fast movement but stays outside zone
    result = rapid_motion_in_restricted_zone([_det((330, 50, 430, 150), track_id=1)], state, [_RESTRICTED], 50)
    assert result is False


# ---------------------------------------------------------------------------
# prolonged_inactivity_in_private_zone
# ---------------------------------------------------------------------------

def test_inactivity_fires_after_threshold():
    kp = _kp_standing()
    det = _det((100, 100, 200, 300), track_id=1)
    person = _person(det, is_fall=False, keypoints=kp)
    state = {}
    # fps=1, threshold=2s → need 2 consecutive still frames after init
    r1 = prolonged_inactivity_in_private_zone([person], state, [_FULL_PRIVATE], 2.0, fps=1)
    assert r1 is False  # init frame — no history yet
    r2 = prolonged_inactivity_in_private_zone([person], state, [_FULL_PRIVATE], 2.0, fps=1)
    assert r2 is False  # 1 still frame, need 2
    r3 = prolonged_inactivity_in_private_zone([person], state, [_FULL_PRIVATE], 2.0, fps=1)
    assert r3 is True   # 2 still frames — threshold reached


def test_inactivity_silent_when_person_moving():
    state = {}
    det = _det((100, 100, 200, 300), track_id=1)
    for i in range(10):
        kp = _kp_standing() + float(i) * 20  # 20px shift each frame, above still_threshold=5
        person = _person(det, is_fall=False, keypoints=kp)
        result = prolonged_inactivity_in_private_zone([person], state, [_FULL_PRIVATE], 2.0, fps=1)
    assert result is False


def test_inactivity_silent_when_person_outside_private_zone():
    kp = _kp_standing()
    # bbox center=(250,300) — outside _SMALL_PRIVATE which is [[0,0],[50,0],[50,50],[0,50]]
    det = _det((200, 250, 300, 350), track_id=1)
    person = _person(det, is_fall=False, keypoints=kp)
    state = {}
    for _ in range(10):
        result = prolonged_inactivity_in_private_zone([person], state, [_SMALL_PRIVATE], 2.0, fps=1)
    assert result is False


# ---------------------------------------------------------------------------
# unattended_minor_in_high_risk_zone
# ---------------------------------------------------------------------------
# High-risk zone polygon: [[0,0],[200,0],[200,200],[0,200]]

def test_minor_in_high_risk_fires_without_adult():
    # bbox (10,10,60,60) → area=2500 < 5000 (minor), center=(35,35) inside zone
    minor = _det((10, 10, 60, 60), track_id=1)
    result = unattended_minor_in_high_risk_zone([minor], [_HIGH_RISK], 5000, 15000)
    assert result is True


def test_minor_in_high_risk_silent_when_adult_present():
    minor = _det((10, 10, 60, 60), track_id=1)        # area=2500
    adult = _det((200, 200, 600, 600), track_id=2)    # area=160000 > 15000
    result = unattended_minor_in_high_risk_zone([minor, adult], [_HIGH_RISK], 5000, 15000)
    assert result is False


def test_minor_silent_when_outside_high_risk_zone():
    # center=(275,275) — outside zone (x=275 > 200)
    minor = _det((250, 250, 300, 300), track_id=1)
    result = unattended_minor_in_high_risk_zone([minor], [_HIGH_RISK], 5000, 15000)
    assert result is False


def test_minor_silent_when_no_high_risk_zones():
    minor = _det((10, 10, 60, 60), track_id=1)
    result = unattended_minor_in_high_risk_zone([minor], [], 5000, 15000)
    assert result is False


# ---------------------------------------------------------------------------
# RulesEngine
# ---------------------------------------------------------------------------

def test_engine_empty_frame_fires_nothing():
    engine = RulesEngine(room_policy={})
    assert engine.evaluate(_frame([])) == []


def test_engine_fires_fall_pose_detected():
    det = _det((0, 0, 100, 200))
    fired = RulesEngine(room_policy={}).evaluate(_frame([_person(det, is_fall=True)]))
    assert "fall_pose_detected" in fired


def test_engine_fires_action_in_target_set():
    fired = RulesEngine(room_policy={}).evaluate(_frame([]), action_label="fighting")
    assert "action_in_target_set" in fired


def test_engine_fires_person_count_exceeds_policy():
    dets = [_det((i * 10, 0, i * 10 + 10, 20), track_id=i) for i in range(3)]
    fired = RulesEngine(room_policy={"max_occupancy": 2}).evaluate(_frame([_person(d) for d in dets]))
    assert "person_count_exceeds_policy" in fired


def test_engine_silent_for_non_target_action():
    fired = RulesEngine(room_policy={}).evaluate(_frame([]), action_label="walking")
    assert "action_in_target_set" not in fired


def test_engine_reset_clears_all_state():
    det_a = _det((0, 0, 100, 100), track_id=1)
    det_b = _det((50, 0, 150, 100), track_id=2)
    engine = RulesEngine(room_policy={"proximity_distance_px": 60, "proximity_duration_sec": 1.0})
    engine.evaluate(_frame([_person(det_a), _person(det_b)]))
    assert len(engine._proximity_state) > 0

    engine.reset()
    assert engine._proximity_state == {}
    assert engine._velocity_state == {}
    assert engine._inactivity_state == {}


def test_engine_silent_when_no_relevant_config():
    det = _det((0, 0, 100, 200))
    persons = [_person(det, is_fall=False)]
    fired = RulesEngine(room_policy={}).evaluate(_frame(persons), action_label="walking")
    assert fired == []
