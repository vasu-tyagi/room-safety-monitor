"""Tests for the L2 aggregator and fall-persistence tracker."""
import numpy as np

from services.perception.detection import Detection
from services.perception.l2 import (
    FallPersistenceTracker,
    L2Perception,
    PersonResult,
)
from services.perception.pose import PersonPose

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _lying_pose():
    kp = np.zeros((17, 2), dtype=float)
    kp[L_SHOULDER], kp[R_SHOULDER] = (100, 150), (100, 170)
    kp[L_HIP], kp[R_HIP] = (220, 150), (220, 170)
    return PersonPose(keypoints=kp, scores=np.ones(17))


def _standing_pose():
    kp = np.zeros((17, 2), dtype=float)
    kp[L_SHOULDER], kp[R_SHOULDER] = (140, 100), (160, 100)
    kp[L_HIP], kp[R_HIP] = (142, 210), (158, 210)
    return PersonPose(keypoints=kp, scores=np.ones(17))


class FakeDetector:
    def __init__(self, n_persons):
        self.n_persons = n_persons

    def persons(self, frame, conf=None):
        return [
            Detection((0, 0, 50, 80), 0.9, 0, "person", "person")
            for _ in range(self.n_persons)
        ]


class FakePose:
    def __init__(self, pose):
        self.pose = pose

    def estimate(self, frame, bboxes):
        return [self.pose for _ in bboxes]


# ---- FallPersistenceTracker ----

def test_tracker_confirms_only_after_persist_frames():
    t = FallPersistenceTracker(persist=3)
    assert [t.update(True) for _ in range(3)] == [False, False, True]


def test_tracker_does_not_reconfirm_while_still_down():
    t = FallPersistenceTracker(persist=3)
    [t.update(True) for _ in range(3)]  # confirmed on 3rd
    assert t.update(True) is False  # still down, no duplicate


def test_tracker_resets_on_non_fall_and_can_reconfirm():
    t = FallPersistenceTracker(persist=2)
    assert t.update(True) is False
    assert t.update(True) is True
    assert t.update(False) is False  # reset
    assert t.update(True) is False
    assert t.update(True) is True


def test_single_dip_does_not_confirm():
    t = FallPersistenceTracker(persist=3)
    assert [t.update(x) for x in (True, True, False, True)] == [
        False, False, False, False
    ]


# ---- L2Perception ----

def test_process_frame_flags_fall_for_lying_person():
    l2 = L2Perception(FakeDetector(1), FakePose(_lying_pose()))
    res = l2.process_frame(frame=None)
    assert res.any_fall is True
    assert isinstance(res.persons[0], PersonResult)
    assert res.persons[0].fall.is_fall is True


def test_process_frame_no_fall_for_standing_person():
    l2 = L2Perception(FakeDetector(1), FakePose(_standing_pose()))
    res = l2.process_frame(frame=None)
    assert res.any_fall is False


def test_process_frame_no_persons():
    l2 = L2Perception(FakeDetector(0), FakePose(_lying_pose()))
    res = l2.process_frame(frame=None)
    assert res.persons == []
    assert res.any_fall is False
