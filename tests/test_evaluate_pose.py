"""Tests for the pose eval's sampling/persistence loop (sequence_has_fall).

The full dataset run needs UR Fall frames + real models; here a fake L2 drives
the loop over generated PNGs so the decision logic is covered hermetically.
"""
import cv2
import numpy as np

from evals.evaluate_pose import sequence_has_fall


class _Res:
    def __init__(self, any_fall):
        self.any_fall = any_fall


class AlwaysFall:
    def process_frame(self, frame):
        return _Res(True)


class NeverFall:
    def process_frame(self, frame):
        return _Res(False)


def _write_frames(d, n):
    for i in range(n):
        cv2.imwrite(str(d / f"f-{i:03d}.png"), np.zeros((10, 10, 3), dtype=np.uint8))


def test_sequence_has_fall_true_when_sustained(tmp_path):
    _write_frames(tmp_path, 30)
    assert sequence_has_fall(tmp_path, AlwaysFall(), persist=3, sample_every=3) is True


def test_sequence_has_fall_false_when_never(tmp_path):
    _write_frames(tmp_path, 30)
    assert sequence_has_fall(tmp_path, NeverFall(), persist=3, sample_every=3) is False
