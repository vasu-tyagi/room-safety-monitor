"""Tests for the RTMPose wrapper (L2), using an injected fake estimator.

The real estimator is rtmlib.RTMPose (ONNX, CPU); here a fake stands in so the
test needs no model download.
"""
import numpy as np

from services.perception.pose import PersonPose, PoseEstimator


class FakeRTMPose:
    """Mimics rtmlib.RTMPose.__call__ -> (keypoints (N,17,2), scores (N,17))."""

    def __call__(self, img, bboxes=None):
        n = len(bboxes) if bboxes is not None else 1
        keypoints = np.zeros((n, 17, 2), dtype=float)
        scores = np.full((n, 17), 0.9, dtype=float)
        return keypoints, scores


def test_estimate_returns_one_pose_per_bbox():
    est = PoseEstimator(model=FakeRTMPose())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    poses = est.estimate(frame, bboxes=[(0, 0, 50, 80), (10, 10, 40, 70)])
    assert len(poses) == 2
    assert all(isinstance(p, PersonPose) for p in poses)
    assert poses[0].keypoints.shape == (17, 2)
    assert poses[0].scores.shape == (17,)


def test_estimate_empty_bboxes_returns_empty():
    est = PoseEstimator(model=FakeRTMPose())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert est.estimate(frame, bboxes=[]) == []
