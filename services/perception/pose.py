"""Pose estimation wrapper (L2): RTMPose via rtmlib (ONNX, CPU).

We use rtmlib instead of mmpose/mmcv: mmcv has no prebuilt wheel for this
torch+CUDA build and cannot source-build without nvcc on a CPU-only box.
rtmlib runs RTMPose through onnxruntime with no CUDA toolchain.

The model is injectable so tests need no ONNX weights download.
"""
from dataclasses import dataclass

import numpy as np

# rtmlib's balanced body pose model (RTMPose-m, body7), returns COCO-17
# keypoints. Downloaded on first use. Input size must match the model.
DEFAULT_RTMPOSE_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
DEFAULT_RTMPOSE_INPUT = (192, 256)


@dataclass
class PersonPose:
    keypoints: np.ndarray  # (17, 2) x,y in image pixels
    scores: np.ndarray  # (17,) per-keypoint confidence


class PoseEstimator:
    """Wraps an rtmlib RTMPose estimator. Returns one PersonPose per bbox."""

    def __init__(
        self,
        model=None,
        onnx_model=DEFAULT_RTMPOSE_ONNX,
        input_size=DEFAULT_RTMPOSE_INPUT,
        device="cpu",
    ):
        self._model = model
        self.onnx_model = onnx_model
        self.input_size = input_size
        self.device = device

    def _get_model(self):
        if self._model is None:
            from rtmlib import RTMPose

            self._model = RTMPose(
                onnx_model=self.onnx_model,
                model_input_size=self.input_size,
                backend="onnxruntime",
                device=self.device,
            )
        return self._model

    def estimate(self, frame, bboxes) -> list:
        if len(bboxes) == 0:
            return []
        model = self._get_model()
        boxes = np.asarray(bboxes, dtype=float)
        keypoints, scores = model(frame, bboxes=boxes)
        poses = []
        for kp, sc in zip(keypoints, scores):
            poses.append(
                PersonPose(
                    keypoints=np.asarray(kp, dtype=float),
                    scores=np.asarray(sc, dtype=float),
                )
            )
        return poses
