"""L2 aggregator: run detection -> pose -> fall assessment over a frame.

Combines the detection and pose wrappers and the pose-geometry fall rule into
one structured per-frame result. Action recognition runs at clip level in the
pipeline, not per frame, so it is not invoked here.

FallPersistenceTracker carries the "stayed down for N frames" persistence that
the v0.5 aspect-ratio rule also used, so a single-frame dip does not raise an
incident.
"""
from dataclasses import dataclass

from services.perception.detection import Detection
from services.perception.fall import FallAssessment, assess_fall
from services.perception.pose import PersonPose

# Consecutive sampled frames a person must look fallen before we confirm.
DEFAULT_PERSIST_FRAMES = 5


@dataclass
class PersonResult:
    detection: Detection
    pose: PersonPose
    fall: FallAssessment


@dataclass
class L2FrameResult:
    persons: list
    any_fall: bool


class FallPersistenceTracker:
    """Confirms a fall only after `persist` consecutive fall frames.

    update() returns True exactly once, on the frame that first reaches the
    threshold. It stays False while the person remains down (no duplicate
    incidents) and resets when a non-fall frame arrives.
    """

    def __init__(self, persist=DEFAULT_PERSIST_FRAMES):
        self.persist = persist
        self._count = 0
        self._confirmed = False

    def update(self, is_fall: bool) -> bool:
        if not is_fall:
            self._count = 0
            self._confirmed = False
            return False
        self._count += 1
        if self._count >= self.persist and not self._confirmed:
            self._confirmed = True
            return True
        return False


class L2Perception:
    def __init__(self, detector, pose_estimator, conf_thr=0.2):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self._conf_thr = conf_thr

    def process_frame(self, frame) -> L2FrameResult:
        persons = self.detector.persons(frame)
        bboxes = [p.bbox for p in persons]
        poses = self.pose_estimator.estimate(frame, bboxes)
        results = []
        any_fall = False
        for det, pose in zip(persons, poses):
            fall = assess_fall(pose.keypoints, pose.scores, conf_thr=self._conf_thr)
            any_fall = any_fall or fall.is_fall
            results.append(PersonResult(detection=det, pose=pose, fall=fall))
        return L2FrameResult(persons=results, any_fall=any_fall)
