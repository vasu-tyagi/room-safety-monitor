"""Tests for the YOLOv8 detection wrapper (L2), using an injected fake model."""
import torch

from services.perception.detection import Detection, Detector, group_for_class


class FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf], dtype=torch.float32)
        self.cls = torch.tensor([float(cls)], dtype=torch.float32)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, img, classes=None, conf=0.25, verbose=False):
        return [FakeResult(self._boxes)]


def test_group_for_class():
    assert group_for_class(0) == "person"
    assert group_for_class(2) == "vehicle"  # car
    assert group_for_class(39) == "object"  # bottle


def test_detector_returns_detections_with_group_and_name():
    boxes = [FakeBox([10, 20, 30, 80], 0.9, 0), FakeBox([0, 0, 50, 40], 0.8, 2)]
    det = Detector(model=FakeYOLO(boxes))
    out = det.detect(frame=None)
    assert len(out) == 2
    assert isinstance(out[0], Detection)
    assert out[0].class_name == "person" and out[0].group == "person"
    assert out[0].bbox == (10.0, 20.0, 30.0, 80.0)
    assert out[1].class_name == "car" and out[1].group == "vehicle"


def test_persons_helper_filters_to_people():
    boxes = [FakeBox([10, 20, 30, 80], 0.9, 0), FakeBox([0, 0, 50, 40], 0.8, 2)]
    det = Detector(model=FakeYOLO(boxes))
    persons = det.persons(frame=None)
    assert len(persons) == 1
    assert persons[0].group == "person"
