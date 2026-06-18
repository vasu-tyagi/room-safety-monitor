"""Object detection wrapper (L2): YOLOv8 person/object/vehicle.

Reuses the v0.5 YOLOv8n detector. The model is injectable so tests run without
weights. Detections are grouped into person / vehicle / object for downstream
rules (the Event Gate in Slice 4).
"""
from dataclasses import dataclass

# Standard COCO-80 class names, in class-id order (YOLOv8 default).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

PERSON_ID = 0
VEHICLE_IDS = {1, 2, 3, 4, 5, 6, 7, 8}  # bicycle..boat


def group_for_class(class_id: int) -> str:
    if class_id == PERSON_ID:
        return "person"
    if class_id in VEHICLE_IDS:
        return "vehicle"
    return "object"


def class_name_for(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return str(class_id)


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    group: str  # person / vehicle / object


class Detector:
    """YOLOv8 wrapper. Pass model=... in tests; otherwise yolov8n.pt loads lazily."""

    def __init__(self, model=None, conf=0.4):
        self._model = model
        self.conf = conf

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO("yolov8n.pt")
        return self._model

    def detect(self, frame, classes=None, conf=None) -> list:
        conf = self.conf if conf is None else conf
        model = self._get_model()
        result = model(frame, classes=classes, conf=conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=class_name_for(class_id),
                    group=group_for_class(class_id),
                )
            )
        return detections

    def persons(self, frame, conf=None) -> list:
        # Post-filter by group so the result is correct regardless of whether
        # the underlying model honors the classes= hint.
        return [d for d in self.detect(frame, conf=conf) if d.group == "person"]
