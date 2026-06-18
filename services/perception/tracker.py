"""ByteTrack wrapper (Slice 3).

Wraps supervision.ByteTrack to assign persistent track IDs to Detection objects.

supervision.ByteTrack is deprecated since v0.28 and removed in v0.30 — requirements.txt
pins supervision<0.30 to avoid surprise removal. The deprecation warning is suppressed
since the API is stable across the pinned range.
"""
import dataclasses
import warnings

import numpy as np
import supervision as sv

from services.perception.detection import Detection


class ByteTracker:
    """Wraps sv.ByteTrack, translating to/from Detection objects.

    minimum_consecutive_frames=1 prioritises recall of track births.
    Production tuning may want 2-3 to reduce spurious tracks.
    """

    def __init__(self, lost_track_buffer: int = 30, frame_rate: int = 30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress FutureWarning + DeprecationWarning from sv.ByteTrack
            self._tracker = sv.ByteTrack(
                minimum_consecutive_frames=1,
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
            )

    def update(self, detections: list) -> list:
        """Assign/update track IDs. Returns Detection objects with track_id filled in.

        Only detections confirmed by the tracker are returned. With
        minimum_consecutive_frames=1 that is every input detection on the first frame
        they appear. Order may differ from input; use detection.bbox to reidentify.
        """
        if not detections:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress FutureWarning + DeprecationWarning from sv.ByteTrack
                self._tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=float)
        confidence = np.array([d.confidence for d in detections], dtype=float)
        class_ids = np.array([d.class_id for d in detections], dtype=int)
        orig_idx = np.arange(len(detections))

        sv_det = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
            data={"orig_idx": orig_idx},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress FutureWarning + DeprecationWarning from sv.ByteTrack
            tracked = self._tracker.update_with_detections(sv_det)

        result = []
        for i in range(len(tracked)):
            orig_i = int(tracked.data["orig_idx"][i])
            result.append(
                dataclasses.replace(detections[orig_i], track_id=int(tracked.tracker_id[i]))
            )
        return result
