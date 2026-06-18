"""Tests for ByteTracker (Slice 3).

Three behavioural requirements:
  1. The same person seen across frames keeps a consistent track_id.
  2. A track survives brief occlusion (5 frames absent) and resumes its original ID.
  3. Two overlapping persons (~33% IoU, bboxes (100,100,200,300) and (150,100,250,300))
     keep distinct IDs and don't swap them over 10 frames.
"""
from services.perception.detection import Detection
from services.perception.tracker import ByteTracker


def _person(x1, y1, x2, y2, conf=0.9):
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=conf,
        class_id=0,
        class_name="person",
        group="person",
    )


def test_same_person_keeps_track_id_across_frames():
    tracker = ByteTracker()
    det = _person(100, 100, 200, 300)
    ids = []
    for _ in range(10):
        result = tracker.update([det])
        if result:
            ids.append(result[0].track_id)
    assert len(ids) == 10, f"Expected 10 tracked frames, got {len(ids)}"
    assert len(set(ids)) == 1, f"ID changed across frames: {ids}"
    assert ids[0] != -1, "track_id was never assigned"


def test_track_survives_5_frame_occlusion():
    """Person disappears for 5 frames then reappears — same ID returned."""
    tracker = ByteTracker(lost_track_buffer=30)
    det = _person(100, 100, 200, 300)

    id_before = None
    for _ in range(5):
        result = tracker.update([det])
        if result:
            id_before = result[0].track_id

    for _ in range(5):
        tracker.update([])

    id_after = None
    for _ in range(5):
        result = tracker.update([det])
        if result:
            id_after = result[0].track_id

    assert id_before is not None, "No track assigned before occlusion"
    assert id_after is not None, "No track assigned after reappearance"
    assert id_before == id_after, f"ID changed across occlusion: {id_before} -> {id_after}"


def test_overlapping_persons_keep_distinct_ids_across_frames():
    """Two persons with ~33% IoU overlap don't swap IDs over 10 frames.

    Person A: (100, 100, 200, 300), Person B: (150, 100, 250, 300).
    Intersection area = 50*200 = 10 000. Union = 30 000. IoU = 33.3%.
    """
    tracker = ByteTracker()
    det_a = _person(100, 100, 200, 300)
    det_b = _person(150, 100, 250, 300)

    id_a = id_b = None
    for _ in range(10):
        result = tracker.update([det_a, det_b])
        if len(result) < 2:
            continue
        sorted_result = sorted(result, key=lambda d: d.bbox[0])
        cur_a = sorted_result[0].track_id
        cur_b = sorted_result[1].track_id
        if id_a is None:
            id_a, id_b = cur_a, cur_b
        else:
            assert cur_a == id_a, f"Person A swapped ID: {id_a} -> {cur_a}"
            assert cur_b == id_b, f"Person B swapped ID: {id_b} -> {cur_b}"

    assert id_a is not None and id_b is not None, "Neither person was tracked"
    assert id_a != id_b, f"Both persons share ID {id_a}"
