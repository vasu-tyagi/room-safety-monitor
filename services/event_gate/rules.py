"""Event Gate rule functions (Slice 4).

Each function is a pure, stateless rule except where the docstring says it
mutates a state dict. State dicts are owned by RulesEngine and passed in so
rules remain unit-testable without an engine instance.

Point-in-polygon uses the ray-casting algorithm (no external geometry library).
"""
import math

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test. polygon is a list of [x, y] pairs."""
    px, py = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Rule 1
# ---------------------------------------------------------------------------

def fall_pose_detected(persons: list) -> bool:
    """True if any person in the current frame has fall.is_fall == True.

    Reads the FallAssessment already computed by L2 — does not re-run geometry.
    """
    return any(p.fall.is_fall for p in persons)


# ---------------------------------------------------------------------------
# Rule 2
# ---------------------------------------------------------------------------

def action_in_target_set(action_label: str, target_set: set) -> bool:
    """True if action_label is a member of target_set."""
    return action_label in target_set


# ---------------------------------------------------------------------------
# Rule 3
# ---------------------------------------------------------------------------

def two_persons_close_proximity_sustained(
    detections: list,
    proximity_state: dict,
    distance_threshold: float,
    duration_threshold_seconds: float,
    fps: float = 30,
) -> bool:
    """True if any two tracked persons have been within distance_threshold for at
    least duration_threshold_seconds consecutive frames.

    proximity_state: mutated in place. Maps frozenset({tid_a, tid_b}) to the
        number of consecutive frames that pair has been within distance_threshold.
        Pairs that separate are removed so the counter resets cleanly.
    distance_threshold: center-to-center pixel distance.
    duration_threshold_seconds: converted to frames internally using fps.
    """
    duration_frames = duration_threshold_seconds * fps
    current_close_pairs = set()

    for i, a in enumerate(detections):
        for j, b in enumerate(detections):
            if j <= i:
                continue
            pair = frozenset({a.track_id, b.track_id})
            if _dist(_bbox_center(a.bbox), _bbox_center(b.bbox)) <= distance_threshold:
                current_close_pairs.add(pair)

    # Increment counters for close pairs, remove pairs that are now far apart.
    for pair in current_close_pairs:
        proximity_state[pair] = proximity_state.get(pair, 0) + 1
    for pair in [p for p in proximity_state if p not in current_close_pairs]:
        del proximity_state[pair]

    return any(count >= duration_frames for count in proximity_state.values())


# ---------------------------------------------------------------------------
# Rule 4
# ---------------------------------------------------------------------------

def person_count_exceeds_policy(detections: list, room_policy: dict) -> bool:
    """True if the number of detected persons exceeds room max_occupancy."""
    return len(detections) > room_policy.get("max_occupancy", float("inf"))


# ---------------------------------------------------------------------------
# Rule 5
# ---------------------------------------------------------------------------

def rapid_motion_in_restricted_zone(
    detections: list,
    velocity_state: dict,
    zones: list,
    velocity_threshold: float,
) -> bool:
    """True if any tracked person is inside a restricted zone AND has center
    displacement (pixels/frame) above velocity_threshold.

    velocity_state: mutated in place. Maps track_id to previous bbox center.
        On the first frame a track is seen, position is stored but no velocity
        check is performed (no previous position to compare against).
    zones: list of zone dicts from room config; only 'restricted'-tagged zones
        are evaluated.
    velocity_threshold: pixels per frame.
    """
    restricted_polygons = [z["polygon"] for z in zones if z.get("tag") == "restricted"]
    fired = False

    for det in detections:
        tid = det.track_id
        center = _bbox_center(det.bbox)

        if tid in velocity_state:
            speed = _dist(center, velocity_state[tid])
            if speed >= velocity_threshold:
                for poly in restricted_polygons:
                    if _point_in_polygon(center, poly):
                        fired = True
                        break

        velocity_state[tid] = center

    return fired


# ---------------------------------------------------------------------------
# Rule 6
# ---------------------------------------------------------------------------

def prolonged_inactivity_in_private_zone(
    persons: list,
    inactivity_state: dict,
    zones: list,
    inactivity_threshold_seconds: float,
    fps: float = 30,
    still_threshold_px: float = 5.0,
) -> bool:
    """True if any tracked person in a private zone has had minimal joint movement
    for at least inactivity_threshold_seconds consecutive frames.

    inactivity_state: mutated in place. Maps track_id to
        {"frames": int, "prev_kp": ndarray}. Entries for tracks no longer in
        the frame or outside private zones are cleaned up.
    zones: list of zone dicts; only 'private'-tagged zones are evaluated.
    still_threshold_px: average keypoint displacement (pixels) below which a
        person is considered still. Configurable at RulesEngine init.
    """
    inactivity_frames = inactivity_threshold_seconds * fps
    private_polygons = [z["polygon"] for z in zones if z.get("tag") == "private"]
    fired = False

    current_tids = {p.track_id for p in persons}
    for tid in [t for t in inactivity_state if t not in current_tids]:
        del inactivity_state[tid]

    for person in persons:
        tid = person.track_id
        center = _bbox_center(person.detection.bbox)
        in_private = any(_point_in_polygon(center, poly) for poly in private_polygons)

        if not in_private:
            inactivity_state.pop(tid, None)
            continue

        kp = person.pose.keypoints  # (17, 2)

        if tid not in inactivity_state:
            inactivity_state[tid] = {"frames": 0, "prev_kp": kp.copy()}
            continue

        avg_disp = float(np.mean(np.linalg.norm(kp - inactivity_state[tid]["prev_kp"], axis=1)))

        if avg_disp < still_threshold_px:
            inactivity_state[tid]["frames"] += 1
        else:
            inactivity_state[tid]["frames"] = 0

        inactivity_state[tid]["prev_kp"] = kp.copy()

        if inactivity_state[tid]["frames"] >= inactivity_frames:
            fired = True

    return fired


# ---------------------------------------------------------------------------
# Rule 7
# ---------------------------------------------------------------------------

def unattended_minor_in_high_risk_zone(
    detections: list,
    zones: list,
    minor_area_px: float = 5000,
    adult_area_px: float = 15000,
) -> bool:
    """True if a small bounding box (minor proxy) is inside a high-risk zone
    with no adult-sized bounding box anywhere in the frame.

    Slice 4 limitation: uses bbox area as a proxy for age. Real deployment
    would use an age classifier on detected faces. Pixel-area proxy is a
    crude stand-in.
    minor_area_px: bbox area below which a detection is classified as a minor.
    adult_area_px: bbox area above which a detection is classified as an adult.
    """
    high_risk_polygons = [z["polygon"] for z in zones if z.get("tag") == "high_risk"]
    if not high_risk_polygons:
        return False

    if any(_bbox_area(d.bbox) >= adult_area_px for d in detections):
        return False

    for det in detections:
        if _bbox_area(det.bbox) < minor_area_px:
            center = _bbox_center(det.bbox)
            for poly in high_risk_polygons:
                if _point_in_polygon(center, poly):
                    return True

    return False
