"""Event Gate rules engine (Slice 4).

RulesEngine runs all seven rules against an L2FrameResult and returns the list
of rule names that fired. Any non-empty list means the frame is escalated to L3.

State dicts (proximity, velocity, inactivity) accumulate across frames for a
single video run. Call reset() between independent runs.
"""
from services.event_gate.rules import (
    action_in_target_set,
    fall_pose_detected,
    person_count_exceeds_policy,
    prolonged_inactivity_in_private_zone,
    rapid_motion_in_restricted_zone,
    two_persons_close_proximity_sustained,
    unattended_minor_in_high_risk_zone,
)

TARGET_ACTIONS = {"falling", "fighting", "running"}

# Default thresholds — overridden by values in room_policy if present.
_DEFAULT_PROXIMITY_DIST_PX    = 100.0
_DEFAULT_PROXIMITY_DUR_SEC    = 3.0
_DEFAULT_VELOCITY_THRESHOLD   = 50.0
_DEFAULT_INACTIVITY_THRESHOLD = 300.0  # 5 minutes
_DEFAULT_MINOR_AREA_PX        = 5000.0
_DEFAULT_ADULT_AREA_PX        = 15000.0


class RulesEngine:
    """Evaluate all Event Gate rules for one frame.

    Args:
        room_policy: dict loaded from config/rooms.yaml for this room.
            Keys: max_occupancy, proximity_distance_px, proximity_duration_sec,
            velocity_threshold_px, inactivity_threshold_sec, minor_area_px,
            adult_area_px, zones (list of zone dicts with name/tag/polygon).
        fps: frames per second of the video stream — used to convert
            duration thresholds from seconds to frames.
        still_threshold_px: average keypoint displacement (pixels) below which
            a person is counted as still for the inactivity rule.
    """

    def __init__(
        self,
        room_policy: dict,
        fps: float = 30,
        still_threshold_px: float = 5.0,
    ):
        self.room_policy = room_policy
        self.fps = fps
        self.still_threshold_px = still_threshold_px
        self._proximity_state: dict = {}
        self._velocity_state: dict = {}
        self._inactivity_state: dict = {}

    def reset(self):
        """Clear all accumulated state.

        Call between independent video runs. Also needed by Slice 7.5 Incident
        Replay, which must start from empty state to replay a clip cleanly.
        """
        self._proximity_state.clear()
        self._velocity_state.clear()
        self._inactivity_state.clear()

    def evaluate(self, frame_result, action_label: str = None) -> list:
        """Run all rules against the current L2FrameResult.

        Returns a list of rule name strings for rules that fired. Empty list
        means the frame is filtered (not escalated to L3).

        Args:
            frame_result: L2FrameResult from the current frame.
            action_label: target_action string from the most recent SlowFast
                inference ("falling", "fighting", "running", "other", or None).
        """
        persons = frame_result.persons
        detections = [p.detection for p in persons]
        zones = self.room_policy.get("zones", [])
        fired = []

        # Rule 1: fall_pose_detected
        if fall_pose_detected(persons):
            fired.append("fall_pose_detected")

        # Rule 2: action_in_target_set
        if action_label is not None and action_in_target_set(action_label, TARGET_ACTIONS):
            fired.append("action_in_target_set")

        # Rule 3: two_persons_close_proximity_sustained
        if len(detections) >= 2:
            if two_persons_close_proximity_sustained(
                detections,
                self._proximity_state,
                self.room_policy.get("proximity_distance_px", _DEFAULT_PROXIMITY_DIST_PX),
                self.room_policy.get("proximity_duration_sec", _DEFAULT_PROXIMITY_DUR_SEC),
                fps=self.fps,
            ):
                fired.append("two_persons_close_proximity_sustained")

        # Rule 4: person_count_exceeds_policy
        if "max_occupancy" in self.room_policy and person_count_exceeds_policy(
            detections, self.room_policy
        ):
            fired.append("person_count_exceeds_policy")

        # Rule 5: rapid_motion_in_restricted_zone (only if restricted zones exist)
        if any(z.get("tag") == "restricted" for z in zones):
            if rapid_motion_in_restricted_zone(
                detections,
                self._velocity_state,
                zones,
                self.room_policy.get("velocity_threshold_px", _DEFAULT_VELOCITY_THRESHOLD),
            ):
                fired.append("rapid_motion_in_restricted_zone")
        else:
            # Still update velocity state so positions are current when zones are added.
            rapid_motion_in_restricted_zone(
                detections, self._velocity_state, [], float("inf")
            )

        # Rule 6: prolonged_inactivity_in_private_zone (only if private zones exist)
        if any(z.get("tag") == "private" for z in zones):
            if prolonged_inactivity_in_private_zone(
                persons,
                self._inactivity_state,
                zones,
                self.room_policy.get("inactivity_threshold_sec", _DEFAULT_INACTIVITY_THRESHOLD),
                fps=self.fps,
                still_threshold_px=self.still_threshold_px,
            ):
                fired.append("prolonged_inactivity_in_private_zone")

        # Rule 7: unattended_minor_in_high_risk_zone (only if high_risk zones exist)
        if any(z.get("tag") == "high_risk" for z in zones):
            if unattended_minor_in_high_risk_zone(
                detections,
                zones,
                self.room_policy.get("minor_area_px", _DEFAULT_MINOR_AREA_PX),
                self.room_policy.get("adult_area_px", _DEFAULT_ADULT_AREA_PX),
            ):
                fired.append("unattended_minor_in_high_risk_zone")

        return fired
