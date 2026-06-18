"""Policy engine for the agent layer (Slice 7).

Loads YAML policy files from config/policies/<facility_id>.yaml and evaluates
rules against the current incident context.

Supported rule types:
  time_window_suppression — suppress alert for a label+room_tag combo between
                            two HH:MM times (e.g. gym falls during PT hours)
  threshold_override      — lower (or raise) confidence threshold for rooms
                            tagged with a specific tag
  severity_filter         — suppress incidents below a minimum severity for
                            rooms tagged with a specific tag

PolicyResult fields:
  suppress             — True if any rule suppresses the incident
  threshold_override   — effective threshold (first rule wins; None = use default)
  severity_override    — override severity string (not used in Slice 7; reserved)
  notes                — human-readable list of which rules fired
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import yaml


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class PolicyResult:
    suppress: bool = False
    threshold_override: Optional[float] = None
    severity_override: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self, policy_dir: str = "config/policies"):
        self._dir = policy_dir
        self._cache: Dict[str, dict] = {}

    def _load(self, facility_id: str) -> dict:
        if facility_id not in self._cache:
            path = os.path.join(self._dir, f"{facility_id}.yaml")
            if os.path.exists(path):
                with open(path) as f:
                    self._cache[facility_id] = yaml.safe_load(f) or {}
            else:
                self._cache[facility_id] = {}
        return self._cache[facility_id]

    def fusion_weights(self, facility_id: str) -> Dict[str, float]:
        """Return fusion weights from policy YAML, or module defaults."""
        from services.agent.fusion import DEFAULT_WEIGHTS
        return self._load(facility_id).get("fusion_weights", dict(DEFAULT_WEIGHTS))

    def evaluate(
        self,
        facility_id: str,
        room_tags: List[str],
        label: str,
        severity: str,
        now: Optional[datetime] = None,
    ) -> PolicyResult:
        """Apply all rules in the policy file and return a PolicyResult."""
        policy = self._load(facility_id)
        now = now or datetime.now()
        result = PolicyResult()

        for rule in policy.get("rules", []):
            rtype = rule.get("type")
            if rtype == "time_window_suppression":
                self._apply_time_window(rule, room_tags, label, now, result)
            elif rtype == "threshold_override":
                self._apply_threshold(rule, room_tags, result)
            elif rtype == "severity_filter":
                self._apply_severity_filter(rule, room_tags, severity, result)

        return result

    # ------------------------------------------------------------------
    # Rule evaluators
    # ------------------------------------------------------------------

    def _tags_match(self, rule_tags: List[str], room_tags: List[str]) -> bool:
        return bool(set(rule_tags) & set(room_tags))

    def _apply_time_window(
        self,
        rule: dict,
        room_tags: List[str],
        label: str,
        now: datetime,
        result: PolicyResult,
    ) -> None:
        match = rule.get("match", {})
        if not self._tags_match(match.get("room_tags", []), room_tags):
            return
        if label not in match.get("incident_labels", []):
            return
        start_str, end_str = rule["suppress_between"]
        start_h, start_m = (int(x) for x in start_str.split(":"))
        end_h, end_m = (int(x) for x in end_str.split(":"))
        now_minutes = now.hour * 60 + now.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= now_minutes < end_minutes:
            result.suppress = True
            result.notes.append(f"rule={rule['name']} suppressed at {now.strftime('%H:%M')}")

    def _apply_threshold(
        self,
        rule: dict,
        room_tags: List[str],
        result: PolicyResult,
    ) -> None:
        match = rule.get("match", {})
        if not self._tags_match(match.get("room_tags", []), room_tags):
            return
        override = float(rule["threshold"])
        if result.threshold_override is None:
            result.threshold_override = override
            result.notes.append(f"rule={rule['name']} threshold={override}")

    def _apply_severity_filter(
        self,
        rule: dict,
        room_tags: List[str],
        severity: str,
        result: PolicyResult,
    ) -> None:
        match = rule.get("match", {})
        if not self._tags_match(match.get("room_tags", []), room_tags):
            return
        min_sev = rule.get("min_severity", "low")
        if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK.get(min_sev, 0):
            result.suppress = True
            result.notes.append(
                f"rule={rule['name']} severity={severity} below min={min_sev}"
            )
