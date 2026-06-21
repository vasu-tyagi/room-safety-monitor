"""Tests for Slice 7 agent layer: nodes, policy engine, fusion, full graph.

Each node is tested in isolation by calling it directly with a synthetic
AgentState dict. The full graph tests exercise the end-to-end path with a
SQLite session.
"""
import os
import tempfile
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.persistence.db import Base
from services.vlm.stub import VLMResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    engine = create_engine(
        "sqlite://", future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def _vlm(label="fall", confidence=0.9, is_stub=False):
    return VLMResult(label=label, rationale="Person fell.", confidence=confidence, is_stub=is_stub)


def _base_state(**overrides):
    """Minimal valid AgentState for tests that only care about one node."""
    state = {
        "session": _make_db(),
        "kb": None,
        "camera_id": "cam0",
        "room_id": "room0",
        "facility_id": "default",
        "room_tags": [],
        "fired_rules": ["fall_pose_detected"],
        "frame_idx": 42,
        "detection_notes": "Torso angle 80 deg from vertical.",
        "yolo_confidence": 0.9,
        "pose_confidence": 0.85,
        "action_confidence": None,
        "vlm_result": _vlm(),
        "vlm_prompt": "sample prompt",
        # Filled by nodes (pre-populated with defaults for nodes under test)
        "label": "fall",
        "severity": "high",
        "rationale": "Person fell.",
        "policy_suppress": False,
        "policy_threshold": 0.7,
        "policy_severity": None,
        "policy_notes": [],
        "fusion_weights": {"yolo": 0.1, "pose": 0.2, "action": 0.2, "vlm": 0.4, "kb": 0.1},
        "fused_confidence": 0.85,
        "confidence_breakdown": {"yolo": 0.9, "pose": 0.85, "vlm": 0.0},
        "kb_matches": [],
        "decision": "alert",
        "incident_state": "alert",
        "decide_reason": "test",
        # Slice 7.5: clip storage defaults
        "clip_frames": [],
        "clips_dir": "clips",
        "dry_run": False,
        # Slice 9: per-frame perception data for overlay rendering
        "frame_perception_data": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# parse_vlm_output node tests
# ---------------------------------------------------------------------------

def test_parse_vlm_output_fall_maps_to_high_severity():
    from services.agent.graph import parse_vlm_output
    state = _base_state(vlm_result=_vlm(label="fall"))
    result = parse_vlm_output(state)
    assert result["severity"] == "high"
    assert result["label"] == "fall"


def test_parse_vlm_output_fight_maps_to_high_severity():
    from services.agent.graph import parse_vlm_output
    state = _base_state(vlm_result=_vlm(label="fight"))
    result = parse_vlm_output(state)
    assert result["severity"] == "high"


def test_parse_vlm_output_inactivity_maps_to_low_severity():
    from services.agent.graph import parse_vlm_output
    state = _base_state(vlm_result=_vlm(label="inactivity"))
    result = parse_vlm_output(state)
    assert result["severity"] == "low"


def test_parse_vlm_output_overcrowding_maps_to_medium_severity():
    from services.agent.graph import parse_vlm_output
    state = _base_state(vlm_result=_vlm(label="overcrowding"))
    result = parse_vlm_output(state)
    assert result["severity"] == "medium"


def test_parse_vlm_output_unknown_uses_detection_notes():
    from services.agent.graph import parse_vlm_output
    state = _base_state(
        vlm_result=_vlm(label="unknown", is_stub=False),
        detection_notes="Torso angle 80 deg.",
    )
    result = parse_vlm_output(state)
    assert result["severity"] == "low"
    assert "Torso angle 80 deg." in result["rationale"]


def test_parse_vlm_output_stub_uses_detection_notes():
    from services.agent.graph import parse_vlm_output
    state = _base_state(
        vlm_result=_vlm(label="fall", is_stub=True),
        detection_notes="Pose-geometry signal only.",
    )
    result = parse_vlm_output(state)
    assert result["severity"] == "low"
    assert "Pose-geometry signal only." in result["rationale"]


# ---------------------------------------------------------------------------
# PolicyEngine tests
# ---------------------------------------------------------------------------

def _write_policy(tmp_path, policy_dict):
    p = tmp_path / "default.yaml"
    p.write_text(yaml.dump(policy_dict))
    return str(tmp_path)


def test_policy_no_rules_uses_defaults(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {"version": 1, "rules": []})
    eng = PolicyEngine(policy_dir=policy_dir)
    result = eng.evaluate("default", room_tags=[], label="fall", severity="high")
    assert result.suppress is False
    assert result.threshold_override is None


def test_policy_gym_suppresses_fall_in_window(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "rules": [{
            "name": "gym_fall",
            "type": "time_window_suppression",
            "match": {"room_tags": ["gym"], "incident_labels": ["fall"]},
            "suppress_between": ["18:00", "20:00"],
        }],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    during = datetime(2026, 6, 19, 19, 0)
    result = eng.evaluate("default", room_tags=["gym"], label="fall", severity="high", now=during)
    assert result.suppress is True


def test_policy_gym_no_suppression_outside_window(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "rules": [{
            "name": "gym_fall",
            "type": "time_window_suppression",
            "match": {"room_tags": ["gym"], "incident_labels": ["fall"]},
            "suppress_between": ["18:00", "20:00"],
        }],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    outside = datetime(2026, 6, 19, 15, 0)
    result = eng.evaluate("default", room_tags=["gym"], label="fall", severity="high", now=outside)
    assert result.suppress is False


def test_policy_fall_risk_lowers_threshold(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "rules": [{
            "name": "fall_risk",
            "type": "threshold_override",
            "match": {"room_tags": ["fall-risk"]},
            "threshold": 0.5,
        }],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    result = eng.evaluate("default", room_tags=["fall-risk"], label="fall", severity="high")
    assert result.threshold_override == pytest.approx(0.5)


def test_policy_bathroom_suppresses_low_severity(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "rules": [{
            "name": "bathroom_filter",
            "type": "severity_filter",
            "match": {"room_tags": ["bathroom"]},
            "min_severity": "high",
        }],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    result = eng.evaluate("default", room_tags=["bathroom"], label="inactivity", severity="low")
    assert result.suppress is True


def test_policy_bathroom_allows_high_severity(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "rules": [{
            "name": "bathroom_filter",
            "type": "severity_filter",
            "match": {"room_tags": ["bathroom"]},
            "min_severity": "high",
        }],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    result = eng.evaluate("default", room_tags=["bathroom"], label="fall", severity="high")
    assert result.suppress is False


def test_policy_fusion_weights_loaded(tmp_path):
    from services.agent.policy import PolicyEngine
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "fusion_weights": {"yolo": 0.05, "pose": 0.25, "action": 0.2, "vlm": 0.45, "kb": 0.05},
        "rules": [],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    weights = eng.fusion_weights("default")
    assert weights["vlm"] == pytest.approx(0.45)
    assert weights["yolo"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Confidence fusion tests
# ---------------------------------------------------------------------------

def test_fusion_all_sources_equal_scores():
    from services.agent.fusion import fuse, DEFAULT_WEIGHTS
    scores = {k: 0.8 for k in DEFAULT_WEIGHTS}
    result = fuse(scores)
    assert result == pytest.approx(0.8)


def test_fusion_all_sources_weighted_sum():
    from services.agent.fusion import fuse
    weights = {"yolo": 0.1, "pose": 0.2, "action": 0.2, "vlm": 0.4, "kb": 0.1}
    scores = {"yolo": 1.0, "pose": 0.0, "action": 0.0, "vlm": 1.0, "kb": 0.0}
    result = fuse(scores, weights)
    assert result == pytest.approx(0.1 + 0.4)  # yolo + vlm


def test_fusion_missing_action_redistributes():
    from services.agent.fusion import fuse
    weights = {"yolo": 0.1, "pose": 0.2, "action": 0.2, "vlm": 0.4, "kb": 0.1}
    # action absent — its 0.2 redistributed proportionally to the remaining 0.8
    scores = {"yolo": 1.0, "pose": 1.0, "vlm": 1.0, "kb": 1.0}
    result = fuse(scores, weights)
    assert result == pytest.approx(1.0)  # all present sources score 1.0 → fused = 1.0


def test_fusion_present_zero_keeps_weight():
    from services.agent.fusion import fuse
    weights = {"yolo": 0.1, "pose": 0.2, "action": 0.2, "vlm": 0.4, "kb": 0.1}
    # action present with 0.0 — weight kept, not redistributed
    scores = {"yolo": 1.0, "pose": 1.0, "action": 0.0, "vlm": 1.0, "kb": 1.0}
    result = fuse(scores, weights)
    assert result == pytest.approx(0.1 + 0.2 + 0.0 + 0.4 + 0.1)  # = 0.8


def test_fusion_clamps_above_one():
    from services.agent.fusion import fuse
    scores = {"yolo": 2.0, "pose": 2.0, "vlm": 2.0}
    result = fuse(scores)
    assert result <= 1.0


def test_fusion_custom_weights_loaded_from_policy(tmp_path):
    from services.agent.policy import PolicyEngine
    from services.agent.fusion import fuse
    policy_dir = _write_policy(tmp_path, {
        "version": 1,
        "fusion_weights": {"yolo": 0.0, "pose": 0.0, "action": 0.0, "vlm": 1.0, "kb": 0.0},
        "rules": [],
    })
    eng = PolicyEngine(policy_dir=policy_dir)
    weights = eng.fusion_weights("default")
    scores = {"yolo": 0.5, "pose": 0.5, "action": 0.5, "vlm": 0.6, "kb": 0.5}
    result = fuse(scores, weights)
    assert result == pytest.approx(0.6)  # only vlm contributes


# ---------------------------------------------------------------------------
# Decide node tests
# ---------------------------------------------------------------------------

def test_decide_alert_when_threshold_met():
    from services.agent.graph import decide
    state = _base_state(
        fused_confidence=0.75, policy_threshold=0.7, policy_suppress=False,
        vlm_result=_vlm(is_stub=False),
    )
    result = decide(state)
    assert result["decision"] == "alert"
    assert result["incident_state"] == "alert"


def test_decide_dismiss_when_suppressed():
    from services.agent.graph import decide
    state = _base_state(
        fused_confidence=0.95, policy_threshold=0.7, policy_suppress=True,
        vlm_result=_vlm(is_stub=False),
    )
    result = decide(state)
    assert result["decision"] == "dismiss"
    assert result["incident_state"] == "dismissed"


def test_decide_dismiss_when_below_threshold():
    from services.agent.graph import decide
    state = _base_state(
        fused_confidence=0.5, policy_threshold=0.7, policy_suppress=False,
        vlm_result=_vlm(is_stub=False),
    )
    result = decide(state)
    assert result["decision"] == "dismiss"


def test_decide_stub_caution_no_rules_dismisses():
    """Stub VLM with no fired rules always dismisses, even with high confidence."""
    from services.agent.graph import decide
    state = _base_state(
        fused_confidence=0.85, policy_threshold=0.7, policy_suppress=False,
        vlm_result=_vlm(is_stub=True),
        fired_rules=[],
    )
    result = decide(state)
    assert result["decision"] == "dismiss"


def test_decide_stub_caution_with_rules_and_high_confidence_alerts():
    """Stub VLM with fired rules and confidence >= threshold+0.1 can alert."""
    from services.agent.graph import decide
    state = _base_state(
        fused_confidence=0.85,  # 0.7 + 0.1 = 0.8, so 0.85 >= 0.8
        policy_threshold=0.7, policy_suppress=False,
        vlm_result=_vlm(is_stub=True),
        fired_rules=["fall_pose_detected"],
    )
    result = decide(state)
    assert result["decision"] == "alert"


def test_decide_stub_caution_with_rules_but_insufficient_confidence_dismisses():
    """Stub VLM with fired rules but fused < threshold+0.1 dismisses."""
    from services.agent.graph import decide
    state = _base_state(
        fused_confidence=0.75,  # threshold+0.1 = 0.8, so 0.75 < 0.8
        policy_threshold=0.7, policy_suppress=False,
        vlm_result=_vlm(is_stub=True),
        fired_rules=["fall_pose_detected"],
    )
    result = decide(state)
    assert result["decision"] == "dismiss"


# ---------------------------------------------------------------------------
# Full graph integration tests
# ---------------------------------------------------------------------------

def _make_graph_session():
    """SQLite session with incidents + incident_audit tables."""
    engine = create_engine(
        "sqlite://", future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_graph_alert_path_writes_incident_and_audit():
    from services.agent.graph import make_agent_graph
    from services.persistence.db import Incident, IncidentAudit
    session = _make_graph_session()
    graph = make_agent_graph(policy_dir="config/policies")

    state = _base_state(
        session=session, kb=None,
        vlm_result=_vlm(label="fall", confidence=0.9, is_stub=False),
        yolo_confidence=0.9, pose_confidence=0.85, action_confidence=None,
        policy_suppress=False, policy_threshold=0.7,
        fired_rules=["fall_pose_detected"],
    )
    graph.invoke(state)

    incidents = session.query(Incident).all()
    assert len(incidents) == 1
    assert incidents[0].state == "alert"
    assert incidents[0].event_type == "fall"
    assert incidents[0].severity == "high"
    assert incidents[0].confidence > 0.0

    audits = session.query(IncidentAudit).all()
    assert len(audits) == 1
    assert audits[0].from_state == "new"
    assert audits[0].to_state == "alert"


def test_graph_dismiss_path_writes_dismissed_incident():
    from services.agent.graph import make_agent_graph
    from services.persistence.db import Incident, IncidentAudit
    session = _make_graph_session()
    graph = make_agent_graph(policy_dir="config/policies")

    state = _base_state(
        session=session, kb=None,
        vlm_result=_vlm(label="fall", confidence=0.3, is_stub=False),
        yolo_confidence=0.4, pose_confidence=0.35, action_confidence=None,
        policy_suppress=False, policy_threshold=0.7,
        fired_rules=["fall_pose_detected"],
    )
    graph.invoke(state)

    incidents = session.query(Incident).all()
    assert len(incidents) == 1
    assert incidents[0].state == "dismissed"

    audits = session.query(IncidentAudit).all()
    assert audits[0].to_state == "dismissed"


def test_graph_kb_writeback_called_on_alert():
    from services.agent.graph import make_agent_graph
    session = _make_graph_session()
    mock_kb = MagicMock()
    mock_kb.find_similar.return_value = []
    graph = make_agent_graph(policy_dir="config/policies")

    state = _base_state(
        session=session, kb=mock_kb,
        vlm_result=_vlm(label="fall", confidence=0.9, is_stub=False),
        yolo_confidence=0.9, pose_confidence=0.85,
    )
    graph.invoke(state)

    mock_kb.write.assert_called_once()
    call_kwargs = mock_kb.write.call_args.kwargs
    assert call_kwargs["operator_decision"] == "pending"


def test_graph_kb_writeback_skipped_when_no_kb():
    from services.agent.graph import make_agent_graph
    session = _make_graph_session()
    graph = make_agent_graph(policy_dir="config/policies")

    state = _base_state(
        session=session, kb=None,
        vlm_result=_vlm(label="fall", confidence=0.9, is_stub=False),
    )
    # Should not raise even with kb=None
    graph.invoke(state)


def test_persist_node_skips_db_write_when_dry_run():
    """persist node makes no DB or filesystem writes when dry_run=True."""
    from unittest.mock import MagicMock
    from services.agent.graph import persist

    session = MagicMock()
    state = _base_state(session=session, dry_run=True)
    persist(state)

    session.add.assert_not_called()
    session.commit.assert_not_called()
