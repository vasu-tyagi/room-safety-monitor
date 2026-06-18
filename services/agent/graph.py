"""LangGraph agent for the Slice 7 agent layer.

Graph topology (linear):
  parse_vlm_output -> policy_check -> confidence_fusion -> decide
    -> kb_writeback -> persist -> END

Each node function is defined at module level so it can be imported and tested
directly without running the full graph.

make_agent_graph() compiles the graph. The policy_dir parameter lets tests
point at a temp directory with fixture YAML files.
"""
import os
import uuid

import cv2
from langgraph.graph import END, StateGraph

from services.agent.fusion import fuse
from services.agent.policy import PolicyEngine
from services.agent.state import AgentState
from services.persistence.db import Incident, IncidentAudit

_LABEL_SEVERITY = {
    "fall": "high",
    "fight": "high",
    "overcrowding": "medium",
    "inactivity": "low",
    "none": "low",
    "unknown": "low",
}


# ---------------------------------------------------------------------------
# Node functions (module-level for direct testability)
# ---------------------------------------------------------------------------

def parse_vlm_output(state: AgentState) -> dict:
    """Normalise VLM result into label/severity/rationale.

    If the VLM returned "unknown" or ran in stub mode, severity defaults to
    "low" and rationale falls back to detection_notes (the L2 pose-geometry
    signal) so the downstream nodes still have a meaningful rationale.
    """
    vlm = state["vlm_result"]
    detection_notes = state.get("detection_notes", "")

    if vlm.is_stub or vlm.label == "unknown":
        return {
            "label": vlm.label,
            "severity": "low",
            "rationale": detection_notes or "No VLM analysis available.",
        }

    severity = _LABEL_SEVERITY.get(vlm.label, "low")
    rationale = vlm.rationale
    if detection_notes:
        rationale = f"{detection_notes} VLM: {vlm.rationale}"

    return {"label": vlm.label, "severity": severity, "rationale": rationale}


def policy_check(state: AgentState, policy_engine: PolicyEngine) -> dict:
    """Evaluate site policy and set suppress/threshold/severity overrides.

    Also loads fusion weights from the policy file and stores them in state
    so confidence_fusion does not need to re-read the YAML.
    """
    result = policy_engine.evaluate(
        facility_id=state["facility_id"],
        room_tags=state["room_tags"],
        label=state["label"],
        severity=state["severity"],
    )
    weights = policy_engine.fusion_weights(state["facility_id"])
    threshold = result.threshold_override if result.threshold_override is not None else 0.7

    return {
        "policy_suppress": result.suppress,
        "policy_threshold": threshold,
        "policy_severity": result.severity_override,
        "policy_notes": result.notes,
        "fusion_weights": weights,
    }


def confidence_fusion(state: AgentState) -> dict:
    """Build the per-source scores dict and compute fused confidence.

    action_confidence is omitted from scores when it is None (SlowFast not
    run) so its weight is redistributed by fuse(). If it is 0.0 the key is
    included, keeping its weight and contributing 0 to the sum.

    The KB is queried here for the best similarity score so it contributes to
    fusion. This is a second KB call (the first happens inside dispatch when
    building VLM prompt context) but is unavoidable given the agent's
    separation from the VLM dispatch path.
    """
    scores = {
        "yolo": state["yolo_confidence"],
        "pose": state["pose_confidence"],
        "vlm": state["vlm_result"].confidence,
    }

    action_conf = state.get("action_confidence")
    if action_conf is not None:
        scores["action"] = action_conf

    kb_matches_serialized = []
    kb = state.get("kb")
    if kb is not None:
        query = " ".join(state["fired_rules"]) if state.get("fired_rules") else "safety incident"
        try:
            similar = kb.find_similar(query, top_k=3, threshold=0.0)
            if similar:
                scores["kb"] = similar[0].similarity_score
                kb_matches_serialized = [
                    {
                        "id": str(e.id),
                        "label": e.label,
                        "rationale": e.rationale,
                        "operator_decision": e.operator_decision,
                        "similarity": e.similarity_score,
                    }
                    for e in similar
                ]
        except Exception:
            pass  # KB unavailable; omit kb score and redistribute its weight

    weights = state.get("fusion_weights")
    fused = fuse(scores, weights)
    return {
        "fused_confidence": fused,
        "confidence_breakdown": scores,
        "kb_matches": kb_matches_serialized,
    }


def decide(state: AgentState) -> dict:
    """Apply policy + fused confidence to determine ALERT or DISMISS.

    Stub caution rule: if the VLM ran in stub mode (no real model output),
    require both at least one gate rule to have fired AND fused_confidence >=
    policy_threshold + 0.1 before alerting. Without these guards, a stub with
    a high pose-geometry score alone could generate false alerts since there is
    no real VLM confirmation.
    """
    suppress = state["policy_suppress"]
    fused = state["fused_confidence"]
    threshold = state["policy_threshold"]
    is_stub = state["vlm_result"].is_stub
    fired_rules = state.get("fired_rules", [])

    if suppress:
        return {
            "decision": "dismiss",
            "incident_state": "dismissed",
            "decide_reason": "policy suppressed",
        }

    if is_stub:
        # Stub caution: only alert when gate rules corroborate AND confidence
        # clears the wider margin (threshold + 0.1).
        stub_threshold = threshold + 0.1
        if fired_rules and fused >= stub_threshold:
            return {
                "decision": "alert",
                "incident_state": "alert",
                "decide_reason": (
                    f"stub-caution alert: fused={fused:.3f} >= "
                    f"threshold+0.1={stub_threshold:.3f}, rules={fired_rules}"
                ),
            }
        return {
            "decision": "dismiss",
            "incident_state": "dismissed",
            "decide_reason": (
                f"stub-caution dismiss: fused={fused:.3f} "
                f"rules={fired_rules} needed={stub_threshold:.3f}"
            ),
        }

    if fused >= threshold:
        return {
            "decision": "alert",
            "incident_state": "alert",
            "decide_reason": f"fused={fused:.3f} >= threshold={threshold:.3f}",
        }
    return {
        "decision": "dismiss",
        "incident_state": "dismissed",
        "decide_reason": f"fused={fused:.3f} < threshold={threshold:.3f}",
    }


def kb_writeback(state: AgentState) -> dict:
    """Write a KB entry for this incident. Replaces the Slice 6 pipeline write.

    operator_decision is "pending" for alerts (awaiting operator review) and
    "agent_dismissed" for dismissals (agent determined no action needed).
    Skipped when kb is None or when the VLM ran in stub mode (no real analysis
    to record).
    """
    kb = state.get("kb")
    vlm = state["vlm_result"]
    if kb is None or vlm.is_stub or state.get("vlm_prompt") is None:
        return {}

    operator_decision = "pending" if state["decision"] == "alert" else "agent_dismissed"
    kb.write(
        prompt=state["vlm_prompt"],
        rationale=state["rationale"],
        label=state["label"],
        operator_decision=operator_decision,
        metadata={
            "camera_id": state["camera_id"],
            "room_id": state["room_id"],
            "frame_idx": state.get("frame_idx", -1),
            "fired_rules": state.get("fired_rules", []),
            "decision": state["decision"],
        },
    )
    return {}


def _save_clip(frames, path):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def persist(state: AgentState) -> dict:
    """Write the incident row and an audit row to Postgres (or SQLite in tests).

    When dry_run=True (replay mode) all writes are skipped so replays never
    alter the database or filesystem.
    """
    if state.get("dry_run", False):
        return {}

    session = state["session"]
    incident_id = uuid.uuid4()

    clip_frames = state.get("clip_frames") or []
    clips_dir = state.get("clips_dir") or "clips"
    evidence_clip_url = None
    if clip_frames:
        os.makedirs(clips_dir, exist_ok=True)
        clip_path = os.path.join(clips_dir, f"{incident_id}.mp4")
        _save_clip(clip_frames, clip_path)
        evidence_clip_url = clip_path

    severity = state.get("policy_severity") or state["severity"]
    incident = Incident(
        id=incident_id,
        camera_id=state["camera_id"],
        room_id=state["room_id"],
        event_type="fall",
        severity=severity,
        confidence=state["fused_confidence"],
        rationale=state["rationale"],
        state=state["incident_state"],
        evidence_clip_url=evidence_clip_url,
        vlm_prompt=state.get("vlm_prompt"),
        fired_rules=state.get("fired_rules") or [],
        confidence_breakdown=state.get("confidence_breakdown") or {},
        kb_matches=state.get("kb_matches") or [],
    )
    session.add(incident)

    audit = IncidentAudit(
        incident_id=incident_id,
        from_state="new",
        to_state=state["incident_state"],
        reason=state.get("decide_reason"),
        agent_node="decide",
    )
    session.add(audit)
    session.commit()
    return {}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def make_agent_graph(policy_dir: str = "config/policies"):
    """Compile and return the agent StateGraph.

    policy_dir: directory containing <facility_id>.yaml files. Override in
                tests to point at a temp directory with fixture policy files.
    """
    engine = PolicyEngine(policy_dir=policy_dir)

    import functools
    policy_node = functools.partial(policy_check, policy_engine=engine)

    graph = StateGraph(AgentState)
    graph.add_node("parse_vlm_output", parse_vlm_output)
    graph.add_node("policy_check", policy_node)
    graph.add_node("confidence_fusion", confidence_fusion)
    graph.add_node("decide", decide)
    graph.add_node("kb_writeback", kb_writeback)
    graph.add_node("persist", persist)

    graph.set_entry_point("parse_vlm_output")
    graph.add_edge("parse_vlm_output", "policy_check")
    graph.add_edge("policy_check", "confidence_fusion")
    graph.add_edge("confidence_fusion", "decide")
    graph.add_edge("decide", "kb_writeback")
    graph.add_edge("kb_writeback", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
