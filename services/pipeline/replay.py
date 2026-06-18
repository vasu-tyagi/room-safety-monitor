"""Incident replay: re-run a saved clip through the current pipeline state.

replay_incident() loads a persisted clip, runs it through the full
L2+Gate+VLM+Agent pipeline in dry-run mode (no DB writes), and returns a
structured comparison of the original outcome versus the replayed outcome.

This lets an operator ask: "given everything the system knows today, would it
still handle this incident the same way?"

Clip retention:
  MAX_CLIP_RETENTION_DAYS controls how long clips should be kept on disk.
  A scheduled cleanup job is not implemented in this slice; a cron job or
  object-store lifecycle rule should delete clips older than this threshold.
"""
import os
import uuid

from services.persistence.db import Incident as IncidentRow
from services.pipeline.perception_pipeline import process_video

MAX_CLIP_RETENTION_DAYS = 30


def replay_incident(
    incident_id: str,
    session,
    kb=None,
    clips_dir: str = "clips",
    policy_dir: str = "config/policies",
) -> dict:
    """Re-run the evidence clip for incident_id through the current pipeline.

    Args:
        incident_id: string UUID of the incident to replay.
        session:     SQLAlchemy session (used to load the incident row).
        kb:          optional KB instance forwarded to process_video so the
                     current KB state influences the replayed outcome.
        clips_dir:   directory for any clips that would be created during a
                     real (non-dry-run) run; unused in replay but forwarded.
        policy_dir:  path to facility policy YAML files.

    Returns:
        dict with keys:
            original_incident  — the stored incident as a plain dict
            replayed_outcome   — metrics and decision from the dry-run
            changed            — structured diff: state_changed, confidence_delta,
                                 rationale_changed, any_change
    """
    parsed_id = uuid.UUID(incident_id) if not isinstance(incident_id, uuid.UUID) else incident_id
    row = session.get(IncidentRow, parsed_id)
    if row is None:
        raise ValueError(f"incident {incident_id} not found")

    clip_url = row.evidence_clip_url
    if not clip_url or not os.path.exists(clip_url):
        raise FileNotFoundError(f"clip not found: {clip_url!r}")

    replay_result = process_video(
        clip_url,
        session,
        room_policy={},  # empty policy enables the gate with no zone constraints
        kb=kb,
        policy_dir=policy_dir,
        clips_dir=clips_dir,
        dry_run=True,
    )

    original_conf = row.confidence
    replayed_conf = replay_result.get("last_fused_confidence", 0.0)
    original_state = row.state
    replayed_state = replay_result.get("last_incident_state", "")
    original_rationale = row.rationale
    replayed_rationale = replay_result.get("last_rationale", "")

    # state_changed is only True when the replay produced an incident AND it
    # differs from the original; "" means no agent was invoked (no falls
    # detected in the replay clip), which is not considered a state change.
    state_changed = bool(replayed_state) and original_state != replayed_state
    confidence_delta = replayed_conf - original_conf
    rationale_changed = bool(replayed_rationale) and original_rationale != replayed_rationale

    return {
        "original_incident": _incident_to_dict(row),
        "replayed_outcome": {
            "last_incident_state": replayed_state,
            "fused_confidence": replayed_conf,
            "rationale": replayed_rationale,
            "incidents_created": replay_result["incidents_created"],
            "frames_processed": replay_result["frames_processed"],
            "frames_escalated": replay_result["frames_escalated"],
            "escalation_ratio": replay_result["escalation_ratio"],
        },
        "changed": {
            "state_changed": state_changed,
            "confidence_delta": confidence_delta,
            "rationale_changed": rationale_changed,
            "any_change": state_changed or confidence_delta != 0.0 or rationale_changed,
        },
    }


def _incident_to_dict(row: IncidentRow) -> dict:
    return {
        "id": str(row.id),
        "camera_id": row.camera_id,
        "room_id": row.room_id,
        "event_type": row.event_type,
        "severity": row.severity,
        "confidence": row.confidence,
        "rationale": row.rationale,
        "state": row.state,
        "evidence_clip_url": row.evidence_clip_url,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
