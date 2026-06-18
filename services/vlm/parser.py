"""Parse structured fields from a Qwen 2.5 VL response (Slice 5).

The VLM is prompted to respond with INCIDENT_TYPE / CONFIDENCE / RATIONALE
lines. If the model ignores the format, regex finds what it can and fills
defaults for the rest — no exception is raised.
"""
import logging
import re

from services.vlm.stub import VLMResult

log = logging.getLogger(__name__)

VALID_LABELS = {"fall", "fight", "overcrowding", "inactivity", "none"}

_TYPE_RE = re.compile(r"^INCIDENT_TYPE:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
_CONF_RE = re.compile(r"^CONFIDENCE:\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
_RAT_RE  = re.compile(r"^RATIONALE:\s*(.+)",       re.IGNORECASE | re.MULTILINE)


def parse_vlm_response(text: str) -> VLMResult:
    """Extract label, confidence, and rationale from raw VLM output.

    Returns VLMResult(is_stub=False) — the model was called even if parsing
    partially failed. Caller can distinguish real-but-garbled from stub via
    is_stub.
    """
    if not text.strip():
        log.warning("VLM returned empty response; using defaults")
        return VLMResult(label="unknown", rationale="", confidence=0.0, is_stub=False)

    label = "unknown"
    m = _TYPE_RE.search(text)
    if m:
        raw = m.group(1).strip().lower().rstrip(".,;")
        if raw in VALID_LABELS:
            label = raw
        else:
            log.warning("VLM INCIDENT_TYPE %r not in valid set; defaulting to 'unknown'", raw)

    confidence = 0.0
    m = _CONF_RE.search(text)
    if m:
        try:
            confidence = min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            log.warning("VLM CONFIDENCE %r is not a float; defaulting to 0.0", m.group(1))

    rationale = ""
    m = _RAT_RE.search(text)
    if m:
        rationale = m.group(1).strip()

    return VLMResult(label=label, rationale=rationale, confidence=confidence, is_stub=False)
