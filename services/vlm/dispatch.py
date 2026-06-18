"""VLM mode dispatcher (Slice 5/6).

Routes escalated frames to the real Qwen 2.5 VL client or the stub based
on the VLM_MODE environment variable. When a KB instance is passed, retrieves
similar prior incidents to ground the prompt before calling the VLM.

VLM_MODE behaviour:
  real (default): HF_TOKEN required. Missing token -> ERROR log + stub fallback,
                  reason="no-token". Transient failures (rate-limit, network,
                  auth) -> WARNING log + stub fallback.
  auto:           HF_TOKEN optional. Missing token -> INFO log + stub (silent
                  fallback, not a misconfiguration). Transient failures -> INFO
                  log + stub.
  stub:           Never calls the real VLM. HF_TOKEN not checked.

The difference between real and auto is intentional: real is for deployments
where a token is expected to be present (missing = operator error), while auto
is for demos and offline runs where graceful degradation is the goal.

Every call logs at INFO or above: "L3 VLM: mode=X result=Y reason=Z".
"""
import logging
import os

from dotenv import load_dotenv

from services.vlm.prompts import SAFETY_ANALYSIS
from services.vlm.qwen_client import AuthError, RateLimitError, VLMNetworkError, analyze_clip
from services.vlm.stub import VLMResult

load_dotenv()

log = logging.getLogger(__name__)


def _stub_result(fired_rules: list) -> VLMResult:
    rule_summary = ", ".join(fired_rules) if fired_rules else "none"
    return VLMResult(
        label="unknown",
        rationale=f"Stub L3 analysis. Rules that fired: {rule_summary}.",
        confidence=0.0,
        is_stub=True,
    )


def _log_fallback(mode: str, detail: str) -> None:
    msg = "L3 VLM: mode=%s falling back to stub (%s)"
    if mode == "real":
        log.warning(msg, mode, detail)
    else:
        log.info(msg, mode, detail)


def _format_kb_context(entries: list) -> str:
    """Format KB retrieval results for injection into the VLM prompt."""
    lines = []
    for e in entries:
        lines.append(
            f"{e.label} (similarity {e.similarity_score:.2f}): "
            f"{e.rationale} [operator: {e.operator_decision}]"
        )
    return "\n".join(lines)


def analyze_escalated(
    clip_frames: list,
    fired_rules: list,
    kb_context: str = "",
    kb=None,
) -> tuple:
    """Route an escalated frame to the real VLM or stub.

    Args:
        clip_frames: BGR numpy arrays from the pipeline frame buffer.
        fired_rules: event gate rule names that fired (used in stub rationale
                     and KB similarity query).
        kb_context: explicit prior incident context; if provided, skips KB query.
        kb: optional KB instance (Slice 6). When present and kb_context is
            empty, queries for top-3 similar incidents above similarity=0.7
            and injects them into the prompt.

    Returns:
        (VLMResult, reason, prompt_used) where:
          reason is one of: success | rate-limited | auth-failed |
                            network-error | no-token | stub-mode
          prompt_used is the full prompt string sent to the model, or None if
                      the stub path was taken (no network call was made).
    """
    mode = os.environ.get("VLM_MODE", "real").lower()

    if mode == "stub":
        log.info("L3 VLM: mode=stub result=stub reason=stub-mode")
        return _stub_result(fired_rules), "stub-mode", None

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        if mode == "real":
            log.error(
                "L3 VLM: mode=real but HF_TOKEN is not set (misconfiguration); "
                "falling back to stub"
            )
        else:
            log.info("L3 VLM: mode=auto HF_TOKEN not set; using stub")
        return _stub_result(fired_rules), "no-token", None

    # Build KB context from prior similar incidents if not explicitly provided.
    if kb is not None and not kb_context:
        query = " ".join(fired_rules) if fired_rules else "safety incident"
        similar = kb.find_similar(query, top_k=3, threshold=0.7)
        if similar:
            kb_context = _format_kb_context(similar)

    prompt = SAFETY_ANALYSIS.format(kb_context=kb_context)

    try:
        result = analyze_clip(clip_frames, prompt, hf_token=token)
        log.info("L3 VLM: mode=%s result=real reason=success", mode)
        return result, "success", prompt
    except RateLimitError as exc:
        _log_fallback(mode, f"rate-limited: {exc}")
        return _stub_result(fired_rules), "rate-limited", None
    except AuthError as exc:
        _log_fallback(mode, f"auth-failed: {exc}")
        return _stub_result(fired_rules), "auth-failed", None
    except VLMNetworkError as exc:
        _log_fallback(mode, f"network-error: {exc}")
        return _stub_result(fired_rules), "network-error", None
