"""VLM mode dispatcher (Slice 5).

Routes escalated frames to the real Qwen 2.5 VL client or the stub based
on the VLM_MODE environment variable.

VLM_MODE behaviour:
  real (default): HF_TOKEN required. Missing token → ERROR log + stub fallback,
                  reason="no-token". Transient failures (rate-limit, network,
                  auth) → WARNING log + stub fallback.
  auto:           HF_TOKEN optional. Missing token → INFO log + stub (silent
                  fallback, not a misconfiguration). Transient failures → INFO
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


def analyze_escalated(
    clip_frames: list,
    fired_rules: list,
    kb_context: str = "",
) -> tuple:
    """Route an escalated frame to the real VLM or stub.

    Args:
        clip_frames: BGR numpy arrays from the pipeline frame buffer.
        fired_rules: event gate rule names that fired (used in stub rationale).
        kb_context: prior incident context from the pgvector KB. Empty in Slice 5;
                    Slice 6 passes real retrieval results without changing this signature.

    Returns:
        (VLMResult, reason) where reason is one of:
        success | rate-limited | auth-failed | network-error | no-token | stub-mode
    """
    mode = os.environ.get("VLM_MODE", "real").lower()

    if mode == "stub":
        log.info("L3 VLM: mode=stub result=stub reason=stub-mode")
        return _stub_result(fired_rules), "stub-mode"

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        if mode == "real":
            log.error(
                "L3 VLM: mode=real but HF_TOKEN is not set (misconfiguration); "
                "falling back to stub"
            )
        else:
            log.info("L3 VLM: mode=auto HF_TOKEN not set; using stub")
        return _stub_result(fired_rules), "no-token"

    prompt = SAFETY_ANALYSIS.format(kb_context=kb_context)

    try:
        result = analyze_clip(clip_frames, prompt, hf_token=token)
        log.info("L3 VLM: mode=%s result=real reason=success", mode)
        return result, "success"
    except RateLimitError as exc:
        _log_fallback(mode, f"rate-limited: {exc}")
        return _stub_result(fired_rules), "rate-limited"
    except AuthError as exc:
        _log_fallback(mode, f"auth-failed: {exc}")
        return _stub_result(fired_rules), "auth-failed"
    except VLMNetworkError as exc:
        _log_fallback(mode, f"network-error: {exc}")
        return _stub_result(fired_rules), "network-error"
