"""Qwen 2.5 VL client via Hugging Face Inference Providers (Slice 5).

Model: Qwen/Qwen2.5-VL-72B-Instruct
Primary provider: featherless-ai (confirmed live 2026-06-19)
Fallback provider: ovhcloud

Note: Qwen2.5-VL-7B-Instruct has no live HF provider (hyperbolic mapping exists
but is in error status). 72B is what the free tier actually serves.

Raises typed exceptions so the caller (dispatch.py) can decide the fallback:
  RateLimitError   — HTTP 429
  AuthError        — HTTP 401
  VLMNetworkError  — connection/timeout/unexpected response format
"""
import base64
import logging

import cv2
import requests

from services.vlm.parser import parse_vlm_response
from services.vlm.stub import VLMResult

log = logging.getLogger(__name__)

HF_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
HF_API_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"
HF_API_URL_FALLBACK = "https://router.huggingface.co/ovhcloud/v1/chat/completions"
_SAMPLE_N = 4  # maximum frames sampled from the clip buffer per API call


class RateLimitError(Exception):
    """HF API returned 429 Too Many Requests."""


class AuthError(Exception):
    """HF API returned 401 Unauthorized — check HF_TOKEN."""


class VLMNetworkError(Exception):
    """Network-level failure or unexpected response format from the HF API."""


def _sample_frames(frames: list, n: int = _SAMPLE_N) -> list:
    """Return up to n evenly-spaced frames from the clip."""
    if len(frames) <= n:
        return list(frames)
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]


def _encode_frame(frame) -> str:
    """Encode a BGR numpy array as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def analyze_clip(
    clip_frames: list,
    prompt: str,
    hf_token: str,
    timeout: int = 30,
) -> VLMResult:
    """Call Qwen 2.5 VL on a sampled clip and return a structured VLMResult.

    Args:
        clip_frames: list of BGR numpy arrays (raw frames from the pipeline buffer).
        prompt: formatted text prompt (from prompts.SAFETY_ANALYSIS).
        hf_token: Hugging Face API token.
        timeout: HTTP request timeout in seconds.

    Raises:
        RateLimitError: on HTTP 429.
        AuthError: on HTTP 401.
        VLMNetworkError: on connection/timeout/unexpected response.
    """
    sampled = _sample_frames(clip_frames)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode_frame(f)}"},
        }
        for f in sampled
    ]
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
    }

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # Try primary provider, then fallback.
    last_exc = None
    for url in (HF_API_URL, HF_API_URL_FALLBACK):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            break
        except requests.Timeout as exc:
            last_exc = VLMNetworkError(f"HF API timed out after {timeout}s")
        except requests.ConnectionError as exc:
            last_exc = VLMNetworkError(f"HF API connection error: {exc}")
    else:
        raise last_exc

    if resp.status_code == 429:
        raise RateLimitError(f"HF rate limit: {resp.text[:120]}")
    if resp.status_code == 401:
        raise AuthError("HF Inference auth failed — check HF_TOKEN")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise VLMNetworkError(f"HF API HTTP error: {exc}") from exc

    try:
        text = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise VLMNetworkError(f"Unexpected HF response format: {exc}") from exc

    return parse_vlm_response(text)
