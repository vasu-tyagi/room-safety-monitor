"""Tests for Slice 5 VLM: parser, client (mocked HTTP), dispatch, integration.

Real-network test is skipped unless both HF_TOKEN and RUN_NETWORK_TESTS=1 are set.
"""
import logging
import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import requests as _req
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.persistence.db import Base
from services.vlm.dispatch import analyze_escalated
from services.vlm.parser import parse_vlm_response
from services.vlm.prompts import SAFETY_ANALYSIS
from services.vlm.qwen_client import AuthError, RateLimitError, VLMNetworkError, analyze_clip
from services.vlm.stub import VLMResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_frame(w=64, h=48):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_resp(status, text="", json_data=None):
    m = MagicMock()
    m.status_code = status
    m.text = text
    if json_data is not None:
        m.json.return_value = json_data
    else:
        m.json.side_effect = ValueError("no json")
    if status >= 400:
        m.raise_for_status.side_effect = _req.HTTPError(f"HTTP {status}")
    return m


_GOOD_JSON = {
    "choices": [{"message": {"content": (
        "INCIDENT_TYPE: fall\nCONFIDENCE: 0.9\nRATIONALE: Person appears to have fallen."
    )}}]
}


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parser_well_formed():
    text = "INCIDENT_TYPE: fall\nCONFIDENCE: 0.85\nRATIONALE: Person fell near the doorway."
    r = parse_vlm_response(text)
    assert r.label == "fall"
    assert abs(r.confidence - 0.85) < 1e-6
    assert "fell" in r.rationale
    assert not r.is_stub


def test_parser_malformed_confidence():
    text = "INCIDENT_TYPE: none\nCONFIDENCE: high\nRATIONALE: All clear."
    r = parse_vlm_response(text)
    assert r.label == "none"
    assert r.confidence == 0.0


def test_parser_missing_incident_type():
    text = "CONFIDENCE: 0.5\nRATIONALE: Something happened."
    r = parse_vlm_response(text)
    assert r.label == "unknown"


def test_parser_missing_rationale():
    text = "INCIDENT_TYPE: fall\nCONFIDENCE: 0.9"
    r = parse_vlm_response(text)
    assert r.label == "fall"
    assert r.rationale == ""


def test_parser_empty_response():
    r = parse_vlm_response("")
    assert r.label == "unknown"
    assert r.confidence == 0.0
    assert r.rationale == ""
    assert not r.is_stub


def test_parser_confidence_clamped_above_one():
    text = "INCIDENT_TYPE: fall\nCONFIDENCE: 1.5\nRATIONALE: Extreme."
    r = parse_vlm_response(text)
    assert r.confidence <= 1.0


def test_parser_all_valid_labels():
    for label in ("fall", "fight", "overcrowding", "inactivity", "none"):
        r = parse_vlm_response(f"INCIDENT_TYPE: {label}\nCONFIDENCE: 0.5\nRATIONALE: x.")
        assert r.label == label


# ---------------------------------------------------------------------------
# Client tests (mocked HTTP)
# ---------------------------------------------------------------------------

def test_client_success():
    with patch("requests.post", return_value=_mock_resp(200, json_data=_GOOD_JSON)):
        r = analyze_clip([_black_frame()], "test prompt", hf_token="fake")
    assert not r.is_stub
    assert r.label == "fall"
    assert r.confidence == 0.9


def test_client_rate_limited():
    with patch("requests.post", return_value=_mock_resp(429, text="rate limited")):
        with pytest.raises(RateLimitError):
            analyze_clip([_black_frame()], "test prompt", hf_token="fake")


def test_client_auth_failed():
    with patch("requests.post", return_value=_mock_resp(401, text="unauthorized")):
        with pytest.raises(AuthError):
            analyze_clip([_black_frame()], "test prompt", hf_token="bad")


def test_client_network_error():
    with patch("requests.post", side_effect=_req.ConnectionError("refused")):
        with pytest.raises(VLMNetworkError):
            analyze_clip([_black_frame()], "test prompt", hf_token="fake")


def test_client_timeout():
    with patch("requests.post", side_effect=_req.Timeout("timed out")):
        with pytest.raises(VLMNetworkError):
            analyze_clip([_black_frame()], "test prompt", hf_token="fake")


def test_client_samples_frames_from_clip():
    """Verify the client encodes only a sample of frames, not all 32."""
    call_args = {}

    def capture_post(url, headers=None, json=None, timeout=None):
        call_args["content"] = json["messages"][0]["content"]
        return _mock_resp(200, json_data=_GOOD_JSON)

    frames = [_black_frame() for _ in range(32)]
    with patch("requests.post", side_effect=capture_post):
        analyze_clip(frames, "prompt", hf_token="fake")

    image_entries = [c for c in call_args["content"] if c["type"] == "image_url"]
    assert len(image_entries) <= 4  # sampled, not all 32


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

def test_dispatch_stub_mode_never_calls_http(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "stub")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("requests.post") as mock_post:
        result, reason = analyze_escalated([_black_frame()], ["fall_pose_detected"])
    mock_post.assert_not_called()
    assert result.is_stub
    assert reason == "stub-mode"


def test_dispatch_real_fallback_on_rate_limit(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    with patch("services.vlm.dispatch.analyze_clip", side_effect=RateLimitError("429")):
        result, reason = analyze_escalated([_black_frame()], ["fall_pose_detected"])
    assert result.is_stub
    assert reason == "rate-limited"


def test_dispatch_real_fallback_on_network_error(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    with patch("services.vlm.dispatch.analyze_clip", side_effect=VLMNetworkError("down")):
        result, reason = analyze_escalated([_black_frame()], ["fall_pose_detected"])
    assert result.is_stub
    assert reason == "network-error"


def test_dispatch_real_fallback_on_auth_error(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.setenv("HF_TOKEN", "bad-token")
    with patch("services.vlm.dispatch.analyze_clip", side_effect=AuthError("401")):
        result, reason = analyze_escalated([_black_frame()], ["fall_pose_detected"])
    assert result.is_stub
    assert reason == "auth-failed"


def test_dispatch_real_missing_token_falls_back(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result, reason = analyze_escalated([_black_frame()], ["fall_pose_detected"])
    assert result.is_stub
    assert reason == "no-token"


def test_dispatch_real_missing_token_logs_error(monkeypatch, caplog):
    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with caplog.at_level(logging.DEBUG, logger="services.vlm.dispatch"):
        analyze_escalated([_black_frame()], [])
    error_records = [r for r in caplog.records
                     if r.name == "services.vlm.dispatch" and r.levelno >= logging.ERROR]
    assert len(error_records) >= 1


def test_dispatch_auto_no_token_silent_stub(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "auto")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("requests.post") as mock_post:
        result, reason = analyze_escalated([_black_frame()], [])
    mock_post.assert_not_called()
    assert result.is_stub
    assert reason == "no-token"


def test_dispatch_auto_rate_limit_logs_info_not_warning(monkeypatch, caplog):
    monkeypatch.setenv("VLM_MODE", "auto")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    with patch("services.vlm.dispatch.analyze_clip", side_effect=RateLimitError("429")):
        with caplog.at_level(logging.DEBUG, logger="services.vlm.dispatch"):
            result, reason = analyze_escalated([_black_frame()], [])
    assert result.is_stub
    assert reason == "rate-limited"
    dispatch_warnings = [r for r in caplog.records
                         if r.name == "services.vlm.dispatch" and r.levelno >= logging.WARNING]
    assert len(dispatch_warnings) == 0


def test_dispatch_real_success_returns_non_stub(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "real")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    fake = VLMResult(label="fall", rationale="Fell.", confidence=0.9, is_stub=False)
    with patch("services.vlm.dispatch.analyze_clip", return_value=fake):
        result, reason = analyze_escalated([_black_frame()], ["fall_pose_detected"])
    assert not result.is_stub
    assert reason == "success"


def test_dispatch_accepts_kb_context(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "stub")
    # kb_context parameter exists; stub mode ignores it but must not error
    result, reason = analyze_escalated([_black_frame()], [], kb_context="prior fall 2026-01-01")
    assert result.is_stub


# ---------------------------------------------------------------------------
# Pipeline integration: VLM_MODE=stub, end-to-end
# ---------------------------------------------------------------------------

def _make_video(path, n_frames, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def _db_session():
    engine = create_engine("sqlite://", future=True,
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_pipeline_stub_mode_no_http(tmp_path, monkeypatch):
    from services.perception.detection import Detection
    from services.perception.fall import FallAssessment
    from services.perception.pose import PersonPose
    from services.pipeline.perception_pipeline import process_video

    monkeypatch.setenv("VLM_MODE", "stub")

    class FakeDetector:
        def persons(self, frame, conf=None):
            return [Detection((0, 0, 50, 80), 0.9, 0, "person", "person")]

    # Horizontal shoulders/hips → fall.is_fall=True from pose geometry
    kp = np.zeros((17, 2), dtype=float)
    kp[5] = (100, 150); kp[6] = (100, 170)   # L/R shoulder — same y
    kp[11] = (220, 150); kp[12] = (220, 170)  # L/R hip — same y

    class FakePose:
        def estimate(self, frame, bboxes):
            return [PersonPose(keypoints=kp, scores=np.ones(17)) for _ in bboxes]

    video = tmp_path / "fall.mp4"
    _make_video(video, 10)
    session = _db_session()

    with patch("requests.post") as mock_post:
        result = process_video(
            video, session,
            detector=FakeDetector(),
            pose_estimator=FakePose(),
            persist=3,
            room_policy={},  # enable gate
        )

    mock_post.assert_not_called()
    assert result["frames_processed"] == 10


# ---------------------------------------------------------------------------
# Real-network test (skipped unless HF_TOKEN and RUN_NETWORK_TESTS=1 are set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") and os.environ.get("RUN_NETWORK_TESTS")),
    reason="requires HF_TOKEN and RUN_NETWORK_TESTS=1",
)
def test_real_network_single_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    prompt = SAFETY_ANALYSIS.format(kb_context="")
    token = os.environ["HF_TOKEN"]
    result = analyze_clip([frame], prompt, hf_token=token)
    assert not result.is_stub
    assert result.label in {"fall", "fight", "overcrowding", "inactivity", "none", "unknown"}
    assert 0.0 <= result.confidence <= 1.0
