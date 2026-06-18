"""Tests for the SlowFast action path (L2).

Covers the hand-rolled preprocessing (we bypass pytorchvideo.transforms, which
is broken on torchvision 0.27) and the Kinetics-400 -> target-action mapping.
Model inference itself is exercised through an injected fake so tests need no
weights download.
"""
import numpy as np
import torch

from services.perception.action import (
    ActionRecognizer,
    pack_pathways,
    preprocess_clip,
    to_target_action,
)


def _frames(n, h=48, w=64):
    return [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]


def test_preprocess_clip_normalizes_to_fixed_shape():
    out = preprocess_clip(_frames(10), num_frames=32, size=256)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 256, 256)
    assert out.dtype == torch.float32


def test_preprocess_clip_handles_more_frames_than_target():
    out = preprocess_clip(_frames(100), num_frames=32, size=224)
    assert out.shape == (3, 32, 224, 224)


def test_pack_pathways_splits_slow_and_fast():
    clip = torch.zeros(3, 32, 256, 256)
    slow, fast = pack_pathways(clip, alpha=4)
    # batch dim added, channels first
    assert fast.shape == (1, 3, 32, 256, 256)
    assert slow.shape == (1, 3, 8, 256, 256)


def test_to_target_action_maps_known_classes():
    assert to_target_action("jogging") == "running"
    assert to_target_action("wrestling") == "fighting"
    assert to_target_action("brushing teeth") == "other"


def test_action_recognizer_returns_label_and_confidence():
    classnames = ["jogging", "brushing teeth", "wrestling"]

    class FakeModel:
        def __call__(self, pathways):
            # highest logit on index 2 -> "wrestling"
            return torch.tensor([[0.1, 0.2, 5.0]])

        def eval(self):
            return self

    rec = ActionRecognizer(model=FakeModel(), classnames=classnames)
    result = rec.recognize(_frames(16))
    assert result.label == "wrestling"
    assert result.target_action == "fighting"
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.5
