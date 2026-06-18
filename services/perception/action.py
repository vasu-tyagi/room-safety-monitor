"""Action recognition path (L2): SlowFast on Kinetics-400.

We use the real slowfast_r50 model from pytorchvideo.models.hub, but implement
the input preprocessing by hand because pytorchvideo.transforms does not import
on torchvision 0.27 (it references the removed torchvision.transforms.functional_tensor).

Honesty note: Kinetics-400 has no clean "falling" class and only fighting-like
classes (wrestling, boxing, ...). The {running, falling, fighting} target set
the architecture asks for is approximated by a curated mapping over K400 class
names; everything else maps to "other". See KINETICS_TARGET_MAP.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Kinetics eval normalization (per the standard SlowFast recipe).
_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

# Curated, deliberately small map from Kinetics-400 class names to the target
# action set used by the Event Gate (Slice 4). Approximate by design.
KINETICS_TARGET_MAP = {
    # running-like
    "jogging": "running",
    "running on treadmill": "running",
    "sprinting": "running",
    "parkour": "running",
    # fighting-like
    "wrestling": "fighting",
    "punching person (boxing)": "fighting",
    "boxing": "fighting",
    "side kick": "fighting",
    "headbutting": "fighting",
    "slapping": "fighting",
    "capoeira": "fighting",
    # falling-like (K400 has no direct "falling"; closest cues)
    "falling off chair": "falling",
    "faceplant": "falling",
}

TARGET_ACTIONS = {"running", "falling", "fighting"}


def to_target_action(class_name: str) -> str:
    """Map a Kinetics-400 class name to {running, falling, fighting, other}."""
    return KINETICS_TARGET_MAP.get(class_name, "other")


def preprocess_clip(frames, num_frames=32, size=256):
    """Turn a list of RGB uint8 frames (H, W, 3) into a normalized (C, T, H, W) tensor.

    Uniformly subsamples (or pads by repetition) to num_frames, resizes each
    frame to size x size, scales to [0, 1], and normalizes with the Kinetics stats.
    """
    if len(frames) == 0:
        raise ValueError("preprocess_clip received an empty clip")

    arr = np.stack(frames).astype(np.float32) / 255.0  # (T0, H, W, C)
    clip = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (C, T0, H, W)

    # Uniform temporal subsample / pad to exactly num_frames.
    t0 = clip.shape[1]
    idx = torch.linspace(0, t0 - 1, num_frames).round().long().clamp(0, t0 - 1)
    clip = clip[:, idx]  # (C, num_frames, H, W)

    # Spatial resize to size x size.
    clip = F.interpolate(
        clip, size=(size, size), mode="bilinear", align_corners=False
    )

    clip = (clip - _MEAN) / _STD
    return clip.to(torch.float32)


def pack_pathways(clip, alpha=4):
    """Pack a (C, T, H, W) clip into SlowFast [slow, fast] pathways with a batch dim.

    The fast pathway sees all T frames; the slow pathway sees every alpha-th frame.
    """
    t = clip.shape[1]
    fast = clip
    slow_idx = torch.linspace(0, t - 1, t // alpha).long()
    slow = clip[:, slow_idx]
    return [slow.unsqueeze(0), fast.unsqueeze(0)]


@dataclass
class ActionResult:
    label: str  # raw Kinetics class name (or index string if names absent)
    target_action: str  # one of running/falling/fighting/other
    confidence: float


class ActionRecognizer:
    """Thin wrapper around a SlowFast model. The model is injectable for tests."""

    def __init__(self, model=None, classnames=None, num_frames=32, size=256, alpha=4):
        self._model = model
        self.classnames = classnames
        self.num_frames = num_frames
        self.size = size
        self.alpha = alpha

    def _get_model(self):
        if self._model is None:
            from pytorchvideo.models.hub import slowfast_r50

            self._model = slowfast_r50(pretrained=True).eval()
        return self._model

    def recognize(self, frames) -> ActionResult:
        clip = preprocess_clip(frames, num_frames=self.num_frames, size=self.size)
        pathways = pack_pathways(clip, alpha=self.alpha)
        model = self._get_model()
        with torch.no_grad():
            logits = model(pathways)
        probs = torch.softmax(logits, dim=1)[0]
        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())

        if self.classnames is not None and top_idx < len(self.classnames):
            label = self.classnames[top_idx]
        else:
            label = str(top_idx)
        return ActionResult(
            label=label,
            target_action=to_target_action(label),
            confidence=confidence,
        )
