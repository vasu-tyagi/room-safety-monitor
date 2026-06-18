"""Pose-geometry fall evaluation over the UR Fall dataset.

The v0.5 aspect-ratio baseline lives in src/evaluate.py and is frozen: it still
produces 12 TP / 18 FN / 7 FP / 33 TN. This script is the pose-based successor.
It reuses the same dataset layout (adl-* = normal, fall-* = fall subfolders of
PNG frames) and the same persistence/sampling parameters, but replaces the
aspect-ratio rule with the RTMPose torso-angle rule in services/perception/fall.py.

Usage (run from project root):
    PYTHONPATH=. python evals/evaluate_pose.py <folder_with_all_sequences> [out.json]
"""
import json
import os
import sys
import time

import cv2

from services.perception.l2 import FallPersistenceTracker, L2Perception

# Same parameters as the aspect-ratio baseline, for a like-for-like comparison.
PERSIST_FRAMES = 5
SAMPLE_EVERY = 3
CONF = 0.4

PROGRESS_EVERY = 10


def sequence_has_fall(folder, l2, persist=PERSIST_FRAMES, sample_every=SAMPLE_EVERY):
    """Apply the pose-fall rule to one sequence of PNG frames. Return True/False."""
    pngs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    pngs = pngs[::sample_every]
    tracker = FallPersistenceTracker(persist=persist)
    for name in pngs:
        frame = cv2.imread(os.path.join(folder, name))
        result = l2.process_frame(frame)
        if tracker.update(result.any_fall):
            return True
    return False


def _build_l2():
    from services.perception.detection import Detector
    from services.perception.pose import PoseEstimator

    return L2Perception(Detector(conf=CONF), PoseEstimator())


def main(frames_root, out_json=None):
    l2 = _build_l2()
    folders = sorted(
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d)) and ("adl" in d or "fall" in d)
    )

    tp = fp = tn = fn = 0
    rows = []
    t0 = time.time()
    for i, d in enumerate(folders, 1):
        truth_is_fall = d.startswith("fall")
        predicted = sequence_has_fall(os.path.join(frames_root, d), l2)
        if truth_is_fall and predicted:
            tp += 1; tag = "TP"
        elif truth_is_fall and not predicted:
            fn += 1; tag = "FN (missed a real fall)"
        elif not truth_is_fall and predicted:
            fp += 1; tag = "FP (false alarm)"
        else:
            tn += 1; tag = "TN"
        rows.append({"sequence": d, "truth": "fall" if truth_is_fall else "normal",
                     "predicted": "fall" if predicted else "normal", "outcome": tag})
        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"[{i}/{len(folders)}] {elapsed:.0f}s elapsed — "
                  f"so far: TP={tp} FN={fn} FP={fp} TN={tn}", flush=True)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n{'sequence':<26}{'truth':<9}{'predicted':<11}outcome")
    print("-" * 70)
    for r in rows:
        print(f"{r['sequence']:<26}{r['truth']:<9}{r['predicted']:<11}{r['outcome']}")
    print("-" * 70)
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"Precision={precision:.1%}  Recall={recall:.1%}  F1={f1:.1%}")
    print(f"Total elapsed: {time.time() - t0:.0f}s")

    result = {
        "dataset": "UR Fall",
        "model": "RTMPose-m + YOLOv8n",
        "params": {"persist_frames": PERSIST_FRAMES, "sample_every": SAMPLE_EVERY, "conf": CONF},
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "sequences": rows,
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_json}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("usage: python evals/evaluate_pose.py <folder_with_all_sequences> [out.json]")
        raise SystemExit(2)
    out = sys.argv[2] if len(sys.argv) == 3 else None
    main(sys.argv[1], out_json=out)
