"""Fall detection evaluation over the Le2i dataset.

Le2i contains AVI videos with per-video annotation files. Each annotation gives
the fall start and end frame (1-indexed). Scenes without annotation files
(Office, Lecture_room) are skipped automatically.

Dataset: http://le2i.cnrs.fr/?action=show&act=publications&id_publi=832
Format: 320x240 25 FPS AVI
Annotation: line 1 = fall_start frame, line 2 = fall_end frame (0,0 = no fall)

Usage (run from project root):
    PYTHONPATH=. python evals/evaluate_le2i.py <le2i_root> [out.json] [--conf-thr 0.3]
"""
import json
import os
import re
import sys
import time

from services.perception.l2 import FallPersistenceTracker
from services.perception.video import iter_frames_ffmpeg, video_dims

LE2I_FPS = 25
PERSIST_FRAMES = 5
SAMPLE_EVERY = 3
DETECTOR_CONF = 0.4
DEFAULT_CONF_THR = 0.3
PROGRESS_EVERY = 10



def parse_annotation(ann_path):
    """Return (fall_start, fall_end) from a Le2i annotation file.

    Returns (0, 0) for no-fall videos. Returns (None, None) if the file does
    not start with two integer lines — 3 files in the dataset have this defect
    (no fall start/end header, data starts directly with per-frame bbox rows).
    Callers should skip (None, None) entries.
    """
    with open(ann_path) as f:
        lines = f.read().strip().splitlines()
    try:
        return int(lines[0]), int(lines[1])
    except (ValueError, IndexError):
        return None, None


def video_eval(video_path, ann_path, l2, persist=PERSIST_FRAMES, sample_every=SAMPLE_EVERY):
    """Process one Le2i video. Returns (outcome, detection_frame_or_None).

    outcome is one of: "TP", "FN", "FP", "TN".
    detection_frame is the video frame number (1-indexed) when the tracker fires,
    or None if no fall was detected.
    """
    fall_start, fall_end = parse_annotation(ann_path)
    truth_is_fall = fall_start != 0 or fall_end != 0

    w, h = video_dims(video_path)
    tracker = FallPersistenceTracker(persist=persist)
    detection_frame = None

    for frame_num, frame in iter_frames_ffmpeg(video_path, w, h, sample_every=sample_every):
        result = l2.process_frame(frame)
        if tracker.update(result.any_fall):
            detection_frame = frame_num
            break

    predicted_fall = detection_frame is not None
    if truth_is_fall and predicted_fall:
        outcome = "TP"
    elif truth_is_fall and not predicted_fall:
        outcome = "FN"
    elif not truth_is_fall and predicted_fall:
        outcome = "FP"
    else:
        outcome = "TN"

    return outcome, detection_frame


def _video_number(name):
    """Extract the integer from 'video (N).ext'."""
    m = re.search(r"\((\d+)\)", name)
    return int(m.group(1)) if m else -1


def _find_scene_dirs(le2i_root):
    """Yield (scene_name, videos_dir, annotations_dir) for annotated scenes."""
    for top in sorted(os.listdir(le2i_root)):
        top_path = os.path.join(le2i_root, top)
        if not os.path.isdir(top_path):
            continue
        # double-nested: SceneName/SceneName/Videos + SceneName/SceneName/Annotation_files
        inner = os.path.join(top_path, top)
        if not os.path.isdir(inner):
            continue
        videos_dir = os.path.join(inner, "Videos")
        if not os.path.isdir(videos_dir):
            continue
        ann_dir = None
        for candidate in ("Annotation_files", "Annotations_files"):
            p = os.path.join(inner, candidate)
            if os.path.isdir(p):
                ann_dir = p
                break
        if ann_dir is None:
            continue
        yield top, videos_dir, ann_dir


def _build_l2(conf_thr):
    from services.perception.detection import Detector
    from services.perception.l2 import L2Perception
    from services.perception.pose import PoseEstimator

    return L2Perception(Detector(conf=DETECTOR_CONF), PoseEstimator(), conf_thr=conf_thr)


def main(le2i_root, out_json=None, conf_thr=DEFAULT_CONF_THR):
    l2 = _build_l2(conf_thr)

    rows = []
    tp = fp = tn = fn = skipped = 0
    ttd_frames = []  # time-to-detect for TPs, in frames relative to fall_start
    processed = 0
    t0 = time.time()

    for scene_name, videos_dir, ann_dir in _find_scene_dirs(le2i_root):
        video_files = sorted(
            [f for f in os.listdir(videos_dir) if f.endswith(".avi")],
            key=_video_number,
        )
        for vf in video_files:
            vnum = _video_number(vf)
            ann_name = f"video ({vnum}).txt"
            ann_path = os.path.join(ann_dir, ann_name)
            if not os.path.isfile(ann_path):
                continue

            video_path = os.path.join(videos_dir, vf)
            fall_start, fall_end = parse_annotation(ann_path)
            if fall_start is None:
                print(f"SKIP {scene_name}/{vf}: annotation missing fall start/end header",
                      flush=True)
                skipped += 1
                continue

            outcome, detect_frame = video_eval(
                video_path, ann_path, l2,
                persist=PERSIST_FRAMES, sample_every=SAMPLE_EVERY,
            )

            ttd = None
            if outcome == "TP" and detect_frame is not None and fall_start > 0:
                ttd_frames.append(detect_frame - fall_start)
                ttd = round((detect_frame - fall_start) / LE2I_FPS, 2)

            if outcome == "TP":
                tp += 1
            elif outcome == "FN":
                fn += 1
            elif outcome == "FP":
                fp += 1
            else:
                tn += 1

            processed += 1
            rows.append({
                "scene": scene_name,
                "video": vf,
                "fall_start": fall_start,
                "fall_end": fall_end,
                "outcome": outcome,
                "detect_frame": detect_frame,
                "time_to_detect_sec": ttd,
            })
            if processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"[{processed}] {elapsed:.0f}s — TP={tp} FN={fn} FP={fp} TN={tn}", flush=True)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_ttd = sum(ttd_frames) / len(ttd_frames) if ttd_frames else None

    print(f"\n{'scene':<22}{'video':<20}{'fall_start':<12}{'outcome'}")
    print("-" * 72)
    for r in rows:
        print(f"{r['scene']:<22}{r['video']:<20}{r['fall_start']:<12}{r['outcome']}"
              + (f"  ttd={r['time_to_detect_sec']:.1f}s" if r["time_to_detect_sec"] is not None else ""))
    print("-" * 72)
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}  skipped={skipped}")
    print(f"Precision={precision:.1%}  Recall={recall:.1%}  F1={f1:.1%}")
    if mean_ttd is not None:
        print(f"Mean time-to-detect (TPs): {mean_ttd / LE2I_FPS:.1f}s  ({mean_ttd:.1f} frames)")
    print(f"Total elapsed: {time.time() - t0:.0f}s")

    result = {
        "dataset": "Le2i",
        "model": "RTMPose-m + YOLOv8n",
        "params": {
            "persist_frames": PERSIST_FRAMES,
            "sample_every": SAMPLE_EVERY,
            "detector_conf": DETECTOR_CONF,
            "conf_thr": conf_thr,
        },
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "skipped": skipped,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_ttd_frames": round(mean_ttd, 2) if mean_ttd is not None else None,
        "mean_ttd_sec": round(mean_ttd / LE2I_FPS, 2) if mean_ttd is not None else None,
        "sequences": rows,
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_json}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("le2i_root")
    parser.add_argument("out_json", nargs="?", default=None)
    parser.add_argument("--conf-thr", type=float, default=DEFAULT_CONF_THR)
    args = parser.parse_args()
    main(args.le2i_root, out_json=args.out_json, conf_thr=args.conf_thr)
