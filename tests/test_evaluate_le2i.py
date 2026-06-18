"""Tests for the Le2i eval: annotation parsing, scene discovery, and sequence eval."""
import cv2
import numpy as np
import pytest

from evals.evaluate_le2i import _find_scene_dirs, parse_annotation, video_eval


def _write_annotation(path, fall_start, fall_end, n_frames=10):
    """Write a minimal Le2i annotation file."""
    lines = [str(fall_start), str(fall_end)]
    for i in range(1, n_frames + 1):
        lines.append(f"{i},1,100,50,160,120")
    path.write_text("\n".join(lines))


class _Res:
    def __init__(self, any_fall):
        self.any_fall = any_fall


class AlwaysFall:
    def process_frame(self, frame):
        return _Res(True)


class NeverFall:
    def process_frame(self, frame):
        return _Res(False)


def _write_avi(path, n_frames=40, w=80, h=60, fps=25):
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
    )
    for _ in range(n_frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()


# ---- scene discovery ----

def _scaffold_scene(root, scene_name, ann_dir_name="Annotation_files", n_videos=2):
    """Create a fake Le2i nested scene: root/<scene>/<scene>/Videos/ + Annotation_files/."""
    inner = root / scene_name / scene_name
    (inner / "Videos").mkdir(parents=True)
    (inner / ann_dir_name).mkdir()
    for i in range(1, n_videos + 1):
        (inner / "Videos" / f"video ({i}).avi").touch()
        ann = inner / ann_dir_name / f"video ({i}).txt"
        ann.write_text(f"10\n20\n1,1,100,50,160,120\n")
    return inner


def test_find_scene_dirs_discovers_annotated_scenes(tmp_path):
    _scaffold_scene(tmp_path, "Home_01")
    _scaffold_scene(tmp_path, "Coffee_room_02", ann_dir_name="Annotations_files")
    # Office scene has no annotation dir — should be skipped
    office_inner = tmp_path / "Office" / "Office"
    (office_inner / "Videos").mkdir(parents=True)
    (office_inner / "Videos" / "video (1).avi").touch()

    scenes = list(_find_scene_dirs(str(tmp_path)))
    names = {s[0] for s in scenes}
    assert names == {"Home_01", "Coffee_room_02"}


def test_find_scene_dirs_handles_both_annotation_spellings(tmp_path):
    _scaffold_scene(tmp_path, "Home_01", ann_dir_name="Annotation_files")
    _scaffold_scene(tmp_path, "Coffee_room_02", ann_dir_name="Annotations_files")
    scenes = list(_find_scene_dirs(str(tmp_path)))
    assert len(scenes) == 2


# ---- annotation parsing ----

def test_parse_annotation_fall(tmp_path):
    ann = tmp_path / "video (1).txt"
    _write_annotation(ann, fall_start=48, fall_end=80)
    start, end = parse_annotation(ann)
    assert start == 48
    assert end == 80


def test_parse_annotation_no_fall(tmp_path):
    ann = tmp_path / "video (1).txt"
    _write_annotation(ann, fall_start=0, fall_end=0)
    start, end = parse_annotation(ann)
    assert start == 0
    assert end == 0


def test_parse_annotation_missing_header_returns_none(tmp_path):
    """3 files in the dataset omit the fall start/end header — skip gracefully."""
    ann = tmp_path / "video (1).txt"
    ann.write_text("1,1,72,58,132,170\n2,1,72,58,132,170\n")
    start, end = parse_annotation(ann)
    assert start is None
    assert end is None


# ---- video_eval: outcome and detection frame ----

def test_video_eval_tp_when_always_fall(tmp_path):
    vid = tmp_path / "video (1).avi"
    ann = tmp_path / "video (1).txt"
    _write_avi(vid, n_frames=60)
    _write_annotation(ann, fall_start=20, fall_end=40)
    outcome, detect_frame = video_eval(vid, ann, AlwaysFall(), persist=3, sample_every=1)
    assert outcome == "TP"
    assert detect_frame is not None


def test_video_eval_fn_when_never_fall(tmp_path):
    vid = tmp_path / "video (1).avi"
    ann = tmp_path / "video (1).txt"
    _write_avi(vid, n_frames=60)
    _write_annotation(ann, fall_start=20, fall_end=40)
    outcome, detect_frame = video_eval(vid, ann, NeverFall(), persist=3, sample_every=1)
    assert outcome == "FN"
    assert detect_frame is None


def test_video_eval_tn_for_no_fall_video(tmp_path):
    vid = tmp_path / "video (1).avi"
    ann = tmp_path / "video (1).txt"
    _write_avi(vid, n_frames=60)
    _write_annotation(ann, fall_start=0, fall_end=0)
    outcome, detect_frame = video_eval(vid, ann, NeverFall(), persist=3, sample_every=1)
    assert outcome == "TN"
    assert detect_frame is None


def test_video_eval_fp_for_no_fall_video_with_false_alarm(tmp_path):
    vid = tmp_path / "video (1).avi"
    ann = tmp_path / "video (1).txt"
    _write_avi(vid, n_frames=60)
    _write_annotation(ann, fall_start=0, fall_end=0)
    outcome, detect_frame = video_eval(vid, ann, AlwaysFall(), persist=3, sample_every=1)
    assert outcome == "FP"
    assert detect_frame is not None
