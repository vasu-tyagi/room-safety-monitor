"""Frame reading utilities that avoid OpenCV's bundled ffmpeg.

cv2.VideoCapture crashes (general protection fault in libc) on AVIs encoded
with rawvideo+mp3 — the codec combination used by the Le2i dataset. The
functions here shell out to system ffmpeg/ffprobe instead, which handles those
files correctly.
"""
import subprocess

import numpy as np


def video_dims(video_path):
    """Return (width, height) of video_path using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
         str(video_path)],
        capture_output=True, text=True, check=True,
    )
    w, h = map(int, result.stdout.strip().split('x'))
    return w, h


def iter_frames_ffmpeg(video_path, width, height, sample_every=1):
    """Yield (frame_num, bgr_array) for sampled frames via ffmpeg subprocess pipe.

    frame_num is 1-indexed and counts every frame in the file (not just
    yielded ones), so the caller can correlate with per-frame annotations.
    """
    frame_size = width * height * 3
    proc = subprocess.Popen(
        ['ffmpeg', '-i', str(video_path), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    frame_num = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame_num += 1
            if frame_num % sample_every == 0:
                yield frame_num, np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    finally:
        proc.stdout.close()
        proc.wait()
