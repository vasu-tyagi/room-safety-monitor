# evaluate.py
# Runs the cheap fall rule across the whole UR Fall dataset and measures how good it is.
#
# It does NOT call YOLO again. It reuses the same idea as detect.py: for each sequence,
# run the detector, read the person box per frame, and apply one rule to decide
# "did this sequence contain a fall, yes or no." Then compare that decision to the
# truth (the folder name tells us: adl = normal, fall = fall) and count the outcomes.
#
# Usage:
#   python evaluate.py <folder_with_all_sequences>
# where the folder contains adl-01-cam0-rgb ... fall-30-cam0-rgb subfolders.

import sys, os
from ultralytics import YOLO

frames_root = sys.argv[1]
model = YOLO("yolov8n.pt")
PERSON = 0

# ---- THE FOUR DECISIONS YOU MUST BE ABLE TO DEFEND ----
# (1) The threshold. A box wider than tall (ratio < 1.0) means "on the floor".
FALL_RATIO = 1.0
# (2) Persistence. One wide frame is noise. We require the box to stay wide for
#     several consecutive sampled frames before we call it a fall. This is the
#     "crossed and stayed down" idea, not "dipped once".
PERSIST_FRAMES = 5
# (3) Sampling. We sample every Nth frame, mirroring the ~5fps the design uses,
#     so the eval reflects what the real system sees, not every raw frame.
SAMPLE_EVERY = 3
# (4) Confidence floor for a person detection, same as detect.py.
CONF = 0.4
# --------------------------------------------------------

def sequence_has_fall(folder):
    """Apply the cheap rule to one sequence. Return True if it looks like a fall."""
    pngs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    pngs = pngs[::SAMPLE_EVERY]
    consecutive_wide = 0
    for name in pngs:
        result = model(os.path.join(folder, name), classes=[PERSON], conf=CONF, verbose=False)[0]
        # take the largest person box in the frame
        best = None
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            if best is None or w * h > best:
                best = w * h
                ratio = h / w if w else 999
        if best is None:
            consecutive_wide = 0          # no person seen, reset
            continue
        if ratio < FALL_RATIO:
            consecutive_wide += 1
            if consecutive_wide >= PERSIST_FRAMES:
                return True                # stayed wide long enough -> call it a fall
        else:
            consecutive_wide = 0
    return False

# truth comes from the folder name: "fall-..." is a real fall, "adl-..." is normal
folders = sorted(d for d in os.listdir(frames_root)
                 if os.path.isdir(os.path.join(frames_root, d)) and ("adl" in d or "fall" in d))

tp = fp = tn = fn = 0
rows = []
for d in folders:
    truth_is_fall = d.startswith("fall")
    predicted_fall = sequence_has_fall(os.path.join(frames_root, d))
    if truth_is_fall and predicted_fall: tp += 1; tag = "TP"
    elif truth_is_fall and not predicted_fall: fn += 1; tag = "FN (missed a real fall)"
    elif not truth_is_fall and predicted_fall: fp += 1; tag = "FP (false alarm)"
    else: tn += 1; tag = "TN"
    rows.append((d, "fall" if truth_is_fall else "normal", "fall" if predicted_fall else "normal", tag))

print(f"{'sequence':<22}{'truth':<9}{'predicted':<11}outcome")
print("-" * 64)
for d, t, p, tag in rows:
    print(f"{d:<22}{t:<9}{p:<11}{tag}")

precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0

print("\n==== CONFUSION MATRIX ====")
print(f"True Positives  (real fall, caught) : {tp}")
print(f"False Negatives (real fall, MISSED) : {fn}")
print(f"False Positives (false alarm)       : {fp}")
print(f"True Negatives  (normal, quiet)     : {tn}")
print(f"\nPrecision (of alerts, how many real): {precision:.0%}")
print(f"Recall    (of real falls, how many caught): {recall:.0%}")
print(f"\nThreshold={FALL_RATIO}, persistence={PERSIST_FRAMES} frames, sampled every {SAMPLE_EVERY}.")
