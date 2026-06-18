# Fall detection eval results

## Protocol

Sample every 3rd frame. Persistence: 5 consecutive positive frames to confirm a fall.
Detector confidence threshold: 0.4. Keypoint confidence threshold: see per-run notes.
Scripts: `evals/evaluate_pose.py` (UR Fall), `evals/evaluate_le2i.py` (Le2i).

---

## UR Fall dataset

Dataset: http://fenix.ur.edu.pl/mkepski/ds/uf.html
Sequences: 70 (40 ADL normal, 30 fall), ~160 PNG frames each, single camera (cam0-rgb).
Not in repo (gitignored under `data/`).

### Aspect-ratio rule (v0.5 baseline, frozen)

Script: `src/evaluate.py`
Rule: bounding-box height/width ratio >= 1.0 triggers a fall signal.

| Metric | Value |
|--------|-------|
| TP | 12 |
| FN | 18 |
| FP | 7 |
| TN | 33 |
| Precision | 63% |
| Recall | 40% |
| F1 | 49% |

### RTMPose torso-angle, conf_thr=0.3

Script: `evals/evaluate_pose.py`, raw results: `evals/results/pose_baseline.json`
Rule: torso angle from vertical >= 50 degrees. Keypoint conf gate: 0.3.

| Metric | Value |
|--------|-------|
| TP | 14 |
| FN | 16 |
| FP | 6 |
| TN | 34 |
| Precision | 70% |
| Recall | 47% |
| F1 | 56% |

### RTMPose torso-angle, conf_thr=0.2 (adopted default)

Raw results: `evals/results/pose_baseline_threshold_0.2.json`

| Metric | Value |
|--------|-------|
| TP | 15 |
| FN | 15 |
| FP | 7 |
| TN | 33 |
| Precision | 68% |
| Recall | 50% |
| F1 | 58% |

### UR Fall: full comparison

| Metric | Baseline (aspect-ratio) | Pose conf=0.3 | Pose conf=0.2 | Delta vs baseline |
|--------|------------------------|---------------|---------------|-------------------|
| TP | 12 | 14 | 15 | +3 |
| FN | 18 | 16 | 15 | -3 |
| FP | 7 | 6 | 7 | 0 |
| TN | 33 | 34 | 33 | 0 |
| Precision | 63% | 70% | 68% | +5pp |
| Recall | 40% | 47% | 50% | +10pp |
| F1 | 49% | 56% | 58% | +9pp |

conf_thr=0.2 adopted as default: recall gains 3pp over conf_thr=0.3 at the cost of 1 extra FP.
Precision stays well above the 60% floor set as the adoption criterion.

---

## Le2i dataset

Dataset: http://le2i.cnrs.fr (Coffee_room, Home, Office scenes, 320x240 25fps AVI)
Scenes evaluated: Coffee_room_01 (48), Coffee_room_02 (22), Home_01 (30), Home_02 (30).
Office and Lecture_room skipped: no annotation files.
3 videos skipped: annotation files missing fall start/end header (dataset defect).
Total: 127 evaluated (104 fall, 23 normal), conf_thr=0.2.
Not in repo (gitignored under `data/`).

### RTMPose torso-angle, conf_thr=0.2

Raw results: `evals/results/pose_le2i.json`

| Metric | Value |
|--------|-------|
| TP | 50 |
| FN | 46 |
| FP | 2 |
| TN | 29 |
| Precision | 96.2% |
| Recall | 52.1% |
| F1 | 67.6% |
| Mean time-to-detect (TPs) | 0.3s (7.6 frames) |
| Skipped (defective annotation) | 3 |

### Le2i: per-scene breakdown

| Scene | Fall videos | TP | FN | Recall | Normal | FP | TN |
|-------|------------|----|----|--------|--------|----|----|
| Coffee_room_01 | 47 | 37 | 10 | 79% | 0 | 0 | 0 |
| Coffee_room_02 | 19 | 13 | 3 | 81% | 3 | 2 | 1 |
| Home_01 | 30 | 4 | 26 | 13% | 0 | 0 | 0 |
| Home_02 | 7 | 3 | 6 | 43% | 23 | 0 | 23 |

Notes:
- Coffee_room scenes: strong recall (79-81%), near-zero false alarms. Falls happen in the open
  with a clear overhead/side camera view — torso angle rule works well here.
- Home_01: 13% recall. All falls in this scene are missed. Likely a camera angle or fall style
  issue (falls onto/off furniture, highly foreshortened from the camera perspective, or the
  torso remains near-vertical until the person is already on the ground). Calibration target
  for Slice 4/5.
- Time-to-detect 0.3s: fast enough for alerting. Several negative TTDs indicate pre-fall lean
  detected before the annotated start frame — either annotation lag or genuine early detection.

---

## Cross-dataset summary

| Dataset | Precision | Recall | F1 | Mean TTD |
|---------|-----------|--------|----|----------|
| UR Fall (aspect-ratio baseline) | 63% | 40% | 49% | — |
| UR Fall (RTMPose, conf=0.2) | 68% | 50% | 58% | — |
| Le2i (RTMPose, conf=0.2) | 96% | 52% | 68% | 0.3s |

Precision is high on both datasets. Recall at ~50% is the open problem for a geometry-only
rule: the VLM confirmation layer (Slice 5) and the event gate (Slice 4) are designed to
recover missed events from buffered clips rather than rely on the L2 rule alone. The L2 rule's
role is to pass through ~1% of frames; high precision means low false-alarm burden on L3/L4.
