# Fall detection eval results

Evaluation protocol: same for both models. 70 UR Fall sequences (40 ADL normal, 30 fall).
Sample every 3rd frame. Persistence: 5 consecutive positive frames to confirm a fall.
Detector confidence threshold: 0.4.

## UR Fall dataset

Dataset: http://fenix.ur.edu.pl/mkepski/ds/uf.html
Sequences: 70 (40 ADL normal, 30 fall), ~160 PNG frames each, single camera (cam0-rgb).
Not in repo (gitignored under `data/`).

### Aspect-ratio rule (v0.5 baseline, frozen)

Script: `src/evaluate.py`
Rule: bounding-box height/width ratio >= 1.0 triggers a fall signal.

| Metric | Value |
|--------|-------|
| TP (real fall, caught) | 12 |
| FN (real fall, missed) | 18 |
| FP (false alarm) | 7 |
| TN (normal, quiet) | 33 |
| Precision | 63% |
| Recall | 40% |
| F1 | 49% |

### RTMPose torso-angle rule (Slice 2)

Script: `evals/evaluate_pose.py`
Rule: torso angle from vertical >= 50 degrees (shoulder midpoint to hip midpoint).
Model: RTMPose-m, 17 COCO keypoints, via rtmlib/ONNX. Detector: YOLOv8n.
Raw results: `evals/results/pose_baseline.json`

| Metric | Value |
|--------|-------|
| TP (real fall, caught) | 14 |
| FN (real fall, missed) | 16 |
| FP (false alarm) | 6 |
| TN (normal, quiet) | 34 |
| Precision | 70% |
| Recall | 47% |
| F1 | 56% |

### Change vs baseline

| Metric | Baseline | Pose | Delta |
|--------|----------|------|-------|
| TP | 12 | 14 | +2 |
| FN | 18 | 16 | -2 |
| FP | 7 | 6 | -1 |
| TN | 33 | 34 | +1 |
| Precision | 63% | 70% | +7pp |
| Recall | 40% | 47% | +7pp |
| F1 | 49% | 56% | +7pp |

Recall at 47% is still low for a safety-critical use case. The 16 missed falls are the primary
calibration target for Slice 4 (event gate) and Slice 5 (VLM confirmation). The torso-angle
threshold (currently 50 degrees) and keypoint confidence gate (0.3) are the two levers.
The 6 false positives are ADL sequences with floor-level activity (adl-04, adl-05, adl-06,
adl-10, adl-17, adl-34) where the subject is prone or crouching — legitimate ambiguous cases
for a geometry-only rule.
