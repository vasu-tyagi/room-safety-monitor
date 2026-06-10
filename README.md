# Room safety monitoring

A 4-tier cascade system for real-time room safety monitoring across 1000+ cameras. Cheap person detection runs on every camera continuously; expensive analysis runs only on the rare clips that a cheaper tier has already flagged.

## What is real and what is simulated

| Component | Status |
|-----------|--------|
| Tier 0: YOLOv8n person detection on UR Fall dataset | **Real** |
| Tier 1: bounding-box aspect-ratio fall signal | **Real** |
| Evaluation across 60 sequences (12 TP, 18 FN, 7 FP, 33 TN) | **Real** |
| Tier 2: VLM clip confirmation (Qwen2-VL) | Simulated |
| Tier 3: deduplication and fusion | Simulated |
| Multi-camera setup, room IDs, timestamps | Simulated |
| Off-hours clock, overcrowding threshold | Simulated |

## Deliverables

| File | Description |
|------|-------------|
| [docs/architecture.md](docs/architecture.md) | End-to-end architecture: diagram, data flow, deployment, scaling |
| [docs/design-writeup.md](docs/design-writeup.md) | Two-page write-up: choices, trade-offs, privacy, two-weeks plan |
| [docs/pitch.md](docs/pitch.md) | One-page pitch summary |
| [demo/console.html](demo/console.html) | Reviewer console: 3 scenarios + aspect-ratio chart |

## Source code

| File | Description |
|------|-------------|
| [src/detect.py](src/detect.py) | YOLOv8n person detector. Runs on a folder of PNG frames, outputs annotated images and a per-frame CSV log. |
| [src/evaluate.py](src/evaluate.py) | Evaluates the aspect-ratio fall rule across the full UR Fall dataset. |

## Running the code

```bash
# Activate the virtual environment
source venv/bin/activate

# Run detection on a folder of frames
python src/detect.py <folder_of_frames> <output_folder>

# Evaluate across the full UR Fall dataset
python src/evaluate.py <folder_with_all_sequences>
```

The `data-results/` folder contains pre-run CSV logs from one fall sequence (`fall-01-cam0-rgb`) and one normal sequence (`adl-01-cam0-rgb`), used in the demo chart.

## Evaluation results

Box aspect-ratio rule alone, 60 sequences (threshold 1.0, persistence 5 frames, sampled every 3 frames):

| Outcome | Count |
|---------|-------|
| True positives (real fall, caught) | 12 |
| False negatives (real fall, missed) | 18 |
| False positives (false alarm) | 7 |
| True negatives (normal, quiet) | 33 |
| Precision | ~63% |
| Recall | ~40% |

The 40% recall is the empirical reason the cascade requires Tier 2 confirmation or a pose model at Tier 1. The cheap rule is designed as a fast pre-filter, not the final decision.
