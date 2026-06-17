# Room safety monitoring

> Currently rebuilding to the six-layer reference architecture. See [docs/SLICES.md](docs/SLICES.md) for build progress.

A real-time room safety monitoring system for 1000+ cameras. Cheap perception runs on every frame; expensive analysis (VLM) runs only on the ~1% of frames an event gate has flagged.

## What is real, what is simulated, and why

Current state after Slice 1 (skeleton). The slice plan is in [docs/SLICES.md](docs/SLICES.md).

| Component | Status | Why |
|-----------|--------|-----|
| L2 person detection (YOLOv8n) | **Real** | Existing v0.5 code, reused as the pipeline detector. |
| Incident schema, Postgres model, Alembic migration | **Real** | Built this slice; incidents persist to Postgres. |
| Service plane API (`/health`, `/process_video`, `/incidents`) | **Real** | FastAPI app built this slice. |
| Docker infra: Postgres+pgvector, MinIO, Redis | **Real** | `deploy/docker-compose.yml`; containers run, MinIO/Redis not yet wired in. |
| Pipeline incidents (event_type, severity, rationale) | **Stub** | Skeleton emits a fake incident every 30th person frame. No real fall/action logic yet. |
| L2 pose (RTMPose) and action (SlowFast) | Not built | Slice 2. |
| L1 ingest, ByteTrack tracker | Not built | Slices 1 ingest is a file path; tracker is Slice 3. |
| Event gate | Not built | Slice 4. |
| L3 VLM deep analysis (Qwen 2.5 VL) | Not built | Slice 5. |
| L5 KB (pgvector), pre-incident buffer | Not built | Slices 5-6. |
| L4 agent (LangGraph), incident FSM, fusion | Not built | Slice 7. |
| L6 Next.js dashboard, WebSocket, feedback loop | Not built | Slice 8. |

### Legacy v0.5 (four-tier cascade)

The previous design and its evaluation remain valid and are recorded in [docs/architecture.md](docs/architecture.md):

| Component | Status |
|-----------|--------|
| Tier 0: YOLOv8n person detection on UR Fall dataset | **Real** |
| Tier 1: bounding-box aspect-ratio fall signal | **Real** |
| Evaluation across 60 sequences (12 TP, 18 FN, 7 FP, 33 TN) | **Real** |
| Tier 2: VLM clip confirmation (Qwen2-VL) | Simulated |
| Tier 3: deduplication and fusion | Simulated |

## Running the rebuild (Slice 1)

```bash
# Use Python 3.11 or 3.12 (not 3.14 - ML wheels lag).
source venv/bin/activate
pip install -r requirements.txt

# Start backing services
docker compose -f deploy/docker-compose.yml up -d

# Apply the database migration
alembic upgrade head

# Run the service plane
uvicorn services.service_plane.app:app --reload

# In another shell: process a video and list incidents
curl -X POST localhost:8000/process_video -H 'content-type: application/json' \
  -d '{"video_path": "demo/sample_videos/your_clip.mp4"}'
curl localhost:8000/incidents

# Run the tests
pytest
```

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
