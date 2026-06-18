# Room safety monitoring

> Currently rebuilding to the six-layer reference architecture. See [docs/SLICES.md](docs/SLICES.md) for build progress.

A real-time room safety monitoring system for 1000+ cameras. Cheap perception runs on every frame; expensive analysis (VLM) runs only on the ~1% of frames an event gate has flagged.

## What is real, what is simulated, and why

Current state after Slice 7.5 (Incident Replay). The slice plan is in [docs/SLICES.md](docs/SLICES.md).

| Component | Status | Why |
|-----------|--------|-----|
| L2 detection (YOLOv8n, person/vehicle/object) | **Real** | Existing v0.5 detector, wrapped with class grouping. |
| L2 pose (RTMPose, 17 COCO keypoints) | **Real** | Runs via rtmlib/ONNX on CPU. Verified on demo fall frames. |
| Pose-geometry fall detection | **Real** | Torso-angle rule over keypoints; replaces the aspect-ratio rule. |
| L2 action (SlowFast, Kinetics-400) | **Real (approx.)** | Real `slowfast_r50`; preprocessing hand-rolled. {running,falling,fighting} is a curated map over K400 names (K400 has no clean "falling"). |
| ByteTrack tracker | **Real** | supervision.ByteTrack; per-track fall persistence and pose history (maxlen=32). Pinned supervision<0.30 (removed in 0.30). |
| Event Gate | **Real** | 7 deterministic rules over L2 outputs. Room policies in `config/rooms.yaml`. `process_video` returns `ProcessVideoResult` with escalation metrics. |
| L3 VLM (Qwen 2.5 VL via HF) | **Real** (stub fallback) | Real Qwen 2.5 VL via HF Inference Providers. Mode controlled by `VLM_MODE` env var: `real` (token required, WARNING on failure), `auto` (token optional, INFO on fallback), `stub` (no network). Every call logs real vs stub and reason. Stub fallback is the demo's safety net against HF free-tier rate limits. |
| KB retrieval (pgvector, sentence-transformers) | **Real** | `services/kb/`: `all-mpnet-base-v2` (768-dim) embedder singleton, HNSW index in Postgres, cosine similarity search. Top-3 similar incidents (threshold=0.7) injected into VLM prompt on each escalated frame. |
| L4 agent (LangGraph) | **Real** | `services/agent/`: 6-node linear StateGraph (parse_vlm_output -> policy_check -> confidence_fusion -> decide -> kb_writeback -> persist). Replaces direct incident writes from the pipeline. |
| Policy engine | **Real** | YAML rules per facility in `config/policies/`. Three rule types: time_window_suppression, threshold_override, severity_filter. Default policy includes gym fall suppression (18:00-20:00), fall-risk threshold=0.5, bathroom high-severity-only. |
| Confidence fusion | **Real** | Weighted combination of yolo=0.1, pose=0.2, action=0.2, vlm=0.4, kb=0.1. Weights loaded from policy YAML per facility. Missing sources (SlowFast not run) redistribute weight proportionally. |
| Stub caution rule | **Real** | When VLM ran in stub mode, agent only alerts if gate rules fired AND fused confidence >= threshold+0.1. Prevents stub-derived false alerts. |
| Incident FSM | **Real** | Incidents transition new -> alert or new -> dismissed. Every transition logged to `incident_audit` table. |
| Unattended-minor rule | **Approximated** | Uses bbox area as age proxy (area < 5000px = minor). Real deployment needs a face age classifier. |
| Incidents + audit | **Real** | Agent persist node writes incident + audit rows to Postgres/SQLite. |
| Incident schema, Postgres model, Alembic migrations | **Real** | Slice 1+6+7; incidents, KB entries, and incident_audit persist to Postgres. |
| Service plane API (`/health`, `/process_video`, `/incidents`, `/incidents/{id}/replay`) | **Real** | Slice 1+7.5 FastAPI app. Replay re-runs a saved clip through the current pipeline in dry-run mode and returns a structured comparison. |
| Incident replay (`POST /incidents/{id}/replay`) | **Real** | `services/pipeline/replay.py`. Structured diff: state_changed, confidence_delta, rationale_changed. Clip saved to `clips/{incident_id}.mp4`. |
| Evidence clips written to MinIO | Not built | Clips saved to local filesystem (`clips/`). MinIO upload deferred to Slice 9. |
| Docker infra: Postgres+pgvector, MinIO, Redis | **Real** | `deploy/docker-compose.yml`; MinIO/Redis not yet wired in. |
| Triton + TensorRT serving, ≤50 ms/frame budget | **Substituted** | Models run in-process on CPU. No Triton/TensorRT; latency target not met on this hardware. |
| ROI crop per camera/room | Not built | Deferred to Slice 9 polish. |
| L1 ingest | Not built | Ingest is a file path; full RTSP ingest is Slice 9. |
| Evidence clips (local filesystem) | **Real** | Clips saved to `clips/{incident_id}.mp4` by agent persist node. MinIO upload deferred to Slice 9. |
| L6 Next.js dashboard, WebSocket, feedback loop | Not built | Slice 8. |

Pose-fall eval numbers over UR Fall are pending: the dataset is not in this repo. The aspect-ratio baseline (`src/evaluate.py`) is frozen and still reports 12 TP / 18 FN / 7 FP / 33 TN; the pose successor is `evals/evaluate_pose.py`.

Install note: mmpose/mmcv are deliberately not used. On this box (CPU-only, torch 2.12+cu130) mmcv has no prebuilt wheel and cannot source-build without nvcc. RTMPose runs through rtmlib/ONNX instead.

### Legacy v0.5 (four-tier cascade)

The previous design and its evaluation remain valid and are recorded in [docs/architecture.md](docs/architecture.md):

| Component | Status |
|-----------|--------|
| Tier 0: YOLOv8n person detection on UR Fall dataset | **Real** |
| Tier 1: bounding-box aspect-ratio fall signal | **Real** |
| Evaluation across 60 sequences (12 TP, 18 FN, 7 FP, 33 TN) | **Real** |
| Tier 2: VLM clip confirmation (Qwen2-VL) | Simulated |
| Tier 3: deduplication and fusion | Simulated |

## Running the rebuild

```bash
# Use Python 3.11 or 3.12 (not 3.14 - ML wheels lag).
source venv/bin/activate
pip install -r requirements.txt

# Start backing services
docker compose -f deploy/docker-compose.yml up -d

# Apply the database migration
alembic upgrade head

# Run the service plane (process_video now runs the L2 pose+action pipeline)
uvicorn services.service_plane.app:app --reload

# In another shell: process a video and list incidents
curl -X POST localhost:8000/process_video -H 'content-type: application/json' \
  -d '{"video_path": "demo/sample_videos/your_clip.mp4"}'
curl localhost:8000/incidents

# Pose-geometry fall evaluation (needs the UR Fall dataset; not in this repo)
python evals/evaluate_pose.py <folder_with_all_sequences>

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
