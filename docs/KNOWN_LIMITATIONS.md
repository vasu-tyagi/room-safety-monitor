# Known limitations

Each item describes the current state, why the limitation exists, and what production would do differently.

---

## Per-stage latency not instrumented

The pipeline does not record how long each layer takes. The `/metrics` endpoint returns aggregate frame counts and a gate filter rate, but not per-stage p50/p95 latency. Production would instrument each pipeline stage with Prometheus histograms and display them on a Grafana dashboard; the FastAPI `/metrics` endpoint would switch from JSON to Prometheus exposition format. This is tracked as future work so the current implementation is not burdened by an observability stack that would complicate the demo setup.

## Hourly severity buckets limited to last 24 hours

The `/metrics` endpoint returns `incidents_by_severity_24h` bucketed over the last 24 hours with no sub-hour granularity. A production deployment would store per-minute or per-hour counts in a time-series database (TimescaleDB or InfluxDB) and expose a configurable time window. The current Postgres query re-scans the incidents table on every metrics poll, which is acceptable at demo scale but would require a materialized view or a separate counter table at production volume.

## SlowFast action recognition is opt-in, not in the default pipeline

`services/perception/l2.py` loads the SlowFast model only when a 32-frame clip is available in the track history. For short videos or the first few seconds of a stream, SlowFast does not run and its weight (0.20) is redistributed proportionally across the other sources by the confidence fusion node. This means the effective VLM weight rises to ~0.50 in practice during early-stream processing. Production would maintain a rolling clip buffer per track so SlowFast always has enough frames; the current ring buffer is sized at 32 frames (CLIP_BUFFER_LEN=90 for replay, but the L2 history deque is maxlen=32).

## Kinetics-400 weights substituted for Kinetics-700

The SlowFast model runs `slowfast_r50` pretrained on Kinetics-400. Kinetics-700 has broader action coverage and better performance on the `falling` and `fighting` categories relevant to this application. The K700 weights are gated behind a Hugging Face access request and were not available during development. The K400 label space does not include a clean `falling` class; we map a curated set of K400 action names to `{falling, fighting, running}` in `services/perception/l2.py`. Swapping in K700 weights is a one-line model path change once access is granted.

## Demo runs on CPU; production path uses GPU-accelerated inference

All three L2 models (YOLOv8n, RTMPose, SlowFast) run in-process on CPU via PyTorch and ONNX Runtime. The <=50 ms/frame latency target is not met on this hardware. A production deployment would serve L2 models with TensorRT-optimized engines on a server-grade GPU, which achieves the latency target. The code structure is compatible with GPU serving clients; switching from in-process inference to GPU-accelerated serving is isolated to `services/perception/l2.py`.

## Per-frame perception artifacts not stored (Inspector limitation)

The Inspector page (`/incidents/[id]/inspect`) shows the VLM prompt, fired gate rules, FSM audit trail, and KB matches for each incident. It does not show the raw L2 outputs (keypoint coordinates, bounding boxes, action logits) for the individual frames that triggered the gate. These artifacts are computed during `process_video` but not persisted. Storing them would require a per-frame artifact table or an object store entry per incident. Production would write a JSON artifact blob to MinIO alongside the evidence clip and retrieve it for the Inspector view.

## In-memory counters reset on service restart

The `_counters` dict in `services/service_plane/app.py` (frames processed, gate filter rate, incident count) is lost on every uvicorn restart. This means the Metrics page shows zeros after a restart even if the database has thousands of incidents. The DB-computed fields (`incidents_total`, `alerts_last_hour`, `operator_decisions`) survive restarts because they query Postgres. The production fix is to replace the in-memory dict with Prometheus counters persisted to a Pushgateway, or to compute all metrics from the DB on each poll.

## Multi-camera production deployment not implemented

The current system accepts a single `video_path` per `POST /process_video` call. There is no concurrent multi-camera ingest worker, no per-camera ByteTrack state isolation beyond a single call, and no camera-level configuration beyond `camera_id` and `room_id` passed in the request body. A production deployment would run one ingest worker process per camera (bounded by CPU/GPU budget), sharing a Postgres instance and a Redis alert bus. The data model is already multi-camera (every incident row stores `camera_id`); the missing piece is the worker orchestration layer.

## Area-as-age proxy is crude

The `unattended_minor_in_high_risk_zone` gate rule uses bounding box area (< 5000 px) as a proxy for detecting children. This works at a fixed camera height and close range but breaks when the camera is mounted high, the room is large, or the child is close to the camera. Production would add a face age estimation model as a separate L2 module (e.g., InsightFace AgeGender) and replace the area threshold with an explicit age prediction. The gate rule interface is unchanged; only the feature used to classify "minor" would change.

## Stub VLM fallback degrades alert quality during HF outages

When `VLM_MODE=auto` and the HF Inference Providers endpoint is unavailable, the pipeline falls back to a deterministic stub that returns a fixed label and confidence. The stub-caution rule in the agent (require gate rules + threshold+0.1) reduces false alerts, but it also reduces recall: real falls that the event gate correctly escalated will be dismissed if fused confidence does not clear the raised threshold. Production would use a dedicated HF Inference Endpoint (not the shared free tier) with an SLA, removing the availability dependency. A local fallback model (a smaller VL model running on-device) is the alternative when network access is restricted.
