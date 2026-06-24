# Architecture: six-layer cascade

## Overview

Cheap perception runs on every frame; expensive analysis runs only on the ~1% of frames an event gate has flagged. This keeps compute and cost proportional to events, not to camera count.

---

## Layer cascade

```
  Video file / RTSP stream
          |
          v
  +---------------------+
  |  L1   Stream Ingest |  ByteTrack persistent identity, 30 fps decode,
  |                     |  batcher, ROI crop per camera/room (stub in demo)
  +---------------------+
          |
          v
  +---------------------+
  |  L2   Fast CV       |  YOLOv8n detection (person/vehicle/object + IDs)
  |                     |  RTMPose: 17 COCO keypoints, torso-angle fall rule
  |                     |  SlowFast Kinetics-400: running/falling/fighting
  |                     |  Target: <=50 ms/frame (CPU in demo; TensorRT in prod)
  +---------------------+
          |
     [ Event Gate ]  <-- ~1% of frames pass; 99% filtered here
     7 deterministic rules over L2 outputs:
       fall_pose_detected (N=3 persistence)
       action_in_target_set {falling, fighting, running}
       two_persons_close_proximity_sustained
       person_count_exceeds_policy
       rapid_motion_in_restricted_zone
       prolonged_inactivity_in_private_zone
       unattended_minor_in_high_risk_zone
          |
          v
  +---------------------+
  |  L3   VLM Analysis  |  Qwen 2.5 VL 72B via HF Inference Providers
  |                     |  pgvector KB context (top-3, cosine >= 0.7)
  |                     |  injected into prompt before each call
  |                     |  Output: label + rationale + confidence
  +---------------------+
          |
          v
  +---------------------+
  |  L4   AI Agent      |  LangGraph 6-node StateGraph:
  |                     |    parse_vlm_output -> policy_check ->
  |                     |    confidence_fusion -> decide ->
  |                     |    kb_writeback -> persist
  |                     |  Incident FSM: new -> alert | dismissed
  |                     |  Every transition logged to incident_audit
  +---------------------+
          |
          v
  +---------------------+
  |  L5   Persistence   |  Postgres: incidents, incident_audit
  |       + KB          |  pgvector: KB entries (768-dim, HNSW index)
  |                     |  MinIO: evidence clips (local fs in demo)
  |                     |  30 s pre-incident ring buffer per camera
  +---------------------+
          |
          v
  +---------------------+
  |  L6   Service Plane |  FastAPI REST + WebSocket /ws/alerts
  |                     |  Next.js 14 operator dashboard
  |                     |  Operator feedback -> KB write-back
  |                     |  Incident replay (dry-run re-inference)
  +---------------------+
```

---

## Layer status (current build)

| Layer | Name | Implementation | Status |
|-------|------|---------------|--------|
| L1 | Stream Ingest | ByteTrack (supervision), video file input | Real (RTSP substituted by file path) |
| L2 | Fast CV | YOLOv8n + RTMPose/ONNX + SlowFast | Real (CPU; TensorRT deferred) |
| Gate | Event Gate | 7 deterministic rules, N=3 persistence | Real |
| L3 | VLM Analysis | Qwen 2.5 VL 72B via HF; stub fallback | Real (stub when HF rate-limited) |
| L4 | AI Agent | LangGraph 6-node graph, policy YAML | Real |
| L5 | Persistence + KB | Postgres + pgvector HNSW + MinIO | Real (clips to local fs) |
| L6 | Service Plane | FastAPI + Next.js 14 | Real (WebSocket in-memory) |

---

## Data flow: single frame end-to-end

1. `POST /process_video` receives a video path, camera ID, and room ID.
2. L1 opens the file, decodes frames, and assigns ByteTrack IDs.
3. L2 runs YOLOv8n detection, RTMPose pose estimation, and (optionally) SlowFast action recognition on each frame.
4. The event gate evaluates the 7 rules against each frame's L2 outputs. Frames that fail all rules are dropped.
5. Escalated frames are packaged with their track history and sent to L3.
6. L3 retrieves up to 3 similar past incidents from the pgvector KB and calls Qwen 2.5 VL with the augmented prompt.
7. L4 parses the VLM output, applies facility policy rules, fuses confidence scores from all sources, and decides alert or dismiss.
8. L5 writes the incident row (+ audit entry) and saves the evidence clip.
9. L6 broadcasts the alert over WebSocket to all connected dashboard clients.
10. The operator reviews, confirms or dismisses, and the feedback writes a KB entry for future retrieval.

---

## Confidence fusion

Weighted sum of five sources, with proportional weight redistribution for any source that did not run:

| Source | Default weight |
|--------|---------------|
| YOLO detection confidence | 0.10 |
| Pose geometry | 0.20 |
| SlowFast action | 0.20 |
| VLM output | 0.40 |
| KB similarity match | 0.10 |

Weights are per-facility and loaded from `config/policies/<facility_id>.yaml`.

---

## Production vs demo differences

| Aspect | Demo | Production |
|--------|------|-----------|
| Input | Video file path | RTSP multicast, hardware-accelerated decode |
| L2 serving | In-process CPU | GPU-accelerated inference, <=50 ms/frame |
| WebSocket backend | In-memory | Redis pub/sub |
| Evidence clips | Local filesystem | MinIO object store |
| Metrics | In-memory counters | Prometheus + Grafana |
| VLM rate limits | Stub fallback | Dedicated HF Inference endpoint |

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the full list.

---

## Related documents

- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — why each implementation choice was made
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — what is not production-ready and why
- [EVAL_RESULTS.md](EVAL_RESULTS.md) — fall detection numbers on UR Fall and Le2i datasets
- [SLICES.md](SLICES.md) — build plan and slice completion status
