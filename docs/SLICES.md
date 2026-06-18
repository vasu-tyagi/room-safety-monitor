# Build plan: six-layer rebuild

Source of truth for the rebuild structure until `docs/ARCHITECTURE.md` is
written in Slice 9. The old four-tier design stays at `docs/architecture.md`
as a historical record of v0.5.

## Target architecture (six layers)

| Layer | Name | What it does |
|-------|------|--------------|
| L1 | Stream Ingest | Multicast RTSP receiver (per-camera workers, auto-reconnect), NVDEC H.264 decode (30 fps, up to 640x360), batcher + ByteTrack persistent identity, ROI crop per camera/room. |
| L2 | Fast Classical CV | Triton + TensorRT, <=50 ms/frame, every frame. Detection (YOLOv8: person/object/vehicle + IDs), Pose (RTMPose: 17 COCO keypoints, fall/contact cues from geometry), Action (SlowFast Kinetics-400: running/falling/fighting). |
| — | Event Gate | ~1% funnel, 99% filtered. Rules over L2: fall_pose_detected, action_in_target_set ({falling, fighting, running}), two_persons_close_proximity_sustained, person_count_exceeds_policy, rapid_motion_in_restricted_zone. |
| L3 | VLM Deep Analysis | Qwen 2.5 VL, 5 s to 1 min. Prompt-based classify ("Is the person in danger? Why?"), KB retrieval via pgvector before the VLM call, NL output (label + rationale + confidence). |
| L4 | AI Agent | LangGraph + rules. Policy rules (guards, schedules, zones), incident FSM (new -> alert -> resolved/dismissed), confidence calibration (CV + VLM + KB fusion with documented weights), KB write-back (no retraining). |
| L5 | Persistence + KB | 7-day local NVMe. Postgres + MinIO (incidents, clips, audit, indexed by time/cam/type), pgvector KB (prompts, reasons, embeddings), 30 s pre-incident ring buffer per camera attached to alerts. |
| L6 | Service Plane | FastAPI + WebSocket + dispatcher. REST API (incidents, search, feedback), Next.js review dashboard, alert dispatcher (webhook stub; SMS/Teams/ACC documented as production paths), feedback loop (operator label -> KB update, no retraining). |

## Slices

Build in order. Each slice must leave the system end-to-end runnable.

- [x] **Slice 1 — Skeleton.** Docker infra (Postgres+pgvector, MinIO, Redis), Incident schema, persistence model + Alembic migration, stub pipeline (video -> YOLOv8n -> stub incident), FastAPI service plane (`/health`, `/process_video`, `/incidents`).
- [x] **Slice 2 — L2 perception.** Add real pose (RTMPose) and action (SlowFast) alongside the existing YOLOv8 detection. Replace the aspect-ratio fall rule with pose geometry. Notes: RTMPose runs via rtmlib/ONNX (mmcv has no wheel for torch 2.12+cu130, no nvcc to source-build); SlowFast preprocessing is hand-rolled (pytorchvideo.transforms broken on torchvision 0.27); pose-fall eval results in `evals/results/pose_baseline.json`.
- [~] **Slice 3 — Tracker.** ByteTrack persistent identity in the L2 pipeline. Per-track fall persistence and pose history (maxlen=32). ROI crop deferred to Slice 9 (not in the ingest path yet).
- [ ] **Slice 4 — Event Gate.** Rules over L2 outputs that funnel ~1% of frames forward; everything else filtered. Reference rules plus two extensions:
  - `prolonged_inactivity_in_private_zone` — pose model shows minimal joint movement over N minutes in a private zone (bathroom, bedroom); possible medical event.
  - `unattended_minor_in_high_risk_zone` — a single small bounding box in a zone tagged kitchen/pool/garage with no adult-sized box nearby.
- [ ] **Slice 5 — Real VLM.** Qwen 2.5 VL deep analysis on gated clips, with the documented Hugging Face free-tier path and a stub fallback. Evidence clips written to MinIO.
- [ ] **Slice 6 — KB and memory.** pgvector knowledge base: store prompts/reasons/embeddings, retrieve similar prior incidents before the VLM call. Note: KB tests require real Postgres + pgvector; SQLite fixtures don't cover similarity search.
- [ ] **Slice 7 — Agent layer.** LangGraph agent: policy rules, incident FSM, confidence calibration (CV + VLM + KB fusion), KB write-back.
- [ ] **Slice 7.5 — Incident Replay.** `POST /incidents/{id}/replay` re-runs the original clip through the current pipeline state (current KB, current rules) and returns the new outcome alongside the original. Tests whether the growing KB changes decisions over time.
- [ ] **Slice 8 — Dashboard.** Next.js operator UI over the REST API + WebSocket alerts, with the operator feedback loop writing back to the KB. Includes:
  - **Pipeline Inspector** (`/incidents/[id]/inspect`): what every layer produced for one incident — frames with YOLO boxes, pose keypoints overlaid, action label + confidence, gate rules that fired, the full VLM prompt sent (including KB context), the raw VLM response, the agent's policy decisions, confidence fusion breakdown with per-source weights, and final outcome.
  - **Operational Metrics** (`/metrics`): live stats — frames processed per camera per hour, event gate filter rate (target 99% filtered), per-layer latency distribution, VLM real-vs-stub call ratio, KB size over time, alerts per hour by severity, operator decision latency, confirmation vs dismissal ratio.
- [ ] **Slice 9 — Polish + docs + demo.** Write `docs/ARCHITECTURE.md`, finalize the real/simulated table, demo script and sample run.

Legend: `[ ]` not started, `[~]` in progress, `[x]` complete.
