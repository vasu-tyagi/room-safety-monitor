# Design decisions

One section per decision. Each explains what was chosen, what was considered, and why.

---

## RTMPose via rtmlib/ONNX instead of MMPose+CUDA

MMPose depends on mmcv, which has no prebuilt wheel for torch 2.12+cu130 and requires nvcc to build from source. rtmlib wraps the same RTMPose models as ONNX exports and runs on onnxruntime-cpu with no CUDA dependency. The accuracy is identical because the weights are the same; only the serving path differs. When the target hardware gains nvcc and a CUDA-compatible mmcv wheel, swapping rtmlib for MMPose is a one-file change in `services/perception/l2.py`.

## SlowFast hand-rolled preprocessing

`pytorchvideo.transforms` raises a shape error on torchvision 0.27 (the `UniformTemporalSubsample` output format changed in a minor release). Rather than pin an older torchvision and break ultralytics, we hand-roll the slow/fast pathway split and normalization directly in `services/perception/l2.py`. The hand-rolled code matches the reference preprocessing from the SlowFast paper (T=32 slow frames, T*alpha=32 fast frames, alpha=4) and passes the same input to the same model weights.

## ByteTrack from supervision

`supervision.ByteTrack` is the simplest zero-dependency ByteTrack wrapper available for Python 3.11. It was pinned below 0.30 because ByteTrack was deprecated in 0.28 and removed in 0.30. The alternative (the original ByteTrack repo) requires a custom CUDA build. For a demo system that processes sequential frames from a single file, supervision's ByteTrack is sufficient; a production multi-camera deployment would use a service-level tracker with cross-camera re-identification.

## pgvector + HNSW

The KB stores incident rationales and operator feedback as 768-dimensional sentence embeddings. pgvector's HNSW index gives approximate nearest-neighbor search in O(log n) with no separate vector database process. This keeps the stack to a single Postgres instance for both structured incident data and vector retrieval. The HNSW parameters (m=16, ef_construction=64) are defaults chosen for correctness over throughput; a production deployment with millions of KB entries would tune these and benchmark recall vs. latency.

## LangGraph for the agent layer

The agent makes a sequence of decisions (parse -> policy -> fuse -> decide -> write) where each step depends on the previous. LangGraph's StateGraph makes the data flow explicit and each node independently testable. The alternative (a single function with nested conditionals) would be harder to extend with new policy types or confidence sources. The 6-node linear graph is the simplest LangGraph topology that achieves this; no branching or loops are used in the current implementation.

## Per-track fall persistence design

A single frame where the torso angle crosses the threshold is not a reliable fall signal; it fires on people bending over. We require N=3 consecutive positive frames per track ID before the gate passes a `fall_pose_detected` event. N=3 was calibrated on Le2i Coffee_room, where it cut false alarms from 16% to 9.6% with 0% pre-fall recall loss. The N parameter lives in the gate config and can be tuned per facility. Per-track counting (not per-frame global counting) is essential so that one person bending does not reset the counter for a different person who is actually falling.

## Stub-caution rule for VLM

When the VLM ran in stub mode, its output is not evidence about the actual scene. If the agent used the stub confidence directly, it would alert on arbitrary frames that happened to pass the event gate. The caution rule requires that: (a) at least one gate rule fired, and (b) fused confidence exceeds the alert threshold plus 0.1. This keeps the stub path from generating false alerts during demos or HF outages, at the cost of reduced recall when the VLM is unavailable. The tradeoff is intentional: a false alert is more disruptive than a missed alert in the demo context.

## Confidence fusion weights and reasoning

The weights (yolo=0.10, pose=0.20, action=0.20, vlm=0.40, kb=0.10) reflect the relative reliability of each source. VLM is the highest because it reasons over the actual image with natural language; its weight is 4x pose to prevent the geometry rule from overriding a clear VLM rejection. Pose and action are equal because both are L2 signals with similar false-alarm rates on the calibration dataset. YOLO detection confidence is low (0.10) because it signals presence, not event type. KB weight (0.10) is a prior from past similar incidents and should not dominate when the current evidence is strong. Weights are loaded from policy YAML so they can be tuned per facility without code changes.

## Gate persistence N=3 calibration

The event gate uses a sliding window of N=3 consecutive frames before passing `fall_pose_detected`. The calibration was done on Le2i Coffee_room: at N=1 (no persistence) the false alarm rate on non-fall sequences was 16.0%; at N=3 it dropped to 9.6%; at N=5 in-fall recall dropped to 79.2% (acceptable but below the 80% target). N=3 was the inflection point where false alarms fell sharply without significant recall loss. The Coffee_room calibration does not generalize to Home scenes (which have very different camera angles), which is documented in EVAL_RESULTS.md.

## Area-as-age proxy for minor detection

The `unattended_minor_in_high_risk_zone` gate rule identifies minors by bounding box area (area < 5000 px at 640x480 = roughly a child-sized box). This is a deliberate approximation: a real face age classifier adds latency and a model dependency that is out of scope for this build. The area proxy works at close camera range with a fixed mounting height; it breaks when the camera is far away or the room is large. This is documented as a known limitation and the production replacement (a face age classifier as a separate L2 module) is described in KNOWN_LIMITATIONS.md.

## In-memory metrics vs Redis production path

Runtime counters (frames processed, gate filter rate) are held in a module-level dictionary in `services/service_plane/app.py`. This is the simplest implementation that satisfies the demo requirement: show a live counter on the metrics page. It resets on service restart and is not shared across uvicorn workers. The production path uses Prometheus counters exposed via `/metrics` (Prometheus format) and scraped by a Grafana instance. The current implementation's `/metrics` endpoint returns JSON; switching to Prometheus format is a one-endpoint change in `app.py`.
