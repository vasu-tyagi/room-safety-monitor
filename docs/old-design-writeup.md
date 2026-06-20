# (Old) Design write-up: room safety monitoring

## Design choices

The system uses a cascade: cheap analysis runs on every camera continuously, and expensive analysis runs only on the events a cheaper tier has already flagged. The alternative, running a capable model on every frame from every camera at 5 fps across 1,000 cameras, would require processing 5,000 frames per second. No available GPU handles that at the required latency, and the cost would be prohibitive even if one could. The cascade makes cost proportional to events, not camera count. A room that is empty or normal consumes almost no GPU time.

A second design choice is to keep Tier 3 as deterministic logic rather than ML. Deduplication and severity fusion expressed as rules are fully auditable: a reviewer can always trace why a specific alert was sent and at what severity. An ML fusion layer would obscure that. For a system that may be cited in a welfare incident, explainability is not optional.

## Models and tools

Tier 0 uses YOLOv8n (6 MB weights, pretrained on COCO class 0: person) running on a Jetson Orin Nano at ~5 fps. It is the smallest YOLO variant fast enough to handle 10-30 cameras per node at the required rate.

Tier 1 uses the bounding-box aspect ratio (height / width of the largest detected person). This is arithmetic on the boxes Tier 0 already produced, not additional inference. The rule: if the ratio stays below 1.0 for 5 or more consecutive sampled frames, raise a fall candidate and buffer a 10-second clip.

Tier 2 uses Qwen2-VL, a vision-language model (simulated in this demo; not yet running on real clips). It was chosen over video action models (SlowFast, X3D) because it produces a natural-language description alongside the confirmation. Staff dispatching a welfare check need to know what the camera saw; a binary label does not give them that, and the description is an audit trail.

Tier 3 is deterministic rule logic (simulated in this demo): deduplication within a time window per room, severity ranking by event type, and priority queue insertion. No trained model is involved.

## Trade-offs

YOLOv8n sacrifices some detection accuracy for size and speed. The cascade is tolerant of occasional Tier 0 misses: a single non-detection resets the Tier 1 persistence counter rather than directly producing a false negative.

The aspect-ratio rule sacrifices recall for near-zero cost. Its 40% recall is acceptable at Tier 1 because Tier 2 is designed to catch what it misses.

Qwen2-VL requires ~5 seconds per clip on an A100 or L4 GPU. A T4 GPU takes 15-20 seconds per clip, pushing the end-to-end safety event latency past the 30-second budget. T4 was evaluated and rejected on this basis.

Deterministic Tier 3 logic cannot self-improve from reviewer feedback. It can be tuned manually. This is an acceptable cost for full explainability.

## Real-time performance and cost

End-to-end latency for a safety event: Tier 0+1 at the edge (~1-2 s), clip transfer over LAN (<1 s), Tier 2 inference (~5 s), Tier 3 routing (<1 s). Total: 7-10 seconds, within the 10-30 s target.

Occupancy alerts (off-hours, overcrowding) bypass Tier 2. Latency is 2-3 seconds, within the 5-15 s target.

GPU cost: at 1,000 cameras with average occupancy, approximately one candidate clip arrives every 7 seconds. Each clip takes ~5 seconds. Average utilisation is ~71% of one GPU. Two to three A100/L4 GPUs cover burst and failover. When no candidates are raised, Tier 2 costs nothing.

## False positive and false negative risks

The box aspect-ratio rule evaluated on 60 sequences of the UR Fall dataset (threshold 1.0, persistence 5 frames, sampled every 3): 12 TP, 18 FN, 7 FP, 33 TN. Precision ~63%, recall ~40%. This is the empirical reason Tier 2 is required. The 18 missed falls are sequences where the person did not produce a persistently wide bounding box: falls toward the camera, slides down a wall, partial occlusions, slow sinks where the person crouches before falling. Tier 2 reviewing the 10-second clip catches many of these. A pose model at Tier 1 would catch more.

False negatives are the primary safety risk: a person injured on the floor who does not receive help in time. The cascade addresses this at Tier 2, and a pose model at Tier 1 is the highest-priority two-weeks item for exactly this reason.

False positives cause alert fatigue. Reviewers who learn that most alerts are false alarms stop responding promptly to real ones. Tier 2 confirmation filters the queue before anything reaches a human. Tier 3 deduplication prevents the same event from generating multiple alerts from adjacent cameras. The raw ~63% precision of the Tier 1 rule improves significantly after both filters.

## Privacy and data handling

Raw video never leaves the edge node under normal operation. The 10-second rolling buffer is held in memory and overwritten continuously. Only a candidate clip travels to the central server, and only when Tier 1 raises a candidate. No persistent video storage is required for rooms where nothing is flagged.

Candidate clips sent to Tier 2 should be encrypted in transit and deleted from the central server after processing. The alert schema uses room and camera identifiers, not personal identifiers. The VLM description should cover posture and position only. Face recognition and identity inference are out of scope and must not be added without a separate privacy review and an appropriate legal basis.

## Two-weeks priorities

1. RTMPose or MoveNet at Tier 1. Directly addresses the 40% recall gap; the highest-impact single change.
2. Real Tier 2 integration. Deploy Qwen2-VL on a test GPU; validate the ~5 s/clip assumption with real video.
3. Tier 3 deduplication and fusion logic in code. Test cases (same fall seen by two cameras, burst of overcrowding alerts) matter more than the logic itself.
4. End-to-end test on a physical multi-camera setup. Real cameras, real edge node, real GPU, alert on the console.
5. Privacy review of VLM output. Confirm Qwen2-VL descriptions do not contain personal information that would require additional data handling controls.
