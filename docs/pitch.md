# Room safety monitoring: one-page pitch

## Core idea

Running a powerful model on every frame from every camera does not scale. At 1,000 cameras and 5 fps that is 5,000 frames per second, which exceeds what any available GPU can process at the required latency, and the cost would be prohibitive anyway. The system instead uses a cascade: cheap analysis runs on every camera continuously, and expensive analysis fires only on the rare events a cheaper tier has already flagged. Cost and compute scale with events, not with camera count. A room that is empty or quiet costs almost nothing to monitor.

## The four tiers

**Tier 0** runs YOLOv8n (6 MB, COCO person class) on a Jetson Orin Nano at the edge, sampling every camera at ~5 fps. It detects and counts people. Frames with no one present are discarded immediately. When people are present, it evaluates two rule-based checks: off-hours presence and overcrowding. Both checks produce alerts directly without involving the GPU server.

**Tier 1** takes the bounding boxes Tier 0 already produced and computes one number: height divided by width of the largest detected person. A standing person has a tall, narrow box (ratio above 1). A person on the floor has a wide, low box (ratio below 1). If the ratio stays below 1.0 for 5 or more consecutive sampled frames, Tier 1 flags a fall candidate, buffers the last 10 seconds of frames, and sends the clip to the central server.

**Tier 2** runs Qwen2-VL on the candidate clip, spending about 5 seconds per clip on an A100 or L4 GPU. It confirms or rejects the fall and writes a natural-language description of what it observed. That description is what staff need to decide how to respond; a binary label alone is not enough. This tier is simulated in the current demo.

**Tier 3** deduplicates events from adjacent cameras covering the same room, fuses Tier 0 rule outputs with Tier 2 results into a single severity score, and places the alert in a priority queue for the reviewer console. It uses deterministic rules rather than ML so that every alert can be explained. This tier is simulated in the current demo.

## Sizing

One Jetson Orin Nano handles 10-30 cameras; a 1,000-camera deployment needs 33-100 edge nodes. For the central GPU, falls are rare: at average occupancy across 1,000 cameras, approximately one candidate clip arrives every 7 seconds. Each clip takes ~5 seconds to process, giving ~71% average utilisation on one A100 or L4 GPU. Two to three GPUs cover burst and failover. The T4 GPU was evaluated and rejected: at 15-20 seconds per clip it exceeds the 30-second end-to-end latency target for safety events.

## Honest eval finding

The Tier 1 aspect-ratio rule was evaluated on 60 sequences of the UR Fall dataset (threshold ratio 1.0, persistence 5 frames, sampled every 3 frames). Result: 12 TP, 18 FN, 7 FP, 33 TN. Precision ~63%, recall ~40%.

The 40% recall is not a surprise and not something to hide. Falls toward the camera, partial occlusions, and slow sinks do not produce a wide bounding box, so the rule misses them. Those 18 missed falls are exactly why Tier 2 exists: a VLM reviewing the full clip catches many of the cases the aspect-ratio rule cannot. A pose model at Tier 1 (the first two-weeks item) would close the gap further.

## What is real, what is simulated

Real and running: YOLOv8n person detection, the bounding-box aspect-ratio fall signal, and the evaluation across all 60 UR Fall sequences. The scripts are in `src/` and the CSV logs are in `data-results/`.

Simulated: Tier 2 VLM confirmation, Tier 3 deduplication and fusion, multi-camera orchestration, room IDs, timestamps, the off-hours clock, and the overcrowding threshold. These are shown in the reviewer console with clear labels.

## Two more weeks

RTMPose or MoveNet at Tier 1 to address the recall gap. Real Qwen2-VL integration to validate the 5 s/clip assumption. Tier 3 logic in code and tested against synthetic event streams. End-to-end test on a physical multi-camera setup.
