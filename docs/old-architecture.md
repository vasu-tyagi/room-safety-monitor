# (Old) Architecture: room safety monitoring

## Overview

The system is a 4-tier cascade. Cheap analysis runs on every camera all the time; expensive analysis runs only on the few events a cheaper tier has already flagged. This keeps compute and cost proportional to events, not to camera count. Most frames across most rooms are empty. Frames with no people are discarded at the edge in under a second and never reach the central server.

---


## Data flow

This traces a single camera frame end-to-end.

**Step 1: Frame capture (edge)**
A camera delivers a frame at ~5 fps. The edge node receives it in memory; no frame is stored to disk.

**Step 2: Tier 0: person detection (edge)**
YOLOv8n runs inference on the frame. It returns a list of bounding boxes for detected persons (class 0, confidence >= 0.4) and a total count.

- If count = 0: the frame is discarded. No further processing occurs. This is the common case in quiet facilities.
- If count >= 1: bounding boxes are passed to Tier 1. The count also triggers the occupancy rules check (step 2a).

**Step 2a: Occupancy rules (edge, parallel)**
If count > 0, two rules are evaluated immediately:

- Off-hours rule: if the local clock is outside permitted hours (e.g., 23:00-06:00), raise an off-hours event. Route directly to Tier 3.
- Overcrowding rule: if count exceeds the room's configured threshold, raise an overcrowding event. Route directly to Tier 3.

Neither rule requires Tier 2 confirmation. Both are deterministic. Both are simulated in the current demo.

**Step 3: Tier 1: aspect-ratio check (edge)**
For each frame where count >= 1, compute height/width of the largest detected person box.

- If ratio > 1.0 (tall box, person upright): no candidate. Continue frame-by-frame monitoring.
- If ratio < 1.0 (wide box, person horizontal): increment a consecutive-low-ratio counter.
- If the counter reaches 5 consecutive sampled frames below the threshold: flag as a fall candidate. Retrieve the last 10 seconds of frames from the rolling buffer and prepare a clip.

**Step 4: Clip transfer (edge to server)**
The candidate clip (~10 s, 50 frames at 5 fps) is sent from the edge node to the central server over the facility network. Raw video for non-candidate frames is never transmitted.

**Step 5: Tier 2: VLM confirmation (central GPU)**
Qwen2-VL processes the clip. It outputs:

- A binary decision: confirmed fall / not confirmed.
- A natural-language description of what it observed (e.g., "Person is stationary on the floor, arms at sides. Posture is consistent with a fall.").
- A confidence score.

The result is passed to Tier 3 regardless of whether the fall was confirmed. Rejected candidates generate a low-severity log entry.

**Step 6: Tier 3: dedup, fusion, routing (central)**
Tier 3 merges the Tier 2 result with any Tier 0 rule events from the same room within a deduplication window (e.g., 30 seconds). It assigns a severity score, produces a structured alert, and places it in the priority queue.

**Step 7: Reviewer console**
The alert surfaces on the reviewer console with all required fields: room ID, camera ID, timestamp, event type, severity, NL description, trigger reason, evidence frame/clip, confidence, and recommended action. A reviewer acts on it.

---

## Escalation logic

The cascade has three escalation boundaries. Each boundary defines what condition must be met to pass work to the next tier.

### Boundary 1: Tier 0 to Tier 1

| | |
|---|---|
| Condition to escalate | `person_count >= 1` |
| What passes | Bounding boxes of all detected persons in the frame |
| What is discarded | The frame itself (not stored); the detection result is discarded if count = 0 |
| Approximate pass rate | Varies by facility. In a quiet office building, >80% of frames are discarded here. |

### Boundary 2: Tier 1 to Tier 2

| | |
|---|---|
| Condition to escalate | `aspect_ratio(largest_box) < 1.0` sustained for `>= 5` consecutive sampled frames |
| What passes | A 10-second buffered clip |
| What is filtered | Any frame sequence where the person stays upright, or where a low ratio lasts fewer than 5 frames (single-frame dips, brief bends) |
| Why 5 frames | A transient drop (e.g., picking something up) does not persist. A fall does. |
| Approximate pass rate | Very low. Falls are rare events. At average occupancy across 1,000 cameras, approximately one candidate clip is generated every 7 seconds. |

### Boundary 3: Tier 2 to Tier 3

| | |
|---|---|
| Condition to escalate | All clips from Tier 2 proceed to Tier 3 (confirmed and rejected alike) |
| What passes | Confirmation result, NL description, confidence score |
| Effect of rejection | A rejected clip produces a low-severity event; it does not generate a reviewer alert but is logged. |
| Why both outcomes proceed | Tier 3 needs the rejection signal to close the deduplication window for that room. |

### Boundary 0a: Tier 0 rules to Tier 3 (direct path)

| | |
|---|---|
| Conditions | Off-hours rule: `person_count > 0` outside permitted hours. Overcrowding rule: `person_count > room_threshold`. |
| What passes | Rule type, count, camera ID, timestamp |
| Tier 2 involvement | None. These alerts do not require clip confirmation. |

---

## Per-tier detail

### Tier 0: edge person detection

**Status: REAL.** Detection runs on the UR Fall dataset in the included scripts; the occupancy rules are simulated.

**Model:** YOLOv8n. 6 MB weights, pretrained on COCO (80 classes). Class 0 is "person"; all other classes are ignored. Confidence threshold: 0.4.

**Hardware:** Jetson Orin Nano. Handles 10-30 cameras at ~5 fps depending on average occupancy. A camera with 95% empty frames uses far less compute than one in a continuously occupied corridor.

**Inputs:** raw frame (PNG/JPEG from camera stream).

**Outputs:**
- Person count for the frame.
- Bounding box coordinates (x1, y1, x2, y2) and confidence for each detected person.

**Why YOLOv8n and not a larger model:** larger YOLO variants or two-stage detectors exceed the latency budget or memory capacity of the Orin Nano at this frame rate. YOLOv8n is accurate enough to detect people reliably; missed detections at Tier 0 are uncommon, and the cascade is tolerant of occasional misses because a single non-detection resets the Tier 1 persistence counter rather than producing a false negative by itself.

---

### Tier 1: edge posture signal

**Status: REAL.** The aspect-ratio signal is computed from the same bounding boxes produced by Tier 0. Evaluation is real.

**Signal:** height divided by width of the largest person bounding box in the frame. A standing person in a typical overhead/side camera view produces a tall, narrow box (ratio well above 1.0, often 2.0-3.5). A person lying on the floor produces a wide, low box (ratio below 1.0, often 0.5-0.7).

**Rule parameters (all evaluated on UR Fall dataset):**
- Fall threshold: ratio < 1.0
- Persistence: 5 consecutive sampled frames
- Sampling rate: every 3 frames (mirrors ~5 fps design rate)

**Evaluation result (60 sequences):** 12 TP, 18 FN, 7 FP, 33 TN. Precision ~63%, recall ~40%.

**Why recall is 40% and what that means:** the aspect-ratio rule misses falls where the person's bounding box does not become wide. This includes falls toward or away from the camera (the depth axis), falls where the person slides down a wall, and falls where partial occlusion causes the bounding box to clip early. These 18 missed falls are the primary reason Tier 2 is required. The rule is intentionally a cheap pre-filter, not the final decision.

**Production path:** replacing the aspect-ratio rule with a skeleton-based pose model (RTMPose or MoveNet) at Tier 1 would substantially increase recall. This is on the two-weeks plan.

---

### Tier 2: central clip confirmation

**Status: SIMULATED in this demo.** The NL descriptions shown in the reviewer console are hand-authored examples.

**Model:** Qwen2-VL (vision-language model). Preferred over video action models (SlowFast, X3D) because it produces a natural-language description alongside the binary confirmation. Staff need to understand what the camera saw before dispatching; a label alone does not provide that.

**Input:** 10-second buffered clip (~50 frames at 5 fps) centred on the Tier 1 trigger point.

**Output:** confirmation decision (fall / not fall), NL description, confidence score.

**Processing time:** ~5 seconds per clip on an A100 or L4 GPU.

**T4 GPU: rejected.** A T4 takes 15-20 seconds per clip. End-to-end safety event latency would be 17-23 seconds for the GPU stage alone, exceeding the 30-second total budget when network and Tier 3 time are added.

---

### Tier 3: central deduplication and fusion

**Status: SIMULATED.**

**Deduplication:** multiple cameras may cover the same room. A fall visible on two cameras generates two Tier 1 candidates. Tier 3 merges events from the same room within a configurable time window (default 30 seconds) into a single alert. The merged alert uses the highest-confidence input.

**Fusion:** combines signals from all tiers into a single severity score.

| Severity | Condition |
|----------|-----------|
| HIGH | Tier 2 confirmed fall, or person down for >60 seconds without Tier 2 rejection |
| MEDIUM | Off-hours presence (Tier 0 rule) or overcrowding (Tier 0 rule) |
| LOW | Tier 1 candidate rejected by Tier 2 (logged, not surfaced to reviewer) |

**Why deterministic logic:** a machine-learning fusion model would be a black box. A rule-based aggregator makes the reason for each severity level explicit and auditable. This matters for liability, compliance, and staff trust.

---

## Deployment: edge, server, and cloud

### Edge layer (Jetson Orin Nano)

One node per 10-30 cameras. The node runs:
- Tier 0 inference (YOLOv8n)
- Tier 1 arithmetic (aspect-ratio rule)
- Rolling frame buffer (10 seconds per camera)
- Occupancy rule evaluation

The edge node does not store frames permanently. The rolling buffer is in memory. When no candidate is triggered, raw frames are never written to disk or transmitted off the node.

**Network requirement:** local area network connection to the central server. Bandwidth is low: only candidate clips are transmitted (~2-5 MB per 10-second clip at typical CCTV quality). No raw video streams leave the edge.

### Central server (on-premises GPU)

One location per facility (or per building cluster). Runs:
- Tier 2: Qwen2-VL inference (2-3 A100 or L4 GPUs provisioned; ~1 GPU average load at 1,000 cameras)
- Tier 3: deduplication, fusion, and priority queue
- Alert database (structured log of all events)

The central server does not need to be internet-connected for core operation. It requires only LAN connectivity to edge nodes and to the reviewer console.

### Reviewer console (web, on-premises or cloud)

The reviewer console is a web application. It can be served from:
- The central server directly (on-premises, lowest complexity)
- A cloud-hosted service (enables remote access, mobile, multi-site aggregation)

For a single facility, on-premises hosting is simpler and keeps all video-derived data within the facility. For multi-site deployments, a cloud-hosted console allows operators to monitor multiple facilities from one interface, provided that alert data (not raw video or clips) is forwarded to the cloud service.

---

## Scaling to 1,000+ cameras

### Edge node count

| Cameras per node | Nodes for 1,000 cameras |
|-----------------|------------------------|
| 30 (low-occupancy facility: storage, corridors) | 34 nodes |
| 10 (high-occupancy facility: clinical, public spaces) | 100 nodes |

Nodes operate independently. Adding cameras means adding edge nodes; no central reconfiguration is required.

### Central GPU utilisation

The Tier 2 load depends on how often Tier 1 raises candidates. Falls are rare. At average occupancy across 1,000 cameras, the expected candidate clip rate is approximately one every 7 seconds.

| Metric | Value |
|--------|-------|
| Candidate clip rate (1,000 cameras, average) | ~1 clip per 7 s |
| Clip processing time on A100/L4 | ~5 s |
| Average GPU utilisation (1 GPU) | 5 / 7 = 71% |
| Provisioned GPUs | 2-3 (burst buffer + failover) |

At 3x burst (e.g., end-of-shift when many people are moving simultaneously, or a fire alarm triggering many detections), 3 GPUs maintain sub-30-second latency. High-severity clips are served first from the priority queue; low-severity candidates are deferred or dropped if the queue depth exceeds the latency budget.

### Latency at scale

| Stage | Duration |
|-------|---------|
| Tier 0: YOLOv8n inference on edge | < 1 s |
| Tier 1: aspect-ratio arithmetic | < 0.1 s |
| Frame buffer retrieval | < 0.1 s |
| Network: clip edge to server (LAN) | < 1 s |
| Tier 2: VLM inference on GPU | ~5 s |
| Tier 3: dedup + routing | < 0.5 s |
| **Total: safety event alert** | **~7-10 s** |
| **Total: occupancy rule alert** | **~2-3 s** |

Both are within the stated targets: 5-15 seconds for occupancy/off-hours events, 10-30 seconds for safety events.

These latency figures hold regardless of whether 1 camera or 1,000 cameras are active, because each pipeline is independent. The only shared resource is the central GPU queue; provisioning 2-3 GPUs keeps queue wait time negligible at the expected candidate rate.

### What does not scale linearly

- **Tier 3 deduplication window** must be sized for the number of cameras per room. A room with 4 cameras generates 4 candidate clips for a single fall. The deduplication window absorbs this; its computational cost is trivial.
- **Alert database** write rate scales with event rate, not camera count. At 1,000 cameras, rule-based alerts (occupancy) may run to thousands per day; fall alerts are rare. A standard relational database handles this comfortably.
- **Reviewer capacity** is the real constraint. At high scale, alert volume must be managed so reviewers are not overwhelmed. Severity filtering and deduplication at Tier 3 are the primary mechanisms; reviewer routing (different staff handle HIGH vs. MEDIUM alerts) is an operational design question outside this architecture.
