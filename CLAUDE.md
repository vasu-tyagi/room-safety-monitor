# Room Safety Monitoring - Architecture & Demo Challenge

## IMPORTANT: scope of this work
This is a DESIGN AND DEMO challenge, not a production software build. Code quality is explicitly NOT evaluated. The design decisions are already made and are FIXED (see below). Your job is to organise the project, regenerate the artifacts, and produce the documents. Do NOT reopen or redesign the architecture. If a decision seems open, treat it as closed per this file.

## The system: a 4-tier cascade
Cheap analysis runs on every camera all the time; expensive analysis runs only on the few events a cheaper layer already flagged.
- Tier 0 (edge, always on): person detection + count. YOLOv8n. Handles off-hours occupancy and overcrowding as simple rules.
- Tier 1 (edge, only when people present): posture/motion. Bounding-box aspect ratio is the cheap fall signal (tall = standing, wide = on floor). A pose model (RTMPose / MoveNet) is the richer production option.
- Tier 2 (central GPU, only on candidates): confirms a candidate from a short buffered clip. Video action model (SlowFast / X3D) or a VLM (Qwen2-VL); VLM preferred because it writes the natural-language description.
- Tier 3 (central): deduplication (merge same-room events within a time window), fusion (combine signals into one severity/confidence), severity ranking. This tier is deterministic logic, not ML, on purpose, for explainability.

## Sizing (FIXED, defensible numbers)
- Edge node: Jetson Orin Nano, ~10-30 cameras depending on occupancy (fewer in busy facilities, more in quiet ones). Sampling ~5 fps.
- Central Tier 2: A100 / L4 class GPU. ~1 GPU on average for 1000 cameras (one candidate clip ~every 7s, ~5s/clip), provision 2-3 for burst + failover. T4 rejected: its per-clip time exceeds the latency budget.
- Latency targets: 5-15s for presence/rules, 10-30s for safety events.
- Burst handling: priority queue + load-shedding, keep high-severity clips first. Designed, simulated in demo.

## Real vs simulated (state this honestly everywhere)
- REAL: Tier 0 YOLOv8n detection on the UR Fall dataset; the Tier 1 box aspect-ratio signal; an eval across all 60 sequences.
- SIMULATED / ASSUMED: Tier 2 confirmation, Tier 3 fusion/dedup, the off-hours clock, the overcrowding threshold, multi-camera setup, room IDs, timestamps.

## Eval result (USE THESE NUMBERS, do not invent new ones)
The cheap box-ratio rule alone, over 60 sequences (threshold ratio 1.0, persistence 5 frames, sampled every 3): 12 TP, 18 FN, 7 FP, 33 TN. Precision ~63%, recall ~40%. This LOW recall is the empirical reason the cascade needs Tier 2 confirmation and/or a pose model at Tier 1. Lead with this honestly; it is a strength, not a weakness to hide.

## Deliverables
1. Architecture document with end-to-end diagram, layer explanations, data flow, escalation logic, deployment design, scaling plan.
2. Demo: a reviewer console (HTML) covering 3 scenarios (normal, off-hours/overcrowding, fall) + the aspect-ratio comparison chart. Badge real vs simulated.
3. Two-page design write-up: design choices, tools, trade-offs, real-time and cost, false positive/negative risks, privacy, two-weeks plan.

## Writing rules (apply to ALL prose)
- No em dashes. Plain, direct register. No marketing language, no filler, no rule-of-three padding. Sentence case.
- Be honest about limitations. Never claim simulated parts are running.

## Existing inputs
src/detect.py (YOLOv8n detector), src/evaluate.py (the eval), and the CSV logs in data-results/ are the real, already-produced inputs. Build from these.
