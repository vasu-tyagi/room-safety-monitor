# Requirements (from the challenge brief)

## Must show
- End-to-end architecture, camera input to alert output.
- Clear responsibility boundaries between layers.
- Near-real-time at 1000+ cameras.
- Cheap continuous processing triggering deeper analysis only when needed.
- Reduced false positives, duplicate alerts, and unnecessary expensive-model use.
- Alerts generated, explained, prioritised, routed for human review.
- A demo on representative video.

## Deliverable 1: Architecture document
Diagram, per-layer explanation, data flow, escalation logic, edge/server/cloud deployment, scaling plan to 1000+.

## Deliverable 2: Demo
3 scenarios (normal daytime; off-hours/overcrowding; safety event/fall), sample alert outputs, clear real/simulated/assumed labels, stated limitations.

## Deliverable 3: Two-page write-up
Design choices; models/tools; trade-offs; real-time and cost; false positive/negative risks; privacy/data handling; two-weeks priorities.

## Alert fields
room/camera ID, timestamp, event type, severity, NL description, trigger reason, evidence clip/frame, confidence, recommended action.

## Latency targets
Presence/off-hours/overcrowding 5-15s; safety events 10-30s; analytics async.
