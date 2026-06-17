# Claude Code working file — Room Safety Monitoring (architecture rebuild)

## Mission

Rebuild this project to match the six-layer reference architecture provided by the team. The existing v0.5 build has YOLOv8n person detection and an aspect-ratio fall rule. Keep that working code as the starting point for L2 detection, but build out all six layers end to end.

## What the system must do

When complete, the system must accept a video file as input, process it through all six layers, store incidents in Postgres, log evidence clips in MinIO, expose alerts via WebSocket to a Next.js dashboard, and accept operator feedback that writes back to the pgvector knowledge base.

## Working principles

1. Build in slices. Each slice must leave the system end-to-end runnable.
2. Honesty over polish. Mark what's simulated, what's substituted, and why. The README has a "What's real / What's simulated / Why" table that gets updated every slice.
3. Tests for new code paths. Existing UR Fall eval must keep producing the same numbers until pose-based fall detection replaces aspect-ratio.
4. No AI tone in docs. No em dashes. No "delve," "comprehensive," "robust framework." Plain professional English.
5. Open source only. No paid APIs except Hugging Face Inference free tier for the VLM (with a documented fallback to a stub).
6. Commit after each slice completes. Commit message format: `feat(L<n>): <what slice>`.

## Commit policy

Commit messages contain only the subject and body specified by the user. Never add Co-Authored-By trailers, Generated-with notes, or any other AI attribution. The user's commit messages are final and complete.

## File locations

- Source: `services/<layer_name>/` (one folder per layer)
- Shared schemas and utilities: `shared/`
- Frontend: `services/dashboard/`
- Docs: `docs/`
- Tests: `tests/`
- Eval data and scripts: `evals/`
- Demo video files: `demo/sample_videos/`
- Docker config: `deploy/docker-compose.yml`

## Session start protocol

Every session:

1. Read this file
2. Read `docs/ARCHITECTURE.md`
3. Read `docs/SLICES.md` (the build plan)
4. State which slice you're working on
5. Confirm with the user before writing code
