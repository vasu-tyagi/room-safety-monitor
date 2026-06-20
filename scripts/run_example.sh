#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

VIDEO="${VIDEO_OVERRIDE:-demo/example_fall.mp4}"

echo "Processing $VIDEO..."
curl -X POST http://localhost:8000/process_video \
  -H "Content-Type: application/json" \
  -d "{\"video_path\": \"$VIDEO\", \"camera_id\": \"cam-demo\", \"room_id\": \"kitchen\"}"
