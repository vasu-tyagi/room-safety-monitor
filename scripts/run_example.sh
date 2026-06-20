#!/usr/bin/env bash
# run_example.sh — process a Le2i video through the full pipeline.
# Run from the repo root after demo.sh is running: bash scripts/run_example.sh
#
# The video must exist at the path below. Le2i is not in this repo;
# download from http://le2i.cnrs.fr and place under data/le2i/.
set -euo pipefail

curl -X POST http://localhost:8000/process_video \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "data/le2i/Coffee_room_02/Coffee_room_02/Videos/video (49).avi",
    "camera_id": "cam-kitchen",
    "room_id":   "kitchen-1"
  }'

echo ""
echo "Incidents created:"
curl -s http://localhost:8000/incidents | python3 -m json.tool
