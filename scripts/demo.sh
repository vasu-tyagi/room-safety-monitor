#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "Starting Postgres..."
(cd deploy && docker compose up -d postgres)

echo "Activating venv and loading .env..."
source venv/bin/activate
set -a
source .env
set +a
export VLM_MODE=${VLM_MODE:-auto}

echo "Starting backend..."
uvicorn services.service_plane.app:app --reload > /tmp/rsm-uvicorn.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID (logs: /tmp/rsm-uvicorn.log)"

echo "Waiting for backend to come up..."
until curl -s http://localhost:8000/health > /dev/null; do sleep 1; done

echo "Starting dashboard..."
(cd services/dashboard && npm run dev > /tmp/rsm-dashboard.log 2>&1) &
DASHBOARD_PID=$!
echo "Dashboard PID: $DASHBOARD_PID (logs: /tmp/rsm-dashboard.log)"

echo ""
echo "================================================"
echo "  Ready."
echo "  Open http://localhost:3000 in browser"
echo "  Run: bash scripts/run_example.sh"
echo "  Stop: kill $BACKEND_PID $DASHBOARD_PID"
echo "================================================"
