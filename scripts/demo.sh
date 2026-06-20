#!/usr/bin/env bash
# demo.sh — start all services for local demonstration.
# Run from the repo root: bash scripts/demo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "Warning: .env not found. Copy .env.example to .env and fill in HF_TOKEN if needed."
fi

# Ensure venv exists
if [ ! -f venv/bin/activate ]; then
  echo "Error: venv not found. Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

echo "Starting Postgres..."
docker compose -f deploy/docker-compose.yml up -d postgres

echo "Waiting for Postgres to be ready..."
until docker exec rsm-postgres pg_isready -U postgres -d room_safety -q 2>/dev/null; do
  sleep 1
done

echo "Running Alembic migrations..."
venv/bin/python -m alembic upgrade head

echo "Starting service plane (uvicorn)..."
DASHBOARD_ORIGIN="${DASHBOARD_ORIGIN:-http://localhost:3000}" \
  venv/bin/uvicorn services.service_plane.app:app --reload --port 8000 &
UVICORN_PID=$!
echo "  Service plane PID: $UVICORN_PID (http://localhost:8000)"

# Give uvicorn a moment to bind
sleep 2

echo "Starting dashboard (npm run dev)..."
(cd services/dashboard && npm run dev) &
DASHBOARD_PID=$!
echo "  Dashboard PID: $DASHBOARD_PID (http://localhost:3000)"

echo ""
echo "Ready. Run: bash scripts/run_example.sh"
echo ""
echo "  Service plane: http://localhost:8000/docs"
echo "  Dashboard:     http://localhost:3000"
echo ""
echo "Stop: kill $UVICORN_PID $DASHBOARD_PID && docker compose -f deploy/docker-compose.yml stop postgres"
