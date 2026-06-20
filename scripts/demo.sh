#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# .env check
if [ ! -f .env ]; then
  echo "Warning: .env not found. Copy .env.example to .env and fill in HF_TOKEN if needed."
fi

# venv check
if [ ! -f venv/bin/activate ]; then
  echo "Error: venv not found. Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

echo "Starting Postgres..."
(cd deploy && docker compose up -d postgres)

echo "Waiting for Postgres to be ready..."
until docker exec rsm-postgres pg_isready -U postgres -d room_safety -q 2>/dev/null; do
  sleep 1
done

echo "Activating venv and loading .env..."
source venv/bin/activate
set -a
[ -f .env ] && source .env
set +a
export VLM_MODE=${VLM_MODE:-auto}

echo "Running Alembic migrations..."
python -m alembic upgrade head

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
