#!/usr/bin/env bash
# start-mrna.sh — Start the .mrna FastAPI sidecar for Garden integration.
#
# Run this before starting Garden's dev server.
# The service initialises PrefillRouter + llama-server on first start (~45-60s).
# Subsequent Garden requests pay only ~500ms SAE routing + generation time.
#
# Usage:
#   ./start-mrna.sh            # default port 7437
#   MRNA_PORT=7438 ./start-mrna.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "[start-mrna] ERROR: .venv not found. Run 'python -m venv .venv && pip install -r requirements.txt' first."
  exit 1
fi

source .venv/bin/activate

echo "[start-mrna] Starting .mrna API sidecar on port ${MRNA_PORT:-7437} …"
exec python -m uvicorn src.mrna.api:app \
  --host 0.0.0.0 \
  --port "${MRNA_PORT:-7437}" \
  --log-level info \
  --no-access-log
