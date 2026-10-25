#!/usr/bin/env bash
# Start the local read-only backtest UI.
# Usage: scripts/serve-ui.sh [--host 127.0.0.1] [--port 8000] [--runs-dir runs]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
python="$root/.venv/bin/python"
dist="$root/frontend/dist/index.html"

if [[ ! -x "$python" ]]; then
  echo "missing $python; create the virtualenv and install dependencies first" >&2
  exit 1
fi

if [[ ! -f "$dist" ]]; then
  echo "missing $dist" >&2
  echo "build it with: npm ci --prefix frontend && npm run build --prefix frontend" >&2
  exit 1
fi

cd "$root"
export PYTHONPATH="$root/lib${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m stolgo.cli.main serve --runs-dir "$root/runs" "$@"
