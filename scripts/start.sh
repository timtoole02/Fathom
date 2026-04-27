#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${FATHOM_LOG_DIR:-$HOME/.fathom/logs}"
RUN_DIR="${FATHOM_RUN_DIR:-$HOME/.fathom/run}"
mkdir -p "$LOG_DIR" "$RUN_DIR"
BACKEND_PORT="${FATHOM_PORT:-8180}"
FRONTEND_PORT="${FATHOM_FRONTEND_PORT:-4185}"
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT/v1/health"
FRONTEND_URL="http://127.0.0.1:$FRONTEND_PORT"
cd "$ROOT"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 127
  }
}

stop_pid_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local pid
  pid="$(cat "$file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
  fi
  rm -f "$file"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local pid="$3"
  local log_file="$4"
  local tries="${5:-240}"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ $label is ready: $url"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited before becoming ready. Last log lines:" >&2
      tail -80 "$log_file" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "Timed out waiting for $label at $url. Last log lines:" >&2
  tail -80 "$log_file" >&2 || true
  exit 1
}

need cargo
need npm
need curl

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Frontend dependencies are missing. Run: npm --prefix frontend install" >&2
  exit 1
fi

stop_pid_file "$RUN_DIR/server.pid"
stop_pid_file "$RUN_DIR/frontend.pid"

nohup env FATHOM_PORT="$BACKEND_PORT" cargo run --release -p fathom-server >"$LOG_DIR/server.log" 2>&1 </dev/null &
echo $! > "$RUN_DIR/server.pid"
SERVER_PID="$!"
nohup npm --prefix "$ROOT/frontend" run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" >"$LOG_DIR/frontend.log" 2>&1 </dev/null &
echo $! > "$RUN_DIR/frontend.pid"
FRONTEND_PID="$!"

echo "Fathom starting. Logs: $LOG_DIR"
wait_for_http "$BACKEND_URL" "backend" "$SERVER_PID" "$LOG_DIR/server.log" 300
wait_for_http "$FRONTEND_URL" "frontend" "$FRONTEND_PID" "$LOG_DIR/frontend.log" 120

echo "Fathom is ready: frontend $FRONTEND_URL backend http://127.0.0.1:$BACKEND_PORT"
