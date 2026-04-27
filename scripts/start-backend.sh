#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${FATHOM_LOG_DIR:-$HOME/.fathom/logs}"
RUN_DIR="${FATHOM_RUN_DIR:-$HOME/.fathom/run}"
BACKEND_PORT="${FATHOM_PORT:-8180}"
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT/v1/health"
FEATURES="${FATHOM_FEATURES:-}"
WAIT_SECONDS="${FATHOM_BACKEND_WAIT_SECONDS:-300}"
SERVER_LOG="$LOG_DIR/server.log"
SERVER_PID_FILE="$RUN_DIR/server.pid"

mkdir -p "$LOG_DIR" "$RUN_DIR"
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
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
  fi
  rm -f "$file"
}

wait_for_http() {
  local url="$1"
  local pid="$2"
  local tries="$3"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ Fathom backend is ready: $url"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Fathom backend exited before becoming ready. Last log lines:" >&2
      tail -80 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "Timed out waiting for Fathom backend at $url. Last log lines:" >&2
  tail -80 "$SERVER_LOG" >&2 || true
  exit 1
}

need cargo
need curl

stop_pid_file "$SERVER_PID_FILE"

cmd=(cargo run --release -p fathom-server)
if [[ -n "$FEATURES" ]]; then
  cmd+=(--features "$FEATURES")
fi

echo "Starting Fathom backend on http://127.0.0.1:$BACKEND_PORT"
if [[ -n "$FEATURES" ]]; then
  echo "Using Cargo features: $FEATURES"
fi
nohup env FATHOM_PORT="$BACKEND_PORT" "${cmd[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
echo $! > "$SERVER_PID_FILE"
SERVER_PID="$!"

echo "Logs: $SERVER_LOG"
wait_for_http "$BACKEND_URL" "$SERVER_PID" "$WAIT_SECONDS"
echo "Backend-only API base: http://127.0.0.1:$BACKEND_PORT"
