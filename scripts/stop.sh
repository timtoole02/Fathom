#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${FATHOM_RUN_DIR:-$HOME/.fathom/run}"
for file in "$RUN_DIR/server.pid" "$RUN_DIR/frontend.pid"; do
  [[ -f "$file" ]] || continue
  pid="$(cat "$file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
  fi
  rm -f "$file"
done
echo "Fathom stopped."
