#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${FATHOM_SMOKE_BACKEND_PORT:-18180}"
FRONTEND_PORT="${FATHOM_SMOKE_FRONTEND_PORT:-14185}"
BASE="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_BASE="http://127.0.0.1:${FRONTEND_PORT}"
BACKEND_WAIT_SECONDS="${FATHOM_SMOKE_BACKEND_WAIT_SECONDS:-300}"
FRONTEND_WAIT_SECONDS="${FATHOM_SMOKE_FRONTEND_WAIT_SECONDS:-120}"
TMP_DIR="${TMPDIR:-/tmp}/fathom-smoke-$$"
LOG_DIR="$TMP_DIR/logs"
STATE_DIR="$TMP_DIR/state"
MODELS_DIR="$TMP_DIR/models"
FIXTURE_DIR="$TMP_DIR/tiny-hf-fixture"
RUNNABLE_MODEL_DIR="${FATHOM_SMOKE_RUNNABLE_MODEL_DIR:-}"
SERVER_PID=""
FRONTEND_PID=""

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "Smoke failed. Logs are in: $LOG_DIR" >&2
    [[ -f "$LOG_DIR/server.log" ]] && { echo "--- server.log ---" >&2; tail -80 "$LOG_DIR/server.log" >&2 || true; }
    [[ -f "$LOG_DIR/frontend.log" ]] && { echo "--- frontend.log ---" >&2; tail -80 "$LOG_DIR/frontend.log" >&2 || true; }
  else
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 127; }
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local tries="${3:-60}"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ $label is responding"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for $label at $url" >&2
  return 1
}

post_json() {
  local url="$1"
  local body="$2"
  local output="$3"
  local status_file="$4"
  curl -sS -o "$output" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -X POST \
    --data "$body" \
    "$url" > "$status_file"
}

json_string() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

detect_runnable_model_dir() {
  if [[ -n "$RUNNABLE_MODEL_DIR" ]]; then
    [[ -d "$RUNNABLE_MODEL_DIR" ]] || { echo "FATHOM_SMOKE_RUNNABLE_MODEL_DIR does not exist: $RUNNABLE_MODEL_DIR" >&2; exit 1; }
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/.fathom/models/vijaymohan-gpt2-tinystories-from-scratch-10m" \
    "$HOME/.fathom/models/distilbert-distilgpt2" \
    "$HOME/.fathom/models/intel-tiny-random-gpt2"; do
    if [[ -d "$candidate" ]]; then
      RUNNABLE_MODEL_DIR="$candidate"
      return 0
    fi
  done
}

need cargo
need npm
need curl
need python3
mkdir -p "$LOG_DIR" "$STATE_DIR" "$MODELS_DIR" "$FIXTURE_DIR"

cat > "$FIXTURE_DIR/config.json" <<'JSON'
{"model_type":"llama","architectures":["LlamaForCausalLM"],"hidden_size":8,"num_hidden_layers":1,"vocab_size":16}
JSON
cat > "$FIXTURE_DIR/tokenizer.json" <<'JSON'
{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","vocab":{},"merges":[]}}
JSON
: > "$FIXTURE_DIR/model.safetensors"

echo "== cargo test =="
cargo test

echo "== frontend build =="
npm --prefix frontend run build

echo "== start backend and frontend on smoke ports =="
FATHOM_PORT="$BACKEND_PORT" \
FATHOM_STATE_DIR="$STATE_DIR" \
FATHOM_MODELS_DIR="$MODELS_DIR" \
cargo run --release -p fathom-server >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
npm --prefix frontend run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" >"$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

wait_for_http "$BASE/v1/health" "backend health" "$BACKEND_WAIT_SECONDS"
wait_for_http "$BASE/api/capabilities" "capabilities API" 30
wait_for_http "$FRONTEND_BASE" "frontend" "$FRONTEND_WAIT_SECONDS"

python3 - "$BASE/v1/health" "$BASE/api/capabilities" <<'PY'
import json, sys, urllib.request
for url in sys.argv[1:]:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.load(response)
    if url.endswith('/v1/health'):
        assert payload.get('ok') is True, payload
        assert payload.get('generation_ready') is False, payload
    else:
        lanes = payload.get('backend_lanes') or []
        assert any(lane.get('id') == 'safetensors-hf' for lane in lanes), payload
PY
echo "✓ health and capabilities payloads are truthful"

REGISTER_JSON="$LOG_DIR/register.json"
REGISTER_STATUS="$LOG_DIR/register.status"
post_json "$BASE/api/models/register" "{\"id\":\"tiny-hf-smoke\",\"name\":\"Tiny HF Smoke\",\"model_path\":\"$FIXTURE_DIR\",\"runtime_model_name\":\"tiny-hf-smoke-runtime\"}" "$REGISTER_JSON" "$REGISTER_STATUS"
[[ "$(cat "$REGISTER_STATUS")" == "200" ]] || { echo "model registration failed: HTTP $(cat "$REGISTER_STATUS")" >&2; cat "$REGISTER_JSON" >&2; exit 1; }
python3 - "$REGISTER_JSON" "$FIXTURE_DIR" <<'PY'
import json, pathlib, sys
payload = json.load(open(sys.argv[1]))
fixture = pathlib.Path(sys.argv[2])
assert payload['id'] == 'tiny-hf-smoke', payload
assert payload['status'] == 'registered', payload
assert payload['provider_kind'] == 'local', payload
assert pathlib.Path(payload['model_path']).resolve() == fixture.resolve(), payload
assert payload['runtime_model_name'] == 'tiny-hf-smoke-runtime', payload
assert payload['format'] == 'SafeTensors', payload
assert payload['capability_status'] == 'planned', payload
assert 'safetensors-hf' in payload['backend_lanes'], payload
assert 'not runnable until' in payload['capability_summary'], payload
assert payload.get('install_error'), payload
PY
echo "✓ tiny HF-style fixture maps to planned safetensors-hf lane without claiming runnable support"

MODELS_JSON="$LOG_DIR/v1-models.json"
curl -fsS "$BASE/v1/models" > "$MODELS_JSON"
python3 - "$MODELS_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload.get('object') == 'list', payload
ids = [item.get('id') for item in payload.get('data', [])]
assert 'tiny-hf-smoke' not in ids, payload
PY
echo "✓ /v1/models excludes non-runnable registered local fixture"

ACTIVATE_JSON="$LOG_DIR/activate.json"
ACTIVATE_STATUS="$LOG_DIR/activate.status"
post_json "$BASE/api/models/tiny-hf-smoke/activate" '{}' "$ACTIVATE_JSON" "$ACTIVATE_STATUS"
[[ "$(cat "$ACTIVATE_STATUS")" == "501" ]] || { echo "expected activation HTTP 501, got $(cat "$ACTIVATE_STATUS")" >&2; cat "$ACTIVATE_JSON" >&2; exit 1; }
echo "✓ local activation refuses to fake loading with HTTP 501"

CHAT_JSON="$LOG_DIR/chat-completions.json"
CHAT_STATUS="$LOG_DIR/chat-completions.status"
post_json "$BASE/v1/chat/completions" '{"model":"tiny-hf-smoke","messages":[{"role":"user","content":"hello"}]}' "$CHAT_JSON" "$CHAT_STATUS"
[[ "$(cat "$CHAT_STATUS")" == "501" ]] || { echo "expected chat completions HTTP 501, got $(cat "$CHAT_STATUS")" >&2; cat "$CHAT_JSON" >&2; exit 1; }
python3 - "$CHAT_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
error = payload.get('error') or {}
assert error.get('type') == 'not_implemented', payload
assert 'no fake inference' in error.get('message', '').lower(), payload
PY
echo "✓ /v1/chat/completions returns truthful no-fake-generation HTTP 501"

detect_runnable_model_dir
if [[ -n "$RUNNABLE_MODEL_DIR" ]]; then
  RUNNABLE_REGISTER_JSON="$LOG_DIR/runnable-register.json"
  RUNNABLE_REGISTER_STATUS="$LOG_DIR/runnable-register.status"
  RUNNABLE_ACTIVATE_JSON="$LOG_DIR/runnable-activate.json"
  RUNNABLE_ACTIVATE_STATUS="$LOG_DIR/runnable-activate.status"
  RUNNABLE_CHAT_JSON="$LOG_DIR/runnable-chat-completions.json"
  RUNNABLE_CHAT_STATUS="$LOG_DIR/runnable-chat-completions.status"

  RUNNABLE_ID="runnable-gpt2-smoke"
  RUNNABLE_DIR_JSON="$(json_string "$RUNNABLE_MODEL_DIR")"
  post_json "$BASE/api/models/register" "{\"id\":\"$RUNNABLE_ID\",\"name\":\"Runnable GPT-2 Smoke\",\"model_path\":$RUNNABLE_DIR_JSON,\"runtime_model_name\":\"$RUNNABLE_ID\"}" "$RUNNABLE_REGISTER_JSON" "$RUNNABLE_REGISTER_STATUS"
  [[ "$(cat "$RUNNABLE_REGISTER_STATUS")" == "200" ]] || { echo "runnable model registration failed: HTTP $(cat "$RUNNABLE_REGISTER_STATUS")" >&2; cat "$RUNNABLE_REGISTER_JSON" >&2; exit 1; }
  python3 - "$RUNNABLE_REGISTER_JSON" "$RUNNABLE_MODEL_DIR" <<'PY'
import json, pathlib, sys
payload = json.load(open(sys.argv[1]))
model_dir = pathlib.Path(sys.argv[2]).resolve()
assert payload['id'] == 'runnable-gpt2-smoke', payload
assert payload['status'] == 'ready', payload
assert pathlib.Path(payload['model_path']).resolve() == model_dir, payload
assert payload['format'] == 'SafeTensors', payload
assert payload['capability_status'] == 'runnable', payload
assert 'safetensors-hf' in payload['backend_lanes'], payload
assert 'Candle GPT-2' in payload['capability_summary'], payload
assert not payload.get('install_error'), payload
PY
  echo "✓ runnable GPT-2 SafeTensors package registers as Candle-backed runnable: $RUNNABLE_MODEL_DIR"

  post_json "$BASE/api/models/$RUNNABLE_ID/activate" '{}' "$RUNNABLE_ACTIVATE_JSON" "$RUNNABLE_ACTIVATE_STATUS"
  [[ "$(cat "$RUNNABLE_ACTIVATE_STATUS")" == "200" ]] || { echo "expected runnable activation HTTP 200, got $(cat "$RUNNABLE_ACTIVATE_STATUS")" >&2; cat "$RUNNABLE_ACTIVATE_JSON" >&2; exit 1; }

  curl -fsS "$BASE/v1/models" > "$MODELS_JSON"
  python3 - "$MODELS_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
ids = [item.get('id') for item in payload.get('data', [])]
assert 'runnable-gpt2-smoke' in ids, payload
PY
  echo "✓ /v1/models includes the runnable Candle GPT-2 model"

  post_json "$BASE/v1/chat/completions" '{"model":"runnable-gpt2-smoke","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}' "$RUNNABLE_CHAT_JSON" "$RUNNABLE_CHAT_STATUS"
  [[ "$(cat "$RUNNABLE_CHAT_STATUS")" == "200" ]] || { echo "expected runnable chat completions HTTP 200, got $(cat "$RUNNABLE_CHAT_STATUS")" >&2; cat "$RUNNABLE_CHAT_JSON" >&2; exit 1; }
  python3 - "$RUNNABLE_CHAT_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
choice = payload.get('choices', [{}])[0]
message = choice.get('message') or {}
usage = payload.get('usage') or {}
assert payload.get('object') == 'chat.completion', payload
assert message.get('role') == 'assistant', payload
assert isinstance(message.get('content'), str), payload
assert usage.get('completion_tokens', 0) > 0, payload
assert usage.get('prompt_tokens', 0) > 0, payload
PY
  echo "✓ /v1/chat/completions produces real Candle GPT-2 tokens for the runnable package"
else
  echo "ℹ skipped runnable GPT-2 smoke: set FATHOM_SMOKE_RUNNABLE_MODEL_DIR or download DistilGPT-2 / Intel tiny-random GPT-2 first"
fi

echo "Fathom smoke passed."
