#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 127; }
}

need cargo
need curl
need python3

PORT="${FATHOM_PUBLIC_CONTRACT_PORT:-$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}"
BASE="http://127.0.0.1:${PORT}"
WAIT_SECONDS="${FATHOM_PUBLIC_CONTRACT_WAIT_SECONDS:-180}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fathom-public-contract.XXXXXX")"
MODELS_DIR="$TMP_ROOT/models"
STATE_DIR="$TMP_ROOT/state"
LOG_DIR="$TMP_ROOT/logs"
RUN_DIR="$TMP_ROOT/run"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ $status -ne 0 && -f "$LOG_DIR/server.log" ]]; then
    echo "Public API contract smoke failed. Sanitized server log tail:" >&2
    sed -E \
      -e 's#(/private)?/tmp/[^[:space:]]+#<temp-path>#g' \
      -e 's#/'"Users"'/[^[:space:]]+#<user-path>#g' \
      "$LOG_DIR/server.log" | tail -80 >&2 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

mkdir -p "$MODELS_DIR" "$STATE_DIR" "$LOG_DIR" "$RUN_DIR"

echo "== public API contract smoke: start isolated fathom-server =="
FATHOM_PORT="$PORT" \
FATHOM_STATE_DIR="$STATE_DIR" \
FATHOM_MODELS_DIR="$MODELS_DIR" \
FATHOM_LOG_DIR="$LOG_DIR" \
cargo run -q -p fathom-server >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 "$WAIT_SECONDS"); do
  if curl -fsS "$BASE/v1/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "fathom-server exited before /v1/health became ready" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "$BASE/v1/health" >/dev/null 2>&1; then
  echo "Timed out waiting for isolated fathom-server /v1/health" >&2
  exit 1
fi

python3 - "$BASE" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")


def request(method, path, body=None):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError as error:
            raise AssertionError(f"{method} {path} returned non-JSON error body: {raw[:120]!r}") from error
        return exc.code, payload


def assert_error(payload, code):
    assert isinstance(payload, dict), payload
    error = payload.get("error")
    assert isinstance(error, dict), payload
    assert isinstance(error.get("message"), str) and error["message"], payload
    assert error.get("type") == code, payload
    assert error.get("code") == code, payload
    assert "param" in error, payload


def assert_no_chat_success(payload):
    assert isinstance(payload, dict), payload
    assert "choices" not in payload, payload
    assert payload.get("object") != "chat.completion", payload


def assert_no_embedding_success(payload):
    assert isinstance(payload, dict), payload
    assert "data" not in payload, payload
    assert payload.get("object") != "list", payload


status, health = request("GET", "/v1/health")
assert status == 200, (status, health)
assert health.get("ok") is True, health
assert health.get("engine") == "fathom", health
assert health.get("generation_ready") is False, health

status, models = request("GET", "/v1/models")
assert status == 200, (status, models)
assert models.get("object") == "list", models
assert models.get("data") == [], models

chat_body = {
    "model": "missing-chat-model",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}
status, stream_refusal = request("POST", "/v1/chat/completions", chat_body)
assert status == 501, (status, stream_refusal)
assert_error(stream_refusal, "not_implemented")
assert_no_chat_success(stream_refusal)

chat_body["stream"] = False
status, missing_chat = request("POST", "/v1/chat/completions", chat_body)
assert status == 400, (status, missing_chat)
assert_error(missing_chat, "model_not_found")
assert_no_chat_success(missing_chat)

embedding_body = {
    "model": "missing-embedding-model",
    "input": "hello",
    "encoding_format": "base64",
}
status, base64_refusal = request("POST", "/v1/embeddings", embedding_body)
assert status == 400, (status, base64_refusal)
assert_error(base64_refusal, "invalid_request")
assert_no_embedding_success(base64_refusal)

embedding_body["encoding_format"] = "float"
status, missing_embedding = request("POST", "/v1/embeddings", embedding_body)
assert status == 404, (status, missing_embedding)
assert_error(missing_embedding, "embedding_model_not_found")
assert_no_embedding_success(missing_embedding)

external_body = {
    "id": "external-placeholder-smoke",
    "name": "External placeholder smoke",
    "source": "OpenAI-compatible",
    "api_base": "https://api.example.test/v1",
    "api_key": "placeholder-key",
    "model_name": "placeholder-model",
}
status, external = request("POST", "/api/models/external", external_body)
assert status == 200, (status, external)
assert external.get("provider_kind") == "external", external
assert external.get("api_key_configured") is True, external
assert "api_key" not in external, external
assert external.get("capability_status") == "planned", external

status, models_after_external = request("GET", "/v1/models")
assert status == 200, (status, models_after_external)
assert models_after_external.get("data") == [], models_after_external

status, activation_refusal = request("POST", "/api/models/external-placeholder-smoke/activate", {})
assert status == 501, (status, activation_refusal)
assert_error(activation_refusal, "external_proxy_not_implemented")

status, external_chat = request(
    "POST",
    "/v1/chat/completions",
    {"model": "external-placeholder-smoke", "messages": [{"role": "user", "content": "hello"}]},
)
assert status == 501, (status, external_chat)
assert_error(external_chat, "not_implemented")
assert_no_chat_success(external_chat)

print("public API contract smoke passed: health, models, chat refusals, embeddings refusals, external placeholder boundary")
PY
