#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${FATHOM_SMOLLM2_ACCEPTANCE:-}" != "1" ]]; then
  cat >&2 <<'EOF'
SmolLM2 optional API acceptance is opt-in because it downloads or reuses about 271 MB of model files and can use several GB of RAM.
Set FATHOM_SMOLLM2_ACCEPTANCE=1 to run it.
EOF
  exit 2
fi

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 127; }
}

need cargo
need curl
need python3

PORT="${FATHOM_SMOLLM2_ACCEPTANCE_PORT:-18186}"
BASE="http://127.0.0.1:${PORT}"
WAIT_SECONDS="${FATHOM_SMOLLM2_ACCEPTANCE_WAIT_SECONDS:-240}"
REQUEST_TIMEOUT="${FATHOM_SMOLLM2_ACCEPTANCE_REQUEST_TIMEOUT:-900}"
KEEP_ARTIFACTS="${FATHOM_SMOLLM2_ACCEPTANCE_KEEP_ARTIFACTS:-0}"
TMP_ROOT="${FATHOM_SMOLLM2_ACCEPTANCE_ROOT:-${TMPDIR:-/tmp}/fathom-smollm2-api-$$}"
MODELS_DIR="${FATHOM_SMOLLM2_ACCEPTANCE_MODELS_DIR:-$TMP_ROOT/models}"
STATE_DIR="${FATHOM_SMOLLM2_ACCEPTANCE_STATE_DIR:-$TMP_ROOT/state}"
ARTIFACT_DIR="${FATHOM_SMOLLM2_ACCEPTANCE_ARTIFACT_DIR:-$TMP_ROOT/artifacts}"
LOG_DIR="${FATHOM_SMOLLM2_ACCEPTANCE_LOG_DIR:-$TMP_ROOT/logs}"
REPO_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ $status -eq 0 && "$KEEP_ARTIFACTS" != "1" && -z "${FATHOM_SMOLLM2_ACCEPTANCE_ROOT:-}" && -z "${FATHOM_SMOLLM2_ACCEPTANCE_ARTIFACT_DIR:-}" ]]; then
    rm -rf "$TMP_ROOT"
  else
    echo "SmolLM2 optional acceptance artifacts: $ARTIFACT_DIR" >&2
    if [[ $status -ne 0 && -f "$LOG_DIR/server.log" ]]; then
      echo "--- server.log tail ---" >&2
      tail -120 "$LOG_DIR/server.log" >&2 || true
    fi
  fi
}
trap cleanup EXIT

if curl -fsS "$BASE/v1/health" >/dev/null 2>&1; then
  echo "Port $PORT already has a responding Fathom backend. Choose another FATHOM_SMOLLM2_ACCEPTANCE_PORT." >&2
  exit 1
fi

mkdir -p "$MODELS_DIR" "$STATE_DIR" "$ARTIFACT_DIR" "$LOG_DIR"

echo "== start isolated Fathom server for optional SmolLM2 API acceptance =="
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
curl -fsS "$BASE/v1/health" >/dev/null

python3 - "$BASE" "$ARTIFACT_DIR" "$REQUEST_TIMEOUT" "$REPO_COMMIT" "$STARTED_AT" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")
artifacts = pathlib.Path(sys.argv[2])
timeout = float(sys.argv[3])
repo_commit = sys.argv[4]
started_at = sys.argv[5]
artifacts.mkdir(parents=True, exist_ok=True)

MODEL_ID = "hf-huggingfacetb-smollm2-135m-instruct"
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"

checks = []

def request(method, path, body=None):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, json.loads(raw) if raw else None


def save(name, status, payload, check, description, expected):
    (artifacts / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks.append({
        "name": check,
        "artifact": name,
        "description": description,
        "expected_http_status": expected,
        "http_status": status,
        "status": "passed" if status == expected else "failed",
    })

status, health = request("GET", "/v1/health")
save("01-v1-health.json", status, health, "health", "Backend health responds before optional SmolLM2 install.", 200)
assert status == 200, health

status, install = request("POST", "/api/models/catalog/install", {"repo_id": REPO_ID, "filename": "model.safetensors"})
save("02-install-smollm2.json", status, install, "install_smollm2", "Pinned SmolLM2 catalog demo installs through catalog verification.", 200)
assert status == 200, install
assert install.get("id") == MODEL_ID, install
assert install.get("status") == "ready", install
assert install.get("capability_status") == "runnable", install
manifest = install.get("download_manifest") or {}
assert manifest.get("repo_id") == REPO_ID, manifest
assert manifest.get("revision") == REVISION, manifest
assert manifest.get("license") == "apache-2.0", manifest
assert manifest.get("license_status") == "permissive", manifest
assert manifest.get("verification_status") == "verified", manifest

status, models = request("GET", "/v1/models")
save("03-v1-models-after-smollm2.json", status, models, "models_include_smollm2", "Validated SmolLM2 demo appears in /v1/models after install.", 200)
assert status == 200, models
model_items = [item for item in models.get("data", []) if item.get("id") == MODEL_ID]
assert model_items, models
fathom = model_items[0].get("fathom") or {}
assert fathom.get("capability_status") == "runnable", model_items[0]
assert "safetensors-hf" in fathom.get("backend_lanes", []), model_items[0]

chat_body = {
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": "Reply with one short sentence about careful local inference."}],
    "max_tokens": 24,
    "temperature": 0.2,
}
for artifact, check, description in [
    ("04-chat-cold.json", "chat_cold_llama", "First same-process SmolLM2 chat call returns real local content and Llama-family metrics."),
    ("05-chat-warm.json", "chat_warm_llama", "Second same-process SmolLM2 chat call returns real local content and warm Llama-family metrics."),
]:
    status, chat = request("POST", "/v1/chat/completions", chat_body)
    save(artifact, status, chat, check, description, 200)
    assert status == 200, chat
    message = chat.get("choices", [{}])[0].get("message") or {}
    assert message.get("role") == "assistant", chat
    assert isinstance(message.get("content"), str) and message.get("content").strip(), chat
    metrics = (chat.get("fathom") or {}).get("metrics") or {}
    assert metrics.get("runtime_family") == "llama", metrics
    assert metrics.get("runtime_residency") in {"cold_loaded", "warm_reused"}, metrics

stream_body = dict(chat_body, stream=True)
status, stream = request("POST", "/v1/chat/completions", stream_body)
save("06-chat-stream-refusal.json", status, stream, "stream_refusal", "Streaming remains truthfully refused for SmolLM2.", 501)
assert status == 501, stream
error = stream.get("error") or {}
assert error.get("type") == "not_implemented" and error.get("code") == "not_implemented", stream
assert "choices" not in stream, stream

summary = {
    "schema": "fathom.smollm2_optional_api_acceptance.summary.v1",
    "passed": all(check["status"] == "passed" for check in checks),
    "repo_commit": repo_commit,
    "started_at": started_at,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "base_url": base,
    "artifact_dir": ".",
    "model_dir": "models/",
    "state_dir": "state/",
    "log_dir": "logs/",
    "model_id": MODEL_ID,
    "repo_id": REPO_ID,
    "revision": REVISION,
    "checks": checks,
    "caveats": [
        "Optional local larger-demo evidence only; not default CI.",
        "Does not prove generation quality, latency, throughput, production readiness, legal suitability, broad SmolLM2/Llama-style compatibility, arbitrary Hugging Face execution, streaming, external proxying, or full OpenAI API parity.",
        "Does not claim GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference.",
    ],
}
(artifacts / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(artifacts / "summary.md").write_text(
    "# SmolLM2 optional API acceptance artifacts\n\n"
    f"- Result: `{'passed' if summary['passed'] else 'failed'}`\n"
    f"- Repo commit: `{repo_commit}`\n"
    f"- Model: `{MODEL_ID}`\n"
    f"- Upstream: `{REPO_ID}` at `{REVISION}`\n"
    "- Scope: optional local larger-demo API evidence only; not default CI.\n"
    "- Artifact directory: `.`\n"
    "- State directory: `state/`\n"
    "- Model directory: `models/`\n"
    "- Server log: `logs/server.log`\n\n"
    "## Checks\n\n"
    + "\n".join(f"- `{check['name']}`: `{check['status']}` ({check['artifact']})" for check in checks)
    + "\n\n## What this does not prove\n\n"
    "- No generation quality, latency, throughput, production readiness, legal suitability, broad SmolLM2/Llama-style compatibility, arbitrary Hugging Face execution, streaming, external proxying, or full OpenAI API parity claim.\n"
    "- No public/runtime GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference claim.\n",
    encoding="utf-8",
)
assert summary["passed"], summary
PY

python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py "$ARTIFACT_DIR"

echo "SmolLM2 optional API acceptance passed. Artifacts: $ARTIFACT_DIR"
