#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${FATHOM_ACCEPTANCE_PORT:-18180}"
BASE="http://127.0.0.1:${PORT}"
WAIT_SECONDS="${FATHOM_ACCEPTANCE_WAIT_SECONDS:-300}"
REQUEST_TIMEOUT="${FATHOM_ACCEPTANCE_REQUEST_TIMEOUT:-900}"
KEEP_ARTIFACTS="${FATHOM_ACCEPTANCE_KEEP_ARTIFACTS:-0}"
TMP_ROOT="${FATHOM_ACCEPTANCE_TMP_DIR:-${TMPDIR:-/tmp}/fathom-backend-acceptance-$$}"
MODELS_DIR="${FATHOM_ACCEPTANCE_MODELS_DIR:-$TMP_ROOT/models}"
STATE_DIR="${FATHOM_ACCEPTANCE_STATE_DIR:-$TMP_ROOT/state}"
ARTIFACT_DIR="${FATHOM_ACCEPTANCE_ARTIFACT_DIR:-$TMP_ROOT/artifacts}"
LOG_DIR="$ARTIFACT_DIR/logs"
SERVER_PID=""
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REPO_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 127; }
}

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi

  if [[ $status -eq 0 && "$KEEP_ARTIFACTS" != "1" ]]; then
    if [[ -z "${FATHOM_ACCEPTANCE_TMP_DIR:-}" && -z "${FATHOM_ACCEPTANCE_MODELS_DIR:-}" && -z "${FATHOM_ACCEPTANCE_STATE_DIR:-}" && -z "${FATHOM_ACCEPTANCE_ARTIFACT_DIR:-}" ]]; then
      rm -rf "$TMP_ROOT"
    else
      [[ -z "${FATHOM_ACCEPTANCE_MODELS_DIR:-}" ]] && rm -rf "$MODELS_DIR"
      [[ -z "${FATHOM_ACCEPTANCE_STATE_DIR:-}" ]] && rm -rf "$STATE_DIR"
      [[ -z "${FATHOM_ACCEPTANCE_ARTIFACT_DIR:-}" ]] && rm -rf "$ARTIFACT_DIR"
    fi
  else
    echo "Backend acceptance artifacts preserved at: $ARTIFACT_DIR" >&2
    if [[ $status -ne 0 && -f "$LOG_DIR/server.log" ]]; then
      echo "--- server.log tail ---" >&2
      tail -120 "$LOG_DIR/server.log" >&2 || true
    fi
  fi
}
trap cleanup EXIT

wait_for_backend() {
  local url="$BASE/v1/health"
  for _ in $(seq 1 "$WAIT_SECONDS"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ backend health is responding at $BASE"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for backend at $url" >&2
  return 1
}

need cargo
need curl
need python3
mkdir -p "$MODELS_DIR" "$STATE_DIR" "$LOG_DIR"

if curl -fsS "$BASE/v1/health" >/dev/null 2>&1; then
  echo "Port $PORT already has a responding Fathom backend. Choose another FATHOM_ACCEPTANCE_PORT." >&2
  exit 1
fi

printf '{ truncated model registry' >"$STATE_DIR/models.json"

echo "== start backend-only acceptance server =="
FATHOM_PORT="$PORT" \
FATHOM_STATE_DIR="$STATE_DIR" \
FATHOM_MODELS_DIR="$MODELS_DIR" \
cargo run --release -p fathom-server >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

wait_for_backend

python3 - "$BASE" "$ARTIFACT_DIR" "$REQUEST_TIMEOUT" "$STATE_DIR" "$PORT" "$MODELS_DIR" "$LOG_DIR" "$REPO_COMMIT" "$STARTED_AT" <<'PY'
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")
artifacts = pathlib.Path(sys.argv[2])
timeout = float(sys.argv[3])
state_dir = pathlib.Path(sys.argv[4])
port = int(sys.argv[5])
models_dir = pathlib.Path(sys.argv[6])
log_dir = pathlib.Path(sys.argv[7])
repo_commit = sys.argv[8]
started_at = sys.argv[9]
artifacts.mkdir(parents=True, exist_ok=True)

TINYSTORIES_ID = "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors"
MINILM_ID = "sentence-transformers-all-minilm-l6-v2-model-safetensors"
GGUF_ID = "aladar-llama-2-tiny-random-gguf-llama-2-tiny-random-gguf"

CHECKS = {
    "00-corrupt-state-runtime.json": ("corrupt_state_runtime_warning", "Runtime starts after seeded corrupt model registry and reports model_state_recovered."),
    "00-corrupt-state-files.json": ("corrupt_state_backup_file", "Recovered corrupt model registry was moved aside as a models.json.corrupt-* file."),
    "01-v1-health.json": ("health", "OpenAI-compatible health endpoint responds."),
    "02-v1-models-after-corrupt-recovery.json": ("models_empty_after_recovery", "Runnable model list is empty after corrupt-state recovery."),
    "02b-api-dashboard-after-corrupt-recovery.json": ("dashboard_recovery_warning", "Dashboard runtime payload exposes the corrupt-state recovery warning."),
    "03-api-capabilities.json": ("capabilities", "Capability discovery includes the SafeTensors/HF backend lane."),
    "04-install-tinystories.json": ("install_tinystories", "Pinned TinyStories SafeTensors/HF fixture installs as runnable."),
    "05-v1-models-after-tinystories.json": ("models_include_chat_fixture", "Runnable /v1/models includes the TinyStories chat fixture."),
    "06-chat-non-stream.json": ("chat_non_stream", "TinyStories fixture returns a real non-streaming chat completion with usage and timing metrics."),
    "07-chat-stream-refusal.json": ("chat_stream_refusal", "Streaming chat is truthfully refused with not_implemented."),
    "08-install-minilm-safetensors.json": ("install_minilm_safetensors", "Pinned MiniLM SafeTensors fixture installs as a text embedding model."),
    "09-api-embedding-models.json": ("embedding_models_include_minilm", "Embedding model listing includes the MiniLM fixture."),
    "09b-v1-models-after-minilm.json": ("embedding_model_excluded_from_v1_models", "MiniLM embedding-only fixture stays out of chat-runnable /v1/models."),
    "10-v1-embeddings-minilm.json": ("v1_embeddings_minilm", "OpenAI-style /v1/embeddings returns one finite 384-dimensional float vector from the verified MiniLM runtime."),
    "10b-v1-embeddings-base64-refusal.json": ("v1_embeddings_base64_refusal", "/v1/embeddings truthfully refuses encoding_format=base64 with invalid_request."),
    "10-minilm-embed.json": ("minilm_embedding", "MiniLM SafeTensors embedding endpoint returns a finite normalized 384-dimensional vector."),
    "11-retrieval-create-index.json": ("retrieval_create_index", "Explicit-vector retrieval index can be created."),
    "12-retrieval-add-chunk.json": ("retrieval_add_chunk", "Explicit-vector retrieval chunk can be stored."),
    "13-retrieval-search.json": ("retrieval_search", "Explicit-vector retrieval search returns the expected chunk."),
    "14-chat-with-retrieval.json": ("chat_with_retrieval", "Chat completion records retrieval-context metadata when explicit vector retrieval is supplied."),
    "15-install-gguf-metadata-only.json": ("install_gguf_metadata_only", "Pinned GGUF fixture installs as metadata_only, not runnable."),
    "16-v1-models-after-gguf.json": ("gguf_excluded_from_v1_models", "GGUF metadata-only fixture remains excluded from /v1/models."),
    "17-chat-gguf-refusal.json": ("gguf_chat_refusal", "GGUF metadata-only fixture is truthfully refused for chat generation."),
}

summary = {
    "base_url": base,
    "port": port,
    "repo_commit": repo_commit,
    "started_at": started_at,
    "artifact_dir": ".",
    "state_dir": "state/",
    "model_dir": "models/",
    "log_dir": "logs/",
    "local_paths_file": "summary.local.json",
    "fixture_model_ids": {
        "chat": TINYSTORIES_ID,
        "embedding": MINILM_ID,
        "gguf_metadata_only": GGUF_ID,
    },
    "checks": [],
}
summary_local = {
    "artifact_dir": str(artifacts),
    "state_dir": str(state_dir),
    "model_dir": str(models_dir),
    "log_dir": str(log_dir),
    "server_log": str(log_dir / "server.log"),
}


def save(name, payload, http_status, expected_http=None):
    path = artifacts / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    check_name, description = CHECKS.get(name, (path.stem, "Captured smoke artifact."))
    check = {
        "name": check_name,
        "description": description,
        "artifact": name,
        "http_status": http_status,
        "status": "passed" if expected_http is not None and http_status == expected_http else "recorded",
    }
    if expected_http is not None:
        check["expected_http_status"] = expected_http
    summary["checks"].append(check)
    return payload


def request(method, path, body=None, artifact=None, expected=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    status = None
    raw = b""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read()
    except Exception as e:
        raise AssertionError(f"{method} {path} failed before HTTP response: {e}") from e

    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text) if text else {}
    except json.JSONDecodeError:
        payload = {"raw": text}
    if artifact:
        save(artifact, payload, status, expected)
    if expected is not None and status != expected:
        raise AssertionError(f"{method} {path} expected HTTP {expected}, got {status}: {payload}")
    return status, payload


def assert_finite_number(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError(f"{label} must be finite number, got {value!r}")


# Discovery surfaces, including corrupt model-state recovery from the seeded registry.
_, health = request("GET", "/v1/health", artifact="01-v1-health.json", expected=200)
assert health.get("ok") is True, health
_, runtime = request("GET", "/api/runtime", artifact="00-corrupt-state-runtime.json", expected=200)
assert runtime.get("ready") is True, runtime
warnings = runtime.get("warnings") or []
assert any(warning.get("type") == "model_state_recovered" for warning in warnings), runtime
assert runtime.get("active_model_id") is None and runtime.get("ready_model_count") == 0, runtime
corrupt_files = sorted(path.name for path in state_dir.glob("models.json.corrupt-*"))
save("00-corrupt-state-files.json", {"corrupt_files": corrupt_files}, 200, 200)
assert len(corrupt_files) == 1, corrupt_files
_, models_before_install = request("GET", "/v1/models", artifact="02-v1-models-after-corrupt-recovery.json", expected=200)
assert models_before_install.get("data") == [], models_before_install
_, dashboard = request("GET", "/api/dashboard", artifact="02b-api-dashboard-after-corrupt-recovery.json", expected=200)
assert any(warning.get("type") == "model_state_recovered" for warning in (dashboard.get("runtime", {}).get("warnings") or [])), dashboard
_, capabilities = request("GET", "/api/capabilities", artifact="03-api-capabilities.json", expected=200)
assert any(lane.get("id") == "safetensors-hf" for lane in capabilities.get("backend_lanes", [])), capabilities

# Pinned TinyStories SafeTensors/HF catalog install and chat.
_, tiny = request(
    "POST",
    "/api/models/catalog/install",
    {"repo_id": "vijaymohan/gpt2-tinystories-from-scratch-10m", "filename": "model.safetensors"},
    artifact="04-install-tinystories.json",
    expected=200,
)
assert tiny.get("id") == TINYSTORIES_ID, tiny
assert tiny.get("status") == "ready", tiny
assert tiny.get("capability_status") == "runnable", tiny
assert "safetensors-hf" in tiny.get("backend_lanes", []), tiny

_, models = request("GET", "/v1/models", artifact="05-v1-models-after-tinystories.json", expected=200)
model_ids = [item.get("id") for item in models.get("data", [])]
assert TINYSTORIES_ID in model_ids, models
for item in models.get("data", []):
    fathom = item.get("fathom") or {}
    assert fathom.get("capability_status") == "runnable", item
    assert "safetensors-hf" in fathom.get("backend_lanes", []) or fathom.get("provider_kind") == "external", item

chat_body = {
    "model": TINYSTORIES_ID,
    "messages": [{"role": "user", "content": "Once upon a time, a little robot found a red ball and"}],
    "max_tokens": 24,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
}
_, chat = request("POST", "/v1/chat/completions", chat_body, artifact="06-chat-non-stream.json", expected=200)
assert chat.get("object") == "chat.completion", chat
message = chat.get("choices", [{}])[0].get("message") or {}
assert message.get("role") == "assistant" and isinstance(message.get("content"), str) and message.get("content").strip(), chat
usage = chat.get("usage") or {}
assert usage.get("prompt_tokens", 0) > 0 and usage.get("completion_tokens", 0) > 0 and usage.get("total_tokens", 0) > 0, chat
metrics = (chat.get("fathom") or {}).get("metrics") or {}
for key in ("generation_ms", "total_ms"):
    assert_finite_number(metrics.get(key), f"chat fathom.metrics.{key}")

stream_body = dict(chat_body, stream=True)
_, stream = request("POST", "/v1/chat/completions", stream_body, artifact="07-chat-stream-refusal.json", expected=501)
assert (stream.get("error") or {}).get("type") == "not_implemented", stream

# Pinned MiniLM SafeTensors embedding package and finite 384d vector.
_, minilm = request(
    "POST",
    "/api/models/catalog/install",
    {"repo_id": "sentence-transformers/all-MiniLM-L6-v2", "filename": "model.safetensors"},
    artifact="08-install-minilm-safetensors.json",
    expected=200,
)
assert minilm.get("id") == MINILM_ID, minilm
assert minilm.get("task") == "text_embedding", minilm
_, embedding_models = request("GET", "/api/embedding-models", artifact="09-api-embedding-models.json", expected=200)
assert MINILM_ID in [item.get("id") for item in embedding_models.get("items", [])], embedding_models
_, models_after_minilm = request("GET", "/v1/models", artifact="09b-v1-models-after-minilm.json", expected=200)
assert MINILM_ID not in [item.get("id") for item in models_after_minilm.get("data", [])], models_after_minilm

v1_embeddings_body = {"model": MINILM_ID, "input": "Rust ownership keeps memory safety explicit.", "encoding_format": "float"}
_, v1_embeddings = request(
    "POST",
    "/v1/embeddings",
    v1_embeddings_body,
    artifact="10-v1-embeddings-minilm.json",
    expected=200,
)
assert v1_embeddings.get("object") == "list", v1_embeddings
assert v1_embeddings.get("model") == MINILM_ID, v1_embeddings
v1_data = v1_embeddings.get("data")
assert isinstance(v1_data, list) and len(v1_data) == 1, v1_embeddings
assert v1_data[0].get("object") == "embedding" and v1_data[0].get("index") == 0, v1_embeddings
v1_vector = v1_data[0].get("embedding")
assert isinstance(v1_vector, list) and len(v1_vector) == 384, v1_embeddings
assert all(isinstance(v, float) and math.isfinite(v) for v in v1_vector), v1_embeddings
v1_fathom = v1_embeddings.get("fathom") or {}
assert v1_fathom.get("runtime") == "candle-bert-embeddings", v1_embeddings
assert v1_fathom.get("embedding_dimension") == 384, v1_embeddings
assert v1_fathom.get("scope") == "verified local embedding runtime only", v1_embeddings

_, v1_embeddings_base64 = request(
    "POST",
    "/v1/embeddings",
    {"model": MINILM_ID, "input": "Rust ownership keeps memory safety explicit.", "encoding_format": "base64"},
    artifact="10b-v1-embeddings-base64-refusal.json",
    expected=400,
)
assert (v1_embeddings_base64.get("error") or {}).get("type") == "invalid_request", v1_embeddings_base64

_, embed = request(
    "POST",
    f"/api/embedding-models/{MINILM_ID}/embed",
    {"input": ["Rust ownership keeps memory safety explicit."], "normalize": True},
    artifact="10-minilm-embed.json",
    expected=200,
)
vector = embed.get("data", [{}])[0].get("embedding")
assert embed.get("embedding_dimension") == 384, embed
assert isinstance(vector, list) and len(vector) == 384 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector), embed

# Retrieval create/add/search with explicit vectors.
_, index = request(
    "POST",
    "/api/retrieval-indexes",
    {"id": "acceptance-notes", "embedding_model_id": "external-demo-3d", "embedding_dimension": 3},
    artifact="11-retrieval-create-index.json",
    expected=200,
)
assert index.get("id") == "acceptance-notes" and index.get("embedding_dimension") == 3, index
_, add = request(
    "POST",
    "/api/retrieval-indexes/acceptance-notes/chunks",
    {
        "chunk": {
            "id": "rust-1",
            "document_id": "dev-notes",
            "text": "Rust ownership keeps memory safety explicit.",
            "byte_start": 0,
            "byte_end": 44,
        },
        "vector": [0.9, 0.1, 0.0],
    },
    artifact="12-retrieval-add-chunk.json",
    expected=200,
)
assert add.get("chunk_count") == 1, add
_, search = request(
    "POST",
    "/api/retrieval-indexes/acceptance-notes/search",
    {"vector": [0.85, 0.15, 0.0], "top_k": 1, "metric": "cosine"},
    artifact="13-retrieval-search.json",
    expected=200,
)
assert search.get("hits") and search["hits"][0].get("chunk", {}).get("id") == "rust-1", search

retrieval_chat_body = {
    "model": TINYSTORIES_ID,
    "messages": [{"role": "user", "content": "Use the note if relevant: why does Rust care about ownership?"}],
    "max_tokens": 24,
    "fathom.retrieval": {
        "index_id": "acceptance-notes",
        "query_vector": [0.85, 0.15, 0.0],
        "top_k": 1,
        "metric": "cosine",
        "max_context_chars": 1000,
    },
}
_, retrieval_chat = request("POST", "/v1/chat/completions", retrieval_chat_body, artifact="14-chat-with-retrieval.json", expected=200)
retrieval_meta = (retrieval_chat.get("fathom") or {}).get("retrieval")
assert retrieval_meta and retrieval_meta.get("index", {}).get("id") == "acceptance-notes", retrieval_chat
assert retrieval_meta.get("hits"), retrieval_chat

# Pinned GGUF fixture remains metadata-only, excluded from /v1/models, and refused for chat.
_, gguf = request(
    "POST",
    "/api/models/catalog/install",
    {"repo_id": "aladar/llama-2-tiny-random-GGUF", "filename": "llama-2-tiny-random.gguf"},
    artifact="15-install-gguf-metadata-only.json",
    expected=200,
)
assert gguf.get("id") == GGUF_ID, gguf
assert gguf.get("capability_status") == "metadata_only", gguf
assert gguf.get("format") == "Gguf", gguf
_, models_after_gguf = request("GET", "/v1/models", artifact="16-v1-models-after-gguf.json", expected=200)
ids_after_gguf = [item.get("id") for item in models_after_gguf.get("data", [])]
assert TINYSTORIES_ID in ids_after_gguf, models_after_gguf
assert MINILM_ID not in ids_after_gguf and GGUF_ID not in ids_after_gguf, models_after_gguf
_, gguf_chat = request(
    "POST",
    "/v1/chat/completions",
    {"model": GGUF_ID, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8},
    artifact="17-chat-gguf-refusal.json",
    expected=501,
)
err = gguf_chat.get("error") or {}
assert err.get("type") == "not_implemented" and "metadata-only" in err.get("message", ""), gguf_chat

summary["passed"] = True
summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with (artifacts / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
with (artifacts / "summary.local.json").open("w", encoding="utf-8") as f:
    json.dump(summary_local, f, indent=2, sort_keys=True)
    f.write("\n")

with (artifacts / "summary.md").open("w", encoding="utf-8") as f:
    f.write("# Fathom backend acceptance artifacts\n\n")
    f.write(f"- Repo commit: `{repo_commit}`\n")
    f.write(f"- Base URL: `{base}`\n")
    f.write(f"- Port: `{port}`\n")
    f.write(f"- Started: `{started_at}`\n")
    f.write(f"- Finished: `{summary['finished_at']}`\n")
    f.write("- Artifact directory: `.`\n")
    f.write("- State directory: `state/`\n")
    f.write("- Model directory: `models/`\n")
    f.write("- Server log: `logs/server.log`\n")
    f.write("- Local-only paths: `summary.local.json`\n\n")
    f.write("## Fixture model ids\n\n")
    for label, model_id in summary["fixture_model_ids"].items():
        f.write(f"- {label}: `{model_id}`\n")
    f.write("\n## Artifact index\n\n")
    f.write("| Check | Artifact | HTTP | What it verifies |\n")
    f.write("| --- | --- | ---: | --- |\n")
    for check in summary["checks"]:
        f.write(
            f"| `{check['name']}` | `{check['artifact']}` | {check['http_status']} | {check['description']} |\n"
        )
    f.write("\n## What this smoke does not prove\n\n")
    f.write("- No arbitrary SafeTensors support claim; only the pinned fixtures above are exercised.\n")
    f.write("- No GGUF runtime, tokenizer execution, or generation claim; GGUF is metadata-only/refusal evidence.\n")
    f.write("- No ONNX chat or general ONNX support claim.\n")
    f.write("- No performance claim or benchmark evidence.\n")
print(f"✓ backend acceptance smoke passed; artifacts written under {artifacts}")
PY

echo "Fathom backend acceptance smoke passed."
if [[ "$KEEP_ARTIFACTS" == "1" ]]; then
  echo "Artifacts: $ARTIFACT_DIR"
else
  echo "Temporary state, models, logs, and artifacts will be removed on exit. Set FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 to keep them."
fi
