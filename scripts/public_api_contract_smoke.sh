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

python3 - "$BASE" "${FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR:-}" "$RUN_DIR" <<'PY'
import json
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

base = sys.argv[1].rstrip("/")
artifact_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
run_dir = Path(sys.argv[3])
root = Path.cwd()
manifest_path = root / "docs" / "api" / "public-contract.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

summary = {
    "schema": "fathom.public_contract_smoke.summary.v1",
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "commit": None,
    "manifest": {
        "path": "docs/api/public-contract.json",
        "name": manifest.get("name"),
        "status": manifest.get("status"),
    },
    "passed": False,
    "proof_scope": "No-download real-backend routing/refusal smoke only. Does not prove model downloads, generation quality, embedding quality, performance, external proxying, a GGUF runtime, tokenizer execution, or generation claim, or broad model support.",
    "endpoint_checks": [],
    "boundary_checks": [],
    "deferred_manifest_boundaries": [],
}

try:
    summary["commit"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    summary["commit"] = "unknown"

endpoint_results = {}
boundary_results = {}
manifest_boundaries = {
    item.get("boundary"): item
    for item in manifest.get("expected_boundary_errors", [])
    if isinstance(item, dict)
}


def write_summary(passed):
    summary["passed"] = bool(passed)
    if not artifact_dir:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "public-contract-smoke-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = "PASS" if passed else "FAIL"
    endpoint_lines = [
        f"- {item['method']} {item['path']}: {'pass' if item['passed'] else 'fail'} ({', '.join(item['checks'])})"
        for item in summary["endpoint_checks"]
    ]
    boundary_lines = []
    for item in summary["boundary_checks"]:
        hint = f"; hint `{item['request_hint']}`" if item.get("request_hint") else ""
        boundary_lines.append(
            f"- {item['boundary']}: {'pass' if item.get('passed') else 'fail'} ({item.get('check')}{hint})"
        )
    deferred_lines = [
        f"- {item['boundary']}: {item['reason']}"
        for item in summary["deferred_manifest_boundaries"]
    ] or ["- none"]
    failure_note = []
    if not passed:
        failure_note = [
            "",
            "This failed smoke summary is partial diagnostic evidence only; it must not be treated as a passed public contract smoke.",
        ]
    md = "\n".join(
        [
            f"# Public contract smoke summary: {status}",
            "",
            f"- Commit: `{summary['commit']}`",
            f"- Manifest: `{summary['manifest']['path']}` / `{summary['manifest']['name']}` (`{summary['manifest']['status']}`)",
            "- Scope: no-download real-backend routing/refusal smoke only; does not prove model downloads, generation quality, embedding quality, performance, external proxying, a GGUF runtime, tokenizer execution, or generation claim, or broad model support.",
            *failure_note,
            "",
            "## Endpoint checks",
            *endpoint_lines,
            "",
            "## Boundary checks",
            *boundary_lines,
            "",
            "## Manifest boundaries not exercised by this no-download smoke",
            *deferred_lines,
            "",
        ]
    )
    (artifact_dir / "public-contract-smoke-summary.md").write_text(md, encoding="utf-8")


def record_endpoint(method, path, check_id):
    endpoint_results.setdefault((method, path), []).append(check_id)


def record_boundary(boundary, check_id, status=None, code=None):
    result = {
        "boundary": boundary,
        "check": check_id,
        "status": status,
        "code": code,
        "passed": True,
    }
    request_hint = manifest_boundaries.get(boundary, {}).get("request_hint")
    if request_hint:
        result["request_hint"] = request_hint
    boundary_results[boundary] = result


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
            raise AssertionError(f"{method} {path} returned non-JSON error body") from error
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


def write_gguf_string(buffer, value):
    encoded = value.encode("utf-8")
    buffer.extend(struct.pack("<Q", len(encoded)))
    buffer.extend(encoded)


def write_gguf_string_kv(buffer, key, value):
    write_gguf_string(buffer, key)
    buffer.extend(struct.pack("<I", 8))
    write_gguf_string(buffer, value)


def write_gguf_u32_kv(buffer, key, value):
    write_gguf_string(buffer, key)
    buffer.extend(struct.pack("<I", 4))
    buffer.extend(struct.pack("<I", value))


def write_minimal_metadata_only_gguf(path):
    buffer = bytearray()
    buffer.extend(b"GGUF")
    buffer.extend(struct.pack("<I", 3))
    buffer.extend(struct.pack("<Q", 0))  # tensor_count
    buffer.extend(struct.pack("<Q", 3))  # metadata_kv_count
    write_gguf_string_kv(buffer, "general.architecture", "llama")
    write_gguf_u32_kv(buffer, "general.alignment", 32)
    write_gguf_string_kv(buffer, "tokenizer.ggml.model", "llama")
    path.write_bytes(buffer)


def verify_manifest_coverage():
    endpoints = manifest.get("supported_endpoints", [])
    summary["endpoint_checks"] = []
    for endpoint in endpoints:
        key = (endpoint.get("method"), endpoint.get("path"))
        checks = endpoint_results.get(key, [])
        if not checks:
            raise AssertionError(f"manifest endpoint {key[0]} {key[1]} has no real-backend smoke check")
        summary["endpoint_checks"].append(
            {"method": key[0], "path": key[1], "checks": checks, "passed": True}
        )

    required_no_download_boundaries = {
        "streaming chat completions",
        "base64 embeddings",
        "external placeholder chat or activation",
        "embedding models in /v1/models",
        "PyTorch .bin execution",
        "unsupported ONNX chat or general ONNX model execution",
        "unverified SafeTensors/Hugging Face model execution",
        "GGUF metadata-only chat attempts",
        "missing chat model",
        "unknown embedding model",
    }
    summary["boundary_checks"] = []
    expected = manifest.get("expected_boundary_errors", [])
    expected_by_name = {item.get("boundary"): item for item in expected}
    for boundary in sorted(required_no_download_boundaries):
        if boundary not in expected_by_name:
            raise AssertionError(f"manifest missing expected no-download boundary {boundary!r}")
        result = boundary_results.get(boundary)
        if not result:
            raise AssertionError(f"manifest boundary {boundary!r} has no real-backend smoke check")
        expected_item = expected_by_name[boundary]
        if "status" in expected_item and result.get("status") != expected_item["status"]:
            raise AssertionError(f"manifest boundary {boundary!r} status drift: {result}")
        if "code" in expected_item and result.get("code") != expected_item["code"]:
            raise AssertionError(f"manifest boundary {boundary!r} code drift: {result}")
        summary["boundary_checks"].append(result)

    deferred = []
    for item in expected:
        boundary = item.get("boundary")
        if boundary not in required_no_download_boundaries:
            deferred.append(
                {
                    "boundary": boundary,
                    "reason": "requires downloaded/registered model state or is a non-claim boundary outside the no-download smoke",
                }
            )
    summary["deferred_manifest_boundaries"] = deferred


try:
    status, health = request("GET", "/v1/health")
    assert status == 200, (status, health)
    assert health.get("ok") is True, health
    assert health.get("engine") == "fathom", health
    assert health.get("generation_ready") is False, health
    record_endpoint("GET", "/v1/health", "empty-health-ready-false")

    status, models = request("GET", "/v1/models")
    assert status == 200, (status, models)
    assert models.get("object") == "list", models
    assert models.get("data") == [], models
    record_endpoint("GET", "/v1/models", "empty-chat-runnable-model-list")
    record_boundary("embedding models in /v1/models", "empty-state-v1-models-exclusion")

    chat_body = {
        "model": "missing-chat-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    status, stream_refusal = request("POST", "/v1/chat/completions", chat_body)
    assert status == 501, (status, stream_refusal)
    assert_error(stream_refusal, "not_implemented")
    assert_no_chat_success(stream_refusal)
    record_endpoint("POST", "/v1/chat/completions", "streaming-refusal")
    record_boundary("streaming chat completions", "stream-true-refusal", status, "not_implemented")

    chat_body["stream"] = False
    status, missing_chat = request("POST", "/v1/chat/completions", chat_body)
    assert status == 400, (status, missing_chat)
    assert_error(missing_chat, "model_not_found")
    assert_no_chat_success(missing_chat)
    record_endpoint("POST", "/v1/chat/completions", "missing-model-refusal")
    record_boundary("missing chat model", "missing-chat-model-refusal", status, "model_not_found")

    embedding_body = {
        "model": "missing-embedding-model",
        "input": "hello",
        "encoding_format": "base64",
    }
    status, base64_refusal = request("POST", "/v1/embeddings", embedding_body)
    assert status == 400, (status, base64_refusal)
    assert_error(base64_refusal, "invalid_request")
    assert_no_embedding_success(base64_refusal)
    record_endpoint("POST", "/v1/embeddings", "base64-refusal")
    record_boundary("base64 embeddings", "base64-encoding-refusal", status, "invalid_request")

    embedding_body["encoding_format"] = "float"
    status, missing_embedding = request("POST", "/v1/embeddings", embedding_body)
    assert status == 404, (status, missing_embedding)
    assert_error(missing_embedding, "embedding_model_not_found")
    assert_no_embedding_success(missing_embedding)
    record_endpoint("POST", "/v1/embeddings", "unknown-embedding-model-refusal")
    record_boundary("unknown embedding model", "unknown-embedding-model-refusal", status, "embedding_model_not_found")

    external_body = {
        "id": "external-placeholder-smoke",
        "name": "External placeholder smoke",
        "source": "OpenAI-compatible",
        "api_base": "https://api.example.test/v1",
        "api_key": "placeholder-key",
        "model_name": "placeholder-model",
    }
    status, capabilities = request("GET", "/api/capabilities")
    assert status == 200, (status, capabilities)
    external_lane = next(
        lane for lane in capabilities.get("backend_lanes", []) if lane.get("id") == "external-openai"
    )
    assert str(external_lane.get("status", "")).lower() == "planned", external_lane
    assert "proxying is not implemented" in external_lane.get("summary", "").lower(), external_lane
    assert external_lane.get("blockers"), external_lane

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
    assert_error(external_chat, "external_proxy_not_implemented")
    assert_no_chat_success(external_chat)
    record_boundary("external placeholder chat or activation", "external-placeholder-capability-planned-exclusion-activation-chat-refusal", 501, "external_proxy_not_implemented")

    pytorch_fixture = run_dir / "pytorch-bin-fixture"
    pytorch_fixture.mkdir(parents=True, exist_ok=True)
    pytorch_artifact = pytorch_fixture / "pytorch_model.bin"
    pytorch_artifact.write_bytes(b"synthetic pickle-like bytes; do not deserialize")
    status, pytorch_model = request(
        "POST",
        "/api/models/register",
        {
            "id": "unsafe-pytorch-bin-smoke",
            "name": "Unsafe PyTorch bin smoke fixture",
            "model_path": str(pytorch_artifact),
        },
    )
    assert status == 200, (status, pytorch_model)
    assert pytorch_model.get("capability_status") == "blocked", pytorch_model
    assert pytorch_model.get("format") == "PyTorchBin", pytorch_model
    assert "pytorch-trusted-import" in pytorch_model.get("backend_lanes", []), pytorch_model
    combined_pytorch_text = " ".join(
        str(pytorch_model.get(key, "")) for key in ("install_error", "capability_summary")
    ).lower()
    assert "pytorch" in combined_pytorch_text, pytorch_model
    assert "pickle" in combined_pytorch_text, pytorch_model
    assert "execute code" in combined_pytorch_text, pytorch_model
    assert "not runnable" in combined_pytorch_text, pytorch_model

    status, models_after_pytorch = request("GET", "/v1/models")
    assert status == 200, (status, models_after_pytorch)
    assert not any(model.get("id") == "unsafe-pytorch-bin-smoke" for model in models_after_pytorch.get("data", [])), models_after_pytorch

    status, pytorch_chat = request(
        "POST",
        "/v1/chat/completions",
        {"model": "unsafe-pytorch-bin-smoke", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 501, (status, pytorch_chat)
    assert_error(pytorch_chat, "not_implemented")
    assert_no_chat_success(pytorch_chat)
    pytorch_message = pytorch_chat["error"]["message"]
    assert "PyTorch" in pytorch_message, pytorch_chat
    assert "pickle" in pytorch_message, pytorch_chat
    assert ".bin" in pytorch_message, pytorch_chat
    assert "blocked" in pytorch_message, pytorch_chat
    assert "No fake inference" in pytorch_message, pytorch_chat
    record_boundary("PyTorch .bin execution", "synthetic-local-pytorch-bin-registration-and-chat-refusal", 501, "not_implemented")

    onnx_fixture = run_dir / "onnx-chat-fixture"
    onnx_fixture.mkdir(parents=True, exist_ok=True)
    onnx_artifact = onnx_fixture / "model.onnx"
    onnx_artifact.write_bytes(b"synthetic onnx bytes; do not load with ONNX Runtime")
    status, onnx_model = request(
        "POST",
        "/api/models/register",
        {
            "id": "unsupported-onnx-chat-smoke",
            "name": "Unsupported ONNX chat smoke fixture",
            "model_path": str(onnx_artifact),
        },
    )
    assert status == 200, (status, onnx_model)
    assert onnx_model.get("capability_status") == "planned", onnx_model
    assert str(onnx_model.get("format", "")).lower() == "onnx", onnx_model
    assert "onnx" in onnx_model.get("backend_lanes", []), onnx_model
    combined_onnx_text = " ".join(
        str(onnx_model.get(key, "")) for key in ("install_error", "capability_summary")
    ).lower()
    assert "onnx" in combined_onnx_text, onnx_model
    assert "not runnable" in combined_onnx_text, onnx_model
    assert "chat-ready" not in combined_onnx_text, onnx_model

    status, models_after_onnx = request("GET", "/v1/models")
    assert status == 200, (status, models_after_onnx)
    assert not any(model.get("id") == "unsupported-onnx-chat-smoke" for model in models_after_onnx.get("data", [])), models_after_onnx

    status, onnx_chat = request(
        "POST",
        "/v1/chat/completions",
        {"model": "unsupported-onnx-chat-smoke", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 501, (status, onnx_chat)
    assert_error(onnx_chat, "not_implemented")
    assert_no_chat_success(onnx_chat)
    onnx_message = onnx_chat["error"]["message"]
    assert "ONNX" in onnx_message, onnx_chat
    assert "not chat-runnable" in onnx_message, onnx_chat
    assert "not supported" in onnx_message, onnx_chat
    assert "non-default pinned MiniLM embedding" in onnx_message, onnx_chat
    assert "No fake inference" in onnx_message, onnx_chat
    record_boundary("unsupported ONNX chat or general ONNX model execution", "synthetic-local-onnx-registration-and-chat-refusal", 501, "not_implemented")

    safetensors_fixture = run_dir / "unverified-safetensors-hf-fixture"
    safetensors_fixture.mkdir(parents=True, exist_ok=True)
    (safetensors_fixture / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (safetensors_fixture / "tokenizer.json").write_text("{}", encoding="utf-8")
    (safetensors_fixture / "model.safetensors").write_bytes(b"synthetic safetensors bytes; do not load")
    status, safetensors_model = request(
        "POST",
        "/api/models/register",
        {
            "id": "unverified-safetensors-hf-smoke",
            "name": "Unverified SafeTensors HF smoke fixture",
            "model_path": str(safetensors_fixture),
        },
    )
    assert status == 200, (status, safetensors_model)
    assert safetensors_model.get("capability_status") == "planned", safetensors_model
    assert safetensors_model.get("format") == "SafeTensors", safetensors_model
    assert "safetensors-hf" in safetensors_model.get("backend_lanes", []), safetensors_model
    combined_safetensors_text = " ".join(
        str(safetensors_model.get(key, "")) for key in ("install_error", "capability_summary")
    ).lower()
    assert "safetensors" in combined_safetensors_text, safetensors_model
    assert "not runnable" in combined_safetensors_text, safetensors_model

    status, models_after_safetensors = request("GET", "/v1/models")
    assert status == 200, (status, models_after_safetensors)
    assert not any(
        model.get("id") == "unverified-safetensors-hf-smoke"
        for model in models_after_safetensors.get("data", [])
    ), models_after_safetensors

    status, safetensors_chat = request(
        "POST",
        "/v1/chat/completions",
        {"model": "unverified-safetensors-hf-smoke", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 501, (status, safetensors_chat)
    assert_error(safetensors_chat, "not_implemented")
    assert_no_chat_success(safetensors_chat)
    safetensors_message = safetensors_chat["error"]["message"]
    assert "No fake inference" in safetensors_message, safetensors_chat
    record_boundary("unverified SafeTensors/Hugging Face model execution", "synthetic-local-unverified-safetensors-hf-registration-and-chat-refusal", 501, "not_implemented")

    gguf_fixture = run_dir / "metadata-only-gguf-fixture"
    gguf_fixture.mkdir(parents=True, exist_ok=True)
    gguf_artifact = gguf_fixture / "synthetic-metadata-only.gguf"
    write_minimal_metadata_only_gguf(gguf_artifact)
    status, gguf_model = request(
        "POST",
        "/api/models/register",
        {
            "id": "metadata-only-gguf-smoke",
            "name": "Metadata-only GGUF smoke fixture",
            "model_path": str(gguf_artifact),
        },
    )
    assert status == 200, (status, gguf_model)
    assert gguf_model.get("capability_status") == "metadata_only", gguf_model
    assert gguf_model.get("format") == "Gguf", gguf_model
    assert "gguf-native" in gguf_model.get("backend_lanes", []), gguf_model
    combined_gguf_text = " ".join(
        str(gguf_model.get(key, "")) for key in ("install_error", "capability_summary")
    ).lower()
    assert "gguf" in combined_gguf_text, gguf_model
    assert "metadata-only" in combined_gguf_text, gguf_model
    assert "not runnable" in combined_gguf_text, gguf_model
    assert "runtime weight loading" in combined_gguf_text, gguf_model
    assert "generation" in combined_gguf_text, gguf_model

    status, models_after_gguf = request("GET", "/v1/models")
    assert status == 200, (status, models_after_gguf)
    assert not any(model.get("id") == "metadata-only-gguf-smoke" for model in models_after_gguf.get("data", [])), models_after_gguf

    status, gguf_chat = request(
        "POST",
        "/v1/chat/completions",
        {"model": "metadata-only-gguf-smoke", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 501, (status, gguf_chat)
    assert_error(gguf_chat, "not_implemented")
    assert_no_chat_success(gguf_chat)
    gguf_message = gguf_chat["error"]["message"]
    assert "GGUF" in gguf_message, gguf_chat
    assert "metadata-only" in gguf_message, gguf_chat
    assert "not runnable" in gguf_message, gguf_chat
    assert "public/runtime tokenizer execution" in gguf_message, gguf_chat
    assert "runtime weight loading" in gguf_message, gguf_chat
    assert "generation" in gguf_message, gguf_chat
    assert "No fake inference" in gguf_message, gguf_chat
    record_boundary("GGUF metadata-only chat attempts", "synthetic-local-gguf-metadata-registration-and-chat-refusal", 501, "not_implemented")

    verify_manifest_coverage()
    write_summary(True)
except Exception:
    try:
        summary["endpoint_checks"] = [
            {"method": method, "path": path, "checks": checks, "passed": True}
            for (method, path), checks in sorted(endpoint_results.items())
        ]
        summary["boundary_checks"] = list(boundary_results.values())
        write_summary(False)
    finally:
        raise

print("public API contract smoke passed: manifest-driven health, models, chat refusals, embeddings refusals, external placeholder boundary, synthetic PyTorch .bin refusal, synthetic ONNX chat/general refusal, synthetic unverified SafeTensors/HF refusal, synthetic GGUF metadata-only refusal, capabilities external metadata-only guard")
PY
