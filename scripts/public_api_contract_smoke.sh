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

python3 - "$BASE" "${FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR:-}" <<'PY'
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

base = sys.argv[1].rstrip("/")
artifact_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
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
    "proof_scope": "No-download real-backend routing/refusal smoke only. Does not prove model downloads, generation quality, embedding quality, performance, external proxying, or broad model support.",
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
    boundary_lines = [
        f"- {item['boundary']}: {'pass' if item.get('passed') else 'fail'} ({item.get('check')})"
        for item in summary["boundary_checks"]
    ]
    deferred_lines = [
        f"- {item['boundary']}: {item['reason']}"
        for item in summary["deferred_manifest_boundaries"]
    ] or ["- none"]
    md = "\n".join(
        [
            f"# Public contract smoke summary: {status}",
            "",
            f"- Commit: `{summary['commit']}`",
            f"- Manifest: `{summary['manifest']['name']}` (`{summary['manifest']['status']}`)",
            "- Scope: no-download real-backend routing/refusal smoke only; not model quality, performance, downloads, external proxying, or broad runtime evidence.",
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
    boundary_results[boundary] = {
        "boundary": boundary,
        "check": check_id,
        "status": status,
        "code": code,
        "passed": True,
    }


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
    record_boundary("external placeholder chat or activation", "external-placeholder-exclusion-activation-chat-refusal", 501, "external_proxy_not_implemented")

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

print("public API contract smoke passed: manifest-driven health, models, chat refusals, embeddings refusals, external placeholder boundary")
PY
