#!/usr/bin/env python3
"""Regression-test API client examples against a fake loopback Fathom API.

This is intentionally dependency-free and CI-safe: it does not start Fathom,
download models, or exercise runtime behavior. It verifies that public examples
send the expected narrow HTTP contract to Fathom-shaped endpoints.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHAT_MODEL_ID = "echarlaix-tiny-random-phiforcausallm-model-safetensors"
CHAT_REPO_ID = "echarlaix/tiny-random-PhiForCausalLM"
EMBEDDING_MODEL_ID = "sentence-transformers-all-minilm-l6-v2-model-safetensors"
EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: Any | None


class Recorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[RecordedRequest] = []

    def append(self, request: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def snapshot(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self.requests)


class FakeFathomHandler(BaseHTTPRequestHandler):
    server_version = "FakeFathom/1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        return

    @property
    def recorder(self) -> Recorder:
        return self.server.recorder  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        self._record_and_respond()

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook
        self._record_and_respond()

    def _record_and_respond(self) -> None:
        body = self._read_json_body()
        self.recorder.append(
            RecordedRequest(
                method=self.command,
                path=self.path,
                headers={key.lower(): value for key, value in self.headers.items()},
                body=body,
            )
        )
        response = self._response_for(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def _read_json_body(self) -> Any | None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return None
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            self.send_error(400, f"invalid JSON: {exc}")
            return None

    def _response_for(self, body: Any | None) -> dict[str, Any]:
        if self.command == "GET" and self.path == "/v1/health":
            return {"ok": True, "engine": "fathom", "generation_ready": True}
        if self.command == "POST" and self.path == "/api/models/catalog/install":
            repo_id = body.get("repo_id", "unknown/unknown") if isinstance(body, dict) else "unknown/unknown"
            filename = body.get("filename", "model.safetensors") if isinstance(body, dict) else "model.safetensors"
            model_id = EMBEDDING_MODEL_ID if repo_id == EMBEDDING_REPO_ID else CHAT_MODEL_ID
            return {
                "ok": True,
                "status": "available",
                "model": {"id": model_id, "repo_id": repo_id, "filename": filename},
            }
        if self.command == "GET" and self.path == "/v1/models":
            return {
                "object": "list",
                "data": [
                    {
                        "id": CHAT_MODEL_ID,
                        "object": "model",
                        "created": 0,
                        "owned_by": "fathom",
                        "fathom": {
                            "provider_kind": "local",
                            "status": "available",
                            "capability_status": "Runnable",
                            "backend_lanes": ["safetensors-hf"],
                        },
                    }
                ],
            }
        if self.command == "POST" and self.path == "/v1/chat/completions":
            return {
                "id": "chatcmpl-fake-loopback",
                "object": "chat.completion",
                "created": 0,
                "model": CHAT_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello from fake Fathom"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
                "fathom": {"runtime": "fake-loopback", "metrics": {"total_ms": 0}},
            }
        if self.command == "POST" and self.path == "/v1/embeddings":
            return {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.125, -0.25, 0.5], "index": 0}],
                "model": EMBEDDING_MODEL_ID,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "fathom": {
                    "runtime": "candle-bert-embeddings",
                    "embedding_dimension": 3,
                    "scope": "verified local embedding runtime only",
                },
            }
        return {"ok": False, "error": {"message": f"unexpected {self.command} {self.path}"}}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_example(command: list[str], *, embeddings: bool) -> list[RecordedRequest]:
    recorder = Recorder()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeFathomHandler)
    server.recorder = recorder  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake_base = f"http://127.0.0.1:{server.server_port}"
    env = os.environ.copy()
    env.update(
        {
            "FATHOM_BASE_URL": fake_base,
            "FATHOM_MODEL_ID": CHAT_MODEL_ID,
            "FATHOM_REPO_ID": CHAT_REPO_ID,
            "FATHOM_FILENAME": "model.safetensors",
            "FATHOM_EMBEDDING_MODEL_ID": EMBEDDING_MODEL_ID,
            "FATHOM_EMBEDDING_REPO_ID": EMBEDDING_REPO_ID,
            "FATHOM_EMBEDDING_FILENAME": "model.safetensors",
        }
    )
    if embeddings:
        env["FATHOM_RUN_EMBEDDINGS"] = "1"
    else:
        env.pop("FATHOM_RUN_EMBEDDINGS", None)
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True, text=True, capture_output=True, timeout=30)
        requests = recorder.snapshot()
        assert_public_contract(requests, embeddings=embeddings, label=" ".join(command))
        return requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def assert_public_contract(requests: list[RecordedRequest], *, embeddings: bool, label: str) -> None:
    paths = [(request.method, request.path) for request in requests]
    expected_prefix = [
        ("GET", "/v1/health"),
        ("POST", "/api/models/catalog/install"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]
    assert_true(paths[:4] == expected_prefix, f"{label}: expected initial endpoints {expected_prefix}, got {paths}")
    assert_true(("GET", "/v1/models") in paths, f"{label}: /v1/models was not called")

    install_requests = [r for r in requests if r.method == "POST" and r.path == "/api/models/catalog/install"]
    assert_true(len(install_requests) == (2 if embeddings else 1), f"{label}: unexpected install request count")
    for request in install_requests:
        assert_json_request(request, label)
        assert_true(isinstance(request.body, dict), f"{label}: install body must be a JSON object")
        repo_id = request.body.get("repo_id")
        filename = request.body.get("filename")
        assert_true(isinstance(repo_id, str) and re.fullmatch(r"[^/\s]+/[^\s]+", repo_id) is not None, f"{label}: invalid catalog repo_id {repo_id!r}")
        assert_true(isinstance(filename, str) and filename.endswith(".safetensors"), f"{label}: invalid catalog filename {filename!r}")

    chat_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/chat/completions"]
    assert_true(len(chat_requests) == 1, f"{label}: expected exactly one chat request")
    chat = chat_requests[0]
    assert_json_request(chat, label)
    assert_true(isinstance(chat.body, dict), f"{label}: chat body must be a JSON object")
    assert_true(chat.body.get("stream") in (None, False), f"{label}: examples must use non-streaming chat")
    assert_true(isinstance(chat.body.get("messages"), list) and chat.body["messages"], f"{label}: chat messages missing")

    embedding_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/embeddings"]
    if embeddings:
        assert_true(len(embedding_requests) == 1, f"{label}: expected one opt-in embeddings request")
        embedding = embedding_requests[0]
        assert_json_request(embedding, label)
        assert_true(isinstance(embedding.body, dict), f"{label}: embeddings body must be a JSON object")
        assert_true(embedding.body.get("encoding_format") == "float", f"{label}: embeddings must request float encoding")
    else:
        assert_true(not embedding_requests, f"{label}: embeddings request must be opt-in only")


def assert_json_request(request: RecordedRequest, label: str) -> None:
    content_type = request.headers.get("content-type", "")
    assert_true(content_type.split(";", 1)[0].lower() == "application/json", f"{label}: {request.path} missing JSON content-type")


def static_checks() -> None:
    http_path = ROOT / "examples/api/fathom.http"
    http_text = http_path.read_text(encoding="utf-8")
    for endpoint in (
        "GET {{base}}/v1/health",
        "POST {{base}}/api/models/catalog/install",
        "GET {{base}}/v1/models",
        "POST {{base}}/v1/chat/completions",
        "POST {{base}}/v1/embeddings",
    ):
        assert_true(endpoint in http_text, f"fathom.http missing endpoint: {endpoint}")
    chat_block = http_text.split("POST {{base}}/v1/chat/completions", 1)[1].split("### Optional:", 1)[0]
    assert_true('"stream": true' not in chat_block, "fathom.http must not document streaming chat")
    embedding_block = http_text.split("POST {{base}}/v1/embeddings", 1)[1]
    assert_true('"encoding_format": "float"' in embedding_block, "fathom.http embeddings must request float encoding")

    sdk_text = (ROOT / "examples/api/openai-sdk.py").read_text(encoding="utf-8")
    assert_true("client.chat.completions.create" in sdk_text, "openai-sdk.py missing chat completion call")
    assert_true("stream=True" not in sdk_text and '"stream": true' not in sdk_text, "openai-sdk.py must stay non-streaming")
    assert_true('encoding_format="float"' in sdk_text, "openai-sdk.py embeddings must request float encoding")


def main() -> int:
    static_checks()
    run_example(["bash", "examples/api/curl-quickstart.sh"], embeddings=False)
    run_example(["python3", "examples/api/python-no-deps.py"], embeddings=False)
    run_example(["bash", "examples/api/curl-quickstart.sh"], embeddings=True)
    run_example(["python3", "examples/api/python-no-deps.py"], embeddings=True)
    print("API client examples regression passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"api client examples regression failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
