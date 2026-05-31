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
import tempfile
import textwrap
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CHAT_MODEL_ID = "echarlaix-tiny-random-phiforcausallm-model-safetensors"
CHAT_REPO_ID = "echarlaix/tiny-random-PhiForCausalLM"
EMBEDDING_MODEL_ID = "sentence-transformers-all-minilm-l6-v2-model-safetensors"
EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
PUBLIC_CONTRACT = ROOT / "docs" / "api" / "public-contract.json"


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
        status, response = self._response_for(body)
        self.send_response(status)
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

    def _response_for(self, body: Any | None) -> tuple[int, dict[str, Any]]:
        if self.command == "GET" and self.path == "/v1/health":
            return 200, {"ok": True, "engine": "fathom", "generation_ready": True}
        if self.command == "POST" and self.path == "/api/models/catalog/install":
            repo_id = body.get("repo_id", "unknown/unknown") if isinstance(body, dict) else "unknown/unknown"
            filename = body.get("filename", "model.safetensors") if isinstance(body, dict) else "model.safetensors"
            model_id = EMBEDDING_MODEL_ID if repo_id == EMBEDDING_REPO_ID else CHAT_MODEL_ID
            return 200, {
                "ok": True,
                "status": "available",
                "model": {"id": model_id, "repo_id": repo_id, "filename": filename},
            }
        if self.command == "GET" and self.path == "/v1/models":
            return 200, {
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
            return 200, {
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
            return 200, {
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
        return 404, {
            "error": {
                "message": f"unexpected fake-loopback endpoint {self.command} {self.path}",
                "type": "invalid_request_error",
                "code": "unexpected_example_endpoint",
                "param": None,
            }
        }


def load_public_contract() -> dict[str, Any]:
    data = json.loads(PUBLIC_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("public contract manifest must be a JSON object")
    return data


def allowed_example_endpoints() -> set[tuple[str, str]]:
    manifest = load_public_contract()
    allowed: set[tuple[str, str]] = set()
    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        if not isinstance(method, str) or not isinstance(path, str):
            raise AssertionError("manifest supported_endpoints entries must include string method/path")
        allowed.add((method, path))
    for item in manifest.get("non_contract_surfaces_allowed_in_examples", []):
        if not isinstance(item, str):
            raise AssertionError("manifest non-contract example surfaces must be strings")
        try:
            method, path = item.split(" ", 1)
        except ValueError as exc:
            raise AssertionError(f"invalid non-contract example surface {item!r}") from exc
        if path.startswith("/v1/"):
            raise AssertionError(f"non-contract example surface must not be a /v1 endpoint: {item}")
        allowed.add((method, path))
    return allowed


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_example(
    command: list[str],
    *,
    embeddings: bool,
    extra_env: dict[str, str] | None = None,
    contract: str = "quickstart",
    stdout_checker: Callable[[str], None] | None = None,
) -> list[RecordedRequest]:
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
    if extra_env:
        env.update(extra_env)
    if embeddings:
        env["FATHOM_RUN_EMBEDDINGS"] = "1"
    else:
        env.pop("FATHOM_RUN_EMBEDDINGS", None)
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, check=True, text=True, capture_output=True, timeout=30)
        if stdout_checker is not None:
            stdout_checker(completed.stdout)
        requests = recorder.snapshot()
        if contract == "sdk":
            assert_openai_sdk_contract(requests, embeddings=embeddings, label=" ".join(command))
        else:
            assert_public_contract(requests, embeddings=embeddings, label=" ".join(command))
        return requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_openai_sdk_example_with_stub(*, embeddings: bool) -> list[RecordedRequest]:
    """Run the OpenAI SDK example through a tiny local openai module stub."""

    stub_source = r'''
import json
import urllib.request


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def model_dump_json(self, indent=None):
        return json.dumps(self._payload, indent=indent)


class _ChatCompletions:
    def __init__(self, base_url):
        self._base_url = base_url

    def create(self, **kwargs):
        return _Response(_post_json(f"{self._base_url}/chat/completions", kwargs))


class _Chat:
    def __init__(self, base_url):
        self.completions = _ChatCompletions(base_url)


class _Embeddings:
    def __init__(self, base_url):
        self._base_url = base_url

    def create(self, **kwargs):
        return _Response(_post_json(f"{self._base_url}/embeddings", kwargs))


class OpenAI:
    def __init__(self, *, base_url, api_key):
        del api_key
        self._base_url = str(base_url).rstrip("/")
        self.chat = _Chat(self._base_url)
        self.embeddings = _Embeddings(self._base_url)


def _post_json(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))
'''
    with tempfile.TemporaryDirectory(prefix="fathom-openai-sdk-stub-") as tmpdir:
        package = Path(tmpdir) / "openai"
        package.mkdir()
        (package / "__init__.py").write_text(textwrap.dedent(stub_source).lstrip(), encoding="utf-8")
        return run_example(
            ["python3", "examples/api/openai-sdk.py"],
            embeddings=embeddings,
            extra_env={"PYTHONPATH": prepend_pythonpath(tmpdir, os.environ.get("PYTHONPATH"))},
            contract="sdk",
            stdout_checker=lambda stdout: assert_openai_sdk_stdout(stdout, embeddings=embeddings),
        )


def prepend_pythonpath(path: str, existing: str | None) -> str:
    if existing:
        return os.pathsep.join((path, existing))
    return path


def assert_public_contract(requests: list[RecordedRequest], *, embeddings: bool, label: str) -> None:
    paths = [(request.method, request.path) for request in requests]
    allowed_paths = allowed_example_endpoints()
    if not embeddings:
        allowed_paths = {item for item in allowed_paths if item != ("POST", "/v1/embeddings")}
    unexpected_paths = [path for path in paths if path not in allowed_paths]
    assert_true(not unexpected_paths, f"{label}: unexpected non-contract endpoints called: {unexpected_paths}")
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
    expected_install_repos = [CHAT_REPO_ID] + ([EMBEDDING_REPO_ID] if embeddings else [])
    actual_install_repos: list[str] = []
    for request in install_requests:
        assert_json_request(request, label)
        assert_true(isinstance(request.body, dict), f"{label}: install body must be a JSON object")
        repo_id = request.body.get("repo_id")
        filename = request.body.get("filename")
        assert_true(isinstance(repo_id, str) and re.fullmatch(r"[^/\s]+/[^\s]+", repo_id) is not None, f"{label}: invalid catalog repo_id {repo_id!r}")
        assert_true(isinstance(filename, str) and filename.endswith(".safetensors"), f"{label}: invalid catalog filename {filename!r}")
        actual_install_repos.append(repo_id)
        assert_true(
            request.body.get("accept_license") in (None, False) and request.body.get("accepted_license") in (None, False),
            f"{label}: default permissive catalog examples should not require a license acknowledgement field",
        )
    assert_true(
        actual_install_repos == expected_install_repos,
        f"{label}: expected install repo order {expected_install_repos}, got {actual_install_repos}",
    )

    chat_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/chat/completions"]
    assert_true(len(chat_requests) == 1, f"{label}: expected exactly one chat request")
    chat = chat_requests[0]
    assert_json_request(chat, label)
    assert_true(isinstance(chat.body, dict), f"{label}: chat body must be a JSON object")
    assert_true(chat.body.get("model") == CHAT_MODEL_ID, f"{label}: chat model id drifted from fake /v1/models fixture")
    assert_true(chat.body.get("stream") in (None, False), f"{label}: examples must use non-streaming chat")
    assert_true(isinstance(chat.body.get("messages"), list) and chat.body["messages"], f"{label}: chat messages missing")

    embedding_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/embeddings"]
    if embeddings:
        assert_true(len(embedding_requests) == 1, f"{label}: expected one opt-in embeddings request")
        embedding = embedding_requests[0]
        assert_json_request(embedding, label)
        assert_true(isinstance(embedding.body, dict), f"{label}: embeddings body must be a JSON object")
        assert_true(
            embedding.body.get("model") == EMBEDDING_MODEL_ID,
            f"{label}: embeddings model id drifted from fake install response",
        )
        assert_true(embedding.body.get("encoding_format") == "float", f"{label}: embeddings must request float encoding")
    else:
        assert_true(not embedding_requests, f"{label}: embeddings request must be opt-in only")


def assert_openai_sdk_contract(requests: list[RecordedRequest], *, embeddings: bool, label: str) -> None:
    paths = [(request.method, request.path) for request in requests]
    expected_paths = [("POST", "/v1/chat/completions")]
    if embeddings:
        expected_paths.extend([("POST", "/api/models/catalog/install"), ("POST", "/v1/embeddings")])
    assert_true(paths == expected_paths, f"{label}: expected SDK endpoints {expected_paths}, got {paths}")

    allowed_paths = allowed_example_endpoints()
    unexpected_paths = [path for path in paths if path not in allowed_paths]
    assert_true(not unexpected_paths, f"{label}: unexpected non-contract endpoints called: {unexpected_paths}")

    chat = requests[0]
    assert_json_request(chat, label)
    assert_true(isinstance(chat.body, dict), f"{label}: SDK chat body must be a JSON object")
    assert_true(chat.body.get("model") == CHAT_MODEL_ID, f"{label}: SDK chat model id drifted")
    assert_true(chat.body.get("stream") in (None, False), f"{label}: SDK example must use non-streaming chat")
    assert_true(isinstance(chat.body.get("messages"), list) and chat.body["messages"], f"{label}: SDK chat messages missing")

    install_requests = [r for r in requests if r.method == "POST" and r.path == "/api/models/catalog/install"]
    embedding_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/embeddings"]
    if embeddings:
        assert_true(len(install_requests) == 1, f"{label}: SDK embeddings should install exactly one embedding fixture")
        install = install_requests[0]
        assert_json_request(install, label)
        assert_true(isinstance(install.body, dict), f"{label}: SDK embedding install body must be a JSON object")
        assert_true(install.body.get("repo_id") == EMBEDDING_REPO_ID, f"{label}: SDK embedding repo id drifted")
        assert_true(install.body.get("filename") == "model.safetensors", f"{label}: SDK embedding filename drifted")

        assert_true(len(embedding_requests) == 1, f"{label}: SDK expected one opt-in embeddings request")
        embedding = embedding_requests[0]
        assert_json_request(embedding, label)
        assert_true(isinstance(embedding.body, dict), f"{label}: SDK embeddings body must be a JSON object")
        assert_true(embedding.body.get("model") == EMBEDDING_MODEL_ID, f"{label}: SDK embeddings model id drifted")
        assert_true(embedding.body.get("encoding_format") == "float", f"{label}: SDK embeddings must request float encoding")
    else:
        assert_true(not install_requests, f"{label}: SDK embedding install must be opt-in only")
        assert_true(not embedding_requests, f"{label}: SDK embeddings request must be opt-in only")


def assert_openai_sdk_stdout(stdout: str, *, embeddings: bool) -> None:
    values = parse_json_stdout_values(stdout)
    expected_count = 3 if embeddings else 1
    assert_true(
        len(values) == expected_count,
        f"openai-sdk.py expected {expected_count} JSON stdout payloads, got {len(values)}",
    )
    chat = values[0]
    assert_true(isinstance(chat, dict), "openai-sdk.py chat stdout payload must be a JSON object")
    assert_true(chat.get("object") == "chat.completion", "openai-sdk.py chat stdout object drifted")
    assert_true(chat.get("model") == CHAT_MODEL_ID, "openai-sdk.py chat stdout model drifted")
    choices = chat.get("choices")
    assert_true(isinstance(choices, list) and choices, "openai-sdk.py chat stdout choices missing")
    first_choice = choices[0]
    assert_true(isinstance(first_choice, dict), "openai-sdk.py chat stdout choice must be an object")
    message = first_choice.get("message")
    assert_true(isinstance(message, dict), "openai-sdk.py chat stdout message missing")
    assert_true(message.get("role") == "assistant", "openai-sdk.py chat stdout assistant role drifted")

    if not embeddings:
        return

    install = values[1]
    assert_true(isinstance(install, dict), "openai-sdk.py embedding install stdout payload must be a JSON object")
    assert_true(install.get("ok") is True, "openai-sdk.py embedding install stdout ok flag drifted")
    install_model = install.get("model")
    assert_true(isinstance(install_model, dict), "openai-sdk.py embedding install stdout model missing")
    assert_true(
        install_model.get("id") == EMBEDDING_MODEL_ID,
        "openai-sdk.py embedding install stdout model id drifted",
    )

    embeddings_payload = values[2]
    assert_true(isinstance(embeddings_payload, dict), "openai-sdk.py embeddings stdout payload must be a JSON object")
    assert_true(embeddings_payload.get("object") == "list", "openai-sdk.py embeddings stdout object drifted")
    assert_true(embeddings_payload.get("model") == EMBEDDING_MODEL_ID, "openai-sdk.py embeddings stdout model drifted")
    data = embeddings_payload.get("data")
    assert_true(isinstance(data, list) and data, "openai-sdk.py embeddings stdout data missing")
    first_embedding = data[0]
    assert_true(isinstance(first_embedding, dict), "openai-sdk.py embeddings stdout item must be an object")
    vector = first_embedding.get("embedding")
    assert_true(
        isinstance(vector, list) and vector and all(isinstance(item, float) for item in vector),
        "openai-sdk.py embeddings stdout vector must be a non-empty float list",
    )


def parse_json_stdout_values(stdout: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(stdout):
        while index < len(stdout) and stdout[index].isspace():
            index += 1
        if index >= len(stdout):
            break
        try:
            value, index = decoder.raw_decode(stdout, index)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"stdout contains non-JSON data at byte {index}: {exc}") from exc
        values.append(value)
    return values


def assert_json_request(request: RecordedRequest, label: str) -> None:
    content_type = request.headers.get("content-type", "")
    assert_true(content_type.split(";", 1)[0].lower() == "application/json", f"{label}: {request.path} missing JSON content-type")


def static_checks() -> None:
    manifest = load_public_contract()

    required_loopback_warning_terms = {
        "README.md": ("no built-in authentication", "loopback"),
        "docs/api/v1-contract.md": ("no built-in authentication", "loopback", "SECURITY.md"),
        "docs/api/backend-only-quickstart.md": ("no built-in authentication", "loopback", "access controls", "SECURITY.md"),
        "docs/api/client-examples.md": ("no built-in authentication", "loopback", "access controls"),
        "docs/public-launch-checklist.md": ("no built-in authentication", "loopback", "access controls", "SECURITY.md"),
        "examples/api/curl-quickstart.sh": ("no built-in authentication", "FATHOM_BASE_URL", "loopback", "access controls"),
        "examples/api/fathom.http": ("no built-in authentication", "@base", "loopback", "access controls"),
        "examples/api/openai-sdk.py": ("no built-in authentication", "FATHOM_BASE_URL", "loopback", "access controls"),
        "examples/api/python-no-deps.py": ("no built-in authentication", "FATHOM_BASE_URL", "loopback", "access controls"),
    }
    for relative_path, required_terms in required_loopback_warning_terms.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        missing_terms = [term for term in required_terms if term not in text]
        assert_true(
            not missing_terms,
            f"{relative_path} missing loopback/no-auth warning terms: {missing_terms}",
        )

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

    executable_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "examples/api/curl-quickstart.sh",
            "examples/api/python-no-deps.py",
            "examples/api/openai-sdk.py",
            "examples/api/fathom.http",
        )
    )
    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        assert_true(
            isinstance(method, str) and isinstance(path, str),
            "manifest supported endpoints must include method/path",
        )
        assert_true(
            path in executable_text,
            f"examples/api missing manifest-supported endpoint {method} {path}",
        )

    allowed_non_contract = manifest.get("non_contract_surfaces_allowed_in_examples", [])
    assert_true(
        isinstance(allowed_non_contract, list),
        "manifest non-contract example allow-list must be a list",
    )
    for item in allowed_non_contract:
        assert_true(
            isinstance(item, str),
            "manifest non-contract example allow-list entries must be strings",
        )
    assert_true(
        "POST /api/models/catalog/install" in allowed_non_contract,
        "manifest must explicitly allow the catalog install surface used by client examples",
    )
    assert_true(
        "/api/models/catalog/install" in executable_text,
        "examples/api missing manifest-allowed catalog install surface",
    )

    forbidden_endpoints = (
        "/v1/responses",
        "/v1/files",
        "/v1/batches",
        "/v1/fine_tuning",
        "/v1/assistants",
        "/v1/audio",
        "/v1/images",
    )
    for endpoint in forbidden_endpoints:
        assert_true(endpoint not in executable_text, f"examples must not mention non-contract endpoint {endpoint}")


def main() -> int:
    static_checks()
    run_example(["bash", "examples/api/curl-quickstart.sh"], embeddings=False)
    run_example(["python3", "examples/api/python-no-deps.py"], embeddings=False)
    run_openai_sdk_example_with_stub(embeddings=False)
    run_example(["bash", "examples/api/curl-quickstart.sh"], embeddings=True)
    run_example(["python3", "examples/api/python-no-deps.py"], embeddings=True)
    run_openai_sdk_example_with_stub(embeddings=True)
    print("API client examples regression passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"api client examples regression failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
