#!/usr/bin/env python3
"""Regression-test API client examples against a fake loopback Fathom API.

This is intentionally dependency-free and CI-safe: it does not start Fathom,
download models, or exercise runtime behavior. It verifies that public examples
send the expected narrow HTTP contract to Fathom-shaped endpoints.
"""

from __future__ import annotations

import argparse
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
MODEL_FILENAME = "model.safetensors"
DEFAULT_PROMPT = "Say hello from a local Fathom API smoke test."
DEFAULT_MAX_TOKENS = "24"
EMBEDDING_MODEL_ID = "sentence-transformers-all-minilm-l6-v2-model-safetensors"
EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_INPUT = "Rust ownership keeps memory safety explicit."
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
    contract: str | None = "quickstart",
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
        elif contract == "quickstart":
            assert_public_contract(requests, embeddings=embeddings, label=" ".join(command))
        return requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_openai_sdk_example_with_stub(
    *,
    embeddings: bool,
    extra_env: dict[str, str] | None = None,
    contract: str | None = "sdk",
) -> list[RecordedRequest]:
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
        env = {"PYTHONPATH": prepend_pythonpath(tmpdir, os.environ.get("PYTHONPATH"))}
        if extra_env:
            env.update(extra_env)
        return run_example(
            ["python3", "examples/api/openai-sdk.py"],
            embeddings=embeddings,
            extra_env=env,
            contract=contract,
            stdout_checker=(lambda stdout: assert_openai_sdk_stdout(stdout, embeddings=embeddings)) if contract == "sdk" else None,
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


def assert_quickstart_override_contract(requests: list[RecordedRequest], label: str) -> None:
    install_requests = [r for r in requests if r.method == "POST" and r.path == "/api/models/catalog/install"]
    assert_true(len(install_requests) == 2, f"{label}: override run should install chat and embedding fixtures")
    assert_true(
        install_requests[0].body == {"repo_id": "override/chat-repo", "filename": "override-chat.safetensors"},
        f"{label}: chat install overrides did not reach request body: {install_requests[0].body}",
    )
    assert_true(
        install_requests[1].body == {"repo_id": "override/embed-repo", "filename": "override-embed.safetensors"},
        f"{label}: embedding install overrides did not reach request body: {install_requests[1].body}",
    )

    chat_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/chat/completions"]
    assert_true(len(chat_requests) == 1, f"{label}: override run should send one chat request")
    chat_body = chat_requests[0].body
    assert_true(isinstance(chat_body, dict), f"{label}: chat override body must be an object")
    assert_true(chat_body.get("model") == "override-chat-model", f"{label}: FATHOM_MODEL_ID override drifted")
    assert_true(chat_body.get("max_tokens") == 7, f"{label}: FATHOM_MAX_TOKENS override drifted")
    messages = chat_body.get("messages")
    assert_true(isinstance(messages, list) and messages, f"{label}: chat override messages missing")
    assert_true(
        isinstance(messages[0], dict) and messages[0].get("content") == "override prompt",
        f"{label}: FATHOM_PROMPT override drifted",
    )

    embedding_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/embeddings"]
    assert_true(len(embedding_requests) == 1, f"{label}: override run should send one embeddings request")
    embedding_body = embedding_requests[0].body
    assert_true(isinstance(embedding_body, dict), f"{label}: embeddings override body must be an object")
    assert_true(embedding_body.get("model") == "override-embedding-model", f"{label}: FATHOM_EMBEDDING_MODEL_ID override drifted")
    assert_true(embedding_body.get("input") == "override embedding input", f"{label}: FATHOM_EMBEDDING_INPUT override drifted")
    assert_true(embedding_body.get("encoding_format") == "float", f"{label}: embeddings override must stay float")


def assert_openai_sdk_override_contract(requests: list[RecordedRequest], label: str) -> None:
    chat_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/chat/completions"]
    assert_true(len(chat_requests) == 1, f"{label}: SDK override run should send one chat request")
    chat_body = chat_requests[0].body
    assert_true(isinstance(chat_body, dict), f"{label}: SDK chat override body must be an object")
    assert_true(chat_body.get("model") == "override-chat-model", f"{label}: SDK FATHOM_MODEL_ID override drifted")
    assert_true(chat_body.get("max_tokens") == 7, f"{label}: SDK FATHOM_MAX_TOKENS override drifted")
    messages = chat_body.get("messages")
    assert_true(isinstance(messages, list) and messages, f"{label}: SDK chat override messages missing")
    assert_true(
        isinstance(messages[0], dict) and messages[0].get("content") == "override prompt",
        f"{label}: SDK FATHOM_PROMPT override drifted",
    )

    install_requests = [r for r in requests if r.method == "POST" and r.path == "/api/models/catalog/install"]
    assert_true(len(install_requests) == 1, f"{label}: SDK override run should install one embedding fixture")
    assert_true(
        install_requests[0].body == {"repo_id": "override/embed-repo", "filename": "override-embed.safetensors"},
        f"{label}: SDK embedding install overrides did not reach request body: {install_requests[0].body}",
    )

    embedding_requests = [r for r in requests if r.method == "POST" and r.path == "/v1/embeddings"]
    assert_true(len(embedding_requests) == 1, f"{label}: SDK override run should send one embeddings request")
    embedding_body = embedding_requests[0].body
    assert_true(isinstance(embedding_body, dict), f"{label}: SDK embeddings override body must be an object")
    assert_true(embedding_body.get("model") == "override-embedding-model", f"{label}: SDK FATHOM_EMBEDDING_MODEL_ID override drifted")
    assert_true(embedding_body.get("input") == "override embedding input", f"{label}: SDK FATHOM_EMBEDDING_INPUT override drifted")
    assert_true(embedding_body.get("encoding_format") == "float", f"{label}: SDK embeddings override must stay float")


def assert_api_client_environment_overrides() -> None:
    override_env = {
        "FATHOM_MODEL_ID": "override-chat-model",
        "FATHOM_REPO_ID": "override/chat-repo",
        "FATHOM_FILENAME": "override-chat.safetensors",
        "FATHOM_PROMPT": "override prompt",
        "FATHOM_MAX_TOKENS": "7",
        "FATHOM_EMBEDDING_MODEL_ID": "override-embedding-model",
        "FATHOM_EMBEDDING_REPO_ID": "override/embed-repo",
        "FATHOM_EMBEDDING_FILENAME": "override-embed.safetensors",
        "FATHOM_EMBEDDING_INPUT": "override embedding input",
    }
    for command, label in (
        (["bash", "examples/api/curl-quickstart.sh"], "curl-quickstart.sh"),
        (["python3", "examples/api/python-no-deps.py"], "python-no-deps.py"),
    ):
        requests = run_example(command, embeddings=True, extra_env=override_env, contract=None)
        assert_quickstart_override_contract(requests, label)

    sdk_requests = run_openai_sdk_example_with_stub(embeddings=True, extra_env=override_env, contract=None)
    assert_openai_sdk_override_contract(sdk_requests, "openai-sdk.py")


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


def parse_headed_json_stdout_sections(stdout: str, label: str) -> list[tuple[str, Any]]:
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in stdout.splitlines():
        heading = re.fullmatch(r"==\s+(.+?)\s+==", line.strip())
        if heading:
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = heading.group(1)
            current_lines = []
            continue
        if current_title is None:
            if line.strip():
                raise AssertionError(f"{label} stdout contains text before first heading: {line!r}")
            continue
        current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, current_lines))
    if not sections:
        raise AssertionError(f"{label} stdout did not contain headed JSON sections")

    parsed_sections: list[tuple[str, Any]] = []
    for title, lines in sections:
        payload_text = "\n".join(lines).strip()
        values = parse_json_stdout_values(payload_text)
        if len(values) != 1:
            raise AssertionError(f"{label} stdout section {title!r} expected one JSON payload, got {len(values)}")
        parsed_sections.append((title, values[0]))
    return parsed_sections


def assert_share_safe_stdout(stdout: str, label: str) -> None:
    forbidden_patterns = (
        (r"(?i)\bapi[_-]?key\b", "API key"),
        (r"(?i)\bauthorization\b", "authorization header"),
        (r"(?i)\bbearer\b", "bearer token"),
        (r"(?i)\bsecret\b", "secret"),
        (r"(?i)\btoken\b", "token"),
        (r"(?i)(^|[\s`'\"])/" + r"Users/[^ \n`'\"]+", "absolute macOS user path"),
        (r"(?i)(^|[\s`'\"])/" + r"private/[^ \n`'\"]+", "absolute private path"),
        (r"(?i)(^|[\s`'\"])[A-Z]:\\[^ \n`'\"]+", "absolute Windows path"),
        (r"(?i)\bproduction[- ]ready\b", "production-readiness overclaim"),
        (r"(?i)\bfull\s+OpenAI\s+(API\s+)?parity\b", "OpenAI parity overclaim"),
        (r"(?i)\bstreaming\s+(chat\s+)?(is\s+)?supported\b", "streaming overclaim"),
        (r"(?i)\bbase64\s+embeddings\s+(are\s+)?supported\b", "base64 embeddings overclaim"),
    )
    for pattern, description in forbidden_patterns:
        if re.search(pattern, stdout):
            raise AssertionError(f"{label} stdout must not expose {description}")


def assert_cli_example_stdout(stdout: str, *, embeddings: bool, label: str) -> None:
    assert_share_safe_stdout(stdout, label)
    sections = parse_headed_json_stdout_sections(stdout, label)
    expected_titles = [
        "Fathom health",
        "Install pinned tiny Phi SafeTensors fixture, if not already present",
        "Runnable chat models",
        "Non-streaming chat completion",
    ]
    if embeddings:
        expected_titles.extend(
            [
                "Install pinned MiniLM SafeTensors embedding fixture, if not already present",
                "Float embeddings from verified local MiniLM runtime",
            ]
        )
    actual_titles = [title for title, _value in sections]
    assert_true(actual_titles == expected_titles, f"{label} stdout headings drifted: expected {expected_titles}, got {actual_titles}")

    health = sections[0][1]
    assert_true(isinstance(health, dict), f"{label} health stdout payload must be an object")
    assert_true(health.get("ok") is True, f"{label} health stdout ok flag drifted")
    assert_true(health.get("engine") == "fathom", f"{label} health stdout engine drifted")

    install = sections[1][1]
    assert_true(isinstance(install, dict), f"{label} install stdout payload must be an object")
    assert_true(install.get("ok") is True, f"{label} install stdout ok flag drifted")
    install_model = install.get("model")
    assert_true(isinstance(install_model, dict), f"{label} install stdout model missing")
    assert_true(install_model.get("id") == CHAT_MODEL_ID, f"{label} install stdout model id drifted")

    models = sections[2][1]
    assert_true(isinstance(models, dict), f"{label} models stdout payload must be an object")
    assert_true(models.get("object") == "list", f"{label} models stdout object drifted")
    model_data = models.get("data")
    assert_true(isinstance(model_data, list) and model_data, f"{label} models stdout data missing")
    first_model = model_data[0]
    assert_true(isinstance(first_model, dict), f"{label} models stdout item must be an object")
    assert_true(first_model.get("id") == CHAT_MODEL_ID, f"{label} models stdout model id drifted")

    chat = sections[3][1]
    assert_true(isinstance(chat, dict), f"{label} chat stdout payload must be an object")
    assert_true(chat.get("object") == "chat.completion", f"{label} chat stdout object drifted")
    assert_true(chat.get("model") == CHAT_MODEL_ID, f"{label} chat stdout model drifted")
    choices = chat.get("choices")
    assert_true(isinstance(choices, list) and choices, f"{label} chat stdout choices missing")

    if not embeddings:
        return

    embedding_install = sections[4][1]
    assert_true(isinstance(embedding_install, dict), f"{label} embedding install stdout payload must be an object")
    assert_true(embedding_install.get("ok") is True, f"{label} embedding install stdout ok flag drifted")
    embedding_install_model = embedding_install.get("model")
    assert_true(isinstance(embedding_install_model, dict), f"{label} embedding install stdout model missing")
    assert_true(
        embedding_install_model.get("id") == EMBEDDING_MODEL_ID,
        f"{label} embedding install stdout model id drifted",
    )

    embeddings_payload = sections[5][1]
    assert_true(isinstance(embeddings_payload, dict), f"{label} embeddings stdout payload must be an object")
    assert_true(embeddings_payload.get("object") == "list", f"{label} embeddings stdout object drifted")
    assert_true(embeddings_payload.get("model") == EMBEDDING_MODEL_ID, f"{label} embeddings stdout model drifted")
    data = embeddings_payload.get("data")
    assert_true(isinstance(data, list) and data, f"{label} embeddings stdout data missing")
    first_embedding = data[0]
    assert_true(isinstance(first_embedding, dict), f"{label} embeddings stdout item must be an object")
    vector = first_embedding.get("embedding")
    assert_true(
        isinstance(vector, list) and vector and all(isinstance(item, float) for item in vector),
        f"{label} embeddings stdout vector must be a non-empty float list",
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


def rest_client_request_blocks(text: str) -> list[tuple[str, str, str]]:
    blocks: list[tuple[str, str, str]] = []
    current_method: str | None = None
    current_path: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(GET|POST)\s+\{\{base\}\}(/(?:v1|api)/[^\s]+)\s*$", line)
        if match:
            if current_method is not None and current_path is not None:
                blocks.append((current_method, current_path, "\n".join(current_lines)))
            current_method, current_path = match.groups()
            current_lines = []
        elif current_method is not None:
            current_lines.append(line)
    if current_method is not None and current_path is not None:
        blocks.append((current_method, current_path, "\n".join(current_lines)))
    return blocks


def rest_client_json_body(block: str, path: str) -> dict[str, Any]:
    start = block.find("{")
    assert_true(start >= 0, f"fathom.http POST {path} missing JSON body")
    decoder = json.JSONDecoder()
    try:
        value, _index = decoder.raw_decode(block[start:])
    except json.JSONDecodeError as exc:
        raise AssertionError(f"fathom.http POST {path} has invalid JSON body: {exc}") from exc
    assert_true(isinstance(value, dict), f"fathom.http POST {path} JSON body must be an object")
    return value


def assert_rest_client_json_bodies(http_blocks: list[tuple[str, str, str]]) -> None:
    post_blocks = [(path, block) for method, path, block in http_blocks if method == "POST"]
    install_bodies = [rest_client_json_body(block, path) for path, block in post_blocks if path == "/api/models/catalog/install"]
    assert_true(len(install_bodies) == 2, f"fathom.http must keep exactly two catalog install request bodies, got {len(install_bodies)}")
    expected_installs = [
        {"repo_id": CHAT_REPO_ID, "filename": "model.safetensors"},
        {"repo_id": EMBEDDING_REPO_ID, "filename": "model.safetensors"},
    ]
    assert_true(
        install_bodies == expected_installs,
        f"fathom.http catalog install bodies drifted: expected {expected_installs}, got {install_bodies}",
    )

    chat_bodies = [rest_client_json_body(block, path) for path, block in post_blocks if path == "/v1/chat/completions"]
    assert_true(len(chat_bodies) == 1, f"fathom.http must keep exactly one chat request body, got {len(chat_bodies)}")
    chat_body = chat_bodies[0]
    assert_true(chat_body.get("model") == "{{model}}", "fathom.http chat body must use the @model variable")
    messages = chat_body.get("messages")
    assert_true(isinstance(messages, list) and messages, "fathom.http chat body must keep non-empty messages")
    for message in messages:
        assert_true(isinstance(message, dict), "fathom.http chat messages must be objects")
        assert_true(isinstance(message.get("role"), str) and message["role"], "fathom.http chat messages need roles")
        assert_true(isinstance(message.get("content"), str) and message["content"], "fathom.http chat messages need content")
    assert_true(chat_body.get("stream") is not True, "fathom.http chat body must not request streaming")
    assert_true("input" not in chat_body and "encoding_format" not in chat_body, "fathom.http chat body must not include embedding fields")

    embedding_bodies = [rest_client_json_body(block, path) for path, block in post_blocks if path == "/v1/embeddings"]
    assert_true(len(embedding_bodies) == 1, f"fathom.http must keep exactly one embeddings request body, got {len(embedding_bodies)}")
    embedding_body = embedding_bodies[0]
    assert_true(embedding_body.get("model") == "{{embedding_model}}", "fathom.http embeddings body must use the @embedding_model variable")
    embedding_input = embedding_body.get("input")
    assert_true(
        isinstance(embedding_input, str)
        or (isinstance(embedding_input, list) and embedding_input and all(isinstance(item, str) for item in embedding_input)),
        "fathom.http embeddings body must keep a string or non-empty string-list input",
    )
    assert_true(embedding_body.get("encoding_format") == "float", "fathom.http embeddings body must request float encoding")
    assert_true("messages" not in embedding_body and "stream" not in embedding_body, "fathom.http embeddings body must not include chat fields")


def assert_rest_client_contract(http_text: str) -> None:
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
    expected_http_request_order = [
        ("GET", "/v1/health"),
        ("POST", "/api/models/catalog/install"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/api/models/catalog/install"),
        ("POST", "/v1/embeddings"),
    ]
    http_blocks = rest_client_request_blocks(http_text)
    actual_http_request_order = [(method, path) for method, path, _block in http_blocks]
    assert_true(
        actual_http_request_order == expected_http_request_order,
        f"fathom.http request order drifted: expected {expected_http_request_order}, got {actual_http_request_order}",
    )
    for method, path, block in http_blocks:
        if method == "POST":
            assert_true(
                "Content-Type: application/json" in block,
                f"fathom.http POST {path} missing JSON Content-Type header",
            )
    assert_rest_client_json_bodies(http_blocks)


def assert_api_client_example_defaults(texts: dict[str, str]) -> None:
    expected_by_path = {
        "docs/api/client-examples.md": (
            "export FATHOM_BASE_URL=http://127.0.0.1:8180",
            f"export FATHOM_MODEL_ID={CHAT_MODEL_ID}",
            f"export FATHOM_REPO_ID={CHAT_REPO_ID}",
            f"export FATHOM_FILENAME={MODEL_FILENAME}",
            f"export FATHOM_PROMPT='{DEFAULT_PROMPT}'",
            f"export FATHOM_MAX_TOKENS={DEFAULT_MAX_TOKENS}",
            "export FATHOM_RUN_EMBEDDINGS=1",
            f"export FATHOM_EMBEDDING_MODEL_ID={EMBEDDING_MODEL_ID}",
            f"export FATHOM_EMBEDDING_REPO_ID={EMBEDDING_REPO_ID}",
            f"export FATHOM_EMBEDDING_FILENAME={MODEL_FILENAME}",
            f"export FATHOM_EMBEDDING_INPUT='{DEFAULT_EMBEDDING_INPUT}'",
        ),
        "examples/api/curl-quickstart.sh": (
            f'MODEL_ID="${{FATHOM_MODEL_ID:-{CHAT_MODEL_ID}}}"',
            f'REPO_ID="${{FATHOM_REPO_ID:-{CHAT_REPO_ID}}}"',
            f'FILENAME="${{FATHOM_FILENAME:-{MODEL_FILENAME}}}"',
            f'PROMPT="${{FATHOM_PROMPT:-{DEFAULT_PROMPT}}}"',
            f'MAX_TOKENS="${{FATHOM_MAX_TOKENS:-{DEFAULT_MAX_TOKENS}}}"',
            'RUN_EMBEDDINGS="${FATHOM_RUN_EMBEDDINGS:-0}"',
            f'EMBEDDING_MODEL_ID="${{FATHOM_EMBEDDING_MODEL_ID:-{EMBEDDING_MODEL_ID}}}"',
            f'EMBEDDING_REPO_ID="${{FATHOM_EMBEDDING_REPO_ID:-{EMBEDDING_REPO_ID}}}"',
            f'EMBEDDING_FILENAME="${{FATHOM_EMBEDDING_FILENAME:-{MODEL_FILENAME}}}"',
            f'EMBEDDING_INPUT="${{FATHOM_EMBEDDING_INPUT:-{DEFAULT_EMBEDDING_INPUT}}}"',
        ),
        "examples/api/python-no-deps.py": (
            f'"FATHOM_MODEL_ID", "{CHAT_MODEL_ID}"',
            f'os.environ.get("FATHOM_REPO_ID", "{CHAT_REPO_ID}")',
            f'os.environ.get("FATHOM_FILENAME", "{MODEL_FILENAME}")',
            f'os.environ.get("FATHOM_PROMPT", "{DEFAULT_PROMPT}")',
            f'os.environ.get("FATHOM_MAX_TOKENS", "{DEFAULT_MAX_TOKENS}")',
            'os.environ.get("FATHOM_RUN_EMBEDDINGS") == "1"',
            f'"FATHOM_EMBEDDING_MODEL_ID", "{EMBEDDING_MODEL_ID}"',
            f'"FATHOM_EMBEDDING_REPO_ID", "{EMBEDDING_REPO_ID}"',
            f'os.environ.get("FATHOM_EMBEDDING_FILENAME", "{MODEL_FILENAME}")',
            f'"FATHOM_EMBEDDING_INPUT", "{DEFAULT_EMBEDDING_INPUT}"',
        ),
        "examples/api/openai-sdk.py": (
            f'"FATHOM_MODEL_ID", "{CHAT_MODEL_ID}"',
            f'os.environ.get("FATHOM_PROMPT", "{DEFAULT_PROMPT}")',
            f'os.environ.get("FATHOM_MAX_TOKENS", "{DEFAULT_MAX_TOKENS}")',
            'os.environ.get("FATHOM_RUN_EMBEDDINGS") == "1"',
            f'"FATHOM_EMBEDDING_MODEL_ID", "{EMBEDDING_MODEL_ID}"',
            f'"FATHOM_EMBEDDING_REPO_ID", "{EMBEDDING_REPO_ID}"',
            f'os.environ.get("FATHOM_EMBEDDING_FILENAME", "{MODEL_FILENAME}")',
            f'"FATHOM_EMBEDDING_INPUT", "{DEFAULT_EMBEDDING_INPUT}"',
        ),
    }
    for relative_path, expected_terms in expected_by_path.items():
        text = texts.get(relative_path)
        assert_true(text is not None, f"missing API client default source: {relative_path}")
        missing_terms = [term for term in expected_terms if term not in text]
        assert_true(
            not missing_terms,
            f"{relative_path} API client defaults drifted or are undocumented: {missing_terms}",
        )


def assert_dependency_light_examples(texts: dict[str, str]) -> None:
    client_examples = texts.get("docs/api/client-examples.md", "")
    assert_true(
        "dependency-light script" in client_examples,
        "client examples docs must describe curl-quickstart.sh as dependency-light",
    )
    assert_true(
        "uses only the Python standard library" in client_examples,
        "client examples docs must describe python-no-deps.py as standard-library only",
    )

    curl_text = texts.get("examples/api/curl-quickstart.sh", "")
    forbidden_curl_terms = (
        "pip install",
        "python3 -m pip",
        "npm install",
        "npm ci",
        "brew install",
        "uv pip",
        "requests",
        "from openai",
        "import openai",
    )
    for term in forbidden_curl_terms:
        assert_true(term not in curl_text, f"curl-quickstart.sh must stay dependency-light and avoid {term!r}")

    python_text = texts.get("examples/api/python-no-deps.py", "")
    forbidden_python_terms = (
        "import requests",
        "from requests",
        "import httpx",
        "from httpx",
        "from openai",
        "import openai",
        "subprocess",
        "pip install",
        "python3 -m pip",
    )
    for term in forbidden_python_terms:
        assert_true(term not in python_text, f"python-no-deps.py must stay standard-library only and avoid {term!r}")

    import_lines = [
        line.strip()
        for line in python_text.splitlines()
        if re.match(r"^(import|from)\s+", line.strip())
    ]
    allowed_import_lines = {
        "from __future__ import annotations",
        "import json",
        "import os",
        "import sys",
        "import urllib.error",
        "import urllib.request",
    }
    unexpected_imports = [line for line in import_lines if line not in allowed_import_lines]
    assert_true(
        not unexpected_imports,
        f"python-no-deps.py import surface drifted from standard-library allow-list: {unexpected_imports}",
    )


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

    expected_loopback_default = "http://127.0.0.1:8180"
    default_base_patterns = {
        "examples/api/curl-quickstart.sh": r'BASE="\$\{FATHOM_BASE_URL:-http://127\.0\.0\.1:8180\}"',
        "examples/api/python-no-deps.py": r'BASE_URL = os\.environ\.get\("FATHOM_BASE_URL", "http://127\.0\.0\.1:8180"\)\.rstrip\("/"\)',
        "examples/api/openai-sdk.py": r'ROOT_URL = os\.environ\.get\("FATHOM_BASE_URL", "http://127\.0\.0\.1:8180"\)\.rstrip\("/"\)',
        "examples/api/fathom.http": r"(?m)^@base = http://127\.0\.0\.1:8180$",
    }
    for relative_path, pattern in default_base_patterns.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert_true(
            re.search(pattern, text) is not None,
            f"{relative_path} must default to loopback {expected_loopback_default}",
        )

    assert_api_client_example_defaults(
        {
            path: (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "docs/api/client-examples.md",
                "examples/api/curl-quickstart.sh",
                "examples/api/python-no-deps.py",
                "examples/api/openai-sdk.py",
            )
        }
    )
    assert_dependency_light_examples(
        {
            path: (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "docs/api/client-examples.md",
                "examples/api/curl-quickstart.sh",
                "examples/api/python-no-deps.py",
            )
        }
    )

    http_path = ROOT / "examples/api/fathom.http"
    http_text = http_path.read_text(encoding="utf-8")
    assert_rest_client_contract(http_text)

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


def run_self_test() -> None:
    good_http = """
@base = http://127.0.0.1:8180
@model = echarlaix-tiny-random-phiforcausallm-model-safetensors
@embedding_model = sentence-transformers-all-minilm-l6-v2-model-safetensors

### Health
GET {{base}}/v1/health

### Install pinned tiny Phi SafeTensors fixture
POST {{base}}/api/models/catalog/install
Content-Type: application/json

{"repo_id": "echarlaix/tiny-random-PhiForCausalLM", "filename": "model.safetensors"}

### List runnable OpenAI-style chat models
GET {{base}}/v1/models

### Non-streaming chat completion
POST {{base}}/v1/chat/completions
Content-Type: application/json

{"model": "{{model}}", "messages": [{"role": "user", "content": "hello"}]}

### Optional: install pinned MiniLM SafeTensors embedding fixture
POST {{base}}/api/models/catalog/install
Content-Type: application/json

{"repo_id": "sentence-transformers/all-MiniLM-L6-v2", "filename": "model.safetensors"}

### Optional: OpenAI-style float embeddings from verified local MiniLM runtime
POST {{base}}/v1/embeddings
Content-Type: application/json

{"model": "{{embedding_model}}", "input": "hello", "encoding_format": "float"}
"""
    assert_rest_client_contract(good_http)

    bad_cases = (
        (
            good_http.replace("Content-Type: application/json\n\n{\"model\": \"{{model}}\"", '{"model": "{{model}}"'),
            "missing JSON Content-Type header",
        ),
        (
            good_http.replace(
                "GET {{base}}/v1/models\n\n### Non-streaming chat completion\nPOST {{base}}/v1/chat/completions",
                "POST {{base}}/v1/chat/completions\nContent-Type: application/json\n\n"
                '{"model": "{{model}}", "messages": [{"role": "user", "content": "hello"}]}\n\n'
                "### List runnable OpenAI-style chat models\nGET {{base}}/v1/models",
            ),
            "request order drifted",
        ),
        (
            good_http.replace('"messages": [{"role": "user", "content": "hello"}]', '"messages": [{"role": "user", "content": "hello"}], "stream": true'),
            "must not document streaming chat",
        ),
        (
            good_http.replace('"encoding_format": "float"', '"encoding_format": "base64"'),
            "embeddings must request float encoding",
        ),
        (
            good_http.replace('"repo_id": "echarlaix/tiny-random-PhiForCausalLM"', '"repo_id": "example/other-chat-model"'),
            "catalog install bodies drifted",
        ),
        (
            good_http.replace('"model": "{{model}}", "messages"', '"model": "{{embedding_model}}", "messages"'),
            "chat body must use the @model variable",
        ),
        (
            good_http.replace('"model": "{{embedding_model}}"', '"model": "{{model}}"'),
            "embeddings body must use the @embedding_model variable",
        ),
        (
            good_http.replace(
                '"model": "{{model}}", "messages": [{"role": "user", "content": "hello"}]',
                '"model": "{{model}}", "messages": [{"role": "user", "content": "hello"}], "encoding_format": "float"',
            ),
            "chat body must not include embedding fields",
        ),
        (
            good_http.replace('"model": "{{embedding_model}}", "input": "hello"', '"model": "{{embedding_model}}", "messages": [{"role": "user", "content": "hello"}], "input": "hello"'),
            "embeddings body must not include chat fields",
        ),
    )
    for text, expected in bad_cases:
        try:
            assert_rest_client_contract(text)
        except AssertionError as exc:
            assert_true(expected in str(exc), f"REST Client self-test failed for the wrong reason: {exc}")
        else:
            raise AssertionError(f"REST Client self-test did not reject drift: {expected}")

    default_sources = {
        "docs/api/client-examples.md": f"""
export FATHOM_BASE_URL=http://127.0.0.1:8180
export FATHOM_MODEL_ID={CHAT_MODEL_ID}
export FATHOM_REPO_ID={CHAT_REPO_ID}
export FATHOM_FILENAME={MODEL_FILENAME}
export FATHOM_PROMPT='{DEFAULT_PROMPT}'
export FATHOM_MAX_TOKENS={DEFAULT_MAX_TOKENS}
export FATHOM_RUN_EMBEDDINGS=1
export FATHOM_EMBEDDING_MODEL_ID={EMBEDDING_MODEL_ID}
export FATHOM_EMBEDDING_REPO_ID={EMBEDDING_REPO_ID}
export FATHOM_EMBEDDING_FILENAME={MODEL_FILENAME}
export FATHOM_EMBEDDING_INPUT='{DEFAULT_EMBEDDING_INPUT}'
""",
        "examples/api/curl-quickstart.sh": f"""
MODEL_ID="${{FATHOM_MODEL_ID:-{CHAT_MODEL_ID}}}"
REPO_ID="${{FATHOM_REPO_ID:-{CHAT_REPO_ID}}}"
FILENAME="${{FATHOM_FILENAME:-{MODEL_FILENAME}}}"
PROMPT="${{FATHOM_PROMPT:-{DEFAULT_PROMPT}}}"
MAX_TOKENS="${{FATHOM_MAX_TOKENS:-{DEFAULT_MAX_TOKENS}}}"
RUN_EMBEDDINGS="${{FATHOM_RUN_EMBEDDINGS:-0}}"
EMBEDDING_MODEL_ID="${{FATHOM_EMBEDDING_MODEL_ID:-{EMBEDDING_MODEL_ID}}}"
EMBEDDING_REPO_ID="${{FATHOM_EMBEDDING_REPO_ID:-{EMBEDDING_REPO_ID}}}"
EMBEDDING_FILENAME="${{FATHOM_EMBEDDING_FILENAME:-{MODEL_FILENAME}}}"
EMBEDDING_INPUT="${{FATHOM_EMBEDDING_INPUT:-{DEFAULT_EMBEDDING_INPUT}}}"
""",
        "examples/api/python-no-deps.py": f'''
MODEL_ID = os.environ.get(
    "FATHOM_MODEL_ID", "{CHAT_MODEL_ID}"
)
REPO_ID = os.environ.get("FATHOM_REPO_ID", "{CHAT_REPO_ID}")
FILENAME = os.environ.get("FATHOM_FILENAME", "{MODEL_FILENAME}")
PROMPT = os.environ.get("FATHOM_PROMPT", "{DEFAULT_PROMPT}")
MAX_TOKENS = int(os.environ.get("FATHOM_MAX_TOKENS", "{DEFAULT_MAX_TOKENS}"))
RUN_EMBEDDINGS = os.environ.get("FATHOM_RUN_EMBEDDINGS") == "1"
EMBEDDING_MODEL_ID = os.environ.get(
    "FATHOM_EMBEDDING_MODEL_ID", "{EMBEDDING_MODEL_ID}"
)
EMBEDDING_REPO_ID = os.environ.get(
    "FATHOM_EMBEDDING_REPO_ID", "{EMBEDDING_REPO_ID}"
)
EMBEDDING_FILENAME = os.environ.get("FATHOM_EMBEDDING_FILENAME", "{MODEL_FILENAME}")
EMBEDDING_INPUT = os.environ.get(
    "FATHOM_EMBEDDING_INPUT", "{DEFAULT_EMBEDDING_INPUT}"
)
''',
        "examples/api/openai-sdk.py": f'''
MODEL_ID = os.environ.get(
    "FATHOM_MODEL_ID", "{CHAT_MODEL_ID}"
)
PROMPT = os.environ.get("FATHOM_PROMPT", "{DEFAULT_PROMPT}")
MAX_TOKENS = int(os.environ.get("FATHOM_MAX_TOKENS", "{DEFAULT_MAX_TOKENS}"))
RUN_EMBEDDINGS = os.environ.get("FATHOM_RUN_EMBEDDINGS") == "1"
EMBEDDING_MODEL_ID = os.environ.get(
    "FATHOM_EMBEDDING_MODEL_ID", "{EMBEDDING_MODEL_ID}"
)
EMBEDDING_REPO_ID = os.environ.get(
    "FATHOM_EMBEDDING_REPO_ID", "{EMBEDDING_REPO_ID}"
)
EMBEDDING_FILENAME = os.environ.get("FATHOM_EMBEDDING_FILENAME", "{MODEL_FILENAME}")
EMBEDDING_INPUT = os.environ.get(
    "FATHOM_EMBEDDING_INPUT", "{DEFAULT_EMBEDDING_INPUT}"
)
''',
    }
    assert_api_client_example_defaults(default_sources)
    drifted_doc_sources = dict(default_sources)
    drifted_doc_sources["docs/api/client-examples.md"] = drifted_doc_sources[
        "docs/api/client-examples.md"
    ].replace(DEFAULT_PROMPT, "Say hello from Fathom.")
    try:
        assert_api_client_example_defaults(drifted_doc_sources)
    except AssertionError as exc:
        assert_true("API client defaults drifted" in str(exc), f"API client default self-test failed for the wrong reason: {exc}")
    else:
        raise AssertionError("API client default self-test did not reject documented prompt drift")

    dependency_light_sources = {
        "docs/api/client-examples.md": """
This dependency-light script checks health.
This version uses only the Python standard library.
""",
        "examples/api/curl-quickstart.sh": """
#!/usr/bin/env bash
curl -fsS "$BASE/v1/health"
python3 -m json.tool
""",
        "examples/api/python-no-deps.py": """
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
""",
    }
    assert_dependency_light_examples(dependency_light_sources)
    drifted_dependency_sources = dict(dependency_light_sources)
    drifted_dependency_sources["examples/api/python-no-deps.py"] += "\nimport requests\n"
    try:
        assert_dependency_light_examples(drifted_dependency_sources)
    except AssertionError as exc:
        assert_true("standard-library only" in str(exc), f"dependency-light self-test failed for the wrong reason: {exc}")
    else:
        raise AssertionError("dependency-light self-test did not reject a third-party Python import")

    bad_override_requests = [
        RecordedRequest("POST", "/api/models/catalog/install", {}, {"repo_id": "override/chat-repo", "filename": "override-chat.safetensors"}),
        RecordedRequest("POST", "/api/models/catalog/install", {}, {"repo_id": "override/embed-repo", "filename": "override-embed.safetensors"}),
        RecordedRequest(
            "POST",
            "/v1/chat/completions",
            {},
            {"model": CHAT_MODEL_ID, "messages": [{"role": "user", "content": "override prompt"}], "max_tokens": 7},
        ),
        RecordedRequest(
            "POST",
            "/v1/embeddings",
            {},
            {"model": "override-embedding-model", "input": "override embedding input", "encoding_format": "float"},
        ),
    ]
    try:
        assert_quickstart_override_contract(bad_override_requests, "synthetic override drift")
    except AssertionError as exc:
        assert_true("FATHOM_MODEL_ID override drifted" in str(exc), f"override self-test failed for the wrong reason: {exc}")
    else:
        raise AssertionError("override self-test did not reject a lost FATHOM_MODEL_ID override")

    safe_cli_stdout = """
== Fathom health ==
{"ok": true, "engine": "fathom", "generation_ready": true}

== Install pinned tiny Phi SafeTensors fixture, if not already present ==
{"ok": true, "model": {"id": "echarlaix-tiny-random-phiforcausallm-model-safetensors"}}

== Runnable chat models ==
{"object": "list", "data": [{"id": "echarlaix-tiny-random-phiforcausallm-model-safetensors"}]}

== Non-streaming chat completion ==
{"object": "chat.completion", "model": "echarlaix-tiny-random-phiforcausallm-model-safetensors", "choices": [{"message": {"role": "assistant", "content": "hello"}}]}
"""
    assert_cli_example_stdout(safe_cli_stdout, embeddings=False, label="synthetic CLI stdout")
    unsafe_cli_stdout = safe_cli_stdout.replace(
        '"content": "hello"',
        '"content": "hello", "authorization": "Bearer local-secret"',
    )
    try:
        assert_cli_example_stdout(unsafe_cli_stdout, embeddings=False, label="synthetic unsafe CLI stdout")
    except AssertionError as exc:
        assert_true("authorization header" in str(exc), f"CLI stdout self-test failed for the wrong reason: {exc}")
    else:
        raise AssertionError("CLI stdout self-test did not reject unsafe stdout")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression-test public API client examples")
    parser.add_argument("--self-test", action="store_true", help="run synthetic static regression checks")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        print("API client examples regression self-test passed")
        return 0

    static_checks()
    run_example(
        ["bash", "examples/api/curl-quickstart.sh"],
        embeddings=False,
        stdout_checker=lambda stdout: assert_cli_example_stdout(stdout, embeddings=False, label="curl-quickstart.sh"),
    )
    run_example(
        ["python3", "examples/api/python-no-deps.py"],
        embeddings=False,
        stdout_checker=lambda stdout: assert_cli_example_stdout(stdout, embeddings=False, label="python-no-deps.py"),
    )
    run_openai_sdk_example_with_stub(embeddings=False)
    run_example(
        ["bash", "examples/api/curl-quickstart.sh"],
        embeddings=True,
        stdout_checker=lambda stdout: assert_cli_example_stdout(stdout, embeddings=True, label="curl-quickstart.sh"),
    )
    run_example(
        ["python3", "examples/api/python-no-deps.py"],
        embeddings=True,
        stdout_checker=lambda stdout: assert_cli_example_stdout(stdout, embeddings=True, label="python-no-deps.py"),
    )
    run_openai_sdk_example_with_stub(embeddings=True)
    assert_api_client_environment_overrides()
    print("API client examples regression passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"api client examples regression failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
