#!/usr/bin/env python3
"""Validate optional Qwen2.5 API acceptance artifacts.

With no artifact directory arguments this runs a dependency-free synthetic self-test.
With one or more directories it validates artifacts produced by
scripts/qwen25_optional_api_acceptance_smoke.sh.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "qwen-qwen2-5-0-5b-instruct-model-safetensors"
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_TOTAL_BYTES = 999_597_690
EXPECTED_FILES = {
    "config.json": (659, "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"),
    "generation_config.json": (242, "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6"),
    "merges.txt": (1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "model.safetensors": (988_097_824, "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
    "tokenizer.json": (7_031_645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": (7_305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "LICENSE": (11_343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
}
REQUIRED_ARTIFACTS = {
    "01-v1-health.json",
    "02-install-qwen25.json",
    "03-v1-models-after-qwen25.json",
    "04-chat-cold.json",
    "05-chat-warm.json",
    "06-chat-stream-refusal.json",
    "summary.json",
    "summary.md",
}
LOCAL_PATH_PATTERNS = [
    re.compile("/" + "Users" + "/", re.IGNORECASE),
    re.compile("/" + "private" + "/" + "tmp", re.IGNORECASE),
    re.compile("/" + "opt" + "/" + "homebrew", re.IGNORECASE),
]
OVERCLAIMS = re.compile(
    r"(quality\s+proved|production\s+ready|legal\s+(approved|review)|full\s+OpenAI\s+parity|"
    r"arbitrary\s+(Hugging\s+Face|SafeTensors)|GGUF\s+(runtime|generation|inference|tokenizer\s+execution)|"
    r"external\s+provider\s+(proxy|call)\s+(works|succeeded|enabled))",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssertionError(f"missing required artifact: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise AssertionError(f"{path.name} must contain a JSON object")
    return data


def assert_share_safe(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"{path.name} contains local path-like leak matching {pattern.pattern}")
    for line in text.splitlines():
        if OVERCLAIMS.search(line) and not re.search(r"\b(no|not|does not|without|doesn't)\b", line, re.IGNORECASE):
            raise AssertionError(f"{path.name} contains an overclaim")


def error_code(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    return error.get("code") if isinstance(error, dict) else None


def validate_install(install: dict[str, Any]) -> None:
    if install.get("id") != MODEL_ID:
        raise AssertionError("install artifact has wrong model id")
    if install.get("status") != "ready" or install.get("capability_status") != "runnable":
        raise AssertionError("install artifact must be ready/runnable")
    if "safetensors-hf" not in (install.get("backend_lanes") or []):
        raise AssertionError("install artifact missing safetensors-hf backend lane")
    manifest = install.get("download_manifest") or {}
    expected_manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "apache-2.0",
        "license_status": "permissive",
        "verification_status": "verified",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise AssertionError(f"download_manifest.{key} expected {expected!r}, got {manifest.get(key)!r}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise AssertionError("download_manifest.files must be a list")
    by_name = {file.get("filename"): file for file in files if isinstance(file, dict)}
    if set(by_name) != set(EXPECTED_FILES):
        raise AssertionError(f"manifest file set mismatch: {sorted(by_name)}")
    total = 0
    for filename, (size, sha256) in EXPECTED_FILES.items():
        item = by_name[filename]
        if item.get("size_bytes") != size or item.get("sha256") != sha256:
            raise AssertionError(f"manifest mismatch for {filename}")
        total += size
    if total != EXPECTED_TOTAL_BYTES:
        raise AssertionError("internal expected byte total mismatch")


def validate_models(models: dict[str, Any]) -> None:
    items = models.get("data")
    if not isinstance(items, list):
        raise AssertionError("/v1/models artifact must contain data list")
    matches = [item for item in items if item.get("id") == MODEL_ID]
    if len(matches) != 1:
        raise AssertionError("/v1/models must contain exactly one Qwen2.5 model entry")
    fathom = matches[0].get("fathom") or {}
    if fathom.get("capability_status") != "runnable":
        raise AssertionError("Qwen2.5 /v1 model entry must be runnable")
    if fathom.get("provider_kind") == "external":
        raise AssertionError("Qwen2.5 /v1 model entry must not be external")
    if "safetensors-hf" not in (fathom.get("backend_lanes") or []):
        raise AssertionError("Qwen2.5 /v1 model entry missing safetensors-hf backend lane")


def validate_chat(chat: dict[str, Any], label: str) -> str:
    if chat.get("object") != "chat.completion":
        raise AssertionError(f"{label} chat must be a chat.completion")
    message = (chat.get("choices") or [{}])[0].get("message") or {}
    if message.get("role") != "assistant" or not str(message.get("content") or "").strip():
        raise AssertionError(f"{label} chat missing real assistant content")
    metrics = (chat.get("fathom") or {}).get("metrics") or {}
    if metrics.get("runtime_family") != "qwen2":
        raise AssertionError(f"{label} chat metrics runtime_family must be qwen2")
    residency = metrics.get("runtime_residency")
    if residency not in {"cold_loaded", "warm_reused"}:
        raise AssertionError(f"{label} chat has invalid runtime_residency {residency!r}")
    return residency


def validate_summary(directory: Path) -> None:
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (directory / name).exists())
    if missing:
        raise AssertionError(f"missing required artifacts: {missing}")
    for name in ("summary.json", "summary.md"):
        assert_share_safe(directory / name)

    summary = load_json(directory / "summary.json")
    if summary.get("schema") != "fathom.qwen25_optional_api_acceptance.summary.v1":
        raise AssertionError("summary schema mismatch")
    if summary.get("passed") is not True:
        raise AssertionError("summary.passed must be true")
    for key in ("artifact_dir", "model_dir", "state_dir", "log_dir"):
        value = summary.get(key)
        if not isinstance(value, str) or value.startswith("/"):
            raise AssertionError(f"summary.{key} must be share-safe relative text")
    if summary.get("model_id") != MODEL_ID or summary.get("repo_id") != REPO_ID or summary.get("revision") != REVISION:
        raise AssertionError("summary model identity mismatch")
    caveats = "\n".join(str(item) for item in summary.get("caveats") or [])
    required_caveats = ["Optional local", "Does not prove generation quality", "arbitrary Hugging Face", "GGUF"]
    for phrase in required_caveats:
        if phrase not in caveats:
            raise AssertionError(f"summary caveats missing {phrase!r}")
    md = (directory / "summary.md").read_text(encoding="utf-8")
    if "Result: `passed`" not in md or "What this does not prove" not in md:
        raise AssertionError("summary.md must clearly mark pass and caveats")

    checks = summary.get("checks")
    if not isinstance(checks, list) or len(checks) < 6:
        raise AssertionError("summary.checks must include all smoke checks")
    for check in checks:
        if check.get("status") != "passed":
            raise AssertionError(f"summary check failed: {check}")

    validate_install(load_json(directory / "02-install-qwen25.json"))
    validate_models(load_json(directory / "03-v1-models-after-qwen25.json"))
    cold = validate_chat(load_json(directory / "04-chat-cold.json"), "cold")
    warm = validate_chat(load_json(directory / "05-chat-warm.json"), "warm")
    if "cold_loaded" not in {cold, warm} or "warm_reused" not in {cold, warm}:
        raise AssertionError("chat artifacts must include both cold_loaded and warm_reused residency evidence")
    stream = load_json(directory / "06-chat-stream-refusal.json")
    if error_code(stream) != "not_implemented" or "choices" in stream:
        raise AssertionError("stream refusal must be not_implemented with no fake choices")


def qwen_install_payload() -> dict[str, Any]:
    return {
        "id": MODEL_ID,
        "status": "ready",
        "capability_status": "runnable",
        "backend_lanes": ["safetensors-hf"],
        "download_manifest": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "license": "apache-2.0",
            "license_status": "permissive",
            "verification_status": "verified",
            "files": [
                {"filename": name, "size_bytes": size, "sha256": sha256}
                for name, (size, sha256) in EXPECTED_FILES.items()
            ],
        },
    }


def qwen_chat_payload(residency: str) -> dict[str, Any]:
    return {
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Local inference is running."}}],
        "fathom": {"metrics": {"runtime_family": "qwen2", "runtime_residency": residency}},
    }


def write_sample(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "01-v1-health.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
    (directory / "02-install-qwen25.json").write_text(json.dumps(qwen_install_payload(), indent=2) + "\n", encoding="utf-8")
    (directory / "03-v1-models-after-qwen25.json").write_text(
        json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "fathom": {
                            "capability_status": "runnable",
                            "provider_kind": "local",
                            "backend_lanes": ["safetensors-hf"],
                        },
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "04-chat-cold.json").write_text(json.dumps(qwen_chat_payload("cold_loaded"), indent=2) + "\n", encoding="utf-8")
    (directory / "05-chat-warm.json").write_text(json.dumps(qwen_chat_payload("warm_reused"), indent=2) + "\n", encoding="utf-8")
    (directory / "06-chat-stream-refusal.json").write_text(
        json.dumps({"error": {"message": "streaming is not implemented", "type": "not_implemented", "code": "not_implemented", "param": None}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    checks = [
        {"name": "health", "artifact": "01-v1-health.json", "description": "health", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "install_qwen25", "artifact": "02-install-qwen25.json", "description": "install", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "models_include_qwen25", "artifact": "03-v1-models-after-qwen25.json", "description": "models", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "chat_cold_qwen2", "artifact": "04-chat-cold.json", "description": "cold chat", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "chat_warm_qwen2", "artifact": "05-chat-warm.json", "description": "warm chat", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "stream_refusal", "artifact": "06-chat-stream-refusal.json", "description": "stream refused", "expected_http_status": 501, "http_status": 501, "status": "passed"},
    ]
    summary = {
        "schema": "fathom.qwen25_optional_api_acceptance.summary.v1",
        "passed": True,
        "repo_commit": "sample",
        "started_at": "2026-04-29T00:00:00Z",
        "finished_at": "2026-04-29T00:00:01Z",
        "base_url": "http://127.0.0.1:18185",
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
            "Does not prove generation quality, arbitrary Hugging Face execution, streaming, external proxying, or full OpenAI API parity.",
            "Does not claim GGUF tokenizer execution, GGUF runtime, generation, or inference.",
        ],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "summary.md").write_text(
        "# Qwen2.5 optional API acceptance artifacts\n\n"
        "- Result: `passed`\n"
        "- Artifact directory: `.`\n"
        "- State directory: `state/`\n"
        "- Model directory: `models/`\n"
        "- Server log: `logs/server.log`\n\n"
        "## What this does not prove\n\n"
        "No generation quality, production readiness, arbitrary Hugging Face execution, streaming, external proxying, full OpenAI API parity, or GGUF runtime claim.\n",
        encoding="utf-8",
    )


def main() -> None:
    import sys

    dirs = [Path(arg) for arg in sys.argv[1:]]
    if not dirs:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample"
            write_sample(sample)
            validate_summary(sample)
        print("Qwen2.5 optional API acceptance artifact QA self-test passed")
        return
    for directory in dirs:
        validate_summary(directory)
    print("Qwen2.5 optional API acceptance artifact QA passed")


if __name__ == "__main__":
    main()
