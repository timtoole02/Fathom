#!/usr/bin/env python3
"""Validate optional SmolLM2 API acceptance artifacts.

With no artifact directory arguments this runs a dependency-free synthetic self-test.
With one or more directories it validates artifacts produced by
scripts/smollm2_optional_api_acceptance_smoke.sh.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "hf-huggingfacetb-smollm2-135m-instruct"
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
EXPECTED_TOTAL_BYTES = 271_170_520
EXPECTED_FILES = {
    "config.json": (861, "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"),
    "generation_config.json": (132, "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013"),
    "model.safetensors": (269_060_552, "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"),
    "tokenizer.json": (2_104_556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
    "tokenizer_config.json": (3_764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
    "special_tokens_map.json": (655, "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52"),
}
REQUIRED_ARTIFACTS = {
    "01-v1-health.json",
    "02-install-smollm2.json",
    "03-v1-models-after-smollm2.json",
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
        raise AssertionError("/v1/models must contain exactly one SmolLM2 model entry")
    fathom = matches[0].get("fathom") or {}
    if fathom.get("capability_status") != "runnable":
        raise AssertionError("SmolLM2 /v1 model entry must be runnable")
    if fathom.get("provider_kind") == "external":
        raise AssertionError("SmolLM2 /v1 model entry must not be external")
    if "safetensors-hf" not in (fathom.get("backend_lanes") or []):
        raise AssertionError("SmolLM2 /v1 model entry missing safetensors-hf backend lane")


def validate_chat(chat: dict[str, Any], label: str) -> str:
    if chat.get("object") != "chat.completion":
        raise AssertionError(f"{label} chat must be a chat.completion")
    message = (chat.get("choices") or [{}])[0].get("message") or {}
    if message.get("role") != "assistant" or not str(message.get("content") or "").strip():
        raise AssertionError(f"{label} chat missing real assistant content")
    metrics = (chat.get("fathom") or {}).get("metrics") or {}
    if metrics.get("runtime_family") != "llama":
        raise AssertionError(f"{label} chat metrics runtime_family must be llama")
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
    if summary.get("schema") != "fathom.smollm2_optional_api_acceptance.summary.v1":
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

    validate_install(load_json(directory / "02-install-smollm2.json"))
    validate_models(load_json(directory / "03-v1-models-after-smollm2.json"))
    cold = validate_chat(load_json(directory / "04-chat-cold.json"), "cold")
    warm = validate_chat(load_json(directory / "05-chat-warm.json"), "warm")
    if "cold_loaded" not in {cold, warm} or "warm_reused" not in {cold, warm}:
        raise AssertionError("chat artifacts must include both cold_loaded and warm_reused residency evidence")
    stream = load_json(directory / "06-chat-stream-refusal.json")
    if error_code(stream) != "not_implemented" or "choices" in stream:
        raise AssertionError("stream refusal must be not_implemented with no fake choices")


def smollm2_install_payload() -> dict[str, Any]:
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


def smollm2_chat_payload(residency: str) -> dict[str, Any]:
    return {
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Local inference is running."}}],
        "fathom": {"metrics": {"runtime_family": "llama", "runtime_residency": residency}},
    }


def write_sample(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "01-v1-health.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
    (directory / "02-install-smollm2.json").write_text(json.dumps(smollm2_install_payload(), indent=2) + "\n", encoding="utf-8")
    (directory / "03-v1-models-after-smollm2.json").write_text(
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
    (directory / "04-chat-cold.json").write_text(json.dumps(smollm2_chat_payload("cold_loaded"), indent=2) + "\n", encoding="utf-8")
    (directory / "05-chat-warm.json").write_text(json.dumps(smollm2_chat_payload("warm_reused"), indent=2) + "\n", encoding="utf-8")
    (directory / "06-chat-stream-refusal.json").write_text(
        json.dumps({"error": {"message": "streaming is not implemented", "type": "not_implemented", "code": "not_implemented", "param": None}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    checks = [
        {"name": "health", "artifact": "01-v1-health.json", "description": "health", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "install_smollm2", "artifact": "02-install-smollm2.json", "description": "install", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "models_include_smollm2", "artifact": "03-v1-models-after-smollm2.json", "description": "models", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "chat_cold_llama", "artifact": "04-chat-cold.json", "description": "cold chat", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "chat_warm_llama", "artifact": "05-chat-warm.json", "description": "warm chat", "expected_http_status": 200, "http_status": 200, "status": "passed"},
        {"name": "stream_refusal", "artifact": "06-chat-stream-refusal.json", "description": "stream refused", "expected_http_status": 501, "http_status": 501, "status": "passed"},
    ]
    summary = {
        "schema": "fathom.smollm2_optional_api_acceptance.summary.v1",
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
        "# SmolLM2 optional API acceptance artifacts\n\n"
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
        print("SmolLM2 optional API acceptance artifact QA self-test passed")
        return
    for directory in dirs:
        validate_summary(directory)
    print("SmolLM2 optional API acceptance artifact QA passed")


if __name__ == "__main__":
    main()
