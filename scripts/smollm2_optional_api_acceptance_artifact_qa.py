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

MODEL_ID = "huggingfacetb-smollm2-135m-instruct-model-safetensors"
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
EXPECTED_CHECK_ARTIFACTS = REQUIRED_ARTIFACTS - {"summary.json", "summary.md"}
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
REQUIRED_CAVEAT_PHRASES = (
    "Optional local",
    "generation quality",
    "latency",
    "throughput",
    "production readiness",
    "legal suitability",
    "broad SmolLM2/Llama-style compatibility",
    "arbitrary Hugging Face",
    "streaming",
    "external proxying",
    "full OpenAI API parity",
    "GGUF tokenizer execution",
    "GGUF " + "runtime",
    "weight loading",
    "generation",
    "dequantization",
    "inference",
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


def assert_required_caveats(text: str, label: str) -> None:
    lowered = text.lower()
    missing = [phrase for phrase in REQUIRED_CAVEAT_PHRASES if phrase.lower() not in lowered]
    if missing:
        raise AssertionError(f"{label} missing caveat phrase(s): {missing}")


def assert_checks_cover_required_artifacts(checks: Any) -> None:
    if not isinstance(checks, list):
        raise AssertionError("summary.checks must be a list")
    artifacts = [check.get("artifact") for check in checks if isinstance(check, dict)]
    missing = sorted(EXPECTED_CHECK_ARTIFACTS - set(artifacts))
    unexpected = sorted(set(artifacts) - EXPECTED_CHECK_ARTIFACTS)
    duplicates = sorted({artifact for artifact in artifacts if artifact and artifacts.count(artifact) > 1})
    if missing or unexpected or duplicates:
        raise AssertionError(
            "summary.checks artifact index mismatch: "
            f"missing={missing} unexpected={unexpected} duplicates={duplicates}"
        )
    for check in checks:
        if not isinstance(check, dict):
            raise AssertionError("summary.checks entries must be objects")
        if check.get("status") != "passed":
            raise AssertionError(f"summary check failed: {check}")


def assert_markdown_checks_match_summary(checks: Any, markdown: str) -> None:
    if not isinstance(checks, list):
        return
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = check.get("name")
        status = check.get("status")
        artifact = check.get("artifact")
        if all(isinstance(value, str) and value for value in (name, status, artifact)):
            row = f"- `{name}`: `{status}` ({artifact})"
            if row not in markdown:
                raise AssertionError(f"summary.md missing check row matching summary.json: {name}")


def assert_markdown_timestamps_match_summary(summary: dict[str, Any], markdown: str) -> None:
    for key, label in (("started_at", "Started"), ("finished_at", "Finished")):
        value = summary.get(key)
        if isinstance(value, str) and value and f"- {label}: `{value}`" not in markdown:
            raise AssertionError(f"summary.md missing {key} row matching summary.json")


def assert_markdown_identity_matches_summary(summary: dict[str, Any], markdown: str) -> None:
    commit = summary.get("repo_commit")
    if not isinstance(commit, str) or not commit:
        raise AssertionError("summary.repo_commit must be non-empty text")
    if f"- Repo commit: `{commit}`" not in markdown:
        raise AssertionError("summary.md missing repo_commit row matching summary.json")
    model_id = summary.get("model_id")
    repo_id = summary.get("repo_id")
    revision = summary.get("revision")
    if not all(isinstance(value, str) and value for value in (model_id, repo_id, revision)):
        raise AssertionError("summary model identity fields must be non-empty text")
    if f"- Model: `{model_id}`" not in markdown:
        raise AssertionError("summary.md missing model_id row matching summary.json")
    if f"- Upstream: `{repo_id}` at `{revision}`" not in markdown:
        raise AssertionError("summary.md missing repo_id/revision row matching summary.json")


def assert_markdown_path_labels_match_summary(summary: dict[str, Any], markdown: str) -> None:
    for key, label in (
        ("artifact_dir", "Artifact directory"),
        ("state_dir", "State directory"),
        ("model_dir", "Model directory"),
    ):
        value = summary.get(key)
        if not isinstance(value, str) or not value:
            raise AssertionError(f"summary.{key} must be non-empty text")
        if f"- {label}: `{value}`" not in markdown:
            raise AssertionError(f"summary.md missing {key} label matching summary.json")

    log_dir = summary.get("log_dir")
    if not isinstance(log_dir, str) or not log_dir:
        raise AssertionError("summary.log_dir must be non-empty text")
    server_log = f"{log_dir.rstrip('/')}/server.log"
    if f"- Server log: `{server_log}`" not in markdown:
        raise AssertionError("summary.md missing log_dir server-log label matching summary.json")


def assert_loopback_base_url(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{2,5}", value):
        raise AssertionError("summary.base_url must be an http://127.0.0.1:<port> loopback URL")
    port = int(value.rsplit(":", 1)[1])
    if not 1 <= port <= 65535:
        raise AssertionError("summary.base_url port must be in range 1-65535")


def assert_utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise AssertionError(f"{label} must be an RFC3339 UTC timestamp ending in Z")


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
    assert_loopback_base_url(summary.get("base_url"))
    assert_utc_timestamp(summary.get("started_at"), "summary.started_at")
    assert_utc_timestamp(summary.get("finished_at"), "summary.finished_at")
    for key in ("artifact_dir", "model_dir", "state_dir", "log_dir"):
        value = summary.get(key)
        if not isinstance(value, str) or value.startswith("/"):
            raise AssertionError(f"summary.{key} must be share-safe relative text")
    if summary.get("model_id") != MODEL_ID or summary.get("repo_id") != REPO_ID or summary.get("revision") != REVISION:
        raise AssertionError("summary model identity mismatch")
    caveats = "\n".join(str(item) for item in summary.get("caveats") or [])
    assert_required_caveats(caveats, "summary.json caveats")
    md = (directory / "summary.md").read_text(encoding="utf-8")
    if "Result: `passed`" not in md or "What this does not prove" not in md:
        raise AssertionError("summary.md must clearly mark pass and caveats")
    assert_required_caveats(md, "summary.md")
    assert_markdown_identity_matches_summary(summary, md)
    assert_markdown_path_labels_match_summary(summary, md)
    assert_markdown_timestamps_match_summary(summary, md)
    assert_markdown_checks_match_summary(summary.get("checks"), md)

    assert_checks_cover_required_artifacts(summary.get("checks"))

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
            "Does not prove generation quality, latency, throughput, production readiness, legal suitability, broad SmolLM2/Llama-style compatibility, arbitrary Hugging Face execution, streaming, external proxying, or full OpenAI API parity.",
            "Does not claim GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference.",
        ],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "summary.md").write_text(
        "# SmolLM2 optional API acceptance artifacts\n\n"
        "- Result: `passed`\n"
        "- Repo commit: `sample`\n"
        "- Started: `2026-04-29T00:00:00Z`\n"
        "- Finished: `2026-04-29T00:00:01Z`\n"
        "- Model: `huggingfacetb-smollm2-135m-instruct-model-safetensors`\n"
        "- Upstream: `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac`\n"
        "- Scope: Optional local larger-demo API evidence only; not default CI.\n"
        "- Artifact directory: `.`\n"
        "- State directory: `state/`\n"
        "- Model directory: `models/`\n"
        "- Server log: `logs/server.log`\n\n"
        "## Checks\n\n"
        + "\n".join(f"- `{check['name']}`: `{check['status']}` ({check['artifact']})" for check in checks)
        + "\n\n"
        "## What this does not prove\n\n"
        "No generation quality, latency, throughput, production readiness, legal suitability, broad SmolLM2/Llama-style compatibility, arbitrary Hugging Face execution, streaming, external proxying, or full OpenAI API parity claim.\n"
        "No public/runtime GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference claim.\n",
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
            bad = Path(tmp) / "missing-caveat"
            write_sample(bad)
            summary = load_json(bad / "summary.json")
            summary["caveats"] = [str(item).replace("latency, ", "") for item in summary["caveats"]]
            (bad / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                validate_summary(bad)
            except AssertionError as exc:
                if "latency" not in str(exc):
                    raise
            else:
                raise AssertionError("missing caveat self-check did not fail")
            bad_index = Path(tmp) / "missing-check-index"
            write_sample(bad_index)
            summary = load_json(bad_index / "summary.json")
            summary["checks"] = [
                check for check in summary["checks"] if check.get("artifact") != "06-chat-stream-refusal.json"
            ]
            (bad_index / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                validate_summary(bad_index)
            except AssertionError as exc:
                if "summary.checks artifact index mismatch" not in str(exc):
                    raise
            else:
                raise AssertionError("missing check artifact self-check did not fail")
            bad_md_index = Path(tmp) / "missing-markdown-check-index"
            write_sample(bad_md_index)
            md = (bad_md_index / "summary.md").read_text(encoding="utf-8")
            (bad_md_index / "summary.md").write_text(
                md.replace("- `stream_refusal`: `passed` (06-chat-stream-refusal.json)\n", ""),
                encoding="utf-8",
            )
            try:
                validate_summary(bad_md_index)
            except AssertionError as exc:
                if "summary.md missing check row matching summary.json" not in str(exc):
                    raise
            else:
                raise AssertionError("missing summary.md check row self-check did not fail")
            bad_base = Path(tmp) / "bad-base-url"
            write_sample(bad_base)
            summary = load_json(bad_base / "summary.json")
            summary["base_url"] = "https://example.invalid"
            (bad_base / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                validate_summary(bad_base)
            except AssertionError as exc:
                if "summary.base_url" not in str(exc):
                    raise
            else:
                raise AssertionError("external summary.base_url self-check did not fail")
            bad_timestamp = Path(tmp) / "bad-timestamp"
            write_sample(bad_timestamp)
            summary = load_json(bad_timestamp / "summary.json")
            summary["finished_at"] = "2026-04-29 00:00:01"
            (bad_timestamp / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                validate_summary(bad_timestamp)
            except AssertionError as exc:
                if "summary.finished_at" not in str(exc):
                    raise
            else:
                raise AssertionError("bad summary timestamp self-check did not fail")
            bad_md_timestamp = Path(tmp) / "bad-markdown-timestamp"
            write_sample(bad_md_timestamp)
            (bad_md_timestamp / "summary.md").write_text(
                (bad_md_timestamp / "summary.md")
                .read_text(encoding="utf-8")
                .replace("- Finished: `2026-04-29T00:00:01Z`\n", "- Finished: `2026-04-29T00:00:02Z`\n"),
                encoding="utf-8",
            )
            try:
                validate_summary(bad_md_timestamp)
            except AssertionError as exc:
                if "summary.md missing finished_at row" not in str(exc):
                    raise
            else:
                raise AssertionError("bad summary.md timestamp self-check did not fail")
            bad_md_commit = Path(tmp) / "bad-markdown-commit"
            write_sample(bad_md_commit)
            (bad_md_commit / "summary.md").write_text(
                (bad_md_commit / "summary.md")
                .read_text(encoding="utf-8")
                .replace("- Repo commit: `sample`\n", "- Repo commit: `stale`\n"),
                encoding="utf-8",
            )
            try:
                validate_summary(bad_md_commit)
            except AssertionError as exc:
                if "summary.md missing repo_commit row" not in str(exc):
                    raise
            else:
                raise AssertionError("bad summary.md repo_commit self-check did not fail")
            bad_md_model = Path(tmp) / "bad-markdown-model"
            write_sample(bad_md_model)
            (bad_md_model / "summary.md").write_text(
                (bad_md_model / "summary.md")
                .read_text(encoding="utf-8")
                .replace(
                    "- Model: `huggingfacetb-smollm2-135m-instruct-model-safetensors`\n",
                    "- Model: `stale-model`\n",
                ),
                encoding="utf-8",
            )
            try:
                validate_summary(bad_md_model)
            except AssertionError as exc:
                if "summary.md missing model_id row" not in str(exc):
                    raise
            else:
                raise AssertionError("bad summary.md model identity self-check did not fail")
            bad_md_upstream = Path(tmp) / "bad-markdown-upstream"
            write_sample(bad_md_upstream)
            (bad_md_upstream / "summary.md").write_text(
                (bad_md_upstream / "summary.md")
                .read_text(encoding="utf-8")
                .replace(
                    "- Upstream: `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac`\n",
                    "- Upstream: `stale/repo` at `stale-revision`\n",
                ),
                encoding="utf-8",
            )
            try:
                validate_summary(bad_md_upstream)
            except AssertionError as exc:
                if "summary.md missing repo_id/revision row" not in str(exc):
                    raise
            else:
                raise AssertionError("bad summary.md upstream identity self-check did not fail")
            bad_md_path_label = Path(tmp) / "bad-markdown-path-label"
            write_sample(bad_md_path_label)
            (bad_md_path_label / "summary.md").write_text(
                (bad_md_path_label / "summary.md")
                .read_text(encoding="utf-8")
                .replace("- Server log: `logs/server.log`\n", "- Server log: `stale-logs/server.log`\n"),
                encoding="utf-8",
            )
            try:
                validate_summary(bad_md_path_label)
            except AssertionError as exc:
                if "summary.md missing log_dir server-log label" not in str(exc):
                    raise
            else:
                raise AssertionError("bad summary.md path-label self-check did not fail")
        print("SmolLM2 optional API acceptance artifact QA self-test passed")
        return
    for directory in dirs:
        validate_summary(directory)
    print("SmolLM2 optional API acceptance artifact QA passed")


if __name__ == "__main__":
    main()
