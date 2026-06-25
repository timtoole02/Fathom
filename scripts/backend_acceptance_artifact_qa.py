#!/usr/bin/env python3
"""Validate backend acceptance smoke artifact summaries.

This is intentionally offline and dependency-free. With no arguments it validates
small synthetic passed/failed summaries so CI can keep artifact wording/schema
honest without starting Fathom or downloading model fixtures. With one or more
artifact directories it validates those generated smoke outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

# Build sensitive local-path probes without embedding the literal private paths
# as contiguous source text; the public risk scan should catch those leaks in
# artifacts/docs, not self-leak through this QA helper.
LOCAL_PATH_PATTERNS = [
    re.compile("/" + "Users" + "/", re.IGNORECASE),
    re.compile("/" + "private" + "/" + "tmp", re.IGNORECASE),
    re.compile("/" + "opt" + "/" + "homebrew", re.IGNORECASE),
]
LEGAL_OVERCLAIM = re.compile(
    r"license\s+(safe|approved|compliant)|legal review completed|legally approved",
    re.IGNORECASE,
)
EXTERNAL_PROXY_OVERCLAIM = re.compile(
    r"(external\s+model\s+(replied|answered|generated)|called\s+(the\s+)?provider|"
    r"provider\s+call\s+(succeeded|completed)|proxied\s+(a\s+)?chat|external\s+proxy\s+support\s+(works|is\s+enabled))",
    re.IGNORECASE,
)
EXTERNAL_PLACEHOLDER_CHECKS = {
    "external_placeholder_connected_metadata_only",
    "external_placeholder_excluded_from_v1_models",
    "external_placeholder_activation_refusal",
    "external_placeholder_v1_chat_refusal",
}
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(r"\b(api[_-]?key|authorization|bearer|token|secret)\b\s*[:=]", re.IGNORECASE),
]
REQUEST_PAYLOAD_PATTERNS = [
    re.compile(r"\b(messages|input|prompt|request_body|provider_payload)\b\s*[:=]", re.IGNORECASE),
    re.compile(r"placeholder-key", re.IGNORECASE),
    re.compile(r"api\.example\.test", re.IGNORECASE),
]


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


def assert_public_summary_share_safe(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"{path.name} contains a local path-like leak matching {pattern.pattern}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"{path.name} contains a secret-like marker matching {pattern.pattern}")
    for pattern in REQUEST_PAYLOAD_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"{path.name} contains request/payload text matching {pattern.pattern}")
    if LEGAL_OVERCLAIM.search(text):
        raise AssertionError(f"{path.name} contains legal/license overclaim wording")
    if EXTERNAL_PROXY_OVERCLAIM.search(text):
        raise AssertionError(f"{path.name} contains external-provider proxy overclaim wording")


def assert_loopback_base_url(value: Any) -> int:
    if not isinstance(value, str):
        raise AssertionError(f"summary.base_url must be text, got {value!r}")
    match = re.fullmatch(r"http://127\.0\.0\.1:(\d{1,5})", value)
    if not match:
        raise AssertionError("summary.base_url must be an http://127.0.0.1:<port> loopback URL")
    port = int(match.group(1))
    if not 1 <= port <= 65535:
        raise AssertionError(f"summary.base_url port must be in range 1-65535, got {port}")
    return port


def assert_utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise AssertionError(f"{label} must be an RFC3339 UTC timestamp ending in Z")


def assert_markdown_checks_match_summary(checks: list[Any], markdown: str) -> None:
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = check.get("name")
        artifact = check.get("artifact")
        http_status = check.get("http_status")
        status = check.get("status")
        if all(value is not None for value in (name, artifact, http_status, status)):
            row_prefix = f"| `{name}` | `{artifact}` | {http_status} | `{status}` |"
            if row_prefix not in markdown:
                raise AssertionError(f"summary.md missing check row matching summary.json: {name}")


def validate_summary_dir(directory: Path) -> None:
    summary_path = directory / "summary.json"
    summary = load_json(summary_path)
    summary_md = directory / "summary.md"
    summary_local = directory / "summary.local.json"
    if not summary_md.exists():
        raise AssertionError("missing summary.md")
    if not summary_local.exists():
        raise AssertionError("missing summary.local.json")

    assert_public_summary_share_safe(summary_path)
    assert_public_summary_share_safe(summary_md)
    summary_md_text = summary_md.read_text(encoding="utf-8")

    base_url = summary.get("base_url")
    base_url_port = assert_loopback_base_url(base_url)
    if "port" in summary and summary["port"] != base_url_port:
        raise AssertionError("summary.port must match summary.base_url port")
    if f"- Base URL: `{base_url}`" not in summary_md_text:
        raise AssertionError("summary.md must include the summary.json Base URL")
    for key in ("started_at", "finished_at"):
        value = summary.get(key)
        assert_utc_timestamp(value, f"summary.{key}")
        markdown_label = "Started" if key == "started_at" else "Finished"
        if f"- {markdown_label}: `{value}`" not in summary_md_text:
            raise AssertionError(f"summary.md must include the summary.json {key} timestamp")

    if summary.get("local_paths_file") != "summary.local.json":
        raise AssertionError("summary.json must point local_paths_file at summary.local.json")
    for key in ("artifact_dir", "state_dir", "model_dir", "log_dir"):
        value = summary.get(key)
        if not isinstance(value, str) or value.startswith("/"):
            raise AssertionError(f"summary.{key} must be a share-safe relative label, got {value!r}")

    checks = summary.get("checks")
    if not isinstance(checks, list) or not checks:
        raise AssertionError("summary.checks must be a non-empty list")
    assert_markdown_checks_match_summary(checks, summary_md_text)
    seen_check_names: set[str] = set()
    seen_check_artifacts: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            raise AssertionError("each check must be an object")
        for key in ("name", "artifact", "description", "status", "http_status"):
            if key not in check:
                raise AssertionError(f"check missing {key}: {check!r}")
        name = check["name"]
        artifact = check["artifact"]
        if not isinstance(name, str) or not name:
            raise AssertionError(f"check name must be non-empty text: {check!r}")
        if not isinstance(artifact, str) or not artifact:
            raise AssertionError(f"check artifact must be non-empty text: {check!r}")
        artifact_path = Path(artifact)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise AssertionError(f"check artifact must be a relative artifact path: {artifact!r}")
        if not (directory / artifact_path).is_file():
            raise AssertionError(f"check artifact is missing from artifact directory: {artifact!r}")
        if name in seen_check_names:
            raise AssertionError(f"duplicate check name: {name!r}")
        if artifact in seen_check_artifacts:
            raise AssertionError(f"duplicate check artifact: {artifact!r}")
        seen_check_names.add(name)
        seen_check_artifacts.add(artifact)
        if check["status"] not in {"passed", "failed"}:
            raise AssertionError(f"check status must be passed/failed: {check!r}")

    check_names = {check.get("name") for check in checks}
    artifacts = {check.get("artifact") for check in checks}
    has_external_placeholder_evidence = any(
        isinstance(artifact, str) and "external-placeholder" in artifact for artifact in artifacts
    )
    if has_external_placeholder_evidence:
        missing = sorted(EXTERNAL_PLACEHOLDER_CHECKS - check_names)
        if missing:
            raise AssertionError(f"external-placeholder evidence is incomplete; missing checks: {missing}")

    passed = summary.get("passed")
    if passed is True:
        forbidden = ["failure_stage", "failure_message", "failure_type", "last_artifact"]
        present = [key for key in forbidden if key in summary]
        if present:
            raise AssertionError(f"passed summary must not include failure fields: {present}")
        if "Result: `passed`" not in summary_md_text:
            raise AssertionError("passed summary.md must clearly mark Result: passed")
    elif passed is False:
        for key in ("failure_stage", "failure_message", "failure_type"):
            if not summary.get(key):
                raise AssertionError(f"failed summary missing {key}")
        if summary.get("model_dir_snapshot_artifact"):
            if not (directory / summary["model_dir_snapshot_artifact"]).exists():
                raise AssertionError("failed summary references missing model_dir_snapshot_artifact")
        if "Result: `failed`" not in summary_md_text:
            raise AssertionError("failed summary.md must clearly mark Result: failed")
        if "must not be treated as a passed acceptance smoke" not in summary_md_text:
            raise AssertionError("failed summary.md must warn that diagnostics are not a pass")
    else:
        raise AssertionError("summary.passed must be true or false")


def write_sample(directory: Path, *, passed: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checks = [
        {
            "name": "health",
            "artifact": "01-v1-health.json",
            "description": "OpenAI-compatible health endpoint responds.",
            "expected_http_status": 200,
            "http_status": 200,
            "status": "passed",
        },
        {
            "name": "external_placeholder_connected_metadata_only",
            "artifact": "05b-connect-external-placeholder.json",
            "description": "Fake external OpenAI-compatible entry is saved as a planned metadata placeholder with no provider call.",
            "expected_http_status": 200,
            "http_status": 200,
            "status": "passed",
        },
        {
            "name": "external_placeholder_excluded_from_v1_models",
            "artifact": "05c-v1-models-after-external-placeholder.json",
            "description": "External placeholder stays excluded from chat-runnable /v1/models.",
            "expected_http_status": 200,
            "http_status": 200,
            "status": "passed",
        },
        {
            "name": "external_placeholder_activation_refusal",
            "artifact": "05d-external-placeholder-activation-refusal.json",
            "description": "External placeholder activation is refused with external_proxy_not_implemented.",
            "expected_http_status": 501,
            "http_status": 501,
            "status": "passed",
        },
        {
            "name": "external_placeholder_v1_chat_refusal",
            "artifact": "05e-v1-chat-external-placeholder-refusal.json",
            "description": "External placeholder chat completion is refused with structured JSON and no fake choices.",
            "expected_http_status": 501,
            "http_status": 501,
            "status": "passed",
        },
    ]
    summary: dict[str, Any] = {
        "artifact_dir": ".",
        "state_dir": "state/",
        "model_dir": "models/",
        "log_dir": "logs/",
        "local_paths_file": "summary.local.json",
        "base_url": "http://127.0.0.1:18180",
        "port": 18180,
        "repo_commit": "sample",
        "started_at": "2026-04-27T00:00:00Z",
        "finished_at": "2026-04-27T00:00:01Z",
        "fixture_model_ids": {"chat": "sample-chat", "external_placeholder": "acceptance-external-placeholder"},
        "checks": checks,
        "passed": passed,
    }
    if not passed:
        summary.update(
            {
                "failure_stage": "forced failure after capabilities",
                "failure_type": "AssertionError",
                "failure_message": "forced acceptance smoke failure after capabilities",
                "last_artifact": "03-api-capabilities.json",
                "model_dir_snapshot_artifact": "failure-model-dir-snapshot.json",
            }
        )
        (directory / "failure-model-dir-snapshot.json").write_text(
            json.dumps({"model_dir_snapshot": []}, indent=2) + "\n", encoding="utf-8"
        )
    for check in checks:
        (directory / check["artifact"]).write_text(
            json.dumps({"sample_artifact": check["name"]}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "summary.local.json").write_text(
        json.dumps(
            {
                "artifact_dir": "/tmp/local-only/artifacts",
                "state_dir": "/tmp/local-only/state",
                "model_dir": "/tmp/local-only/models",
                "log_dir": "/tmp/local-only/logs",
                "server_log": "/tmp/local-only/logs/server.log",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = "passed" if passed else "failed"
    extra = ""
    if not passed:
        extra = (
            "\n## Failure diagnostics\n\n"
            "- Failure stage: `forced failure after capabilities`\n"
            "- Last artifact: `03-api-capabilities.json`\n"
            "- Message: forced acceptance smoke failure after capabilities\n"
            "- Model directory snapshot: `failure-model-dir-snapshot.json`\n\n"
            "This failed-run summary is diagnostic evidence only; it must not be treated as a passed acceptance smoke.\n"
        )
    (directory / "summary.md").write_text(
        f"# Fathom backend acceptance artifacts{' — failed run' if not passed else ''}\n\n"
        f"- Result: `{result}`\n"
        "- Base URL: `http://127.0.0.1:18180`\n"
        "- Started: `2026-04-27T00:00:00Z`\n"
        "- Finished: `2026-04-27T00:00:01Z`\n"
        "- Artifact directory: `.`\n"
        "- State directory: `state/`\n"
        "- Model directory: `models/`\n"
        "- Server log: `logs/server.log`\n"
        "- Local-only paths: `summary.local.json`\n"
        f"{extra}\n"
        "## Artifact index\n\n"
        "| Check | Artifact | HTTP | Status | What it verifies |\n"
        "| --- | --- | ---: | --- | --- |\n"
        "| `health` | `01-v1-health.json` | 200 | `passed` | OpenAI-compatible health endpoint responds. |\n"
        "| `external_placeholder_connected_metadata_only` | `05b-connect-external-placeholder.json` | 200 | `passed` | Fake external entry is metadata only; no provider call is made. |\n"
        "| `external_placeholder_excluded_from_v1_models` | `05c-v1-models-after-external-placeholder.json` | 200 | `passed` | External placeholder stays excluded from /v1/models. |\n"
        "| `external_placeholder_activation_refusal` | `05d-external-placeholder-activation-refusal.json` | 501 | `passed` | Activation is refused with external_proxy_not_implemented. |\n"
        "| `external_placeholder_v1_chat_refusal` | `05e-v1-chat-external-placeholder-refusal.json` | 501 | `passed` | Chat completion is refused with structured JSON and no fake choices. |\n"
        "\n## What this smoke does not prove\n\n"
        "- Connected external API entries are metadata placeholders only; this smoke does not prove provider calls, external replies, or proxy support.\n"
        "- Catalog license checks prove metadata visibility and gating, not legal review, legal advice, or compatibility for any use case.\n",
        encoding="utf-8",
    )


def run_self_check() -> None:
    with tempfile.TemporaryDirectory(prefix="fathom-acceptance-artifact-qa-") as raw:
        root = Path(raw)
        for name, passed in (("passed", True), ("failed", False)):
            sample = root / name
            write_sample(sample, passed=passed)
            validate_summary_dir(sample)

        overclaim = root / "external-overclaim"
        write_sample(overclaim, passed=True)
        summary_md = overclaim / "summary.md"
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8")
            + "\nUnsafe claim: external model replied after a provider call succeeded.\n",
            encoding="utf-8",
        )
        try:
            validate_summary_dir(overclaim)
        except AssertionError as exc:
            if "external-provider proxy overclaim" not in str(exc):
                raise
        else:
            raise AssertionError("external-provider proxy overclaim self-check did not fail")

        json_leak = root / "summary-json-leak"
        write_sample(json_leak, passed=True)
        summary = load_json(json_leak / "summary.json")
        summary["debug_note"] = "/" + "Users" + "/example/private-run"
        (json_leak / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(json_leak)
        except AssertionError as exc:
            if "summary.json contains a local path-like leak" not in str(exc):
                raise
        else:
            raise AssertionError("summary.json share-safety self-check did not fail")

        secret_leak = root / "summary-secret-leak"
        write_sample(secret_leak, passed=True)
        summary = load_json(secret_leak / "summary.json")
        summary["debug_note"] = "author" + "ization: Bear" + "er sample-token-value"
        (secret_leak / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(secret_leak)
        except AssertionError as exc:
            if "summary.json contains a secret-like marker" not in str(exc):
                raise
        else:
            raise AssertionError("summary.json secret-marker self-check did not fail")

        payload_leak = root / "summary-payload-leak"
        write_sample(payload_leak, passed=True)
        summary_md = payload_leak / "summary.md"
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8")
            + "\nDebug request body: messages: [{'role': 'user', 'content': 'private'}]\n",
            encoding="utf-8",
        )
        try:
            validate_summary_dir(payload_leak)
        except AssertionError as exc:
            if "summary.md contains request/payload text" not in str(exc):
                raise
        else:
            raise AssertionError("summary.md request/payload self-check did not fail")

        external_base_url = root / "external-base-url"
        write_sample(external_base_url, passed=True)
        summary = load_json(external_base_url / "summary.json")
        summary["base_url"] = "https://example.invalid"
        (external_base_url / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(external_base_url)
        except AssertionError as exc:
            if "summary.base_url must be an http://127.0.0.1:<port> loopback URL" not in str(exc):
                raise
        else:
            raise AssertionError("external base URL self-check did not fail")

        markdown_base_url_mismatch = root / "markdown-base-url-mismatch"
        write_sample(markdown_base_url_mismatch, passed=True)
        summary_md = markdown_base_url_mismatch / "summary.md"
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8").replace(
                "- Base URL: `http://127.0.0.1:18180`",
                "- Base URL: `http://127.0.0.1:18181`",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(markdown_base_url_mismatch)
        except AssertionError as exc:
            if "summary.md must include the summary.json Base URL" not in str(exc):
                raise
        else:
            raise AssertionError("Markdown base URL mismatch self-check did not fail")

        mismatched_port = root / "mismatched-port"
        write_sample(mismatched_port, passed=True)
        summary = load_json(mismatched_port / "summary.json")
        summary["port"] = 18181
        (mismatched_port / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(mismatched_port)
        except AssertionError as exc:
            if "summary.port must match summary.base_url port" not in str(exc):
                raise
        else:
            raise AssertionError("mismatched port self-check did not fail")

        malformed_timestamp = root / "malformed-timestamp"
        write_sample(malformed_timestamp, passed=True)
        summary = load_json(malformed_timestamp / "summary.json")
        summary["finished_at"] = "2026-04-27 00:00:01"
        (malformed_timestamp / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(malformed_timestamp)
        except AssertionError as exc:
            if "summary.finished_at must be an RFC3339 UTC timestamp ending in Z" not in str(exc):
                raise
        else:
            raise AssertionError("malformed timestamp self-check did not fail")

        markdown_timestamp_mismatch = root / "markdown-timestamp-mismatch"
        write_sample(markdown_timestamp_mismatch, passed=True)
        summary_md = markdown_timestamp_mismatch / "summary.md"
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8").replace(
                "- Finished: `2026-04-27T00:00:01Z`",
                "- Finished: `2026-04-27T00:00:02Z`",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(markdown_timestamp_mismatch)
        except AssertionError as exc:
            if "summary.md must include the summary.json finished_at timestamp" not in str(exc):
                raise
        else:
            raise AssertionError("Markdown timestamp mismatch self-check did not fail")

        markdown_missing_check = root / "markdown-missing-check"
        write_sample(markdown_missing_check, passed=True)
        summary_md = markdown_missing_check / "summary.md"
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8").replace(
                "| `external_placeholder_v1_chat_refusal` | `05e-v1-chat-external-placeholder-refusal.json` | 501 | `passed` | Chat completion is refused with structured JSON and no fake choices. |\n",
                "",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(markdown_missing_check)
        except AssertionError as exc:
            if "summary.md missing check row matching summary.json" not in str(exc):
                raise
        else:
            raise AssertionError("Markdown check row self-check did not fail")

        missing_external_check = root / "missing-external-placeholder-check"
        write_sample(missing_external_check, passed=True)
        summary = load_json(missing_external_check / "summary.json")
        summary["checks"] = [
            check
            for check in summary["checks"]
            if check.get("name") != "external_placeholder_v1_chat_refusal"
        ]
        (missing_external_check / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(missing_external_check)
        except AssertionError as exc:
            if "external-placeholder evidence is incomplete" not in str(exc):
                raise
        else:
            raise AssertionError("missing external-placeholder check self-check did not fail")

        duplicate_check = root / "duplicate-check-artifact"
        write_sample(duplicate_check, passed=True)
        summary = load_json(duplicate_check / "summary.json")
        summary["checks"].append(dict(summary["checks"][0]))
        (duplicate_check / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            validate_summary_dir(duplicate_check)
        except AssertionError as exc:
            if "duplicate check name" not in str(exc):
                raise
        else:
            raise AssertionError("duplicate check self-check did not fail")

        missing_artifact = root / "missing-check-artifact"
        write_sample(missing_artifact, passed=True)
        (missing_artifact / "05e-v1-chat-external-placeholder-refusal.json").unlink()
        try:
            validate_summary_dir(missing_artifact)
        except AssertionError as exc:
            if "check artifact is missing" not in str(exc):
                raise
        else:
            raise AssertionError("missing check artifact self-check did not fail")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Fathom backend acceptance smoke summaries.")
    parser.add_argument("artifact_dirs", nargs="*", type=Path, help="Artifact directories containing summary.json/summary.md.")
    args = parser.parse_args()
    if not args.artifact_dirs:
        run_self_check()
    else:
        for directory in args.artifact_dirs:
            validate_summary_dir(directory)
    print("backend acceptance artifact QA passed")


if __name__ == "__main__":
    main()
