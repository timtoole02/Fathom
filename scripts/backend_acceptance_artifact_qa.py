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


def assert_no_public_path_leaks(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"{path.name} contains a local path-like leak matching {pattern.pattern}")
    if LEGAL_OVERCLAIM.search(text):
        raise AssertionError(f"{path.name} contains legal/license overclaim wording")
    if EXTERNAL_PROXY_OVERCLAIM.search(text):
        raise AssertionError(f"{path.name} contains external-provider proxy overclaim wording")


def validate_summary_dir(directory: Path) -> None:
    summary = load_json(directory / "summary.json")
    summary_md = directory / "summary.md"
    summary_local = directory / "summary.local.json"
    if not summary_md.exists():
        raise AssertionError("missing summary.md")
    if not summary_local.exists():
        raise AssertionError("missing summary.local.json")

    assert_no_public_path_leaks(summary_md)

    if summary.get("local_paths_file") != "summary.local.json":
        raise AssertionError("summary.json must point local_paths_file at summary.local.json")
    for key in ("artifact_dir", "state_dir", "model_dir", "log_dir"):
        value = summary.get(key)
        if not isinstance(value, str) or value.startswith("/"):
            raise AssertionError(f"summary.{key} must be a share-safe relative label, got {value!r}")

    checks = summary.get("checks")
    if not isinstance(checks, list) or not checks:
        raise AssertionError("summary.checks must be a non-empty list")
    for check in checks:
        if not isinstance(check, dict):
            raise AssertionError("each check must be an object")
        for key in ("name", "artifact", "description", "status", "http_status"):
            if key not in check:
                raise AssertionError(f"check missing {key}: {check!r}")
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
        if "Result: `passed`" not in summary_md.read_text(encoding="utf-8"):
            raise AssertionError("passed summary.md must clearly mark Result: passed")
    elif passed is False:
        for key in ("failure_stage", "failure_message", "failure_type"):
            if not summary.get(key):
                raise AssertionError(f"failed summary missing {key}")
        if summary.get("model_dir_snapshot_artifact"):
            if not (directory / summary["model_dir_snapshot_artifact"]).exists():
                raise AssertionError("failed summary references missing model_dir_snapshot_artifact")
        md = summary_md.read_text(encoding="utf-8")
        if "Result: `failed`" not in md:
            raise AssertionError("failed summary.md must clearly mark Result: failed")
        if "must not be treated as a passed acceptance smoke" not in md:
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
