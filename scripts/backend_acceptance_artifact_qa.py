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

LOCAL_PATH_PATTERNS = [
    re.compile(r"/Users/", re.IGNORECASE),
    re.compile(r"/private/tmp", re.IGNORECASE),
    re.compile(r"/opt/homebrew", re.IGNORECASE),
]
LEGAL_OVERCLAIM = re.compile(
    r"license\s+(safe|approved|compliant)|legal review completed|legally approved",
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


def assert_no_public_path_leaks(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"{path.name} contains a local path-like leak matching {pattern.pattern}")
    if LEGAL_OVERCLAIM.search(text):
        raise AssertionError(f"{path.name} contains legal/license overclaim wording")


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
        }
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
        "fixture_model_ids": {"chat": "sample-chat"},
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
        "\n## What this smoke does not prove\n\n"
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
