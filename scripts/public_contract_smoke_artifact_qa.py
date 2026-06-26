#!/usr/bin/env python3
"""Validate public API contract smoke handoff artifacts.

This checker is offline and dependency-free. With no arguments it validates
synthetic passed/failed `public-contract-smoke-summary.*` artifacts so default CI
can regression-test the artifact schema/copy without starting Fathom. With one
or more directories it validates real artifacts produced by
`FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR=... bash scripts/public_api_contract_smoke.sh`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "api" / "public-contract.json"
SUMMARY_JSON = "public-contract-smoke-summary.json"
SUMMARY_MD = "public-contract-smoke-summary.md"
SCHEMA = "fathom.public_contract_smoke.summary.v1"
GENERATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{7,40}|unknown)$")
SCOPE_PHRASES = [
    "no-download",
    "routing/refusal",
    "does not prove model downloads",
    "generation quality",
    "embedding quality",
    "performance",
    "external proxying",
    "gguf runtime, tokenizer execution, or generation claim",
    "broad model support",
]
REQUIRED_NO_DOWNLOAD_BOUNDARIES = {
    "streaming chat completions",
    "base64 embeddings",
    "missing chat model",
    "malformed /v1 JSON request body",
    "unknown embedding model",
    "embedding models in /v1/models",
    "external placeholder chat or activation",
    "PyTorch .bin execution",
    "unsupported ONNX chat or general ONNX model execution",
    "unverified SafeTensors/Hugging Face model execution",
    "GGUF metadata-only chat attempts",
    "unsupported /v1 endpoint",
    "unsupported /v1 method",
}
REQUIRED_EXPECTED_BEHAVIOR_NO_DOWNLOAD_BOUNDARIES = {
    "embedding models in /v1/models",
}
REFUSAL_ONLY_BOUNDARY = "external placeholder chat or activation"
REFUSAL_ONLY_CODE = "external_proxy_not_implemented"

# Build local/private probes without embedding one known user's literal paths.
LOCAL_PATH_PATTERNS = [
    re.compile("/" + "Users" + r"/[^\s`'\"]+", re.IGNORECASE),
    re.compile("/" + "private" + "/" + "tmp", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])/" + "tmp" + r"/", re.IGNORECASE),
    re.compile("/var/" + "folders", re.IGNORECASE),
    re.compile("/" + "opt" + "/" + "homebrew", re.IGNORECASE),
    re.compile(r"\." + "open" + "claw", re.IGNORECASE),
]
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(api[_-]?key|authorization|bearer|token|secret)\b\s*[:=]", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
]
FORBIDDEN_PAYLOAD_PATTERNS = [
    re.compile(r"Sanitized server log tail", re.IGNORECASE),
    re.compile(r"server log tail", re.IGNORECASE),
    re.compile(r"\b(messages|input|prompt|api_base|model_name|provider payload)\b\s*[:=]", re.IGNORECASE),
    re.compile(r"placeholder-key", re.IGNORECASE),
    re.compile(r"api\.example\.test", re.IGNORECASE),
]
OVERCLAIM_PATTERNS = [
    re.compile(r"\b(generation|embedding) quality\b[^\n.]{0,80}\b(pass|passed|verified|proven|works|ready)\b", re.IGNORECASE),
    re.compile(r"\b(performance|throughput|latency|capacity|production readiness)\b[^\n.]{0,80}\b(pass|passed|verified|proven|works|ready)\b", re.IGNORECASE),
    re.compile(r"\b(full|complete|drop-in|100%)\b[^\n.]{0,80}\bOpenAI\b[^\n.]{0,40}\b(parity|compatible|compatibility|replacement)\b", re.IGNORECASE),
    re.compile(r"\b(proxy|proxies|proxied|forwards?|calls?)\b[^\n.]{0,80}\b(external|provider)\b", re.IGNORECASE),
    re.compile(r"\b(GGUF|ONNX|PyTorch|\.bin|SafeTensors|Hugging Face)\b[^\n.]{0,100}\b(runtime|execution|generation|loading|support)\b[^\n.]{0,80}\b(pass|passed|verified|proven|works|ready|enabled)\b", re.IGNORECASE),
]
SAFE_CAVEAT = re.compile(r"\b(no|not|does not|do not|without|unsupported|refused|refusal|excluded|metadata placeholders? only|not implemented|not claimed|deferred)\b", re.IGNORECASE)


def load_manifest() -> dict[str, Any]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("public-contract.json must be an object")
    return data


def manifest_expected_boundaries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for item in manifest.get("expected_boundary_errors", []):
        if not isinstance(item, dict):
            raise AssertionError("manifest expected_boundary_errors entries must be objects")
        boundary = item.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise AssertionError(f"manifest boundary entry missing boundary name: {item!r}")
        if boundary in expected:
            raise AssertionError(f"manifest duplicate boundary entry: {boundary}")
        expected[boundary] = item
    return expected


def assert_no_download_boundary_policy(manifest: dict[str, Any]) -> None:
    """Keep the artifact QA coverage set aligned with public-contract.json."""
    expected = manifest_expected_boundaries(manifest)
    status_code_boundaries = {
        boundary
        for boundary, item in expected.items()
        if "status" in item or "code" in item
    }
    expected_required = status_code_boundaries | REQUIRED_EXPECTED_BEHAVIOR_NO_DOWNLOAD_BOUNDARIES
    if REQUIRED_NO_DOWNLOAD_BOUNDARIES != expected_required:
        raise AssertionError(
            "REQUIRED_NO_DOWNLOAD_BOUNDARIES must match public-contract.json status/code boundaries "
            "plus explicit no-download expected-behavior boundaries: "
            f"missing={sorted(expected_required - REQUIRED_NO_DOWNLOAD_BOUNDARIES)} "
            f"unexpected={sorted(REQUIRED_NO_DOWNLOAD_BOUNDARIES - expected_required)}"
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


def all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_strings(child)


def assert_share_safe(label: str, text: str) -> None:
    for pattern in [*LOCAL_PATH_PATTERNS, *SECRET_PATTERNS, *FORBIDDEN_PAYLOAD_PATTERNS]:
        if pattern.search(text):
            raise AssertionError(f"{label} contains non-share-safe text matching {pattern.pattern!r}")
    for pattern in OVERCLAIM_PATTERNS:
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) and not SAFE_CAVEAT.search(line):
                raise AssertionError(f"{label}:{line_no} contains uncaveated overclaim: {line.strip()}")


def assert_scope(text: str, label: str) -> None:
    lowered = text.lower()
    missing = [phrase for phrase in SCOPE_PHRASES if phrase not in lowered]
    if missing:
        raise AssertionError(f"{label} missing scope caveat phrases: {missing}")


def assert_markdown(summary: dict[str, Any], markdown_path: Path) -> str:
    try:
        md = markdown_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AssertionError(f"missing required artifact: {markdown_path.name}") from exc
    assert_share_safe(markdown_path.name, md)
    status = "PASS" if summary.get("passed") is True else "FAIL"
    required = [
        f"# Public contract smoke summary: {status}",
        "Commit:",
        "Manifest:",
        "Scope:",
        "## Endpoint checks",
        "## Boundary checks",
        "## Manifest boundaries not exercised by this no-download smoke",
    ]
    for needle in required:
        if needle not in md:
            raise AssertionError(f"{markdown_path.name} missing {needle!r}")
    if "docs/api/public-contract.json" not in md:
        raise AssertionError(f"{markdown_path.name} must include the manifest path")
    commit = summary.get("commit")
    if isinstance(commit, str) and commit and f"Commit: `{commit}`" not in md:
        raise AssertionError(f"{markdown_path.name} commit line must match summary.commit")
    generated_at = summary.get("generated_at")
    if isinstance(generated_at, str) and generated_at and f"Generated at: `{generated_at}`" not in md:
        raise AssertionError(f"{markdown_path.name} generated-at line must match summary.generated_at")
    manifest = summary.get("manifest")
    if isinstance(manifest, dict):
        manifest_path = manifest.get("path")
        manifest_name = manifest.get("name")
        manifest_status = manifest.get("status")
        if isinstance(manifest_path, str) and manifest_path and f"`{manifest_path}`" not in md:
            raise AssertionError(f"{markdown_path.name} manifest line must match summary.manifest.path")
        if isinstance(manifest_name, str) and manifest_name and manifest_name not in md:
            raise AssertionError(f"{markdown_path.name} manifest line must match summary.manifest.name")
        if isinstance(manifest_status, str) and manifest_status and manifest_status not in md:
            raise AssertionError(f"{markdown_path.name} manifest line must match summary.manifest.status")
    proof_scope = summary.get("proof_scope")
    if isinstance(proof_scope, str) and proof_scope and f"- Scope: {proof_scope}" not in md:
        raise AssertionError(f"{markdown_path.name} scope line must match summary.proof_scope")
    assert_scope(md, markdown_path.name)
    if summary.get("passed") is False:
        lowered = md.lower()
        if "partial" not in lowered or "must not be treated as a passed public contract smoke" not in lowered:
            raise AssertionError("failed markdown summary must mark checks as partial diagnostics, not a pass")
    return md


def assert_markdown_rows_match_summary(summary: dict[str, Any], md: str, markdown_path: Path) -> None:
    """Keep the shareable Markdown summary aligned with the JSON evidence rows."""
    label = markdown_path.name
    endpoint_checks = summary.get("endpoint_checks")
    boundary_checks = summary.get("boundary_checks")
    deferred = summary.get("deferred_manifest_boundaries")
    if not isinstance(endpoint_checks, list) or not isinstance(boundary_checks, list) or not isinstance(deferred, list):
        return

    for item in endpoint_checks:
        if not isinstance(item, dict):
            continue
        method = item.get("method")
        path = item.get("path")
        if isinstance(method, str) and method and isinstance(path, str) and path:
            expected = f"- {method} {path}: pass"
            if expected not in md:
                raise AssertionError(f"{label} missing endpoint row matching summary JSON: {method} {path}")
            endpoint_line = next((line for line in md.splitlines() if line.startswith(expected)), "")
            checks = item.get("checks")
            if isinstance(checks, list) and checks:
                expected_checks = f"({', '.join(str(check) for check in checks)})"
                if expected_checks not in endpoint_line:
                    raise AssertionError(f"{label} missing endpoint check ids matching summary JSON: {method} {path}")

    for item in boundary_checks:
        if not isinstance(item, dict):
            continue
        boundary = item.get("boundary")
        if isinstance(boundary, str) and boundary:
            expected = f"- {boundary}: pass"
            if expected not in md:
                raise AssertionError(f"{label} missing boundary row matching summary JSON: {boundary}")
            boundary_line = next((line for line in md.splitlines() if line.startswith(expected)), "")
            check = item.get("check")
            if isinstance(check, str) and check and f"({check}" not in boundary_line:
                raise AssertionError(f"{label} missing boundary check id matching summary JSON: {boundary}")
            request_hint = item.get("request_hint")
            if isinstance(request_hint, str) and request_hint and f"hint `{request_hint}`" not in boundary_line:
                raise AssertionError(f"{label} missing boundary request hint matching summary JSON: {boundary}")
            status = item.get("status")
            code = item.get("code")
            if isinstance(status, int) and isinstance(code, str) and code and f"`{status} {code}`" not in boundary_line:
                raise AssertionError(f"{label} missing boundary status/code matching summary JSON: {boundary}")

    for item in deferred:
        if not isinstance(item, dict):
            continue
        boundary = item.get("boundary")
        reason = item.get("reason")
        if isinstance(boundary, str) and boundary and isinstance(reason, str) and reason:
            expected = f"- {boundary}: {reason}"
            if expected not in md:
                raise AssertionError(f"{label} missing deferred boundary row matching summary JSON: {boundary}")


def assert_boundary_result_matches_manifest(
    boundary: str,
    result: dict[str, Any],
    expected_by_name: dict[str, dict[str, Any]],
) -> None:
    expected = expected_by_name.get(boundary)
    if expected is None:
        raise AssertionError(f"summary has boundary not present in public-contract.json: {boundary!r}")
    if "status" in expected and result.get("status") != expected["status"]:
        raise AssertionError(
            f"summary boundary {boundary!r} status {result.get('status')!r} "
            f"does not match public-contract.json {expected['status']!r}"
        )
    if "code" in expected and result.get("code") != expected["code"]:
        raise AssertionError(
            f"summary boundary {boundary!r} code {result.get('code')!r} "
            f"does not match public-contract.json {expected['code']!r}"
        )
    if "request_hint" in expected and result.get("request_hint") != expected["request_hint"]:
        raise AssertionError(
            f"summary boundary {boundary!r} request_hint {result.get('request_hint')!r} "
            f"does not match public-contract.json {expected['request_hint']!r}"
        )


def validate_summary_dir(directory: Path) -> None:
    summary = load_json(directory / SUMMARY_JSON)
    md = assert_markdown(summary, directory / SUMMARY_MD)
    assert_markdown_rows_match_summary(summary, md, directory / SUMMARY_MD)
    assert_share_safe(SUMMARY_JSON, json.dumps(summary, sort_keys=True))

    if summary.get("schema") != SCHEMA:
        raise AssertionError(f"summary.schema must be {SCHEMA!r}")
    generated_at = summary.get("generated_at")
    if not isinstance(generated_at, str) or not GENERATED_AT_RE.fullmatch(generated_at):
        raise AssertionError("summary.generated_at must be an RFC3339 UTC timestamp ending in Z")
    parsed_generated_at = datetime.fromisoformat(generated_at.removesuffix("Z") + "+00:00")
    if parsed_generated_at.tzinfo != timezone.utc:
        raise AssertionError("summary.generated_at must use UTC")
    if not isinstance(summary.get("passed"), bool):
        raise AssertionError("summary.passed must be a boolean")
    commit = summary.get("commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise AssertionError("summary.commit must be a 7-40 character hex git SHA or unknown")
    manifest = summary.get("manifest")
    if not isinstance(manifest, dict):
        raise AssertionError("summary.manifest must be an object")
    for key in ("path", "name", "status"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise AssertionError(f"summary.manifest.{key} must be a non-empty string")
    if manifest["path"] != "docs/api/public-contract.json":
        raise AssertionError("summary.manifest.path must be docs/api/public-contract.json")
    manifest_data = load_manifest()
    assert_no_download_boundary_policy(manifest_data)
    for key in ("name", "status"):
        if manifest[key] != manifest_data.get(key):
            raise AssertionError(f"summary.manifest.{key} must match docs/api/public-contract.json")
    proof_scope = summary.get("proof_scope")
    if not isinstance(proof_scope, str) or not proof_scope:
        raise AssertionError("summary.proof_scope must be a non-empty string")
    assert_scope(proof_scope, "summary.proof_scope")

    endpoint_checks = summary.get("endpoint_checks")
    boundary_checks = summary.get("boundary_checks")
    deferred = summary.get("deferred_manifest_boundaries")
    if not isinstance(endpoint_checks, list):
        raise AssertionError("summary.endpoint_checks must be a list")
    if not isinstance(boundary_checks, list):
        raise AssertionError("summary.boundary_checks must be a list")
    if not isinstance(deferred, list):
        raise AssertionError("summary.deferred_manifest_boundaries must be a list")

    seen_endpoint_checks: set[tuple[str, str]] = set()
    for item in endpoint_checks:
        if not isinstance(item, dict):
            raise AssertionError("endpoint check entries must be objects")
        for key in ("method", "path", "checks", "passed"):
            if key not in item:
                raise AssertionError(f"endpoint check missing {key}: {item!r}")
        method = item.get("method")
        path = item.get("path")
        if not isinstance(method, str) or not method:
            raise AssertionError(f"endpoint check method must be a non-empty string: {item!r}")
        if not isinstance(path, str) or not path:
            raise AssertionError(f"endpoint check path must be a non-empty string: {item!r}")
        endpoint_key = (method, path)
        if endpoint_key in seen_endpoint_checks:
            raise AssertionError(f"duplicate endpoint check entry: {item!r}")
        seen_endpoint_checks.add(endpoint_key)
        if item["passed"] is not True:
            raise AssertionError(f"recorded endpoint checks must be completed passes: {item!r}")
        if not isinstance(item["checks"], list) or not item["checks"]:
            raise AssertionError(f"endpoint check must name at least one check id: {item!r}")
        for check in item["checks"]:
            if not isinstance(check, str) or not check:
                raise AssertionError(f"endpoint check ids must be non-empty strings: {item!r}")

    boundary_by_name: dict[str, dict[str, Any]] = {}
    for item in boundary_checks:
        if not isinstance(item, dict):
            raise AssertionError("boundary check entries must be objects")
        boundary = item.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise AssertionError(f"boundary check missing boundary name: {item!r}")
        if boundary in boundary_by_name:
            raise AssertionError(f"duplicate boundary check entry: {boundary!r}")
        if item.get("passed") is not True:
            raise AssertionError(f"recorded boundary checks must be completed passes: {item!r}")
        if not isinstance(item.get("check"), str) or not item["check"]:
            raise AssertionError(f"boundary check must name a check id: {item!r}")
        boundary_by_name[boundary] = item

    seen_deferred_boundaries: set[str] = set()
    for item in deferred:
        if not isinstance(item, dict):
            raise AssertionError("deferred boundary entries must be objects")
        reason = item.get("reason")
        if not isinstance(item.get("boundary"), str) or not item["boundary"]:
            raise AssertionError(f"deferred boundary missing boundary name: {item!r}")
        boundary = item["boundary"]
        if boundary in seen_deferred_boundaries:
            raise AssertionError(f"duplicate deferred boundary entry: {boundary!r}")
        seen_deferred_boundaries.add(boundary)
        if not isinstance(reason, str) or "requires downloaded/registered model state" not in reason or "outside the no-download smoke" not in reason:
            raise AssertionError(f"deferred boundary reason must preserve no-download caveat: {item!r}")

    checked_and_deferred = sorted(set(boundary_by_name) & seen_deferred_boundaries)
    if checked_and_deferred:
        raise AssertionError(f"boundary cannot be both checked and deferred: {checked_and_deferred}")

    if REFUSAL_ONLY_BOUNDARY in boundary_by_name:
        external = boundary_by_name[REFUSAL_ONLY_BOUNDARY]
        if external.get("status") != 501 or external.get("code") != REFUSAL_ONLY_CODE:
            raise AssertionError("external placeholder boundary must remain a 501 external_proxy_not_implemented refusal")
        external_text = json.dumps(external, sort_keys=True) + "\n" + md
        if REFUSAL_ONLY_CODE not in external_text or "refusal" not in external_text.lower():
            raise AssertionError("external placeholder wording must be refusal-only")

    expected_by_name = manifest_expected_boundaries(manifest_data)
    for boundary, result in boundary_by_name.items():
        assert_boundary_result_matches_manifest(boundary, result, expected_by_name)

    manifest_endpoints = {(item["method"], item["path"]) for item in manifest_data.get("supported_endpoints", [])}
    checked_endpoints = {(item.get("method"), item.get("path")) for item in endpoint_checks}
    unexpected_endpoints = sorted(checked_endpoints - manifest_endpoints)
    if unexpected_endpoints:
        raise AssertionError(f"summary has endpoint checks not present in public-contract.json: {unexpected_endpoints}")

    if summary["passed"] is True:
        missing_endpoints = sorted(manifest_endpoints - checked_endpoints)
        if missing_endpoints:
            raise AssertionError(f"passed summary missing endpoint coverage: {missing_endpoints}")
        missing_boundaries = sorted(REQUIRED_NO_DOWNLOAD_BOUNDARIES - set(boundary_by_name))
        if missing_boundaries:
            raise AssertionError(f"passed summary missing no-download boundary coverage: {missing_boundaries}")
        deferred_names = {item.get("boundary") for item in deferred}
        expected_names = set(expected_by_name)
        expected_deferred_names = expected_names - REQUIRED_NO_DOWNLOAD_BOUNDARIES
        if deferred_names != expected_deferred_names:
            raise AssertionError(
                "passed summary deferred manifest boundaries must exactly match public-contract.json minus no-download boundaries: "
                f"missing={sorted(expected_deferred_names - deferred_names)} unexpected={sorted(deferred_names - expected_deferred_names)}"
            )


def passed_sample() -> dict[str, Any]:
    manifest = load_manifest()
    endpoint_checks = [
        {"method": item["method"], "path": item["path"], "checks": ["synthetic-check"], "passed": True}
        for item in manifest.get("supported_endpoints", [])
    ]
    boundary_checks = []
    for boundary in sorted(REQUIRED_NO_DOWNLOAD_BOUNDARIES):
        item: dict[str, Any] = {"boundary": boundary, "check": "synthetic-refusal-check", "passed": True}
        if boundary == "streaming chat completions":
            item.update({"status": 501, "code": "not_implemented"})
        elif boundary == "base64 embeddings":
            item.update({"status": 400, "code": "invalid_request"})
        elif boundary == "missing chat model":
            item.update({"status": 400, "code": "model_not_found"})
        elif boundary == "malformed /v1 JSON request body":
            item.update({"status": 400, "code": "invalid_request"})
        elif boundary == "unknown embedding model":
            item.update({"status": 404, "code": "embedding_model_not_found"})
        elif boundary == REFUSAL_ONLY_BOUNDARY:
            item.update({"status": 501, "code": REFUSAL_ONLY_CODE})
        elif boundary == "PyTorch .bin execution":
            item.update({"status": 501, "code": "not_implemented"})
        elif boundary == "unsupported ONNX chat or general ONNX model execution":
            item.update({"status": 501, "code": "not_implemented"})
        elif boundary == "unverified SafeTensors/Hugging Face model execution":
            item.update({"status": 501, "code": "not_implemented"})
        elif boundary == "GGUF metadata-only chat attempts":
            item.update({"status": 501, "code": "not_implemented"})
        elif boundary == "unsupported /v1 endpoint":
            item.update({"status": 404, "code": "not_found"})
        elif boundary == "unsupported /v1 method":
            item.update({"status": 405, "code": "method_not_allowed"})
        request_hint = manifest_expected_boundaries(manifest).get(boundary, {}).get("request_hint")
        if request_hint:
            item["request_hint"] = request_hint
        boundary_checks.append(item)
    return {
        "schema": SCHEMA,
        "generated_at": "2026-04-27T00:00:00Z",
        "commit": "abcdef1",
        "manifest": {"path": "docs/api/public-contract.json", "name": manifest.get("name"), "status": manifest.get("status")},
        "passed": True,
        "proof_scope": "No-download real-backend routing/refusal smoke only. Does not prove model downloads, generation quality, embedding quality, performance, external proxying, a GGUF runtime, tokenizer execution, or generation claim, or broad model support.",
        "endpoint_checks": endpoint_checks,
        "boundary_checks": boundary_checks,
        "deferred_manifest_boundaries": [
            {
                "boundary": item["boundary"],
                "reason": "requires downloaded/registered model state or is a non-claim boundary outside the no-download smoke",
            }
            for item in manifest.get("expected_boundary_errors", [])
            if item.get("boundary") not in REQUIRED_NO_DOWNLOAD_BOUNDARIES
        ],
    }


def failed_sample() -> dict[str, Any]:
    sample = passed_sample()
    sample["passed"] = False
    sample["endpoint_checks"] = sample["endpoint_checks"][:1]
    sample["boundary_checks"] = []
    sample["deferred_manifest_boundaries"] = []
    return sample


def write_sample(directory: Path, summary: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "PASS" if summary["passed"] else "FAIL"
    endpoints = [
        f"- {item['method']} {item['path']}: pass ({', '.join(item['checks'])})" for item in summary["endpoint_checks"]
    ] or ["- none completed before failure"]
    boundaries = []
    for item in summary["boundary_checks"]:
        status_code = f"; `{item['status']} {item['code']}`" if item.get("status") and item.get("code") else ""
        hint = f"; hint `{item['request_hint']}`" if item.get("request_hint") else ""
        boundaries.append(f"- {item['boundary']}: pass ({item['check']}{status_code}{hint})")
    boundaries = boundaries or ["- none completed before failure"]
    deferred = [
        f"- {item['boundary']}: {item['reason']}" for item in summary["deferred_manifest_boundaries"]
    ] or ["- none"]
    failure_note = []
    if not summary["passed"]:
        failure_note = [
            "",
            "This failed smoke summary is partial diagnostic evidence only; it must not be treated as a passed public contract smoke.",
        ]
    md = "\n".join(
        [
            f"# Public contract smoke summary: {status}",
            "",
            f"- Commit: `{summary['commit']}`",
            f"- Generated at: `{summary['generated_at']}`",
            f"- Manifest: `docs/api/public-contract.json` / `{summary['manifest']['name']}` (`{summary['manifest']['status']}`)",
            f"- Scope: {summary['proof_scope']}",
            *failure_note,
            "",
            "## Endpoint checks",
            *endpoints,
            "",
            "## Boundary checks",
            *boundaries,
            "",
            "## Manifest boundaries not exercised by this no-download smoke",
            *deferred,
            "",
        ]
    )
    (directory / SUMMARY_MD).write_text(md, encoding="utf-8")


def run_self_check() -> None:
    with tempfile.TemporaryDirectory(prefix="fathom-public-contract-artifact-qa-") as raw:
        root = Path(raw)
        for name, summary in (("passed", passed_sample()), ("failed", failed_sample())):
            sample_dir = root / name
            write_sample(sample_dir, summary)
            validate_summary_dir(sample_dir)

        unsafe = root / "unsafe"
        write_sample(unsafe, passed_sample())
        (unsafe / SUMMARY_MD).write_text(
            (unsafe / SUMMARY_MD).read_text(encoding="utf-8")
            + "\nUnsafe: provider call succeeded with api_key: "
            + "sk-"
            + "this-is-not-share-safe\n",
            encoding="utf-8",
        )
        try:
            validate_summary_dir(unsafe)
        except AssertionError as exc:
            if "non-share-safe" not in str(exc) and "overclaim" not in str(exc):
                raise
        else:
            raise AssertionError("share-safety/overclaim self-check did not fail")

        bad_status = root / "bad-status"
        mutated = passed_sample()
        for item in mutated["boundary_checks"]:
            if item["boundary"] == "streaming chat completions":
                item["status"] = 400
                item["code"] = "invalid_request"
        write_sample(bad_status, mutated)
        try:
            validate_summary_dir(bad_status)
        except AssertionError as exc:
            if "public-contract.json" not in str(exc):
                raise
        else:
            raise AssertionError("manifest status/code drift self-check did not fail")

        bad_request_hint = root / "bad-request-hint"
        mutated = passed_sample()
        for item in mutated["boundary_checks"]:
            if item["boundary"] == "base64 embeddings":
                item["request_hint"] = "encoding_format: float"
        write_sample(bad_request_hint, mutated)
        try:
            validate_summary_dir(bad_request_hint)
        except AssertionError as exc:
            if "request_hint" not in str(exc):
                raise
        else:
            raise AssertionError("manifest request_hint drift self-check did not fail")

        missing_boundary = root / "missing-boundary"
        mutated = passed_sample()
        mutated["boundary_checks"] = [
            item for item in mutated["boundary_checks"] if item["boundary"] != "GGUF metadata-only chat attempts"
        ]
        write_sample(missing_boundary, mutated)
        try:
            validate_summary_dir(missing_boundary)
        except AssertionError as exc:
            if "missing no-download boundary coverage" not in str(exc):
                raise
        else:
            raise AssertionError("missing required no-download boundary self-check did not fail")

        bad_markdown = root / "bad-markdown"
        write_sample(bad_markdown, passed_sample())
        (bad_markdown / SUMMARY_MD).write_text(
            (bad_markdown / SUMMARY_MD).read_text(encoding="utf-8").replace("Commit: `abcdef1`", "Commit: `stale`"),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown)
        except AssertionError as exc:
            if "commit line" not in str(exc):
                raise
        else:
            raise AssertionError("markdown/JSON consistency self-check did not fail")

        bad_markdown_scope = root / "bad-markdown-scope"
        write_sample(bad_markdown_scope, passed_sample())
        (bad_markdown_scope / SUMMARY_MD).write_text(
            (bad_markdown_scope / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace("No-download real-backend routing/refusal smoke only.", "Narrow local smoke only."),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown_scope)
        except AssertionError as exc:
            if "summary.proof_scope" not in str(exc):
                raise
        else:
            raise AssertionError("markdown/proof-scope consistency self-check did not fail")

        bad_proof_scope = root / "bad-proof-scope"
        mutated = passed_sample()
        mutated["proof_scope"] = (
            "No-download real-backend routing/refusal smoke only. "
            "Does not prove model downloads."
        )
        write_sample(bad_proof_scope, mutated)
        try:
            validate_summary_dir(bad_proof_scope)
        except AssertionError as exc:
            if "missing scope caveat phrases" not in str(exc):
                raise
        else:
            raise AssertionError("proof-scope caveat self-check did not fail")

        missing_markdown_boundary = root / "missing-markdown-boundary"
        write_sample(missing_markdown_boundary, passed_sample())
        (missing_markdown_boundary / SUMMARY_MD).write_text(
            (missing_markdown_boundary / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace(
                "- GGUF metadata-only chat attempts: pass (synthetic-refusal-check; `501 not_implemented`; hint `metadata-only GGUF model id in /v1/chat/completions`)\n",
                "",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(missing_markdown_boundary)
        except AssertionError as exc:
            if "missing boundary row" not in str(exc):
                raise
        else:
            raise AssertionError("markdown summary row consistency self-check did not fail")

        missing_markdown_endpoint = root / "missing-markdown-endpoint"
        write_sample(missing_markdown_endpoint, passed_sample())
        first_endpoint = passed_sample()["endpoint_checks"][0]
        (missing_markdown_endpoint / SUMMARY_MD).write_text(
            (missing_markdown_endpoint / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace(
                f"- {first_endpoint['method']} {first_endpoint['path']}: pass (synthetic-check)\n",
                "",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(missing_markdown_endpoint)
        except AssertionError as exc:
            if "missing endpoint row" not in str(exc):
                raise
        else:
            raise AssertionError("markdown endpoint row consistency self-check did not fail")

        bad_markdown_endpoint_checks = root / "bad-markdown-endpoint-checks"
        write_sample(bad_markdown_endpoint_checks, passed_sample())
        first_endpoint = passed_sample()["endpoint_checks"][0]
        (bad_markdown_endpoint_checks / SUMMARY_MD).write_text(
            (bad_markdown_endpoint_checks / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace(
                f"- {first_endpoint['method']} {first_endpoint['path']}: pass (synthetic-check)\n",
                f"- {first_endpoint['method']} {first_endpoint['path']}: pass (stale-check)\n",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown_endpoint_checks)
        except AssertionError as exc:
            if "endpoint check ids" not in str(exc):
                raise
        else:
            raise AssertionError("markdown endpoint check-id consistency self-check did not fail")

        bad_markdown_boundary_check = root / "bad-markdown-boundary-check"
        write_sample(bad_markdown_boundary_check, passed_sample())
        (bad_markdown_boundary_check / SUMMARY_MD).write_text(
            (bad_markdown_boundary_check / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace(
                "- streaming chat completions: pass (synthetic-refusal-check; `501 not_implemented`; hint `stream: true`)\n",
                "- streaming chat completions: pass (stale-check; `501 not_implemented`; hint `stream: true`)\n",
            ),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown_boundary_check)
        except AssertionError as exc:
            if "boundary check id" not in str(exc):
                raise
        else:
            raise AssertionError("markdown boundary check-id consistency self-check did not fail")

        missing_markdown_status_code = root / "missing-markdown-status-code"
        write_sample(missing_markdown_status_code, passed_sample())
        (missing_markdown_status_code / SUMMARY_MD).write_text(
            (missing_markdown_status_code / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace("; `501 not_implemented`; hint `stream: true`", "; hint `stream: true`"),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(missing_markdown_status_code)
        except AssertionError as exc:
            if "status/code" not in str(exc):
                raise
        else:
            raise AssertionError("markdown summary status/code consistency self-check did not fail")

        bad_commit = root / "bad-commit"
        mutated = passed_sample()
        mutated["commit"] = "not-a-sha"
        write_sample(bad_commit, mutated)
        try:
            validate_summary_dir(bad_commit)
        except AssertionError as exc:
            if "summary.commit" not in str(exc):
                raise
        else:
            raise AssertionError("commit shape self-check did not fail")

        bad_generated_at = root / "bad-generated-at"
        mutated = passed_sample()
        mutated["generated_at"] = "2026-04-27 00:00:00"
        write_sample(bad_generated_at, mutated)
        try:
            validate_summary_dir(bad_generated_at)
        except AssertionError as exc:
            if "summary.generated_at" not in str(exc):
                raise
        else:
            raise AssertionError("generated_at timestamp self-check did not fail")

        bad_markdown_generated_at = root / "bad-markdown-generated-at"
        write_sample(bad_markdown_generated_at, passed_sample())
        (bad_markdown_generated_at / SUMMARY_MD).write_text(
            (bad_markdown_generated_at / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace("Generated at: `2026-04-27T00:00:00Z`", "Generated at: `2026-04-27T00:00:01Z`"),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown_generated_at)
        except AssertionError as exc:
            if "generated-at line" not in str(exc):
                raise
        else:
            raise AssertionError("markdown/generated_at consistency self-check did not fail")

        bad_markdown_manifest_status = root / "bad-markdown-manifest-status"
        write_sample(bad_markdown_manifest_status, passed_sample())
        (bad_markdown_manifest_status / SUMMARY_MD).write_text(
            (bad_markdown_manifest_status / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace("(`launch-supported-narrow-local-api`)", "(`stale-status`)"),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown_manifest_status)
        except AssertionError as exc:
            if "manifest line must match summary.manifest.status" not in str(exc):
                raise
        else:
            raise AssertionError("markdown/manifest status consistency self-check did not fail")

        bad_markdown_manifest_name = root / "bad-markdown-manifest-name"
        write_sample(bad_markdown_manifest_name, passed_sample())
        (bad_markdown_manifest_name / SUMMARY_MD).write_text(
            (bad_markdown_manifest_name / SUMMARY_MD)
            .read_text(encoding="utf-8")
            .replace("Fathom public launch API contract", "Stale public launch API contract"),
            encoding="utf-8",
        )
        try:
            validate_summary_dir(bad_markdown_manifest_name)
        except AssertionError as exc:
            if "manifest line must match summary.manifest.name" not in str(exc):
                raise
        else:
            raise AssertionError("markdown/manifest name consistency self-check did not fail")

        stale_manifest = root / "stale-manifest"
        mutated = passed_sample()
        mutated["manifest"]["status"] = "stale-status"
        write_sample(stale_manifest, mutated)
        try:
            validate_summary_dir(stale_manifest)
        except AssertionError as exc:
            if "summary.manifest.status" not in str(exc):
                raise
        else:
            raise AssertionError("manifest metadata drift self-check did not fail")

        stale_manifest_name = root / "stale-manifest-name"
        mutated = passed_sample()
        mutated["manifest"]["name"] = "stale contract name"
        write_sample(stale_manifest_name, mutated)
        try:
            validate_summary_dir(stale_manifest_name)
        except AssertionError as exc:
            if "summary.manifest.name" not in str(exc):
                raise
        else:
            raise AssertionError("manifest name drift self-check did not fail")

        stale_manifest_path = root / "stale-manifest-path"
        mutated = passed_sample()
        mutated["manifest"]["path"] = "docs/api/stale-public-contract.json"
        write_sample(stale_manifest_path, mutated)
        try:
            validate_summary_dir(stale_manifest_path)
        except AssertionError as exc:
            if "summary.manifest.path" not in str(exc):
                raise
        else:
            raise AssertionError("manifest path drift self-check did not fail")

        bad_deferred = root / "bad-deferred"
        mutated = passed_sample()
        mutated["deferred_manifest_boundaries"].append(
            {
                "boundary": "invented boundary",
                "reason": "requires downloaded/registered model state outside the no-download smoke",
            }
        )
        write_sample(bad_deferred, mutated)
        try:
            validate_summary_dir(bad_deferred)
        except AssertionError as exc:
            if "deferred manifest boundaries" not in str(exc):
                raise
        else:
            raise AssertionError("unexpected deferred boundary self-check did not fail")

        checked_deferred_overlap = root / "checked-deferred-overlap"
        mutated = failed_sample()
        checked_boundary = passed_sample()["boundary_checks"][0]
        mutated["boundary_checks"] = [checked_boundary]
        mutated["deferred_manifest_boundaries"] = [
            {
                "boundary": checked_boundary["boundary"],
                "reason": "requires downloaded/registered model state or is a non-claim boundary outside the no-download smoke",
            }
        ]
        write_sample(checked_deferred_overlap, mutated)
        try:
            validate_summary_dir(checked_deferred_overlap)
        except AssertionError as exc:
            if "both checked and deferred" not in str(exc):
                raise
        else:
            raise AssertionError("checked/deferred boundary overlap self-check did not fail")

        failed_boundary_drift = root / "failed-boundary-drift"
        mutated = failed_sample()
        drifted_boundary = dict(passed_sample()["boundary_checks"][0])
        drifted_boundary["status"] = 400
        drifted_boundary["code"] = "invalid_request"
        mutated["boundary_checks"] = [drifted_boundary]
        write_sample(failed_boundary_drift, mutated)
        try:
            validate_summary_dir(failed_boundary_drift)
        except AssertionError as exc:
            if "public-contract.json" not in str(exc):
                raise
        else:
            raise AssertionError("failed summary boundary manifest drift self-check did not fail")

        failed_endpoint_drift = root / "failed-endpoint-drift"
        mutated = failed_sample()
        mutated["endpoint_checks"] = [
            {"method": "GET", "path": "/v1/files", "checks": ["synthetic-check"], "passed": True}
        ]
        write_sample(failed_endpoint_drift, mutated)
        try:
            validate_summary_dir(failed_endpoint_drift)
        except AssertionError as exc:
            if "endpoint checks not present in public-contract.json" not in str(exc):
                raise
        else:
            raise AssertionError("failed summary endpoint manifest drift self-check did not fail")

        duplicate_endpoint = root / "duplicate-endpoint"
        mutated = passed_sample()
        mutated["endpoint_checks"].append(dict(mutated["endpoint_checks"][0]))
        write_sample(duplicate_endpoint, mutated)
        try:
            validate_summary_dir(duplicate_endpoint)
        except AssertionError as exc:
            if "duplicate endpoint check entry" not in str(exc):
                raise
        else:
            raise AssertionError("duplicate endpoint self-check did not fail")

        unexpected_endpoint = root / "unexpected-endpoint"
        mutated = passed_sample()
        mutated["endpoint_checks"].append(
            {"method": "GET", "path": "/v1/files", "checks": ["synthetic-check"], "passed": True}
        )
        write_sample(unexpected_endpoint, mutated)
        try:
            validate_summary_dir(unexpected_endpoint)
        except AssertionError as exc:
            if "endpoint checks not present in public-contract.json" not in str(exc):
                raise
        else:
            raise AssertionError("unexpected endpoint self-check did not fail")

        bad_endpoint_check_id = root / "bad-endpoint-check-id"
        mutated = passed_sample()
        mutated["endpoint_checks"][0]["checks"] = [""]
        write_sample(bad_endpoint_check_id, mutated)
        try:
            validate_summary_dir(bad_endpoint_check_id)
        except AssertionError as exc:
            if "endpoint check ids" not in str(exc):
                raise
        else:
            raise AssertionError("endpoint check-id schema self-check did not fail")

        duplicate_boundary = root / "duplicate-boundary"
        mutated = passed_sample()
        mutated["boundary_checks"].append(dict(mutated["boundary_checks"][0]))
        write_sample(duplicate_boundary, mutated)
        try:
            validate_summary_dir(duplicate_boundary)
        except AssertionError as exc:
            if "duplicate boundary check entry" not in str(exc):
                raise
        else:
            raise AssertionError("duplicate boundary self-check did not fail")

        bad_boundary_check_id = root / "bad-boundary-check-id"
        mutated = passed_sample()
        mutated["boundary_checks"][0]["check"] = ""
        write_sample(bad_boundary_check_id, mutated)
        try:
            validate_summary_dir(bad_boundary_check_id)
        except AssertionError as exc:
            if "boundary check must name a check id" not in str(exc):
                raise
        else:
            raise AssertionError("boundary check-id schema self-check did not fail")

        duplicate_deferred = root / "duplicate-deferred"
        mutated = passed_sample()
        if len(mutated["deferred_manifest_boundaries"]) < 1:
            raise AssertionError("duplicate deferred self-check needs a deferred sample boundary")
        mutated["deferred_manifest_boundaries"].append(dict(mutated["deferred_manifest_boundaries"][0]))
        write_sample(duplicate_deferred, mutated)
        try:
            validate_summary_dir(duplicate_deferred)
        except AssertionError as exc:
            if "duplicate deferred boundary entry" not in str(exc):
                raise
        else:
            raise AssertionError("duplicate deferred self-check did not fail")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate public contract smoke summary artifacts.")
    parser.add_argument("artifact_dirs", nargs="*", type=Path, help=f"Directories containing {SUMMARY_JSON} and {SUMMARY_MD}.")
    args = parser.parse_args()
    if not args.artifact_dirs:
        run_self_check()
    else:
        for directory in args.artifact_dirs:
            validate_summary_dir(directory)
    print("public contract smoke artifact QA passed")


if __name__ == "__main__":
    main()
