#!/usr/bin/env python3
"""Static QA for Fathom's public launch API contract.

This gate is intentionally dependency-free and offline. It does not start
Fathom, download models, call external APIs, or enable non-default runtime
features. It keeps public docs, examples, README copy, and CI policy aligned
with docs/api/public-contract.json.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "api" / "public-contract.json"
V1_CONTRACT = ROOT / "docs" / "api" / "v1-contract.md"
CLIENT_EXAMPLES = ROOT / "docs" / "api" / "client-examples.md"
BACKEND_QUICKSTART = ROOT / "docs" / "api" / "backend-only-quickstart.md"
LAUNCH_CHECKLIST = ROOT / "docs" / "public-launch-checklist.md"
LAUNCH_EVIDENCE = ROOT / "docs" / "public-launch-evidence.md"
REFUSAL_MATRIX = ROOT / "docs" / "api" / "refusal-boundary-matrix.md"
MINILM_OPTIONAL_ACCEPTANCE = ROOT / "docs" / "api" / "minilm-embeddings-optional-acceptance.md"
SMOLLM2_OPTIONAL_ACCEPTANCE = ROOT / "docs" / "api" / "smollm2-optional-acceptance.md"
QWEN25_OPTIONAL_ACCEPTANCE = ROOT / "docs" / "api" / "qwen25-optional-acceptance.md"
ROADMAP = ROOT / "roadmap.md"
README = ROOT / "README.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
SECURITY = ROOT / "SECURITY.md"
LICENSE_FILE = ROOT / "LICENSE"
GITATTRIBUTES = ROOT / ".gitattributes"
ROOT_CARGO = ROOT / "Cargo.toml"
SERVER_CARGO = ROOT / "crates" / "fathom-server" / "Cargo.toml"
CORE_CARGO = ROOT / "crates" / "fathom-core" / "Cargo.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
API_CONTRACT_ISSUE_TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "api_contract.yml"
MODEL_RUNTIME_ISSUE_TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "model_runtime_request.yml"
BUG_REPORT_ISSUE_TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
SECURITY_PRIVACY_ISSUE_TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "security_or_privacy.yml"
ISSUE_TEMPLATE_CONFIG = ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml"
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"
SMOKE = ROOT / "scripts" / "public_api_contract_smoke.sh"
EXAMPLES_DIR = ROOT / "examples" / "api"

DOC_PATHS = [V1_CONTRACT, CLIENT_EXAMPLES, BACKEND_QUICKSTART, LAUNCH_CHECKLIST, LAUNCH_EVIDENCE, REFUSAL_MATRIX, README]
OPTIONAL_DOC_PATHS = [MINILM_OPTIONAL_ACCEPTANCE, SMOLLM2_OPTIONAL_ACCEPTANCE, QWEN25_OPTIONAL_ACCEPTANCE]
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*"))
TEXT_PATHS = DOC_PATHS + OPTIONAL_DOC_PATHS + EXAMPLE_PATHS + [
    CI,
    API_CONTRACT_ISSUE_TEMPLATE,
    MODEL_RUNTIME_ISSUE_TEMPLATE,
    BUG_REPORT_ISSUE_TEMPLATE,
    SECURITY_PRIVACY_ISSUE_TEMPLATE,
    ISSUE_TEMPLATE_CONFIG,
    PR_TEMPLATE,
]
OFFLINE_QA_PYTHON_PATHS = (
    "scripts/api_client_examples_regression.py",
    "scripts/backend_acceptance_artifact_qa.py",
    "scripts/bench_backend.py",
    "scripts/ci_static_policy.py",
    "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
    "scripts/public_api_contract_qa.py",
    "scripts/public_contract_smoke_artifact_qa.py",
    "scripts/qwen25_optional_api_acceptance_artifact_qa.py",
    "scripts/reference_llama3_tokenizer_ids.py",
    "scripts/smollm2_optional_api_acceptance_artifact_qa.py",
)
OFFLINE_CLIENT_EXAMPLE_PYTHON_PATHS = (
    "examples/api/openai-sdk.py",
    "examples/api/python-no-deps.py",
)
OFFLINE_CLIENT_EXAMPLE_SHELL_PATHS = (
    "examples/api/curl-quickstart.sh",
)
OFFLINE_SHELL_SYNTAX_PATHS = (
    *OFFLINE_CLIENT_EXAMPLE_SHELL_PATHS,
    "scripts/public_risk_scan.sh",
    "scripts/public_api_contract_smoke.sh",
    "scripts/backend_acceptance_smoke.sh",
    "scripts/minilm_embeddings_optional_api_acceptance_smoke.sh",
    "scripts/smollm2_optional_api_acceptance_smoke.sh",
    "scripts/qwen25_optional_api_acceptance_smoke.sh",
    "scripts/smoke.sh",
    "scripts/start-backend.sh",
    "scripts/start.sh",
    "scripts/stop.sh",
)
OFFLINE_ARTIFACT_QA_RUN_PATHS = (
    "scripts/public_contract_smoke_artifact_qa.py",
    "scripts/backend_acceptance_artifact_qa.py",
    "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
    "scripts/smollm2_optional_api_acceptance_artifact_qa.py",
    "scripts/qwen25_optional_api_acceptance_artifact_qa.py",
)
OPTIONAL_ACCEPTANCE_DOCS = (
    (
        MINILM_OPTIONAL_ACCEPTANCE,
        "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE=1",
        "scripts/minilm_embeddings_optional_api_acceptance_smoke.sh",
        "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
        "embedding quality",
    ),
    (
        SMOLLM2_OPTIONAL_ACCEPTANCE,
        "FATHOM_SMOLLM2_ACCEPTANCE=1",
        "scripts/smollm2_optional_api_acceptance_smoke.sh",
        "scripts/smollm2_optional_api_acceptance_artifact_qa.py",
        "larger-demo evidence only",
    ),
    (
        QWEN25_OPTIONAL_ACCEPTANCE,
        "FATHOM_QWEN25_ACCEPTANCE=1",
        "scripts/qwen25_optional_api_acceptance_smoke.sh",
        "scripts/qwen25_optional_api_acceptance_artifact_qa.py",
        "larger-demo evidence only",
    ),
)
PUBLIC_CONTRACT_QA_HARDENING_SUBJECT_PATTERN = (
    r"^(Harden public .+ QA|Expose refusal request hints in matrix|Guard public .+ artifact .+|"
    r"Track public smoke artifact QA evidence|Derive public smoke boundaries from manifest|"
    r"Tighten public smoke .+|Guard refusal matrix row drift|Guard failed public smoke .+ drift|"
    r"Standardize v1 unsupported endpoint refusals|Standardize v1 malformed JSON refusals|"
    r"Harden API contract issue privacy checks|Guard PR template truthfulness privacy checks|"
    r"Guard public issue template privacy checks|Guard public issue template required fields|"
    r"Guard issue template config privacy checks|"
    r"Guard OpenAI SDK example regression|Guard CI token permissions|Guard offline shell syntax coverage|"
    r"Guard offline Python syntax coverage|Guard API example loopback defaults|"
    r"Guard REST Client example headers|Guard API example regression self-test|"
    r"Guard CI frontend launch gates|Guard launch syntax checklist consistency|"
    r"Guard contributing syntax gate consistency|Guard launch clean install consistency|"
    r"Guard launch text normalization metadata|"
    r"Guard public risk scan .+|Guard tracked credential config files|"
    r"Guard tracked workspace instruction files|Guard tracked local runtime artifacts)$"
)
NO_DOWNLOAD_REFUSAL_EVIDENCE_SUBJECT_PATTERN = (
    r"^(Promote GGUF refusal to public smoke|Standardize v1 unsupported endpoint refusals|"
    r"Standardize v1 malformed JSON refusals|Prove embeddings malformed JSON refusal)$"
)
ALLOWED_EXTRA_REFUSAL_MATRIX_ROWS = {
    "Production readiness, performance, quality, legal/license suitability",
    "Real external provider proxying",
}

REQUIRED_ERROR_CODES = {
    "invalid_request",
    "model_not_found",
    "embedding_model_not_found",
    "not_found",
    "method_not_allowed",
    "not_implemented",
    "external_proxy_not_implemented",
}

DANGEROUS_POSITIVE_PATTERNS = [
    (
        "streaming chat appears supported",
        re.compile(r"\b(supports?|enabled|provides|offers|uses|send|sends)\b[^\n.]{0,80}\bstream(ing)?\b", re.I),
        re.compile(r"\b(not supported|unsupported|rejected|refused|must be omitted|false|non-streaming|stream\s*[:=]\s*(false|False))\b", re.I),
    ),
    (
        "external provider proxying appears supported",
        re.compile(r"\b(proxy|proxies|proxied|forwards?|calls?)\b[^\n.]{0,100}\bexternal\b", re.I),
        re.compile(r"\b(not implemented|not proxied|metadata placeholders? only|excluded|refused|does not|do not|without calling)\b", re.I),
    ),
    (
        "full OpenAI parity appears claimed",
        re.compile(r"\b(full|complete|drop-in|100%)\b[^\n.]{0,80}\bOpenAI\b[^\n.]{0,40}\b(parity|compatible|compatibility|replacement)\b", re.I),
        re.compile(r"\b(not|no|does not|do not|without|smaller than|doesn't)\b", re.I),
    ),
    (
        "broad model execution appears claimed",
        re.compile(r"\b(arbitrary|any|all|general)\b[^\n.]{0,80}\b(SafeTensors|Hugging Face|ONNX|GGUF)\b[^\n.]{0,80}\b(execution|inference|runtime|generation|support)\b", re.I),
        re.compile(r"\b(no|not|unsupported|blocked|refused|metadata-only|does not|do not|without claiming|not supported)\b", re.I),
    ),
    (
        "base64 embeddings appear supported",
        re.compile(r"\bbase64\b[^\n.]{0,80}\b(embedding|embeddings)\b", re.I),
        re.compile(r"\b(not supported|unsupported|rejected|refused|invalid_request|only float|float only|do not|don't|no)\b", re.I),
    ),
    (
        "production readiness appears claimed",
        re.compile(r"\b(production[- ]ready|production readiness|ready for production|production deployment)\b", re.I),
        re.compile(r"\b(not|no|does not|do not|without|doesn't|does not prove|not prove|unproven|overclaim)\b", re.I),
    ),
    (
        "legal/license suitability appears claimed",
        re.compile(
            r"\b(legal|license|licensing)\b[^\n.]{0,80}\b(safe|approved|compliant|suitable|cleared|ready|verified|proven)\b",
            re.I,
        ),
        re.compile(r"\b(not|no|does not|do not|without|doesn't|advice|review before|does not prove|not prove)\b", re.I),
    ),
]


def assert_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{label} must be a non-empty string")


def assert_manifest_shape(manifest: dict[str, Any]) -> None:
    for key in ("name", "status", "base_url", "scope_note"):
        assert_non_empty_string(manifest.get(key), f"manifest.{key}")

    endpoints = manifest.get("supported_endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise AssertionError("manifest.supported_endpoints must be a non-empty list")
    seen_endpoints: set[tuple[str, str]] = set()
    for index, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, dict):
            raise AssertionError(f"manifest.supported_endpoints[{index}] must be an object")
        method = endpoint.get("method")
        path = endpoint.get("path")
        purpose = endpoint.get("purpose")
        assert_non_empty_string(method, f"manifest.supported_endpoints[{index}].method")
        assert_non_empty_string(path, f"manifest.supported_endpoints[{index}].path")
        assert_non_empty_string(purpose, f"manifest.supported_endpoints[{index}].purpose")
        if method not in {"GET", "POST"}:
            raise AssertionError(f"manifest endpoint uses unsupported method {method!r}: {endpoint!r}")
        if not path.startswith("/v1/"):
            raise AssertionError(f"manifest supported endpoint must stay in the /v1 public contract: {endpoint!r}")
        endpoint_key = (method, path)
        if endpoint_key in seen_endpoints:
            raise AssertionError(f"manifest duplicate supported endpoint: {method} {path}")
        seen_endpoints.add(endpoint_key)
        if "required_boundary" in endpoint:
            assert_non_empty_string(endpoint.get("required_boundary"), f"manifest.supported_endpoints[{index}].required_boundary")

    envelope = manifest.get("standard_error_envelope")
    if not isinstance(envelope, dict):
        raise AssertionError("manifest.standard_error_envelope must be an object")
    if envelope.get("top_level") != ["error"]:
        raise AssertionError("manifest.standard_error_envelope.top_level must be ['error']")
    if envelope.get("error_fields") != ["message", "type", "code", "param"]:
        raise AssertionError("manifest.standard_error_envelope.error_fields must be message/type/code/param")

    boundaries = manifest.get("expected_boundary_errors")
    if not isinstance(boundaries, list) or not boundaries:
        raise AssertionError("manifest.expected_boundary_errors must be a non-empty list")
    seen_boundaries: set[str] = set()
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, dict):
            raise AssertionError(f"manifest.expected_boundary_errors[{index}] must be an object")
        name = boundary.get("boundary")
        assert_non_empty_string(name, f"manifest.expected_boundary_errors[{index}].boundary")
        if name in seen_boundaries:
            raise AssertionError(f"manifest duplicate expected boundary: {name}")
        seen_boundaries.add(name)

        has_status = "status" in boundary
        has_code = "code" in boundary
        has_expected_behavior = "expected_behavior" in boundary
        if has_status != has_code:
            raise AssertionError(f"manifest boundary must include status and code together: {boundary!r}")
        if has_status:
            status = boundary.get("status")
            code = boundary.get("code")
            if not isinstance(status, int) or status < 400 or status > 599:
                raise AssertionError(f"manifest boundary status must be a 4xx/5xx integer: {boundary!r}")
            assert_non_empty_string(code, f"manifest.expected_boundary_errors[{index}].code")
            if code not in REQUIRED_ERROR_CODES:
                raise AssertionError(f"manifest boundary code {code!r} is not in the documented public error-code set")
            assert_non_empty_string(boundary.get("request_hint"), f"manifest.expected_boundary_errors[{index}].request_hint")
        if has_expected_behavior:
            assert_non_empty_string(
                boundary.get("expected_behavior"),
                f"manifest.expected_boundary_errors[{index}].expected_behavior",
            )
        if not has_status and not has_expected_behavior:
            raise AssertionError(f"manifest boundary must include status/code or expected_behavior: {boundary!r}")
        if "request_hint" in boundary and not has_status:
            assert_non_empty_string(boundary.get("request_hint"), f"manifest.expected_boundary_errors[{index}].request_hint")

    allowed = manifest.get("non_contract_surfaces_allowed_in_examples")
    if not isinstance(allowed, list):
        raise AssertionError("manifest.non_contract_surfaces_allowed_in_examples must be a list")
    seen_allowed: set[str] = set()
    for index, item in enumerate(allowed):
        assert_non_empty_string(item, f"manifest.non_contract_surfaces_allowed_in_examples[{index}]")
        if item in seen_allowed:
            raise AssertionError(f"manifest duplicate non-contract example surface: {item}")
        seen_allowed.add(item)

    ci_policy = manifest.get("ci_policy")
    if not isinstance(ci_policy, dict):
        raise AssertionError("manifest.ci_policy must be an object")
    assert_non_empty_string(ci_policy.get("offline_static_gate"), "manifest.ci_policy.offline_static_gate")
    for key in ("no_default_network_acceptance_smoke", "no_default_onnx_embeddings_ort", "no_default_model_downloads"):
        if not isinstance(ci_policy.get(key), bool):
            raise AssertionError(f"manifest.ci_policy.{key} must be a boolean")


def load_manifest() -> dict[str, Any]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("public contract manifest must be a JSON object")
    return data


def allowed_example_endpoints(manifest: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        if not isinstance(method, str) or not isinstance(path, str):
            raise AssertionError("manifest supported_endpoints entries must include string method/path")
        allowed.add(f"{method} {path}")

    for item in manifest.get("non_contract_surfaces_allowed_in_examples", []):
        if not isinstance(item, str):
            raise AssertionError("manifest non-contract example surfaces must be strings")
        try:
            _method, path = item.split(" ", 1)
        except ValueError as exc:
            raise AssertionError(f"invalid non-contract example surface {item!r}") from exc
        if path.startswith("/v1/"):
            raise AssertionError(f"non-contract example surface must not be a /v1 endpoint: {item}")
        allowed.add(item)
    return allowed


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} missing {needle!r}")


def issue_template_field_block(template_text: str, field_id: str, label: str) -> str:
    lines = template_text.splitlines()
    matches = [
        index
        for index, line in enumerate(lines)
        if re.match(rf"^\s+id:\s*{re.escape(field_id)}\s*(?:#.*)?$", line)
    ]
    if len(matches) != 1:
        raise AssertionError(f"{label} must contain exactly one issue-template field id {field_id!r}")

    start = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^  - type:\s+", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def assert_issue_template_required_field(template_text: str, label: str, field_id: str) -> None:
    block = issue_template_field_block(template_text, field_id, label)
    if "validations:" not in block or not re.search(r"(?m)^\s*required:\s*true\s*(?:#.*)?$", block):
        raise AssertionError(f"{label} field {field_id!r} must remain required")


def assert_issue_template_required_fields(template_text: str, label: str, field_ids: tuple[str, ...]) -> None:
    for field_id in field_ids:
        assert_issue_template_required_field(template_text, label, field_id)


def assert_issue_template_required_checkbox_options(template_text: str, label: str, field_id: str) -> None:
    block = issue_template_field_block(template_text, field_id, label)
    lines = block.splitlines()
    option_starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s+- label:\s+", line)
    ]
    if not option_starts:
        raise AssertionError(f"{label} checkbox field {field_id!r} must contain checkbox options")

    for position, start in enumerate(option_starts):
        end = option_starts[position + 1] if position + 1 < len(option_starts) else len(lines)
        option_block = "\n".join(lines[start:end])
        if not re.search(r"(?m)^\s*required:\s*true\s*(?:#.*)?$", option_block):
            option_label = lines[start].strip()
            raise AssertionError(f"{label} checkbox option must remain required: {option_label}")


def py_compile_command_paths(text: str, label: str) -> set[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("python3 -m py_compile"):
            continue
        command_lines = [line.strip()]
        cursor = index
        while command_lines[-1].endswith("\\"):
            cursor += 1
            if cursor >= len(lines):
                raise AssertionError(f"{label} has an unterminated py_compile continuation")
            command_lines.append(lines[cursor].strip())
        command = " ".join(part.removesuffix("\\").strip() for part in command_lines)
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise AssertionError(f"{label} has an invalid py_compile command: {exc}") from exc
        if tokens[:3] != ["python3", "-m", "py_compile"]:
            raise AssertionError(f"{label} has an unexpected py_compile command shape")
        return set(tokens[3:])
    raise AssertionError(f"{label} is missing a python3 -m py_compile command")


def bash_syntax_command_paths(text: str) -> set[str]:
    paths: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("bash -n "):
            continue
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            raise AssertionError(f"invalid bash -n command: {exc}") from exc
        if len(tokens) != 3 or tokens[:2] != ["bash", "-n"]:
            raise AssertionError(f"unexpected bash -n command shape: {stripped}")
        paths.add(tokens[2])
    return paths


def assert_python_syntax_paths(text: str, label: str, required: set[str]) -> None:
    paths = py_compile_command_paths(text, label)
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(f"{label} py_compile gate is missing offline Python gate(s): {missing}")
    unexpected = sorted(paths - required)
    if unexpected:
        raise AssertionError(f"{label} py_compile gate includes unexpected Python path(s): {unexpected}")


def assert_shell_syntax_paths(text: str, label: str, required: set[str]) -> None:
    paths = bash_syntax_command_paths(text)
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(f"{label} bash -n gate is missing offline shell gate(s): {missing}")
    unexpected = sorted(paths - required)
    if unexpected:
        raise AssertionError(f"{label} bash -n gate includes unexpected shell path(s): {unexpected}")


def assert_launch_checklist_python_syntax_gate() -> None:
    required = set(OFFLINE_QA_PYTHON_PATHS) | set(OFFLINE_CLIENT_EXAMPLE_PYTHON_PATHS)
    assert_python_syntax_paths(read(LAUNCH_CHECKLIST), "launch checklist", required)


def tracked_python_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return set(completed.stdout.splitlines())


def tracked_shell_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "*.sh"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return set(completed.stdout.splitlines())


def assert_tracked_python_syntax_coverage() -> None:
    covered = set(OFFLINE_QA_PYTHON_PATHS) | set(OFFLINE_CLIENT_EXAMPLE_PYTHON_PATHS)
    missing = sorted(tracked_python_paths() - covered)
    if missing:
        raise AssertionError(f"tracked Python files missing from offline syntax coverage: {missing}")


def assert_tracked_shell_syntax_coverage() -> None:
    covered = set(OFFLINE_SHELL_SYNTAX_PATHS)
    missing = sorted(tracked_shell_paths() - covered)
    if missing:
        raise AssertionError(f"tracked shell scripts missing from offline syntax coverage: {missing}")


def assert_launch_checklist_client_example_syntax_gates() -> None:
    assert_shell_syntax_paths(read(LAUNCH_CHECKLIST), "launch checklist", set(OFFLINE_SHELL_SYNTAX_PATHS))


def assert_launch_checklist_artifact_qa_run_gates() -> None:
    checklist_text = read(LAUNCH_CHECKLIST)
    for path in OFFLINE_ARTIFACT_QA_RUN_PATHS:
        assert_contains(checklist_text, f"python3 {path}", "launch checklist artifact QA run gates")


def assert_launch_checklist_frontend_gates() -> None:
    checklist_text = read(LAUNCH_CHECKLIST)
    assert_frontend_launch_gates(checklist_text, "launch checklist frontend gate")


def assert_clean_install_gate(text: str, label: str) -> None:
    assert_contains(text, "npm --prefix frontend ci", label)
    if re.search(r"\bnpm\s+--prefix\s+frontend\s+install\b", text):
        raise AssertionError(f"{label} must use npm ci, not npm install")


def assert_launch_checklist_clean_install_gate() -> None:
    assert_clean_install_gate(read(LAUNCH_CHECKLIST), "launch checklist clean install gate")


def assert_frontend_lockfile_evidence(evidence_text: str) -> None:
    assert_contains(
        evidence_text,
        "lockfile-reproducible frontend clean install",
        "launch evidence frontend clean-install QA scope",
    )


def assert_contributing_common_gates() -> None:
    contributing_text = read(CONTRIBUTING)
    assert_python_syntax_paths(
        contributing_text,
        "contributing",
        set(OFFLINE_QA_PYTHON_PATHS) | set(OFFLINE_CLIENT_EXAMPLE_PYTHON_PATHS),
    )
    assert_shell_syntax_paths(contributing_text, "contributing", set(OFFLINE_SHELL_SYNTAX_PATHS))
    required_commands = (
        "git diff --check",
        "python3 -m py_compile",
        "bash -n",
        "python3 scripts/api_client_examples_regression.py",
        "python3 scripts/api_client_examples_regression.py --self-test",
        "python3 scripts/public_api_contract_qa.py",
        "python3 scripts/public_api_contract_qa.py --self-test",
        "python3 scripts/public_contract_smoke_artifact_qa.py",
        "python3 scripts/backend_acceptance_artifact_qa.py",
        "python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
        "python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py",
        "python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py",
        "python3 scripts/ci_static_policy.py",
        "python3 scripts/ci_static_policy.py --self-test",
        "bash scripts/public_risk_scan.sh --self-test",
        "bash scripts/public_risk_scan.sh",
    )
    for command in required_commands:
        assert_contains(contributing_text, command, "contributing common gates")


def assert_frontend_launch_gates(text: str, label: str) -> None:
    for command in (
        "npm --prefix frontend run build",
        "npm --prefix frontend run qa:copy",
    ):
        assert_contains(text, command, label)


def assert_public_security_docs() -> None:
    risk_scan_caveat = "not a complete privacy audit"
    requirements = {
        SECURITY: (
            "no built-in authentication",
            "Do not expose Fathom directly to the public internet or an untrusted LAN",
            "proxy, tunnel",
            "hostnames",
            "private documents",
            "sensitive artifact details",
            "model-store details",
            "reproduction steps",
            risk_scan_caveat,
        ),
        README: (
            "no built-in authentication",
            "loopback development",
            "not direct internet or untrusted-LAN exposure",
            "SECURITY.md",
            risk_scan_caveat,
        ),
        BACKEND_QUICKSTART: (
            "no built-in authentication",
            "loopback development",
            "internet or an untrusted LAN",
            "../../SECURITY.md",
            risk_scan_caveat,
        ),
        V1_CONTRACT: (
            "no built-in authentication",
            "loopback development",
            "not direct internet exposure",
            "../../SECURITY.md",
        ),
        LAUNCH_CHECKLIST: (
            "no built-in authentication",
            "loopback development",
            "internet or an untrusted LAN",
            "SECURITY.md",
            risk_scan_caveat,
        ),
        LAUNCH_EVIDENCE: (
            "local API no-auth/loopback security-note coverage",
            "internet/untrusted-LAN exposure",
            "`SECURITY.md` review warnings",
            risk_scan_caveat,
        ),
    }
    for path, phrases in requirements.items():
        text = read(path)
        label = f"{path.relative_to(ROOT)} local API security note"
        for phrase in phrases:
            assert_contains(text, phrase, label)


def assert_license_metadata() -> None:
    license_text = read(LICENSE_FILE)
    if not license_text.startswith("MIT License\n"):
        raise AssertionError("LICENSE must keep the repository MIT license header")

    root_cargo_text = read(ROOT_CARGO)
    assert_contains(root_cargo_text, 'license = "MIT"', "workspace Cargo license metadata")
    for path in (SERVER_CARGO, CORE_CARGO):
        assert_contains(read(path), "license.workspace = true", f"{path.relative_to(ROOT)} license inheritance")

    readme_text = read(README)
    assert_contains(readme_text, "## License", "README license section")
    assert_contains(readme_text, "MIT License", "README license section")
    assert_contains(readme_text, "[`LICENSE`](LICENSE)", "README license section")

    evidence_text = read(LAUNCH_EVIDENCE)
    assert_contains(
        evidence_text,
        "repository-level MIT license metadata/docs consistency guard",
        "launch evidence license metadata scope",
    )
    assert_contains(
        evidence_text,
        "Cargo manifests, README, and `LICENSE`",
        "launch evidence license metadata proof",
    )


def assert_gitattributes_text_normalization(text: str | None = None, label: str = ".gitattributes") -> None:
    if text is None:
        text = read(GITATTRIBUTES)
    active_lines = {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required_lines = {
        "* text=auto eol=lf",
        "*.7z binary",
        "*.aab binary",
        "*.aac binary",
        "*.apk binary",
        "*.app binary",
        "*.avi binary",
        "*.bin binary",
        "*.bz2 binary",
        "*.ckpt binary",
        "*.deb binary",
        "*.dmg binary",
        "*.gguf binary",
        "*.gz binary",
        "*.dSYM binary",
        "*.egg binary",
        "*.flac binary",
        "*.gif binary",
        "*.ico binary",
        "*.ipa binary",
        "*.jpeg binary",
        "*.jpg binary",
        "*.m4a binary",
        "*.m4v binary",
        "*.mkv binary",
        "*.mobileprovision binary",
        "*.mov binary",
        "*.mp3 binary",
        "*.mp4 binary",
        "*.msi binary",
        "*.npy binary",
        "*.npz binary",
        "*.onnx binary",
        "*.pdf binary",
        "*.pkg binary",
        "*.png binary",
        "*.provisionprofile binary",
        "*.pt binary",
        "*.pth binary",
        "*.rar binary",
        "*.rpm binary",
        "*.safetensors binary",
        "*.tar binary",
        "*.tfplan binary",
        "*.tgz binary",
        "*.tflite binary",
        "*.wav binary",
        "*.webm binary",
        "*.webp binary",
        "*.zip binary",
        "*.whl binary",
        "*.xcarchive binary",
        "*.xcresult binary",
        "*.xz binary",
        "*.zst binary",
    }
    missing = sorted(required_lines - active_lines)
    if missing:
        raise AssertionError(f"{label} missing text-normalization metadata: {missing}")
    if re.search(r"(?m)^\*\s+-text\b", text):
        raise AssertionError(f"{label} must not disable text normalization for the whole repository")
    if re.search(r"(?m)^\*\s+binary\b", text):
        raise AssertionError(f"{label} must not mark the whole repository as binary")


def latest_public_contract_qa_hardening_commit() -> tuple[str, str]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                f"--grep={PUBLIC_CONTRACT_QA_HARDENING_SUBJECT_PATTERN}",
                "--extended-regexp",
                "--format=%H%x00%s",
                "--",
                "docs/api/refusal-boundary-matrix.md",
                ".github/ISSUE_TEMPLATE/api_contract.yml",
                ".github/ISSUE_TEMPLATE/bug_report.yml",
                ".github/ISSUE_TEMPLATE/config.yml",
                ".github/ISSUE_TEMPLATE/security_or_privacy.yml",
                ".github/pull_request_template.md",
                "scripts/api_client_examples_regression.py",
                "scripts/backend_acceptance_artifact_qa.py",
                "scripts/ci_static_policy.py",
                "scripts/public_api_contract_qa.py",
                "scripts/public_contract_smoke_artifact_qa.py",
                "scripts/public_risk_scan.sh",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise AssertionError("could not resolve latest public-contract QA hardening commit from local git history") from exc

    if not output:
        raise AssertionError("local git history has no recognized public-contract QA hardening commit")
    commit, subject = output.split("\0", 1)
    return commit, subject


def assert_latest_public_contract_qa_hardening_evidence(evidence_text: str) -> None:
    match = re.search(
        r"^- Latest public-contract QA hardening commit: `([0-9a-f]{40})` \(`([^`]+)`\)$",
        evidence_text,
        re.MULTILINE,
    )
    if not match:
        raise AssertionError("launch evidence latest public-contract QA hardening commit line is missing or malformed")

    evidence_commit, evidence_subject = match.groups()
    latest_commit, latest_subject = latest_public_contract_qa_hardening_commit()
    if evidence_commit != latest_commit or evidence_subject != latest_subject:
        raise AssertionError(
            "launch evidence public-contract QA hardening commit is stale: "
            f"expected `{latest_commit}` (`{latest_subject}`), found `{evidence_commit}` (`{evidence_subject}`)"
        )


def latest_no_download_refusal_evidence_commit() -> tuple[str, str]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                f"--grep={NO_DOWNLOAD_REFUSAL_EVIDENCE_SUBJECT_PATTERN}",
                "--extended-regexp",
                "--format=%H%x00%s",
                "--",
                "docs/public-launch-evidence.md",
                "scripts/public_api_contract_smoke.sh",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise AssertionError("could not resolve latest no-download refusal evidence commit from local git history") from exc

    if not output:
        raise AssertionError("local git history has no recognized no-download refusal evidence commit")
    commit, subject = output.split("\0", 1)
    return commit, subject


def assert_latest_no_download_refusal_evidence(evidence_text: str) -> None:
    match = re.search(
        r"^- Latest no-download refusal evidence commit: `([0-9a-f]{40})` \(`([^`]+)`\)$",
        evidence_text,
        re.MULTILINE,
    )
    if not match:
        raise AssertionError("launch evidence latest no-download refusal evidence commit line is missing or malformed")

    evidence_commit, evidence_subject = match.groups()
    latest_commit, latest_subject = latest_no_download_refusal_evidence_commit()
    if evidence_commit != latest_commit or evidence_subject != latest_subject:
        raise AssertionError(
            "launch evidence no-download refusal evidence commit is stale: "
            f"expected `{latest_commit}` (`{latest_subject}`), found `{evidence_commit}` (`{evidence_subject}`)"
        )


def latest_commit_date_for_path(path: Path) -> date:
    rel_path = str(path.relative_to(ROOT))
    try:
        output = subprocess.check_output(
            ["git", "log", "-1", "--format=%cs", "--", rel_path],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise AssertionError(f"could not resolve latest commit date for {rel_path}") from exc

    if not output:
        raise AssertionError(f"local git history has no commits for {rel_path}")
    return datetime.strptime(output, "%Y-%m-%d").date()


def assert_roadmap_last_updated_freshness() -> None:
    roadmap_text = read(ROADMAP)
    match = re.search(r"^_Last updated: (\d{4}-\d{2}-\d{2})_$", roadmap_text, re.MULTILINE)
    if not match:
        raise AssertionError("roadmap last-updated line is missing or malformed")

    stated_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
    latest_roadmap_commit_date = latest_commit_date_for_path(ROADMAP)
    if stated_date < latest_roadmap_commit_date:
        raise AssertionError(
            "roadmap last-updated date is stale: "
            f"expected at least {latest_roadmap_commit_date.isoformat()}, found {stated_date.isoformat()}"
        )
    if stated_date > date.today():
        raise AssertionError(f"roadmap last-updated date is in the future: {stated_date.isoformat()}")


def assert_endpoint_docs(manifest: dict[str, Any]) -> None:
    v1_text = read(V1_CONTRACT)
    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint["method"]
        path = endpoint["path"]
        assert_contains(v1_text, f"{method} {path}", "docs/api/v1-contract.md")
        if path.startswith("/v1/"):
            assert_contains(read(CLIENT_EXAMPLES) + read(README) + read(BACKEND_QUICKSTART), path, "public docs")

    for field in manifest["standard_error_envelope"]["error_fields"]:
        assert_contains(v1_text, f'"{field}"', "standard error envelope")

    for code in REQUIRED_ERROR_CODES:
        combined_docs = "\n".join(read(path) for path in DOC_PATHS)
        assert_contains(combined_docs, code, "public docs error-code set")


def assert_boundary_docs() -> None:
    v1_text = read(V1_CONTRACT)
    client_text = read(CLIENT_EXAMPLES)
    readme_text = read(README)
    combined = "\n".join(read(path) for path in DOC_PATHS)

    required_phrases = [
        "stream: true",
        "501 not_implemented",
        "encoding_format: \"base64\"",
        "400 invalid_request",
        "external_proxy_not_implemented",
        "Embedding-only models are excluded",
        "Metadata-only GGUF packages are excluded",
        "PyTorch `.bin`",
        "No ONNX chat/LLM runtime yet",
        "no arbitrary SafeTensors/Hugging Face execution",
        "POST /v1/responses",
        "404 not_found",
        "GET /v1/chat/completions",
        "405 method_not_allowed",
        "not full OpenAI API parity",
    ]
    for phrase in required_phrases:
        assert_contains(combined, phrase, "public boundary docs")

    assert_contains(client_text, "Adopter checklist", "client examples")
    assert_contains(client_text, "/v1/models", "adopter checklist")
    assert_contains(client_text, "expected boundaries", "adopter checklist")
    assert_contains(readme_text, "public launch API contract", "README API section")
    assert_contains(readme_text, "scripts/public_api_contract_smoke.sh", "README public contract smoke")
    launch_text = read(LAUNCH_CHECKLIST)
    checklist_required_gates = [
        "git diff --check",
        "scripts/api_client_examples_regression.py",
        "scripts/api_client_examples_regression.py --self-test",
        "scripts/ci_static_policy.py --self-test",
        "scripts/public_api_contract_qa.py --self-test",
        "scripts/public_risk_scan.sh --self-test",
    ]
    for gate in checklist_required_gates:
        assert_contains(launch_text, gate, "launch checklist no-download gates")
    assert_launch_checklist_python_syntax_gate()
    assert_launch_checklist_client_example_syntax_gates()
    assert_launch_checklist_artifact_qa_run_gates()
    assert_launch_checklist_clean_install_gate()
    assert_contains(read(BACKEND_QUICKSTART), "scripts/public_api_contract_smoke.sh", "backend quickstart public contract smoke")
    assert_contains(read(CONTRIBUTING), "scripts/public_api_contract_smoke.sh", "contributing public contract smoke")
    assert_contains(readme_text, "docs/public-launch-checklist.md", "README launch checklist link")
    assert_contains(read(BACKEND_QUICKSTART), "../public-launch-checklist.md", "backend quickstart launch checklist link")
    assert_contains(read(CONTRIBUTING), "docs/public-launch-checklist.md", "contributing launch checklist link")
    assert_contributing_common_gates()
    assert_public_security_docs()
    assert_contains(
        launch_text,
        "tracked-file privacy patterns including macOS, Linux, and Windows home/profile paths",
        "launch checklist cross-platform home/profile path risk-scan scope",
    )
    assert_contains(v1_text, "public-contract.json", "v1 contract manifest link")
    assert_contains(launch_text, "api/public-contract.json", "launch checklist manifest link")
    assert_contains(launch_text, "api/v1-contract.md", "launch checklist v1 contract link")
    assert_contains(launch_text, "scripts/public_api_contract_smoke.sh", "launch checklist contract smoke")
    assert_contains(launch_text, "root `.gitattributes` text-normalization metadata", "launch checklist text-normalization metadata scope")
    assert_contains(
        launch_text,
        "tracked Git LFS pointer files that can hide external artifact downloads",
        "launch checklist Git LFS pointer risk-scan scope",
    )
    assert_contains(launch_text, "tracked local model/checkpoint artifacts", "launch checklist model/checkpoint artifact risk-scan scope")
    assert_contains(launch_text, "root `.gitignore` coverage for local model/checkpoint artifacts", "launch checklist model/checkpoint artifact ignore scope")
    assert_contains(launch_text, "tracked local Docker/container artifacts", "launch checklist Docker/container artifact risk-scan scope")
    assert_contains(launch_text, "root `.gitignore` coverage for local Docker/container artifacts", "launch checklist Docker/container artifact ignore scope")
    assert_contains(
        launch_text,
        "tracked local Terraform/OpenTofu state artifacts",
        "launch checklist infrastructure state artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local infrastructure state artifacts",
        "launch checklist infrastructure state artifact ignore scope",
    )
    assert_contains(launch_text, "tracked local mobile/Xcode/Android build artifacts", "launch checklist mobile build artifact risk-scan scope")
    assert_contains(launch_text, "root `.gitignore` coverage for local mobile/Xcode/Android build artifacts", "launch checklist mobile build artifact ignore scope")
    assert_contains(
        launch_text,
        "tracked local mobile/Xcode/Android signing/provisioning artifacts",
        "launch checklist mobile signing artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local mobile/Xcode/Android signing/provisioning artifacts",
        "launch checklist mobile signing artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked local screenshot/screen-recording artifacts",
        "launch checklist screenshot/screen-recording artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local screenshot/screen-recording artifacts",
        "launch checklist screenshot/screen-recording artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked local audio/video capture/export artifacts",
        "launch checklist audio/video capture artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local audio/video capture/export artifacts",
        "launch checklist audio/video capture artifact ignore scope",
    )
    assert_contains(launch_text, "root `.gitignore` coverage for local Python cache/build artifacts", "launch checklist Python artifact ignore scope")
    assert_contains(
        launch_text,
        "tracked Python virtualenv/dependency artifacts",
        "launch checklist Python virtualenv/dependency artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local Python virtualenv/dependency artifacts",
        "launch checklist Python virtualenv/dependency artifact ignore scope",
    )
    assert_contains(launch_text, "root `.gitignore` coverage for local frontend/Node cache/build artifacts", "launch checklist frontend artifact ignore scope")
    assert_contains(launch_text, "root `.gitignore` coverage for local Rust/Cargo cache/build artifacts", "launch checklist Rust/Cargo artifact ignore scope")
    assert_contains(launch_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "launch checklist public contract artifact env")
    assert_contains(launch_text, "scripts/backend_acceptance_artifact_qa.py", "launch checklist backend acceptance artifact QA")
    assert_contains(launch_text, "backend acceptance smoke success/failure summaries", "launch checklist backend acceptance artifact QA scope")
    assert_contains(launch_text, "backend acceptance artifact summaries reject local paths, secret markers, and request/payload text", "launch checklist backend acceptance artifact share-safety scope")
    assert_contains(
        launch_text,
        "optional backend acceptance smoke itself remains networked",
        "launch checklist backend acceptance artifact QA non-runtime caveat",
    )
    assert_contains(launch_text, "scripts/backend_acceptance_smoke.sh", "launch checklist optional acceptance smoke")
    assert_contains(launch_text, "public-launch-evidence.md", "launch checklist evidence link")
    assert_contains(launch_text, "What this launch does not prove", "launch checklist boundaries")

    evidence_text = read(LAUNCH_EVIDENCE)
    assert_contains(evidence_text, "a32505eadac6539865d224a8b4195656003a0032", "launch evidence commit")
    assert_contains(evidence_text, "e9195bc7462999284960f5631d3a74aa5391bffc", "launch evidence optional artifact QA CI commit")
    assert_latest_no_download_refusal_evidence(evidence_text)
    assert_latest_public_contract_qa_hardening_evidence(evidence_text)
    assert_frontend_lockfile_evidence(evidence_text)
    assert_contains(evidence_text, "repository text-normalization metadata guard", "launch evidence text-normalization metadata scope")
    assert_contains(evidence_text, "root `.gitattributes` text-normalization metadata", "launch evidence text-normalization metadata proof")
    assert_contains(evidence_text, "Git LFS pointer-file guard", "launch evidence Git LFS pointer risk-scan scope")
    assert_contains(
        evidence_text,
        "tracked Git LFS pointer files that can hide external artifact downloads",
        "launch evidence Git LFS pointer examples",
    )
    assert_contains(evidence_text, "scripts/public_contract_smoke_artifact_qa.py", "launch evidence artifact QA")
    assert_contains(evidence_text, "offline public-contract and backend acceptance artifact QA", "launch evidence backend acceptance artifact QA scope")
    assert_contains(
        evidence_text,
        "backend acceptance artifact summary share-safety guard rejects local paths, secret markers, and request/payload text",
        "launch evidence backend acceptance artifact share-safety scope",
    )
    assert_contains(evidence_text, "public-contract smoke Markdown/status/proof-scope row consistency", "launch evidence public smoke row QA scope")
    assert_contains(evidence_text, "manifest shape validation", "launch evidence manifest shape gate")
    assert_contains(evidence_text, "manifest-to-`/v1` docs boundary coverage", "launch evidence manifest docs boundary gate")
    assert_contains(
        evidence_text,
        "request hints for status/code refusal boundaries to be exposed in the refusal matrix",
        "launch evidence refusal request hint gate",
    )
    assert_contains(evidence_text, "public overclaim scanner self-test coverage", "launch evidence overclaim self-test scope")
    assert_contains(
        evidence_text,
        "required public issue-template field/privacy checkbox guard",
        "launch evidence issue-template required-field scope",
    )
    assert_contains(evidence_text, "synthetic refused/unsupported public overclaim examples", "launch evidence overclaim self-test proof")
    assert_contains(
        evidence_text,
        "production-readiness/legal-license public overclaim examples",
        "launch evidence production/legal overclaim self-test proof",
    )
    assert_contains(
        evidence_text,
        "workspace/personal agent context-file guard",
        "launch evidence workspace context risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked-file secret-token/private-key/cloud API-key and macOS/Linux/Windows home/profile path guards",
        "launch evidence cross-platform home/profile path risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "macOS, Linux, and Windows home/profile paths",
        "launch evidence cross-platform home/profile path examples",
    )
    assert_contains(
        evidence_text,
        "local assistant state such as `.codex/`, `.claude/`, `.continue/`, `.cursor/`, `.windsurf/`, and `.aider.*`",
        "launch evidence local assistant state risk-scan examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for personal workspace context including local assistant state",
        "launch checklist local assistant state ignore scope",
    )
    assert_contains(
        evidence_text,
        "local shell/REPL command history-file guard",
        "launch evidence command history risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local shell/REPL command history files such as `.bash_history`, `.zsh_history`, `.python_history`, `.node_repl_history`, `.psql_history`, `.sqlite_history`, `.mysql_history`, `.rediscli_history`, and `.fish_history`",
        "launch evidence command history examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local shell/REPL command history files",
        "launch evidence command history ignore examples",
    )
    assert_contains(
        launch_text,
        "tracked local shell/REPL command history files",
        "launch checklist command history risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local shell/REPL command history files",
        "launch checklist command history ignore scope",
    )
    assert_contains(
        evidence_text,
        "OS/platform metadata-file",
        "launch evidence OS/platform metadata risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked OS/platform metadata files such as `.DS_Store`, `Thumbs.db`, `ehthumbs.db`, `desktop.ini`, `$RECYCLE.BIN/`, `__MACOSX/`, `._*`, `.AppleDouble`, `.fseventsd/`, `.Spotlight-V100/`, `.TemporaryItems/`, `.Trashes/`, `.LSOverride`, and `.localized`",
        "launch evidence OS/platform metadata examples",
    )
    assert_contains(
        evidence_text,
        "editor backup/swap artifact guard",
        "launch evidence editor backup/swap risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "local editor history artifact guard",
        "launch evidence editor history risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "IDE workspace/config artifact guard",
        "launch evidence IDE workspace/config risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local OS/platform metadata files",
        "launch checklist OS/platform metadata ignore scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local editor backup/swap files",
        "launch checklist editor backup/swap ignore scope",
    )
    assert_contains(
        launch_text,
        "local editor history directories",
        "launch checklist editor history risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local IDE workspace/config artifacts",
        "launch checklist IDE workspace/config ignore scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local OS/platform metadata-file guard",
        "launch evidence OS/platform metadata ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local OS/platform metadata files",
        "launch evidence OS/platform metadata ignore examples",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local editor backup/swap artifact guard",
        "launch evidence editor backup/swap ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local editor backup/swap files",
        "launch evidence editor backup/swap ignore examples",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local IDE workspace/config artifact guard",
        "launch evidence IDE workspace/config ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local IDE workspace/config artifacts",
        "launch evidence IDE workspace/config ignore examples",
    )
    assert_contains(
        evidence_text,
        "credential/config filename guard including SSH private-key filenames, `.ssh/` directories, direnv config/state, and generic secret material paths",
        "launch evidence credential/config SSH risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local credential/config guard including SSH private-key filenames, `.ssh/`, direnv config/state, and generic secret material patterns",
        "launch evidence credential/config SSH ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked cloud SDK credential/config guard",
        "launch evidence cloud SDK credential risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local cloud SDK credential/config guard",
        "launch evidence cloud SDK credential ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked cloud SDK credential/config files such as `.aws/`, `.azure/`, `.config/gcloud/`, `.boto`, `boto.cfg`, `application_default_credentials.json`, `service-account.json`, `service_account.json`, and `serviceAccountKey.json`",
        "launch evidence cloud SDK credential examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local cloud SDK credential/config files",
        "launch evidence cloud SDK credential ignore examples",
    )
    assert_contains(
        evidence_text,
        "tracked Kubernetes credential/config guard",
        "launch evidence Kubernetes credential risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local Kubernetes credential/config guard",
        "launch evidence Kubernetes credential ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked Kubernetes credential/config files such as `.kube/`, `kubeconfig`, `kubeconfig.yaml`, and `kubeconfig.yml`",
        "launch evidence Kubernetes credential examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Kubernetes credential/config files",
        "launch evidence Kubernetes credential ignore examples",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local credential/config guard",
        "launch evidence credential/config ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked `.direnv/`, `.ssh/`, `secrets/`, and `private/` directories",
        "launch evidence credential/config SSH examples",
    )
    assert_contains(
        evidence_text,
        "tracked generic secret material files such as `secrets.json`, `secrets.yaml`, `secrets.yml`, `*.secret`, and `*.secrets`",
        "launch evidence generic secret material examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local credential/config files including `/.ssh/`, SSH private-key filenames, `/.direnv/`, `.envrc`, `/secrets/`, `/private/`, and generic secret material patterns",
        "launch evidence credential/config ignore examples",
    )
    assert_contains(
        launch_text,
        "tracked credential/config filenames including SSH private-key filenames, `.ssh/` directories, direnv config/state, generic secret material paths, and Java/Android/Apple signing key material",
        "launch checklist credential/config SSH risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local credential/config files including SSH private-key filenames, `.ssh/`, direnv config/state, generic secret material patterns, and Java/Android/Apple signing key material patterns",
        "launch checklist credential/config SSH ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked local cloud SDK credential/config files",
        "launch checklist cloud SDK credential risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local cloud SDK credential/config files",
        "launch checklist cloud SDK credential ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked Kubernetes credential/config files",
        "launch checklist Kubernetes credential risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local Kubernetes credential/config files",
        "launch checklist Kubernetes credential ignore scope",
    )
    assert_contains(
        evidence_text,
        "local runtime/artifact detail-file guard",
        "launch evidence runtime artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local runtime/artifact detail files",
        "launch checklist runtime artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local runtime/artifact detail-file guard",
        "launch evidence runtime artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local runtime/artifact detail files",
        "launch evidence runtime artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local log/trace/profiling/debug-output artifacts",
        "launch checklist diagnostic artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "local log/trace/profiling/debug-output artifact guard",
        "launch evidence diagnostic artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local log/trace/profiling/debug-output artifact guard",
        "launch evidence diagnostic artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local log/trace/profiling/debug-output artifacts such as top-level `logs/`, `traces/`, `profiles/`, `debug-output/`, `.trace`, `.cpuprofile`, `.heapsnapshot`, `.perf`, `.prof`, `.sarif`, `.sarif.json`, `.core`, `.crash`, `.dmp`, `.ips`, and root `core` files",
        "launch evidence diagnostic artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local log/trace/profiling/debug-output artifacts including `*.log`, SARIF/code-scanning reports, and crash dump patterns",
        "launch evidence diagnostic artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Python cache/build artifact guard",
        "launch evidence Python artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local Python cache/build artifact guard",
        "launch evidence Python artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Python cache/build artifacts",
        "launch evidence Python artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Python virtualenv/dependency artifact guard",
        "launch evidence Python virtualenv/dependency artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local Python virtualenv/dependency artifact guard",
        "launch evidence Python virtualenv/dependency artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked Python virtualenv/dependency artifacts such as `.venv/`, `venv/`, `env/`, `.tox/`, `.nox/`, `wheelhouse/`, `pip-wheel-metadata/`, and `site-packages/`",
        "launch evidence Python virtualenv/dependency artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Python virtualenv/dependency artifacts",
        "launch evidence Python virtualenv/dependency artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "frontend/Node cache/build artifact guard",
        "launch evidence frontend artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local frontend/Node cache/build artifact guard",
        "launch evidence frontend artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local frontend/Node cache/build artifacts",
        "launch evidence frontend artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Rust/Cargo cache/build artifact guard",
        "launch evidence Rust artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "Playwright/browser-test report directories such as `playwright-report/`, `blob-report/`, and `.playwright/`",
        "launch evidence Playwright/browser-test report examples",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local Rust/Cargo cache/build artifact guard",
        "launch evidence Rust artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Rust/Cargo cache/build artifacts",
        "launch evidence Rust artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local release/package artifacts",
        "launch checklist release/package artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local backup/dump artifacts",
        "launch checklist backup/dump artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local release/package artifact guard",
        "launch evidence release/package artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local release/package artifacts",
        "launch evidence release/package artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "backup/dump artifact guard",
        "launch evidence backup/dump artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root `.gitignore` local backup/dump artifact guard",
        "launch evidence backup/dump artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked backup/dump artifacts",
        "launch evidence backup/dump artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local backup/dump artifacts",
        "launch evidence backup/dump artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local notebook artifacts",
        "launch checklist notebook artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "local notebook checkpoint artifact guard",
        "launch evidence notebook checkpoint artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked notebook execution-output guard",
        "launch evidence notebook execution-output risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local notebook checkpoint artifacts such as `.ipynb_checkpoints/`",
        "launch evidence notebook checkpoint artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local notebook artifacts",
        "launch evidence notebook artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local model/checkpoint artifact guard",
        "launch evidence model/checkpoint artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local model/checkpoint artifacts such as top-level `models/`, `model-store/`, `weights/`, `checkpoints/`, `.safetensors`, `.gguf`, `.onnx`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.npz`, `.npy`, and `.tflite` files",
        "launch evidence model/checkpoint artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local model/checkpoint artifacts",
        "launch evidence model/checkpoint artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local Docker/container artifact guard",
        "launch evidence Docker/container artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Docker/container artifacts such as top-level `.docker/`, `docker-data/`, `docker-volumes/`, `docker-compose.override.yml`, `docker-compose.override.yaml`, `compose.override.yml`, and `compose.override.yaml`",
        "launch evidence Docker/container artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Docker/container artifacts",
        "launch evidence Docker/container artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local Terraform/OpenTofu infrastructure state artifact guard",
        "launch evidence infrastructure state artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Terraform/OpenTofu infrastructure state artifacts such as `.terraform/`, `.terraform.lock.hcl`, `.tfstate`, `.tfvars`, `.tfvars.json`, and `.tfplan` files",
        "launch evidence infrastructure state artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local infrastructure state artifacts",
        "launch evidence infrastructure state artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local mobile/Xcode/Android build artifact guard",
        "launch evidence mobile build artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local mobile/Xcode/Android build artifacts such as `DerivedData/`, `.gradle/`, `xcuserdata/`, `local.properties`, `.xcuserstate`, `.xcresult`, `.ipa`, `.apk`, `.aab`, and `.dSYM` files",
        "launch evidence mobile build artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local mobile/Xcode/Android build artifacts",
        "launch evidence mobile build artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "app-signing key stores (`*.p8`, `*.jks`, `*.keystore`)",
        "launch evidence mobile signing key artifact examples",
    )
    assert_contains(
        evidence_text,
        "mobile provisioning/build archives (`*.mobileprovision`, `*.provisionprofile`, `*.xcarchive`)",
        "launch evidence mobile provisioning artifact examples",
    )
    assert_contains(
        evidence_text,
        "matching root `.gitignore` coverage",
        "launch evidence mobile signing artifact ignore coverage",
    )
    assert_contains(
        evidence_text,
        "local screenshot/screen-recording artifact guard",
        "launch evidence screenshot/screen-recording artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local screenshot/screen-recording artifacts with default macOS capture prefixes",
        "launch evidence screenshot/screen-recording artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local screenshot/screen-recording artifacts",
        "launch evidence screenshot/screen-recording artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local audio/video capture/export artifact guard",
        "launch evidence audio/video capture artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local audio/video capture/export artifacts with common media extensions and default voice memo/audio recording prefixes",
        "launch evidence audio/video capture artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local audio/video capture/export artifacts",
        "launch evidence audio/video capture artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "symlink escape and target-resolution guards",
        "launch evidence symlink risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked symlinks that escape the repository or resolve only to missing/untracked local targets",
        "launch evidence symlink risk-scan examples",
    )
    assert_contains(
        launch_text,
        "tracked symlinks that escape the repository or resolve only to missing/untracked local targets",
        "launch checklist symlink risk-scan scope",
    )
    assert_contains(evidence_text, "offline MiniLM/SmolLM2/Qwen2.5 optional API acceptance artifact QA self-tests", "launch evidence optional artifact QA self-test scope")
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA and optional MiniLM, SmolLM2, and Qwen2.5 API acceptance artifact QA self-tests run offline",
        "launch evidence backend acceptance artifact QA non-runtime scope",
    )
    assert_contains(evidence_text, "they do not download models, start the backend, or add runtime proof", "launch evidence optional artifact QA non-runtime caveat")
    assert_contains(evidence_text, "What this evidence does not prove", "launch evidence caveats")
    assert_contains(evidence_text, "external_proxy_not_implemented", "launch evidence external placeholder refusal")
    assert_contains(evidence_text, "synthetic PyTorch `.bin` public-contract refusal smoke", "launch evidence PyTorch refusal scope")
    assert_contains(evidence_text, "without deserializing pickle bytes or faking inference", "launch evidence PyTorch refusal boundary")
    assert_contains(evidence_text, "unsupported ONNX chat/general public-contract refusal smoke", "launch evidence ONNX refusal scope")
    assert_contains(evidence_text, "without enabling ONNX Runtime, loading the graph, or faking inference", "launch evidence ONNX refusal boundary")
    assert_contains(evidence_text, "unverified SafeTensors/Hugging Face public-contract refusal smoke", "launch evidence SafeTensors/HF refusal scope")
    assert_contains(evidence_text, "without loading weights or faking inference", "launch evidence SafeTensors/HF refusal boundary")
    assert_contains(evidence_text, "metadata-only GGUF public-contract refusal smoke", "launch evidence GGUF refusal scope")
    assert_contains(evidence_text, "without making a GGUF runtime, tokenizer execution, or generation claim", "launch evidence GGUF refusal boundary")
    assert_contains(evidence_text, "unsupported `/v1` route/method public-contract refusal smoke", "launch evidence unsupported v1 routing scope")
    assert_contains(evidence_text, "standard JSON error envelopes for unsupported `/v1` routes and methods", "launch evidence unsupported v1 routing boundary")
    assert_contains(evidence_text, "malformed `/v1` JSON request body refusal smoke", "launch evidence malformed v1 JSON scope")
    assert_contains(
        evidence_text,
        "malformed JSON on `POST /v1/chat/completions` and `POST /v1/embeddings` return `400 invalid_request`",
        "launch evidence malformed v1 JSON boundary",
    )

    matrix_text = read(REFUSAL_MATRIX)
    matrix_required_phrases = (
        "Streamed chat-completion requests",
        "Base64 embeddings",
        "Missing chat model",
        "Unknown embedding model",
        "External placeholder chat or activation",
        "GGUF metadata-only chat attempts",
        "PyTorch `.bin` execution",
        "Unsupported ONNX chat or general ONNX model execution",
        "Unverified SafeTensors/Hugging Face model execution",
        "Full OpenAI API parity",
        "not a runtime expansion plan",
    )
    for phrase in matrix_required_phrases:
        assert_contains(matrix_text, phrase, "refusal boundary matrix")

    manifest = load_manifest()
    matrix_aliases = {
        "streaming chat completions": "Streamed chat-completion requests",
        "base64 embeddings": "Base64 embeddings",
        "missing chat model": "Missing chat model",
        "malformed /v1 JSON request body": "Malformed `/v1` JSON request body",
        "unknown embedding model": "Unknown embedding model",
        "external placeholder chat or activation": "External placeholder chat or activation",
        "embedding models in /v1/models": "Embedding models in `/v1/models`",
        "GGUF metadata-only chat attempts": "GGUF metadata-only chat attempts",
        "PyTorch .bin execution": "PyTorch `.bin` execution",
        "unsupported ONNX chat or general ONNX model execution": "Unsupported ONNX chat or general ONNX model execution",
        "unverified SafeTensors/Hugging Face model execution": "Unverified SafeTensors/Hugging Face model execution",
        "unsupported /v1 endpoint": "Unsupported `/v1` endpoint",
        "unsupported /v1 method": "Unsupported `/v1` method",
        "full OpenAI API parity": "Full OpenAI API parity",
    }
    assert_refusal_matrix_row_set(matrix_text, manifest, matrix_aliases)
    for boundary in manifest.get("expected_boundary_errors", []):
        name = boundary.get("boundary")
        expected = matrix_aliases.get(name)
        if expected is None:
            raise AssertionError(f"docs/api/public-contract.json boundary {name!r} is missing a refusal matrix alias")
        assert_contains(matrix_text, expected, "refusal boundary matrix manifest coverage")
        matrix_line = next((line for line in matrix_text.splitlines() if expected in line), "")
        if not matrix_line:
            raise AssertionError(f"refusal boundary matrix missing row for manifest boundary {name!r}")
        if "status" in boundary and "code" in boundary:
            expected_status_code = f"`{boundary['status']} {boundary['code']}`"
            assert_contains(matrix_line, expected_status_code, f"refusal boundary matrix row for {name!r}")
            request_hint = boundary.get("request_hint")
            assert_non_empty_string(request_hint, f"manifest boundary {name!r}.request_hint")
            assert_contains(matrix_line, request_hint, f"refusal boundary matrix request hint for {name!r}")
        elif name == "embedding models in /v1/models":
            assert_contains(matrix_line.lower(), "excluded", f"refusal boundary matrix row for {name!r}")
        elif name == "full OpenAI API parity":
            assert_contains(matrix_line.lower(), "not claimed", f"refusal boundary matrix row for {name!r}")

    v1_aliases = {
        "streaming chat completions": "Streaming chat completions.",
        "base64 embeddings": "Base64 embeddings are not supported.",
        "missing chat model": "`400 model_not_found`",
        "malformed /v1 JSON request body": "Malformed JSON bodies on `/v1/chat/completions` or `/v1/embeddings` return `400 invalid_request`",
        "unknown embedding model": "`404 embedding_model_not_found`",
        "external placeholder chat or activation": "`501 external_proxy_not_implemented`",
        "embedding models in /v1/models": "Embedding-only models are excluded from `/v1/models`",
        "GGUF metadata-only chat attempts": "No native GGUF chat/inference",
        "PyTorch .bin execution": "no PyTorch `.bin` execution",
        "unsupported ONNX chat or general ONNX model execution": "no ONNX chat/LLM generation",
        "unverified SafeTensors/Hugging Face model execution": "no arbitrary SafeTensors/Hugging Face execution",
        "unsupported /v1 endpoint": "`404 not_found`",
        "unsupported /v1 method": "`405 method_not_allowed`",
        "full OpenAI API parity": "not full OpenAI API parity",
    }
    for boundary in manifest.get("expected_boundary_errors", []):
        name = boundary.get("boundary")
        expected = v1_aliases.get(name)
        if expected is None:
            raise AssertionError(f"docs/api/public-contract.json boundary {name!r} is missing a /v1 docs alias")
        assert_contains(v1_text, expected, "/v1 contract manifest boundary coverage")

    assert_contains(v1_text, "refusal-boundary-matrix.md", "v1 contract refusal matrix link")
    assert_contains(launch_text, "refusal-boundary-matrix.md", "launch checklist refusal matrix link")

    quickstart_text = read(BACKEND_QUICKSTART)
    contributing_text = read(CONTRIBUTING)
    assert_contains(quickstart_text, "manifest-driven", "backend quickstart public contract smoke")
    assert_contains(quickstart_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "backend quickstart public contract artifact env")
    assert_contains(contributing_text, "public-contract.json", "contributing manifest-driven public contract smoke")
    assert_contains(contributing_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "contributing public contract artifact env")


def markdown_table_first_cells(markdown: str) -> list[str]:
    rows: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or cells[0] == "Boundary" or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells[0])
    return rows


def assert_refusal_matrix_row_set(
    matrix_text: str,
    manifest: dict[str, Any],
    matrix_aliases: dict[str, str],
) -> None:
    expected_rows: set[str] = set(ALLOWED_EXTRA_REFUSAL_MATRIX_ROWS)
    for boundary in manifest.get("expected_boundary_errors", []):
        name = boundary.get("boundary")
        if not isinstance(name, str):
            raise AssertionError(f"manifest boundary entry missing boundary name: {boundary!r}")
        expected = matrix_aliases.get(name)
        if expected is None:
            raise AssertionError(f"docs/api/public-contract.json boundary {name!r} is missing a refusal matrix alias")
        expected_rows.add(expected)

    rows = markdown_table_first_cells(matrix_text)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        if row in seen:
            duplicates.add(row)
        seen.add(row)
    if duplicates:
        raise AssertionError(f"refusal boundary matrix has duplicate boundary rows: {sorted(duplicates)}")

    unexpected = seen - expected_rows
    if unexpected:
        raise AssertionError(f"refusal boundary matrix has non-manifest boundary rows without allow-list coverage: {sorted(unexpected)}")

    missing = expected_rows - seen
    if missing:
        raise AssertionError(f"refusal boundary matrix is missing expected boundary rows: {sorted(missing)}")


def example_endpoints_from_text(text: str) -> set[str]:
    endpoints = set()
    for method in ("GET", "POST"):
        for match in re.finditer(rf"\b{method}\b[^\n\"']*(/(?:v1|api)/[A-Za-z0-9_./{{}}:-]+)", text):
            endpoint = match.group(1).replace("{{base}}", "").split("?", 1)[0]
            endpoint = endpoint.replace("$BASE", "").replace("base_url", "")
            endpoint = re.sub(r"/\$\{?EMBED_MODEL_ID\}?/embed", "/:id/embed", endpoint)
            if "/api/embedding-models" in endpoint:
                # Older embedding endpoint is documented by the backend quickstart, but the launch
                # OpenAI-style contract only covers /v1/embeddings.
                continue
            if endpoint.startswith("/v1/") or endpoint.startswith("/api/models/catalog"):
                endpoints.add(f"{method} {endpoint}")
    return endpoints


def assert_example_text_contract(text: str, allowed_endpoints: set[str], label: str) -> None:
    if re.search(r"['\"]?stream['\"]?\s*[:=]\s*(true|True)", text):
        raise AssertionError(f"{label} must not send streaming chat requests")
    if re.search(r"['\"]?encoding_format['\"]?\s*[:=]\s*['\"]base64['\"]", text):
        raise AssertionError(f"{label} must not request base64 embeddings")

    unexpected = {item for item in example_endpoints_from_text(text) if item not in allowed_endpoints}
    if unexpected:
        raise AssertionError(f"{label} uses endpoints outside public examples allow-list: {sorted(unexpected)}")


def assert_examples_static(manifest: dict[str, Any]) -> None:
    allowed_endpoints = allowed_example_endpoints(manifest)
    example_text = "\n".join(read(path) for path in EXAMPLE_PATHS)
    assert_contains(example_text, "/v1/health", "examples/api")
    assert_contains(example_text, "/v1/models", "examples/api")
    assert_contains(example_text, "/v1/chat/completions", "examples/api")
    assert_contains(example_text, "/v1/embeddings", "examples/api")
    assert_contains(example_text, "encoding_format", "examples/api embeddings")
    assert_example_text_contract(example_text, allowed_endpoints, "examples/api")

    for path in EXAMPLE_PATHS:
        assert_example_text_contract(read(path), allowed_endpoints, str(path.relative_to(ROOT)))


def positive_overclaim_failures(items: list[tuple[str, str]]) -> list[str]:
    failures = []
    for label_path, text in items:
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern, caveat in DANGEROUS_POSITIVE_PATTERNS:
                if pattern.search(line) and not caveat.search(line):
                    failures.append(f"{label_path}:{line_no}: {label}: {line.strip()}")
    return failures


def assert_no_positive_overclaims() -> None:
    items = [(str(path.relative_to(ROOT)), read(path)) for path in TEXT_PATHS]
    failures = positive_overclaim_failures(items)
    if failures:
        raise AssertionError(failures[0])


def run_self_test() -> None:
    repo_manifest = load_manifest()
    allowed_endpoints = allowed_example_endpoints(repo_manifest)
    good_example = """
GET {{base}}/v1/health
POST {{base}}/api/models/catalog/install
POST {{base}}/v1/chat/completions
{
  "stream": false,
  "encoding_format": "float"
}
"""
    assert_example_text_contract(good_example, allowed_endpoints, "synthetic good example")
    for text, expected in (
        ("POST {{base}}/v1/responses\n", "outside public examples allow-list"),
        ('POST {{base}}/v1/chat/completions\n{"stream": true}\n', "streaming chat requests"),
        ('POST {{base}}/v1/embeddings\n{"encoding_format": "base64"}\n', "base64 embeddings"),
    ):
        try:
            assert_example_text_contract(text, allowed_endpoints, "synthetic bad example")
        except AssertionError as exc:
            if expected not in str(exc):
                raise
        else:
            raise AssertionError("examples/api static self-test did not reject unsafe example drift")

    bad_lines = [
        ("docs/api/example.md", "Fathom supports streaming chat completions for clients."),
        ("docs/api/example.md", "The API proxies external provider requests."),
        ("README.md", "Fathom is a complete OpenAI compatible replacement."),
        ("README.md", "Fathom supports arbitrary SafeTensors runtime execution."),
        ("docs/api/example.md", "The embeddings endpoint supports base64 embeddings."),
        ("docs/public-launch-checklist.md", "Fathom is production-ready for deployment."),
        ("docs/public-launch-evidence.md", "Fathom license suitability is verified for public launch."),
    ]
    failures = positive_overclaim_failures(bad_lines)
    if len(failures) != len(bad_lines):
        raise AssertionError("public API contract QA self-test did not reject every unsafe overclaim example")

    allowed_lines = [
        ("docs/api/example.md", "Streaming chat completions are not supported."),
        ("docs/api/example.md", "External entries are metadata placeholders only and are not proxied."),
        ("README.md", "Fathom is not full OpenAI API parity."),
        ("README.md", "Arbitrary SafeTensors execution is refused."),
        ("docs/api/example.md", "Base64 embeddings are unsupported."),
        ("docs/public-launch-checklist.md", "Fathom does not prove production readiness."),
        ("docs/public-launch-evidence.md", "This evidence does not prove legal/license suitability."),
    ]
    allowed_failures = positive_overclaim_failures(allowed_lines)
    if allowed_failures:
        raise AssertionError(
            "public API contract QA self-test rejected allowed caveated examples:\n" + "\n".join(allowed_failures)
        )

    valid_issue_template = """
body:
  - type: input
    id: endpoint
    attributes:
      label: Endpoint
    validations:
      required: true
  - type: checkboxes
    id: privacy
    attributes:
      label: Privacy and artifact check
      options:
        - label: I removed credentials.
          required: true
        - label: I used synthetic prompts.
          required: true
"""
    assert_issue_template_required_fields(valid_issue_template, "synthetic issue template", ("endpoint",))
    assert_issue_template_required_checkbox_options(valid_issue_template, "synthetic issue template", "privacy")
    for text, expected in (
        (valid_issue_template.replace("    validations:\n      required: true\n", ""), "must remain required"),
        (
            valid_issue_template.replace("        - label: I used synthetic prompts.\n          required: true", "        - label: I used synthetic prompts."),
            "checkbox option must remain required",
        ),
    ):
        try:
            assert_issue_template_required_fields(text, "synthetic bad issue template", ("endpoint",))
            assert_issue_template_required_checkbox_options(text, "synthetic bad issue template", "privacy")
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("issue-template required-field self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("issue-template required-field self-test did not reject optional public intake fields")

    manifest = {
        "expected_boundary_errors": [
            {"boundary": "streaming chat completions", "status": 501, "code": "not_implemented"},
            {"boundary": "full OpenAI API parity", "expected_behavior": "not claimed"},
        ]
    }
    aliases = {
        "streaming chat completions": "Streamed chat-completion requests",
        "full OpenAI API parity": "Full OpenAI API parity",
    }
    allowed_matrix = """
| Boundary | Request hint | Expected behavior | Evidence |
| --- | --- | --- | --- |
| Streamed chat-completion requests | `stream: true` | `501 not_implemented`, no `choices` | public contract smoke |
| Full OpenAI API parity | n/a; non-claim boundary | not claimed | docs/static contract QA |
| Production readiness, performance, quality, legal/license suitability | n/a; non-claim boundary | not claimed | public launch checklist and evidence caveats |
| Real external provider proxying | external placeholder activation or chat model id | not implemented | public contract smoke and docs/static QA |
"""
    assert_refusal_matrix_row_set(allowed_matrix, manifest, aliases)

    unexpected_matrix = allowed_matrix.replace(
        "| Full OpenAI API parity |",
        "| Full OpenAI API parity |\n| Unreviewed runtime expansion |",
    )
    try:
        assert_refusal_matrix_row_set(unexpected_matrix, manifest, aliases)
    except AssertionError as exc:
        if "non-manifest boundary rows without allow-list coverage" not in str(exc):
            raise AssertionError("refusal matrix row-set self-test failed for the wrong reason") from exc
    else:
        raise AssertionError("refusal matrix row-set self-test did not reject an unexpected boundary row")

    required_python = {"scripts/public_api_contract_qa.py", "examples/api/python-no-deps.py"}
    valid_python_gate = "python3 -m py_compile scripts/public_api_contract_qa.py examples/api/python-no-deps.py\n"
    assert_python_syntax_paths(valid_python_gate, "synthetic Python syntax gate", required_python)
    for text, expected in (
        ("# scripts/public_api_contract_qa.py is mentioned only in prose\n", "missing a python3 -m py_compile command"),
        (
            valid_python_gate.replace("examples/api/python-no-deps.py", "scripts/untracked_helper.py"),
            "missing offline Python gate",
        ),
        (
            "python3 -m py_compile scripts/public_api_contract_qa.py examples/api/python-no-deps.py scripts/untracked_helper.py\n",
            "unexpected Python path",
        ),
    ):
        try:
            assert_python_syntax_paths(text, "synthetic bad Python syntax gate", required_python)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("Python syntax gate self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("Python syntax gate self-test did not reject command drift")

    required_shell = {"scripts/public_risk_scan.sh", "scripts/public_api_contract_smoke.sh"}
    valid_shell_gate = """
bash -n scripts/public_risk_scan.sh
bash -n scripts/public_api_contract_smoke.sh
"""
    assert_shell_syntax_paths(valid_shell_gate, "synthetic shell syntax gate", required_shell)
    for text, expected in (
        ("# bash -n scripts/public_risk_scan.sh\n", "missing offline shell gate"),
        (
            valid_shell_gate.replace("scripts/public_api_contract_smoke.sh", "scripts/deleted_smoke.sh"),
            "missing offline shell gate",
        ),
        (
            valid_shell_gate + "bash -n scripts/deleted_smoke.sh\n",
            "unexpected shell path",
        ),
    ):
        try:
            assert_shell_syntax_paths(text, "synthetic bad shell syntax gate", required_shell)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("shell syntax gate self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("shell syntax gate self-test did not reject command drift")

    valid_frontend_gates = """
npm --prefix frontend run build
npm --prefix frontend run qa:copy
"""
    assert_frontend_launch_gates(valid_frontend_gates, "synthetic frontend launch gate")
    for text, expected in (
        ("npm --prefix frontend run build\n", "qa:copy"),
        ("npm --prefix frontend run qa:copy\n", "run build"),
        (
            "npm --prefix frontend build\nnpm --prefix frontend run qa:copy\n",
            "run build",
        ),
    ):
        try:
            assert_frontend_launch_gates(text, "synthetic bad frontend launch gate")
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("frontend launch gate self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("frontend launch gate self-test did not reject command drift")

    assert_clean_install_gate("npm --prefix frontend ci\n", "synthetic clean install gate")
    for text, expected in (
        ("npm --prefix frontend install\n", "npm --prefix frontend ci"),
        ("npm --prefix frontend ci\nnpm --prefix frontend install\n", "npm ci"),
    ):
        try:
            assert_clean_install_gate(text, "synthetic bad clean install gate")
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("clean install gate self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("clean install gate self-test did not reject command drift")

    valid_gitattributes = """
* text=auto eol=lf
*.7z binary
*.aab binary
*.aac binary
*.apk binary
*.app binary
*.avi binary
*.bin binary
*.bz2 binary
*.ckpt binary
*.deb binary
*.dmg binary
*.dSYM binary
*.egg binary
*.flac binary
*.gif binary
*.gguf binary
*.gz binary
*.ico binary
*.ipa binary
*.jpeg binary
*.jpg binary
*.m4a binary
*.m4v binary
*.mkv binary
*.mobileprovision binary
*.mov binary
*.mp3 binary
*.mp4 binary
*.msi binary
*.npy binary
*.npz binary
*.onnx binary
*.pdf binary
*.pkg binary
*.png binary
*.provisionprofile binary
*.pt binary
*.pth binary
*.rar binary
*.rpm binary
*.safetensors binary
*.tar binary
*.tfplan binary
*.tgz binary
*.tflite binary
*.wav binary
*.webm binary
*.webp binary
*.whl binary
*.xcarchive binary
*.xcresult binary
*.xz binary
*.zip binary
*.zst binary
"""
    assert_gitattributes_text_normalization(valid_gitattributes, "synthetic .gitattributes")
    for text, expected in (
        (valid_gitattributes.replace("* text=auto eol=lf\n", ""), "missing text-normalization metadata"),
        (valid_gitattributes.replace("*.mp4 binary\n", ""), "missing text-normalization metadata"),
        (valid_gitattributes.replace("*.safetensors binary\n", ""), "missing text-normalization metadata"),
        ("* binary\n" + valid_gitattributes, "must not mark the whole repository as binary"),
        ("* -text\n" + valid_gitattributes, "must not disable text normalization"),
    ):
        try:
            assert_gitattributes_text_normalization(text, "synthetic bad .gitattributes")
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError(".gitattributes self-test failed for the wrong reason") from exc
        else:
            raise AssertionError(".gitattributes self-test did not reject unsafe text-normalization drift")


def assert_smoke_manifest_wiring() -> None:
    smoke_text = read(SMOKE)
    assert_contains(smoke_text, "docs/api/public-contract.json", "public contract smoke manifest load")
    assert_contains(smoke_text, "supported_endpoints", "public contract smoke endpoint coverage")
    assert_contains(smoke_text, "expected_boundary_errors", "public contract smoke boundary coverage")
    assert_contains(smoke_text, "expected_behavior_no_download_boundaries", "public contract smoke manifest-derived boundary coverage")
    assert_contains(smoke_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "public contract smoke artifact env")
    assert_contains(smoke_text, "fathom.public_contract_smoke.summary.v1", "public contract smoke artifact schema")
    assert_contains(smoke_text, "partial diagnostic evidence", "public contract smoke failed artifact caveat")


def assert_optional_acceptance_docs() -> None:
    required_boundaries = [
        "Do not add this flow to default CI",
        "downloads or reuses",
        "It is intentionally not part of default CI",
        "With no arguments, the artifact QA runs a dependency-free synthetic self-test",
        "Run `bash scripts/public_risk_scan.sh`",
        "external provider proxying",
        "arbitrary SafeTensors/HF execution",
        "streaming",
        "full OpenAI API parity",
    ]
    for path, opt_in_env, smoke_script, artifact_qa_script, evidence_scope in OPTIONAL_ACCEPTANCE_DOCS:
        text = read(path)
        label = str(path.relative_to(ROOT))
        for phrase in required_boundaries:
            assert_contains(text, phrase, label)
        assert_contains(text, opt_in_env, label)
        assert_contains(text, smoke_script, label)
        assert_contains(text, artifact_qa_script, label)
        assert_contains(text, evidence_scope, label)


def assert_ci_wiring(manifest: dict[str, Any]) -> None:
    ci_text = read(CI)
    expected = manifest["ci_policy"]["offline_static_gate"]
    assert_frontend_launch_gates(ci_text, "CI frontend launch gate")
    assert_contains(ci_text, "python3 -m py_compile", "CI Python syntax step")
    assert_python_syntax_paths(
        ci_text,
        "CI",
        set(OFFLINE_QA_PYTHON_PATHS) | set(OFFLINE_CLIENT_EXAMPLE_PYTHON_PATHS),
    )
    assert_shell_syntax_paths(ci_text, "CI", set(OFFLINE_SHELL_SYNTAX_PATHS))
    assert_contains(ci_text, "scripts/public_api_contract_qa.py", "CI public API contract QA wiring")
    assert_contains(ci_text, "scripts/public_contract_smoke_artifact_qa.py", "CI public contract smoke artifact QA wiring")
    assert_contains(ci_text, "scripts/backend_acceptance_artifact_qa.py", "CI backend acceptance artifact QA wiring")
    assert_contains(ci_text, "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py", "CI MiniLM optional artifact QA wiring")
    assert_contains(ci_text, "scripts/smollm2_optional_api_acceptance_artifact_qa.py", "CI SmolLM2 optional artifact QA wiring")
    assert_contains(ci_text, "scripts/qwen25_optional_api_acceptance_artifact_qa.py", "CI Qwen2.5 optional artifact QA wiring")
    assert_contains(ci_text, "git diff --check", "CI whitespace gate")
    assert_contains(ci_text, expected, "CI public API contract QA run step")
    assert_contains(
        ci_text,
        "python3 scripts/api_client_examples_regression.py --self-test",
        "CI API client example regression self-test run step",
    )
    assert_contains(ci_text, "python3 scripts/public_api_contract_qa.py --self-test", "CI public API contract QA self-test run step")
    assert_contains(ci_text, "python3 scripts/public_contract_smoke_artifact_qa.py", "CI public contract smoke artifact QA run step")
    assert_contains(ci_text, "python3 scripts/backend_acceptance_artifact_qa.py", "CI backend acceptance artifact QA run step")
    assert_contains(ci_text, "bash scripts/public_risk_scan.sh --self-test", "CI public risk scan self-test run step")
    assert_contains(ci_text, "bash scripts/public_risk_scan.sh", "CI public risk scan run step")
    assert_contains(
        ci_text,
        "python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
        "CI MiniLM optional artifact QA self-test run step",
    )
    assert_contains(
        ci_text,
        "python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py",
        "CI SmolLM2 optional artifact QA self-test run step",
    )
    assert_contains(
        ci_text,
        "python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py",
        "CI Qwen2.5 optional artifact QA self-test run step",
    )
    assert_contains(ci_text, "bash scripts/public_api_contract_smoke.sh", "CI public API contract smoke run step")
    if re.search(r"cargo\s+test\b[^\n]*--features\s+[^\n]*onnx-embeddings-ort", ci_text):
        raise AssertionError("default CI must not enable onnx-embeddings-ort")
    for line_no, line in enumerate(ci_text.splitlines(), start=1):
        if "scripts/backend_acceptance_smoke.sh" in line and line.strip() != "bash -n scripts/backend_acceptance_smoke.sh":
            raise AssertionError(f"default CI must only syntax-check backend acceptance smoke, line {line_no}")


def assert_api_contract_issue_template(manifest: dict[str, Any]) -> None:
    template_text = read(API_CONTRACT_ISSUE_TEMPLATE)
    label = ".github/ISSUE_TEMPLATE/api_contract.yml"
    required_phrases = [
        "docs/api/v1-contract.md",
        "docs/api/public-contract.json",
        "docs/api/refusal-boundary-matrix.md",
        "not full OpenAI API parity",
        "synthetic prompts only",
        "credentials",
        "auth headers",
        "local paths",
        "hostnames",
        "logs/artifacts",
        "model-store details",
    ]
    for phrase in required_phrases:
        assert_contains(template_text, phrase, label)

    assert_issue_template_required_fields(template_text, label, ("endpoint", "request", "response", "client", "notes"))
    assert_issue_template_required_checkbox_options(template_text, label, "privacy")

    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        assert_non_empty_string(method, f"manifest endpoint method for {label}")
        assert_non_empty_string(path, f"manifest endpoint path for {label}")
        assert_contains(template_text, f"{method} {path}", f"{label} endpoint placeholder")


def assert_model_runtime_issue_template() -> None:
    template_text = read(MODEL_RUNTIME_ISSUE_TEMPLATE)
    label = ".github/ISSUE_TEMPLATE/model_runtime_request.yml"
    required_phrases = [
        "does not imply broad GGUF runtime/tokenizer/generation support",
        "ONNX",
        "chat or general",
        "general ONNX support",
        "PyTorch `.bin` loading",
        "arbitrary SafeTensors execution",
        "streaming or full OpenAI API parity",
        "Do not include private prompts",
        "credentials",
        "local paths",
        "hostnames",
        "raw logs/artifacts",
        "model-store details",
        "Use public URLs or share-safe identifiers only",
        "Privacy and artifact check",
        "I removed private prompts, documents, and credentials.",
        "I removed absolute local paths, hostnames, usernames, and model-store details.",
        "I reviewed attached logs/artifacts/screenshots for sensitive data and ran a risk scan when applicable.",
    ]
    for phrase in required_phrases:
        assert_contains(template_text, phrase, label)
    assert_issue_template_required_fields(template_text, label, ("format", "family", "task", "layout", "license", "provenance"))
    assert_issue_template_required_checkbox_options(template_text, label, "privacy")


def assert_bug_report_issue_template() -> None:
    template_text = read(BUG_REPORT_ISSUE_TEMPLATE)
    label = ".github/ISSUE_TEMPLATE/bug_report.yml"
    required_phrases = [
        "synthetic prompts",
        "share-safe fixtures",
        "private prompts",
        "credentials",
        "auth headers",
        "private documents",
        "local paths",
        "hostnames",
        "usernames",
        "model-store details",
        "raw logs/artifacts",
        "bash scripts/public_risk_scan.sh",
        "Privacy and artifact check",
        "I removed private prompts, documents, and credentials.",
        "I removed absolute local paths, hostnames, usernames, and model-store details.",
        "I reviewed attached logs/artifacts/screenshots for sensitive data and ran a risk scan when applicable.",
        "I used synthetic prompts/share-safe fixtures",
    ]
    for phrase in required_phrases:
        assert_contains(template_text, phrase, label)
    assert_issue_template_required_fields(template_text, label, ("os", "version", "command", "area", "expected", "actual", "repro"))
    assert_issue_template_required_checkbox_options(template_text, label, "privacy")


def assert_security_privacy_issue_template() -> None:
    template_text = read(SECURITY_PRIVACY_ISSUE_TEMPLATE)
    label = ".github/ISSUE_TEMPLATE/security_or_privacy.yml"
    required_phrases = [
        "SECURITY.md",
        "private channel",
        "Do not include exploit details",
        "reproduction steps",
        "private prompts",
        "credentials",
        "sensitive local paths",
        "hostnames",
        "private documents",
        "sensitive artifact details",
        "model-store details",
        "public-safe summary",
        "Public disclosure check",
        "I did not include exploit steps, private prompts, credentials, local paths, hostnames, or sensitive artifact details.",
        "I read SECURITY.md and understand sensitive details should wait for a private channel.",
        "I included only a public-safe summary",
    ]
    for phrase in required_phrases:
        assert_contains(template_text, phrase, label)
    assert_issue_template_required_fields(template_text, label, ("category", "area", "public_summary"))
    assert_issue_template_required_checkbox_options(template_text, label, "safety")


def assert_issue_template_config() -> None:
    config_text = read(ISSUE_TEMPLATE_CONFIG)
    label = ".github/ISSUE_TEMPLATE/config.yml"
    blank_issue_settings = re.findall(r"(?m)^\s*blank_issues_enabled\s*:\s*(true|false)\s*(?:#.*)?$", config_text)
    if blank_issue_settings != ["false"]:
        raise AssertionError(f"{label} must set exactly one `blank_issues_enabled: false` entry")
    if re.search(r"(?m)^\s*blank_issues_enabled\s*:\s*true\s*(?:#.*)?$", config_text):
        raise AssertionError(f"{label} must not enable blank public issues")


def assert_pull_request_template() -> None:
    template_text = read(PR_TEMPLATE)
    label = ".github/pull_request_template.md"
    required_phrases = [
        "does not fake inference, embeddings, installs, downloads, readiness, benchmark results, or runtime availability",
        "format, family, task, feature flag, fixture, endpoint, and known blockers",
        " ".join(("broad GGUF", "runtime/tokenizer/generation support")),
        " ".join(("ONNX", "chat/general ONNX support")),
        "PyTorch `.bin` loading",
        "arbitrary SafeTensors execution",
        "streaming/full OpenAI parity",
        "GPU support",
        "batching",
        "performance claims",
        "private prompts",
        "credentials",
        "usernames",
        "hostnames",
        "absolute local paths",
        "model-store details",
        "synthetic/share-safe prompts",
        "bash scripts/public_risk_scan.sh",
        "not a complete privacy audit",
        "README, CONTRIBUTING, SECURITY, API docs, UI copy, and tests",
        "runnable, planned, blocked, metadata-only, and unavailable states",
        "git diff --check",
        "python3 -m py_compile",
        "bash -n",
        "python3 scripts/api_client_examples_regression.py",
        "python3 scripts/api_client_examples_regression.py --self-test",
        "python3 scripts/public_api_contract_qa.py",
        "python3 scripts/public_api_contract_qa.py --self-test",
        "python3 scripts/public_contract_smoke_artifact_qa.py",
        "python3 scripts/backend_acceptance_artifact_qa.py",
        "python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
        "python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py",
        "python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py",
        "python3 scripts/ci_static_policy.py",
        "python3 scripts/ci_static_policy.py --self-test",
        "cargo fmt --all --check",
        "cargo test -q",
        "npm --prefix frontend run build",
        "npm --prefix frontend run qa:copy",
        "bash scripts/public_risk_scan.sh --self-test",
        "FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh",
        "cargo test -q --features onnx-embeddings-ort",
        "exact evidence and caveats",
    ]
    for phrase in required_phrases:
        assert_contains(template_text, phrase, label)


def main() -> int:
    parser = argparse.ArgumentParser(description="Static QA for Fathom's public launch API contract")
    parser.add_argument("--self-test", action="store_true", help="run synthetic public-overclaim scanner regression checks")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        print("public API contract QA self-test passed")
        return 0

    manifest = load_manifest()
    assert_manifest_shape(manifest)
    assert_gitattributes_text_normalization()
    assert_endpoint_docs(manifest)
    assert_roadmap_last_updated_freshness()
    assert_license_metadata()
    assert_tracked_python_syntax_coverage()
    assert_tracked_shell_syntax_coverage()
    assert_launch_checklist_clean_install_gate()
    assert_boundary_docs()
    assert_examples_static(manifest)
    assert_launch_checklist_frontend_gates()
    assert_no_positive_overclaims()
    assert_smoke_manifest_wiring()
    assert_optional_acceptance_docs()
    assert_ci_wiring(manifest)
    assert_api_contract_issue_template(manifest)
    assert_model_runtime_issue_template()
    assert_bug_report_issue_template()
    assert_security_privacy_issue_template()
    assert_issue_template_config()
    assert_pull_request_template()
    print("public API contract QA passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"public API contract QA failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
