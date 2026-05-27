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
CI = ROOT / ".github" / "workflows" / "ci.yml"
API_CONTRACT_ISSUE_TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "api_contract.yml"
SMOKE = ROOT / "scripts" / "public_api_contract_smoke.sh"
EXAMPLES_DIR = ROOT / "examples" / "api"

DOC_PATHS = [V1_CONTRACT, CLIENT_EXAMPLES, BACKEND_QUICKSTART, LAUNCH_CHECKLIST, LAUNCH_EVIDENCE, REFUSAL_MATRIX, README]
OPTIONAL_DOC_PATHS = [MINILM_OPTIONAL_ACCEPTANCE, SMOLLM2_OPTIONAL_ACCEPTANCE, QWEN25_OPTIONAL_ACCEPTANCE]
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*"))
TEXT_PATHS = DOC_PATHS + OPTIONAL_DOC_PATHS + EXAMPLE_PATHS + [CI, API_CONTRACT_ISSUE_TEMPLATE]
OFFLINE_QA_PYTHON_PATHS = (
    "scripts/api_client_examples_regression.py",
    "scripts/backend_acceptance_artifact_qa.py",
    "scripts/ci_static_policy.py",
    "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
    "scripts/public_api_contract_qa.py",
    "scripts/public_contract_smoke_artifact_qa.py",
    "scripts/qwen25_optional_api_acceptance_artifact_qa.py",
    "scripts/smollm2_optional_api_acceptance_artifact_qa.py",
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
PUBLIC_CONTRACT_QA_HARDENING_SUBJECT_PATTERN = r"^(Harden public .+ QA|Expose refusal request hints in matrix)$"

REQUIRED_ERROR_CODES = {
    "invalid_request",
    "model_not_found",
    "embedding_model_not_found",
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


def assert_launch_checklist_python_syntax_gate() -> None:
    paths = py_compile_command_paths(read(LAUNCH_CHECKLIST), "launch checklist")
    missing = sorted(set(OFFLINE_QA_PYTHON_PATHS) - paths)
    if missing:
        raise AssertionError(f"launch checklist py_compile gate is missing offline QA helper(s): {missing}")


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
                "scripts/public_api_contract_qa.py",
                "scripts/public_contract_smoke_artifact_qa.py",
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
        "scripts/ci_static_policy.py --self-test",
        "scripts/public_api_contract_qa.py --self-test",
        "scripts/public_risk_scan.sh --self-test",
    ]
    for gate in checklist_required_gates:
        assert_contains(launch_text, gate, "launch checklist no-download gates")
    assert_launch_checklist_python_syntax_gate()
    assert_contains(read(BACKEND_QUICKSTART), "scripts/public_api_contract_smoke.sh", "backend quickstart public contract smoke")
    assert_contains(read(CONTRIBUTING), "scripts/public_api_contract_smoke.sh", "contributing public contract smoke")
    assert_contains(readme_text, "docs/public-launch-checklist.md", "README launch checklist link")
    assert_contains(read(BACKEND_QUICKSTART), "../public-launch-checklist.md", "backend quickstart launch checklist link")
    assert_contains(read(CONTRIBUTING), "docs/public-launch-checklist.md", "contributing launch checklist link")
    assert_contains(v1_text, "public-contract.json", "v1 contract manifest link")
    assert_contains(launch_text, "api/public-contract.json", "launch checklist manifest link")
    assert_contains(launch_text, "api/v1-contract.md", "launch checklist v1 contract link")
    assert_contains(launch_text, "scripts/public_api_contract_smoke.sh", "launch checklist contract smoke")
    assert_contains(launch_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "launch checklist public contract artifact env")
    assert_contains(launch_text, "scripts/backend_acceptance_artifact_qa.py", "launch checklist backend acceptance artifact QA")
    assert_contains(launch_text, "backend acceptance smoke success/failure summaries", "launch checklist backend acceptance artifact QA scope")
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
    assert_contains(evidence_text, "687aaebc27fdaa00588dd889d9ae3226f5b26000", "launch evidence latest no-download refusal commit")
    assert_contains(evidence_text, "e9195bc7462999284960f5631d3a74aa5391bffc", "launch evidence optional artifact QA CI commit")
    assert_latest_public_contract_qa_hardening_evidence(evidence_text)
    assert_contains(evidence_text, "scripts/public_contract_smoke_artifact_qa.py", "launch evidence artifact QA")
    assert_contains(evidence_text, "offline public-contract and backend acceptance artifact QA", "launch evidence backend acceptance artifact QA scope")
    assert_contains(evidence_text, "manifest shape validation", "launch evidence manifest shape gate")
    assert_contains(evidence_text, "manifest-to-`/v1` docs boundary coverage", "launch evidence manifest docs boundary gate")
    assert_contains(
        evidence_text,
        "request hints for status/code refusal boundaries to be exposed in the refusal matrix",
        "launch evidence refusal request hint gate",
    )
    assert_contains(evidence_text, "public overclaim scanner self-test coverage", "launch evidence overclaim self-test scope")
    assert_contains(evidence_text, "synthetic refused/unsupported public overclaim examples", "launch evidence overclaim self-test proof")
    assert_contains(
        evidence_text,
        "production-readiness/legal-license public overclaim examples",
        "launch evidence production/legal overclaim self-test proof",
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
        "unknown embedding model": "Unknown embedding model",
        "external placeholder chat or activation": "External placeholder chat or activation",
        "embedding models in /v1/models": "Embedding models in `/v1/models`",
        "GGUF metadata-only chat attempts": "GGUF metadata-only chat attempts",
        "PyTorch .bin execution": "PyTorch `.bin` execution",
        "unsupported ONNX chat or general ONNX model execution": "Unsupported ONNX chat or general ONNX model execution",
        "unverified SafeTensors/Hugging Face model execution": "Unverified SafeTensors/Hugging Face model execution",
        "full OpenAI API parity": "Full OpenAI API parity",
    }
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
        "unknown embedding model": "`404 embedding_model_not_found`",
        "external placeholder chat or activation": "`501 external_proxy_not_implemented`",
        "embedding models in /v1/models": "Embedding-only models are excluded from `/v1/models`",
        "GGUF metadata-only chat attempts": "No native GGUF chat/inference",
        "PyTorch .bin execution": "no PyTorch `.bin` execution",
        "unsupported ONNX chat or general ONNX model execution": "no ONNX chat/LLM generation",
        "unverified SafeTensors/Hugging Face model execution": "no arbitrary SafeTensors/Hugging Face execution",
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


def assert_examples_static(manifest: dict[str, Any]) -> None:
    allowed_endpoints = allowed_example_endpoints(manifest)
    example_text = "\n".join(read(path) for path in EXAMPLE_PATHS)
    assert_contains(example_text, "/v1/health", "examples/api")
    assert_contains(example_text, "/v1/models", "examples/api")
    assert_contains(example_text, "/v1/chat/completions", "examples/api")
    assert_contains(example_text, "/v1/embeddings", "examples/api")
    assert_contains(example_text, "encoding_format", "examples/api embeddings")
    if re.search(r"stream\s*[:=]\s*(true|True)", example_text):
        raise AssertionError("examples/api must not send streaming chat requests")
    if re.search(r"encoding_format\s*[:=]\s*['\"]base64['\"]", example_text):
        raise AssertionError("examples/api must not request base64 embeddings")

    for path in EXAMPLE_PATHS:
        text = read(path)
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
        unexpected = {item for item in endpoints if item not in allowed_endpoints}
        if unexpected:
            raise AssertionError(f"{path.relative_to(ROOT)} uses endpoints outside public examples allow-list: {sorted(unexpected)}")


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


def assert_smoke_manifest_wiring() -> None:
    smoke_text = read(SMOKE)
    assert_contains(smoke_text, "docs/api/public-contract.json", "public contract smoke manifest load")
    assert_contains(smoke_text, "supported_endpoints", "public contract smoke endpoint coverage")
    assert_contains(smoke_text, "expected_boundary_errors", "public contract smoke boundary coverage")
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
    assert_contains(ci_text, "python3 -m py_compile", "CI Python syntax step")
    for path in OFFLINE_QA_PYTHON_PATHS:
        assert_contains(ci_text, path, "CI Python offline QA syntax step")
    assert_contains(ci_text, "scripts/public_api_contract_qa.py", "CI public API contract QA wiring")
    assert_contains(ci_text, "scripts/public_contract_smoke_artifact_qa.py", "CI public contract smoke artifact QA wiring")
    assert_contains(ci_text, "scripts/backend_acceptance_artifact_qa.py", "CI backend acceptance artifact QA wiring")
    assert_contains(ci_text, "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py", "CI MiniLM optional artifact QA wiring")
    assert_contains(ci_text, "scripts/smollm2_optional_api_acceptance_artifact_qa.py", "CI SmolLM2 optional artifact QA wiring")
    assert_contains(ci_text, "scripts/qwen25_optional_api_acceptance_artifact_qa.py", "CI Qwen2.5 optional artifact QA wiring")
    assert_contains(ci_text, expected, "CI public API contract QA run step")
    assert_contains(ci_text, "python3 scripts/public_api_contract_qa.py --self-test", "CI public API contract QA self-test run step")
    assert_contains(ci_text, "python3 scripts/public_contract_smoke_artifact_qa.py", "CI public contract smoke artifact QA run step")
    assert_contains(ci_text, "python3 scripts/backend_acceptance_artifact_qa.py", "CI backend acceptance artifact QA run step")
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
    assert_contains(ci_text, "bash -n scripts/public_api_contract_smoke.sh", "CI public API contract smoke syntax step")
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
    ]
    for phrase in required_phrases:
        assert_contains(template_text, phrase, label)

    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        assert_non_empty_string(method, f"manifest endpoint method for {label}")
        assert_non_empty_string(path, f"manifest endpoint path for {label}")
        assert_contains(template_text, f"{method} {path}", f"{label} endpoint placeholder")


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
    assert_endpoint_docs(manifest)
    assert_roadmap_last_updated_freshness()
    assert_boundary_docs()
    assert_examples_static(manifest)
    assert_no_positive_overclaims()
    assert_smoke_manifest_wiring()
    assert_optional_acceptance_docs()
    assert_ci_wiring(manifest)
    assert_api_contract_issue_template(manifest)
    print("public API contract QA passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"public API contract QA failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
