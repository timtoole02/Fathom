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

from ci_static_policy import DEFAULT_CI_GATE_COMMANDS, assert_default_ci_gate_inventory

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
SERVER_MAIN = ROOT / "crates" / "fathom-server" / "src" / "main.rs"
FRONTEND_PACKAGE = ROOT / "frontend" / "package.json"
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
PUBLIC_BASE_URL_PATHS = (
    V1_CONTRACT,
    CLIENT_EXAMPLES,
    BACKEND_QUICKSTART,
    LAUNCH_CHECKLIST,
    README,
    *EXAMPLE_PATHS,
)
TEXT_PATHS = DOC_PATHS + OPTIONAL_DOC_PATHS + EXAMPLE_PATHS + [
    CI,
    API_CONTRACT_ISSUE_TEMPLATE,
    MODEL_RUNTIME_ISSUE_TEMPLATE,
    BUG_REPORT_ISSUE_TEMPLATE,
    SECURITY_PRIVACY_ISSUE_TEMPLATE,
    ISSUE_TEMPLATE_CONFIG,
    PR_TEMPLATE,
]
PUBLIC_DOC_LOCAL_LINK_PATHS = (
    README,
    CONTRIBUTING,
    SECURITY,
    PR_TEMPLATE,
    LAUNCH_CHECKLIST,
    LAUNCH_EVIDENCE,
    *sorted((ROOT / "docs" / "api").glob("*.md")),
)
PR_TEMPLATE_REQUIRED_CHECKBOXES = (
    "This PR does not fake inference, embeddings, installs, downloads, readiness, benchmark results, or runtime availability.",
    "Any new/changed support is described narrowly by format, family, task, feature flag, fixture, endpoint, and known blockers.",
    "This PR does not imply broad GGUF runtime/tokenizer/generation support, ONNX chat/general ONNX support, PyTorch `.bin` loading, arbitrary SafeTensors execution, streaming/full OpenAI parity, GPU support, batching, or performance claims unless implemented and verified here.",
    "Public text, logs, screenshots, fixtures, and generated artifacts were reviewed for private prompts, credentials, usernames, hostnames, absolute local paths, and model-store details.",
    "Attached evidence uses synthetic/share-safe prompts and minimal local context.",
    "`bash scripts/public_risk_scan.sh` was run when public-facing text or artifacts changed; this is a guardrail, not a complete privacy audit.",
    "README, CONTRIBUTING, SECURITY, API docs, UI copy, and tests tell the same capability story where relevant.",
    "`/v1` behavior changes are reflected in `docs/api/v1-contract.md` when relevant.",
    "User-facing copy distinguishes runnable, planned, blocked, metadata-only, and unavailable states truthfully.",
    "`git diff --check`",
    "`npm --prefix frontend ci`",
    "Python syntax gate from `docs/public-launch-checklist.md` was run: `python3 -m py_compile ...`",
    "Shell syntax gate from `docs/public-launch-checklist.md` was run: `bash -n ...`",
    "`bash scripts/public_api_contract_smoke.sh`",
    "`python3 scripts/public_api_contract_qa.py`",
    "`python3 scripts/public_api_contract_qa.py --self-test`",
    "`python3 scripts/ci_static_policy.py`",
    "`python3 scripts/ci_static_policy.py --self-test`",
    "`bash scripts/public_risk_scan.sh --self-test`",
    "`bash scripts/public_risk_scan.sh`",
    "Not applicable / explained below",
    "For backend/API changes, I considered the optional networked acceptance smoke. It remains non-default CI and should be run only as a targeted manual check when appropriate: `FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh`.",
    "For ONNX embedding changes, I considered the targeted/manual feature gate: `cargo test -q --features onnx-embeddings-ort`.",
    "Release-facing claims include exact evidence and caveats: commit, feature flags, model or fixture, request shape, warm/cold state, and what the evidence does not prove.",
)
ALLOWED_NON_CONTRACT_EXAMPLE_SURFACES = {
    "GET /api/models/catalog",
    "POST /api/models/catalog/install",
}
EXPECTED_MANIFEST_METADATA = {
    "name": "Fathom public launch API contract",
    "status": "launch-supported-narrow-local-api",
    "base_url": "http://127.0.0.1:8180",
    "scope_note": (
        "Machine-checkable launch contract for the public docs and examples. "
        "This is intentionally smaller than OpenAPI and does not imply full OpenAI API parity."
    ),
}
EXPECTED_SUPPORTED_ENDPOINTS = [
    {
        "method": "GET",
        "path": "/v1/health",
        "purpose": "local API health and chat-generation readiness",
    },
    {
        "method": "GET",
        "path": "/v1/models",
        "purpose": "runnable local chat/generation models only",
    },
    {
        "method": "POST",
        "path": "/v1/chat/completions",
        "purpose": "non-streaming local chat completion for verified runnable SafeTensors/Hugging Face chat lanes",
        "required_boundary": "stream must be omitted or false",
    },
    {
        "method": "POST",
        "path": "/v1/embeddings",
        "purpose": "narrow OpenAI-style float embeddings for verified local MiniLM embedding runtimes",
        "required_boundary": "encoding_format must be omitted or float",
    },
]
EXPECTED_BOUNDARY_ERRORS = [
    {
        "boundary": "streaming chat completions",
        "request_hint": "stream: true",
        "status": 501,
        "code": "not_implemented",
    },
    {
        "boundary": "base64 embeddings",
        "request_hint": "encoding_format: base64",
        "status": 400,
        "code": "invalid_request",
    },
    {
        "boundary": "missing chat model",
        "request_hint": "unknown local chat model id",
        "status": 400,
        "code": "model_not_found",
    },
    {
        "boundary": "malformed /v1 JSON request body",
        "request_hint": "malformed JSON body on /v1/chat/completions or /v1/embeddings",
        "status": 400,
        "code": "invalid_request",
    },
    {
        "boundary": "unknown embedding model",
        "request_hint": "unknown local embedding model id",
        "status": 404,
        "code": "embedding_model_not_found",
    },
    {
        "boundary": "external placeholder chat or activation",
        "request_hint": "external placeholder activation or chat model id",
        "status": 501,
        "code": "external_proxy_not_implemented",
    },
    {
        "boundary": "embedding models in /v1/models",
        "expected_behavior": "excluded from /v1/models because they are not chat/generation models",
    },
    {
        "boundary": "GGUF metadata-only chat attempts",
        "request_hint": "metadata-only GGUF model id in /v1/chat/completions",
        "status": 501,
        "code": "not_implemented",
    },
    {
        "boundary": "PyTorch .bin execution",
        "request_hint": "PyTorch .bin model id in /v1/chat/completions",
        "status": 501,
        "code": "not_implemented",
    },
    {
        "boundary": "unsupported ONNX chat or general ONNX model execution",
        "request_hint": "unsupported ONNX model id in /v1/chat/completions",
        "status": 501,
        "code": "not_implemented",
    },
    {
        "boundary": "unverified SafeTensors/Hugging Face model execution",
        "request_hint": "unverified SafeTensors/Hugging Face model id in /v1/chat/completions",
        "status": 501,
        "code": "not_implemented",
    },
    {
        "boundary": "unsupported /v1 endpoint",
        "request_hint": "POST /v1/responses",
        "status": 404,
        "code": "not_found",
    },
    {
        "boundary": "unsupported /v1 method",
        "request_hint": "GET /v1/chat/completions",
        "status": 405,
        "code": "method_not_allowed",
    },
    {
        "boundary": "full OpenAI API parity",
        "expected_behavior": "not claimed",
    },
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
        (
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_PORT",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_WAIT_SECONDS",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_REQUEST_TIMEOUT",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_KEEP_ARTIFACTS",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ROOT",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_MODELS_DIR",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_STATE_DIR",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ARTIFACT_DIR",
            "FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_LOG_DIR",
        ),
        (
            "Default port: `18187`",
            "Default wait: `240` seconds",
            "Default request timeout: `900` seconds",
            "Default root: `${TMPDIR:-/tmp}/fathom-minilm-embeddings-api-$$`",
            "Default artifacts directory: `$TMP_ROOT/artifacts`",
            "By default, successful runs delete the temporary root unless `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_KEEP_ARTIFACTS=1`",
        ),
    ),
    (
        SMOLLM2_OPTIONAL_ACCEPTANCE,
        "FATHOM_SMOLLM2_ACCEPTANCE=1",
        "scripts/smollm2_optional_api_acceptance_smoke.sh",
        "scripts/smollm2_optional_api_acceptance_artifact_qa.py",
        "larger-demo evidence only",
        (
            "FATHOM_SMOLLM2_ACCEPTANCE_PORT",
            "FATHOM_SMOLLM2_ACCEPTANCE_WAIT_SECONDS",
            "FATHOM_SMOLLM2_ACCEPTANCE_REQUEST_TIMEOUT",
            "FATHOM_SMOLLM2_ACCEPTANCE_KEEP_ARTIFACTS",
            "FATHOM_SMOLLM2_ACCEPTANCE_ROOT",
            "FATHOM_SMOLLM2_ACCEPTANCE_MODELS_DIR",
            "FATHOM_SMOLLM2_ACCEPTANCE_STATE_DIR",
            "FATHOM_SMOLLM2_ACCEPTANCE_ARTIFACT_DIR",
            "FATHOM_SMOLLM2_ACCEPTANCE_LOG_DIR",
        ),
        (
            "Default port: `18186`",
            "Default wait: `240` seconds",
            "Default request timeout: `900` seconds",
            "Default root: `${TMPDIR:-/tmp}/fathom-smollm2-api-$$`",
            "Default artifacts directory: `$TMP_ROOT/artifacts`",
            "By default, successful runs delete the temporary root unless `FATHOM_SMOLLM2_ACCEPTANCE_KEEP_ARTIFACTS=1`",
        ),
    ),
    (
        QWEN25_OPTIONAL_ACCEPTANCE,
        "FATHOM_QWEN25_ACCEPTANCE=1",
        "scripts/qwen25_optional_api_acceptance_smoke.sh",
        "scripts/qwen25_optional_api_acceptance_artifact_qa.py",
        "larger-demo evidence only",
        (
            "FATHOM_QWEN25_ACCEPTANCE_PORT",
            "FATHOM_QWEN25_ACCEPTANCE_WAIT_SECONDS",
            "FATHOM_QWEN25_ACCEPTANCE_REQUEST_TIMEOUT",
            "FATHOM_QWEN25_ACCEPTANCE_KEEP_ARTIFACTS",
            "FATHOM_QWEN25_ACCEPTANCE_ROOT",
            "FATHOM_QWEN25_ACCEPTANCE_MODELS_DIR",
            "FATHOM_QWEN25_ACCEPTANCE_STATE_DIR",
            "FATHOM_QWEN25_ACCEPTANCE_ARTIFACT_DIR",
            "FATHOM_QWEN25_ACCEPTANCE_LOG_DIR",
        ),
        (
            "Default port: `18185`",
            "Default wait: `240` seconds",
            "Default request timeout: `900` seconds",
            "Default root: `${TMPDIR:-/tmp}/fathom-qwen25-api-$$`",
            "Default artifacts directory: `$TMP_ROOT/artifacts`",
            "By default, successful runs delete the temporary root unless `FATHOM_QWEN25_ACCEPTANCE_KEEP_ARTIFACTS=1`",
        ),
    ),
)
PUBLIC_CONTRACT_QA_HARDENING_SUBJECT_PATTERN = (
    r"^(Harden public .+ QA|Expose refusal request hints in matrix|Guard public .+ artifact .+|"
    r"Track public smoke artifact QA evidence|Derive public smoke boundaries from manifest|"
    r"Tighten public smoke .+|Guard refusal matrix row drift|Guard failed public smoke .+ drift|"
    r"Standardize v1 unsupported endpoint refusals|Standardize v1 malformed JSON refusals|"
    r"Harden API contract issue privacy checks|Guard PR template truthfulness privacy checks|"
    r"Guard PR template checkbox integrity|"
    r"Guard public issue template privacy checks|Guard public issue template required fields|"
    r"Guard public issue template routing metadata|"
    r"Guard public docs link QA|"
    r"Guard non-contract example surface metadata|"
    r"Guard public contract manifest identity|"
    r"Guard public contract endpoint metadata|"
    r"Guard public contract boundary metadata|"
    r"Guard public contract error envelope docs|"
    r"Guard v1 example JSON boundary docs|"
    r"Guard issue template contact link routing|"
    r"Guard issue template config privacy checks|"
    r"Guard OpenAI SDK example regression|Guard CI token permissions|Guard offline shell syntax coverage|"
    r"Guard CI checkout credential persistence|Guard CI Node cache scope|"
    r"Guard CI PR trigger and cache scope|"
    r"Guard CI secret access|"
    r"Guard CI action allowlist|"
    r"Guard CI runner image policy|"
    r"Guard CI concurrency cancellation|"
    r"Guard CI privileged PR triggers|"
    r"Guard Default CI Gate Inventory Parity|"
    r"Guard offline Python syntax coverage|Guard API example loopback defaults|"
    r"Guard API Client Example Defaults|"
    r"Guard API client dependency boundaries|"
    r"Guard API Client Environment Overrides|"
    r"Guard optional acceptance env docs|"
    r"Guard optional acceptance default docs|"
    r"Guard backend acceptance artifact summary loopback URLs|"
    r"Guard backend acceptance artifact summary timestamps|"
    r"Guard backend acceptance artifact summary markdown index|"
    r"Guard backend acceptance artifact summary identity metadata|"
    r"Guard backend acceptance artifact summary path labels|"
    r"Guard backend acceptance artifact summary port row|"
    r"Guard backend acceptance artifact failure diagnostics|"
    r"Guard backend acceptance artifact caveats|"
    r"Guard optional artifact summary loopback URLs|"
    r"Guard optional artifact summary timestamps|"
    r"Guard optional artifact timestamp markdown rows|"
    r"Guard optional artifact model identity markdown rows|"
    r"Guard optional artifact summary path labels|"
    r"Guard optional artifact summary caveats|"
    r"Guard optional artifact summary markdown index|"
    r"Guard API example stdout share safety|"
    r"Guard REST Client example headers|Guard REST Client JSON body boundaries|"
    r"Guard API example regression self-test|"
    r"Guard CI frontend launch gates|Guard launch syntax checklist consistency|"
    r"Guard frontend package script safety|"
    r"Guard Rust crate publish-safety|"
    r"Guard contributing syntax gate consistency|Guard launch clean install consistency|"
    r"Guard README clean install consistency|"
    r"Guard launch text normalization metadata|Guard public contract smoke endpoint rows|"
    r"Guard backend v1 router manifest drift|"
    r"Guard .+ build artifacts|"
    r"Guard .+benchmark artifacts|"
    r"Guard .+ inventory artifacts|"
    r"Guard .+ benchmark artifacts|"
    r"Guard .+ XML reports|"
    r"Guard .+ coverage artifacts|Guard .+ coverage profile artifacts|"
    r"Guard .+ coverage reports|Guard .+ report artifacts|Guard .+ test report artifacts|"
    r"Guard standalone .+ reports|"
    r"Guard public risk scan .+|Guard tracked credential config files|"
    r"Guard tracked workspace instruction files|Guard tracked local runtime artifacts)$"
)
NO_DOWNLOAD_REFUSAL_EVIDENCE_SUBJECT_PATTERN = (
    r"^(Promote GGUF refusal to public smoke|Standardize v1 unsupported endpoint refusals|"
    r"Standardize v1 malformed JSON refusals|Prove embeddings malformed JSON refusal)$"
)
PUBLIC_RISK_SCAN_HARDENING_SUBJECT_PATTERN = (
    r"^(Guard .+ artifacts|Guard .+ report artifacts|Guard .+ test report artifacts|"
    r"Guard .+ scan artifacts|Guard .+ scanner report artifacts|Guard .+ inventory artifacts|"
    r"Guard .+ build artifacts|Guard .+ benchmark artifacts|Guard .+ coverage artifacts|"
    r"Guard .+ coverage profile artifacts|Guard .+ coverage reports|"
    r"Guard tracked credential config files|Guard tracked workspace instruction files|"
    r"Guard tracked local runtime artifacts|Guard public risk scan .+)$"
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


def assert_non_contract_example_surfaces(allowed: Any) -> None:
    if not isinstance(allowed, list):
        raise AssertionError("manifest.non_contract_surfaces_allowed_in_examples must be a list")
    seen_allowed: set[str] = set()
    for index, item in enumerate(allowed):
        assert_non_empty_string(item, f"manifest.non_contract_surfaces_allowed_in_examples[{index}]")
        if item in seen_allowed:
            raise AssertionError(f"manifest duplicate non-contract example surface: {item}")
        seen_allowed.add(item)

        match = re.fullmatch(r"(GET|POST) (/api/[A-Za-z0-9_./{}:-]+)", item)
        if match is None:
            raise AssertionError(
                "manifest non-contract example surfaces must use uppercase GET/POST local /api paths: "
                f"{item!r}"
            )
        if item not in ALLOWED_NON_CONTRACT_EXAMPLE_SURFACES:
            raise AssertionError(f"manifest non-contract example surface is not reviewed for public examples: {item!r}")


def assert_manifest_shape(manifest: dict[str, Any]) -> None:
    for key in ("name", "status", "base_url", "scope_note"):
        assert_non_empty_string(manifest.get(key), f"manifest.{key}")
        expected = EXPECTED_MANIFEST_METADATA[key]
        if manifest.get(key) != expected:
            raise AssertionError(f"manifest.{key} must remain {expected!r}; found {manifest.get(key)!r}")

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
    if endpoints != EXPECTED_SUPPORTED_ENDPOINTS:
        raise AssertionError("manifest.supported_endpoints must match the narrow public launch endpoint inventory")

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
    if boundaries != EXPECTED_BOUNDARY_ERRORS:
        raise AssertionError("manifest.expected_boundary_errors must match the narrow public launch boundary inventory")

    assert_non_contract_example_surfaces(manifest.get("non_contract_surfaces_allowed_in_examples"))

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


def assert_required_unchecked_task_list_items(text: str, label: str, required_items: tuple[str, ...]) -> None:
    checklist_items = {
        match.group(2).strip(): match.group(1)
        for match in re.finditer(r"(?m)^- \[([ xX])\]\s+(.+?)\s*$", text)
    }
    for item in required_items:
        state = checklist_items.get(item)
        if state is None:
            raise AssertionError(f"{label} must keep unchecked task-list item: {item!r}")
        if state != " ":
            raise AssertionError(f"{label} task-list item must remain unchecked: {item!r}")


def markdown_inline_link_targets(text: str) -> list[tuple[int, str]]:
    targets = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in re.finditer(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)", line):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and raw_target.endswith(">"):
                raw_target = raw_target[1:-1].strip()
            try:
                target = shlex.split(raw_target)[0] if raw_target else ""
            except ValueError as exc:
                raise AssertionError(f"invalid Markdown link target quoting on line {line_no}: {raw_target!r}") from exc
            targets.append((line_no, target))
    return targets


def public_doc_local_link_failures(path: Path, text: str) -> list[str]:
    failures = []
    root = ROOT.resolve()
    label = str(path.relative_to(ROOT))
    for line_no, target in markdown_inline_link_targets(text):
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue
        target_without_fragment = target.split("#", 1)[0].split("?", 1)[0]
        if not target_without_fragment:
            continue
        resolved = (path.parent / target_without_fragment).resolve()
        if resolved != root and root not in resolved.parents:
            failures.append(f"{label}:{line_no}: local Markdown link escapes repository: {target}")
            continue
        if not resolved.exists():
            failures.append(f"{label}:{line_no}: local Markdown link target does not exist: {target}")
    return failures


def assert_public_docs_local_links() -> None:
    failures = []
    for path in PUBLIC_DOC_LOCAL_LINK_PATHS:
        failures.extend(public_doc_local_link_failures(path, read(path)))
    if failures:
        raise AssertionError(failures[0])


def issue_template_header_value(template_text: str, key: str, label: str) -> str:
    matches = re.findall(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", template_text)
    if len(matches) != 1:
        raise AssertionError(f"{label} must contain exactly one top-level {key!r} metadata entry")
    return matches[0].strip().strip("\"'")


def issue_template_labels(template_text: str, label: str) -> list[str]:
    raw_labels = issue_template_header_value(template_text, "labels", label)
    if not raw_labels.startswith("[") or not raw_labels.endswith("]"):
        raise AssertionError(f"{label} labels metadata must stay as an inline list")
    return [item.strip().strip("\"'") for item in raw_labels[1:-1].split(",") if item.strip()]


def assert_issue_template_metadata(
    template_text: str,
    label: str,
    *,
    expected_name: str,
    expected_description: str,
    expected_title: str,
    expected_labels: tuple[str, ...],
) -> None:
    actual_name = issue_template_header_value(template_text, "name", label)
    actual_description = issue_template_header_value(template_text, "description", label)
    actual_title = issue_template_header_value(template_text, "title", label)
    actual_labels = tuple(issue_template_labels(template_text, label))
    if actual_name != expected_name:
        raise AssertionError(f"{label} name metadata must remain {expected_name!r}; found {actual_name!r}")
    if actual_description != expected_description:
        raise AssertionError(
            f"{label} description metadata must remain {expected_description!r}; found {actual_description!r}"
        )
    if actual_title != expected_title:
        raise AssertionError(f"{label} title metadata must remain {expected_title!r}; found {actual_title!r}")
    if actual_labels != expected_labels:
        raise AssertionError(f"{label} labels metadata must remain {expected_labels!r}; found {actual_labels!r}")


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


def assert_readme_clean_install_gate() -> None:
    assert_clean_install_gate(read(README), "README clean install gate")


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


def assert_frontend_package_scripts(package: dict[str, Any] | None = None, label: str = "frontend/package.json") -> None:
    if package is None:
        package = json.loads(read(FRONTEND_PACKAGE))
    if not isinstance(package, dict):
        raise AssertionError(f"{label} must be a JSON object")
    if package.get("private") is not True:
        raise AssertionError(f"{label} must keep private: true")

    scripts = package.get("scripts")
    if not isinstance(scripts, dict):
        raise AssertionError(f"{label} scripts must be an object")
    expected_scripts = {
        "dev": "vite --host 127.0.0.1 --port 4185",
        "preview": "vite preview --host 127.0.0.1 --port 4185",
        "build": "vite build",
        "qa:copy": "node scripts/ui-copy-qa.mjs",
    }
    for script_name, expected in expected_scripts.items():
        actual = scripts.get(script_name)
        if actual != expected:
            raise AssertionError(f"{label} script {script_name!r} must remain {expected!r}; found {actual!r}")
    for script_name in ("dev", "preview"):
        script = scripts[script_name]
        if "--host 0.0.0.0" in script or "--host ::" in script:
            raise AssertionError(f"{label} script {script_name!r} must not bind externally")


def assert_cargo_publish_safety(manifests: dict[str, str] | None = None) -> None:
    expected_paths = {
        "crates/fathom-core/Cargo.toml": CORE_CARGO,
        "crates/fathom-server/Cargo.toml": SERVER_CARGO,
    }
    if manifests is None:
        manifests = {relative_path: read(path) for relative_path, path in expected_paths.items()}

    for relative_path in expected_paths:
        text = manifests.get(relative_path)
        if not isinstance(text, str):
            raise AssertionError(f"{relative_path} must be readable text")
        package_match = re.search(r"(?ms)^\[package\]\n(?P<body>.*?)(?:\n\[|\Z)", text)
        if package_match is None:
            raise AssertionError(f"{relative_path} must contain a [package] section")
        if not re.search(r"(?m)^publish\s*=\s*false\s*$", package_match.group("body")):
            raise AssertionError(f"{relative_path} must keep explicit publish = false")


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
                "scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
                "scripts/public_api_contract_qa.py",
                "scripts/public_contract_smoke_artifact_qa.py",
                "scripts/qwen25_optional_api_acceptance_artifact_qa.py",
                "scripts/smollm2_optional_api_acceptance_artifact_qa.py",
                "crates/fathom-server/src/main.rs",
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


def latest_public_risk_scan_hardening_commit() -> tuple[str, str]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                f"--grep={PUBLIC_RISK_SCAN_HARDENING_SUBJECT_PATTERN}",
                "--extended-regexp",
                "--format=%H%x00%s",
                "--",
                ".gitignore",
                "scripts/public_risk_scan.sh",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise AssertionError("could not resolve latest public risk-scan hardening commit from local git history") from exc

    if not output:
        raise AssertionError("local git history has no recognized public risk-scan hardening commit")
    commit, subject = output.split("\0", 1)
    return commit, subject


def assert_latest_public_risk_scan_hardening_evidence(evidence_text: str) -> None:
    match = re.search(
        r"^- Latest public risk-scan hardening commit: `([0-9a-f]{40})` \(`([^`]+)`\)$",
        evidence_text,
        re.MULTILINE,
    )
    if not match:
        raise AssertionError("launch evidence latest public risk-scan hardening commit line is missing or malformed")

    evidence_commit, evidence_subject = match.groups()
    latest_commit, latest_subject = latest_public_risk_scan_hardening_commit()
    if evidence_commit != latest_commit or evidence_subject != latest_subject:
        raise AssertionError(
            "launch evidence public risk-scan hardening commit is stale: "
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

    assert_standard_error_envelope_docs(manifest)
    assert_v1_contract_json_examples(manifest)


def json_code_blocks(text: str) -> list[Any]:
    parsed: list[Any] = []
    for match in re.finditer(r"```json\s*\n(.*?)\n```", text, re.S):
        try:
            parsed.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return parsed


def assert_standard_error_envelope_docs(
    manifest: dict[str, Any],
    texts: dict[Path, str] | None = None,
) -> None:
    envelope = manifest.get("standard_error_envelope")
    if not isinstance(envelope, dict):
        raise AssertionError("manifest.standard_error_envelope must be an object")
    top_level = envelope.get("top_level")
    fields = envelope.get("error_fields")
    if top_level != ["error"] or fields != ["message", "type", "code", "param"]:
        raise AssertionError("manifest.standard_error_envelope must stay at error.message/type/code/param")

    if texts is None:
        texts = {
            V1_CONTRACT: read(V1_CONTRACT),
            REFUSAL_MATRIX: read(REFUSAL_MATRIX),
            LAUNCH_CHECKLIST: read(LAUNCH_CHECKLIST),
            LAUNCH_EVIDENCE: read(LAUNCH_EVIDENCE),
        }

    v1_text = texts.get(V1_CONTRACT, "")
    assert_contains(v1_text, "Standard error envelope", "standard error envelope docs")
    assert_contains(v1_text, '"error": {', "standard error envelope docs")
    for field in fields:
        assert_contains(v1_text, f"`error.{field}`", "standard error envelope docs")

    for path in (REFUSAL_MATRIX, LAUNCH_CHECKLIST, LAUNCH_EVIDENCE):
        text = texts.get(path, "")
        assert_contains(text, "standard error envelope", f"{path.relative_to(ROOT)} standard error envelope docs")
        for field in fields:
            assert_contains(text, f"`error.{field}`", f"{path.relative_to(ROOT)} standard error envelope docs")

    envelope_fields = set(fields)
    for path, text in texts.items():
        for block in json_code_blocks(text):
            if not isinstance(block, dict):
                continue
            top_level_error_fields = envelope_fields.intersection(block)
            if "error" not in block:
                if top_level_error_fields:
                    raise AssertionError(
                        f"{path.relative_to(ROOT)} documents bare top-level error fields: "
                        f"{sorted(top_level_error_fields)}"
                    )
                continue
            error = block.get("error")
            if not isinstance(error, dict):
                raise AssertionError(f"{path.relative_to(ROOT)} documents a non-object top-level error envelope")
            missing = [field for field in fields if field not in error]
            if missing:
                raise AssertionError(f"{path.relative_to(ROOT)} standard error envelope missing fields: {missing}")


def assert_v1_contract_json_examples(manifest: dict[str, Any], text: str | None = None) -> None:
    if text is None:
        text = read(V1_CONTRACT)
    blocks = [block for block in json_code_blocks(text) if isinstance(block, dict)]

    for block in blocks:
        if block.get("stream") is True:
            raise AssertionError("docs/api/v1-contract.md JSON examples must not show streaming chat requests")
        if block.get("encoding_format") == "base64":
            raise AssertionError("docs/api/v1-contract.md JSON examples must not show base64 embeddings requests")
        if "error" not in block and set(block).intersection({"message", "type", "code", "param"}):
            raise AssertionError("docs/api/v1-contract.md JSON examples must not show bare error fields")

    health_examples = [block for block in blocks if {"ok", "engine", "generation_ready"}.issubset(block)]
    if len(health_examples) != 1:
        raise AssertionError("docs/api/v1-contract.md must keep exactly one /v1/health response JSON example")
    health = health_examples[0]
    if health.get("engine") != "fathom" or not isinstance(health.get("generation_ready"), bool):
        raise AssertionError("docs/api/v1-contract.md /v1/health example must stay on the Fathom readiness shape")

    model_list_examples = [
        block
        for block in blocks
        if block.get("object") == "list"
        and isinstance(block.get("data"), list)
        and block["data"]
        and isinstance(block["data"][0], dict)
        and isinstance(block["data"][0].get("fathom"), dict)
    ]
    if len(model_list_examples) != 1:
        raise AssertionError("docs/api/v1-contract.md must keep exactly one /v1/models response JSON example")
    model_entry = model_list_examples[0]["data"][0]
    fathom_model = model_entry["fathom"]
    if fathom_model.get("provider_kind") != "local":
        raise AssertionError("docs/api/v1-contract.md /v1/models example must not show external provider placeholders")
    if fathom_model.get("capability_status") != "Runnable":
        raise AssertionError("docs/api/v1-contract.md /v1/models example must show only runnable chat/generation models")
    if "safetensors-hf" not in fathom_model.get("backend_lanes", []):
        raise AssertionError("docs/api/v1-contract.md /v1/models example must remain a verified SafeTensors/HF chat lane")
    forbidden_model_keys = {"embedding_dimension", "embedding", "metadata_only", "external_api"}
    if forbidden_model_keys.intersection(fathom_model) or forbidden_model_keys.intersection(model_entry):
        raise AssertionError("docs/api/v1-contract.md /v1/models example must not show embedding, metadata-only, or external models")

    chat_requests = [block for block in blocks if isinstance(block.get("messages"), list)]
    if len(chat_requests) != 1:
        raise AssertionError("docs/api/v1-contract.md must keep exactly one chat request JSON example")
    chat_request = chat_requests[0]
    if chat_request.get("stream") is not False:
        raise AssertionError("docs/api/v1-contract.md chat request example must keep stream false")
    assert_non_empty_string(chat_request.get("model"), "docs/api/v1-contract.md chat request model")
    if not chat_request.get("messages"):
        raise AssertionError("docs/api/v1-contract.md chat request example must keep non-empty messages")
    if "encoding_format" in chat_request:
        raise AssertionError("docs/api/v1-contract.md chat request example must not include embedding fields")

    chat_responses = [block for block in blocks if isinstance(block.get("choices"), list)]
    if len(chat_responses) != 1:
        raise AssertionError("docs/api/v1-contract.md must keep exactly one chat response JSON example")
    chat_response = chat_responses[0]
    if "data" in chat_response or "error" in chat_response:
        raise AssertionError("docs/api/v1-contract.md chat response example must not mix embedding data or errors")
    if not chat_response["choices"] or not isinstance(chat_response["choices"][0].get("message"), dict):
        raise AssertionError("docs/api/v1-contract.md chat response example must keep assistant message choices")

    embedding_requests = [block for block in blocks if "input" in block]
    if len(embedding_requests) != 1:
        raise AssertionError("docs/api/v1-contract.md must keep exactly one embeddings request JSON example")
    embedding_request = embedding_requests[0]
    assert_non_empty_string(embedding_request.get("model"), "docs/api/v1-contract.md embeddings request model")
    if embedding_request.get("encoding_format") != "float":
        raise AssertionError("docs/api/v1-contract.md embeddings request example must keep encoding_format float")
    if "messages" in embedding_request:
        raise AssertionError("docs/api/v1-contract.md embeddings request example must not include chat fields")

    embedding_responses = [
        block
        for block in blocks
        if block.get("object") == "list"
        and isinstance(block.get("data"), list)
        and block["data"]
        and isinstance(block["data"][0], dict)
        and "embedding" in block["data"][0]
    ]
    if len(embedding_responses) != 1:
        raise AssertionError("docs/api/v1-contract.md must keep exactly one embeddings response JSON example")
    embedding_response = embedding_responses[0]
    if "choices" in embedding_response or "error" in embedding_response:
        raise AssertionError("docs/api/v1-contract.md embeddings response example must not mix chat choices or errors")
    fathom_embedding = embedding_response.get("fathom")
    if not isinstance(fathom_embedding, dict) or fathom_embedding.get("scope") != "verified local embedding runtime only":
        raise AssertionError("docs/api/v1-contract.md embeddings response example must keep verified local embedding scope")

    supported_paths = {(endpoint["method"], endpoint["path"]) for endpoint in manifest.get("supported_endpoints", [])}
    if ("POST", "/v1/chat/completions") not in supported_paths or ("POST", "/v1/embeddings") not in supported_paths:
        raise AssertionError("manifest must keep chat and embeddings endpoints for /v1 JSON example QA")


def backend_v1_router_block(server_text: str) -> str:
    match = re.search(
        r"let\s+v1_router\s*=\s*Router::new\(\)(?P<router>.*?)\n\s*Router::new\(\)",
        server_text,
        re.S,
    )
    if not match:
        raise AssertionError("crates/fathom-server/src/main.rs is missing the v1_router Router::new() block")
    return match.group("router")


def assert_backend_v1_router_matches_manifest(manifest: dict[str, Any], server_text: str | None = None) -> None:
    if server_text is None:
        server_text = read(SERVER_MAIN)
    router_text = backend_v1_router_block(server_text)

    method_helpers = {
        "GET": "get",
        "POST": "post",
    }
    for endpoint in manifest.get("supported_endpoints", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        if not isinstance(method, str) or not isinstance(path, str):
            raise AssertionError("manifest supported_endpoints entries must include string method/path")
        if method not in method_helpers:
            raise AssertionError(f"unsupported backend router method in manifest: {method!r}")
        if not path.startswith("/v1/"):
            raise AssertionError(f"backend v1 router check only supports /v1 endpoints: {method} {path}")

        nested_path = path.removeprefix("/v1")
        route_pattern = re.compile(
            rf'\.route\(\s*"{re.escape(nested_path)}"\s*,\s*'
            rf"{method_helpers[method]}\([^)]+\)\.fallback\(v1_method_not_allowed\)",
            re.S,
        )
        if not route_pattern.search(router_text):
            raise AssertionError(
                "backend v1 router is missing manifest endpoint with method fallback: "
                f"{method} {path}"
            )

    if ".fallback(v1_not_found)" not in router_text:
        raise AssertionError("backend v1 router must keep v1_not_found fallback for unsupported /v1 routes")

    if 'nest("/v1", v1_router)' not in server_text:
        raise AssertionError("backend app router must keep the /v1 router nested at /v1")

    for function_name, expected_code in (
        ("v1_json_rejection_error", "invalid_request"),
        ("v1_not_found", "not_found"),
        ("v1_method_not_allowed", "method_not_allowed"),
    ):
        function_match = re.search(rf"fn\s+{function_name}\b|async\s+fn\s+{function_name}\b", server_text)
        if not function_match:
            raise AssertionError(f"backend is missing {function_name}")
        snippet = server_text[function_match.start() : function_match.start() + 900]
        if f'"{expected_code}"' not in snippet:
            raise AssertionError(f"backend {function_name} must keep {expected_code!r} error envelope code")


def assert_manifest_base_url_alignment(manifest: dict[str, Any], texts: dict[str, str] | None = None) -> None:
    base_url = manifest.get("base_url")
    assert_non_empty_string(base_url, "manifest.base_url")

    if texts is None:
        texts = {str(path.relative_to(ROOT)): read(path) for path in PUBLIC_BASE_URL_PATHS}

    for label, text in texts.items():
        assert_contains(text, base_url, f"{label} public API base URL")


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
        *DEFAULT_CI_GATE_COMMANDS,
    ]
    for gate in checklist_required_gates:
        assert_contains(launch_text, gate, "launch checklist no-download gates")
    assert_contains(
        launch_text,
        "keeps `docs/api/client-examples.md` aligned with the example script defaults",
        "launch checklist API client default QA scope",
    )
    assert_contains(
        launch_text,
        "fake loopback `FATHOM_*` overrides so documented environment overrides keep reaching install, chat, and embedding request bodies",
        "launch checklist API client environment override QA scope",
    )
    assert_contains(
        launch_text,
        "keeps the cURL quickstart dependency-light and the no-deps Python example on a standard-library import allow-list",
        "launch checklist API client dependency QA scope",
    )
    assert_contains(
        launch_text,
        "parses the REST Client `.http` JSON bodies",
        "launch checklist REST Client body-boundary QA scope",
    )
    assert_contains(
        launch_text,
        "include commit, generated UTC timestamp, manifest path/name/status, endpoint checks, boundary checks, and scope caveats only",
        "launch checklist public contract smoke artifact field scope",
    )
    assert_contains(
        launch_text,
        "intentionally omit local temp paths, server log tails, request secrets, and model/provider payloads",
        "launch checklist public contract smoke artifact share-safety scope",
    )
    assert_contains(
        launch_text,
        "Validate them offline before sharing with `python3 scripts/public_contract_smoke_artifact_qa.py public-contract-artifacts`",
        "launch checklist public contract smoke artifact QA command",
    )
    assert_contains(
        launch_text,
        "keeps `summary.model_id`, `summary.repo_id`, and `summary.revision` aligned between `summary.json` and `summary.md`",
        "launch checklist optional artifact model identity scope",
    )
    assert_contains(
        launch_text,
        "keeps share-safe artifact/state/model/log labels aligned between `summary.json` and `summary.md`",
        "launch checklist optional artifact path-label scope",
    )
    assert_contains(
        launch_text,
        "keeps required boundary caveats present in `summary.json` and `summary.md`",
        "launch checklist optional artifact caveat scope",
    )
    assert_contains(
        launch_text,
        "Markdown/JSON check-index, timestamp, model-identity, path-label, loopback, and caveat consistency",
        "launch checklist optional artifact complete proof scope",
    )
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
    assert_contains(
        launch_text,
        "non-contract example surfaces constrained to reviewed local catalog helper paths",
        "launch checklist non-contract example surface metadata scope",
    )
    assert_contains(
        launch_text,
        "pins the public contract manifest identity metadata",
        "launch checklist manifest identity metadata scope",
    )
    assert_contains(
        launch_text,
        "pins the public contract manifest's supported endpoint inventory",
        "launch checklist manifest endpoint inventory scope",
    )
    assert_contains(
        launch_text,
        "pins the public contract manifest's refusal/boundary inventory",
        "launch checklist manifest boundary inventory scope",
    )
    assert_contains(
        launch_text,
        "pins concrete JSON examples in `docs/api/v1-contract.md` to the narrow launch boundary",
        "launch checklist v1 JSON example boundary scope",
    )
    assert_contains(launch_text, "root `.gitattributes` text-normalization metadata", "launch checklist text-normalization metadata scope")
    assert_contains(
        launch_text,
        "launch-facing relative Markdown links",
        "launch checklist local Markdown link QA scope",
    )
    assert_contains(
        launch_text,
        "unchecked Markdown task-list items",
        "launch checklist PR template checkbox QA scope",
    )
    assert_contains(
        launch_text,
        "tracked Git LFS pointer files that can hide external artifact downloads",
        "launch checklist Git LFS pointer risk-scan scope",
    )
    assert_contains(
        launch_text,
        "tracked Git submodule metadata for local/relative, SSH-only, authenticated, secret, or private source URLs",
        "launch checklist Git submodule metadata risk-scan scope",
    )
    assert_contains(
        launch_text,
        "model/checkpoint artifact guard also rejects tracked local model/checkpoint artifacts",
        "launch checklist model/checkpoint artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "top-level `models/`, `model-store/`, `weights/`, `checkpoints/`, and model binary/data files including `.safetensors`, `.gguf`, `.onnx`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.npz`, `.npy`, and `.tflite`",
        "launch checklist model/checkpoint artifact examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local model/checkpoint artifacts",
        "launch checklist model/checkpoint artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "ML experiment/tracking artifact guard rejects tracked local W&B, MLflow, Lightning, and TensorBoard run outputs",
        "launch checklist ML experiment/tracking artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.wandb/`, `wandb/`, `mlruns/`, `lightning_logs/`, and `events.out.tfevents.*`",
        "launch checklist ML experiment/tracking artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local ML experiment/tracking artifacts",
        "launch checklist ML experiment/tracking artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Docker/container artifact guard rejects tracked local container runtime and override artifacts",
        "launch checklist Docker/container artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as top-level `.docker/`, `docker-data/`, `docker-volumes/`, `docker-compose.override.yml`, `docker-compose.override.yaml`, `compose.override.yml`, and `compose.override.yaml`",
        "launch checklist Docker/container artifact examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local Docker/container artifacts",
        "launch checklist Docker/container artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "deployment platform artifact guard rejects tracked local Vercel and Netlify state directories",
        "launch checklist deployment platform artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.vercel/` and `.netlify/` at any tree depth",
        "launch checklist deployment platform artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage",
        "launch checklist deployment platform artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Terraform/OpenTofu/Terragrunt infrastructure state artifact guard rejects tracked local infrastructure state artifacts",
        "launch checklist infrastructure state artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.terraform/`, `.tofu/`, `.terragrunt-cache/`, `.terraform.lock.hcl`, `.tofu.lock.hcl`, `.tfstate`, `.tfvars`, `.tfvars.json`, and `.tfplan` files",
        "launch checklist infrastructure state artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage for local infrastructure state artifacts",
        "launch checklist infrastructure state artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Nix build result artifact guard rejects tracked local result symlinks/directories",
        "launch checklist Nix artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as root `result` and `result-*` outputs",
        "launch checklist Nix artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage for local Nix build result artifacts",
        "launch checklist Nix artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Nix files such as `flake.nix`, `flake.lock`, `default.nix`, `shell.nix`, or `*.nix` files",
        "launch checklist Nix source-file allowance",
    )
    assert_contains(
        launch_text,
        "blocks tracked local Bazel output symlinks/directories",
        "launch checklist Bazel artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `bazel-bin/`, `bazel-out/`, `bazel-testlogs/`, and root `bazel-*` outputs",
        "launch checklist Bazel artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage for local Bazel build artifacts",
        "launch checklist Bazel artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "It does not treat source-of-truth Bazel files such as `BUILD`, `BUILD.bazel`, `MODULE.bazel`, or `.bzl` files as build artifacts",
        "launch checklist Bazel source-file allowance",
    )
    assert_contains(
        launch_text,
        "blocks tracked local Buck/Buck2 build artifacts",
        "launch checklist Buck/Buck2 artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.buckd/` and `buck-out/`",
        "launch checklist Buck/Buck2 artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage for local Buck/Buck2 build artifacts",
        "launch checklist Buck/Buck2 artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Buck files such as `BUCK`, `BUCK.v2`, or `.buckconfig`",
        "launch checklist Buck/Buck2 source allowance",
    )
    assert_contains(
        launch_text,
        "tracked local Swift Package Manager build/workspace artifacts such as `.build/` and `.swiftpm/` at any tree depth",
        "launch checklist SwiftPM artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Swift Package Manager artifacts",
        "launch checklist SwiftPM artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Swift package files such as `Package.swift` or Swift source files",
        "launch checklist SwiftPM source-file allowance",
    )
    assert_contains(
        launch_text,
        "tracked local Zig build artifacts such as `.zig-cache/`, `zig-cache/`, and `zig-out/` at any tree depth",
        "launch checklist Zig artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Zig build artifacts",
        "launch checklist Zig artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Zig files such as `build.zig`, `build.zig.zon`, or Zig source files",
        "launch checklist Zig source-file allowance",
    )
    assert_contains(
        launch_text,
        "blocks tracked local Dart/Flutter artifacts",
        "launch checklist Dart/Flutter artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.dart_tool/`, `.pub-cache/`, `.pub/`, `.packages`, `.flutter-plugins`, and `.flutter-plugins-dependencies` at any tree depth",
        "launch checklist Dart/Flutter artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Dart/Flutter artifacts",
        "launch checklist Dart/Flutter artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Dart/Flutter files such as `pubspec.yaml`, `pubspec.lock`, Dart source, Android Gradle project files, or iOS Xcode project files",
        "launch checklist Dart/Flutter source-file allowance",
    )
    assert_contains(
        launch_text,
        "general mobile/Xcode/Android build artifact guard blocks tracked local build outputs",
        "launch checklist mobile build artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `DerivedData/`, `.gradle/`, `xcuserdata/`, `local.properties`, `.xcuserstate`, `.xcresult`, `.ipa`, `.apk`, `.aab`, and `.dSYM` files",
        "launch checklist mobile build artifact examples",
    )
    assert_contains(
        launch_text,
        "root, nested, Android-specific, and platform artifact `.gitignore` coverage for local mobile build outputs",
        "launch checklist mobile build artifact ignore scope",
    )
    assert_contains(launch_text, "Expo local project state such as `.expo/` and `.expo-shared/`", "launch checklist Expo local state scope")
    assert_contains(
        launch_text,
        "Expo local project state such as `.expo/` and `.expo-shared/` at any tree depth, with matching root and nested `.gitignore` coverage",
        "launch checklist Expo local state ignore scope",
    )
    assert_contains(launch_text, "source-of-truth mobile app code, Expo config, Android Gradle project files, or iOS Xcode project files", "launch checklist Expo source allowance")
    assert_contains(launch_text, "Android native build intermediates such as `.cxx/` and `.externalNativeBuild/`", "launch checklist Android native build intermediate scope")
    assert_contains(
        launch_text,
        "including the same directories under `android/`, with matching root, nested, and `android/` `.gitignore` coverage",
        "launch checklist Android native build intermediate ignore scope",
    )
    assert_contains(launch_text, "source-of-truth Android Gradle project files, JNI/C++ source files, or checked-in native build configuration", "launch checklist Android native source allowance")
    assert_contains(launch_text, "CocoaPods dependency outputs such as root `Pods/` and `ios/Pods/`", "launch checklist CocoaPods dependency output scope")
    assert_contains(
        launch_text,
        "CocoaPods dependency outputs such as root `Pods/` and `ios/Pods/`, with matching root `.gitignore` coverage",
        "launch checklist CocoaPods dependency output ignore scope",
    )
    assert_contains(launch_text, "source-of-truth CocoaPods files such as `Podfile` or `Podfile.lock`", "launch checklist CocoaPods source allowance")
    assert_contains(launch_text, "Carthage dependency outputs such as `Carthage/Build/` and `Carthage/Checkouts/`", "launch checklist Carthage dependency output scope")
    assert_contains(
        launch_text,
        "Carthage dependency outputs such as `Carthage/Build/` and `Carthage/Checkouts/`, with matching root `.gitignore` coverage",
        "launch checklist Carthage dependency output ignore scope",
    )
    assert_contains(launch_text, "source-of-truth Carthage files such as `Cartfile` or `Cartfile.resolved`", "launch checklist Carthage source allowance")
    assert_contains(
        launch_text,
        "Fastlane generated report/test artifacts such as `fastlane/report.xml`, `fastlane/Preview.html`, and `fastlane/test_output/`",
        "launch checklist Fastlane report/test artifact scope",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage through the local mobile/Xcode/Android build artifact guard",
        "launch checklist Fastlane report/test artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Fastlane configuration or App Store metadata such as `Fastfile`, `Appfile`, or `fastlane/metadata/`",
        "launch checklist Fastlane source allowance",
    )
    assert_contains(
        launch_text,
        "mobile/Xcode/Android signing/provisioning artifact guard also blocks tracked local provisioning/build archives",
        "launch checklist mobile signing artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `*.mobileprovision`, `*.provisionprofile`, and `*.xcarchive`",
        "launch checklist mobile signing artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage for local mobile/Xcode/Android signing/provisioning artifacts",
        "launch checklist mobile signing artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "screenshot/screen-recording artifact guard blocks tracked local captures with default macOS capture prefixes",
        "launch checklist screenshot/screen-recording artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `Screenshot *`, `Screen Shot *`, and `Screen Recording *`",
        "launch checklist screenshot/screen-recording artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage for local screenshot/screen-recording artifacts",
        "launch checklist screenshot/screen-recording artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "audio/video capture/export artifact guard blocks tracked local captures and exports",
        "launch checklist audio/video capture artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `Audio Recording *`, `Voice Memo *`, `*.m4a`, `*.mov`, `*.mp3`, `*.mp4`, and `*.wav`",
        "launch checklist audio/video capture artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage for local audio/video capture/export artifacts",
        "launch checklist audio/video capture artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Python cache/build artifact guard also rejects tracked local Pyre/Pytype and mypy daemon state",
        "launch checklist Python artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.pyre/`, `.pytype/`, and `.dmypy.json`",
        "launch checklist Python type-checker artifact examples",
    )
    assert_contains(
        launch_text,
        "plus Python package metadata/build artifacts such as `.eggs/`, `*.egg-info/`, and `*.dist-info/`, with matching root and nested `.gitignore` coverage",
        "launch checklist Python package metadata artifact scope",
    )
    assert_contains(
        launch_text,
        "Python virtualenv/dependency artifact guard rejects local PDM/PEP 582 artifacts",
        "launch checklist Python virtualenv/dependency artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.pdm-build/` and `__pypackages__/`",
        "launch checklist Python virtualenv/dependency artifact examples",
    )
    assert_contains(
        launch_text,
        "plus virtualenv and package-manager cache directories at any tree depth, with matching root and nested `.gitignore` coverage",
        "launch checklist Python virtualenv/dependency artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Python files such as `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements*.txt`, `Pipfile`, `Pipfile.lock`, `poetry.lock`, `uv.lock`, Python source files, or Python docs",
        "launch checklist Python source-file allowance",
    )
    assert_contains(
        launch_text,
        "The Python benchmark artifact guard rejects tracked local pytest-benchmark output such as `.benchmarks/` and `pytest-benchmark.json` exports",
        "launch checklist Python benchmark artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage for local Python benchmark artifacts",
        "launch checklist Python benchmark artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "It does not treat curated benchmark evidence under `docs/benchmarks/` or source helpers such as `scripts/bench_backend.py` as local benchmark artifacts",
        "launch checklist Python benchmark source/evidence allowance",
    )
    assert_contains(
        launch_text,
        "tracked local frontend/static-site framework caches and build outputs",
        "launch checklist frontend/static-site artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "Vite/Vitest cache/config timestamp artifacts",
        "launch checklist Vite/Vitest frontend artifact scope",
    )
    assert_contains(
        launch_text,
        "Storybook static build output",
        "launch checklist Storybook frontend artifact scope",
    )
    assert_contains(
        launch_text,
        "Astro `.astro/`, Docusaurus `.docusaurus/`, Hugo `resources/_gen/` and `.hugo_build.lock`, Jekyll `_site/`, `.jekyll-cache/`, and `.sass-cache/`, Vite `.vite/`, VitePress `.vitepress/cache/` and `.vitepress/dist/`, Metro `.metro-cache/`, Nx `.nx/`, SWC `.swc/`, Rollup TypeScript plugin `.rpt2_cache/`, Webpack `.webpack-cache/`, Next.js `.next/`, SvelteKit `.svelte-kit/`, Nuxt `.nuxt/` and `.output/`, and Angular `.angular/cache/` directories",
        "launch checklist frontend framework artifact scope",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage for those local frontend/static-site artifacts",
        "launch checklist frontend/static-site artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "selected nested Yarn cache/state artifacts such as `.yarn/cache/`, `.yarn/unplugged/`, `.yarn/build-state.yml`, and `.yarn/install-state.gz`",
        "launch checklist Yarn cache/state artifact scope",
    )
    assert_contains(
        launch_text,
        "while preserving source-of-truth lockfiles and package manifests",
        "launch checklist frontend package source allowance",
    )
    assert_contains(
        launch_text,
        "source-of-truth Metro configuration such as `metro.config.js`",
        "launch checklist Metro source allowance",
    )
    assert_contains(
        launch_text,
        "tracked local Lighthouse/LHCI audit outputs such as `.lighthouseci/`, `lhci_reports/`, `lighthouse-report.html`, and `lighthouse-report.json`",
        "launch checklist Lighthouse/LHCI artifact scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated audit outputs",
        "launch checklist Lighthouse/LHCI artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "preserving source-of-truth Lighthouse config files such as `.lighthouserc.*`",
        "launch checklist Lighthouse config allowance",
    )
    assert_contains(
        launch_text,
        "tracked ESLint report outputs such as `eslint-report/`, `eslint-reports/`, `eslint-junit.xml`, `eslint-report.json`, and `eslint-results.xml`",
        "launch checklist ESLint report artifact scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated lint report artifacts",
        "launch checklist ESLint report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth ESLint configuration such as `eslint.config.*` or `.eslintrc.*`",
        "launch checklist ESLint config allowance",
    )
    assert_contains(
        launch_text,
        "tracked Stylelint report outputs such as `stylelint-report/`, `stylelint-reports/`, `stylelint-junit.xml`, `stylelint-report.json`, and `stylelint-results.xml`",
        "launch checklist Stylelint report artifact scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated lint report artifacts",
        "launch checklist Stylelint report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Stylelint configuration such as `stylelint.config.*` or `.stylelintrc*`",
        "launch checklist Stylelint config allowance",
    )
    assert_contains(
        launch_text,
        "tracked Biome report outputs such as `biome-report/`, `biome-reports/`, `biome-report.html`, `biome-report.json`, `biome-report.xml`, `biome-results.json`, and timestamped `biome-report-*` or `biome-results-*` files",
        "launch checklist Biome report artifact scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated lint/formatter report artifacts",
        "launch checklist Biome report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Biome configuration such as `biome.json` or `biome.jsonc`",
        "launch checklist Biome config allowance",
    )
    assert_contains(
        launch_text,
        "tracked Markdown/prose lint report outputs such as `markdownlint-report/`, `markdownlint-reports/`, `markdownlint-junit.xml`, `markdownlint-report.json`, `markdownlint-results.xml`, `remark-report/`, `remark-reports/`, `remark-junit.xml`, `remark-report.json`, `remark-results.xml`, `proselint-report/`, `proselint-reports/`, `proselint-junit.xml`, `proselint-report.json`, and `proselint-results.xml`",
        "launch checklist Markdown/prose lint report artifact scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated documentation lint report artifacts",
        "launch checklist Markdown/prose lint report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Markdown/prose lint configuration such as `.markdownlint.*`, `.remarkrc`, `remark.config.*`, or `.proselintrc`",
        "launch checklist Markdown/prose lint config allowance",
    )
    assert_contains(
        launch_text,
        "Deno local cache artifacts such as `.deno/` and `deno-dir/`",
        "launch checklist Deno artifact scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Deno project files such as `deno.json`, `deno.jsonc`, `deno.lock`, JavaScript files, and TypeScript files",
        "launch checklist Deno source-file allowance",
    )
    assert_contains(
        launch_text,
        "Rust/Cargo cache/build artifacts such as `.cargo/`, `target/`, compiler outputs, and coverage profiles, with matching root and nested `.gitignore` coverage",
        "launch checklist Rust/Cargo artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Gradle/JVM build artifact guard also rejects tracked local Gradle cache/state",
        "launch checklist Gradle/JVM artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "common Gradle `build/` output subtrees such as `classes/`, `reports/`, `test-results/`, `tmp/`, `generated/`, `intermediates/`, and `libs/`",
        "launch checklist Gradle/JVM artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for Gradle cache/state",
        "launch checklist Gradle/JVM artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Gradle project files such as `build.gradle`, `settings.gradle`, `gradle.properties`, `gradlew`, or `gradle/wrapper/gradle-wrapper.properties`",
        "launch checklist Gradle/JVM source-file allowance",
    )
    assert_contains(
        launch_text,
        "tracked local JVM dependency artifacts",
        "launch checklist JVM dependency artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "Maven repository/cache state such as `.m2/` at any tree depth",
        "launch checklist JVM dependency artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local JVM dependency artifacts",
        "launch checklist JVM dependency artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Maven/JVM project files such as `pom.xml`, Java source files, Kotlin source files, Scala source files, or JVM docs",
        "launch checklist JVM dependency source-file allowance",
    )
    assert_contains(
        launch_text,
        "JVM compiler artifact guard also rejects tracked local bytecode outputs such as `*.class` and `*.tasty`",
        "launch checklist JVM compiler artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "tracked local bytecode outputs such as `*.class` and `*.tasty`",
        "launch checklist JVM compiler artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist JVM compiler artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth JVM project files such as `pom.xml`, `build.gradle`, `build.sbt`, Java source files, Kotlin source files, or Scala source files",
        "launch checklist JVM compiler source-file allowance",
    )
    assert_contains(
        launch_text,
        ".NET/NuGet artifact guard also rejects tracked local dependency/build/user-state artifacts",
        "launch checklist .NET/NuGet artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.nuget/`, `packages/`, `bin/Debug/`, `obj/Release/`, `project.assets.json`, `project.nuget.cache`, `*.nupkg`, `*.snupkg`, `*.csproj.user`, and `*.suo`",
        "launch checklist .NET/NuGet artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage",
        "launch checklist .NET/NuGet artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth .NET files such as `*.csproj`, `*.fsproj`, `*.vbproj`, `*.sln`, `Directory.Build.props`, C# source files, F# source files, or NuGet lock files",
        "launch checklist .NET/NuGet source-file allowance",
    )
    assert_contains(
        launch_text,
        "tracked local Clojure/Leiningen artifacts such as `.lein/`, `.cpcache/`, and `.shadow-cljs/` at any tree depth, plus `.nrepl-port`",
        "launch checklist Clojure/Leiningen artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Clojure/Leiningen artifacts",
        "launch checklist Clojure/Leiningen artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Clojure files such as `project.clj`, `deps.edn`, `bb.edn`, `shadow-cljs.edn`, or Clojure/ClojureScript source files",
        "launch checklist Clojure/Leiningen source allowance",
    )
    assert_contains(
        launch_text,
        "tracked local CI runner artifacts/config such as `.act/`, `.actrc`, `actions-runner/`, `_work/`, `_diag/`, `.runner`, `.credentials`, and `.credentials_rsaparams`",
        "launch checklist local CI runner artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage, so local workflow runner state cannot be mistaken for launch evidence",
        "launch checklist local CI runner artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked local Swift Package Manager build/workspace artifacts such as `.build/` and `.swiftpm/` at any tree depth",
        "launch checklist Swift Package Manager artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for local Swift Package Manager artifacts",
        "launch checklist Swift Package Manager artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Swift package files such as `Package.swift` or Swift source files",
        "launch checklist Swift Package Manager source-file allowance",
    )
    assert_contains(
        launch_text,
        "tracked local Zig build artifacts such as `.zig-cache/`, `zig-cache/`, and `zig-out/` at any tree depth",
        "launch checklist Zig build artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for local Zig build artifacts",
        "launch checklist Zig build artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Zig files such as `build.zig`, `build.zig.zon`, or Zig source files",
        "launch checklist Zig build source-file allowance",
    )
    assert_contains(
        launch_text,
        "Go cache/test artifact guard also rejects tracked local generated outputs",
        "launch checklist Go artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".gocache/` and `.gomodcache/` at any tree depth, `cover.out`, `coverage.out`, `*.coverprofile`, and `*.test`",
        "launch checklist Go artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Go artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Go files such as `go.mod`, `go.sum`, Go source files, or Go docs",
        "launch checklist Go source-file allowance",
    )
    assert_contains(
        launch_text,
        "Ruby/Bundler guard also rejects tracked local dependency artifacts such as `.bundle/`, `vendor/bundle/`, and `vendor/cache/` at any tree depth",
        "launch checklist Ruby/Bundler artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Ruby/Bundler artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Ruby files such as `Gemfile`, `Gemfile.lock`, Ruby source files, Ruby docs, or non-artifact vendor paths",
        "launch checklist Ruby/Bundler source-file allowance",
    )
    assert_contains(
        launch_text,
        "PHP Composer guard also rejects tracked local dependency/test artifacts",
        "launch checklist PHP Composer artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "vendor/autoload.php`, `vendor/bin/`, `vendor/composer/`, and `.phpunit.cache/` at any tree depth, plus `.phpunit.result.cache` and `composer.phar`",
        "launch checklist PHP Composer artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist PHP Composer artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Composer files such as `composer.json`, `composer.lock`, PHP source files, or PHP docs",
        "launch checklist PHP Composer source-file allowance",
    )
    assert_contains(
        launch_text,
        "Perl/CPAN build/dependency artifact guard rejects tracked local CPAN client, build, and local::lib outputs",
        "launch checklist Perl/CPAN artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".cpan/`, `.cpanm/`, `blib/`, `local/lib/perl5/`, `MYMETA.json`, `MYMETA.yml`, and `pm_to_blib`",
        "launch checklist Perl/CPAN artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Perl/CPAN artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Perl files such as `cpanfile`, `cpanfile.snapshot`, `Makefile.PL`, `Build.PL`, Perl source files, or Perl docs",
        "launch checklist Perl/CPAN source-file allowance",
    )
    assert_contains(
        launch_text,
        "Julia depot/preference artifact guard also rejects tracked local Julia depot, preference, coverage, and allocation artifacts",
        "launch checklist Julia artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".julia/` at any tree depth, `LocalPreferences.toml`, `*.jl.cov`, and `*.jl.mem`",
        "launch checklist Julia artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Julia artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Julia project files such as `Project.toml`, `Manifest.toml`, `Artifacts.toml`, or Julia source files as local depot/preference artifacts by filename alone",
        "launch checklist Julia source-file allowance",
    )
    assert_contains(
        launch_text,
        "R/RStudio artifact guard also rejects tracked local session and dependency artifacts",
        "launch checklist R/RStudio artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".Rproj.user/` and `renv/library/` at any tree depth, plus `.Rhistory`, `.RData`, and `.Ruserdata`",
        "launch checklist R/RStudio artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for local R/RStudio artifacts",
        "launch checklist R/RStudio artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth R files such as `*.R`, `*.Rmd`, `.Rproj` project files, `renv.lock`, `DESCRIPTION`, or R docs",
        "launch checklist R/RStudio source-file allowance",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage",
        "launch checklist PHP Composer artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage",
        "launch checklist Go artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Haskell Stack/Cabal build artifact guard also rejects tracked local build outputs",
        "launch checklist Haskell Stack/Cabal artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".stack-work/`, `dist-newstyle/`, `.cabal-sandbox/`, and `cabal.sandbox.config` at any tree depth",
        "launch checklist Haskell Stack/Cabal artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage",
        "launch checklist Haskell Stack/Cabal artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Haskell files such as `.cabal` files, `stack.yaml`, `cabal.project`, or Haskell source files as local Stack/Cabal build artifacts by filename alone",
        "launch checklist Haskell Stack/Cabal source-file allowance scope",
    )
    assert_contains(
        launch_text,
        "OCaml/opam local switch artifact guard also rejects tracked local switch/dependency artifacts",
        "launch checklist OCaml/opam artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "_opam/` and `.opam-switch/` at any tree depth",
        "launch checklist OCaml/opam artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage",
        "launch checklist OCaml/opam artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth OCaml project files such as `dune`, `dune-project`, `*.opam`, or OCaml source files",
        "launch checklist OCaml/opam source-file allowance scope",
    )
    assert_contains(
        launch_text,
        "Lua/LuaRocks artifact guard also rejects tracked local dependency/build/package artifacts",
        "launch checklist Lua/LuaRocks artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".luarocks/`, `lua_modules/`, and `*.rock` at any tree depth",
        "launch checklist Lua/LuaRocks artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage",
        "launch checklist Lua/LuaRocks artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Lua files such as `*.lua`, `*.rockspec`, LuaRocks lock files, or Lua config files",
        "launch checklist Lua/LuaRocks source-file allowance scope",
    )
    assert_contains(
        launch_text,
        "tracked local Elixir/Mix build/dependency artifacts",
        "launch checklist Elixir/Mix artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".elixir_ls/`, `_build/`, and `deps/` at any tree depth",
        "launch checklist Elixir/Mix artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Elixir/Mix build/dependency artifacts",
        "launch checklist Elixir/Mix artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Elixir/Mix files such as `mix.exs`, `mix.lock`, Elixir source files, or Elixir docs",
        "launch checklist Elixir/Mix source-file allowance",
    )
    assert_contains(
        launch_text,
        "The Erlang/Rebar3 artifact guard also rejects tracked local cache and crash artifacts",
        "launch checklist Erlang/Rebar3 artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".rebar3/` at any tree depth, `rebar3.crashdump`, and `erl_crash.dump`",
        "launch checklist Erlang/Rebar3 artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Erlang/Rebar3 artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Erlang/Rebar3 files such as `rebar.config`, `rebar.lock`, Erlang source files, or Erlang header files",
        "launch checklist Erlang/Rebar3 source-file allowance",
    )
    assert_contains(
        launch_text,
        "native/CMake build artifact guard rejects tracked local native/CMake build/user-local artifacts",
        "launch checklist native/CMake artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "CTest `Testing/Temporary/` output, CPack staging output such as `_CPack_Packages/`, generated CTest/Dart files such as `CTestTestfile.cmake` and `DartConfiguration.tcl`",
        "launch checklist native/CMake artifact examples",
    )
    assert_contains(
        launch_text,
        "generated CPack files such as `CPackConfig.cmake` and `CPackSourceConfig.cmake`",
        "launch checklist native/CMake CPack artifact examples",
    )
    assert_contains(
        launch_text,
        "`CMakeCache.txt`, user-local `CMakeUserPresets.json`, `cmake_install.cmake`, `compile_commands.json`, `install_manifest.txt`",
        "launch checklist native/CMake generated local file examples",
    )
    assert_contains(
        launch_text,
        "Ninja local state files such as `.ninja_deps` and `.ninja_log`",
        "launch checklist native/CMake Ninja local state examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local native/CMake build artifacts",
        "launch checklist native/CMake artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth CMake files such as `CMakeLists.txt` and `CMakePresets.json`",
        "launch checklist native/CMake source-file allowance",
    )
    assert_contains(launch_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "launch checklist public contract artifact env")
    assert_contains(launch_text, "scripts/backend_acceptance_artifact_qa.py", "launch checklist backend acceptance artifact QA")
    assert_contains(launch_text, "backend acceptance smoke success/failure summaries", "launch checklist backend acceptance artifact QA scope")
    assert_contains(launch_text, "backend acceptance artifact summaries reject local paths, secret markers, and request/payload text", "launch checklist backend acceptance artifact share-safety scope")
    assert_contains(
        launch_text,
        "keep `summary.base_url` loopback-only and aligned between `summary.json` and `summary.md`",
        "launch checklist backend acceptance artifact base URL scope",
    )
    assert_contains(
        launch_text,
        "keep `summary.started_at` and `summary.finished_at` as RFC3339 UTC timestamps aligned between `summary.json` and `summary.md`",
        "launch checklist backend acceptance artifact timestamp scope",
    )
    assert_contains(
        launch_text,
        "keep `summary.md` artifact-index rows aligned with `summary.json` checks",
        "launch checklist backend acceptance artifact markdown index scope",
    )
    assert_contains(
        launch_text,
        "keep `repo_commit` and `fixture_model_ids` aligned between `summary.json` and `summary.md`",
        "launch checklist backend acceptance artifact identity metadata scope",
    )
    assert_contains(
        launch_text,
        "keep share-safe artifact/state/model/log/local-path labels aligned between `summary.json` and `summary.md`",
        "launch checklist backend acceptance artifact path-label scope",
    )
    assert_contains(
        launch_text,
        "keep the Markdown Port row aligned with `summary.port` and `summary.base_url`",
        "launch checklist backend acceptance artifact port-row scope",
    )
    assert_contains(
        launch_text,
        "keep failed-run diagnostics aligned between `summary.json` and `summary.md`",
        "launch checklist backend acceptance artifact failure diagnostics scope",
    )
    assert_contains(
        launch_text,
        "keep backend acceptance boundary caveats present in `summary.md`",
        "launch checklist backend acceptance artifact caveat scope",
    )
    assert_contains(
        launch_text,
        "Keep `summary.local.json` private unless you have reviewed it",
        "launch checklist backend acceptance local-path artifact privacy warning",
    )
    assert_contains(
        launch_text,
        "manually inspect logs/full payloads for local paths or request text",
        "launch checklist backend acceptance manual artifact review warning",
    )
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
    assert_latest_public_risk_scan_hardening_evidence(evidence_text)
    assert_frontend_lockfile_evidence(evidence_text)
    assert_contains(evidence_text, "repository text-normalization metadata guard", "launch evidence text-normalization metadata scope")
    assert_contains(evidence_text, "root `.gitattributes` text-normalization metadata", "launch evidence text-normalization metadata proof")
    assert_contains(
        evidence_text,
        "launch-facing relative Markdown links",
        "launch evidence local Markdown link QA scope",
    )
    assert_contains(
        evidence_text,
        "unchecked Markdown task-list items",
        "launch evidence PR template checkbox QA scope",
    )
    assert_contains(evidence_text, "Git LFS pointer-file guard", "launch evidence Git LFS pointer risk-scan scope")
    assert_contains(
        evidence_text,
        "tracked Git LFS pointer files that can hide external artifact downloads",
        "launch evidence Git LFS pointer examples",
    )
    assert_contains(
        evidence_text,
        "Git submodule metadata local/private source guard",
        "launch evidence Git submodule metadata risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked Git submodule metadata such as `.gitmodules` local/relative, SSH-only, authenticated, secret, or private source URLs",
        "launch evidence Git submodule metadata examples",
    )
    assert_contains(evidence_text, "scripts/public_contract_smoke_artifact_qa.py", "launch evidence artifact QA")
    assert_contains(evidence_text, "offline public-contract and backend acceptance artifact QA", "launch evidence backend acceptance artifact QA scope")
    assert_contains(
        evidence_text,
        "Public-contract smoke summary artifacts include commit, generated UTC timestamp, manifest path/name/status, endpoint checks, boundary checks, and scope caveats only",
        "launch evidence public contract smoke artifact field scope",
    )
    assert_contains(
        evidence_text,
        "Public-contract smoke summary artifacts intentionally omit local temp paths, server log tails, request secrets, and model/provider payloads",
        "launch evidence public contract smoke artifact share-safety scope",
    )
    assert_contains(
        evidence_text,
        "backend acceptance artifact summary share-safety guard rejects local paths, secret markers, and request/payload text",
        "launch evidence backend acceptance artifact share-safety scope",
    )
    assert_contains(
        evidence_text,
        "keeps `summary.base_url` loopback-only and aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact base URL scope",
    )
    assert_contains(
        evidence_text,
        "keeps `summary.started_at` and `summary.finished_at` as RFC3339 UTC timestamps aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact timestamp scope",
    )
    assert_contains(
        evidence_text,
        "keeps `summary.md` artifact-index rows aligned with `summary.json` checks",
        "launch evidence backend acceptance artifact markdown index scope",
    )
    assert_contains(
        evidence_text,
        "keeps `repo_commit` and `fixture_model_ids` aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact identity metadata scope",
    )
    assert_contains(
        evidence_text,
        "keeps share-safe artifact/state/model/log/local-path labels aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact path-label scope",
    )
    assert_contains(
        evidence_text,
        "keeps the Markdown Port row aligned with `summary.port` and `summary.base_url`",
        "launch evidence backend acceptance artifact port-row scope",
    )
    assert_contains(
        evidence_text,
        "keeps failed-run diagnostics aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact failure diagnostics scope",
    )
    assert_contains(
        evidence_text,
        "keeps backend acceptance boundary caveats present in `summary.md`",
        "launch evidence backend acceptance artifact caveat scope",
    )
    assert_contains(
        evidence_text,
        "keeps `docs/api/client-examples.md` aligned with the example script defaults",
        "launch evidence API client default QA scope",
    )
    assert_contains(
        evidence_text,
        "fake loopback `FATHOM_*` overrides so documented environment overrides keep reaching install, chat, and embedding request bodies",
        "launch evidence API client environment override QA scope",
    )
    assert_contains(
        evidence_text,
        "keeps the cURL quickstart dependency-light and the no-deps Python example on a standard-library import allow-list",
        "launch evidence API client dependency QA scope",
    )
    assert_contains(
        evidence_text,
        "parses POST JSON bodies so catalog installs keep reviewed repo/filename pairs",
        "launch evidence REST Client body-boundary QA scope",
    )
    assert_contains(evidence_text, "public-contract smoke Markdown/status/proof-scope row consistency", "launch evidence public smoke row QA scope")
    assert_contains(evidence_text, "manifest shape validation", "launch evidence manifest shape gate")
    assert_contains(
        evidence_text,
        "pins `docs/api/public-contract.json` identity metadata",
        "launch evidence manifest identity metadata scope",
    )
    assert_contains(
        evidence_text,
        "pins `docs/api/public-contract.json` supported endpoint inventory",
        "launch evidence manifest endpoint inventory scope",
    )
    assert_contains(
        evidence_text,
        "pins `docs/api/public-contract.json` refusal/boundary inventory",
        "launch evidence manifest boundary inventory scope",
    )
    assert_contains(
        evidence_text,
        "pins concrete JSON examples in `docs/api/v1-contract.md` to the narrow launch boundary",
        "launch evidence v1 JSON example boundary scope",
    )
    assert_contains(
        evidence_text,
        "non-contract example surfaces constrained to reviewed local catalog helper paths",
        "launch evidence non-contract example surface metadata scope",
    )
    assert_contains(evidence_text, "manifest-to-`/v1` docs boundary coverage", "launch evidence manifest docs boundary gate")
    assert_contains(evidence_text, "manifest `base_url` alignment across public docs/examples", "launch evidence manifest base URL gate")
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
        "Haskell Stack/Cabal local build artifact guard",
        "launch evidence Haskell Stack/Cabal artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local build outputs such as `.stack-work/`, `dist-newstyle/`, `.cabal-sandbox/`, and `cabal.sandbox.config` at any tree depth",
        "launch evidence Haskell Stack/Cabal artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for local Haskell Stack/Cabal build artifacts",
        "launch evidence Haskell Stack/Cabal artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Haskell files such as `.cabal` files, `stack.yaml`, `cabal.project`, and Haskell source files by filename alone",
        "launch evidence Haskell Stack/Cabal source-file allowance scope",
    )
    assert_contains(
        evidence_text,
        "local OCaml/opam artifact guard",
        "launch evidence OCaml/opam artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "local OCaml/opam artifact guard for `_opam/` and `.opam-switch/` at any tree depth",
        "launch evidence OCaml/opam current-scope examples",
    )
    assert_contains(
        evidence_text,
        "tracked local OCaml/opam local switch artifacts such as `_opam/` and `.opam-switch/` at any tree depth",
        "launch evidence OCaml/opam artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local OCaml/opam artifacts",
        "launch evidence OCaml/opam artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth OCaml project files such as `dune`, `dune-project`, `*.opam`, and OCaml source files",
        "launch evidence OCaml/opam source allowance",
    )
    assert_contains(
        evidence_text,
        "Lua/LuaRocks artifact guard",
        "launch evidence Lua/LuaRocks artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "Lua/LuaRocks local dependency/build/package artifact guard for `.luarocks/`, `lua_modules/`, and `*.rock` at any tree depth",
        "launch evidence Lua/LuaRocks current-scope examples",
    )
    assert_contains(
        evidence_text,
        "tracked local Lua/LuaRocks dependency/build/package artifacts such as `.luarocks/`, `lua_modules/`, and `*.rock` at any tree depth",
        "launch evidence Lua/LuaRocks artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Lua/LuaRocks artifacts",
        "launch evidence Lua/LuaRocks artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Lua files such as `*.lua`, `*.rockspec`, LuaRocks lock files, and Lua config files",
        "launch evidence Lua/LuaRocks source allowance",
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
        "root and nested `.gitignore` coverage for personal workspace context including local assistant state",
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
        "missing root and nested `.gitignore` coverage for local shell/REPL command history files",
        "launch evidence command history ignore examples",
    )
    assert_contains(
        launch_text,
        "tracked local shell/REPL command history files",
        "launch checklist command history risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local shell/REPL command history files",
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
        "root and nested `.gitignore` coverage for local IDE workspace/config artifacts",
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
        "root and nested `.gitignore` local IDE workspace/config artifact guard",
        "launch evidence IDE workspace/config ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "local `act` workflow runner state/config (`.act/` and `.actrc`) and GitHub Actions self-hosted runner state/config (`actions-runner/`, `_work/`, `_diag/`, `.runner`, `.credentials`, and `.credentials_rsaparams`) are blocked as tracked launch-facing material",
        "launch evidence local CI runner artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "with matching root and nested `.gitignore` coverage",
        "launch evidence local CI runner artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local IDE workspace/config artifacts",
        "launch evidence IDE workspace/config ignore examples",
    )
    assert_contains(
        evidence_text,
        "credential/config filename guard including SSH private-key filenames, `.ssh/` directories, direnv config/state, Git credential/config files, generic secret material paths, and TLS certificate/request artifacts",
        "launch evidence credential/config SSH risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local credential/config guard including SSH private-key filenames, `.ssh/`, direnv config/state, Git credential/config files, generic secret material patterns, and TLS certificate/request artifact patterns",
        "launch evidence credential/config SSH ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked cloud SDK credential/config guard",
        "launch evidence cloud SDK credential risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local cloud SDK credential/config guard",
        "launch evidence cloud SDK credential ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked cloud SDK credential/config files such as `.aws/`, `.azure/`, `.config/gcloud/`, `.boto`, `boto.cfg`, `application_default_credentials.json`, `service-account.json`, `service_account.json`, and `serviceAccountKey.json`",
        "launch evidence cloud SDK credential examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local cloud SDK credential/config files",
        "launch evidence cloud SDK credential ignore examples",
    )
    assert_contains(
        evidence_text,
        "tracked Kubernetes credential/config guard",
        "launch evidence Kubernetes credential risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local Kubernetes credential/config guard",
        "launch evidence Kubernetes credential ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked Kubernetes credential/config files such as `.kube/`, `kubeconfig`, `kubeconfig.yaml`, and `kubeconfig.yml`",
        "launch evidence Kubernetes credential examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Kubernetes credential/config files",
        "launch evidence Kubernetes credential ignore examples",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local credential/config guard",
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
        "missing root and nested `.gitignore` coverage for local credential/config files including `.git-credentials`, `.gitconfig`, `.gitconfig.local`, `/.ssh/`, SSH private-key filenames, `/.direnv/`, `.envrc`, `/secrets/`, `/private/`, and generic secret material patterns",
        "launch evidence credential/config ignore examples",
    )
    assert_contains(
        launch_text,
        "tracked credential/config filenames including SSH private-key filenames, `.ssh/` directories, direnv config/state, Git credential/config files, generic secret material paths, TLS certificate/request artifacts, and Java/Android/Apple signing key material",
        "launch checklist credential/config SSH risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local credential/config files including SSH private-key filenames, `.ssh/`, direnv config/state, Git credential/config files, generic secret material patterns, TLS certificate/request artifact patterns, and Java/Android/Apple signing key material patterns",
        "launch checklist credential/config SSH ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked local cloud SDK credential/config files",
        "launch checklist cloud SDK credential risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local cloud SDK credential/config files",
        "launch checklist cloud SDK credential ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked Kubernetes credential/config files",
        "launch checklist Kubernetes credential risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Kubernetes credential/config files",
        "launch checklist Kubernetes credential ignore scope",
    )
    assert_contains(
        evidence_text,
        "local runtime/artifact detail-file guard",
        "launch evidence runtime artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local runtime/artifact detail files",
        "launch checklist runtime artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local runtime/artifact detail-file guard",
        "launch evidence runtime artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local runtime/artifact detail files",
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
        launch_text,
        "SonarQube/SonarScanner local analysis outputs such as `.scannerwork/` and `.sonar/`",
        "launch checklist code-quality scanner artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not treat source-of-truth scanner configuration such as `sonar-project.properties` as a local artifact",
        "launch checklist code-quality scanner config allowance",
    )
    assert_contains(
        evidence_text,
        "SonarQube/SonarScanner local analysis outputs (`.scannerwork/`, `.sonar/`) as local code-quality scanner artifacts",
        "launch evidence code-quality scanner artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth scanner configuration such as `sonar-project.properties`",
        "launch evidence code-quality scanner config allowance",
    )
    assert_contains(
        launch_text,
        "Semgrep local cache/state directories such as `.semgrep/`",
        "launch checklist Semgrep code-quality scanner artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not treat source-of-truth Semgrep configuration such as `.semgrep.yml` or `semgrep.yml` as a local artifact",
        "launch checklist Semgrep config allowance",
    )
    assert_contains(
        evidence_text,
        "Semgrep local cache/state directories (`.semgrep/`) as local code-quality scanner artifacts",
        "launch evidence Semgrep code-quality scanner artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Semgrep configuration such as `.semgrep.yml` and `semgrep.yml`",
        "launch evidence Semgrep config allowance",
    )
    assert_contains(
        launch_text,
        "Python static-analysis report outputs from Ruff, Pylint, Mypy, and Pyright",
        "launch checklist Python static-analysis report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `ruff-report/`, `ruff-report.json`, `pylint-report.txt`, `mypy-report/`, `mypy-results.xml`, `pyright-report.json`, and timestamped `*-report-*` or `*-results-*` files",
        "launch checklist Python static-analysis report artifact examples",
    )
    assert_contains(
        launch_text,
        "preserving source-of-truth Python analysis configuration such as `pyproject.toml`, `ruff.toml`, `.ruff.toml`, `.pylintrc`, `mypy.ini`, and `pyrightconfig.json`",
        "launch checklist Python static-analysis report config allowance",
    )
    assert_contains(
        evidence_text,
        "Python static-analysis report outputs from Ruff, Pylint, Mypy, and Pyright (`ruff-report/`, `ruff-report.json`, `pylint-report.txt`, `mypy-report/`, `mypy-results.xml`, `pyright-report.json`, and timestamped `*-report-*` or `*-results-*` files) as local code-quality report artifacts",
        "launch evidence Python static-analysis report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Python analysis configuration such as `pyproject.toml`, `ruff.toml`, `.ruff.toml`, `.pylintrc`, `mypy.ini`, and `pyrightconfig.json`",
        "launch evidence Python static-analysis report config allowance",
    )
    assert_contains(
        launch_text,
        "TypeScript type-check report outputs such as `tsc-report/`, `tsc-report.txt`, `typecheck-report/`, `typecheck-results.json`, `typescript-report.xml`, and timestamped `tsc-report-*`, `typecheck-results-*`, or `typescript-results-*` files",
        "launch checklist TypeScript type-check report artifact examples",
    )
    assert_contains(
        launch_text,
        "preserving source-of-truth TypeScript project configuration such as `tsconfig.json`, `tsconfig.*.json`, frontend package manifests, and JavaScript/TypeScript source files",
        "launch checklist TypeScript type-check report config allowance",
    )
    assert_contains(
        evidence_text,
        "TypeScript type-check report artifact guard for `tsc-report/`, `tsc-report.txt`, `typecheck-report/`, `typecheck-results.json`, `typescript-report.xml`, and timestamped `tsc-report-*`, `typecheck-results-*`, or `typescript-results-*` files",
        "launch evidence TypeScript type-check report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth TypeScript project configuration such as `tsconfig.json`, `tsconfig.*.json`, frontend package manifests, and JavaScript/TypeScript source files",
        "launch evidence TypeScript type-check report config allowance",
    )
    assert_contains(
        evidence_text,
        "Python cache/build artifact guard",
        "launch evidence Python artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "Pyre/Pytype and mypy daemon state",
        "launch evidence Python type-checker artifact scope",
    )
    assert_contains(
        evidence_text,
        "Python package metadata/build artifacts such as `.eggs/`, `*.egg-info/`, and `*.dist-info/`",
        "launch evidence Python package metadata artifact scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local Python cache/build artifact guard",
        "launch evidence Python artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Python cache/build artifacts",
        "launch evidence Python artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Python virtualenv/dependency artifact guard",
        "launch evidence Python virtualenv/dependency artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local Python virtualenv/dependency artifact guard",
        "launch evidence Python virtualenv/dependency artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked Python virtualenv/dependency artifacts such as `.venv/`, `venv/`, `env/`, `.tox/`, `.nox/`, `wheelhouse/`, `pip-wheel-metadata/`, and `site-packages/`",
        "launch evidence Python virtualenv/dependency artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Python virtualenv/dependency artifacts",
        "launch evidence Python virtualenv/dependency artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Python files such as `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements*.txt`, `Pipfile`, `Pipfile.lock`, `poetry.lock`, `uv.lock`, Python source files, and Python docs",
        "launch evidence Python source-file allowance",
    )
    assert_contains(
        evidence_text,
        "Python benchmark artifact guard",
        "launch evidence Python benchmark artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local pytest-benchmark artifacts such as `.benchmarks/` at any tree depth and `pytest-benchmark.json` exports",
        "launch evidence Python benchmark artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Python benchmark artifacts",
        "launch evidence Python benchmark artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving curated benchmark evidence under `docs/benchmarks/` and source helpers such as `scripts/bench_backend.py`",
        "launch evidence Python benchmark source/evidence allowance",
    )
    assert_contains(
        evidence_text,
        "frontend/Node cache/build artifact guard",
        "launch evidence frontend artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "Vite/Vitest cache/config timestamp artifacts",
        "launch evidence Vite/Vitest frontend artifact scope",
    )
    assert_contains(
        evidence_text,
        "Storybook static build output",
        "launch evidence Storybook frontend artifact scope",
    )
    assert_contains(
        evidence_text,
        "Astro/Docusaurus/Hugo/Jekyll/VitePress/Metro/Nx/SWC/Next.js/SvelteKit/Nuxt/Angular framework cache/build outputs such as `.astro/`, `.docusaurus/`, `resources/_gen/`, `.hugo_build.lock`, `_site/`, `.jekyll-cache/`, `.sass-cache/`, `.vite/`, `.vitepress/cache/`, `.vitepress/dist/`, `.metro-cache/`, `.nx/`, `.swc/`, Rollup TypeScript plugin `.rpt2_cache/`, Webpack `.webpack-cache/`, `.next/`, `.svelte-kit/`, `.nuxt/`, `.output/`, and `.angular/cache/`",
        "launch evidence frontend framework artifact scope",
    )
    assert_contains(
        evidence_text,
        "Lighthouse/LHCI audit outputs such as `.lighthouseci/`, `lhci_reports/`, `lighthouse-report.html`, `lighthouse-report.json`, and timestamped `lighthouse-report-*` files",
        "launch evidence Lighthouse/LHCI artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root, nested, and `frontend/` `.gitignore` coverage for generated Lighthouse/LHCI audit outputs",
        "launch evidence Lighthouse/LHCI artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Lighthouse config files such as `.lighthouserc.*`",
        "launch evidence Lighthouse config allowance",
    )
    assert_contains(
        launch_text,
        "Axe/Pa11y accessibility audit outputs such as `axe-report/`, `axe-reports/`, `axe-report.html`, `axe-results.json`, `pa11y-report/`, `pa11y-reports/`, `pa11y-ci-report.json`, and `pa11y-results.json`",
        "launch checklist Axe/Pa11y accessibility report artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated accessibility reports",
        "launch checklist Axe/Pa11y accessibility report artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "source-of-truth accessibility configuration such as `axe.config.*`, `.pa11yci`, or `pa11yci.json`",
        "launch checklist Axe/Pa11y config allowance",
    )
    assert_contains(
        evidence_text,
        "Axe/Pa11y accessibility audit report guard rejects tracked local accessibility report outputs such as `axe-report/`, `axe-reports/`, `axe-report.html`, `axe-results.json`, `pa11y-report/`, `pa11y-reports/`, `pa11y-ci-report.json`, and `pa11y-results.json`",
        "launch evidence Axe/Pa11y accessibility report artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for generated accessibility reports",
        "launch evidence Axe/Pa11y accessibility report artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth accessibility configuration such as `axe.config.*`, `.pa11yci`, and `pa11yci.json`",
        "launch evidence Axe/Pa11y config allowance",
    )
    assert_contains(
        evidence_text,
        "ESLint report artifact guard rejects tracked local lint report outputs such as `eslint-report/`, `eslint-reports/`, `eslint-junit.xml`, `eslint-report.json`, `eslint-report.xml`, `eslint-results.json`, and timestamped `eslint-report-*` or `eslint-results-*` files",
        "launch evidence ESLint report artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for generated ESLint reports",
        "launch evidence ESLint report artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth ESLint configuration such as `eslint.config.*` and `.eslintrc.*`",
        "launch evidence ESLint config allowance",
    )
    assert_contains(
        evidence_text,
        "Stylelint report artifact guard rejects tracked local lint report outputs such as `stylelint-report/`, `stylelint-reports/`, `stylelint-junit.xml`, `stylelint-report.json`, `stylelint-report.xml`, `stylelint-results.json`, and timestamped `stylelint-report-*` or `stylelint-results-*` files",
        "launch evidence Stylelint report artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for generated Stylelint reports",
        "launch evidence Stylelint report artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Stylelint configuration such as `stylelint.config.*` and `.stylelintrc*`",
        "launch evidence Stylelint config allowance",
    )
    assert_contains(
        evidence_text,
        "Biome report artifact guard rejects tracked local lint/formatter report outputs such as `biome-report/`, `biome-reports/`, `biome-report.html`, `biome-report.json`, `biome-report.xml`, `biome-results.json`, and timestamped `biome-report-*` or `biome-results-*` files",
        "launch evidence Biome report artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for generated Biome reports",
        "launch evidence Biome report artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Biome configuration such as `biome.json` and `biome.jsonc`",
        "launch evidence Biome config allowance",
    )
    assert_contains(
        evidence_text,
        "Markdown/prose lint report artifact guard rejects tracked local documentation lint report outputs such as `markdownlint-report/`, `markdownlint-reports/`, `markdownlint-junit.xml`, `markdownlint-report.json`, `markdownlint-results.xml`, `remark-report/`, `remark-reports/`, `remark-junit.xml`, `remark-report.json`, `remark-results.xml`, `proselint-report/`, `proselint-reports/`, `proselint-junit.xml`, `proselint-report.json`, and `proselint-results.xml`",
        "launch evidence Markdown/prose lint report artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for generated Markdown/prose lint reports",
        "launch evidence Markdown/prose lint report artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Markdown/prose lint configuration such as `.markdownlint.*`, `.remarkrc`, `remark.config.*`, and `.proselintrc`",
        "launch evidence Markdown/prose lint config allowance",
    )
    assert_contains(
        launch_text,
        "security/dependency scan report outputs such as `snyk-report/`, `snyk-reports/`, `cargo-audit-report/`, `cargo-deny-report/`, `npm-audit-report/`, `pnpm-audit-report/`, `yarn-audit-report/`, `osv-scanner-report/`, `trivy-report/`, `grype-report/`, `dependency-check-report/`, `snyk-report.json`, `cargo-audit-results.xml`, `cargo-deny-results.txt`, `npm-audit.json`, `osv-scanner-results.json`, `trivy-results.json`, `grype-results.txt`, and `dependency-check-report.html`",
        "launch checklist security/dependency scan report artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated security/dependency scan reports",
        "launch checklist security/dependency scan report artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "source-of-truth dependency lockfiles or scanner configuration such as `Cargo.lock`, `package-lock.json`, `.snyk`, `deny.toml`, `cargo-deny.toml`, `osv-scanner.toml`, `trivy.yaml`, `grype.yaml`, or `dependency-check.properties`",
        "launch checklist security/dependency scan config allowance",
    )
    assert_contains(
        evidence_text,
        "security/dependency scan report artifact guard for generated Snyk, Cargo Audit, Cargo Deny, npm audit, pnpm audit, Yarn audit, OSV Scanner, Trivy, Grype, and OWASP Dependency-Check outputs",
        "launch evidence security/dependency scan report artifact scope",
    )
    assert_contains(
        evidence_text,
        "with matching root and nested `.gitignore` coverage while preserving source-of-truth lockfiles and scanner configuration such as `Cargo.lock`, `package-lock.json`, `.snyk`, `deny.toml`, `cargo-deny.toml`, `osv-scanner.toml`, `trivy.yaml`, `grype.yaml`, and `dependency-check.properties`",
        "launch evidence security/dependency scan report artifact ignore and config allowance",
    )
    assert_contains(
        launch_text,
        "generated SBOM/license inventory outputs such as `sbom/`, `sboms/`, `cyclonedx/`, `cyclonedx-reports/`, `license-report/`, `license-reports/`, `licenses-report/`, `licenses-reports/`, `syft-report/`, `syft-reports/`, `sbom.json`, `sbom.spdx.json`, `bom.json`, `cyclonedx.json`, `syft-report.json`, and `license-report.json`",
        "launch checklist SBOM/license inventory artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for generated SBOM/license inventory outputs",
        "launch checklist SBOM/license inventory ignore examples",
    )
    assert_contains(
        launch_text,
        "source-of-truth SBOM or inventory tool configuration such as `cdxgen.yml`, `.cdxgenrc`, `syft.yaml`, and `.syft.yaml`",
        "launch checklist SBOM/license inventory config allowance",
    )
    assert_contains(
        evidence_text,
        "SBOM/license inventory artifact guard for generated CycloneDX, SPDX, Syft, and license inventory outputs",
        "launch evidence SBOM/license inventory artifact scope",
    )
    assert_contains(
        evidence_text,
        "with matching root and nested `.gitignore` coverage while preserving source-of-truth SBOM or inventory tool configuration such as `cdxgen.yml`, `.cdxgenrc`, `syft.yaml`, and `.syft.yaml`",
        "launch evidence SBOM/license inventory ignore and config allowance",
    )
    assert_contains(
        launch_text,
        "Rollup TypeScript plugin `.rpt2_cache/`",
        "launch checklist Rollup TypeScript plugin cache scope",
    )
    assert_contains(
        launch_text,
        "Webpack `.webpack-cache/`",
        "launch checklist Webpack cache scope",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth lockfiles, package manifests, and Metro configuration such as `metro.config.js`",
        "launch evidence Metro source allowance",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local frontend/Node cache/build artifact guard",
        "launch evidence frontend artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local frontend/Node cache/build artifacts",
        "launch evidence frontend artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "frontend bundle-analysis outputs such as `bundle-analyzer-report.html`, `bundle-analysis.json`, `bundle-stats.html`, `webpack-stats.json`, `rollup-stats.html`, and `vite-bundle-visualizer.html`",
        "launch checklist frontend bundle-analysis artifact examples",
    )
    assert_contains(
        launch_text,
        "It preserves source-of-truth bundler configuration and package metadata such as `webpack.config.*`, `rollup.config.*`, `vite.config.*`, and frontend package manifests",
        "launch checklist frontend bundle-analysis source allowance",
    )
    assert_contains(
        evidence_text,
        "frontend bundle-analysis output guard for `bundle-analyzer-report.html`, `bundle-analysis.json`, `bundle-stats.html`, `webpack-stats.json`, `rollup-stats.html`, `vite-bundle-visualizer.html`, and name-suffixed variants",
        "launch evidence frontend bundle-analysis artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for those generated reports",
        "launch evidence frontend bundle-analysis ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth bundler configuration and package metadata such as `webpack.config.*`, `rollup.config.*`, `vite.config.*`, and frontend package manifests",
        "launch evidence frontend bundle-analysis source allowance",
    )
    assert_contains(
        launch_text,
        "tracked local temporary/scratch artifacts",
        "launch checklist temporary/scratch artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "tracked Watchman local state cookies such as `.watchman-cookie` and `.watchman-cookie-*`",
        "launch checklist Watchman local cache artifact scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Watchman configuration such as `.watchmanconfig`",
        "launch checklist Watchman source allowance",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local temporary/scratch artifacts",
        "launch checklist temporary/scratch artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "local temporary/scratch artifact guard",
        "launch evidence temporary/scratch artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local temporary/scratch artifact guard",
        "launch evidence temporary/scratch artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local temporary/scratch artifacts such as `tmp/`, `temp/`, `*.tmp`, and `*.temp`",
        "launch evidence temporary/scratch artifact examples",
    )
    assert_contains(
        evidence_text,
        "tracked Watchman local state cookies such as `.watchman-cookie` and `.watchman-cookie-*`",
        "launch evidence Watchman local cache artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for those local Watchman state artifacts",
        "launch evidence Watchman local cache artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "source-of-truth Watchman configuration such as `.watchmanconfig`",
        "launch evidence Watchman source allowance",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local temporary/scratch artifacts",
        "launch evidence temporary/scratch artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Rust/Cargo cache/build artifact guard",
        "launch evidence Rust artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked Rust/Cargo cache/build artifacts such as `.cargo/`, `target/`, compiler outputs, and coverage profiles",
        "launch evidence Rust artifact examples",
    )
    assert_contains(
        evidence_text,
        "Playwright/browser-test report directories such as `playwright-report/`, `blob-report/`, and `.playwright/`",
        "launch evidence Playwright/browser-test report examples",
    )
    assert_contains(
        launch_text,
        "browser-test artifact guard also blocks Cypress screenshot, video, and download output directories",
        "launch checklist Cypress browser-test artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "in addition to Playwright report directories",
        "launch checklist Cypress browser-test artifact Playwright relation",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for Cypress outputs",
        "launch checklist Cypress browser-test artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Mochawesome report outputs such as `mochawesome-report/`, `mochawesome.html`, and `mochawesome.json` as local test report artifacts",
        "launch checklist Mochawesome test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Mochawesome test report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Cucumber/BDD report outputs such as `cucumber-report/`, `cucumber-reports/`, `cucumber-report.html`, `cucumber-report.ndjson`, `cucumber.json`, `cucumber-report.json`, `cucumber-report.xml`, and `cucumber.xml` as local test report artifacts",
        "launch checklist Cucumber/BDD test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage where directory-shaped",
        "launch checklist Cucumber/BDD test report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "standalone xUnit XML report outputs such as `xunit.xml` and `*.xunit.xml` as local test report artifacts",
        "launch checklist standalone xUnit XML test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage",
        "launch checklist standalone xUnit XML test report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "standalone JUnit XML suite report outputs such as `TEST-*.xml` as local test report artifacts",
        "launch checklist standalone JUnit XML suite report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage for generated JUnit suite report files",
        "launch checklist standalone JUnit XML suite report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "standalone JUnit XML aggregate suite report outputs such as `TESTS-*.xml` as local test report artifacts",
        "launch checklist standalone JUnit XML aggregate suite report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage for generated JUnit aggregate suite report files",
        "launch checklist standalone JUnit XML aggregate suite report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "JUnit report directories and aggregate result files such as `junit-report/`, `junit-reports/`, and `junit-results.xml` as local test report artifacts",
        "launch checklist JUnit report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for JUnit report directories and root `.gitignore` coverage for generated JUnit result files",
        "launch checklist JUnit report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "TestNG generated report outputs such as `test-output/`, `testng-results.xml`, `testng-failed.xml`, and `emailable-report.html` as local test report artifacts",
        "launch checklist TestNG test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for `test-output/` and root `.gitignore` coverage for generated TestNG report files",
        "launch checklist TestNG test report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "VSTest/.NET test result outputs such as `TestResults/`, `*.trx`, `*.coverage`, and `*.coveragexml` as local test report artifacts",
        "launch checklist VSTest/.NET test result artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for `TestResults/` and root `.gitignore` coverage for generated report files",
        "launch checklist VSTest/.NET test result artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "NUnit XML report outputs such as `nunit.xml`, `*.nunit.xml`, `TestResult.xml`, and `TestResult-*.xml` as local test report artifacts",
        "launch checklist NUnit XML test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage for generated NUnit report files",
        "launch checklist NUnit XML test report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Go test runner report exports such as `gotestsum.json`, `gotestsum-report.xml`, `go-test-report.json`, and `go-test-results.xml` as local test report artifacts",
        "launch checklist Go test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage for generated Go test report files",
        "launch checklist Go test report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Allure report/result directories as local test report artifacts",
        "launch checklist Allure test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "Postman/Newman API test report outputs such as `newman/`, `newman-report.html`, `newman-report.json`, and `newman-report.xml` as local test report artifacts",
        "launch checklist Newman API test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "Robot Framework result/report outputs such as `robot-results/`, `robot-reports/`, `robot-output.xml`, `robot-log.html`, and `robot-report.html` as local test report artifacts",
        "launch checklist Robot Framework test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not block generic default names such as `output.xml`, `log.html`, or `report.html` outside Robot-specific output directories",
        "launch checklist Robot Framework generic report allowance",
    )
    assert_contains(
        launch_text,
        "Apache JMeter generated result/report outputs such as `jmeter-report/`, `jmeter-reports/`, `jmeter-results/`, and `*.jtl` as local test report artifacts",
        "launch checklist JMeter test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not block source-of-truth JMeter test plans such as `*.jmx`",
        "launch checklist JMeter source test plan allowance",
    )
    assert_contains(
        launch_text,
        "Jest generated report outputs such as `jest-stare/`, `jest-html-reporters-attach/`, `jest_html_reporters.html`, `jest-junit.xml`, and `jest-results.json` as local test report artifacts",
        "launch checklist Jest test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not block source-of-truth Jest configuration files such as `jest.config.js`",
        "launch checklist Jest config allowance",
    )
    assert_contains(
        launch_text,
        "Vitest generated report outputs such as `vitest-report/`, `vitest-report.html`, `vitest-report.json`, `vitest-report-*` variants, `vitest-junit.xml`, and `vitest-results.json` as local test report artifacts",
        "launch checklist Vitest test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not block source-of-truth Vitest configuration files such as `vitest.config.js` or `vitest.config.ts`",
        "launch checklist Vitest config allowance",
    )
    assert_contains(
        launch_text,
        "TAP generated report outputs such as `tap-report/`, `tap-reports/`, `tap-report.html`, `tap-report.json`, `tap-report.xml`, `tap-report.tap`, and `tap-results.tap` as local test report artifacts",
        "launch checklist TAP test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not block arbitrary `*.tap` files outside explicit TAP report names or directories",
        "launch checklist TAP source/data allowance",
    )
    assert_contains(
        launch_text,
        "JaCoCo coverage outputs as local test report artifacts",
        "launch checklist JaCoCo test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "Cobertura/OpenCover coverage XML outputs (`cobertura.xml`, `*.cobertura.xml`, `opencover.xml`, and `*.opencover.xml`) as local test report artifacts",
        "launch checklist Cobertura test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "LCOV HTML report directories such as `lcov-report/` at any tree depth",
        "launch checklist LCOV HTML coverage artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for LCOV HTML report directories",
        "launch checklist LCOV HTML coverage artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "Istanbul/nyc coverage outputs (`.nyc_output/`, `coverage-final.json`, `coverage-summary.json`, `clover.xml`, and `*.clover.xml`) as local test report artifacts",
        "launch checklist Istanbul/nyc coverage artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for `.nyc_output/` and root `.gitignore` coverage for generated Istanbul/nyc report files",
        "launch checklist Istanbul/nyc coverage artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "coverage.py JSON report output (`coverage.json`) as a local coverage artifact",
        "launch checklist coverage.py JSON artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage",
        "launch checklist coverage.py JSON artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "coverage upload JSON exports such as `codecov.json`, `coveralls.json`, `*.codecov.json`, and `*.coveralls.json` as local coverage artifacts",
        "launch checklist coverage upload JSON artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "does not treat service configuration files such as `codecov.yml`, `.codecov.yml`, or `.coveralls.yml` as local report artifacts",
        "launch checklist coverage upload config allowance",
    )
    assert_contains(
        launch_text,
        "cargo-tarpaulin coverage report outputs (`tarpaulin-report.html` and `tarpaulin-report.json`) as local test report artifacts",
        "launch checklist cargo-tarpaulin test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "native compiler coverage outputs (`*.gcda`, `*.gcno`, and `*.gcov`) as local test report artifacts",
        "launch checklist native compiler coverage artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "pytest generated report outputs such as `.report.json`, `pytest-report.html`, `pytest-report.json`, `pytest-report.xml`, and timestamped or named `pytest-report-*` variants",
        "launch checklist pytest report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root `.gitignore` coverage",
        "launch checklist pytest report artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "mutation-testing report/cache outputs such as `.mutmut-cache/`, `.stryker-tmp/`, `mutation-report/`, `pit-reports/`, `mutation-report.html`, and `mutmut.sqlite`",
        "launch checklist mutation-testing artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist mutation-testing artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "CTRF generated test report outputs such as `ctrf/`, `ctrf-report.json`, and `ctrf-report-*.json` as local test report artifacts",
        "launch checklist CTRF test report artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage for report directories and root `.gitignore` coverage for generated CTRF report files",
        "launch checklist CTRF test report artifact ignore scope",
    )
    assert_contains(
        evidence_text,
        "Cypress screenshot, video, and download output directories at any tree depth",
        "launch evidence Cypress browser-test artifact examples",
    )
    assert_contains(
        evidence_text,
        "Mochawesome report outputs (`mochawesome-report/`, `mochawesome.html`, `mochawesome.json`)",
        "launch evidence Mochawesome test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "Cucumber/BDD report outputs (`cucumber-report/`, `cucumber-reports/`, `cucumber-report.html`, `cucumber-report.ndjson`, `cucumber.json`, `cucumber-report.json`, `cucumber-report.xml`, `cucumber.xml`)",
        "launch evidence Cucumber/BDD test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "standalone xUnit XML report outputs (`xunit.xml`, `*.xunit.xml`)",
        "launch evidence standalone xUnit XML test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "standalone JUnit XML suite report outputs (`TEST-*.xml`)",
        "launch evidence standalone JUnit XML suite report artifact examples",
    )
    assert_contains(
        evidence_text,
        "standalone JUnit XML aggregate suite report outputs (`TESTS-*.xml`)",
        "launch evidence standalone JUnit XML aggregate suite report artifact examples",
    )
    assert_contains(
        evidence_text,
        "JUnit report directories and aggregate result files (`junit-report/`, `junit-reports/`, `junit-results.xml`)",
        "launch evidence JUnit report artifact examples",
    )
    assert_contains(
        evidence_text,
        "TestNG generated report outputs (`test-output/`, `testng-results.xml`, `testng-failed.xml`, `emailable-report.html`)",
        "launch evidence TestNG test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "VSTest/.NET test result outputs (`TestResults/`, `*.trx`, `*.coverage`, `*.coveragexml`)",
        "launch evidence VSTest/.NET test result artifact examples",
    )
    assert_contains(
        evidence_text,
        "NUnit XML report outputs (`nunit.xml`, `*.nunit.xml`, `TestResult.xml`, `TestResult-*.xml`)",
        "launch evidence NUnit XML test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "Go test runner report exports (`gotestsum.json`, `gotestsum-report.xml`, `go-test-report.json`, `go-test-results.xml`)",
        "launch evidence Go test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "Allure report/result directories (`allure-report/`, `allure-results/`)",
        "launch evidence Allure test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "Postman/Newman API test report outputs (`newman/`, `newman-report.html`, `newman-report.json`, `newman-report.xml`)",
        "launch evidence Newman API test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "Robot Framework result/report outputs (`robot-results/`, `robot-reports/`, `robot-output.xml`, `robot-log.html`, `robot-report.html`)",
        "launch evidence Robot Framework test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving generic default names outside Robot-specific directories",
        "launch evidence Robot Framework generic report allowance",
    )
    assert_contains(
        evidence_text,
        "Apache JMeter generated result/report outputs (`jmeter-report/`, `jmeter-reports/`, `jmeter-results/`, `*.jtl`)",
        "launch evidence JMeter test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth JMeter test plans such as `*.jmx`",
        "launch evidence JMeter source test plan allowance",
    )
    assert_contains(
        evidence_text,
        "Jest generated report outputs (`jest-stare/`, `jest-html-reporters-attach/`, `jest_html_reporters.html`, `jest-junit.xml`, `jest-results.json`)",
        "launch evidence Jest test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Jest configuration such as `jest.config.js`",
        "launch evidence Jest config allowance",
    )
    assert_contains(
        evidence_text,
        "Vitest generated report outputs (`vitest-report/`, `vitest-report.html`, `vitest-report.json`, `vitest-report-*` variants, `vitest-junit.xml`, `vitest-results.json`)",
        "launch evidence Vitest test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Vitest configuration such as `vitest.config.js` and `vitest.config.ts`",
        "launch evidence Vitest config allowance",
    )
    assert_contains(
        evidence_text,
        "TAP generated report outputs (`tap-report/`, `tap-reports/`, `tap-report.html`, `tap-report.json`, `tap-report.xml`, `tap-report.tap`, `tap-results.tap`)",
        "launch evidence TAP test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving arbitrary `*.tap` files outside explicit TAP report names or directories",
        "launch evidence TAP source/data allowance",
    )
    assert_contains(
        evidence_text,
        "JaCoCo coverage outputs (`jacocoHtml/`, `jacoco.exec`, `jacoco.xml`, `jacoco.csv`)",
        "launch evidence JaCoCo test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "Cobertura/OpenCover coverage XML outputs (`cobertura.xml`, `*.cobertura.xml`, `opencover.xml`, `*.opencover.xml`)",
        "launch evidence Cobertura test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "coverage.py JSON report output (`coverage.json`)",
        "launch evidence coverage.py JSON artifact examples",
    )
    assert_contains(
        evidence_text,
        "coverage upload JSON exports (`codecov.json`, `coveralls.json`, `*.codecov.json`, `*.coveralls.json`)",
        "launch evidence coverage upload JSON artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving service configuration files such as `codecov.yml`, `.codecov.yml`, and `.coveralls.yml`",
        "launch evidence coverage upload config allowance",
    )
    assert_contains(
        evidence_text,
        "LCOV HTML report directories (`lcov-report/`)",
        "launch evidence LCOV HTML coverage artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects missing root and nested `.gitignore` coverage for those local coverage report artifacts",
        "launch evidence LCOV HTML coverage artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "cargo-tarpaulin coverage report outputs (`tarpaulin-report.html`, `tarpaulin-report.json`)",
        "launch evidence cargo-tarpaulin test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "native compiler coverage outputs such as `*.gcda`, `*.gcno`, and `*.gcov`",
        "launch evidence native compiler coverage artifact examples",
    )
    assert_contains(
        evidence_text,
        "pytest generated report outputs (`.report.json`, `pytest-report.html`, `pytest-report.json`, `pytest-report.xml`, and `pytest-report-*` variants)",
        "launch evidence pytest report artifact examples",
    )
    assert_contains(
        evidence_text,
        "mutation-testing report/cache outputs (`.mutmut-cache/`, `.stryker-tmp/`, `mutation-report/`, `pit-reports/`, `mutation-report.html`, and `mutmut.sqlite`)",
        "launch evidence mutation-testing artifact examples",
    )
    assert_contains(
        evidence_text,
        "CTRF generated test report outputs (`ctrf/`, `ctrf-report.json`, `ctrf-report-*.json`)",
        "launch evidence CTRF test report artifact examples",
    )
    assert_contains(
        evidence_text,
        "NYC/Istanbul coverage artifacts such as `.nyc_output/`, `coverage-final.json`, `coverage-summary.json`, `clover.xml`, and `*.clover.xml`",
        "launch evidence JS coverage artifact examples",
    )
    assert_contains(
        evidence_text,
        "coverage.py JSON report output (`coverage.json`)",
        "launch evidence coverage.py JSON hardening detail",
    )
    assert_contains(
        evidence_text,
        "coverage upload JSON exports (`codecov.json`, `coveralls.json`, `*.codecov.json`, `*.coveralls.json`)",
        "launch evidence coverage upload JSON hardening detail",
    )
    assert_contains(
        evidence_text,
        "pytest generated report outputs (`.report.json`, `pytest-report.html`, `pytest-report.json`, `pytest-report.xml`, and `pytest-report-*` variants)",
        "launch evidence pytest report hardening detail",
    )
    assert_contains(
        evidence_text,
        "Deno local cache artifact guard",
        "launch evidence Deno artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Deno cache artifacts such as `.deno/` and `deno-dir/`",
        "launch evidence Deno artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Deno cache artifacts",
        "launch evidence Deno artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "blocks tracked local `.cache/` directories at any tree depth",
        "launch checklist local cache artifact examples",
    )
    assert_contains(
        evidence_text,
        "rejects tracked local `.cache/` directories at any tree depth",
        "launch evidence local cache artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local cache artifacts",
        "launch evidence local cache artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Deno project files such as `deno.json`, `deno.jsonc`, `deno.lock`, JavaScript files, and TypeScript files",
        "launch evidence Deno source allowance",
    )
    assert_contains(
        evidence_text,
        "root and nested `.gitignore` local Rust/Cargo cache/build artifact guard",
        "launch evidence Rust artifact ignore risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Rust/Cargo cache/build artifacts",
        "launch evidence Rust artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Go cache/test artifact guard",
        "launch evidence Go artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Go cache/test artifacts such as `.gocache/` and `.gomodcache/` at any tree depth, `cover.out`, `coverage.out`, `*.coverprofile`, and `*.test`",
        "launch evidence Go artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Go cache/test artifacts",
        "launch evidence Go artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Go files such as `go.mod`, `go.sum`, Go source files, and Go docs",
        "launch evidence Go source-file allowance",
    )
    assert_contains(
        evidence_text,
        "Elixir/Mix build/dependency artifact guard",
        "launch evidence Elixir/Mix artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Elixir/Mix build/dependency artifacts such as `.elixir_ls/`, `_build/`, and `deps/` at any tree depth",
        "launch evidence Elixir/Mix artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Elixir/Mix build/dependency artifacts",
        "launch evidence Elixir/Mix artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Elixir/Mix files such as `mix.exs`, `mix.lock`, Elixir source files, and Elixir docs",
        "launch evidence Elixir/Mix source-file allowance",
    )
    assert_contains(
        launch_text,
        "The Erlang/Rebar3 artifact guard also rejects tracked local cache and crash artifacts",
        "launch checklist Erlang/Rebar3 artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        ".rebar3/` at any tree depth, `rebar3.crashdump`, and `erl_crash.dump`",
        "launch checklist Erlang/Rebar3 artifact examples",
    )
    assert_contains(
        launch_text,
        "matching root and nested `.gitignore` coverage",
        "launch checklist Erlang/Rebar3 artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "source-of-truth Erlang/Rebar3 files such as `rebar.config`, `rebar.lock`, Erlang source files, or Erlang header files",
        "launch checklist Erlang/Rebar3 source allowance",
    )
    assert_contains(
        evidence_text,
        "Erlang/Rebar3 artifact guard rejects tracked local cache and crash artifacts such as `.rebar3/` at any tree depth, `rebar3.crashdump`, and `erl_crash.dump`",
        "launch evidence Erlang/Rebar3 artifact examples",
    )
    assert_contains(
        evidence_text,
        "Erlang/Rebar3 local cache/crash artifact guard for `.rebar3/`, `rebar3.crashdump`, and `erl_crash.dump`",
        "launch evidence Erlang/Rebar3 current-scope examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Erlang/Rebar3 artifacts",
        "launch evidence Erlang/Rebar3 artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Erlang/Rebar3 files such as `rebar.config`, `rebar.lock`, Erlang source files, and Erlang header files",
        "launch evidence Erlang/Rebar3 source allowance",
    )
    assert_contains(
        evidence_text,
        "JVM dependency artifact guard",
        "launch evidence JVM dependency artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local JVM dependency artifacts such as `.m2/` at any tree depth",
        "launch evidence JVM dependency artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local JVM dependency artifacts",
        "launch evidence JVM dependency artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Maven/JVM project files such as `pom.xml`, Java source files, Kotlin source files, Scala source files, and JVM docs",
        "launch evidence JVM dependency source allowance",
    )
    assert_contains(
        launch_text,
        "The Scala/SBT build artifact guard also rejects tracked local Scala build server/IDE outputs",
        "launch checklist Scala/SBT artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.bloop/`, `.bsp/`, `.metals/`, and `.scala-build/` at any tree depth",
        "launch checklist Scala/SBT artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Scala/SBT files",
        "launch checklist Scala/SBT artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "It does not treat source-of-truth Scala/SBT files such as `build.sbt`, `project/build.properties`, or Scala source files as local tool artifacts.",
        "launch checklist Scala/SBT source-file allowance",
    )
    assert_contains(
        evidence_text,
        "The Scala/SBT build artifact guard rejects tracked local Scala build server/IDE outputs such as `.bloop/`, `.bsp/`, `.metals/`, and `.scala-build/` at any tree depth",
        "launch evidence Scala/SBT artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Scala/SBT build artifacts",
        "launch evidence Scala/SBT artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "while preserving source-of-truth Scala/SBT files such as `build.sbt`, `project/build.properties`, and Scala source files",
        "launch evidence Scala/SBT source allowance",
    )
    assert_contains(
        evidence_text,
        "R/RStudio artifact guard",
        "launch evidence R/RStudio artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local R/RStudio artifacts such as `.Rproj.user/` and `renv/library/` at any tree depth, plus `.Rhistory`, `.RData`, and `.Ruserdata`",
        "launch evidence R/RStudio artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local R/RStudio artifacts",
        "launch evidence R/RStudio artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth R files such as `*.R`, `*.Rmd`, `.Rproj` project files, `renv.lock`, `DESCRIPTION`, and R docs",
        "launch evidence R/RStudio source-file allowance",
    )
    assert_contains(
        evidence_text,
        "Julia depot/preference artifact guard",
        "launch evidence Julia artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Julia depot/preference artifacts such as `.julia/` at any tree depth, `LocalPreferences.toml`, `*.jl.cov`, and `*.jl.mem`",
        "launch evidence Julia artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Julia depot/preference artifacts",
        "launch evidence Julia artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Julia project files such as `Project.toml`, `Manifest.toml`, `Artifacts.toml`, and Julia source files by filename alone",
        "launch evidence Julia source allowance",
    )
    assert_contains(
        evidence_text,
        "Ruby/Bundler dependency artifact guard",
        "launch evidence Ruby/Bundler artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Bundler state/cache paths such as `.bundle/`, `vendor/bundle/`, and `vendor/cache/` at any tree depth",
        "launch evidence Ruby/Bundler artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Ruby/Bundler dependency artifacts plus missing nested `.gitignore` coverage",
        "launch evidence Ruby/Bundler artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Ruby files such as `Gemfile`, `Gemfile.lock`, Ruby source files, Ruby docs, and non-artifact vendor paths",
        "launch evidence Ruby/Bundler source allowance",
    )
    assert_contains(
        evidence_text,
        "PHP Composer dependency/test artifact guard",
        "launch evidence PHP Composer artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Composer/PHPUnit artifacts such as `vendor/autoload.php`, `vendor/bin/`, `vendor/composer/`, `.phpunit.cache/`, `.phpunit.result.cache`, and `composer.phar`",
        "launch evidence PHP Composer artifact examples",
    )
    assert_contains(
        evidence_text,
        "including Composer vendor/PHPUnit cache artifacts at any tree depth",
        "launch evidence PHP Composer nested artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local PHP Composer dependency/test artifacts plus missing nested `.gitignore` coverage",
        "launch evidence PHP Composer artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Composer files such as `composer.json`, `composer.lock`, PHP source files, and PHP docs",
        "launch evidence PHP Composer source allowance",
    )
    assert_contains(
        evidence_text,
        "Perl/CPAN build/dependency artifact guard",
        "launch evidence Perl/CPAN artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local CPAN client, build, and local::lib artifacts such as `.cpan/`, `.cpanm/`, `blib/`, `local/lib/perl5/`, `MYMETA.json`, `MYMETA.yml`, and `pm_to_blib`",
        "launch evidence Perl/CPAN artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Perl/CPAN build/dependency artifacts",
        "launch evidence Perl/CPAN artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Perl files such as `cpanfile`, `cpanfile.snapshot`, `Makefile.PL`, `Build.PL`, Perl source files, and Perl docs",
        "launch evidence Perl/CPAN source allowance",
    )
    assert_contains(
        launch_text,
        "tracked local release/package artifacts including common archive formats such as `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`, `.7z`, `.rar`, and `.zst`",
        "launch checklist release/package archive examples",
    )
    assert_contains(
        launch_text,
        "installer/package formats such as `.dmg`, `.pkg`, `.whl`, `.egg`, `.deb`, `.rpm`, and `.msi`",
        "launch checklist release/package installer examples",
    )
    assert_contains(
        launch_text,
        "JVM package archives such as `.jar`, `.war`, and `.ear`; and root `artifacts/` outputs",
        "launch checklist release/package JVM and artifact examples",
    )
    assert_contains(
        launch_text,
        "root `.gitignore` coverage for local release/package artifacts",
        "launch checklist release/package artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "backup and data dump artifacts such as root `backups/`, root `dumps/`, `*.bak`, `*.backup`, `*.dump`, and `*.sql`",
        "launch checklist backup/dump artifact examples",
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
        "tracked local notebook checkpoint/runtime config artifacts",
        "launch checklist notebook runtime/config artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "tracked notebook execution outputs",
        "launch checklist notebook execution-output risk-scan scope",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local notebook artifacts",
        "launch checklist notebook artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "tracked local documentation build artifacts",
        "launch checklist documentation build artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "LaTeX auxiliary files (`*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.synctex.gz`)",
        "launch checklist documentation LaTeX artifact examples",
    )
    assert_contains(
        launch_text,
        "_minted-*` caches",
        "launch checklist documentation minted artifact examples",
    )
    assert_contains(
        launch_text,
        "generated API documentation exports such as `swagger-ui/`, `swagger-ui-dist/`, `redoc-static/`, `openapi-generated/`, `api-docs-generated/`, `swagger-ui.html`, `redoc-static.html`, and `openapi-bundle.json`",
        "launch checklist generated API documentation artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local documentation build artifacts",
        "launch checklist documentation build artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "It intentionally does not block all PDFs",
        "launch checklist documentation PDF source allowance",
    )
    assert_contains(
        launch_text,
        "source-of-truth API specification/configuration files such as `openapi.yaml`, `openapi.json`, `.redocly.yaml`, `redocly.yaml`, `openapitools.json`, and `openapi-generator-config.*`",
        "launch checklist API documentation source allowance",
    )
    assert_contains(
        evidence_text,
        "local notebook checkpoint artifact guard",
        "launch evidence notebook checkpoint artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "local notebook runtime/config artifact guard",
        "launch evidence notebook runtime/config artifact risk-scan scope",
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
        "tracked local notebook runtime/config artifacts such as `.jupyter/` and `.nbhistory`",
        "launch evidence notebook runtime/config artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local notebook artifacts",
        "launch evidence notebook artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local documentation build artifact guard",
        "launch evidence documentation build artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "LaTeX/Pandoc-adjacent build byproducts such as `*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.synctex.gz`, `_minted-*` caches, and generated API documentation exports",
        "launch evidence documentation build artifact examples",
    )
    assert_contains(
        evidence_text,
        "`swagger-ui/`, `swagger-ui-dist/`, `redoc-static/`, `openapi-generated/`, `api-docs-generated/`, `swagger-ui.html`, `redoc-static.html`, and `openapi-bundle.json`",
        "launch evidence generated API documentation artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local documentation build artifacts",
        "launch evidence documentation build artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth API specification/configuration files such as `openapi.yaml`, `openapi.json`, `.redocly.yaml`, `redocly.yaml`, `openapitools.json`, and `openapi-generator-config.*`",
        "launch evidence API documentation source allowance",
    )
    assert_contains(
        launch_text,
        "Meson build artifact guard also rejects tracked local Meson build artifacts",
        "launch checklist Meson build artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `.mesonpy-*`, `meson-info/`, `meson-logs/`, and `meson-private/`",
        "launch checklist Meson build artifact examples",
    )
    assert_contains(
        launch_text,
        "`meson-private/`, with matching root and nested `.gitignore` coverage",
        "launch checklist Meson build artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Meson files such as `meson.build`, `meson_options.txt`, or `meson.options`",
        "launch checklist Meson source-file allowance scope",
    )
    assert_contains(
        evidence_text,
        "local Meson build artifact guard",
        "launch evidence Meson build artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Meson build artifacts such as `.mesonpy-*` build directories at any tree depth, `builddir/`, `meson-info/`, `meson-logs/`, and `meson-private/`",
        "launch evidence Meson build artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Meson build artifacts",
        "launch evidence Meson build artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Meson files such as `meson.build`, `meson_options.txt`, and `meson.options`",
        "launch evidence Meson source allowance",
    )
    assert_contains(
        launch_text,
        "Autotools build artifact guard also rejects tracked local Autotools configure and Libtool outputs",
        "launch checklist Autotools build artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "such as `autom4te.cache/`, `.deps/`, `.libs/`, `config.log`, `config.status`, generated `libtool` files, `*.lo`, and `*.la`",
        "launch checklist Autotools build artifact examples",
    )
    assert_contains(
        launch_text,
        "with matching root `.gitignore` coverage",
        "launch checklist Autotools build artifact ignore scope",
    )
    assert_contains(
        launch_text,
        "source-of-truth Autotools files such as `configure.ac`, `Makefile.am`, or Autotools docs",
        "launch checklist Autotools source-file allowance",
    )
    assert_contains(
        evidence_text,
        "Autotools local configure/Libtool output guard",
        "launch evidence Autotools build artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "Autotools configure/Libtool outputs (`autom4te.cache/`, `.deps/`, `.libs/`, `config.log`, `config.status`, generated `libtool` files, `*.lo`, `*.la`)",
        "launch evidence Autotools hardening detail examples",
    )
    assert_contains(
        evidence_text,
        "tracked local Autotools configure and Libtool outputs such as `autom4te.cache/`, `.deps/`, `.libs/`, `config.log`, `config.status`, generated `libtool` files, `*.lo`, and `*.la`",
        "launch evidence Autotools build artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Autotools files such as `configure.ac`, `Makefile.am`, and Autotools docs",
        "launch evidence Autotools source allowance",
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
        "local ML experiment/tracking artifact guard",
        "launch evidence ML experiment/tracking artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked W&B, MLflow, Lightning, and TensorBoard run outputs such as `.wandb/`, `wandb/`, `mlruns/`, `lightning_logs/`, and `events.out.tfevents.*`",
        "launch evidence ML experiment/tracking artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local ML experiment/tracking artifacts",
        "launch evidence ML experiment/tracking artifact ignore examples",
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
        "local Terraform/OpenTofu/Terragrunt infrastructure state artifact guard",
        "launch evidence infrastructure state artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Terraform/OpenTofu/Terragrunt infrastructure state artifacts such as `.terraform/`, `.tofu/`, `.terragrunt-cache/`, `.terraform.lock.hcl`, `.tofu.lock.hcl`, `.tfstate`, `.tfvars`, `.tfvars.json`, and `.tfplan` files",
        "launch evidence infrastructure state artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local infrastructure state artifacts",
        "launch evidence infrastructure state artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "local Nix build result artifact guard",
        "launch evidence Nix artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Nix build result artifacts such as root `result` and `result-*` outputs",
        "launch evidence Nix artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Nix build result artifacts",
        "launch evidence Nix artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth Nix files such as `flake.nix`, `flake.lock`, `default.nix`, `shell.nix`, and `*.nix` files",
        "launch evidence Nix source-file allowance",
    )
    assert_contains(
        evidence_text,
        "local Bazel build artifact guard",
        "launch evidence Bazel artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Bazel build artifacts such as `bazel-bin/`, `bazel-out/`, `bazel-testlogs/`, and root `bazel-*` outputs",
        "launch evidence Bazel artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local Bazel build artifacts",
        "launch evidence Bazel artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "Buck/Buck2 local build output guard for `.buckd/` and `buck-out/`, with matching root and nested `.gitignore` coverage",
        "launch evidence Buck/Buck2 artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Buck/Buck2 build artifacts",
        "launch evidence Buck/Buck2 artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "source-file allowances for `BUCK`, `BUCK.v2`, and `.buckconfig`",
        "launch evidence Buck/Buck2 source allowance",
    )
    assert_contains(
        evidence_text,
        "local Swift Package Manager artifact guard",
        "launch evidence SwiftPM artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Swift Package Manager build/workspace artifacts such as `.build/` and `.swiftpm/` at any tree depth",
        "launch evidence SwiftPM artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Swift Package Manager artifacts",
        "launch evidence SwiftPM artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Swift package files such as `Package.swift` and Swift source files",
        "launch evidence SwiftPM source allowance",
    )
    assert_contains(
        evidence_text,
        "Zig local build artifact guard for `.zig-cache/`, `zig-cache/`, and `zig-out/` at any tree depth",
        "launch evidence Zig current-scope examples",
    )
    assert_contains(
        evidence_text,
        "tracked local Zig build artifacts such as `.zig-cache/`, `zig-cache/`, and `zig-out/` at any tree depth",
        "launch evidence Zig artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Zig build artifacts",
        "launch evidence Zig artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Zig files such as `build.zig`, `build.zig.zon`, and Zig source files",
        "launch evidence Zig source allowance",
    )
    assert_contains(
        evidence_text,
        "local Dart/Flutter artifact guard",
        "launch evidence Dart/Flutter artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Dart/Flutter artifacts such as `.dart_tool/`, `.pub-cache/`, `.pub/`, `.packages`, `.flutter-plugins`, and `.flutter-plugins-dependencies` at any tree depth",
        "launch evidence Dart/Flutter artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Dart/Flutter artifacts",
        "launch evidence Dart/Flutter artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth project files such as `pubspec.yaml`, `pubspec.lock`, Dart source, Android Gradle project files, and iOS Xcode project files",
        "launch evidence Dart/Flutter source-file allowance",
    )
    assert_contains(
        launch_text,
        "The Kotlin/Kotlin Native compiler artifact guard also rejects tracked local Kotlin/Kotlin Native compiler artifacts",
        "launch checklist Kotlin/Kotlin Native artifact risk-scan scope",
    )
    assert_contains(
        launch_text,
        "tracked local Kotlin/Kotlin Native compiler artifacts such as `.kotlin/` and `.konan/` at any tree depth",
        "launch checklist Kotlin/Kotlin Native artifact examples",
    )
    assert_contains(
        launch_text,
        "root and nested `.gitignore` coverage for local Kotlin/Kotlin Native artifacts",
        "launch checklist Kotlin/Kotlin Native artifact ignore examples",
    )
    assert_contains(
        launch_text,
        "source-of-truth Kotlin files such as `build.gradle.kts`, `settings.gradle.kts`, Kotlin source files, or Gradle version catalog files",
        "launch checklist Kotlin/Kotlin Native source allowance",
    )
    assert_contains(
        evidence_text,
        "local Kotlin/Kotlin Native compiler artifact guard",
        "launch evidence Kotlin/Kotlin Native artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Kotlin/Kotlin Native compiler artifacts such as `.kotlin/` and `.konan/` at any tree depth",
        "launch evidence Kotlin/Kotlin Native artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Kotlin/Kotlin Native artifacts",
        "launch evidence Kotlin/Kotlin Native artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Kotlin files such as `build.gradle.kts`, `settings.gradle.kts`, Kotlin source files, and Gradle version catalog files",
        "launch evidence Kotlin/Kotlin Native source allowance",
    )
    assert_contains(
        evidence_text,
        "Gradle/JVM local build artifact guard",
        "launch evidence Gradle/JVM artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local Gradle cache/state and build outputs such as `.gradle/`, `examples/android/.gradle/`, `app/build/classes/`, `service/build/reports/`, `service/build/test-results/`, `service/build/tmp/`, `service/build/generated/`, and `service/build/libs/`",
        "launch evidence Gradle/JVM artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Gradle/JVM cache/state artifacts",
        "launch evidence Gradle/JVM artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "JVM compiler bytecode artifact guard",
        "launch evidence JVM compiler artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local bytecode outputs such as `*.class` and `*.tasty`",
        "launch evidence JVM compiler artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local JVM compiler artifacts",
        "launch evidence JVM compiler artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth JVM project files such as `pom.xml`, `build.gradle`, `build.sbt`, Java source files, Kotlin source files, and Scala source files",
        "launch evidence JVM compiler source allowance",
    )
    assert_contains(
        evidence_text,
        ".NET/NuGet local dependency/build/user-state artifact guard",
        "launch evidence .NET/NuGet artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "local dependency/build/user-state artifacts such as `.nuget/`, `packages/`, `bin/Debug/`, `obj/Release/`, `project.assets.json`, `project.nuget.cache`, `*.nupkg`, `*.snupkg`, `*.csproj.user`, and `*.suo`",
        "launch evidence .NET/NuGet artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root `.gitignore` coverage for local .NET/NuGet artifacts",
        "launch evidence .NET/NuGet artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth .NET project files such as `*.csproj`, `*.fsproj`, `*.vbproj`, `*.sln`, `Directory.Build.props`, C# source files, F# source files, and NuGet lock files",
        "launch evidence .NET/NuGet source allowance",
    )
    assert_contains(
        evidence_text,
        "Clojure/Leiningen local artifact guard",
        "launch evidence Clojure/Leiningen artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "local artifacts such as `.lein/`, `.cpcache/`, and `.shadow-cljs/` at any tree depth, plus `.nrepl-port`",
        "launch evidence Clojure/Leiningen artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local Clojure/Leiningen artifacts",
        "launch evidence Clojure/Leiningen artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth Clojure files such as `project.clj`, `deps.edn`, `bb.edn`, `shadow-cljs.edn`, and Clojure/ClojureScript source files",
        "launch evidence Clojure/Leiningen source allowance",
    )
    assert_contains(
        evidence_text,
        "local mobile/Xcode/Android build artifact guard",
        "launch evidence mobile build artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "Expo local project state guard for `.expo/` and `.expo-shared/`",
        "launch evidence Expo local state scope",
    )
    assert_contains(
        evidence_text,
        "source-file allowances for mobile app source, Expo config, Android Gradle project files, and iOS Xcode project files",
        "launch evidence Expo source allowance",
    )
    assert_contains(
        evidence_text,
        "Android native build intermediate guard for `.cxx/` and `.externalNativeBuild/`",
        "launch evidence Android native build intermediate scope",
    )
    assert_contains(
        evidence_text,
        "source-file allowances for Android Gradle project files, JNI/C++ source files, and checked-in native build configuration",
        "launch evidence Android native source allowance",
    )
    assert_contains(
        evidence_text,
        "CocoaPods dependency output guard for root `Pods/` and `ios/Pods/`",
        "launch evidence CocoaPods dependency output scope",
    )
    assert_contains(
        evidence_text,
        "source-file allowances for `Podfile` and `Podfile.lock`",
        "launch evidence CocoaPods source allowance",
    )
    assert_contains(
        evidence_text,
        "Carthage dependency output guard for `Carthage/Build/` and `Carthage/Checkouts/`",
        "launch evidence Carthage dependency output scope",
    )
    assert_contains(
        evidence_text,
        "source-file allowances for `Cartfile` and `Cartfile.resolved`",
        "launch evidence Carthage source allowance",
    )
    assert_contains(
        evidence_text,
        "Fastlane generated report/test artifact guard for `fastlane/report.xml`, `fastlane/Preview.html`, and `fastlane/test_output/`",
        "launch evidence Fastlane report/test artifact scope",
    )
    assert_contains(
        evidence_text,
        "source-file allowances for Fastlane configuration and App Store metadata such as `Fastfile`, `Appfile`, and `fastlane/metadata/`",
        "launch evidence Fastlane source allowance",
    )
    assert_contains(
        evidence_text,
        "tracked local mobile/Xcode/Android build artifacts such as `DerivedData/`, `.gradle/`, `xcuserdata/`, `local.properties`, `.xcuserstate`, `.xcresult`, `.ipa`, `.apk`, `.aab`, and `.dSYM` files",
        "launch evidence mobile build artifact examples",
    )
    assert_contains(
        evidence_text,
        "missing root, nested, Android-specific, and platform artifact `.gitignore` coverage for local mobile build outputs",
        "launch evidence mobile build artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "preserves source-of-truth mobile files such as `Package.swift`, `Podfile`, `Podfile.lock`, `*.xcodeproj/project.pbxproj`, Android Gradle build files, `AndroidManifest.xml`, Fastlane configuration, app-store metadata, and mobile docs",
        "launch evidence mobile source allowance",
    )
    assert_contains(
        evidence_text,
        "local native/CMake build artifact guard",
        "launch evidence native/CMake artifact risk-scan scope",
    )
    assert_contains(
        evidence_text,
        "tracked local native/CMake build/user-local artifacts such as `cmake-build-*`, `CMakeFiles/` at any tree depth, CTest `Testing/Temporary/` output, CPack staging output such as `_CPack_Packages/`, generated CTest/Dart files such as `CTestTestfile.cmake` and `DartConfiguration.tcl`, generated CPack files such as `CPackConfig.cmake` and `CPackSourceConfig.cmake`, `CMakeCache.txt`, user-local `CMakeUserPresets.json`, `cmake_install.cmake`, `compile_commands.json`, `install_manifest.txt`, and Ninja local state files such as `.ninja_deps` and `.ninja_log`",
        "launch evidence native/CMake artifact examples",
    )
    assert_contains(
        evidence_text,
        "preserving source-of-truth CMake files such as `CMakeLists.txt` and `CMakePresets.json`",
        "launch evidence native/CMake source allowance",
    )
    assert_contains(
        evidence_text,
        "missing root and nested `.gitignore` coverage for local native/CMake build artifacts",
        "launch evidence native/CMake artifact ignore examples",
    )
    assert_contains(
        evidence_text,
        "TLS certificate/request artifacts (`*.crt`, `*.csr`, `*.der`, `*.p7b`, `*.p7c`)",
        "launch evidence TLS certificate/request artifact examples",
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
        "Markdown/JSON check-index, timestamp, model-identity, path-label, loopback, and caveat consistency",
        "launch evidence optional artifact QA complete proof scope",
    )
    assert_contains(
        evidence_text,
        "optional acceptance artifact QA also keeps shareable `summary.md` check rows aligned with `summary.json`",
        "launch evidence optional artifact QA markdown index scope",
    )
    assert_contains(
        evidence_text,
        "optional acceptance artifact QA keeps `summary.base_url` loopback-only",
        "launch evidence optional artifact QA loopback base URL scope",
    )
    assert_contains(
        evidence_text,
        "optional acceptance artifact QA keeps `summary.started_at` and `summary.finished_at` as RFC3339 UTC timestamps",
        "launch evidence optional artifact QA timestamp scope",
    )
    assert_contains(
        evidence_text,
        "optional acceptance artifact QA keeps `summary.model_id`, `summary.repo_id`, and `summary.revision` aligned between `summary.json` and `summary.md`",
        "launch evidence optional artifact QA model identity scope",
    )
    assert_contains(
        evidence_text,
        "optional acceptance artifact QA keeps share-safe artifact/state/model/log labels aligned between `summary.json` and `summary.md`",
        "launch evidence optional artifact QA path-label scope",
    )
    assert_contains(
        evidence_text,
        "optional acceptance artifact QA keeps required boundary caveats present in `summary.json` and `summary.md`",
        "launch evidence optional artifact QA caveat scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps `summary.base_url` loopback-only and aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact base URL boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps `summary.started_at` and `summary.finished_at` as RFC3339 UTC timestamps aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact timestamp boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps `summary.md` artifact-index rows aligned with `summary.json` checks",
        "launch evidence backend acceptance artifact markdown index boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps `repo_commit` and `fixture_model_ids` aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact identity metadata boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps share-safe artifact/state/model/log/local-path labels aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact path-label boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps the Markdown Port row aligned with `summary.port` and `summary.base_url`",
        "launch evidence backend acceptance artifact port-row boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps failed-run diagnostics aligned between `summary.json` and `summary.md`",
        "launch evidence backend acceptance artifact failure diagnostics boundary scope",
    )
    assert_contains(
        evidence_text,
        "Backend acceptance artifact QA keeps backend acceptance boundary caveats present in `summary.md`",
        "launch evidence backend acceptance artifact caveat boundary scope",
    )
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
    assert_manifest_shape(repo_manifest)
    for key, replacement in (
        ("name", "Fathom preview API contract"),
        ("status", "experimental-preview"),
        ("base_url", "https://example.invalid"),
        ("scope_note", "Full OpenAI API parity."),
    ):
        bad_manifest = json.loads(json.dumps(repo_manifest))
        bad_manifest[key] = replacement
        try:
            assert_manifest_shape(bad_manifest)
        except AssertionError as exc:
            if f"manifest.{key} must remain" not in str(exc):
                raise AssertionError("manifest identity self-test failed for the wrong reason") from exc
        else:
            raise AssertionError(f"manifest identity self-test did not reject {key} drift")

    for mutate in (
        lambda manifest: manifest["supported_endpoints"].append(
            {"method": "POST", "path": "/v1/responses", "purpose": "Responses API preview"}
        ),
        lambda manifest: manifest["supported_endpoints"].pop(),
        lambda manifest: manifest["supported_endpoints"][0].update({"method": "POST"}),
        lambda manifest: manifest["supported_endpoints"][2].update({"purpose": "full OpenAI-compatible streaming chat"}),
        lambda manifest: manifest["supported_endpoints"][3].update({"required_boundary": "any encoding_format"}),
    ):
        bad_manifest = json.loads(json.dumps(repo_manifest))
        mutate(bad_manifest)
        try:
            assert_manifest_shape(bad_manifest)
        except AssertionError as exc:
            if "narrow public launch endpoint inventory" not in str(exc):
                raise AssertionError("manifest endpoint inventory self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("manifest endpoint inventory self-test did not reject endpoint metadata drift")

    for mutate in (
        lambda manifest: manifest["expected_boundary_errors"].append(
            {
                "boundary": "preview responses API",
                "request_hint": "POST /v1/responses",
                "status": 501,
                "code": "not_implemented",
            }
        ),
        lambda manifest: manifest["expected_boundary_errors"].pop(),
        lambda manifest: manifest["expected_boundary_errors"][0].update({"status": 200}),
        lambda manifest: manifest["expected_boundary_errors"][1].update({"code": "not_implemented"}),
        lambda manifest: manifest["expected_boundary_errors"][11].update({"request_hint": "GET /v1/responses"}),
        lambda manifest: manifest["expected_boundary_errors"][13].update({"expected_behavior": "preview"}),
    ):
        bad_manifest = json.loads(json.dumps(repo_manifest))
        mutate(bad_manifest)
        try:
            assert_manifest_shape(bad_manifest)
        except AssertionError as exc:
            message = str(exc)
            if (
                "narrow public launch boundary inventory" not in message
                and "manifest boundary status must be a 4xx/5xx integer" not in message
            ):
                raise AssertionError("manifest boundary inventory self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("manifest boundary inventory self-test did not reject boundary metadata drift")

    valid_envelope_texts = {
        V1_CONTRACT: """
## Standard error envelope

All application errors use `error.message`, `error.type`, `error.code`, and `error.param`.

```json
{"error": {"message": "Nope.", "type": "invalid_request", "code": "invalid_request", "param": null}}
```
""",
        REFUSAL_MATRIX: "Rows use the standard error envelope with `error.message`, `error.type`, `error.code`, and `error.param`.",
        LAUNCH_CHECKLIST: "Static QA pins standard error envelope fields (`error.message`, `error.type`, `error.code`, and `error.param`).",
        LAUNCH_EVIDENCE: "Static QA pins standard error envelope fields (`error.message`, `error.type`, `error.code`, and `error.param`).",
    }
    assert_standard_error_envelope_docs(repo_manifest, valid_envelope_texts)
    for texts, expected in (
        (
            {
                **valid_envelope_texts,
                LAUNCH_CHECKLIST: "Static QA pins standard error envelope fields (`error.message`, `error.type`, and `error.code`).",
            },
            "`error.param`",
        ),
        (
            {
                **valid_envelope_texts,
                V1_CONTRACT: """
## Standard error envelope

All application errors use `"error": {` with `error.message`, `error.type`, `error.code`, and `error.param`.

```json
{"message": "Nope.", "type": "invalid_request", "code": "invalid_request", "param": null}
```
""",
            },
            "bare top-level error fields",
        ),
    ):
        try:
            assert_standard_error_envelope_docs(repo_manifest, texts)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("standard error envelope docs self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("standard error envelope docs self-test did not reject docs drift")

    valid_v1_json_examples = """
```json
{"error": {"message": "Nope.", "type": "invalid_request", "code": "invalid_request", "param": null}}
```

```json
{"ok": true, "engine": "fathom", "generation_ready": true}
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "chat-model",
      "object": "model",
      "created": 0,
      "owned_by": "fathom",
      "fathom": {
        "provider_kind": "local",
        "status": "available",
        "capability_status": "Runnable",
        "capability_summary": "Real local generation is available through a verified backend lane.",
        "backend_lanes": ["safetensors-hf"]
      }
    }
  ]
}
```

```json
{"model": "chat-model", "messages": [{"role": "user", "content": "Hi"}], "stream": false}
```

```json
{"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "chat-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}]}
```

```json
{"model": "embedding-model", "input": ["hello"], "encoding_format": "float"}
```

```json
{"object": "list", "data": [{"object": "embedding", "embedding": [0.1], "index": 0}], "model": "embedding-model", "usage": {"prompt_tokens": 0, "total_tokens": 0}, "fathom": {"runtime": "candle-bert-embeddings", "embedding_dimension": 384, "scope": "verified local embedding runtime only"}}
```
"""
    assert_v1_contract_json_examples(repo_manifest, valid_v1_json_examples)
    for text, expected in (
        (
            valid_v1_json_examples.replace('"stream": false', '"stream": true'),
            "streaming chat requests",
        ),
        (
            valid_v1_json_examples.replace('"encoding_format": "float"', '"encoding_format": "base64"'),
            "base64 embeddings requests",
        ),
        (
            valid_v1_json_examples.replace('"capability_status": "Runnable"', '"capability_status": "EmbeddingOnly"'),
            "runnable chat/generation models",
        ),
        (
            valid_v1_json_examples.replace('"choices": [', '"data": [], "choices": ['),
            "mix embedding data or errors",
        ),
    ):
        try:
            assert_v1_contract_json_examples(repo_manifest, text)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("v1 JSON examples self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("v1 JSON examples self-test did not reject docs drift")

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

    assert_non_contract_example_surfaces(
        ["GET /api/models/catalog", "POST /api/models/catalog/install"]
    )
    for surfaces, expected in (
        (["get /api/models/catalog"], "uppercase GET/POST local /api paths"),
        (["GET /v1/responses"], "uppercase GET/POST local /api paths"),
        (["GET https://example.invalid/catalog"], "uppercase GET/POST local /api paths"),
        (["GET /api/admin/debug"], "not reviewed for public examples"),
        (["DELETE /api/models/catalog"], "uppercase GET/POST local /api paths"),
        (["GET /api/models/catalog", "GET /api/models/catalog"], "duplicate non-contract example surface"),
    ):
        try:
            assert_non_contract_example_surfaces(surfaces)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("non-contract example surface self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("non-contract example surface self-test did not reject unsafe manifest drift")

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

    good_links = "\n".join(
        [
            "[README](../README.md)",
            "[contract](api/v1-contract.md#supported-endpoints)",
            "[external](https://example.invalid/fathom)",
            "[anchor](#local-section)",
            "[mailto](mailto:security@example.invalid)",
        ]
    )
    good_link_failures = public_doc_local_link_failures(ROOT / "docs" / "synthetic.md", good_links)
    if good_link_failures:
        raise AssertionError(
            "public docs local-link self-test rejected allowed links:\n" + "\n".join(good_link_failures)
        )

    for text, expected in (
        ("[missing](missing.md)", "target does not exist"),
        ("[escape](../../outside.md)", "escapes repository"),
    ):
        failures = public_doc_local_link_failures(ROOT / "docs" / "synthetic.md", text)
        if not any(expected in failure for failure in failures):
            raise AssertionError(f"public docs local-link self-test did not reject {expected}: {failures}")

    synthetic_pr_template = "- [ ] Keep public claims narrow.\n- [ ] Run static QA.\n"
    assert_required_unchecked_task_list_items(
        synthetic_pr_template,
        "synthetic PR template",
        ("Keep public claims narrow.", "Run static QA."),
    )
    for text, expected in (
        ("- [x] Keep public claims narrow.\n- [ ] Run static QA.\n", "must remain unchecked"),
        ("- Keep public claims narrow.\n- [ ] Run static QA.\n", "must keep unchecked task-list item"),
    ):
        try:
            assert_required_unchecked_task_list_items(
                text,
                "synthetic bad PR template",
                ("Keep public claims narrow.", "Run static QA."),
            )
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("PR template checkbox self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("PR template checkbox self-test did not reject checklist drift")

    valid_issue_template = """
name: API contract issue
description: Report or request a narrow `/v1` API contract change.
title: "api: "
labels: [api]
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
    assert_issue_template_metadata(
        valid_issue_template,
        "synthetic issue template",
        expected_name="API contract issue",
        expected_description="Report or request a narrow `/v1` API contract change.",
        expected_title="api: ",
        expected_labels=("api",),
    )
    assert_issue_template_required_fields(valid_issue_template, "synthetic issue template", ("endpoint",))
    assert_issue_template_required_checkbox_options(valid_issue_template, "synthetic issue template", "privacy")
    for text, expected in (
        (valid_issue_template.replace("name: API contract issue", "name: General support"), "name metadata"),
        (
            valid_issue_template.replace(
                "description: Report or request a narrow `/v1` API contract change.",
                "description: Ask anything about Fathom.",
            ),
            "description metadata",
        ),
        (valid_issue_template.replace('title: "api: "', 'title: "question: "'), "title metadata"),
        (valid_issue_template.replace("labels: [api]", "labels: [question]"), "labels metadata"),
        (valid_issue_template.replace("labels: [api]", "labels: api"), "labels metadata must stay as an inline list"),
    ):
        try:
            assert_issue_template_metadata(
                text,
                "synthetic bad issue template",
                expected_name="API contract issue",
                expected_description="Report or request a narrow `/v1` API contract change.",
                expected_title="api: ",
                expected_labels=("api",),
            )
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("issue-template metadata self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("issue-template metadata self-test did not reject routing metadata drift")

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

    assert_issue_template_config_text("blank_issues_enabled: false\n", "synthetic issue template config")
    for text, expected in (
        ("blank_issues_enabled: true\n", "must set exactly one `blank_issues_enabled: false` entry"),
        ("blank_issues_enabled: false\ncontact_links:\n", "must not add public contact links"),
        (
            "blank_issues_enabled: false\nblank_issues_enabled: false\n",
            "must set exactly one `blank_issues_enabled: false` entry",
        ),
    ):
        try:
            assert_issue_template_config_text(text, "synthetic bad issue template config")
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("issue-template config self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("issue-template config self-test did not reject unsafe routing drift")

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

    base_url_manifest = {"base_url": "http://127.0.0.1:8180"}
    valid_base_url_texts = {
        "docs/api/v1-contract.md": "Base URL in local development: `http://127.0.0.1:8180`.",
        "examples/api/python-no-deps.py": 'BASE_URL = "http://127.0.0.1:8180"',
    }
    assert_manifest_base_url_alignment(base_url_manifest, valid_base_url_texts)
    try:
        assert_manifest_base_url_alignment(
            base_url_manifest,
            {
                **valid_base_url_texts,
                "docs/api/client-examples.md": "Use http://127.0.0.1:8280 as the local base URL.",
            },
        )
    except AssertionError as exc:
        if "public API base URL" not in str(exc):
            raise AssertionError("manifest base URL self-test failed for the wrong reason") from exc
    else:
        raise AssertionError("manifest base URL self-test did not reject stale public docs/examples")

    router_manifest = {
        "supported_endpoints": [
            {"method": "GET", "path": "/v1/health"},
            {"method": "POST", "path": "/v1/chat/completions"},
        ]
    }
    valid_router = """
fn app_router(state: AppState) -> Router {
    let v1_router = Router::new()
        .route("/health", get(v1_health).fallback(v1_method_not_allowed))
        .route(
            "/chat/completions",
            post(v1_chat_completions_route).fallback(v1_method_not_allowed),
        )
        .fallback(v1_not_found);

    Router::new()
        .nest("/v1", v1_router)
        .with_state(state)
}

fn v1_json_rejection_error(rejection: JsonRejection) -> (StatusCode, Json<serde_json::Value>) {
    error_json(StatusCode::BAD_REQUEST, "Malformed JSON request body", "invalid_request")
}

async fn v1_not_found(uri: Uri) -> ApiError {
    error_json(StatusCode::NOT_FOUND, "missing", "not_found")
}

async fn v1_method_not_allowed(method: Method, uri: Uri) -> ApiError {
    error_json(StatusCode::METHOD_NOT_ALLOWED, "wrong method", "method_not_allowed")
}
"""
    assert_backend_v1_router_matches_manifest(router_manifest, valid_router)
    for text, expected in (
        (
            valid_router.replace('.route("/health", get(v1_health).fallback(v1_method_not_allowed))\n', ""),
            "GET /v1/health",
        ),
        (
            valid_router.replace(".fallback(v1_method_not_allowed)", ".fallback(other_method_not_allowed)", 1),
            "GET /v1/health",
        ),
        (
            valid_router.replace(".fallback(v1_not_found);", ";"),
            "v1_not_found fallback",
        ),
        (
            valid_router.replace('.nest("/v1", v1_router)', '.nest("/api/v1", v1_router)'),
            "nested at /v1",
        ),
        (
            valid_router.replace('"method_not_allowed"', '"invalid_request"'),
            "method_not_allowed",
        ),
    ):
        try:
            assert_backend_v1_router_matches_manifest(router_manifest, text)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("backend v1 router self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("backend v1 router self-test did not reject manifest/router drift")

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

    valid_frontend_package = {
        "private": True,
        "scripts": {
            "dev": "vite --host 127.0.0.1 --port 4185",
            "preview": "vite preview --host 127.0.0.1 --port 4185",
            "build": "vite build",
            "qa:copy": "node scripts/ui-copy-qa.mjs",
        },
    }
    assert_frontend_package_scripts(valid_frontend_package, "synthetic frontend package")
    for package, expected in (
        ({**valid_frontend_package, "private": False}, "private: true"),
        (
            {
                **valid_frontend_package,
                "scripts": {**valid_frontend_package["scripts"], "dev": "vite --host 0.0.0.0 --port 4185"},
            },
            "127.0.0.1",
        ),
        (
            {
                **valid_frontend_package,
                "scripts": {**valid_frontend_package["scripts"], "preview": "vite preview --port 4185"},
            },
            "127.0.0.1",
        ),
        (
            {
                **valid_frontend_package,
                "scripts": {**valid_frontend_package["scripts"], "build": "vite build --mode production"},
            },
            "vite build",
        ),
        (
            {
                **valid_frontend_package,
                "scripts": {**valid_frontend_package["scripts"], "qa:copy": "node scripts/other.mjs"},
            },
            "ui-copy-qa.mjs",
        ),
    ):
        try:
            assert_frontend_package_scripts(package, "synthetic bad frontend package")
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("frontend package script self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("frontend package script self-test did not reject unsafe package drift")

    valid_cargo_manifests = {
        "crates/fathom-core/Cargo.toml": "[package]\nname = \"fathom-core\"\npublish = false\n\n[dependencies]\n",
        "crates/fathom-server/Cargo.toml": "[package]\nname = \"fathom-server\"\npublish = false\n\n[dependencies]\n",
    }
    assert_cargo_publish_safety(valid_cargo_manifests)
    for manifests, expected in (
        (
            {
                **valid_cargo_manifests,
                "crates/fathom-core/Cargo.toml": "[package]\nname = \"fathom-core\"\n\n[dependencies]\n",
            },
            "publish = false",
        ),
        (
            {
                **valid_cargo_manifests,
                "crates/fathom-server/Cargo.toml": "[package]\nname = \"fathom-server\"\npublish = true\n\n[dependencies]\n",
            },
            "publish = false",
        ),
    ):
        try:
            assert_cargo_publish_safety(manifests)
        except AssertionError as exc:
            if expected not in str(exc):
                raise AssertionError("Cargo publish-safety self-test failed for the wrong reason") from exc
        else:
            raise AssertionError("Cargo publish-safety self-test did not reject unsafe manifest drift")

    assert_clean_install_gate("npm --prefix frontend ci\n", "synthetic clean install gate")
    for text, expected in (
        ("npm --prefix frontend install\n", "npm --prefix frontend ci"),
        ("npm --prefix frontend ci\nnpm --prefix frontend install\n", "npm ci"),
        ("## Quick start\nnpm --prefix frontend install\n", "npm --prefix frontend ci"),
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


def assert_public_contract_smoke_artifact_wiring() -> None:
    smoke_text = read(SMOKE)
    artifact_qa_text = read(ROOT / "scripts/public_contract_smoke_artifact_qa.py")
    assert_contains(smoke_text, '"generated_at"', "public contract smoke generated_at summary field")
    assert_contains(smoke_text, "datetime.now(timezone.utc)", "public contract smoke generated_at UTC source")
    assert_contains(smoke_text, "Generated at:", "public contract smoke generated-at markdown row")
    assert_contains(smoke_text, "public-contract-smoke-summary.json", "public contract smoke summary JSON artifact")
    assert_contains(smoke_text, "public-contract-smoke-summary.md", "public contract smoke summary Markdown artifact")
    assert_contains(
        artifact_qa_text,
        "summary.generated_at must be an RFC3339 UTC timestamp ending in Z",
        "public contract smoke artifact QA generated_at timestamp guard",
    )
    assert_contains(
        artifact_qa_text,
        "generated-at line must match summary.generated_at",
        "public contract smoke artifact QA generated_at markdown guard",
    )
    assert_contains(
        artifact_qa_text,
        "markdown/generated_at consistency self-check did not fail",
        "public contract smoke artifact QA generated_at markdown negative self-test",
    )
    assert_contains(
        artifact_qa_text,
        "assert_markdown_rows_match_summary",
        "public contract smoke artifact QA markdown row guard",
    )


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
    for path, opt_in_env, smoke_script, artifact_qa_script, evidence_scope, env_vars, default_phrases in OPTIONAL_ACCEPTANCE_DOCS:
        text = read(path)
        label = str(path.relative_to(ROOT))
        for phrase in required_boundaries:
            assert_contains(text, phrase, label)
        assert_contains(text, opt_in_env, label)
        assert_contains(text, smoke_script, label)
        assert_contains(text, artifact_qa_script, label)
        assert_contains(text, evidence_scope, label)
        for env_var in env_vars:
            assert_contains(text, env_var, label)
        for phrase in default_phrases:
            assert_contains(text, phrase, label)


def assert_optional_acceptance_timestamp_wiring() -> None:
    for _, _, smoke_script, artifact_qa_script, _, _, _ in OPTIONAL_ACCEPTANCE_DOCS:
        smoke_text = read(ROOT / smoke_script)
        artifact_qa_text = read(ROOT / artifact_qa_script)
        assert_contains(smoke_text, "Started:", f"{smoke_script} summary.md started timestamp row")
        assert_contains(smoke_text, "summary['started_at']", f"{smoke_script} summary.md started timestamp source")
        assert_contains(smoke_text, "Finished:", f"{smoke_script} summary.md finished timestamp row")
        assert_contains(smoke_text, "summary['finished_at']", f"{smoke_script} summary.md finished timestamp source")
        assert_contains(
            artifact_qa_text,
            "assert_markdown_timestamps_match_summary",
            f"{artifact_qa_script} markdown timestamp guard",
        )
        assert_contains(
            artifact_qa_text,
            "bad summary.md timestamp self-check did not fail",
            f"{artifact_qa_script} markdown timestamp negative self-test",
        )


def assert_optional_acceptance_loopback_wiring() -> None:
    for _, _, smoke_script, artifact_qa_script, _, _, _ in OPTIONAL_ACCEPTANCE_DOCS:
        smoke_text = read(ROOT / smoke_script)
        artifact_qa_text = read(ROOT / artifact_qa_script)
        assert_contains(smoke_text, "base_url", f"{smoke_script} summary base_url field")
        assert_contains(smoke_text, "127.0.0.1", f"{smoke_script} loopback default host")
        assert_contains(
            artifact_qa_text,
            "assert_loopback_base_url",
            f"{artifact_qa_script} summary base_url loopback guard",
        )
        assert_contains(
            artifact_qa_text,
            "summary.base_url must be an http://127.0.0.1:<port> loopback URL",
            f"{artifact_qa_script} summary base_url loopback error",
        )
        assert_contains(
            artifact_qa_text,
            "external summary.base_url self-check did not fail",
            f"{artifact_qa_script} external base_url negative self-test",
        )


def assert_optional_acceptance_markdown_index_wiring() -> None:
    for _, _, smoke_script, artifact_qa_script, _, _, _ in OPTIONAL_ACCEPTANCE_DOCS:
        smoke_text = read(ROOT / smoke_script)
        artifact_qa_text = read(ROOT / artifact_qa_script)
        assert_contains(smoke_text, "## Checks", f"{smoke_script} summary.md checks section")
        assert_contains(smoke_text, "status", f"{smoke_script} summary.md check status source")
        assert_contains(smoke_text, "artifact", f"{smoke_script} summary.md check artifact source")
        assert_contains(
            artifact_qa_text,
            "assert_markdown_checks_match_summary",
            f"{artifact_qa_script} markdown check-row guard",
        )
        assert_contains(
            artifact_qa_text,
            "summary.md missing check row matching summary.json",
            f"{artifact_qa_script} markdown check-row error",
        )
        assert_contains(
            artifact_qa_text,
            "missing summary.md check row self-check did not fail",
            f"{artifact_qa_script} markdown check-row negative self-test",
        )
        assert_contains(
            artifact_qa_text,
            "summary.checks artifact index mismatch",
            f"{artifact_qa_script} summary checks artifact index guard",
        )


def assert_optional_acceptance_model_identity_wiring() -> None:
    for _, _, smoke_script, artifact_qa_script, _, _, _ in OPTIONAL_ACCEPTANCE_DOCS:
        smoke_text = read(ROOT / smoke_script)
        artifact_qa_text = read(ROOT / artifact_qa_script)
        assert_contains(smoke_text, "Model:", f"{smoke_script} summary.md model identity row")
        assert_contains(smoke_text, "Upstream:", f"{smoke_script} summary.md upstream identity row")
        assert_contains(smoke_text, "MODEL_ID", f"{smoke_script} summary.md model identity source")
        assert_contains(smoke_text, "REPO_ID", f"{smoke_script} summary.md repo identity source")
        assert_contains(smoke_text, "REVISION", f"{smoke_script} summary.md revision identity source")
        assert_contains(
            artifact_qa_text,
            "summary.md missing model_id row matching summary.json",
            f"{artifact_qa_script} markdown model identity guard",
        )
        assert_contains(
            artifact_qa_text,
            "summary.md missing repo_id/revision row matching summary.json",
            f"{artifact_qa_script} markdown upstream identity guard",
        )
        assert_contains(
            artifact_qa_text,
            "bad summary.md model identity self-check did not fail",
            f"{artifact_qa_script} markdown model identity negative self-test",
        )
        assert_contains(
            artifact_qa_text,
            "bad summary.md upstream identity self-check did not fail",
            f"{artifact_qa_script} markdown upstream identity negative self-test",
        )


def assert_optional_acceptance_path_label_wiring() -> None:
    for _, _, smoke_script, artifact_qa_script, _, _, _ in OPTIONAL_ACCEPTANCE_DOCS:
        smoke_text = read(ROOT / smoke_script)
        artifact_qa_text = read(ROOT / artifact_qa_script)
        for label in ("Artifact directory", "State directory", "Model directory", "Server log"):
            assert_contains(smoke_text, label, f"{smoke_script} summary.md {label} row")
        for key in ("artifact_dir", "state_dir", "model_dir", "log_dir"):
            assert_contains(smoke_text, key, f"{smoke_script} summary {key} field")
        assert_contains(
            artifact_qa_text,
            "assert_markdown_path_labels_match_summary",
            f"{artifact_qa_script} markdown path-label guard",
        )
        assert_contains(
            artifact_qa_text,
            "summary.md missing log_dir server-log label matching summary.json",
            f"{artifact_qa_script} markdown server-log path-label guard",
        )
        assert_contains(
            artifact_qa_text,
            "bad summary.md path-label self-check did not fail",
            f"{artifact_qa_script} markdown path-label negative self-test",
        )


def assert_optional_acceptance_caveat_wiring() -> None:
    for _, _, smoke_script, artifact_qa_script, _, _, _ in OPTIONAL_ACCEPTANCE_DOCS:
        smoke_text = read(ROOT / smoke_script)
        artifact_qa_text = read(ROOT / artifact_qa_script)
        assert_contains(smoke_text, "What this does not prove", f"{smoke_script} summary.md caveat section")
        for phrase in (
            "production readiness",
            "legal suitability",
            "external proxying",
            "full OpenAI API parity",
            "dequantization",
        ):
            assert_contains(smoke_text, phrase, f"{smoke_script} summary.md required caveat phrase")
            assert_contains(artifact_qa_text, phrase, f"{artifact_qa_script} required caveat phrase")
        assert_contains(
            artifact_qa_text,
            "REQUIRED_CAVEAT_PHRASES",
            f"{artifact_qa_script} required caveat inventory",
        )
        assert_contains(
            artifact_qa_text,
            "assert_required_caveats(caveats",
            f"{artifact_qa_script} summary.json caveat guard",
        )
        assert_contains(
            artifact_qa_text,
            "assert_required_caveats(md,",
            f"{artifact_qa_script} summary.md caveat guard",
        )
        assert_contains(
            artifact_qa_text,
            "missing caveat self-check did not fail",
            f"{artifact_qa_script} missing caveat negative self-test",
        )


def assert_ci_wiring(manifest: dict[str, Any]) -> None:
    ci_text = read(CI)
    expected = manifest["ci_policy"]["offline_static_gate"]
    assert_default_ci_gate_inventory()
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
    assert_issue_template_metadata(
        template_text,
        label,
        expected_name="API contract issue",
        expected_description="Report or request a narrow `/v1` API contract change.",
        expected_title="api: ",
        expected_labels=("api",),
    )
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
    assert_issue_template_metadata(
        template_text,
        label,
        expected_name="Model/runtime request",
        expected_description="Request support for a specific model format, family, task, or artifact layout.",
        expected_title="model/runtime: ",
        expected_labels=("model-runtime",),
    )
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
    assert_issue_template_metadata(
        template_text,
        label,
        expected_name="Bug report",
        expected_description="Report a reproducible issue without sharing private local data.",
        expected_title="bug: ",
        expected_labels=("bug",),
    )
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
    assert_issue_template_metadata(
        template_text,
        label,
        expected_name="Security or privacy concern",
        expected_description="Ask for a private channel or report a public-safe security/privacy concern.",
        expected_title="security/privacy: ",
        expected_labels=("security", "privacy"),
    )
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


def assert_issue_template_config_text(config_text: str, label: str) -> None:
    blank_issue_settings = re.findall(r"(?m)^\s*blank_issues_enabled\s*:\s*(true|false)\s*(?:#.*)?$", config_text)
    if blank_issue_settings != ["false"]:
        raise AssertionError(f"{label} must set exactly one `blank_issues_enabled: false` entry")
    if re.search(r"(?m)^\s*blank_issues_enabled\s*:\s*true\s*(?:#.*)?$", config_text):
        raise AssertionError(f"{label} must not enable blank public issues")
    if re.search(r"(?m)^\s*contact_links\s*:\s*(?:#.*)?$", config_text):
        raise AssertionError(f"{label} must not add public contact links without a reviewed routing guard")


def assert_issue_template_config() -> None:
    assert_issue_template_config_text(read(ISSUE_TEMPLATE_CONFIG), ".github/ISSUE_TEMPLATE/config.yml")


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
        "npm --prefix frontend ci",
        "python3 -m py_compile",
        "bash -n",
        "bash scripts/public_api_contract_smoke.sh",
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
    assert_required_unchecked_task_list_items(template_text, label, PR_TEMPLATE_REQUIRED_CHECKBOXES)


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
    assert_backend_v1_router_matches_manifest(manifest)
    assert_manifest_base_url_alignment(manifest)
    assert_roadmap_last_updated_freshness()
    assert_license_metadata()
    assert_tracked_python_syntax_coverage()
    assert_tracked_shell_syntax_coverage()
    assert_launch_checklist_clean_install_gate()
    assert_readme_clean_install_gate()
    assert_boundary_docs()
    assert_examples_static(manifest)
    assert_public_docs_local_links()
    assert_launch_checklist_frontend_gates()
    assert_frontend_package_scripts()
    assert_cargo_publish_safety()
    assert_no_positive_overclaims()
    assert_smoke_manifest_wiring()
    assert_public_contract_smoke_artifact_wiring()
    assert_optional_acceptance_docs()
    assert_optional_acceptance_loopback_wiring()
    assert_optional_acceptance_markdown_index_wiring()
    assert_optional_acceptance_timestamp_wiring()
    assert_optional_acceptance_model_identity_wiring()
    assert_optional_acceptance_path_label_wiring()
    assert_optional_acceptance_caveat_wiring()
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
