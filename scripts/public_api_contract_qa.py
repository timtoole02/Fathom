#!/usr/bin/env python3
"""Static QA for Fathom's public launch API contract.

This gate is intentionally dependency-free and offline. It does not start
Fathom, download models, call external APIs, or enable non-default runtime
features. It keeps public docs, examples, README copy, and CI policy aligned
with docs/api/public-contract.json.
"""

from __future__ import annotations

import json
import re
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
README = ROOT / "README.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
SMOKE = ROOT / "scripts" / "public_api_contract_smoke.sh"
EXAMPLES_DIR = ROOT / "examples" / "api"

DOC_PATHS = [V1_CONTRACT, CLIENT_EXAMPLES, BACKEND_QUICKSTART, LAUNCH_CHECKLIST, LAUNCH_EVIDENCE, REFUSAL_MATRIX, README]
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*"))
TEXT_PATHS = DOC_PATHS + EXAMPLE_PATHS + [CI]

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
]

ALLOWED_EXAMPLE_ENDPOINTS = {
    "GET /v1/health",
    "GET /v1/models",
    "POST /v1/chat/completions",
    "POST /v1/embeddings",
    "GET /api/models/catalog",
    "POST /api/models/catalog/install",
}


def load_manifest() -> dict[str, Any]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("public contract manifest must be a JSON object")
    return data


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} missing {needle!r}")


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
    assert_contains(launch_text, "scripts/backend_acceptance_smoke.sh", "launch checklist optional acceptance smoke")
    assert_contains(launch_text, "public-launch-evidence.md", "launch checklist evidence link")
    assert_contains(launch_text, "What this launch does not prove", "launch checklist boundaries")

    evidence_text = read(LAUNCH_EVIDENCE)
    assert_contains(evidence_text, "a32505eadac6539865d224a8b4195656003a0032", "launch evidence commit")
    assert_contains(evidence_text, "scripts/public_contract_smoke_artifact_qa.py", "launch evidence artifact QA")
    assert_contains(evidence_text, "What this evidence does not prove", "launch evidence caveats")
    assert_contains(evidence_text, "external_proxy_not_implemented", "launch evidence external placeholder refusal")

    matrix_text = read(REFUSAL_MATRIX)
    for phrase in (
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
    ):
        assert_contains(matrix_text, phrase, "refusal boundary matrix")
    assert_contains(v1_text, "refusal-boundary-matrix.md", "v1 contract refusal matrix link")
    assert_contains(launch_text, "refusal-boundary-matrix.md", "launch checklist refusal matrix link")

    quickstart_text = read(BACKEND_QUICKSTART)
    contributing_text = read(CONTRIBUTING)
    assert_contains(quickstart_text, "manifest-driven", "backend quickstart public contract smoke")
    assert_contains(quickstart_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "backend quickstart public contract artifact env")
    assert_contains(contributing_text, "public-contract.json", "contributing manifest-driven public contract smoke")
    assert_contains(contributing_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "contributing public contract artifact env")


def assert_examples_static() -> None:
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
        unexpected = {item for item in endpoints if item not in ALLOWED_EXAMPLE_ENDPOINTS}
        if unexpected:
            raise AssertionError(f"{path.relative_to(ROOT)} uses endpoints outside public examples allow-list: {sorted(unexpected)}")


def assert_no_positive_overclaims() -> None:
    for path in TEXT_PATHS:
        text = read(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern, caveat in DANGEROUS_POSITIVE_PATTERNS:
                if pattern.search(line) and not caveat.search(line):
                    raise AssertionError(f"{path.relative_to(ROOT)}:{line_no}: {label}: {line.strip()}")


def assert_smoke_manifest_wiring() -> None:
    smoke_text = read(SMOKE)
    assert_contains(smoke_text, "docs/api/public-contract.json", "public contract smoke manifest load")
    assert_contains(smoke_text, "supported_endpoints", "public contract smoke endpoint coverage")
    assert_contains(smoke_text, "expected_boundary_errors", "public contract smoke boundary coverage")
    assert_contains(smoke_text, "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR", "public contract smoke artifact env")
    assert_contains(smoke_text, "fathom.public_contract_smoke.summary.v1", "public contract smoke artifact schema")
    assert_contains(smoke_text, "partial diagnostic evidence", "public contract smoke failed artifact caveat")


def assert_ci_wiring(manifest: dict[str, Any]) -> None:
    ci_text = read(CI)
    expected = manifest["ci_policy"]["offline_static_gate"]
    assert_contains(ci_text, "python3 -m py_compile", "CI Python syntax step")
    assert_contains(ci_text, "scripts/public_api_contract_qa.py", "CI public API contract QA wiring")
    assert_contains(ci_text, "scripts/public_contract_smoke_artifact_qa.py", "CI public contract smoke artifact QA wiring")
    assert_contains(ci_text, expected, "CI public API contract QA run step")
    assert_contains(ci_text, "python3 scripts/public_contract_smoke_artifact_qa.py", "CI public contract smoke artifact QA run step")
    assert_contains(ci_text, "bash scripts/public_api_contract_smoke.sh", "CI public API contract smoke run step")
    assert_contains(ci_text, "bash -n scripts/public_api_contract_smoke.sh", "CI public API contract smoke syntax step")
    if re.search(r"cargo\s+test\b[^\n]*--features\s+[^\n]*onnx-embeddings-ort", ci_text):
        raise AssertionError("default CI must not enable onnx-embeddings-ort")
    for line_no, line in enumerate(ci_text.splitlines(), start=1):
        if "scripts/backend_acceptance_smoke.sh" in line and line.strip() != "bash -n scripts/backend_acceptance_smoke.sh":
            raise AssertionError(f"default CI must only syntax-check backend acceptance smoke, line {line_no}")


def main() -> int:
    manifest = load_manifest()
    assert_endpoint_docs(manifest)
    assert_boundary_docs()
    assert_examples_static()
    assert_no_positive_overclaims()
    assert_smoke_manifest_wiring()
    assert_ci_wiring(manifest)
    print("public API contract QA passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"public API contract QA failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
