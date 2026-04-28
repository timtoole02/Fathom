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
SCOPE_PHRASES = [
    "no-download",
    "routing/refusal",
    "does not prove model downloads",
    "generation quality",
    "embedding quality",
    "performance",
    "external proxying",
    "broad model support",
]
REQUIRED_NO_DOWNLOAD_BOUNDARIES = {
    "streaming chat completions",
    "base64 embeddings",
    "missing chat model",
    "unknown embedding model",
    "embedding models in /v1/models",
    "external placeholder chat or activation",
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
SAFE_CAVEAT = re.compile(r"\b(no|not|does not|do not|without|unsupported|refused|excluded|metadata placeholders? only|not implemented|not claimed|deferred)\b", re.IGNORECASE)


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
        expected[boundary] = item
    return expected


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
    assert_scope(md, markdown_path.name)
    if summary.get("passed") is False:
        lowered = md.lower()
        if "partial" not in lowered or "must not be treated as a passed public contract smoke" not in lowered:
            raise AssertionError("failed markdown summary must mark checks as partial diagnostics, not a pass")
    return md


def validate_summary_dir(directory: Path) -> None:
    summary = load_json(directory / SUMMARY_JSON)
    md = assert_markdown(summary, directory / SUMMARY_MD)
    assert_share_safe(SUMMARY_JSON, json.dumps(summary, sort_keys=True))

    if summary.get("schema") != SCHEMA:
        raise AssertionError(f"summary.schema must be {SCHEMA!r}")
    if not isinstance(summary.get("passed"), bool):
        raise AssertionError("summary.passed must be a boolean")
    commit = summary.get("commit")
    if not isinstance(commit, str) or not commit:
        raise AssertionError("summary.commit must be a non-empty string")
    manifest = summary.get("manifest")
    if not isinstance(manifest, dict):
        raise AssertionError("summary.manifest must be an object")
    for key in ("path", "name", "status"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise AssertionError(f"summary.manifest.{key} must be a non-empty string")
    if manifest["path"] != "docs/api/public-contract.json":
        raise AssertionError("summary.manifest.path must be docs/api/public-contract.json")
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

    for item in endpoint_checks:
        if not isinstance(item, dict):
            raise AssertionError("endpoint check entries must be objects")
        for key in ("method", "path", "checks", "passed"):
            if key not in item:
                raise AssertionError(f"endpoint check missing {key}: {item!r}")
        if item["passed"] is not True:
            raise AssertionError(f"recorded endpoint checks must be completed passes: {item!r}")
        if not isinstance(item["checks"], list) or not item["checks"]:
            raise AssertionError(f"endpoint check must name at least one check id: {item!r}")

    boundary_by_name: dict[str, dict[str, Any]] = {}
    for item in boundary_checks:
        if not isinstance(item, dict):
            raise AssertionError("boundary check entries must be objects")
        boundary = item.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise AssertionError(f"boundary check missing boundary name: {item!r}")
        if item.get("passed") is not True:
            raise AssertionError(f"recorded boundary checks must be completed passes: {item!r}")
        if not isinstance(item.get("check"), str) or not item["check"]:
            raise AssertionError(f"boundary check must name a check id: {item!r}")
        boundary_by_name[boundary] = item

    for item in deferred:
        if not isinstance(item, dict):
            raise AssertionError("deferred boundary entries must be objects")
        reason = item.get("reason")
        if not isinstance(item.get("boundary"), str) or not item["boundary"]:
            raise AssertionError(f"deferred boundary missing boundary name: {item!r}")
        if not isinstance(reason, str) or "requires downloaded/registered model state" not in reason or "outside the no-download smoke" not in reason:
            raise AssertionError(f"deferred boundary reason must preserve no-download caveat: {item!r}")

    if REFUSAL_ONLY_BOUNDARY in boundary_by_name:
        external = boundary_by_name[REFUSAL_ONLY_BOUNDARY]
        if external.get("status") != 501 or external.get("code") != REFUSAL_ONLY_CODE:
            raise AssertionError("external placeholder boundary must remain a 501 external_proxy_not_implemented refusal")
        external_text = json.dumps(external, sort_keys=True) + "\n" + md
        if REFUSAL_ONLY_CODE not in external_text or "refusal" not in external_text.lower():
            raise AssertionError("external placeholder wording must be refusal-only")

    manifest_data = load_manifest()
    expected_by_name = manifest_expected_boundaries(manifest_data)
    if summary["passed"] is True:
        manifest_endpoints = {(item["method"], item["path"]) for item in manifest_data.get("supported_endpoints", [])}
        checked_endpoints = {(item.get("method"), item.get("path")) for item in endpoint_checks}
        missing_endpoints = sorted(manifest_endpoints - checked_endpoints)
        if missing_endpoints:
            raise AssertionError(f"passed summary missing endpoint coverage: {missing_endpoints}")
        missing_boundaries = sorted(REQUIRED_NO_DOWNLOAD_BOUNDARIES - set(boundary_by_name))
        if missing_boundaries:
            raise AssertionError(f"passed summary missing no-download boundary coverage: {missing_boundaries}")
        for boundary, result in boundary_by_name.items():
            expected = expected_by_name.get(boundary)
            if expected is None:
                raise AssertionError(f"passed summary has boundary not present in public-contract.json: {boundary!r}")
            if "status" in expected and result.get("status") != expected["status"]:
                raise AssertionError(
                    f"passed summary boundary {boundary!r} status {result.get('status')!r} does not match public-contract.json {expected['status']!r}"
                )
            if "code" in expected and result.get("code") != expected["code"]:
                raise AssertionError(
                    f"passed summary boundary {boundary!r} code {result.get('code')!r} does not match public-contract.json {expected['code']!r}"
                )
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
        item: dict[str, Any] = {"boundary": boundary, "check": "synthetic-check", "passed": True}
        if boundary == "streaming chat completions":
            item.update({"status": 501, "code": "not_implemented"})
        elif boundary == "base64 embeddings":
            item.update({"status": 400, "code": "invalid_request"})
        elif boundary == "missing chat model":
            item.update({"status": 400, "code": "model_not_found"})
        elif boundary == "unknown embedding model":
            item.update({"status": 404, "code": "embedding_model_not_found"})
        elif boundary == REFUSAL_ONLY_BOUNDARY:
            item.update({"status": 501, "code": REFUSAL_ONLY_CODE})
        boundary_checks.append(item)
    return {
        "schema": SCHEMA,
        "generated_at": "2026-04-27T00:00:00Z",
        "commit": "sample",
        "manifest": {"path": "docs/api/public-contract.json", "name": manifest.get("name"), "status": manifest.get("status")},
        "passed": True,
        "proof_scope": "No-download real-backend routing/refusal smoke only. Does not prove model downloads, generation quality, embedding quality, performance, external proxying, or broad model support.",
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
    boundaries = [
        f"- {item['boundary']}: pass ({item['check']})" for item in summary["boundary_checks"]
    ] or ["- none completed before failure"]
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
            f"- Manifest: `docs/api/public-contract.json` / `{summary['manifest']['name']}` (`{summary['manifest']['status']}`)",
            "- Scope: no-download real-backend routing/refusal smoke only; does not prove model downloads, generation quality, embedding quality, performance, external proxying, or broad model support.",
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
            (unsafe / SUMMARY_MD).read_text(encoding="utf-8") + "\nUnsafe: provider call succeeded with api_key: sk-this-is-not-share-safe\n",
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
