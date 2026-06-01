#!/usr/bin/env python3
"""Guard Fathom's default CI safety posture.

Default PR/push CI must stay offline/lightweight: no networked backend
acceptance smoke, no model downloads, no persisted public-contract smoke
artifacts, and no non-default ONNX Runtime feature binary path. This helper is
intentionally small and dependency-free.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
OPTIONAL_ACCEPTANCE_SMOKE_SCRIPTS = (
    "scripts/minilm_embeddings_optional_api_acceptance_smoke.sh",
    "scripts/smollm2_optional_api_acceptance_smoke.sh",
    "scripts/qwen25_optional_api_acceptance_smoke.sh",
)
WRITE_TOKEN_PERMISSION_PATTERN = re.compile(
    r"^\s*(?:"
    r"actions|attestations|checks|contents|deployments|discussions|id-token|"
    r"issues|models|packages|pages|pull-requests|security-events|statuses"
    r")\s*:\s*write\s*(?:#.*)?$"
)


def top_level_permissions_block(text: str) -> list[str] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^permissions:\s*(?:#.*)?$", line):
            block: list[str] = []
            for child in lines[index + 1 :]:
                if child and not child.startswith((" ", "\t")):
                    break
                if child.strip():
                    block.append(child)
            return block
        if re.match(r"^permissions:\s*(?:read-all|write-all)\s*(?:#.*)?$", line):
            return [line]
    return None


def evaluate_ci_text(text: str) -> list[str]:
    failures: list[str] = []

    permissions_block = top_level_permissions_block(text)
    if permissions_block is None:
        failures.append("default CI must set top-level permissions: contents: read")
    else:
        permissions_text = "\n".join(permissions_block)
        if re.search(r"^\s*permissions:\s*write-all\s*(?:#.*)?$", permissions_text, re.MULTILINE):
            failures.append("default CI must not grant write-all token permissions")
        if re.search(r"^\s*permissions:\s*read-all\s*(?:#.*)?$", permissions_text, re.MULTILINE):
            failures.append("default CI must not grant read-all token permissions")
        if not re.search(r"^\s*contents:\s*read\s*(?:#.*)?$", permissions_text, re.MULTILINE):
            failures.append("default CI token permissions must include contents: read")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if WRITE_TOKEN_PERMISSION_PATTERN.match(line):
                failures.append(
                    f"default CI must not grant write token permissions, line {line_number}: {line.strip()}"
                )

    if re.search(r"cargo\s+test\b[^\n]*--features\s+[^\n]*onnx-embeddings-ort", text):
        failures.append("default CI must not run onnx-embeddings-ort feature tests")

    saw_public_contract_smoke = False
    saw_ci_static_policy_self_test = False
    saw_public_risk_scan_self_test = False
    saw_public_risk_scan = False
    saw_diff_check = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped in {"git diff --check", "run: git diff --check", "- run: git diff --check"}:
            saw_diff_check = True
        if "scripts/backend_acceptance_smoke.sh" in line:
            if stripped != "bash -n scripts/backend_acceptance_smoke.sh":
                failures.append(
                    f"default CI must only syntax-check backend_acceptance_smoke.sh, line {line_number}: {stripped}"
                )
        if "scripts/public_api_contract_smoke.sh" in line:
            if stripped in {"bash scripts/public_api_contract_smoke.sh", "run: bash scripts/public_api_contract_smoke.sh"}:
                saw_public_contract_smoke = True
            elif stripped not in {"bash -n scripts/public_api_contract_smoke.sh", "run: bash -n scripts/public_api_contract_smoke.sh"}:
                failures.append(
                    f"default CI may only syntax-check or run public_api_contract_smoke.sh, line {line_number}: {stripped}"
                )
        for script in OPTIONAL_ACCEPTANCE_SMOKE_SCRIPTS:
            if script in line and stripped not in {f"bash -n {script}", f"run: bash -n {script}"}:
                failures.append(
                    f"default CI must only syntax-check optional acceptance smoke {script}, line {line_number}: {stripped}"
                )
        if re.search(r"\bpython3\s+scripts/ci_static_policy\.py\s+--self-test\b", stripped):
            saw_ci_static_policy_self_test = True
        if stripped in {
            "bash scripts/public_risk_scan.sh --self-test",
            "run: bash scripts/public_risk_scan.sh --self-test",
            "- run: bash scripts/public_risk_scan.sh --self-test",
        }:
            saw_public_risk_scan_self_test = True
        if stripped in {
            "bash scripts/public_risk_scan.sh",
            "run: bash scripts/public_risk_scan.sh",
            "- run: bash scripts/public_risk_scan.sh",
        }:
            saw_public_risk_scan = True

    if not saw_public_contract_smoke:
        failures.append("default CI must run the no-download public API contract smoke")
    if not saw_ci_static_policy_self_test:
        failures.append("default CI must run the CI static policy self-test")
    if not saw_public_risk_scan_self_test:
        failures.append("default CI must run the public risk scan self-test")
    if not saw_public_risk_scan:
        failures.append("default CI must run the public risk scan")
    if not saw_diff_check:
        failures.append("default CI must run git diff --check")

    if re.search(r"FATHOM_ACCEPTANCE_KEEP_ARTIFACTS|FATHOM_ACCEPTANCE_PORT|backend_acceptance_smoke\.sh\s*$", text):
        failures.append("default CI must not invoke networked backend acceptance smoke")

    if "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR" in text:
        failures.append("default CI must not persist public contract smoke artifacts")

    if re.search(r"actions/(upload-artifact|download-artifact|cache)@", text):
        failures.append("default CI must not upload/download or cache QA/model artifacts")

    if re.search(r"(models|artifacts|target)\s*:", text, re.IGNORECASE):
        failures.append("default CI must not define broad model/artifact/target caches")

    return failures


def assert_policy_passes(text: str) -> None:
    failures = evaluate_ci_text(text)
    if failures:
        raise AssertionError("CI static policy failed:\n- " + "\n- ".join(failures))


def run_self_test() -> None:
    valid = """
name: CI
permissions:
  contents: read
jobs:
  rust:
    steps:
      run: cargo test -q
      run: bash scripts/public_api_contract_smoke.sh
  static-safety:
    steps:
      - run: |
          bash -n scripts/public_api_contract_smoke.sh
          bash -n scripts/backend_acceptance_smoke.sh
      - run: python3 scripts/ci_static_policy.py --self-test
      - run: git diff --check
      - run: bash scripts/public_risk_scan.sh --self-test
      - run: bash scripts/public_risk_scan.sh
      - run: echo done
"""
    assert_policy_passes(valid)

    cases = {
        "missing token permissions": "run: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "write-all token permissions": "permissions: write-all\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "read-all token permissions": "permissions: read-all\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "contents write token permission": "permissions:\n  contents: write\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "id-token write token permission": "permissions:\n  contents: read\n  id-token: write\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "onnx feature": "permissions:\n  contents: read\nrun: cargo test -q --features onnx-embeddings-ort\nrun: bash scripts/public_api_contract_smoke.sh",
        "networked acceptance": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nrun: bash scripts/backend_acceptance_smoke.sh",
        "public contract artifacts": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nenv:\n  FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR: artifacts",
        "optional acceptance smoke run": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nrun: bash scripts/qwen25_optional_api_acceptance_smoke.sh",
        "upload artifact": "permissions:\n  contents: read\nuses: actions/upload-artifact@v4\nrun: bash scripts/public_api_contract_smoke.sh",
        "download artifact": "permissions:\n  contents: read\nuses: actions/download-artifact@v4\nrun: bash scripts/public_api_contract_smoke.sh",
        "cache action": "permissions:\n  contents: read\nuses: actions/cache@v4\nrun: bash scripts/public_api_contract_smoke.sh",
        "broad target cache": "permissions:\n  contents: read\ntarget:\n  path: target\nrun: bash scripts/public_api_contract_smoke.sh",
        "missing public smoke": "run: cargo test -q",
        "missing static policy self-test": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py",
        "missing public risk scan self-test": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test\nrun: bash scripts/public_risk_scan.sh",
        "missing public risk scan": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test\nrun: bash scripts/public_risk_scan.sh --self-test",
        "missing diff check": "permissions:\n  contents: read\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test\nrun: bash scripts/public_risk_scan.sh --self-test\nrun: bash scripts/public_risk_scan.sh",
    }
    for label, text in cases.items():
        if not evaluate_ci_text(text):
            raise AssertionError(f"CI policy self-test did not reject {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard Fathom's default CI safety posture")
    parser.add_argument("--self-test", action="store_true", help="run synthetic policy regression checks")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("CI static policy self-test passed")
        return

    assert_policy_passes(CI.read_text(encoding="utf-8"))
    print("CI static policy passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        raise SystemExit(str(exc)) from exc
