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


def evaluate_ci_text(text: str) -> list[str]:
    failures: list[str] = []

    if re.search(r"cargo\s+test\b[^\n]*--features\s+[^\n]*onnx-embeddings-ort", text):
        failures.append("default CI must not run onnx-embeddings-ort feature tests")

    saw_public_contract_smoke = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
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

    if not saw_public_contract_smoke:
        failures.append("default CI must run the no-download public API contract smoke")

    if re.search(r"FATHOM_ACCEPTANCE_KEEP_ARTIFACTS|FATHOM_ACCEPTANCE_PORT|backend_acceptance_smoke\.sh\s*$", text):
        failures.append("default CI must not invoke networked backend acceptance smoke")

    if "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR" in text:
        failures.append("default CI must not persist public contract smoke artifacts")

    if re.search(r"actions/(upload-artifact|cache)@", text):
        failures.append("default CI must not upload or cache QA/model artifacts")

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
jobs:
  rust:
    steps:
      run: cargo test -q
      run: bash scripts/public_api_contract_smoke.sh
  static-safety:
    steps:
      run: |
        bash -n scripts/public_api_contract_smoke.sh
        bash -n scripts/backend_acceptance_smoke.sh
      run: echo done
"""
    assert_policy_passes(valid)

    cases = {
        "onnx feature": "run: cargo test -q --features onnx-embeddings-ort\nrun: bash scripts/public_api_contract_smoke.sh",
        "networked acceptance": "run: bash scripts/public_api_contract_smoke.sh\nrun: bash scripts/backend_acceptance_smoke.sh",
        "public contract artifacts": "run: bash scripts/public_api_contract_smoke.sh\nenv:\n  FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR: artifacts",
        "upload artifact": "uses: actions/upload-artifact@v4\nrun: bash scripts/public_api_contract_smoke.sh",
        "cache action": "uses: actions/cache@v4\nrun: bash scripts/public_api_contract_smoke.sh",
        "broad target cache": "target:\n  path: target\nrun: bash scripts/public_api_contract_smoke.sh",
        "missing public smoke": "run: cargo test -q",
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
