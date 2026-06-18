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
REQUIRED_ARTIFACT_QA_COMMANDS = (
    "python3 scripts/public_contract_smoke_artifact_qa.py",
    "python3 scripts/backend_acceptance_artifact_qa.py",
    "python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py",
    "python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py",
    "python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py",
)
WRITE_TOKEN_PERMISSION_PATTERN = re.compile(
    r"^\s*(?:"
    r"actions|attestations|checks|contents|deployments|discussions|id-token|"
    r"issues|models|packages|pages|pull-requests|security-events|statuses"
    r")\s*:\s*write\s*(?:#.*)?$"
)
SECRET_CONTEXT_PATTERN = re.compile(r"\$\{\{\s*secrets\.", re.IGNORECASE)
TOKEN_CONTEXT_PATTERN = re.compile(r"\$\{\{\s*github\.token\s*\}\}", re.IGNORECASE)
SENSITIVE_ENV_PATTERN = re.compile(
    r"^\s*(?:"
    r"GITHUB_TOKEN|GH_TOKEN|NPM_TOKEN|CARGO_REGISTRY_TOKEN|"
    r"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY"
    r")\s*:",
    re.IGNORECASE,
)
MAX_JOB_TIMEOUT_MINUTES = 30


def top_level_concurrency_block(text: str) -> list[str] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^concurrency:\s*(?:#.*)?$", line):
            block: list[str] = []
            for child in lines[index + 1 :]:
                if child and not child.startswith((" ", "\t")):
                    break
                if child.strip():
                    block.append(child)
            return block
    return None


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


def job_timeout_failures(text: str) -> list[str]:
    failures: list[str] = []
    lines = text.splitlines()
    in_jobs = False
    current_job: str | None = None
    saw_timeout = False

    def finish_job() -> None:
        if current_job is not None and not saw_timeout:
            failures.append(f"default CI job {current_job!r} must set timeout-minutes")

    for line_number, line in enumerate(lines, start=1):
        if re.match(r"^jobs:\s*(?:#.*)?$", line):
            in_jobs = True
            continue
        if in_jobs and line and not line.startswith((" ", "\t")):
            finish_job()
            in_jobs = False
            break
        if not in_jobs:
            continue

        job_match = re.match(r"^  ([A-Za-z0-9_-]+):\s*(?:#.*)?$", line)
        if job_match:
            finish_job()
            current_job = job_match.group(1)
            saw_timeout = False
            continue

        timeout_match = re.match(r"^    timeout-minutes:\s*([0-9]+)\s*(?:#.*)?$", line)
        if timeout_match and current_job is not None:
            saw_timeout = True
            timeout_minutes = int(timeout_match.group(1))
            if timeout_minutes <= 0 or timeout_minutes > MAX_JOB_TIMEOUT_MINUTES:
                failures.append(
                    "default CI job "
                    f"{current_job!r} timeout-minutes must be 1-{MAX_JOB_TIMEOUT_MINUTES}, "
                    f"line {line_number}: {timeout_minutes}"
                )

    if in_jobs:
        finish_job()
    return failures


def evaluate_ci_text(text: str) -> list[str]:
    failures: list[str] = []

    failures.extend(job_timeout_failures(text))

    if SECRET_CONTEXT_PATTERN.search(text):
        failures.append("default CI must not reference GitHub Actions secrets")
    if TOKEN_CONTEXT_PATTERN.search(text):
        failures.append("default CI must not reference github.token directly")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if SENSITIVE_ENV_PATTERN.match(line):
            failures.append(
                f"default CI must not define sensitive token environment variables, line {line_number}: {line.strip()}"
            )

    concurrency_block = top_level_concurrency_block(text)
    if concurrency_block is None:
        failures.append("default CI must set top-level concurrency cancellation")
    else:
        concurrency_text = "\n".join(concurrency_block)
        if not re.search(
            r"^\s*group:\s*\$\{\{\s*github\.workflow\s*\}\}-\$\{\{\s*github\.ref\s*\}\}\s*(?:#.*)?$",
            concurrency_text,
            re.MULTILINE,
        ):
            failures.append("default CI concurrency group must scope to workflow and ref")
        if not re.search(r"^\s*cancel-in-progress:\s*true\s*(?:#.*)?$", concurrency_text, re.MULTILINE):
            failures.append("default CI concurrency must cancel in-progress runs")

    if re.search(r"(?m)^\s*-?\s*pull_request_target\s*:", text) or re.search(
        r"(?m)^\s*on\s*:\s*\[[^\]]*\bpull_request_target\b", text
    ):
        failures.append("default CI must not use pull_request_target")

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
    saw_artifact_qa_commands = {command: False for command in REQUIRED_ARTIFACT_QA_COMMANDS}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if re.match(r"uses:\s*actions/checkout@v[0-9]+", stripped):
            lookahead = "\n".join(text.splitlines()[line_number : line_number + 8])
            if not re.search(r"^\s*persist-credentials:\s*false\s*(?:#.*)?$", lookahead, re.MULTILINE):
                failures.append(
                    "default CI checkout must set persist-credentials: false, "
                    f"line {line_number}: {stripped}"
                )
        if re.match(r"uses:\s*actions/setup-node@v[0-9]+", stripped):
            lookahead = "\n".join(text.splitlines()[line_number : line_number + 8])
            if re.search(r"^\s*cache\s*:", lookahead, re.MULTILINE):
                if not re.search(r"^\s*cache:\s*npm\s*(?:#.*)?$", lookahead, re.MULTILINE):
                    failures.append(
                        "default CI setup-node cache must use npm only, "
                        f"line {line_number}: {stripped}"
                    )
                if not re.search(
                    r"^\s*cache-dependency-path:\s*frontend/package-lock\.json\s*(?:#.*)?$",
                    lookahead,
                    re.MULTILINE,
                ):
                    failures.append(
                        "default CI setup-node cache must be scoped to frontend/package-lock.json, "
                        f"line {line_number}: {stripped}"
                    )
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
        for command in saw_artifact_qa_commands:
            if re.search(rf"\b{re.escape(command)}\b", stripped):
                saw_artifact_qa_commands[command] = True

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
    for command, saw_command in saw_artifact_qa_commands.items():
        if not saw_command:
            failures.append(f"default CI must run offline artifact QA: {command}")

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
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  rust:
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      run: cargo test -q
      run: bash scripts/public_api_contract_smoke.sh
  static-safety:
    timeout-minutes: 15
    steps:
      - run: |
          bash -n scripts/public_api_contract_smoke.sh
          bash -n scripts/backend_acceptance_smoke.sh
      - run: python3 scripts/ci_static_policy.py --self-test
      - run: git diff --check
      - run: python3 scripts/public_contract_smoke_artifact_qa.py
      - run: python3 scripts/backend_acceptance_artifact_qa.py
      - run: |
          python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py
          python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py
          python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py
      - run: bash scripts/public_risk_scan.sh --self-test
      - run: bash scripts/public_risk_scan.sh
      - run: echo done
"""
    assert_policy_passes(valid)

    cases = {
        "missing token permissions": "run: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "missing concurrency": valid.replace(
            "concurrency:\n  group: ${{ github.workflow }}-${{ github.ref }}\n  cancel-in-progress: true\n",
            "",
        ),
        "wrong concurrency group": valid.replace(
            "  group: ${{ github.workflow }}-${{ github.ref }}\n",
            "  group: ${{ github.workflow }}\n",
        ),
        "missing concurrency cancellation": valid.replace("  cancel-in-progress: true\n", ""),
        "missing job timeout": valid.replace("    timeout-minutes: 30\n", ""),
        "too-large job timeout": valid.replace("    timeout-minutes: 30\n", "    timeout-minutes: 60\n"),
        "secrets context": valid + "      - run: echo ${{ secrets.NPM_TOKEN }}\n",
        "github token context": valid + "      - run: echo ${{ github.token }}\n",
        "sensitive token env": valid + "env:\n  CARGO_REGISTRY_TOKEN: placeholder\n",
        "write-all token permissions": "permissions: write-all\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "read-all token permissions": "permissions: read-all\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "contents write token permission": "permissions:\n  contents: write\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "id-token write token permission": "permissions:\n  contents: read\n  id-token: write\nrun: bash scripts/public_api_contract_smoke.sh\nrun: python3 scripts/ci_static_policy.py --self-test",
        "privileged pull_request_target trigger": "permissions:\n  contents: read\non:\n  pull_request_target:\nrun: bash scripts/public_api_contract_smoke.sh",
        "privileged pull_request_target list trigger": "permissions:\n  contents: read\non: [push, pull_request_target]\nrun: bash scripts/public_api_contract_smoke.sh",
        "checkout persisted credentials": "permissions:\n  contents: read\nuses: actions/checkout@v4\nrun: bash scripts/public_api_contract_smoke.sh",
        "setup-node cache without dependency path": "permissions:\n  contents: read\nuses: actions/setup-node@v4\nwith:\n  cache: npm\nrun: bash scripts/public_api_contract_smoke.sh",
        "setup-node broad dependency path": "permissions:\n  contents: read\nuses: actions/setup-node@v4\nwith:\n  cache: npm\n  cache-dependency-path: package-lock.json\nrun: bash scripts/public_api_contract_smoke.sh",
        "setup-node non-npm cache": "permissions:\n  contents: read\nuses: actions/setup-node@v4\nwith:\n  cache: yarn\n  cache-dependency-path: frontend/package-lock.json\nrun: bash scripts/public_api_contract_smoke.sh",
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
        "missing public contract artifact QA": valid.replace(
            "      - run: python3 scripts/public_contract_smoke_artifact_qa.py\n", ""
        ),
        "missing backend artifact QA": valid.replace(
            "      - run: python3 scripts/backend_acceptance_artifact_qa.py\n", ""
        ),
        "missing optional artifact QA": valid.replace(
            "          python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py\n", ""
        ),
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
