#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$@" <<'PY'
import pathlib
import re
import subprocess
import sys

# Keep scanner patterns constructed where practical so ordinary greps of this
# file do not become the thing they are meant to catch. The script itself is
# skipped during normal repository scans.
personal_user = "".join(["ti", "m", "to", "ole"])
personal_owner = personal_user + "02"
personal_name_pattern = r"\b" + "Ti" + "m" + r"\s+" + "Too" + "le" + r"\b"
personal_machine_pattern = "|".join(["Ti" + "ms-M" + "ac", "M" + "ac m" + "ini", "m" + "ac-m" + "ini"])
private_temp_pattern = "|".join(["/pri" + "vate/t" + "mp", "/var/fo" + "lders"])
homebrew_pattern = "/opt/home" + "brew"
volume_root_pattern = "/Vol" + "umes/"
generic_tmp_pattern = r"(?<![\w.-])/t" + "mp/(?!fathom[-_/])"

public_repo_urls = [
    "https://github.com/" + personal_owner + "/Fathom/",
    "https://github.com/" + personal_owner + "/Fathom.git",
]

def privacy_patterns():
    return [
        ("personal GitHub owner", re.compile(r"github\.com/" + re.escape(personal_owner), re.IGNORECASE)),
        ("personal username", re.compile(re.escape(personal_user), re.IGNORECASE)),
        ("personal name", re.compile(personal_name_pattern, re.IGNORECASE)),
        ("personal machine hostname", re.compile(personal_machine_pattern, re.IGNORECASE)),
        ("personal home path", re.compile(r"/Users/[^\s`'\"]+")),
        ("private macOS temp path", re.compile(private_temp_pattern, re.IGNORECASE)),
        ("machine-local Homebrew path", re.compile(re.escape(homebrew_pattern), re.IGNORECASE)),
        ("private workspace path", re.compile(r"\.openclaw", re.IGNORECASE)),
    ]

def docs_or_evidence_local_path_patterns():
    return [
        ("machine-local volume path in public docs/evidence", re.compile(re.escape(volume_root_pattern), re.IGNORECASE)),
        ("machine-local tmp path in public docs/evidence", re.compile(generic_tmp_pattern, re.IGNORECASE)),
        ("machine-local model/reference checkout path in public docs/evidence", re.compile(r"\b(llama\.cpp|model-store|models/)\b.*(/Users/|/Vol|/t" + "mp/|\.openclaw)", re.IGNORECASE)),
    ]

claim_patterns = [
    ("uncaveated GGUF runtime claim", re.compile(r"\bGGUF runtime\b(?!, tokenizer execution, or generation claim)", re.IGNORECASE)),
    ("uncaveated ONNX chat claim", re.compile(r"\bONNX chat\b(?! or general ONNX support claim)", re.IGNORECASE)),
    ("arbitrary SafeTensors claim", re.compile(r"\barbitrary SafeTensors support\b(?! claim)", re.IGNORECASE)),
    ("unsafe torch load", re.compile(r"\btorch\.load\b")),
    ("unsafe pickle load", re.compile(r"\bpickle\.load\b")),
]

skip_suffixes = {".lock"}
skip_paths = {"scripts/public_risk_scan.sh", "frontend/scripts/ui-copy-qa.mjs"}
docs_evidence_prefixes = (
    "docs/benchmarks/",
    "docs/api/",
)
docs_evidence_paths = {"docs/public-launch-evidence.md", "docs/public-launch-checklist.md"}

def scan_items(items):
    failures = []
    for rel, text in items:
        for line_no, line in enumerate(text.splitlines(), 1):
            privacy_line = line
            for public_repo_url in public_repo_urls:
                privacy_line = privacy_line.replace(public_repo_url, "")
            for label, pattern in privacy_patterns():
                if pattern.search(privacy_line):
                    failures.append(f"{rel}:{line_no}: {label}: {line.strip()}")
            if rel in docs_evidence_paths or rel.startswith(docs_evidence_prefixes):
                for label, pattern in docs_or_evidence_local_path_patterns():
                    if pattern.search(privacy_line):
                        failures.append(f"{rel}:{line_no}: {label}: {line.strip()}")
            lowered = line.lower()
            caveated = any(marker in lowered for marker in ["no ", "not ", "unsupported", "blocked", "refused", "metadata-only", "does not", "without claiming"])
            for label, pattern in claim_patterns:
                if pattern.search(line) and not caveated:
                    failures.append(f"{rel}:{line_no}: {label}: {line.strip()}")
    return failures

def tracked_items():
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    items = []
    for rel in tracked:
        if rel in skip_paths or pathlib.Path(rel).suffix in skip_suffixes:
            continue
        path = pathlib.Path(rel)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        items.append((rel, text))
    return items

def self_test():
    bad_lines = [
        ("docs/benchmarks/example.md", "Reference repo: /Users/example/.openclaw/workspace/projects/llama.cpp"),
        ("docs/benchmarks/example.md", "Model: /Volumes/External/models/model.gguf"),
        ("docs/api/example.md", "Artifact: /tmp/private-output.json"),
        ("docs/public-launch-evidence.md", "Binary: /path/then/model-store under /Users/example"),
    ]
    allowed_lines = [
        ("docs/benchmarks/example.md", "Reference repo: local llama.cpp checkout"),
        ("docs/benchmarks/example.md", "Binary: /path/to/llama.cpp/build/bin/llama-tokenize"),
        ("README.md", "Canonical repo: https://github.com/" + personal_owner + "/Fathom/"),
    ]
    failures = scan_items(bad_lines)
    if len(failures) < len(bad_lines):
        raise AssertionError("public risk self-test did not reject all local path examples")
    allowed_failures = scan_items(allowed_lines)
    if allowed_failures:
        raise AssertionError("public risk self-test rejected allowed examples:\n" + "\n".join(allowed_failures))
    print("public risk scan self-test passed")

if "--self-test" in sys.argv[1:]:
    self_test()
    raise SystemExit(0)

failures = scan_items(tracked_items())
if failures:
    print("Public risk scan failed:", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print("public risk scan passed")
PY
