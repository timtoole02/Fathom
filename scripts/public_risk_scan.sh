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
        ("machine-local volume path", re.compile(re.escape(volume_root_pattern), re.IGNORECASE)),
        ("private workspace path", re.compile(r"\.openclaw", re.IGNORECASE)),
    ]

def docs_or_evidence_local_path_patterns():
    return [
        ("machine-local volume path in public docs/evidence", re.compile(re.escape(volume_root_pattern), re.IGNORECASE)),
        ("machine-local tmp path in public docs/evidence", re.compile(generic_tmp_pattern, re.IGNORECASE)),
        ("machine-local model/reference checkout path in public docs/evidence", re.compile(r"\b(llama\.cpp|model-store|models/)\b.*(/Users/|/Vol|/t" + "mp/|\.openclaw)", re.IGNORECASE)),
    ]

def secret_value_patterns():
    return [
        ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")),
        ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
        ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
        ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
        ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
        ("Bearer token value", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE)),
        ("PEM private key block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
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
max_tracked_file_bytes = 1024 * 1024
blocked_tracked_filenames = {".DS_Store", "Thumbs.db", "desktop.ini"}
allowed_tracked_credential_filenames = {".env.example"}
blocked_tracked_credential_filenames = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
}
blocked_tracked_credential_suffixes = {
    ".key",
    ".p12",
    ".pem",
    ".pfx",
}
blocked_tracked_workspace_filenames = {
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "MEMORY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}
blocked_tracked_runtime_artifact_filenames = {
    "server.log",
    "summary.local.json",
}
blocked_tracked_runtime_artifact_suffixes = {
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".log",
    ".sqlite",
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3",
}
blocked_tracked_python_artifact_dirs = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
blocked_tracked_python_artifact_suffixes = {
    ".pyc",
    ".pyo",
}
blocked_tracked_frontend_artifact_dirs = {
    ".next",
    ".parcel-cache",
    ".turbo",
    ".vite",
    "coverage",
    "node_modules",
}
blocked_tracked_frontend_build_dirs = {
    "build",
    "dist",
}
blocked_tracked_frontend_artifact_filenames = {
    "npm-debug.log",
    "pnpm-debug.log",
    "yarn-debug.log",
    "yarn-error.log",
}
blocked_tracked_frontend_artifact_suffixes = {
    ".tsbuildinfo",
}
blocked_tracked_rust_artifact_dirs = {
    "target",
}
blocked_tracked_rust_artifact_suffixes = {
    ".a",
    ".dll",
    ".dylib",
    ".o",
    ".obj",
    ".profdata",
    ".profraw",
    ".rlib",
    ".rmeta",
    ".so",
}
tracked_symlink_mode = "120000"
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
            for label, pattern in secret_value_patterns():
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
        if path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        items.append((rel, text))
    return items

def tracked_large_file_failures(tracked_paths=None, sizes=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.Path(rel)
        try:
            if path.is_symlink():
                continue
            size = sizes[rel] if sizes is not None else path.stat().st_size
        except (FileNotFoundError, KeyError):
            continue
        if size > max_tracked_file_bytes:
            failures.append(f"{rel}: tracked file is {size} bytes; public launch tracked files must stay <= {max_tracked_file_bytes} bytes")
    return failures

def tracked_blocked_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        if pathlib.PurePosixPath(rel).name in blocked_tracked_filenames:
            failures.append(f"{rel}: OS/editor metadata files must not be tracked for public launch")
    return failures

def tracked_credential_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        name = path.name
        if name in allowed_tracked_credential_filenames:
            continue
        if name in blocked_tracked_credential_filenames or name.startswith(".env."):
            failures.append(f"{rel}: credential/config files must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_credential_suffixes:
            failures.append(f"{rel}: credential/key files must not be tracked for public launch")
    return failures

def tracked_workspace_context_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.name in blocked_tracked_workspace_filenames or rel.startswith("memory/"):
            failures.append(f"{rel}: workspace/personal agent context files must not be tracked for public launch")
    return failures

def tracked_runtime_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] == ".fathom":
            failures.append(f"{rel}: local Fathom runtime state must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_runtime_artifact_filenames:
            failures.append(f"{rel}: local runtime/artifact detail files must not be tracked for public launch")
            continue
        if any(rel.endswith(suffix) for suffix in blocked_tracked_runtime_artifact_suffixes):
            failures.append(f"{rel}: local runtime/artifact detail files must not be tracked for public launch")
    return failures

def tracked_python_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_python_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: Python cache/build artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_python_artifact_suffixes:
            failures.append(f"{rel}: Python cache/build artifacts must not be tracked for public launch")
    return failures

def tracked_frontend_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_frontend_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
            continue
        if path.parts and path.parts[0] == "frontend" and any(part in blocked_tracked_frontend_build_dirs for part in path.parts[1:]):
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_frontend_artifact_filenames:
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_frontend_artifact_suffixes:
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
    return failures

def tracked_rust_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_rust_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: Rust/Cargo cache/build artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_rust_artifact_suffixes:
            failures.append(f"{rel}: Rust/Cargo cache/build artifacts must not be tracked for public launch")
    return failures

def tracked_index_entries():
    entries = []
    for line in subprocess.check_output(["git", "ls-files", "-s"], text=True).splitlines():
        metadata, rel = line.split("\t", 1)
        mode, _blob_sha, _stage = metadata.split()
        entries.append((mode, rel))
    return entries

def tracked_symlink_failures(tracked_entries=None, targets=None):
    if tracked_entries is None:
        tracked_entries = tracked_index_entries()
    failures = []
    for mode, rel in tracked_entries:
        if mode != tracked_symlink_mode:
            continue
        try:
            target = targets[rel] if targets is not None else pathlib.Path(rel).readlink().as_posix()
        except (FileNotFoundError, KeyError, OSError):
            failures.append(f"{rel}: tracked symlink target could not be inspected")
            continue
        target_path = pathlib.PurePosixPath(target)
        if target.startswith("~") or target_path.is_absolute() or ".." in target_path.parts:
            failures.append(f"{rel}: tracked symlink must use a relative in-repository target, found {target}")
    return failures

def self_test():
    bad_lines = [
        ("README.md", "Maintainer: " + "Ti" + "m Too" + "le"),
        ("README.md", "Local maintainer user: " + personal_user),
        ("docs/api/example.md", "Tool path: " + homebrew_pattern + "/bin/fathom-helper"),
        ("docs/api/example.md", "Workspace cache: .openclaw/workspace/artifacts"),
        ("docs/benchmarks/example.md", "Reference repo: /Users/example/.openclaw/workspace/projects/llama.cpp"),
        ("docs/benchmarks/example.md", "Model: /Volumes/External/models/model.gguf"),
        ("scripts/example.py", "DEFAULT_MODEL = Path('/Volumes/External/models/model.gguf')"),
        ("docs/api/example.md", "Artifact: /tmp/private-output.json"),
        ("docs/public-launch-evidence.md", "Binary: /path/then/model-store under /Users/example"),
        ("docs/api/example.md", "Authorization: Bearer secret-token-value-123456"),
        ("README.md", "OPENAI_API_KEY=sk-this-is-not-share-safe"),
        ("docs/api/example.md", "Hugging Face token: hf_abcdefghijklmnopqrstuvwxyz"),
        ("docs/api/example.md", "GitHub token: ghp_abcdefghijklmnopqrstuvwxyz"),
        ("docs/api/example.md", "-----BEGIN OPENSSH PRIVATE KEY-----"),
        ("docs/api/example.md", "Fathom includes a GGUF runtime for local inference."),
        ("docs/api/example.md", "Fathom uses torch.load to inspect PyTorch weights."),
    ]
    allowed_lines = [
        ("docs/benchmarks/example.md", "Reference repo: local llama.cpp checkout"),
        ("docs/benchmarks/example.md", "Binary: /path/to/llama.cpp/build/bin/llama-tokenize"),
        ("README.md", "Canonical repo: https://github.com/" + personal_owner + "/Fathom/"),
        ("docs/api/example.md", "Use API key placeholders such as placeholder-key or fathom-local."),
        ("docs/api/example.md", "-----BEGIN PUBLIC KEY-----"),
        ("docs/api/example.md", "No GGUF runtime, tokenizer execution, or generation claim is made."),
    ]
    failures = scan_items(bad_lines)
    if len(failures) < len(bad_lines):
        raise AssertionError("public risk self-test did not reject all local path examples")
    allowed_failures = scan_items(allowed_lines)
    if allowed_failures:
        raise AssertionError("public risk self-test rejected allowed examples:\n" + "\n".join(allowed_failures))
    large_file_failures = tracked_large_file_failures(
        tracked_paths=["docs/api/small.md", "docs/api/large.bin"],
        sizes={"docs/api/small.md": max_tracked_file_bytes, "docs/api/large.bin": max_tracked_file_bytes + 1},
    )
    if large_file_failures != [
        f"docs/api/large.bin: tracked file is {max_tracked_file_bytes + 1} bytes; public launch tracked files must stay <= {max_tracked_file_bytes} bytes"
    ]:
        raise AssertionError("public risk self-test did not reject oversized tracked files")
    blocked_file_failures = tracked_blocked_file_failures(
        tracked_paths=["docs/api/public-contract.json", "docs/.DS_Store", "frontend/Thumbs.db", "desktop.ini"],
    )
    if blocked_file_failures != [
        "docs/.DS_Store: OS/editor metadata files must not be tracked for public launch",
        "frontend/Thumbs.db: OS/editor metadata files must not be tracked for public launch",
        "desktop.ini: OS/editor metadata files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked OS/editor metadata files")
    credential_file_failures = tracked_credential_file_failures(
        tracked_paths=[
            ".env",
            ".env.local",
            ".env.example",
            ".npmrc",
            "docs/public.pem",
            "docs/public-key.txt",
            "frontend/.pypirc",
            "crates/fathom-server/credentials",
        ],
    )
    if credential_file_failures != [
        ".env: credential/config files must not be tracked for public launch",
        ".env.local: credential/config files must not be tracked for public launch",
        ".npmrc: credential/config files must not be tracked for public launch",
        "docs/public.pem: credential/key files must not be tracked for public launch",
        "frontend/.pypirc: credential/config files must not be tracked for public launch",
        "crates/fathom-server/credentials: credential/config files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked credential/config files")
    workspace_context_failures = tracked_workspace_context_failures(
        tracked_paths=[
            "AGENTS.md",
            "docs/AGENTS.md",
            "MEMORY.md",
            "memory/2026-05-31.md",
            "docs/api/public-contract.json",
        ],
    )
    if workspace_context_failures != [
        "AGENTS.md: workspace/personal agent context files must not be tracked for public launch",
        "docs/AGENTS.md: workspace/personal agent context files must not be tracked for public launch",
        "MEMORY.md: workspace/personal agent context files must not be tracked for public launch",
        "memory/2026-05-31.md: workspace/personal agent context files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked workspace/personal agent context files")
    runtime_artifact_failures = tracked_runtime_artifact_file_failures(
        tracked_paths=[
            ".fathom/state/registry.json",
            "docs/api/public-contract.json",
            "logs/server.log",
            "public-contract-artifacts/summary.local.json",
            "state/fathom.sqlite",
            "state/fathom.sqlite-wal",
            "frontend/package-lock.json",
        ],
    )
    if runtime_artifact_failures != [
        ".fathom/state/registry.json: local Fathom runtime state must not be tracked for public launch",
        "logs/server.log: local runtime/artifact detail files must not be tracked for public launch",
        "public-contract-artifacts/summary.local.json: local runtime/artifact detail files must not be tracked for public launch",
        "state/fathom.sqlite: local runtime/artifact detail files must not be tracked for public launch",
        "state/fathom.sqlite-wal: local runtime/artifact detail files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local runtime/artifact files")
    python_artifact_failures = tracked_python_artifact_file_failures(
        tracked_paths=[
            "scripts/__pycache__/public_api_contract_qa.cpython-312.pyc",
            ".pytest_cache/v/cache/nodeids",
            ".ruff_cache/0.12.0/file",
            "docs/api/public-contract.json",
            "scripts/public_api_contract_qa.py",
        ],
    )
    if python_artifact_failures != [
        "scripts/__pycache__/public_api_contract_qa.cpython-312.pyc: Python cache/build artifacts must not be tracked for public launch",
        ".pytest_cache/v/cache/nodeids: Python cache/build artifacts must not be tracked for public launch",
        ".ruff_cache/0.12.0/file: Python cache/build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Python cache/build artifacts")
    frontend_artifact_failures = tracked_frontend_artifact_file_failures(
        tracked_paths=[
            "frontend/node_modules/.package-lock.json",
            "frontend/dist/index.html",
            "frontend/build/assets/app.js",
            "frontend/coverage/lcov.info",
            "frontend/.next/server/app.js",
            "frontend/.vite/deps/react.js",
            ".turbo/cache/build.log",
            ".parcel-cache/data.mdb",
            "frontend/src/tsconfig.tsbuildinfo",
            "npm-debug.log",
            "yarn-error.log",
            "pnpm-debug.log",
            "frontend/package-lock.json",
            "frontend/yarn.lock",
            "frontend/pnpm-lock.yaml",
            "frontend/package.json",
            "frontend/vite.config.js",
            "frontend/src/App.jsx",
        ],
    )
    if frontend_artifact_failures != [
        "frontend/node_modules/.package-lock.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/dist/index.html: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/build/assets/app.js: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/coverage/lcov.info: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.next/server/app.js: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.vite/deps/react.js: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".turbo/cache/build.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".parcel-cache/data.mdb: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/src/tsconfig.tsbuildinfo: frontend/Node cache/build artifacts must not be tracked for public launch",
        "npm-debug.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        "yarn-error.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        "pnpm-debug.log: frontend/Node cache/build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked frontend/Node cache/build artifacts")
    rust_artifact_failures = tracked_rust_artifact_file_failures(
        tracked_paths=[
            "target/debug/fathom",
            "crates/fathom-core/target/release/deps/libfathom_core.rlib",
            "crates/fathom-core/target/release/deps/fathom_core.rmeta",
            "crates/fathom-core/target/debug/incremental/state.o",
            "crates/fathom-core/target/debug/incremental/state.obj",
            "crates/fathom-server/target/debug/deps/libserver.so",
            "crates/fathom-server/target/debug/deps/libserver.dylib",
            "crates/fathom-server/target/debug/deps/server.dll",
            "crates/fathom-core/target/debug/deps/libnative.a",
            "coverage/default.profraw",
            "coverage/merged.profdata",
            "Cargo.lock",
            "crates/fathom-core/Cargo.toml",
            "docs/research/performance-strategy.md",
            "src/main.rs",
        ],
    )
    if rust_artifact_failures != [
        "target/debug/fathom: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-core/target/release/deps/libfathom_core.rlib: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-core/target/release/deps/fathom_core.rmeta: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-core/target/debug/incremental/state.o: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-core/target/debug/incremental/state.obj: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-server/target/debug/deps/libserver.so: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-server/target/debug/deps/libserver.dylib: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-server/target/debug/deps/server.dll: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "crates/fathom-core/target/debug/deps/libnative.a: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "coverage/default.profraw: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        "coverage/merged.profdata: Rust/Cargo cache/build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Rust/Cargo cache/build artifacts")
    symlink_failures = tracked_symlink_failures(
        tracked_entries=[
            ("100644", "README.md"),
            (tracked_symlink_mode, "docs/internal-home"),
            (tracked_symlink_mode, "docs/private-temp"),
            (tracked_symlink_mode, "docs/parent"),
            (tracked_symlink_mode, "docs/public-contract-link"),
        ],
        targets={
            "docs/internal-home": "/Users/example/private-notes.md",
            "docs/private-temp": "~/private-output.json",
            "docs/parent": "../outside-repo.md",
            "docs/public-contract-link": "api/public-contract.json",
        },
    )
    if symlink_failures != [
        "docs/internal-home: tracked symlink must use a relative in-repository target, found /Users/example/private-notes.md",
        "docs/private-temp: tracked symlink must use a relative in-repository target, found ~/private-output.json",
        "docs/parent: tracked symlink must use a relative in-repository target, found ../outside-repo.md",
    ]:
        raise AssertionError("public risk self-test did not reject symlinks escaping the repository")
    print("public risk scan self-test passed")

if "--self-test" in sys.argv[1:]:
    self_test()
    raise SystemExit(0)

failures = scan_items(tracked_items())
failures.extend(tracked_large_file_failures())
failures.extend(tracked_blocked_file_failures())
failures.extend(tracked_credential_file_failures())
failures.extend(tracked_workspace_context_failures())
failures.extend(tracked_runtime_artifact_file_failures())
failures.extend(tracked_python_artifact_file_failures())
failures.extend(tracked_frontend_artifact_file_failures())
failures.extend(tracked_rust_artifact_file_failures())
failures.extend(tracked_symlink_failures())
if failures:
    print("Public risk scan failed:", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print("public risk scan passed")
PY
