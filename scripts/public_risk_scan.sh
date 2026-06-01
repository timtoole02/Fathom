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
blocked_tracked_os_metadata_filenames = {
    ".AppleDouble",
    ".DS_Store",
    ".localized",
    ".LSOverride",
    "desktop.ini",
    "Thumbs.db",
}
blocked_tracked_os_metadata_dirs = {
    "__MACOSX",
    ".AppleDouble",
}
blocked_tracked_editor_artifact_suffixes = {
    ".orig",
    ".rej",
    ".swo",
    ".swp",
}
blocked_tracked_ide_artifact_dirs = {
    ".idea",
    ".vscode",
}
blocked_tracked_ide_artifact_suffixes = {
    ".code-workspace",
}
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
    "IDENTITY.md",
    "MEMORY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}
required_workspace_gitignore_patterns = {
    "/.openclaw/",
    "/memory/",
    "/AGENTS.md",
    "/BOOTSTRAP.md",
    "/HEARTBEAT.md",
    "/IDENTITY.md",
    "/MEMORY.md",
    "/SOUL.md",
    "/TOOLS.md",
    "/USER.md",
}
required_credential_gitignore_patterns = {
    ".env",
    ".env.*",
    "!.env.example",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    ".netrc",
    ".npmrc",
    ".pypirc",
}
required_model_artifact_gitignore_patterns = {
    "/checkpoints/",
    "/model-store/",
    "/models/",
    "/weights/",
    "*.bin",
    "*.ckpt",
    "*.gguf",
    "*.npy",
    "*.npz",
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.tflite",
}
required_container_artifact_gitignore_patterns = {
    "/.docker/",
    "/docker-data/",
    "/docker-volumes/",
    "compose.override.yaml",
    "compose.override.yml",
    "docker-compose.override.yaml",
    "docker-compose.override.yml",
}
required_infra_state_gitignore_patterns = {
    "/.terraform/",
    ".terraform.lock.hcl",
    "*.tfplan",
    "*.tfstate",
    "*.tfstate.*",
    "*.tfvars",
    "*.tfvars.json",
}
required_mobile_build_gitignore_patterns = {
    "/.gradle/",
    "/DerivedData/",
    "*.aab",
    "*.apk",
    "*.dSYM",
    "*.ipa",
    "*.xcresult",
    "*.xcuserstate",
    "local.properties",
    "xcuserdata/",
}
required_package_artifact_gitignore_patterns = {
    "/artifacts/",
    "/release/",
    "/releases/",
    "*.7z",
    "*.app",
    "*.bz2",
    "*.dmg",
    "*.egg",
    "*.gz",
    "*.pkg",
    "*.tar",
    "*.tgz",
    "*.whl",
    "*.xz",
    "*.zip",
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
blocked_tracked_root_frontend_build_dirs = {
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
blocked_tracked_package_artifact_suffixes = {
    ".7z",
    ".app",
    ".bz2",
    ".dmg",
    ".egg",
    ".gz",
    ".pkg",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
}
blocked_tracked_package_artifact_dirs = {
    "artifacts",
    "release",
    "releases",
}
blocked_tracked_backup_artifact_dirs = {
    "backups",
    "dumps",
}
blocked_tracked_backup_artifact_suffixes = {
    ".bak",
    ".backup",
    ".dump",
    ".sql",
}
blocked_tracked_model_artifact_dirs = {
    "checkpoints",
    "model-store",
    "models",
    "weights",
}
blocked_tracked_model_artifact_suffixes = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}
blocked_tracked_container_artifact_dirs = {
    ".docker",
    "docker-data",
    "docker-volumes",
}
blocked_tracked_container_artifact_filenames = {
    "compose.override.yaml",
    "compose.override.yml",
    "docker-compose.override.yaml",
    "docker-compose.override.yml",
}
blocked_tracked_infra_state_dirs = {
    ".terraform",
}
blocked_tracked_infra_state_filenames = {
    ".terraform.lock.hcl",
}
blocked_tracked_infra_state_suffixes = {
    ".tfplan",
    ".tfstate",
    ".tfvars",
}
blocked_tracked_mobile_build_dirs = {
    ".gradle",
    "DerivedData",
    "xcuserdata",
}
blocked_tracked_mobile_build_filenames = {
    "local.properties",
}
blocked_tracked_mobile_build_suffixes = {
    ".aab",
    ".apk",
    ".dSYM",
    ".ipa",
    ".xcresult",
    ".xcuserstate",
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
            if rel == ".gitignore" and line.strip() in required_workspace_gitignore_patterns:
                continue
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
        path = pathlib.PurePosixPath(rel)
        if (
            path.name in blocked_tracked_os_metadata_filenames
            or path.name.startswith("._")
            or any(part in blocked_tracked_os_metadata_dirs for part in path.parts)
        ):
            failures.append(f"{rel}: OS/platform metadata files must not be tracked for public launch")
            continue
        if path.name.endswith("~") or path.suffix.lower() in blocked_tracked_editor_artifact_suffixes:
            failures.append(f"{rel}: editor backup/swap artifacts must not be tracked for public launch")
            continue
        if any(part in blocked_tracked_ide_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: IDE workspace/config artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_ide_artifact_suffixes:
            failures.append(f"{rel}: IDE workspace/config artifacts must not be tracked for public launch")
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
        if path.name in blocked_tracked_workspace_filenames or rel.startswith(("memory/", ".openclaw/")):
            failures.append(f"{rel}: workspace/personal agent context files must not be tracked for public launch")
    return failures

def gitignore_workspace_context_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing workspace/personal context ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_workspace_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing workspace/personal context ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_credential_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local credential/config ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_credential_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local credential/config ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_model_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local model/checkpoint artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_model_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local model/checkpoint artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_container_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local container artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_container_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local container artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_infra_state_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local infrastructure state ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_infra_state_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local infrastructure state ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_mobile_build_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local mobile/Xcode/Android build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_mobile_build_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local mobile/Xcode/Android build artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_package_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local release/package artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_package_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local release/package artifact ignore patterns: {', '.join(missing)}"]
    return []

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
        if path.parts and path.parts[0] in blocked_tracked_root_frontend_build_dirs:
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

def tracked_package_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        lower_name = path.name.lower()
        if path.parts and path.parts[0] in blocked_tracked_package_artifact_dirs:
            failures.append(f"{rel}: release/package artifacts must not be tracked for public launch")
            continue
        if any(part.lower().endswith((".app", ".egg")) for part in path.parts):
            failures.append(f"{rel}: release/package artifacts must not be tracked for public launch")
            continue
        if lower_name.endswith(".tar.gz") or lower_name.endswith(".tar.bz2") or lower_name.endswith(".tar.xz"):
            failures.append(f"{rel}: release/package artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_package_artifact_suffixes:
            failures.append(f"{rel}: release/package artifacts must not be tracked for public launch")
    return failures

def tracked_backup_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] in blocked_tracked_backup_artifact_dirs:
            failures.append(f"{rel}: backup/dump artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_backup_artifact_suffixes:
            failures.append(f"{rel}: backup/dump artifacts must not be tracked for public launch")
    return failures

def tracked_model_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] in blocked_tracked_model_artifact_dirs:
            failures.append(f"{rel}: local model/checkpoint artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_model_artifact_suffixes:
            failures.append(f"{rel}: local model/checkpoint artifacts must not be tracked for public launch")
    return failures

def tracked_container_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] in blocked_tracked_container_artifact_dirs:
            failures.append(f"{rel}: local Docker/container artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_container_artifact_filenames:
            failures.append(f"{rel}: local Docker/container override files must not be tracked for public launch")
    return failures

def tracked_infra_state_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        lower_name = path.name.lower()
        if any(part in blocked_tracked_infra_state_dirs for part in path.parts):
            failures.append(f"{rel}: local infrastructure state artifacts must not be tracked for public launch")
            continue
        if lower_name in blocked_tracked_infra_state_filenames:
            failures.append(f"{rel}: local infrastructure state artifacts must not be tracked for public launch")
            continue
        if lower_name.endswith(".tfstate.backup") or lower_name.endswith(".tfvars.json"):
            failures.append(f"{rel}: local infrastructure state artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_infra_state_suffixes:
            failures.append(f"{rel}: local infrastructure state artifacts must not be tracked for public launch")
    return failures

def tracked_mobile_build_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_mobile_build_dirs for part in path.parts):
            failures.append(f"{rel}: local mobile/Xcode/Android build artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_mobile_build_filenames:
            failures.append(f"{rel}: local mobile/Xcode/Android build artifacts must not be tracked for public launch")
            continue
        if any(part.endswith((".dSYM", ".xcresult")) for part in path.parts):
            failures.append(f"{rel}: local mobile/Xcode/Android build artifacts must not be tracked for public launch")
            continue
        if path.suffix in blocked_tracked_mobile_build_suffixes:
            failures.append(f"{rel}: local mobile/Xcode/Android build artifacts must not be tracked for public launch")
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
        tracked_paths=[
            "docs/api/public-contract.json",
            "docs/.DS_Store",
            "docs/__MACOSX/._public-launch-evidence.md",
            "docs/.AppleDouble/public-launch-checklist.md",
            "frontend/._vite.config.ts",
            "frontend/.LSOverride",
            "frontend/.localized",
            "frontend/Thumbs.db",
            "desktop.ini",
            "README.md~",
            "docs/api/client-examples.md.swp",
            "docs/public-launch-checklist.md.orig",
            "docs/public-launch-evidence.md.rej",
            ".vscode/settings.json",
            ".idea/workspace.xml",
            "fathom.code-workspace",
        ],
    )
    if blocked_file_failures != [
        "docs/.DS_Store: OS/platform metadata files must not be tracked for public launch",
        "docs/__MACOSX/._public-launch-evidence.md: OS/platform metadata files must not be tracked for public launch",
        "docs/.AppleDouble/public-launch-checklist.md: OS/platform metadata files must not be tracked for public launch",
        "frontend/._vite.config.ts: OS/platform metadata files must not be tracked for public launch",
        "frontend/.LSOverride: OS/platform metadata files must not be tracked for public launch",
        "frontend/.localized: OS/platform metadata files must not be tracked for public launch",
        "frontend/Thumbs.db: OS/platform metadata files must not be tracked for public launch",
        "desktop.ini: OS/platform metadata files must not be tracked for public launch",
        "README.md~: editor backup/swap artifacts must not be tracked for public launch",
        "docs/api/client-examples.md.swp: editor backup/swap artifacts must not be tracked for public launch",
        "docs/public-launch-checklist.md.orig: editor backup/swap artifacts must not be tracked for public launch",
        "docs/public-launch-evidence.md.rej: editor backup/swap artifacts must not be tracked for public launch",
        ".vscode/settings.json: IDE workspace/config artifacts must not be tracked for public launch",
        ".idea/workspace.xml: IDE workspace/config artifacts must not be tracked for public launch",
        "fathom.code-workspace: IDE workspace/config artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked OS/platform metadata, backup/swap, or IDE config files")
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
            "IDENTITY.md",
            "MEMORY.md",
            "memory/2026-05-31.md",
            ".openclaw/session.json",
            "docs/api/public-contract.json",
        ],
    )
    if workspace_context_failures != [
        "AGENTS.md: workspace/personal agent context files must not be tracked for public launch",
        "docs/AGENTS.md: workspace/personal agent context files must not be tracked for public launch",
        "IDENTITY.md: workspace/personal agent context files must not be tracked for public launch",
        "MEMORY.md: workspace/personal agent context files must not be tracked for public launch",
        "memory/2026-05-31.md: workspace/personal agent context files must not be tracked for public launch",
        ".openclaw/session.json: workspace/personal agent context files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked workspace/personal agent context files")
    allowed_gitignore = "\n".join(sorted(required_workspace_gitignore_patterns)) + "\n"
    if gitignore_workspace_context_failures(allowed_gitignore):
        raise AssertionError("public risk self-test rejected complete workspace/personal context ignore patterns")
    gitignore_failures = gitignore_workspace_context_failures(allowed_gitignore.replace("/IDENTITY.md\n", ""))
    if gitignore_failures != [".gitignore: missing workspace/personal context ignore patterns: /IDENTITY.md"]:
        raise AssertionError("public risk self-test did not reject missing workspace/personal context ignore patterns")
    allowed_credential_gitignore = "\n".join(sorted(required_credential_gitignore_patterns)) + "\n"
    if gitignore_credential_failures(allowed_credential_gitignore):
        raise AssertionError("public risk self-test rejected complete local credential/config ignore patterns")
    credential_gitignore_failures = gitignore_credential_failures(allowed_credential_gitignore.replace(".npmrc\n", ""))
    if credential_gitignore_failures != [".gitignore: missing local credential/config ignore patterns: .npmrc"]:
        raise AssertionError("public risk self-test did not reject missing local credential/config ignore patterns")
    allowed_model_artifact_gitignore = "\n".join(sorted(required_model_artifact_gitignore_patterns)) + "\n"
    if gitignore_model_artifact_failures(allowed_model_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local model/checkpoint artifact ignore patterns")
    model_artifact_gitignore_failures = gitignore_model_artifact_failures(allowed_model_artifact_gitignore.replace("*.gguf\n", ""))
    if model_artifact_gitignore_failures != [".gitignore: missing local model/checkpoint artifact ignore patterns: *.gguf"]:
        raise AssertionError("public risk self-test did not reject missing local model/checkpoint artifact ignore patterns")
    allowed_container_artifact_gitignore = "\n".join(sorted(required_container_artifact_gitignore_patterns)) + "\n"
    if gitignore_container_artifact_failures(allowed_container_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local container artifact ignore patterns")
    container_artifact_gitignore_failures = gitignore_container_artifact_failures(
        allowed_container_artifact_gitignore.replace("docker-compose.override.yml\n", "")
    )
    if container_artifact_gitignore_failures != [
        ".gitignore: missing local container artifact ignore patterns: docker-compose.override.yml"
    ]:
        raise AssertionError("public risk self-test did not reject missing local container artifact ignore patterns")
    allowed_infra_state_gitignore = "\n".join(sorted(required_infra_state_gitignore_patterns)) + "\n"
    if gitignore_infra_state_failures(allowed_infra_state_gitignore):
        raise AssertionError("public risk self-test rejected complete local infrastructure state ignore patterns")
    infra_state_gitignore_failures = gitignore_infra_state_failures(
        allowed_infra_state_gitignore.replace("*.tfvars\n", "")
    )
    if infra_state_gitignore_failures != [".gitignore: missing local infrastructure state ignore patterns: *.tfvars"]:
        raise AssertionError("public risk self-test did not reject missing local infrastructure state ignore patterns")
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
            "dist/index.html",
            "build/assets/app.js",
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
        "dist/index.html: frontend/Node cache/build artifacts must not be tracked for public launch",
        "build/assets/app.js: frontend/Node cache/build artifacts must not be tracked for public launch",
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
    package_artifact_failures = tracked_package_artifact_file_failures(
        tracked_paths=[
            "artifacts/public-contract-smoke-summary.json",
            "dist/fathom.zip",
            "release/fathom",
            "releases/fathom-macos.dmg",
            "releases/fathom.pkg",
            "python/fathom-0.1.0-py3-none-any.whl",
            "python/fathom.egg",
            "Fathom.app/Contents/Info.plist",
            "snapshots/fathom.tar",
            "snapshots/fathom.tar.gz",
            "snapshots/fathom.tar.bz2",
            "snapshots/fathom.tar.xz",
            "snapshots/fathom.tgz",
            "snapshots/fathom.7z",
            "docs/api/public-contract.json",
            "frontend/package-lock.json",
            "Cargo.lock",
        ],
    )
    if package_artifact_failures != [
        "artifacts/public-contract-smoke-summary.json: release/package artifacts must not be tracked for public launch",
        "dist/fathom.zip: release/package artifacts must not be tracked for public launch",
        "release/fathom: release/package artifacts must not be tracked for public launch",
        "releases/fathom-macos.dmg: release/package artifacts must not be tracked for public launch",
        "releases/fathom.pkg: release/package artifacts must not be tracked for public launch",
        "python/fathom-0.1.0-py3-none-any.whl: release/package artifacts must not be tracked for public launch",
        "python/fathom.egg: release/package artifacts must not be tracked for public launch",
        "Fathom.app/Contents/Info.plist: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.gz: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.bz2: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.xz: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tgz: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.7z: release/package artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked release/package artifacts")
    allowed_package_artifact_gitignore = "\n".join(sorted(required_package_artifact_gitignore_patterns)) + "\n"
    if gitignore_package_artifact_failures(allowed_package_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local release/package artifact ignore patterns")
    package_artifact_gitignore_failures = gitignore_package_artifact_failures(
        allowed_package_artifact_gitignore.replace("*.whl\n", "")
    )
    if package_artifact_gitignore_failures != [
        ".gitignore: missing local release/package artifact ignore patterns: *.whl"
    ]:
        raise AssertionError("public risk self-test did not reject missing local release/package artifact ignore patterns")
    backup_artifact_failures = tracked_backup_artifact_file_failures(
        tracked_paths=[
            "backups/model-registry.json",
            "dumps/fathom-state.json",
            "state/models.json.bak",
            "state/models.json.backup",
            "state/fathom.dump",
            "state/fathom.sql",
            "docs/api/public-contract.json",
            "docs/research/runtime-safety-policy.md",
        ],
    )
    if backup_artifact_failures != [
        "backups/model-registry.json: backup/dump artifacts must not be tracked for public launch",
        "dumps/fathom-state.json: backup/dump artifacts must not be tracked for public launch",
        "state/models.json.bak: backup/dump artifacts must not be tracked for public launch",
        "state/models.json.backup: backup/dump artifacts must not be tracked for public launch",
        "state/fathom.dump: backup/dump artifacts must not be tracked for public launch",
        "state/fathom.sql: backup/dump artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked backup/dump artifacts")
    model_artifact_failures = tracked_model_artifact_file_failures(
        tracked_paths=[
            "models/tinystories/model.safetensors",
            "model-store/qwen/model.gguf",
            "weights/lora/adapter.bin",
            "checkpoints/run-042/model.ckpt",
            "fixtures/test.onnx",
            "fixtures/embed.npy",
            "fixtures/embed.npz",
            "fixtures/model.pt",
            "fixtures/model.pth",
            "fixtures/model.tflite",
            "docs/api/public-contract.json",
            "docs/research/model-format-landscape.md",
            "crates/fathom-server/src/main.rs",
        ],
    )
    if model_artifact_failures != [
        "models/tinystories/model.safetensors: local model/checkpoint artifacts must not be tracked for public launch",
        "model-store/qwen/model.gguf: local model/checkpoint artifacts must not be tracked for public launch",
        "weights/lora/adapter.bin: local model/checkpoint artifacts must not be tracked for public launch",
        "checkpoints/run-042/model.ckpt: local model/checkpoint artifacts must not be tracked for public launch",
        "fixtures/test.onnx: local model/checkpoint artifacts must not be tracked for public launch",
        "fixtures/embed.npy: local model/checkpoint artifacts must not be tracked for public launch",
        "fixtures/embed.npz: local model/checkpoint artifacts must not be tracked for public launch",
        "fixtures/model.pt: local model/checkpoint artifacts must not be tracked for public launch",
        "fixtures/model.pth: local model/checkpoint artifacts must not be tracked for public launch",
        "fixtures/model.tflite: local model/checkpoint artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local model/checkpoint artifacts")
    container_artifact_failures = tracked_container_artifact_file_failures(
        tracked_paths=[
            ".docker/cache/buildkit.db",
            "docker-data/postgres/base/0001",
            "docker-volumes/fathom-model-store/metadata.json",
            "docker-compose.override.yml",
            "docker-compose.override.yaml",
            "compose.override.yml",
            "compose.override.yaml",
            "Dockerfile",
            "docs/deployment/docker.md",
            "docker-compose.yml",
            "compose.yml",
        ],
    )
    if container_artifact_failures != [
        ".docker/cache/buildkit.db: local Docker/container artifacts must not be tracked for public launch",
        "docker-data/postgres/base/0001: local Docker/container artifacts must not be tracked for public launch",
        "docker-volumes/fathom-model-store/metadata.json: local Docker/container artifacts must not be tracked for public launch",
        "docker-compose.override.yml: local Docker/container override files must not be tracked for public launch",
        "docker-compose.override.yaml: local Docker/container override files must not be tracked for public launch",
        "compose.override.yml: local Docker/container override files must not be tracked for public launch",
        "compose.override.yaml: local Docker/container override files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local Docker/container artifacts")
    infra_state_failures = tracked_infra_state_file_failures(
        tracked_paths=[
            ".terraform/providers/registry.terraform.io/example/provider",
            ".terraform.lock.hcl",
            "infra/default.tfstate",
            "infra/default.tfstate.backup",
            "infra/prod.tfvars",
            "infra/prod.auto.tfvars",
            "infra/prod.tfvars.json",
            "infra/plan.tfplan",
            "infra/main.tf",
            "docs/deployment/terraform.md",
        ],
    )
    if infra_state_failures != [
        ".terraform/providers/registry.terraform.io/example/provider: local infrastructure state artifacts must not be tracked for public launch",
        ".terraform.lock.hcl: local infrastructure state artifacts must not be tracked for public launch",
        "infra/default.tfstate: local infrastructure state artifacts must not be tracked for public launch",
        "infra/default.tfstate.backup: local infrastructure state artifacts must not be tracked for public launch",
        "infra/prod.tfvars: local infrastructure state artifacts must not be tracked for public launch",
        "infra/prod.auto.tfvars: local infrastructure state artifacts must not be tracked for public launch",
        "infra/prod.tfvars.json: local infrastructure state artifacts must not be tracked for public launch",
        "infra/plan.tfplan: local infrastructure state artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local infrastructure state artifacts")
    allowed_mobile_build_gitignore = "\n".join(sorted(required_mobile_build_gitignore_patterns)) + "\n"
    if gitignore_mobile_build_failures(allowed_mobile_build_gitignore):
        raise AssertionError("public risk self-test rejected complete local mobile/Xcode/Android build artifact ignore patterns")
    mobile_build_gitignore_failures = gitignore_mobile_build_failures(
        allowed_mobile_build_gitignore.replace("*.xcresult\n", "")
    )
    if mobile_build_gitignore_failures != [
        ".gitignore: missing local mobile/Xcode/Android build artifact ignore patterns: *.xcresult"
    ]:
        raise AssertionError("public risk self-test did not reject missing local mobile/Xcode/Android build artifact ignore patterns")
    mobile_build_failures = tracked_mobile_build_file_failures(
        tracked_paths=[
            "DerivedData/Fathom/Build/Products/Debug/Fathom.app",
            ".gradle/caches/modules-2/files-2.1/metadata.bin",
            "android/local.properties",
            "ios/Fathom.xcodeproj/xcuserdata/tim.xcuserdatad/UserInterfaceState.xcuserstate",
            "TestResults/Fathom.xcresult/Data/data.0~",
            "builds/Fathom.ipa",
            "android/app/release/app-release.apk",
            "android/app/release/app-release.aab",
            "Fathom.app.dSYM/Contents/Resources/DWARF/Fathom",
            "ios/Fathom.xcodeproj/project.pbxproj",
            "android/app/build.gradle",
            "docs/mobile.md",
        ],
    )
    if mobile_build_failures != [
        "DerivedData/Fathom/Build/Products/Debug/Fathom.app: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        ".gradle/caches/modules-2/files-2.1/metadata.bin: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "android/local.properties: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "ios/Fathom.xcodeproj/xcuserdata/tim.xcuserdatad/UserInterfaceState.xcuserstate: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "TestResults/Fathom.xcresult/Data/data.0~: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "builds/Fathom.ipa: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "android/app/release/app-release.apk: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "android/app/release/app-release.aab: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "Fathom.app.dSYM/Contents/Resources/DWARF/Fathom: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local mobile/Xcode/Android build artifacts")
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
failures.extend(gitignore_workspace_context_failures())
failures.extend(gitignore_credential_failures())
failures.extend(gitignore_model_artifact_failures())
failures.extend(gitignore_container_artifact_failures())
failures.extend(gitignore_infra_state_failures())
failures.extend(gitignore_mobile_build_failures())
failures.extend(gitignore_package_artifact_failures())
failures.extend(tracked_runtime_artifact_file_failures())
failures.extend(tracked_python_artifact_file_failures())
failures.extend(tracked_frontend_artifact_file_failures())
failures.extend(tracked_rust_artifact_file_failures())
failures.extend(tracked_package_artifact_file_failures())
failures.extend(tracked_backup_artifact_file_failures())
failures.extend(tracked_model_artifact_file_failures())
failures.extend(tracked_container_artifact_file_failures())
failures.extend(tracked_infra_state_file_failures())
failures.extend(tracked_mobile_build_file_failures())
failures.extend(tracked_symlink_failures())
if failures:
    print("Public risk scan failed:", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print("public risk scan passed")
PY
