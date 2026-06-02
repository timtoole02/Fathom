#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$@" <<'PY'
import pathlib
import json
import posixpath
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
        ("AWS access key ID", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
        ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
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
    ".Spotlight-V100",
    ".TemporaryItems",
    "__MACOSX",
    ".AppleDouble",
    ".Trashes",
    ".fseventsd",
}
blocked_tracked_editor_artifact_suffixes = {
    ".diff",
    ".orig",
    ".patch",
    ".rej",
    ".swo",
    ".swp",
}
blocked_tracked_ide_artifact_dirs = {
    ".idea",
    ".settings",
    ".vscode",
}
blocked_tracked_ide_artifact_filenames = {
    ".classpath",
    ".project",
}
blocked_tracked_ide_artifact_suffixes = {
    ".code-workspace",
    ".iml",
}
allowed_tracked_credential_filenames = {".env.example"}
blocked_tracked_credential_filenames = {
    ".env",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
}
blocked_tracked_credential_dirs = {
    ".direnv",
    ".ssh",
    "private",
    "secrets",
}
blocked_tracked_credential_suffixes = {
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".p8",
    ".pem",
    ".pfx",
    ".secret",
    ".secrets",
}
blocked_tracked_cloud_credential_dirs = {
    ".aws",
    ".azure",
}
blocked_tracked_cloud_credential_filenames = {
    ".boto",
    "application_default_credentials.json",
    "boto.cfg",
    "service-account.json",
    "serviceAccountKey.json",
    "service_account.json",
}
blocked_tracked_kubernetes_credential_dirs = {
    ".kube",
}
blocked_tracked_kubernetes_credential_filenames = {
    "kubeconfig",
    "kubeconfig.yaml",
    "kubeconfig.yml",
}
blocked_tracked_workspace_filenames = {
    ".aider.chat.history.md",
    ".aider.input.history",
    ".aider.tags.cache.v4",
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "MEMORY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}
blocked_tracked_workspace_dirs = {
    ".claude",
    ".codex",
    ".continue",
}
blocked_tracked_command_history_filenames = {
    ".bash_history",
    ".fish_history",
    ".mysql_history",
    ".node_repl_history",
    ".psql_history",
    ".python_history",
    ".rediscli_history",
    ".sqlite_history",
    ".zsh_history",
}
required_workspace_gitignore_patterns = {
    "/.aider.chat.history.md",
    "/.aider.input.history",
    "/.aider.tags.cache.v4",
    "/.claude/",
    "/.codex/",
    "/.continue/",
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
required_command_history_gitignore_patterns = {
    ".bash_history",
    ".fish_history",
    ".mysql_history",
    ".node_repl_history",
    ".psql_history",
    ".python_history",
    ".rediscli_history",
    ".sqlite_history",
    ".zsh_history",
}
required_credential_gitignore_patterns = {
    "/.direnv/",
    ".env",
    ".env.*",
    ".envrc",
    "/.ssh/",
    "/private/",
    "/secrets/",
    "!.env.example",
    "*.jks",
    "*.key",
    "*.keystore",
    "*.p12",
    "*.p8",
    "*.pem",
    "*.pfx",
    "*.secret",
    "*.secrets",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
}
required_cloud_credential_gitignore_patterns = {
    "/.aws/",
    "/.azure/",
    "/.config/gcloud/",
    ".boto",
    "application_default_credentials.json",
    "boto.cfg",
    "service-account.json",
    "serviceAccountKey.json",
    "service_account.json",
}
required_kubernetes_credential_gitignore_patterns = {
    "/.kube/",
    "kubeconfig",
    "kubeconfig.yaml",
    "kubeconfig.yml",
}
required_os_metadata_gitignore_patterns = {
    ".AppleDouble/",
    ".DS_Store",
    ".LSOverride",
    ".Spotlight-V100/",
    ".TemporaryItems/",
    ".Trashes/",
    ".fseventsd/",
    ".localized",
    "._*",
    "__MACOSX/",
    "Thumbs.db",
    "desktop.ini",
}
required_editor_artifact_gitignore_patterns = {
    "*~",
    "*.diff",
    "*.orig",
    "*.patch",
    "*.rej",
    "*.swo",
    "*.swp",
}
required_ide_artifact_gitignore_patterns = {
    "/.idea/",
    "/.vscode/",
    ".classpath",
    ".project",
    ".settings/",
    "*.code-workspace",
    "*.iml",
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
    "*.mobileprovision",
    "*.provisionprofile",
    "*.xcarchive",
    "*.xcresult",
    "*.xcuserstate",
    "local.properties",
    "xcuserdata/",
}
required_screen_capture_gitignore_patterns = {
    "Screen Recording *",
    "Screen Shot *",
    "Screenshot *",
}
required_media_capture_gitignore_patterns = {
    "Audio Recording *",
    "Voice Memo *",
    "*.aac",
    "*.avi",
    "*.flac",
    "*.m4a",
    "*.m4v",
    "*.mkv",
    "*.mov",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.webm",
}
required_rust_artifact_gitignore_patterns = {
    "/.cargo/",
    "/target/",
    "**/*.rs.bk",
    "*.profdata",
    "*.profraw",
    "target/",
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
required_backup_artifact_gitignore_patterns = {
    "/backups/",
    "/dumps/",
    "*.bak",
    "*.backup",
    "*.dump",
    "*.sql",
}
required_diagnostic_artifact_gitignore_patterns = {
    "/debug-output/",
    "/logs/",
    "/profiles/",
    "/traces/",
    "/core",
    "/core.[0-9]*",
    "*.cpuprofile",
    "*.core",
    "*.crash",
    "*.dmp",
    "*.heapsnapshot",
    "*.ips",
    "*.log",
    "*.perf",
    "*.prof",
    "*.trace",
}
required_python_artifact_gitignore_patterns = {
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "*.pyc",
    "*.pyo",
}
required_python_env_artifact_gitignore_patterns = {
    "/.nox/",
    "/.tox/",
    "/.venv/",
    "/env/",
    "/venv/",
    "/wheelhouse/",
    "pip-wheel-metadata/",
    "site-packages/",
}
required_frontend_artifact_gitignore_patterns = {
    ".npm/",
    ".parcel-cache/",
    ".pnpm-store/",
    ".turbo/",
    ".yarn/build-state.yml",
    ".yarn/cache/",
    ".yarn/install-state.gz",
    ".yarn/unplugged/",
    "*.tsbuildinfo",
    "build/",
    "coverage/",
    "dist/",
    "frontend/.next/",
    "frontend/.npm/",
    "frontend/.pnpm-store/",
    "frontend/.vite/",
    "frontend/.yarn/build-state.yml",
    "frontend/.yarn/cache/",
    "frontend/.yarn/install-state.gz",
    "frontend/.yarn/unplugged/",
    "frontend/build/",
    "frontend/coverage/",
    "frontend/dist/",
    "frontend/node_modules/",
    "node_modules/",
    "npm-debug.log",
    "pnpm-debug.log",
    "yarn-debug.log",
    "yarn-error.log",
}
required_test_report_artifact_gitignore_patterns = {
    "/.playwright/",
    "/blob-report/",
    "/htmlcov/",
    "/playwright-report/",
    "/reports/",
    "/test-reports/",
    "/test-results/",
    ".coverage",
    ".coverage.*",
    "*.lcov",
    "*.junit.xml",
    "coverage.xml",
    "lcov.info",
    "junit.xml",
}
required_notebook_artifact_gitignore_patterns = {
    ".ipynb_checkpoints/",
}
required_runtime_artifact_gitignore_patterns = {
    "*.db",
    "*.db-journal",
    "*.db-shm",
    "*.db-wal",
    "*.pid",
    "*.sqlite",
    "*.sqlite-journal",
    "*.sqlite-shm",
    "*.sqlite-wal",
    "*.sqlite3",
    ".fathom/",
    "summary.local.json",
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
    ".pid",
    ".sqlite",
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3",
}
blocked_tracked_diagnostic_artifact_dirs = {
    "debug-output",
    "logs",
    "profiles",
    "traces",
}
blocked_tracked_diagnostic_artifact_suffixes = {
    ".core",
    ".cpuprofile",
    ".crash",
    ".dmp",
    ".heapsnapshot",
    ".ips",
    ".perf",
    ".prof",
    ".trace",
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
blocked_tracked_python_env_artifact_dirs = {
    ".nox",
    ".tox",
    ".venv",
    "env",
    "pip-wheel-metadata",
    "site-packages",
    "venv",
    "wheelhouse",
}
blocked_tracked_frontend_artifact_dirs = {
    ".npm",
    ".next",
    ".parcel-cache",
    ".pnpm-store",
    ".turbo",
    ".vite",
    "coverage",
    "node_modules",
}
blocked_tracked_frontend_yarn_artifact_dirs = {
    "cache",
    "unplugged",
}
blocked_tracked_frontend_yarn_artifact_filenames = {
    "build-state.yml",
    "install-state.gz",
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
    ".mobileprovision",
    ".provisionprofile",
    ".xcarchive",
    ".xcresult",
    ".xcuserstate",
}
blocked_tracked_screen_capture_prefixes = (
    "Screen Recording ",
    "Screen Shot ",
    "Screenshot ",
)
blocked_tracked_media_capture_prefixes = (
    "Audio Recording ",
    "Voice Memo ",
)
blocked_tracked_media_capture_suffixes = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
    ".webm",
}
blocked_tracked_test_report_artifact_dirs = {
    ".playwright",
    "blob-report",
    "htmlcov",
    "playwright-report",
    "reports",
    "test-reports",
    "test-results",
}
blocked_tracked_test_report_artifact_filenames = {
    ".coverage",
    "coverage.xml",
    "junit.xml",
    "lcov.info",
}
blocked_tracked_test_report_artifact_suffixes = {
    ".lcov",
    ".junit.xml",
}
blocked_tracked_notebook_artifact_dirs = {
    ".ipynb_checkpoints",
}
tracked_symlink_mode = "120000"
dependency_lockfile_names = {
    "Cargo.lock",
    "Pipfile.lock",
    "bun.lockb",
    "npm-shrinkwrap.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}
dependency_lock_source_patterns = [
    (
        "local file dependency source in lockfile",
        re.compile(r"\b(?:file|link):(?:\.\.?/|/|~)|\bpath\s*=\s*[\"'](?:\.\.?/|/|~)", re.IGNORECASE),
    ),
    (
        "SSH dependency source in lockfile",
        re.compile(r"\b(?:git\+)?ssh://|git@[A-Za-z0-9_.-]+:", re.IGNORECASE),
    ),
    (
        "authenticated dependency URL in lockfile",
        re.compile(r"https?://[^/\s:@]+:[^@\s]+@", re.IGNORECASE),
    ),
]
git_lfs_pointer_version = "version https://git-lfs.github.com/spec/v1"
docs_evidence_prefixes = (
    "docs/benchmarks/",
    "docs/api/",
)
docs_evidence_paths = {"docs/public-launch-evidence.md", "docs/public-launch-checklist.md"}

def is_dependency_lockfile(rel):
    path = pathlib.PurePosixPath(rel)
    return path.name in dependency_lockfile_names or path.name.endswith(".lock")

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

def tracked_dependency_lock_source_failures(tracked_paths=None, texts=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        if not is_dependency_lockfile(rel):
            continue
        path = pathlib.Path(rel)
        try:
            if path.is_symlink():
                continue
            text = texts[rel] if texts is not None else path.read_text(encoding="utf-8")
        except (FileNotFoundError, KeyError, OSError, UnicodeDecodeError):
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            privacy_line = line
            for public_repo_url in public_repo_urls:
                privacy_line = privacy_line.replace(public_repo_url, "")
            for label, pattern in privacy_patterns():
                if pattern.search(privacy_line):
                    failures.append(f"{rel}:{line_no}: {label} in dependency lockfile: {line.strip()}")
            for label, pattern in secret_value_patterns():
                if pattern.search(privacy_line):
                    failures.append(f"{rel}:{line_no}: {label} in dependency lockfile: {line.strip()}")
            for label, pattern in dependency_lock_source_patterns:
                if pattern.search(privacy_line):
                    failures.append(f"{rel}:{line_no}: {label}: {line.strip()}")
    return failures

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

def tracked_git_lfs_pointer_failures(tracked_paths=None, texts=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.Path(rel)
        try:
            if path.is_symlink():
                continue
            text = texts[rel] if texts is not None else path.read_text(encoding="utf-8")
        except (FileNotFoundError, KeyError, OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip() == git_lfs_pointer_version
            and re.fullmatch(r"oid sha256:[0-9a-f]{64}", lines[1].strip())
            and re.fullmatch(r"size [1-9][0-9]*", lines[2].strip())
        ):
            failures.append(f"{rel}: Git LFS pointer files must not be tracked for public launch")
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
        if path.name in blocked_tracked_ide_artifact_filenames:
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
        if any(part in blocked_tracked_credential_dirs for part in path.parts):
            failures.append(f"{rel}: credential/config files must not be tracked for public launch")
            continue
        if name in blocked_tracked_credential_filenames or name.startswith(".env."):
            failures.append(f"{rel}: credential/config files must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_credential_suffixes:
            failures.append(f"{rel}: credential/key files must not be tracked for public launch")
    return failures

def tracked_cloud_credential_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_cloud_credential_dirs for part in path.parts):
            failures.append(f"{rel}: cloud SDK credential/config files must not be tracked for public launch")
            continue
        if len(path.parts) >= 3 and path.parts[0] == ".config" and path.parts[1] == "gcloud":
            failures.append(f"{rel}: cloud SDK credential/config files must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_cloud_credential_filenames:
            failures.append(f"{rel}: cloud SDK credential/config files must not be tracked for public launch")
    return failures

def tracked_kubernetes_credential_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_kubernetes_credential_dirs for part in path.parts):
            failures.append(f"{rel}: Kubernetes credential/config files must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_kubernetes_credential_filenames:
            failures.append(f"{rel}: Kubernetes credential/config files must not be tracked for public launch")
    return failures

def tracked_workspace_context_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if (
            path.name in blocked_tracked_workspace_filenames
            or rel.startswith(("memory/", ".openclaw/"))
            or any(part in blocked_tracked_workspace_dirs for part in path.parts)
        ):
            failures.append(f"{rel}: workspace/personal agent context files must not be tracked for public launch")
    return failures

def tracked_command_history_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.name in blocked_tracked_command_history_filenames:
            failures.append(f"{rel}: local shell/REPL command history files must not be tracked for public launch")
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

def gitignore_command_history_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local shell/REPL command history ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_command_history_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local shell/REPL command history ignore patterns: {', '.join(missing)}"]
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

def gitignore_cloud_credential_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local cloud SDK credential/config ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_cloud_credential_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local cloud SDK credential/config ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_kubernetes_credential_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Kubernetes credential/config ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_kubernetes_credential_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Kubernetes credential/config ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_os_metadata_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local OS/platform metadata ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_os_metadata_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local OS/platform metadata ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_editor_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local editor backup/swap ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_editor_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local editor backup/swap ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_ide_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local IDE workspace/config ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_ide_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local IDE workspace/config ignore patterns: {', '.join(missing)}"]
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

def gitignore_screen_capture_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local screenshot/screen-recording ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_screen_capture_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local screenshot/screen-recording ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_media_capture_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local audio/video capture/export ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_media_capture_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local audio/video capture/export ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_rust_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Rust/Cargo cache/build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_rust_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Rust/Cargo cache/build artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_backup_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local backup/dump artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_backup_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local backup/dump artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_diagnostic_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local log/trace/profiling/debug-output artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_diagnostic_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local log/trace/profiling/debug-output artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_python_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Python cache/build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_python_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Python cache/build artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_python_env_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Python virtualenv/dependency artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_python_env_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Python virtualenv/dependency artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_frontend_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local frontend/Node cache/build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_frontend_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local frontend/Node cache/build artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_test_report_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local test report artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_test_report_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local test report artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_notebook_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local notebook artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_notebook_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local notebook artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_runtime_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local runtime/artifact detail-file ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_runtime_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local runtime/artifact detail-file ignore patterns: {', '.join(missing)}"]
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

def tracked_diagnostic_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.name == "core" or re.fullmatch(r"core\.[0-9]+", path.name):
            failures.append(f"{rel}: local log/trace/profiling/debug-output artifacts must not be tracked for public launch")
            continue
        if path.parts and path.parts[0] in blocked_tracked_diagnostic_artifact_dirs:
            failures.append(f"{rel}: local log/trace/profiling/debug-output artifacts must not be tracked for public launch")
            continue
        if any(rel.endswith(suffix) for suffix in blocked_tracked_diagnostic_artifact_suffixes):
            failures.append(f"{rel}: local log/trace/profiling/debug-output artifacts must not be tracked for public launch")
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

def tracked_python_env_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_python_env_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: Python virtualenv/dependency artifacts must not be tracked for public launch")
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
        if ".yarn" in path.parts and any(part in blocked_tracked_frontend_yarn_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
            continue
        if ".yarn" in path.parts and path.name in blocked_tracked_frontend_yarn_artifact_filenames:
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
        if any(part.endswith((".dSYM", ".xcarchive", ".xcresult")) for part in path.parts):
            failures.append(f"{rel}: local mobile/Xcode/Android build artifacts must not be tracked for public launch")
            continue
        if path.suffix in blocked_tracked_mobile_build_suffixes:
            failures.append(f"{rel}: local mobile/Xcode/Android build artifacts must not be tracked for public launch")
    return failures

def tracked_screen_capture_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.name.startswith(blocked_tracked_screen_capture_prefixes):
            failures.append(f"{rel}: local screenshot/screen-recording artifacts must not be tracked for public launch")
    return failures

def tracked_media_capture_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.name.startswith(blocked_tracked_media_capture_prefixes):
            failures.append(f"{rel}: local audio/video capture/export artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_media_capture_suffixes:
            failures.append(f"{rel}: local audio/video capture/export artifacts must not be tracked for public launch")
    return failures

def tracked_test_report_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] in blocked_tracked_test_report_artifact_dirs:
            failures.append(f"{rel}: local test report artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_test_report_artifact_filenames:
            failures.append(f"{rel}: local test report artifacts must not be tracked for public launch")
            continue
        if path.name.startswith(".coverage."):
            failures.append(f"{rel}: local test report artifacts must not be tracked for public launch")
            continue
        if any(rel.endswith(suffix) for suffix in blocked_tracked_test_report_artifact_suffixes):
            failures.append(f"{rel}: local test report artifacts must not be tracked for public launch")
    return failures

def tracked_notebook_artifact_file_failures(tracked_paths=None, notebook_texts=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_notebook_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: local notebook checkpoint artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() != ".ipynb":
            continue
        try:
            text = notebook_texts[rel] if notebook_texts is not None else pathlib.Path(rel).read_text(encoding="utf-8")
            notebook = json.loads(text)
        except (FileNotFoundError, KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for cell in notebook.get("cells", []):
            if cell.get("execution_count") is not None or cell.get("outputs"):
                failures.append(f"{rel}: notebook execution outputs must not be tracked for public launch")
                break
    return failures

def tracked_index_entries():
    entries = []
    for line in subprocess.check_output(["git", "ls-files", "-s"], text=True).splitlines():
        metadata, rel = line.split("\t", 1)
        mode, _blob_sha, _stage = metadata.split()
        entries.append((mode, rel))
    return entries

def tracked_symlink_failures(tracked_entries=None, targets=None, tracked_paths=None):
    if tracked_entries is None:
        tracked_entries = tracked_index_entries()
    if tracked_paths is None:
        tracked_paths = {rel for _mode, rel in tracked_entries}
    else:
        tracked_paths = set(tracked_paths)
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
            continue
        target_rel = posixpath.normpath((pathlib.PurePosixPath(rel).parent / target_path).as_posix())
        target_prefix = target_rel.rstrip("/") + "/"
        if target_rel == "." or (
            target_rel not in tracked_paths and not any(path.startswith(target_prefix) for path in tracked_paths)
        ):
            failures.append(f"{rel}: tracked symlink target must resolve to an existing tracked in-repository path, found {target}")
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
        ("docs/api/example.md", "AWS key id: AKIAABCDEFGHIJKLMNOP"),
        ("docs/api/example.md", "Google API key: AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"),
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
    git_lfs_pointer_failures = tracked_git_lfs_pointer_failures(
        tracked_paths=["docs/api/public-contract.json", "weights/model.safetensors"],
        texts={
            "docs/api/public-contract.json": '{"version":"local"}\n',
            "weights/model.safetensors": (
                git_lfs_pointer_version
                + "\n"
                + "oid sha256:"
                + ("a" * 64)
                + "\n"
                + "size 2048\n"
            ),
        },
    )
    if git_lfs_pointer_failures != [
        "weights/model.safetensors: Git LFS pointer files must not be tracked for public launch"
    ]:
        raise AssertionError("public risk self-test did not reject tracked Git LFS pointer files")
    blocked_file_failures = tracked_blocked_file_failures(
        tracked_paths=[
            "docs/api/public-contract.json",
            "docs/.DS_Store",
            "docs/.fseventsd/fseventsd-uuid",
            "docs/.Spotlight-V100/Store-V2/index",
            "docs/.TemporaryItems/folders.501/TemporaryItems",
            "docs/.Trashes/501/deleted.md",
            "docs/__MACOSX/._public-launch-evidence.md",
            "docs/.AppleDouble/public-launch-checklist.md",
            "frontend/._vite.config.ts",
            "frontend/.LSOverride",
            "frontend/.localized",
            "frontend/Thumbs.db",
            "desktop.ini",
            "README.md~",
            "docs/launch-review.patch",
            "docs/launch-review.diff",
            "docs/api/client-examples.md.swp",
            "docs/public-launch-checklist.md.orig",
            "docs/public-launch-evidence.md.rej",
            ".vscode/settings.json",
            ".idea/workspace.xml",
            ".settings/org.eclipse.jdt.core.prefs",
            ".classpath",
            ".project",
            "fathom.code-workspace",
            "fathom.iml",
        ],
    )
    if blocked_file_failures != [
        "docs/.DS_Store: OS/platform metadata files must not be tracked for public launch",
        "docs/.fseventsd/fseventsd-uuid: OS/platform metadata files must not be tracked for public launch",
        "docs/.Spotlight-V100/Store-V2/index: OS/platform metadata files must not be tracked for public launch",
        "docs/.TemporaryItems/folders.501/TemporaryItems: OS/platform metadata files must not be tracked for public launch",
        "docs/.Trashes/501/deleted.md: OS/platform metadata files must not be tracked for public launch",
        "docs/__MACOSX/._public-launch-evidence.md: OS/platform metadata files must not be tracked for public launch",
        "docs/.AppleDouble/public-launch-checklist.md: OS/platform metadata files must not be tracked for public launch",
        "frontend/._vite.config.ts: OS/platform metadata files must not be tracked for public launch",
        "frontend/.LSOverride: OS/platform metadata files must not be tracked for public launch",
        "frontend/.localized: OS/platform metadata files must not be tracked for public launch",
        "frontend/Thumbs.db: OS/platform metadata files must not be tracked for public launch",
        "desktop.ini: OS/platform metadata files must not be tracked for public launch",
        "README.md~: editor backup/swap artifacts must not be tracked for public launch",
        "docs/launch-review.patch: editor backup/swap artifacts must not be tracked for public launch",
        "docs/launch-review.diff: editor backup/swap artifacts must not be tracked for public launch",
        "docs/api/client-examples.md.swp: editor backup/swap artifacts must not be tracked for public launch",
        "docs/public-launch-checklist.md.orig: editor backup/swap artifacts must not be tracked for public launch",
        "docs/public-launch-evidence.md.rej: editor backup/swap artifacts must not be tracked for public launch",
        ".vscode/settings.json: IDE workspace/config artifacts must not be tracked for public launch",
        ".idea/workspace.xml: IDE workspace/config artifacts must not be tracked for public launch",
        ".settings/org.eclipse.jdt.core.prefs: IDE workspace/config artifacts must not be tracked for public launch",
        ".classpath: IDE workspace/config artifacts must not be tracked for public launch",
        ".project: IDE workspace/config artifacts must not be tracked for public launch",
        "fathom.code-workspace: IDE workspace/config artifacts must not be tracked for public launch",
        "fathom.iml: IDE workspace/config artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked OS/platform metadata, backup/swap, or IDE config files")
    allowed_os_metadata_gitignore = "\n".join(sorted(required_os_metadata_gitignore_patterns)) + "\n"
    if gitignore_os_metadata_failures(allowed_os_metadata_gitignore):
        raise AssertionError("public risk self-test rejected complete local OS/platform metadata ignore patterns")
    os_metadata_gitignore_failures = gitignore_os_metadata_failures(allowed_os_metadata_gitignore.replace(".DS_Store\n", ""))
    if os_metadata_gitignore_failures != [".gitignore: missing local OS/platform metadata ignore patterns: .DS_Store"]:
        raise AssertionError("public risk self-test did not reject missing local OS/platform metadata ignore patterns")
    allowed_editor_artifact_gitignore = "\n".join(sorted(required_editor_artifact_gitignore_patterns)) + "\n"
    if gitignore_editor_artifact_failures(allowed_editor_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local editor backup/swap ignore patterns")
    editor_artifact_gitignore_failures = gitignore_editor_artifact_failures(
        allowed_editor_artifact_gitignore.replace("*.patch\n", "")
    )
    if editor_artifact_gitignore_failures != [".gitignore: missing local editor backup/swap ignore patterns: *.patch"]:
        raise AssertionError("public risk self-test did not reject missing local editor backup/swap ignore patterns")
    allowed_ide_artifact_gitignore = "\n".join(sorted(required_ide_artifact_gitignore_patterns)) + "\n"
    if gitignore_ide_artifact_failures(allowed_ide_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local IDE workspace/config ignore patterns")
    ide_artifact_gitignore_failures = gitignore_ide_artifact_failures(allowed_ide_artifact_gitignore.replace("*.iml\n", ""))
    if ide_artifact_gitignore_failures != [".gitignore: missing local IDE workspace/config ignore patterns: *.iml"]:
        raise AssertionError("public risk self-test did not reject missing local IDE workspace/config ignore patterns")
    credential_file_failures = tracked_credential_file_failures(
        tracked_paths=[
            ".env",
            ".env.local",
            ".envrc",
            ".env.example",
            ".direnv/allow",
            ".npmrc",
            ".ssh/config",
            "secrets/api-keys.json",
            "private/prod-env.txt",
            "docs/id_ed25519",
            "docs/deploy.secret",
            "docs/deploy.secrets",
            "ops/secrets.json",
            "ops/secrets.yaml",
            "ops/secrets.yml",
            "mobile/AuthKey_ABC123.p8",
            "android/release.jks",
            "android/release.keystore",
            "docs/public.pem",
            "docs/public-key.txt",
            "frontend/.pypirc",
            "crates/fathom-server/credentials",
        ],
    )
    if credential_file_failures != [
        ".env: credential/config files must not be tracked for public launch",
        ".env.local: credential/config files must not be tracked for public launch",
        ".envrc: credential/config files must not be tracked for public launch",
        ".direnv/allow: credential/config files must not be tracked for public launch",
        ".npmrc: credential/config files must not be tracked for public launch",
        ".ssh/config: credential/config files must not be tracked for public launch",
        "secrets/api-keys.json: credential/config files must not be tracked for public launch",
        "private/prod-env.txt: credential/config files must not be tracked for public launch",
        "docs/id_ed25519: credential/config files must not be tracked for public launch",
        "docs/deploy.secret: credential/key files must not be tracked for public launch",
        "docs/deploy.secrets: credential/key files must not be tracked for public launch",
        "ops/secrets.json: credential/config files must not be tracked for public launch",
        "ops/secrets.yaml: credential/config files must not be tracked for public launch",
        "ops/secrets.yml: credential/config files must not be tracked for public launch",
        "mobile/AuthKey_ABC123.p8: credential/key files must not be tracked for public launch",
        "android/release.jks: credential/key files must not be tracked for public launch",
        "android/release.keystore: credential/key files must not be tracked for public launch",
        "docs/public.pem: credential/key files must not be tracked for public launch",
        "frontend/.pypirc: credential/config files must not be tracked for public launch",
        "crates/fathom-server/credentials: credential/config files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked credential/config files")
    cloud_credential_file_failures = tracked_cloud_credential_file_failures(
        tracked_paths=[
            ".aws/config",
            ".aws/credentials",
            ".azure/accessTokens.json",
            ".config/gcloud/application_default_credentials.json",
            ".boto",
            "boto.cfg",
            "secrets/service-account.json",
            "secrets/service_account.json",
            "firebase/serviceAccountKey.json",
            "docs/api/public-contract.json",
            "docs/cloud-auth.md",
        ],
    )
    if cloud_credential_file_failures != [
        ".aws/config: cloud SDK credential/config files must not be tracked for public launch",
        ".aws/credentials: cloud SDK credential/config files must not be tracked for public launch",
        ".azure/accessTokens.json: cloud SDK credential/config files must not be tracked for public launch",
        ".config/gcloud/application_default_credentials.json: cloud SDK credential/config files must not be tracked for public launch",
        ".boto: cloud SDK credential/config files must not be tracked for public launch",
        "boto.cfg: cloud SDK credential/config files must not be tracked for public launch",
        "secrets/service-account.json: cloud SDK credential/config files must not be tracked for public launch",
        "secrets/service_account.json: cloud SDK credential/config files must not be tracked for public launch",
        "firebase/serviceAccountKey.json: cloud SDK credential/config files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked cloud SDK credential/config files")
    kubernetes_credential_file_failures = tracked_kubernetes_credential_file_failures(
        tracked_paths=[
            ".kube/config",
            ".kube/cache/discovery/example.json",
            "ops/kubeconfig",
            "ops/kubeconfig.yaml",
            "ops/kubeconfig.yml",
            "docs/deployment/kubernetes.md",
            "docs/api/public-contract.json",
        ],
    )
    if kubernetes_credential_file_failures != [
        ".kube/config: Kubernetes credential/config files must not be tracked for public launch",
        ".kube/cache/discovery/example.json: Kubernetes credential/config files must not be tracked for public launch",
        "ops/kubeconfig: Kubernetes credential/config files must not be tracked for public launch",
        "ops/kubeconfig.yaml: Kubernetes credential/config files must not be tracked for public launch",
        "ops/kubeconfig.yml: Kubernetes credential/config files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Kubernetes credential/config files")
    workspace_context_failures = tracked_workspace_context_failures(
        tracked_paths=[
            "AGENTS.md",
            "docs/AGENTS.md",
            "IDENTITY.md",
            "MEMORY.md",
            "memory/2026-05-31.md",
            ".openclaw/session.json",
            ".codex/config.toml",
            ".claude/settings.local.json",
            ".continue/config.json",
            ".aider.chat.history.md",
            ".aider.input.history",
            ".aider.tags.cache.v4",
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
        ".codex/config.toml: workspace/personal agent context files must not be tracked for public launch",
        ".claude/settings.local.json: workspace/personal agent context files must not be tracked for public launch",
        ".continue/config.json: workspace/personal agent context files must not be tracked for public launch",
        ".aider.chat.history.md: workspace/personal agent context files must not be tracked for public launch",
        ".aider.input.history: workspace/personal agent context files must not be tracked for public launch",
        ".aider.tags.cache.v4: workspace/personal agent context files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked workspace/personal agent context files")
    allowed_gitignore = "\n".join(sorted(required_workspace_gitignore_patterns)) + "\n"
    if gitignore_workspace_context_failures(allowed_gitignore):
        raise AssertionError("public risk self-test rejected complete workspace/personal context ignore patterns")
    gitignore_failures = gitignore_workspace_context_failures(allowed_gitignore.replace("/.codex/\n", ""))
    if gitignore_failures != [".gitignore: missing workspace/personal context ignore patterns: /.codex/"]:
        raise AssertionError("public risk self-test did not reject missing workspace/personal context ignore patterns")
    command_history_failures = tracked_command_history_file_failures(
        tracked_paths=[
            ".bash_history",
            ".zsh_history",
            "frontend/.node_repl_history",
            "db/.psql_history",
            "db/.sqlite_history",
            "db/.mysql_history",
            "cache/.rediscli_history",
            "scripts/.python_history",
            "docs/.fish_history",
            "docs/api/public-contract.json",
        ],
    )
    if command_history_failures != [
        ".bash_history: local shell/REPL command history files must not be tracked for public launch",
        ".zsh_history: local shell/REPL command history files must not be tracked for public launch",
        "frontend/.node_repl_history: local shell/REPL command history files must not be tracked for public launch",
        "db/.psql_history: local shell/REPL command history files must not be tracked for public launch",
        "db/.sqlite_history: local shell/REPL command history files must not be tracked for public launch",
        "db/.mysql_history: local shell/REPL command history files must not be tracked for public launch",
        "cache/.rediscli_history: local shell/REPL command history files must not be tracked for public launch",
        "scripts/.python_history: local shell/REPL command history files must not be tracked for public launch",
        "docs/.fish_history: local shell/REPL command history files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local shell/REPL command history files")
    allowed_command_history_gitignore = "\n".join(sorted(required_command_history_gitignore_patterns)) + "\n"
    if gitignore_command_history_failures(allowed_command_history_gitignore):
        raise AssertionError("public risk self-test rejected complete local shell/REPL command history ignore patterns")
    command_history_gitignore_failures = gitignore_command_history_failures(
        allowed_command_history_gitignore.replace(".zsh_history\n", "")
    )
    if command_history_gitignore_failures != [
        ".gitignore: missing local shell/REPL command history ignore patterns: .zsh_history"
    ]:
        raise AssertionError("public risk self-test did not reject missing local shell/REPL command history ignore patterns")
    allowed_credential_gitignore = "\n".join(sorted(required_credential_gitignore_patterns)) + "\n"
    if gitignore_credential_failures(allowed_credential_gitignore):
        raise AssertionError("public risk self-test rejected complete local credential/config ignore patterns")
    credential_gitignore_failures = gitignore_credential_failures(allowed_credential_gitignore.replace("/.ssh/\n", ""))
    if credential_gitignore_failures != [".gitignore: missing local credential/config ignore patterns: /.ssh/"]:
        raise AssertionError("public risk self-test did not reject missing local credential/config ignore patterns")
    allowed_cloud_credential_gitignore = "\n".join(sorted(required_cloud_credential_gitignore_patterns)) + "\n"
    if gitignore_cloud_credential_failures(allowed_cloud_credential_gitignore):
        raise AssertionError("public risk self-test rejected complete local cloud SDK credential/config ignore patterns")
    cloud_credential_gitignore_failures = gitignore_cloud_credential_failures(
        allowed_cloud_credential_gitignore.replace("/.config/gcloud/\n", "")
    )
    if cloud_credential_gitignore_failures != [
        ".gitignore: missing local cloud SDK credential/config ignore patterns: /.config/gcloud/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local cloud SDK credential/config ignore patterns")
    allowed_kubernetes_credential_gitignore = "\n".join(sorted(required_kubernetes_credential_gitignore_patterns)) + "\n"
    if gitignore_kubernetes_credential_failures(allowed_kubernetes_credential_gitignore):
        raise AssertionError("public risk self-test rejected complete local Kubernetes credential/config ignore patterns")
    kubernetes_credential_gitignore_failures = gitignore_kubernetes_credential_failures(
        allowed_kubernetes_credential_gitignore.replace("/.kube/\n", "")
    )
    if kubernetes_credential_gitignore_failures != [
        ".gitignore: missing local Kubernetes credential/config ignore patterns: /.kube/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Kubernetes credential/config ignore patterns")
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
            "run/server.pid",
            "state/fathom.sqlite",
            "state/fathom.sqlite-wal",
            "frontend/package-lock.json",
        ],
    )
    if runtime_artifact_failures != [
        ".fathom/state/registry.json: local Fathom runtime state must not be tracked for public launch",
        "logs/server.log: local runtime/artifact detail files must not be tracked for public launch",
        "public-contract-artifacts/summary.local.json: local runtime/artifact detail files must not be tracked for public launch",
        "run/server.pid: local runtime/artifact detail files must not be tracked for public launch",
        "state/fathom.sqlite: local runtime/artifact detail files must not be tracked for public launch",
        "state/fathom.sqlite-wal: local runtime/artifact detail files must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local runtime/artifact files")
    allowed_runtime_artifact_gitignore = "\n".join(sorted(required_runtime_artifact_gitignore_patterns)) + "\n"
    if gitignore_runtime_artifact_failures(allowed_runtime_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local runtime/artifact detail-file ignore patterns")
    runtime_artifact_gitignore_failures = gitignore_runtime_artifact_failures(
        allowed_runtime_artifact_gitignore.replace("*.pid\n", "")
    )
    if runtime_artifact_gitignore_failures != [
        ".gitignore: missing local runtime/artifact detail-file ignore patterns: *.pid"
    ]:
        raise AssertionError("public risk self-test did not reject missing local runtime/artifact detail-file ignore patterns")
    allowed_diagnostic_artifact_gitignore = "\n".join(sorted(required_diagnostic_artifact_gitignore_patterns)) + "\n"
    if gitignore_diagnostic_artifact_failures(allowed_diagnostic_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local log/trace/profiling/debug-output artifact ignore patterns")
    diagnostic_artifact_gitignore_failures = gitignore_diagnostic_artifact_failures(
        allowed_diagnostic_artifact_gitignore.replace("*.dmp\n", "")
    )
    if diagnostic_artifact_gitignore_failures != [
        ".gitignore: missing local log/trace/profiling/debug-output artifact ignore patterns: *.dmp"
    ]:
        raise AssertionError("public risk self-test did not reject missing local log/trace/profiling/debug-output artifact ignore patterns")
    diagnostic_artifact_failures = tracked_diagnostic_artifact_file_failures(
        tracked_paths=[
            "logs/backend.trace",
            "traces/public-contract.trace",
            "profiles/backend.cpuprofile",
            "debug-output/request-dump.json",
            "core",
            "core.123",
            "crashes/fathom.dmp",
            "DiagnosticReports/Fathom.crash",
            "DiagnosticReports/Fathom.ips",
            "fathom.core",
            "crates/fathom-server/heap.heapsnapshot",
            "crates/fathom-core/flame.perf",
            "crates/fathom-core/flame.prof",
            "docs/research/core.md",
            "docs/api/public-contract.json",
            "docs/research/performance-strategy.md",
        ],
    )
    if diagnostic_artifact_failures != [
        "logs/backend.trace: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "traces/public-contract.trace: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "profiles/backend.cpuprofile: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "debug-output/request-dump.json: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "core: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "core.123: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "crashes/fathom.dmp: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "DiagnosticReports/Fathom.crash: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "DiagnosticReports/Fathom.ips: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "fathom.core: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "crates/fathom-server/heap.heapsnapshot: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "crates/fathom-core/flame.perf: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "crates/fathom-core/flame.prof: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local log/trace/profiling/debug-output artifacts")
    python_artifact_failures = tracked_python_artifact_file_failures(
        tracked_paths=[
            "scripts/__pycache__/public_api_contract_qa.cpython-312.pyc",
            ".pytest_cache/v/cache/nodeids",
            ".mypy_cache/3.12/scripts/public_api_contract_qa.data.json",
            ".ruff_cache/0.12.0/file",
            "scripts/public_api_contract_qa.pyo",
            "docs/api/public-contract.json",
            "scripts/public_api_contract_qa.py",
        ],
    )
    if python_artifact_failures != [
        "scripts/__pycache__/public_api_contract_qa.cpython-312.pyc: Python cache/build artifacts must not be tracked for public launch",
        ".pytest_cache/v/cache/nodeids: Python cache/build artifacts must not be tracked for public launch",
        ".mypy_cache/3.12/scripts/public_api_contract_qa.data.json: Python cache/build artifacts must not be tracked for public launch",
        ".ruff_cache/0.12.0/file: Python cache/build artifacts must not be tracked for public launch",
        "scripts/public_api_contract_qa.pyo: Python cache/build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Python cache/build artifacts")
    allowed_python_artifact_gitignore = "\n".join(sorted(required_python_artifact_gitignore_patterns)) + "\n"
    if gitignore_python_artifact_failures(allowed_python_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Python cache/build artifact ignore patterns")
    python_artifact_gitignore_failures = gitignore_python_artifact_failures(
        allowed_python_artifact_gitignore.replace(".mypy_cache/\n", "")
    )
    if python_artifact_gitignore_failures != [
        ".gitignore: missing local Python cache/build artifact ignore patterns: .mypy_cache/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Python cache/build artifact ignore patterns")
    python_env_artifact_failures = tracked_python_env_artifact_file_failures(
        tracked_paths=[
            ".venv/bin/python",
            "venv/lib/python3.12/site.py",
            "env/pyvenv.cfg",
            ".tox/py312/log/result.json",
            ".nox/tests/tmp/output.json",
            "wheelhouse/fathom-0.1.0-py3-none-any.whl",
            "pip-wheel-metadata/fathom.json",
            "python/site-packages/fathom/__init__.py",
            "docs/api/public-contract.json",
            "scripts/public_api_contract_qa.py",
        ],
    )
    if python_env_artifact_failures != [
        ".venv/bin/python: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "venv/lib/python3.12/site.py: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "env/pyvenv.cfg: Python virtualenv/dependency artifacts must not be tracked for public launch",
        ".tox/py312/log/result.json: Python virtualenv/dependency artifacts must not be tracked for public launch",
        ".nox/tests/tmp/output.json: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "wheelhouse/fathom-0.1.0-py3-none-any.whl: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "pip-wheel-metadata/fathom.json: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "python/site-packages/fathom/__init__.py: Python virtualenv/dependency artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Python virtualenv/dependency artifacts")
    allowed_python_env_artifact_gitignore = "\n".join(sorted(required_python_env_artifact_gitignore_patterns)) + "\n"
    if gitignore_python_env_artifact_failures(allowed_python_env_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Python virtualenv/dependency artifact ignore patterns")
    python_env_artifact_gitignore_failures = gitignore_python_env_artifact_failures(
        allowed_python_env_artifact_gitignore.replace("/.venv/\n", "")
    )
    if python_env_artifact_gitignore_failures != [
        ".gitignore: missing local Python virtualenv/dependency artifact ignore patterns: /.venv/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Python virtualenv/dependency artifact ignore patterns")
    frontend_artifact_failures = tracked_frontend_artifact_file_failures(
        tracked_paths=[
            "frontend/node_modules/.package-lock.json",
            ".npm/_cacache/index-v5/00/cache-entry",
            ".pnpm-store/v3/files/00/package",
            ".yarn/cache/react-npm-18.2.0.zip",
            ".yarn/unplugged/esbuild/package.json",
            ".yarn/build-state.yml",
            ".yarn/install-state.gz",
            "frontend/.npm/_logs/install.log",
            "frontend/.pnpm-store/v3/files/00/package",
            "frontend/.yarn/cache/vite-npm-5.0.0.zip",
            "frontend/.yarn/unplugged/rollup/package.json",
            "frontend/.yarn/build-state.yml",
            "frontend/.yarn/install-state.gz",
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
        ".npm/_cacache/index-v5/00/cache-entry: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".pnpm-store/v3/files/00/package: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/cache/react-npm-18.2.0.zip: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/unplugged/esbuild/package.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/build-state.yml: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/install-state.gz: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.npm/_logs/install.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.pnpm-store/v3/files/00/package: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.yarn/cache/vite-npm-5.0.0.zip: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.yarn/unplugged/rollup/package.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.yarn/build-state.yml: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.yarn/install-state.gz: frontend/Node cache/build artifacts must not be tracked for public launch",
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
    allowed_frontend_artifact_gitignore = "\n".join(sorted(required_frontend_artifact_gitignore_patterns)) + "\n"
    if gitignore_frontend_artifact_failures(allowed_frontend_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local frontend/Node cache/build artifact ignore patterns")
    frontend_artifact_gitignore_failures = gitignore_frontend_artifact_failures(
        allowed_frontend_artifact_gitignore.replace("frontend/.vite/\n", "")
    )
    if frontend_artifact_gitignore_failures != [
        ".gitignore: missing local frontend/Node cache/build artifact ignore patterns: frontend/.vite/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local frontend/Node cache/build artifact ignore patterns")
    allowed_test_report_artifact_gitignore = "\n".join(sorted(required_test_report_artifact_gitignore_patterns)) + "\n"
    if gitignore_test_report_artifact_failures(allowed_test_report_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local test report artifact ignore patterns")
    test_report_artifact_gitignore_failures = gitignore_test_report_artifact_failures(
        allowed_test_report_artifact_gitignore.replace("*.lcov\n", "")
    )
    if test_report_artifact_gitignore_failures != [
        ".gitignore: missing local test report artifact ignore patterns: *.lcov"
    ]:
        raise AssertionError("public risk self-test did not reject missing local test report artifact ignore patterns")
    test_report_artifact_failures = tracked_test_report_artifact_file_failures(
        tracked_paths=[
            "playwright-report/index.html",
            "blob-report/report.zip",
            ".playwright/screenshots/home.png",
            "test-results/junit.xml",
            "test-reports/backend.xml",
            "reports/public-risk/index.html",
            "htmlcov/index.html",
            ".coverage",
            ".coverage.public-risk",
            "coverage.xml",
            "lcov.info",
            "crates/fathom-core/coverage.lcov",
            "junit.xml",
            "crates/fathom-core/test-output.junit.xml",
            "docs/api/public-contract.json",
            "docs/research/runtime-safety-policy.md",
        ],
    )
    if test_report_artifact_failures != [
        "playwright-report/index.html: local test report artifacts must not be tracked for public launch",
        "blob-report/report.zip: local test report artifacts must not be tracked for public launch",
        ".playwright/screenshots/home.png: local test report artifacts must not be tracked for public launch",
        "test-results/junit.xml: local test report artifacts must not be tracked for public launch",
        "test-reports/backend.xml: local test report artifacts must not be tracked for public launch",
        "reports/public-risk/index.html: local test report artifacts must not be tracked for public launch",
        "htmlcov/index.html: local test report artifacts must not be tracked for public launch",
        ".coverage: local test report artifacts must not be tracked for public launch",
        ".coverage.public-risk: local test report artifacts must not be tracked for public launch",
        "coverage.xml: local test report artifacts must not be tracked for public launch",
        "lcov.info: local test report artifacts must not be tracked for public launch",
        "crates/fathom-core/coverage.lcov: local test report artifacts must not be tracked for public launch",
        "junit.xml: local test report artifacts must not be tracked for public launch",
        "crates/fathom-core/test-output.junit.xml: local test report artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local test report artifacts")
    allowed_notebook_artifact_gitignore = "\n".join(sorted(required_notebook_artifact_gitignore_patterns)) + "\n"
    if gitignore_notebook_artifact_failures(allowed_notebook_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local notebook artifact ignore patterns")
    notebook_artifact_gitignore_failures = gitignore_notebook_artifact_failures("")
    if notebook_artifact_gitignore_failures != [
        ".gitignore: missing local notebook artifact ignore patterns: .ipynb_checkpoints/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local notebook artifact ignore patterns")
    notebook_artifact_failures = tracked_notebook_artifact_file_failures(
        tracked_paths=[
            "notebooks/.ipynb_checkpoints/demo-checkpoint.ipynb",
            "notebooks/with-output.ipynb",
            "notebooks/clean.ipynb",
            "docs/api/public-contract.json",
        ],
        notebook_texts={
            "notebooks/with-output.ipynb": json.dumps(
                {"cells": [{"cell_type": "code", "execution_count": 1, "outputs": [{"output_type": "stream"}]}]}
            ),
            "notebooks/clean.ipynb": json.dumps(
                {"cells": [{"cell_type": "code", "execution_count": None, "outputs": []}]}
            ),
        },
    )
    if notebook_artifact_failures != [
        "notebooks/.ipynb_checkpoints/demo-checkpoint.ipynb: local notebook checkpoint artifacts must not be tracked for public launch",
        "notebooks/with-output.ipynb: notebook execution outputs must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local notebook artifacts")
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
    allowed_rust_artifact_gitignore = "\n".join(sorted(required_rust_artifact_gitignore_patterns)) + "\n"
    if gitignore_rust_artifact_failures(allowed_rust_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Rust/Cargo cache/build artifact ignore patterns")
    rust_artifact_gitignore_failures = gitignore_rust_artifact_failures(
        allowed_rust_artifact_gitignore.replace("*.profraw\n", "")
    )
    if rust_artifact_gitignore_failures != [
        ".gitignore: missing local Rust/Cargo cache/build artifact ignore patterns: *.profraw"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Rust/Cargo cache/build artifact ignore patterns")
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
    allowed_backup_artifact_gitignore = "\n".join(sorted(required_backup_artifact_gitignore_patterns)) + "\n"
    if gitignore_backup_artifact_failures(allowed_backup_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local backup/dump artifact ignore patterns")
    backup_artifact_gitignore_failures = gitignore_backup_artifact_failures(
        allowed_backup_artifact_gitignore.replace("*.sql\n", "")
    )
    if backup_artifact_gitignore_failures != [
        ".gitignore: missing local backup/dump artifact ignore patterns: *.sql"
    ]:
        raise AssertionError("public risk self-test did not reject missing local backup/dump artifact ignore patterns")
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
    allowed_screen_capture_gitignore = "\n".join(sorted(required_screen_capture_gitignore_patterns)) + "\n"
    if gitignore_screen_capture_failures(allowed_screen_capture_gitignore):
        raise AssertionError("public risk self-test rejected complete local screenshot/screen-recording ignore patterns")
    screen_capture_gitignore_failures = gitignore_screen_capture_failures(
        allowed_screen_capture_gitignore.replace("Screenshot *\n", "")
    )
    if screen_capture_gitignore_failures != [
        ".gitignore: missing local screenshot/screen-recording ignore patterns: Screenshot *"
    ]:
        raise AssertionError("public risk self-test did not reject missing local screenshot/screen-recording ignore patterns")
    allowed_media_capture_gitignore = "\n".join(sorted(required_media_capture_gitignore_patterns)) + "\n"
    if gitignore_media_capture_failures(allowed_media_capture_gitignore):
        raise AssertionError("public risk self-test rejected complete local audio/video capture/export ignore patterns")
    media_capture_gitignore_failures = gitignore_media_capture_failures(
        allowed_media_capture_gitignore.replace("*.mp4\n", "")
    )
    if media_capture_gitignore_failures != [
        ".gitignore: missing local audio/video capture/export ignore patterns: *.mp4"
    ]:
        raise AssertionError("public risk self-test did not reject missing local audio/video capture/export ignore patterns")
    mobile_build_failures = tracked_mobile_build_file_failures(
        tracked_paths=[
            "DerivedData/Fathom/Build/Products/Debug/Fathom.app",
            ".gradle/caches/modules-2/files-2.1/metadata.bin",
            "android/local.properties",
            "ios/Fathom.xcodeproj/xcuserdata/tim.xcuserdatad/UserInterfaceState.xcuserstate",
            "TestResults/Fathom.xcresult/Data/data.0~",
            "archives/Fathom.xcarchive/Info.plist",
            "builds/Fathom.ipa",
            "android/app/release/app-release.apk",
            "android/app/release/app-release.aab",
            "Fathom.app.dSYM/Contents/Resources/DWARF/Fathom",
            "profiles/Fathom.mobileprovision",
            "profiles/Fathom.provisionprofile",
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
        "archives/Fathom.xcarchive/Info.plist: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "builds/Fathom.ipa: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "android/app/release/app-release.apk: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "android/app/release/app-release.aab: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "Fathom.app.dSYM/Contents/Resources/DWARF/Fathom: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "profiles/Fathom.mobileprovision: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
        "profiles/Fathom.provisionprofile: local mobile/Xcode/Android build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local mobile/Xcode/Android build artifacts")
    screen_capture_failures = tracked_screen_capture_file_failures(
        tracked_paths=[
            "Screenshot 2026-06-01 at 11.42.00 AM.png",
            "Screen Shot 2026-06-01 at 11.42.00 AM.png",
            "Screen Recording 2026-06-01 at 11.42.00 AM.mov",
            "docs/screenshots/public-contract.png",
            "frontend/public/pacman-favicon.svg",
        ],
    )
    if screen_capture_failures != [
        "Screenshot 2026-06-01 at 11.42.00 AM.png: local screenshot/screen-recording artifacts must not be tracked for public launch",
        "Screen Shot 2026-06-01 at 11.42.00 AM.png: local screenshot/screen-recording artifacts must not be tracked for public launch",
        "Screen Recording 2026-06-01 at 11.42.00 AM.mov: local screenshot/screen-recording artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local screenshot/screen-recording artifacts")
    media_capture_failures = tracked_media_capture_file_failures(
        tracked_paths=[
            "captures/backend-demo.mp4",
            "captures/public-contract.webm",
            "captures/fathom-demo.mov",
            "captures/model-output.m4v",
            "captures/manual-pass.avi",
            "captures/manual-pass.mkv",
            "captures/launch-demo.wav",
            "captures/launch-demo.m4a",
            "captures/launch-demo.mp3",
            "captures/launch-demo.aac",
            "captures/launch-demo.flac",
            "Audio Recording 2026-06-01 at 11.42.00 AM.m4a",
            "Voice Memo 2026-06-01 at 11.42.00 AM.m4a",
            "docs/api/public-contract.json",
            "frontend/public/pacman-favicon.svg",
        ],
    )
    if media_capture_failures != [
        "captures/backend-demo.mp4: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/public-contract.webm: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/fathom-demo.mov: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/model-output.m4v: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/manual-pass.avi: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/manual-pass.mkv: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/launch-demo.wav: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/launch-demo.m4a: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/launch-demo.mp3: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/launch-demo.aac: local audio/video capture/export artifacts must not be tracked for public launch",
        "captures/launch-demo.flac: local audio/video capture/export artifacts must not be tracked for public launch",
        "Audio Recording 2026-06-01 at 11.42.00 AM.m4a: local audio/video capture/export artifacts must not be tracked for public launch",
        "Voice Memo 2026-06-01 at 11.42.00 AM.m4a: local audio/video capture/export artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local audio/video capture/export artifacts")
    symlink_failures = tracked_symlink_failures(
        tracked_entries=[
            ("100644", "README.md"),
            (tracked_symlink_mode, "docs/internal-home"),
            (tracked_symlink_mode, "docs/private-temp"),
            (tracked_symlink_mode, "docs/parent"),
            (tracked_symlink_mode, "docs/public-contract-link"),
            (tracked_symlink_mode, "docs/missing-contract-link"),
        ],
        targets={
            "docs/internal-home": "/Users/example/private-notes.md",
            "docs/private-temp": "~/private-output.json",
            "docs/parent": "../outside-repo.md",
            "docs/public-contract-link": "api/public-contract.json",
            "docs/missing-contract-link": "api/missing-contract.json",
        },
        tracked_paths=[
            "README.md",
            "docs/api/public-contract.json",
            "docs/public-contract-link",
            "docs/missing-contract-link",
        ],
    )
    if symlink_failures != [
        "docs/internal-home: tracked symlink must use a relative in-repository target, found /Users/example/private-notes.md",
        "docs/private-temp: tracked symlink must use a relative in-repository target, found ~/private-output.json",
        "docs/parent: tracked symlink must use a relative in-repository target, found ../outside-repo.md",
        "docs/missing-contract-link: tracked symlink target must resolve to an existing tracked in-repository path, found api/missing-contract.json",
    ]:
        raise AssertionError("public risk self-test did not reject symlinks escaping the repository or resolving to local-only targets")
    lockfile_failures = tracked_dependency_lock_source_failures(
        tracked_paths=[
            "Cargo.lock",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "docs/api/public-contract.json",
        ],
        texts={
            "Cargo.lock": (
                '[[package]]\n'
                'name = "safe"\n'
                'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
                '[[package]]\n'
                'name = "local-helper"\n'
                'path = "../private-helper"\n'
                '[[package]]\n'
                'name = "ssh-only"\n'
                'source = "git+ssh://git@github.com/example/private#abc123"\n'
            ),
            "package-lock.json": (
                '{"packages":{"node_modules/local":{"resolved":"file:../local.tgz"},'
                '"node_modules/private":{"resolved":"https://user:token@example.invalid/private.tgz"}}}'
            ),
            "pnpm-lock.yaml": "packages:\n  /private:\n    resolution: {tarball: /Users/example/private.tgz}\n",
            "yarn.lock": '"safe@npm:^1.0.0":\n  resolution: "safe@npm:1.0.0"\n',
            "docs/api/public-contract.json": '{"source":"file:../ignored.json"}\n',
        },
    )
    if lockfile_failures != [
        'Cargo.lock:6: local file dependency source in lockfile: path = "../private-helper"',
        'Cargo.lock:9: SSH dependency source in lockfile: source = "git+ssh://git@github.com/example/private#abc123"',
        'package-lock.json:1: local file dependency source in lockfile: {"packages":{"node_modules/local":{"resolved":"file:../local.tgz"},"node_modules/private":{"resolved":"https://user:token@example.invalid/private.tgz"}}}',
        'package-lock.json:1: authenticated dependency URL in lockfile: {"packages":{"node_modules/local":{"resolved":"file:../local.tgz"},"node_modules/private":{"resolved":"https://user:token@example.invalid/private.tgz"}}}',
        "pnpm-lock.yaml:3: personal home path in dependency lockfile: resolution: {tarball: /Users/example/private.tgz}",
    ]:
        raise AssertionError("public risk self-test did not reject local/private dependency lockfile sources")
    print("public risk scan self-test passed")

if "--self-test" in sys.argv[1:]:
    self_test()
    raise SystemExit(0)

failures = scan_items(tracked_items())
failures.extend(tracked_dependency_lock_source_failures())
failures.extend(tracked_large_file_failures())
failures.extend(tracked_git_lfs_pointer_failures())
failures.extend(tracked_blocked_file_failures())
failures.extend(tracked_credential_file_failures())
failures.extend(tracked_cloud_credential_file_failures())
failures.extend(tracked_kubernetes_credential_file_failures())
failures.extend(tracked_workspace_context_failures())
failures.extend(tracked_command_history_file_failures())
failures.extend(gitignore_workspace_context_failures())
failures.extend(gitignore_command_history_failures())
failures.extend(gitignore_credential_failures())
failures.extend(gitignore_cloud_credential_failures())
failures.extend(gitignore_kubernetes_credential_failures())
failures.extend(gitignore_os_metadata_failures())
failures.extend(gitignore_editor_artifact_failures())
failures.extend(gitignore_ide_artifact_failures())
failures.extend(gitignore_model_artifact_failures())
failures.extend(gitignore_container_artifact_failures())
failures.extend(gitignore_infra_state_failures())
failures.extend(gitignore_mobile_build_failures())
failures.extend(gitignore_screen_capture_failures())
failures.extend(gitignore_media_capture_failures())
failures.extend(gitignore_rust_artifact_failures())
failures.extend(gitignore_package_artifact_failures())
failures.extend(gitignore_backup_artifact_failures())
failures.extend(gitignore_diagnostic_artifact_failures())
failures.extend(gitignore_python_artifact_failures())
failures.extend(gitignore_python_env_artifact_failures())
failures.extend(gitignore_frontend_artifact_failures())
failures.extend(gitignore_test_report_artifact_failures())
failures.extend(gitignore_notebook_artifact_failures())
failures.extend(gitignore_runtime_artifact_failures())
failures.extend(tracked_runtime_artifact_file_failures())
failures.extend(tracked_diagnostic_artifact_file_failures())
failures.extend(tracked_python_artifact_file_failures())
failures.extend(tracked_python_env_artifact_file_failures())
failures.extend(tracked_frontend_artifact_file_failures())
failures.extend(tracked_rust_artifact_file_failures())
failures.extend(tracked_package_artifact_file_failures())
failures.extend(tracked_backup_artifact_file_failures())
failures.extend(tracked_model_artifact_file_failures())
failures.extend(tracked_container_artifact_file_failures())
failures.extend(tracked_infra_state_file_failures())
failures.extend(tracked_mobile_build_file_failures())
failures.extend(tracked_screen_capture_file_failures())
failures.extend(tracked_media_capture_file_failures())
failures.extend(tracked_test_report_artifact_file_failures())
failures.extend(tracked_notebook_artifact_file_failures())
failures.extend(tracked_symlink_failures())
if failures:
    print("Public risk scan failed:", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print("public risk scan passed")
PY
