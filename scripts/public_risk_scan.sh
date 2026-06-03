#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$@" <<'PY'
import pathlib
import fnmatch
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
        ("Linux home path", re.compile(r"(?<![\w.-])/home/[^\s`'\"<>]+")),
        ("Windows user profile path", re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+[^\s`'\"<>]+", re.IGNORECASE)),
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
    "ehthumbs.db",
    "Thumbs.db",
}
blocked_tracked_os_metadata_dirs = {
    ".Spotlight-V100",
    ".TemporaryItems",
    "__MACOSX",
    ".AppleDouble",
    ".Trashes",
    "$RECYCLE.BIN",
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
blocked_tracked_editor_artifact_dirs = {
    ".history",
}
blocked_tracked_ide_artifact_dirs = {
    ".idea",
    ".settings",
    ".vscode",
    ".zed",
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
    ".crt",
    ".csr",
    ".der",
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".p7b",
    ".p7c",
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
    ".cursor",
    ".windsurf",
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
blocked_tracked_local_ci_artifact_dirs = {
    ".act",
}
blocked_tracked_local_ci_artifact_filenames = {
    ".actrc",
}
required_workspace_gitignore_patterns = {
    "/.aider.chat.history.md",
    "/.aider.input.history",
    "/.aider.tags.cache.v4",
    "/.claude/",
    "/.codex/",
    "/.continue/",
    "/.cursor/",
    "/.openclaw/",
    "/.windsurf/",
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
required_local_ci_artifact_gitignore_patterns = {
    "/.act/",
    ".actrc",
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
    "*.crt",
    "*.csr",
    "*.der",
    "*.p7b",
    "*.p7c",
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
    "$RECYCLE.BIN/",
    "__MACOSX/",
    "ehthumbs.db",
    "Thumbs.db",
    "desktop.ini",
}
required_editor_artifact_gitignore_patterns = {
    ".history/",
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
    "/.zed/",
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
required_experiment_tracking_artifact_gitignore_patterns = {
    "/.wandb/",
    "/lightning_logs/",
    "/mlruns/",
    "/wandb/",
    "events.out.tfevents.*",
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
required_deployment_platform_artifact_gitignore_patterns = {
    "/.netlify/",
    "/.vercel/",
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
required_nix_artifact_gitignore_patterns = {
    "/result",
    "/result-*",
}
required_bazel_artifact_gitignore_patterns = {
    "/bazel-*/",
    "/bazel-bin/",
    "/bazel-out/",
    "/bazel-testlogs/",
}
required_swiftpm_artifact_gitignore_patterns = {
    "/.build/",
    "/.swiftpm/",
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
required_native_build_artifact_gitignore_patterns = {
    "/CMakeFiles/",
    "/cmake-build-*/",
    "CMakeCache.txt",
    "cmake_install.cmake",
    "compile_commands.json",
}
required_package_artifact_gitignore_patterns = {
    "/artifacts/",
    "/release/",
    "/releases/",
    "*.7z",
    "*.app",
    "*.bz2",
    "*.deb",
    "*.dmg",
    "*.egg",
    "*.ear",
    "*.gz",
    "*.jar",
    "*.msi",
    "*.pkg",
    "*.rar",
    "*.rpm",
    "*.tar",
    "*.tgz",
    "*.war",
    "*.whl",
    "*.xz",
    "*.zip",
    "*.zst",
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
    "*.sarif",
    "*.sarif.json",
    "*.trace",
}
required_python_artifact_gitignore_patterns = {
    "__pycache__/",
    ".dmypy.json",
    ".hypothesis/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".pyre/",
    ".pytype/",
    ".ruff_cache/",
    "*.pyc",
    "*.pyo",
}
required_python_env_artifact_gitignore_patterns = {
    "/.pip-cache/",
    "/.uv-cache/",
    "/.uv/",
    "/.nox/",
    "/.tox/",
    "/.venv/",
    "/env/",
    "/pip-cache/",
    "/uv-cache/",
    "/venv/",
    "/wheelhouse/",
    "pip-wheel-metadata/",
    "site-packages/",
}
required_ruby_bundle_artifact_gitignore_patterns = {
    "/.bundle/",
    "/vendor/bundle/",
    "/vendor/cache/",
}
required_php_composer_artifact_gitignore_patterns = {
    "/.phpunit.cache/",
    "/vendor/autoload.php",
    "/vendor/bin/",
    "/vendor/composer/",
    ".phpunit.result.cache",
    "composer.phar",
}
required_r_artifact_gitignore_patterns = {
    "/.Rproj.user/",
    "/renv/library/",
    ".RData",
    ".Rhistory",
    ".Ruserdata",
}
required_go_artifact_gitignore_patterns = {
    "/.gocache/",
    "/.gomodcache/",
    "*.test",
    "coverage.out",
}
required_jvm_dependency_artifact_gitignore_patterns = {
    "/.m2/",
}
required_gradle_artifact_gitignore_patterns = {
    "/.gradle/",
    "build/",
}
required_frontend_artifact_gitignore_patterns = {
    ".bun/",
    ".eslintcache",
    ".npm/",
    ".parcel-cache/",
    ".pnpm-store/",
    ".stylelintcache",
    ".turbo/",
    ".vitest/",
    ".yarn/build-state.yml",
    ".yarn/cache/",
    ".yarn/install-state.gz",
    ".yarn/unplugged/",
    "*.tsbuildinfo",
    "build/",
    "bun-debug.log",
    "coverage/",
    "dist/",
    "frontend/.next/",
    "frontend/.npm/",
    "frontend/.pnpm-store/",
    "frontend/.vite/",
    "frontend/.vitest/",
    "frontend/.yarn/build-state.yml",
    "frontend/.yarn/cache/",
    "frontend/.yarn/install-state.gz",
    "frontend/.yarn/unplugged/",
    "frontend/build/",
    "frontend/coverage/",
    "frontend/dist/",
    "frontend/node_modules/",
    "frontend/.eslintcache",
    "frontend/.bun/",
    "frontend/.stylelintcache",
    "frontend/vite.config.*.timestamp-*",
    "frontend/vitest.config.*.timestamp-*",
    "node_modules/",
    "npm-debug.log",
    "pnpm-debug.log",
    "vite.config.*.timestamp-*",
    "vitest.config.*.timestamp-*",
    "yarn-debug.log",
    "yarn-error.log",
}
required_local_cache_artifact_gitignore_patterns = {
    ".cache/",
}
required_temp_artifact_gitignore_patterns = {
    "/temp/",
    "/tmp/",
    "*.temp",
    "*.tmp",
}
required_test_report_artifact_gitignore_patterns = {
    "/.nyc_output/",
    "/.playwright/",
    "/blob-report/",
    "/cypress/downloads/",
    "/cypress/screenshots/",
    "/cypress/videos/",
    "/htmlcov/",
    "/playwright-report/",
    "/reports/",
    "/test-reports/",
    "/test-results/",
    ".coverage",
    ".coverage.*",
    "*.lcov",
    "*.junit.xml",
    "clover.xml",
    "coverage-final.json",
    "coverage.xml",
    "frontend/cypress/downloads/",
    "frontend/cypress/screenshots/",
    "frontend/cypress/videos/",
    "lcov.info",
    "junit.xml",
}
required_notebook_artifact_gitignore_patterns = {
    ".jupyter/",
    ".ipynb_checkpoints/",
    ".nbhistory",
}
required_doc_build_artifact_gitignore_patterns = {
    "_minted-*/",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.synctex.gz",
}
required_runtime_artifact_gitignore_patterns = {
    "*.db",
    "*.db-journal",
    "*.db-shm",
    "*.db-wal",
    "*.duckdb",
    "*.duckdb.wal",
    "*.pid",
    "*.rdb",
    "*.sqlite",
    "*.sqlite-journal",
    "*.sqlite-shm",
    "*.sqlite-wal",
    "*.sqlite3",
    ".fathom/",
    "dump.rdb",
    "summary.local.json",
}
blocked_tracked_runtime_artifact_filenames = {
    "dump.rdb",
    "server.log",
    "summary.local.json",
}
blocked_tracked_runtime_artifact_suffixes = {
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".duckdb",
    ".duckdb.wal",
    ".log",
    ".pid",
    ".rdb",
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
    ".sarif",
    ".sarif.json",
    ".trace",
}
blocked_tracked_python_artifact_dirs = {
    "__pycache__",
    ".hypothesis",
    ".mypy_cache",
    ".pyre",
    ".pytest_cache",
    ".pytype",
    ".ruff_cache",
}
blocked_tracked_python_artifact_filenames = {
    ".dmypy.json",
}
blocked_tracked_python_artifact_suffixes = {
    ".pyc",
    ".pyo",
}
blocked_tracked_python_env_artifact_dirs = {
    ".nox",
    ".pip-cache",
    ".tox",
    ".uv",
    ".uv-cache",
    ".venv",
    "env",
    "pip-cache",
    "pip-wheel-metadata",
    "site-packages",
    "uv-cache",
    "venv",
    "wheelhouse",
}
blocked_tracked_ruby_bundle_artifact_dirs = {
    ".bundle",
}
blocked_tracked_ruby_vendor_artifact_dirs = {
    ("vendor", "bundle"),
    ("vendor", "cache"),
}
blocked_tracked_php_composer_artifact_dirs = {
    ".phpunit.cache",
}
blocked_tracked_php_composer_vendor_artifact_dirs = {
    ("vendor", "bin"),
    ("vendor", "composer"),
}
blocked_tracked_php_composer_artifact_filenames = {
    ".phpunit.result.cache",
    "composer.phar",
}
blocked_tracked_php_composer_vendor_artifact_paths = {
    ("vendor", "autoload.php"),
}
blocked_tracked_r_artifact_dirs = {
    ".Rproj.user",
}
blocked_tracked_r_dependency_artifact_dirs = {
    ("renv", "library"),
}
blocked_tracked_r_artifact_filenames = {
    ".RData",
    ".Rhistory",
    ".Ruserdata",
}
blocked_tracked_go_artifact_dirs = {
    ".gocache",
    ".gomodcache",
}
blocked_tracked_go_artifact_filenames = {
    "coverage.out",
}
blocked_tracked_go_artifact_suffixes = {
    ".test",
}
blocked_tracked_jvm_dependency_artifact_dirs = {
    ".m2",
}
blocked_tracked_gradle_artifact_dirs = {
    ".gradle",
}
blocked_tracked_gradle_build_output_dirs = {
    "build",
}
blocked_tracked_gradle_build_output_children = {
    ".transforms",
    "classes",
    "distributions",
    "generated",
    "intermediates",
    "kotlin",
    "libs",
    "outputs",
    "reports",
    "resources",
    "test-results",
    "tmp",
}
blocked_tracked_frontend_artifact_dirs = {
    ".bun",
    ".npm",
    ".next",
    ".parcel-cache",
    ".pnpm-store",
    ".turbo",
    ".vite",
    ".vitest",
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
    ".eslintcache",
    ".stylelintcache",
    "bun-debug.log",
    "npm-debug.log",
    "pnpm-debug.log",
    "yarn-debug.log",
    "yarn-error.log",
}
blocked_tracked_frontend_artifact_filename_patterns = {
    "vite.config.*.timestamp-*",
    "vitest.config.*.timestamp-*",
}
blocked_tracked_frontend_artifact_suffixes = {
    ".tsbuildinfo",
}
blocked_tracked_local_cache_artifact_dirs = {
    ".cache",
}
blocked_tracked_temp_artifact_dirs = {
    "temp",
    "tmp",
}
blocked_tracked_temp_artifact_suffixes = {
    ".temp",
    ".tmp",
}
blocked_tracked_rust_artifact_dirs = {
    ".cargo",
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
blocked_tracked_native_build_artifact_dirs = {
    "CMakeFiles",
}
blocked_tracked_native_build_artifact_filenames = {
    "CMakeCache.txt",
    "cmake_install.cmake",
    "compile_commands.json",
}
blocked_tracked_package_artifact_suffixes = {
    ".7z",
    ".app",
    ".bz2",
    ".deb",
    ".dmg",
    ".egg",
    ".ear",
    ".gz",
    ".jar",
    ".msi",
    ".pkg",
    ".rar",
    ".rpm",
    ".tar",
    ".tgz",
    ".war",
    ".whl",
    ".xz",
    ".zip",
    ".zst",
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
blocked_tracked_experiment_tracking_artifact_root_names = {
    ".wandb",
    "lightning_logs",
    "mlruns",
    "wandb",
}
blocked_tracked_experiment_tracking_artifact_filename_patterns = (
    "events.out.tfevents.*",
)
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
blocked_tracked_deployment_platform_artifact_dirs = {
    ".netlify",
    ".vercel",
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
blocked_tracked_nix_artifact_root_names = {
    "result",
}
blocked_tracked_bazel_artifact_root_names = {
    "bazel-bin",
    "bazel-out",
    "bazel-testlogs",
}
blocked_tracked_bazel_output_tree_children = {
    "bazel-out",
    "execroot",
    "external",
}
blocked_tracked_swiftpm_artifact_dirs = {
    ".build",
    ".swiftpm",
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
    ".nyc_output",
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
    "clover.xml",
    "coverage-final.json",
    "coverage.xml",
    "junit.xml",
    "lcov.info",
}
blocked_tracked_test_report_artifact_suffixes = {
    ".lcov",
    ".junit.xml",
}
blocked_tracked_cypress_artifact_dirs = {
    "downloads",
    "screenshots",
    "videos",
}
blocked_tracked_notebook_artifact_dirs = {
    ".jupyter",
    ".ipynb_checkpoints",
}
blocked_tracked_notebook_artifact_filenames = {
    ".nbhistory",
}
blocked_tracked_doc_build_artifact_suffixes = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".synctex.gz",
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
        re.compile(
            r"\b(?:file|link|portal):(?:\.\.?/|/|~)"
            r"|\bpath\s*=\s*[\"'](?:\.\.?/|/|~)"
            r"|\bpatch:[^\s\"']*#(?:\.\.?/|/|~)",
            re.IGNORECASE,
        ),
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
gitmodules_source_patterns = [
    (
        "local or relative submodule URL in .gitmodules",
        re.compile(r"\burl\s*=\s*(?:file:|(?:\.\.?|~|/)[^\s]+)", re.IGNORECASE),
    ),
    (
        "SSH submodule URL in .gitmodules",
        re.compile(r"\b(?:git\+)?ssh://|git@[A-Za-z0-9_.-]+:", re.IGNORECASE),
    ),
    (
        "authenticated submodule URL in .gitmodules",
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

def tracked_gitmodules_source_failures(tracked_paths=None, texts=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        if rel != ".gitmodules":
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
                    failures.append(f"{rel}:{line_no}: {label} in Git submodule metadata: {line.strip()}")
            for label, pattern in secret_value_patterns():
                if pattern.search(privacy_line):
                    failures.append(f"{rel}:{line_no}: {label} in Git submodule metadata: {line.strip()}")
            for label, pattern in gitmodules_source_patterns:
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
        if (
            path.name.endswith("~")
            or path.suffix.lower() in blocked_tracked_editor_artifact_suffixes
            or any(part in blocked_tracked_editor_artifact_dirs for part in path.parts)
        ):
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
        if len(path.parts) >= 2 and path.parts[0] == ".config" and path.parts[1] == "gcloud":
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

def tracked_local_ci_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_local_ci_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: local CI runner artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_local_ci_artifact_filenames:
            failures.append(f"{rel}: local CI runner config must not be tracked for public launch")
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

def gitignore_local_ci_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local CI runner artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_local_ci_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local CI runner artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_experiment_tracking_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local ML experiment/tracking artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_experiment_tracking_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local ML experiment/tracking artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_deployment_platform_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local deployment platform artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_deployment_platform_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local deployment platform artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_nix_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Nix build result artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_nix_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Nix build result artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_bazel_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Bazel build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_bazel_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Bazel build artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_swiftpm_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Swift Package Manager artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_swiftpm_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Swift Package Manager artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_native_build_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local native/CMake build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_native_build_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local native/CMake build artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_ruby_bundle_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Ruby/Bundler dependency artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_ruby_bundle_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Ruby/Bundler dependency artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_php_composer_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local PHP Composer dependency/test artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_php_composer_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local PHP Composer dependency/test artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_r_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local R/RStudio artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_r_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local R/RStudio artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_go_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Go cache/test artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_go_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Go cache/test artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_jvm_dependency_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local JVM dependency artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_jvm_dependency_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local JVM dependency artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_gradle_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local Gradle/JVM build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_gradle_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local Gradle/JVM build artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_local_cache_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local cache artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_local_cache_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local cache artifact ignore patterns: {', '.join(missing)}"]
    return []

def gitignore_temp_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local temporary/scratch artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_temp_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local temporary/scratch artifact ignore patterns: {', '.join(missing)}"]
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

def gitignore_doc_build_artifact_failures(gitignore_text=None):
    if gitignore_text is None:
        try:
            gitignore_text = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        except FileNotFoundError:
            return [".gitignore: missing local documentation build artifact ignore patterns"]
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_doc_build_artifact_gitignore_patterns - active_patterns)
    if missing:
        return [f".gitignore: missing local documentation build artifact ignore patterns: {', '.join(missing)}"]
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
        if path.name in blocked_tracked_python_artifact_filenames:
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

def tracked_ruby_bundle_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_ruby_bundle_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: Ruby/Bundler dependency artifacts must not be tracked for public launch")
            continue
        if len(path.parts) >= 2 and tuple(path.parts[:2]) in blocked_tracked_ruby_vendor_artifact_dirs:
            failures.append(f"{rel}: Ruby/Bundler dependency artifacts must not be tracked for public launch")
    return failures

def tracked_php_composer_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_php_composer_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: PHP Composer dependency/test artifacts must not be tracked for public launch")
            continue
        if len(path.parts) >= 2 and tuple(path.parts[:2]) in blocked_tracked_php_composer_vendor_artifact_dirs:
            failures.append(f"{rel}: PHP Composer dependency artifacts must not be tracked for public launch")
            continue
        if len(path.parts) >= 2 and tuple(path.parts[:2]) in blocked_tracked_php_composer_vendor_artifact_paths:
            failures.append(f"{rel}: PHP Composer dependency artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_php_composer_artifact_filenames:
            failures.append(f"{rel}: PHP Composer dependency/test artifacts must not be tracked for public launch")
    return failures

def tracked_r_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_r_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: R/RStudio local artifacts must not be tracked for public launch")
            continue
        if len(path.parts) >= 2 and tuple(path.parts[:2]) in blocked_tracked_r_dependency_artifact_dirs:
            failures.append(f"{rel}: R/RStudio local artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_r_artifact_filenames:
            failures.append(f"{rel}: R/RStudio local artifacts must not be tracked for public launch")
    return failures

def tracked_go_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_go_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: Go cache/test artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_go_artifact_filenames:
            failures.append(f"{rel}: Go cache/test artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_go_artifact_suffixes:
            failures.append(f"{rel}: Go cache/test artifacts must not be tracked for public launch")
    return failures

def tracked_jvm_dependency_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_jvm_dependency_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: JVM dependency artifacts must not be tracked for public launch")
    return failures

def tracked_gradle_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_gradle_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: Gradle/JVM build artifacts must not be tracked for public launch")
            continue
        for index, part in enumerate(path.parts[:-1]):
            if part not in blocked_tracked_gradle_build_output_dirs:
                continue
            if path.parts[index + 1] in blocked_tracked_gradle_build_output_children:
                failures.append(f"{rel}: Gradle/JVM build artifacts must not be tracked for public launch")
                break
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
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in blocked_tracked_frontend_artifact_filename_patterns):
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_frontend_artifact_suffixes:
            failures.append(f"{rel}: frontend/Node cache/build artifacts must not be tracked for public launch")
    return failures

def tracked_local_cache_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_local_cache_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: local cache artifacts must not be tracked for public launch")
    return failures

def tracked_temp_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] in blocked_tracked_temp_artifact_dirs:
            failures.append(f"{rel}: local temporary/scratch artifacts must not be tracked for public launch")
            continue
        if path.suffix.lower() in blocked_tracked_temp_artifact_suffixes:
            failures.append(f"{rel}: local temporary/scratch artifacts must not be tracked for public launch")
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

def tracked_native_build_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0].startswith("cmake-build-"):
            failures.append(f"{rel}: native/CMake build artifacts must not be tracked for public launch")
            continue
        if any(part in blocked_tracked_native_build_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: native/CMake build artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_native_build_artifact_filenames:
            failures.append(f"{rel}: native/CMake build artifacts must not be tracked for public launch")
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

def tracked_experiment_tracking_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if path.parts and path.parts[0] in blocked_tracked_experiment_tracking_artifact_root_names:
            failures.append(f"{rel}: local ML experiment/tracking artifacts must not be tracked for public launch")
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in blocked_tracked_experiment_tracking_artifact_filename_patterns):
            failures.append(f"{rel}: local ML experiment/tracking artifacts must not be tracked for public launch")
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

def tracked_deployment_platform_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_deployment_platform_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: local deployment platform artifacts must not be tracked for public launch")
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

def tracked_nix_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if not path.parts:
            continue
        root_name = path.parts[0]
        if root_name in blocked_tracked_nix_artifact_root_names or root_name.startswith("result-"):
            failures.append(f"{rel}: local Nix build result artifacts must not be tracked for public launch")
    return failures

def tracked_bazel_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if not path.parts:
            continue
        root_name = path.parts[0]
        if root_name in blocked_tracked_bazel_artifact_root_names:
            failures.append(f"{rel}: local Bazel build artifacts must not be tracked for public launch")
            continue
        if (
            root_name.startswith("bazel-")
            and len(path.parts) > 1
            and path.parts[1] in blocked_tracked_bazel_output_tree_children
        ):
            failures.append(f"{rel}: local Bazel build artifacts must not be tracked for public launch")
    return failures

def tracked_swiftpm_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        if any(part in blocked_tracked_swiftpm_artifact_dirs for part in path.parts):
            failures.append(f"{rel}: local Swift Package Manager artifacts must not be tracked for public launch")
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
        if any(
            part == "cypress" and path.parts[index + 1] in blocked_tracked_cypress_artifact_dirs
            for index, part in enumerate(path.parts[:-1])
        ):
            failures.append(f"{rel}: local browser-test artifacts must not be tracked for public launch")
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
            if ".jupyter" in path.parts:
                failures.append(f"{rel}: local notebook runtime/config artifacts must not be tracked for public launch")
            else:
                failures.append(f"{rel}: local notebook checkpoint artifacts must not be tracked for public launch")
            continue
        if path.name in blocked_tracked_notebook_artifact_filenames:
            failures.append(f"{rel}: local notebook runtime/config artifacts must not be tracked for public launch")
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

def tracked_doc_build_artifact_file_failures(tracked_paths=None):
    if tracked_paths is None:
        tracked_paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in tracked_paths:
        path = pathlib.PurePosixPath(rel)
        lower_name = path.name.lower()
        if any(part.startswith("_minted-") for part in path.parts):
            failures.append(f"{rel}: local documentation build artifacts must not be tracked for public launch")
            continue
        if any(lower_name.endswith(suffix) for suffix in blocked_tracked_doc_build_artifact_suffixes):
            failures.append(f"{rel}: local documentation build artifacts must not be tracked for public launch")
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
        ("docs/benchmarks/example.md", "Reference repo: /home/example/fathom-private/llama.cpp"),
        ("docs/api/example.md", r"Artifact output: C:\Users\example\AppData\Local\Temp\fathom-summary.json"),
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
            "frontend/ehthumbs.db",
            "$RECYCLE.BIN/S-1-5-21-1234567890/deleted.md",
            "desktop.ini",
            "README.md~",
            "docs/.history/public-launch-evidence.md",
            "docs/launch-review.patch",
            "docs/launch-review.diff",
            "docs/api/client-examples.md.swp",
            "docs/public-launch-checklist.md.orig",
            "docs/public-launch-evidence.md.rej",
            ".vscode/settings.json",
            ".idea/workspace.xml",
            ".settings/org.eclipse.jdt.core.prefs",
            ".zed/settings.json",
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
        "frontend/ehthumbs.db: OS/platform metadata files must not be tracked for public launch",
        "$RECYCLE.BIN/S-1-5-21-1234567890/deleted.md: OS/platform metadata files must not be tracked for public launch",
        "desktop.ini: OS/platform metadata files must not be tracked for public launch",
        "README.md~: editor backup/swap artifacts must not be tracked for public launch",
        "docs/.history/public-launch-evidence.md: editor backup/swap artifacts must not be tracked for public launch",
        "docs/launch-review.patch: editor backup/swap artifacts must not be tracked for public launch",
        "docs/launch-review.diff: editor backup/swap artifacts must not be tracked for public launch",
        "docs/api/client-examples.md.swp: editor backup/swap artifacts must not be tracked for public launch",
        "docs/public-launch-checklist.md.orig: editor backup/swap artifacts must not be tracked for public launch",
        "docs/public-launch-evidence.md.rej: editor backup/swap artifacts must not be tracked for public launch",
        ".vscode/settings.json: IDE workspace/config artifacts must not be tracked for public launch",
        ".idea/workspace.xml: IDE workspace/config artifacts must not be tracked for public launch",
        ".settings/org.eclipse.jdt.core.prefs: IDE workspace/config artifacts must not be tracked for public launch",
        ".zed/settings.json: IDE workspace/config artifacts must not be tracked for public launch",
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
            "tls/fathom-local.crt",
            "tls/fathom-local.csr",
            "tls/fathom-local.der",
            "tls/fathom-local.p7b",
            "tls/fathom-local.p7c",
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
        "tls/fathom-local.crt: credential/key files must not be tracked for public launch",
        "tls/fathom-local.csr: credential/key files must not be tracked for public launch",
        "tls/fathom-local.der: credential/key files must not be tracked for public launch",
        "tls/fathom-local.p7b: credential/key files must not be tracked for public launch",
        "tls/fathom-local.p7c: credential/key files must not be tracked for public launch",
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
            ".config/gcloud",
            ".config/gcloud/application_default_credentials.json",
            ".config/gcloud/configurations/config_default",
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
        ".config/gcloud: cloud SDK credential/config files must not be tracked for public launch",
        ".config/gcloud/application_default_credentials.json: cloud SDK credential/config files must not be tracked for public launch",
        ".config/gcloud/configurations/config_default: cloud SDK credential/config files must not be tracked for public launch",
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
            ".cursor/rules/project.mdc",
            ".windsurf/config.json",
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
        ".cursor/rules/project.mdc: workspace/personal agent context files must not be tracked for public launch",
        ".windsurf/config.json: workspace/personal agent context files must not be tracked for public launch",
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
    local_ci_artifact_failures = tracked_local_ci_artifact_file_failures(
        tracked_paths=[
            ".act/event.json",
            ".act/workflows/ci-1/container-options.json",
            ".actrc",
            "docs/api/public-contract.json",
        ],
    )
    if local_ci_artifact_failures != [
        ".act/event.json: local CI runner artifacts must not be tracked for public launch",
        ".act/workflows/ci-1/container-options.json: local CI runner artifacts must not be tracked for public launch",
        ".actrc: local CI runner config must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local CI runner artifacts")
    allowed_local_ci_artifact_gitignore = "\n".join(sorted(required_local_ci_artifact_gitignore_patterns)) + "\n"
    if gitignore_local_ci_artifact_failures(allowed_local_ci_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local CI runner artifact ignore patterns")
    local_ci_artifact_gitignore_failures = gitignore_local_ci_artifact_failures(
        allowed_local_ci_artifact_gitignore.replace("/.act/\n", "")
    )
    if local_ci_artifact_gitignore_failures != [
        ".gitignore: missing local CI runner artifact ignore patterns: /.act/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local CI runner artifact ignore patterns")
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
            "state/fathom.duckdb",
            "state/fathom.duckdb.wal",
            "state/dump.rdb",
            "redis/dump.rdb",
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
        "state/fathom.duckdb: local runtime/artifact detail files must not be tracked for public launch",
        "state/fathom.duckdb.wal: local runtime/artifact detail files must not be tracked for public launch",
        "state/dump.rdb: local runtime/artifact detail files must not be tracked for public launch",
        "redis/dump.rdb: local runtime/artifact detail files must not be tracked for public launch",
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
        allowed_diagnostic_artifact_gitignore.replace("*.sarif.json\n", "")
    )
    if diagnostic_artifact_gitignore_failures != [
        ".gitignore: missing local log/trace/profiling/debug-output artifact ignore patterns: *.sarif.json"
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
            "reports/codeql.sarif",
            "reports/semgrep.sarif.json",
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
        "reports/codeql.sarif: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
        "reports/semgrep.sarif.json: local log/trace/profiling/debug-output artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local log/trace/profiling/debug-output artifacts")
    python_artifact_failures = tracked_python_artifact_file_failures(
        tracked_paths=[
            "scripts/__pycache__/public_api_contract_qa.cpython-312.pyc",
            ".hypothesis/examples/7c1b4d6c3d8f2a1e",
            ".pytest_cache/v/cache/nodeids",
            ".mypy_cache/3.12/scripts/public_api_contract_qa.data.json",
            ".dmypy.json",
            ".pyre/server/server.stderr",
            ".pytype/pyi/scripts/public_api_contract_qa.pyi",
            ".ruff_cache/0.12.0/file",
            "scripts/public_api_contract_qa.pyo",
            "docs/api/public-contract.json",
            "scripts/public_api_contract_qa.py",
        ],
    )
    if python_artifact_failures != [
        "scripts/__pycache__/public_api_contract_qa.cpython-312.pyc: Python cache/build artifacts must not be tracked for public launch",
        ".hypothesis/examples/7c1b4d6c3d8f2a1e: Python cache/build artifacts must not be tracked for public launch",
        ".pytest_cache/v/cache/nodeids: Python cache/build artifacts must not be tracked for public launch",
        ".mypy_cache/3.12/scripts/public_api_contract_qa.data.json: Python cache/build artifacts must not be tracked for public launch",
        ".dmypy.json: Python cache/build artifacts must not be tracked for public launch",
        ".pyre/server/server.stderr: Python cache/build artifacts must not be tracked for public launch",
        ".pytype/pyi/scripts/public_api_contract_qa.pyi: Python cache/build artifacts must not be tracked for public launch",
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
            ".uv/cache/archive-v0/package",
            ".uv-cache/archive-v0/package",
            ".pip-cache/http-v2/cache-entry",
            "uv-cache/archive-v0/package",
            "pip-cache/http-v2/cache-entry",
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
        ".uv/cache/archive-v0/package: Python virtualenv/dependency artifacts must not be tracked for public launch",
        ".uv-cache/archive-v0/package: Python virtualenv/dependency artifacts must not be tracked for public launch",
        ".pip-cache/http-v2/cache-entry: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "uv-cache/archive-v0/package: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "pip-cache/http-v2/cache-entry: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "wheelhouse/fathom-0.1.0-py3-none-any.whl: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "pip-wheel-metadata/fathom.json: Python virtualenv/dependency artifacts must not be tracked for public launch",
        "python/site-packages/fathom/__init__.py: Python virtualenv/dependency artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Python virtualenv/dependency artifacts")
    allowed_python_env_artifact_gitignore = "\n".join(sorted(required_python_env_artifact_gitignore_patterns)) + "\n"
    if gitignore_python_env_artifact_failures(allowed_python_env_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Python virtualenv/dependency artifact ignore patterns")
    python_env_artifact_gitignore_failures = gitignore_python_env_artifact_failures(
        allowed_python_env_artifact_gitignore.replace("/.uv-cache/\n", "")
    )
    if python_env_artifact_gitignore_failures != [
        ".gitignore: missing local Python virtualenv/dependency artifact ignore patterns: /.uv-cache/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Python virtualenv/dependency artifact ignore patterns")
    ruby_bundle_artifact_failures = tracked_ruby_bundle_artifact_file_failures(
        tracked_paths=[
            ".bundle/config",
            "vendor/bundle/ruby/3.3.0/gems/rack-3.0.0/lib/rack.rb",
            "vendor/cache/rack-3.0.0.gem",
            "docs/api/public-contract.json",
            "Gemfile",
            "Gemfile.lock",
            "vendor/safe/README.md",
        ],
    )
    if ruby_bundle_artifact_failures != [
        ".bundle/config: Ruby/Bundler dependency artifacts must not be tracked for public launch",
        "vendor/bundle/ruby/3.3.0/gems/rack-3.0.0/lib/rack.rb: Ruby/Bundler dependency artifacts must not be tracked for public launch",
        "vendor/cache/rack-3.0.0.gem: Ruby/Bundler dependency artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked Ruby/Bundler dependency artifacts")
    allowed_ruby_bundle_artifact_gitignore = "\n".join(sorted(required_ruby_bundle_artifact_gitignore_patterns)) + "\n"
    if gitignore_ruby_bundle_artifact_failures(allowed_ruby_bundle_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Ruby/Bundler dependency artifact ignore patterns")
    ruby_bundle_artifact_gitignore_failures = gitignore_ruby_bundle_artifact_failures(
        allowed_ruby_bundle_artifact_gitignore.replace("/vendor/cache/\n", "")
    )
    if ruby_bundle_artifact_gitignore_failures != [
        ".gitignore: missing local Ruby/Bundler dependency artifact ignore patterns: /vendor/cache/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Ruby/Bundler dependency artifact ignore patterns")
    php_composer_artifact_failures = tracked_php_composer_artifact_file_failures(
        tracked_paths=[
            "vendor/autoload.php",
            "vendor/bin/phpunit",
            "vendor/composer/installed.json",
            ".phpunit.cache/test-results",
            ".phpunit.result.cache",
            "composer.phar",
            "composer.json",
            "composer.lock",
            "src/Fathom.php",
            "vendor/local-source/README.md",
        ],
    )
    if php_composer_artifact_failures != [
        "vendor/autoload.php: PHP Composer dependency artifacts must not be tracked for public launch",
        "vendor/bin/phpunit: PHP Composer dependency artifacts must not be tracked for public launch",
        "vendor/composer/installed.json: PHP Composer dependency artifacts must not be tracked for public launch",
        ".phpunit.cache/test-results: PHP Composer dependency/test artifacts must not be tracked for public launch",
        ".phpunit.result.cache: PHP Composer dependency/test artifacts must not be tracked for public launch",
        "composer.phar: PHP Composer dependency/test artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local PHP Composer dependency/test artifacts")
    allowed_php_composer_artifact_gitignore = "\n".join(sorted(required_php_composer_artifact_gitignore_patterns)) + "\n"
    if gitignore_php_composer_artifact_failures(allowed_php_composer_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local PHP Composer dependency/test artifact ignore patterns")
    php_composer_artifact_gitignore_failures = gitignore_php_composer_artifact_failures(
        allowed_php_composer_artifact_gitignore.replace("/vendor/composer/\n", "")
    )
    if php_composer_artifact_gitignore_failures != [
        ".gitignore: missing local PHP Composer dependency/test artifact ignore patterns: /vendor/composer/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local PHP Composer dependency/test artifact ignore patterns")
    r_artifact_failures = tracked_r_artifact_file_failures(
        tracked_paths=[
            ".Rproj.user/123/session.json",
            "analysis/.Rhistory",
            "analysis/.RData",
            "analysis/.Ruserdata",
            "renv/library/macos/R-4.4/x86_64-apple-darwin20/dplyr/DESCRIPTION",
            "analysis/fathom.R",
            "analysis/fathom.Rproj",
            "renv.lock",
            "docs/r-analysis.md",
        ],
    )
    if r_artifact_failures != [
        ".Rproj.user/123/session.json: R/RStudio local artifacts must not be tracked for public launch",
        "analysis/.Rhistory: R/RStudio local artifacts must not be tracked for public launch",
        "analysis/.RData: R/RStudio local artifacts must not be tracked for public launch",
        "analysis/.Ruserdata: R/RStudio local artifacts must not be tracked for public launch",
        "renv/library/macos/R-4.4/x86_64-apple-darwin20/dplyr/DESCRIPTION: R/RStudio local artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local R/RStudio artifacts")
    allowed_r_artifact_gitignore = "\n".join(sorted(required_r_artifact_gitignore_patterns)) + "\n"
    if gitignore_r_artifact_failures(allowed_r_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local R/RStudio artifact ignore patterns")
    r_artifact_gitignore_failures = gitignore_r_artifact_failures(
        allowed_r_artifact_gitignore.replace("/renv/library/\n", "")
    )
    if r_artifact_gitignore_failures != [
        ".gitignore: missing local R/RStudio artifact ignore patterns: /renv/library/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local R/RStudio artifact ignore patterns")
    go_artifact_failures = tracked_go_artifact_file_failures(
        tracked_paths=[
            ".gocache/00/abcdef-a",
            ".gomodcache/cache/download/example.com/fathom/@v/v0.1.0.mod",
            "cmd/fathom/fathom.test",
            "coverage.out",
            "go.mod",
            "go.sum",
            "docs/go-testing.md",
        ],
    )
    if go_artifact_failures != [
        ".gocache/00/abcdef-a: Go cache/test artifacts must not be tracked for public launch",
        ".gomodcache/cache/download/example.com/fathom/@v/v0.1.0.mod: Go cache/test artifacts must not be tracked for public launch",
        "cmd/fathom/fathom.test: Go cache/test artifacts must not be tracked for public launch",
        "coverage.out: Go cache/test artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local Go cache/test artifacts")
    allowed_go_artifact_gitignore = "\n".join(sorted(required_go_artifact_gitignore_patterns)) + "\n"
    if gitignore_go_artifact_failures(allowed_go_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Go cache/test artifact ignore patterns")
    go_artifact_gitignore_failures = gitignore_go_artifact_failures(
        allowed_go_artifact_gitignore.replace("coverage.out\n", "")
    )
    if go_artifact_gitignore_failures != [
        ".gitignore: missing local Go cache/test artifact ignore patterns: coverage.out"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Go cache/test artifact ignore patterns")
    jvm_dependency_artifact_failures = tracked_jvm_dependency_artifact_file_failures(
        tracked_paths=[
            ".m2/repository/com/example/private-lib/1.0/private-lib-1.0.jar",
            ".m2/settings.xml",
            "docs/.m2/repository/com/example/private-lib/1.0/private-lib-1.0.pom",
            "pom.xml",
            "docs/java-build.md",
        ],
    )
    if jvm_dependency_artifact_failures != [
        ".m2/repository/com/example/private-lib/1.0/private-lib-1.0.jar: JVM dependency artifacts must not be tracked for public launch",
        ".m2/settings.xml: JVM dependency artifacts must not be tracked for public launch",
        "docs/.m2/repository/com/example/private-lib/1.0/private-lib-1.0.pom: JVM dependency artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local JVM dependency artifacts")
    allowed_jvm_dependency_artifact_gitignore = "\n".join(sorted(required_jvm_dependency_artifact_gitignore_patterns)) + "\n"
    if gitignore_jvm_dependency_artifact_failures(allowed_jvm_dependency_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local JVM dependency artifact ignore patterns")
    jvm_dependency_artifact_gitignore_failures = gitignore_jvm_dependency_artifact_failures("")
    if jvm_dependency_artifact_gitignore_failures != [
        ".gitignore: missing local JVM dependency artifact ignore patterns: /.m2/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local JVM dependency artifact ignore patterns")
    gradle_artifact_failures = tracked_gradle_artifact_file_failures(
        tracked_paths=[
            ".gradle/caches/modules-2/files-2.1/metadata.bin",
            ".gradle/buildOutputCleanup/cache.properties",
            "app/build/classes/java/main/App.class",
            "android/app/build/intermediates/merged_manifest/debug/AndroidManifest.xml",
            "service/build/reports/tests/test/index.html",
            "service/build/test-results/test/TEST-service.xml",
            "service/build/tmp/compileJava/previous-compilation-data.bin",
            "service/build/generated/sources/annotationProcessor/java/main/Generated.java",
            "build.gradle",
            "settings.gradle",
            "gradle.properties",
            "gradlew",
            "gradle/wrapper/gradle-wrapper.properties",
            "gradle/wrapper/gradle-wrapper.jar",
            "docs/gradle-build.md",
        ],
    )
    if gradle_artifact_failures != [
        ".gradle/caches/modules-2/files-2.1/metadata.bin: Gradle/JVM build artifacts must not be tracked for public launch",
        ".gradle/buildOutputCleanup/cache.properties: Gradle/JVM build artifacts must not be tracked for public launch",
        "app/build/classes/java/main/App.class: Gradle/JVM build artifacts must not be tracked for public launch",
        "android/app/build/intermediates/merged_manifest/debug/AndroidManifest.xml: Gradle/JVM build artifacts must not be tracked for public launch",
        "service/build/reports/tests/test/index.html: Gradle/JVM build artifacts must not be tracked for public launch",
        "service/build/test-results/test/TEST-service.xml: Gradle/JVM build artifacts must not be tracked for public launch",
        "service/build/tmp/compileJava/previous-compilation-data.bin: Gradle/JVM build artifacts must not be tracked for public launch",
        "service/build/generated/sources/annotationProcessor/java/main/Generated.java: Gradle/JVM build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local Gradle/JVM build artifacts")
    allowed_gradle_artifact_gitignore = "\n".join(sorted(required_gradle_artifact_gitignore_patterns)) + "\n"
    if gitignore_gradle_artifact_failures(allowed_gradle_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Gradle/JVM build artifact ignore patterns")
    gradle_artifact_gitignore_failures = gitignore_gradle_artifact_failures(
        allowed_gradle_artifact_gitignore.replace("build/\n", "")
    )
    if gradle_artifact_gitignore_failures != [
        ".gitignore: missing local Gradle/JVM build artifact ignore patterns: build/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Gradle/JVM build artifact ignore patterns")
    frontend_artifact_failures = tracked_frontend_artifact_file_failures(
        tracked_paths=[
            "frontend/node_modules/.package-lock.json",
            ".bun/install/cache/react/package.json",
            ".npm/_cacache/index-v5/00/cache-entry",
            ".pnpm-store/v3/files/00/package",
            ".yarn/cache/react-npm-18.2.0.zip",
            ".yarn/unplugged/esbuild/package.json",
            ".yarn/build-state.yml",
            ".yarn/install-state.gz",
            "frontend/.bun/install/cache/vite/package.json",
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
            "frontend/.vitest/cache/results.json",
            ".turbo/cache/build.log",
            ".parcel-cache/data.mdb",
            ".vitest/cache/results.json",
            ".eslintcache",
            ".stylelintcache",
            "frontend/.eslintcache",
            "frontend/.stylelintcache",
            "frontend/src/tsconfig.tsbuildinfo",
            "vite.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs",
            "vitest.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs",
            "frontend/vite.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs",
            "frontend/vitest.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs",
            "bun-debug.log",
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
        ".bun/install/cache/react/package.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".npm/_cacache/index-v5/00/cache-entry: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".pnpm-store/v3/files/00/package: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/cache/react-npm-18.2.0.zip: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/unplugged/esbuild/package.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/build-state.yml: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".yarn/install-state.gz: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.bun/install/cache/vite/package.json: frontend/Node cache/build artifacts must not be tracked for public launch",
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
        "frontend/.vitest/cache/results.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".turbo/cache/build.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".parcel-cache/data.mdb: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".vitest/cache/results.json: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".eslintcache: frontend/Node cache/build artifacts must not be tracked for public launch",
        ".stylelintcache: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.eslintcache: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/.stylelintcache: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/src/tsconfig.tsbuildinfo: frontend/Node cache/build artifacts must not be tracked for public launch",
        "vite.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs: frontend/Node cache/build artifacts must not be tracked for public launch",
        "vitest.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/vite.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs: frontend/Node cache/build artifacts must not be tracked for public launch",
        "frontend/vitest.config.ts.timestamp-1710000000000-a1b2c3d4e5f6.mjs: frontend/Node cache/build artifacts must not be tracked for public launch",
        "bun-debug.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        "npm-debug.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        "yarn-error.log: frontend/Node cache/build artifacts must not be tracked for public launch",
        "pnpm-debug.log: frontend/Node cache/build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked frontend/Node cache/build artifacts")
    allowed_frontend_artifact_gitignore = "\n".join(sorted(required_frontend_artifact_gitignore_patterns)) + "\n"
    if gitignore_frontend_artifact_failures(allowed_frontend_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local frontend/Node cache/build artifact ignore patterns")
    frontend_artifact_gitignore_failures = gitignore_frontend_artifact_failures(
        allowed_frontend_artifact_gitignore.replace("frontend/.vitest/\n", "")
    )
    if frontend_artifact_gitignore_failures != [
        ".gitignore: missing local frontend/Node cache/build artifact ignore patterns: frontend/.vitest/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local frontend/Node cache/build artifact ignore patterns")
    local_cache_artifact_failures = tracked_local_cache_artifact_file_failures(
        tracked_paths=[
            ".cache/huggingface/download/model.safetensors",
            "docs/.cache/rendered/preview.html",
            "docs/api/public-contract.json",
        ],
    )
    if local_cache_artifact_failures != [
        ".cache/huggingface/download/model.safetensors: local cache artifacts must not be tracked for public launch",
        "docs/.cache/rendered/preview.html: local cache artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local cache artifacts")
    allowed_local_cache_artifact_gitignore = "\n".join(sorted(required_local_cache_artifact_gitignore_patterns)) + "\n"
    if gitignore_local_cache_artifact_failures(allowed_local_cache_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local cache artifact ignore patterns")
    local_cache_artifact_gitignore_failures = gitignore_local_cache_artifact_failures("")
    if local_cache_artifact_gitignore_failures != [
        ".gitignore: missing local cache artifact ignore patterns: .cache/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local cache artifact ignore patterns")
    temp_artifact_failures = tracked_temp_artifact_file_failures(
        tracked_paths=[
            "tmp/public-contract-smoke-summary.json",
            "temp/model-registry.json",
            "docs/api/generated.tmp",
            "frontend/src/App.jsx",
        ],
    )
    if temp_artifact_failures != [
        "tmp/public-contract-smoke-summary.json: local temporary/scratch artifacts must not be tracked for public launch",
        "temp/model-registry.json: local temporary/scratch artifacts must not be tracked for public launch",
        "docs/api/generated.tmp: local temporary/scratch artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local temporary/scratch artifacts")
    allowed_temp_artifact_gitignore = "\n".join(sorted(required_temp_artifact_gitignore_patterns)) + "\n"
    if gitignore_temp_artifact_failures(allowed_temp_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local temporary/scratch artifact ignore patterns")
    temp_artifact_gitignore_failures = gitignore_temp_artifact_failures(
        allowed_temp_artifact_gitignore.replace("*.tmp\n", "")
    )
    if temp_artifact_gitignore_failures != [
        ".gitignore: missing local temporary/scratch artifact ignore patterns: *.tmp"
    ]:
        raise AssertionError("public risk self-test did not reject missing local temporary/scratch artifact ignore patterns")
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
            "cypress/screenshots/chat/refusal.png",
            "cypress/videos/chat/refusal.mp4",
            "cypress/downloads/public-contract-summary.json",
            "frontend/cypress/screenshots/chat/refusal.png",
            "test-results/junit.xml",
            "test-reports/backend.xml",
            "reports/public-risk/index.html",
            "htmlcov/index.html",
            ".nyc_output/processinfo/index.json",
            ".coverage",
            ".coverage.public-risk",
            "coverage.xml",
            "coverage-final.json",
            "clover.xml",
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
        "cypress/screenshots/chat/refusal.png: local browser-test artifacts must not be tracked for public launch",
        "cypress/videos/chat/refusal.mp4: local browser-test artifacts must not be tracked for public launch",
        "cypress/downloads/public-contract-summary.json: local browser-test artifacts must not be tracked for public launch",
        "frontend/cypress/screenshots/chat/refusal.png: local browser-test artifacts must not be tracked for public launch",
        "test-results/junit.xml: local test report artifacts must not be tracked for public launch",
        "test-reports/backend.xml: local test report artifacts must not be tracked for public launch",
        "reports/public-risk/index.html: local test report artifacts must not be tracked for public launch",
        "htmlcov/index.html: local test report artifacts must not be tracked for public launch",
        ".nyc_output/processinfo/index.json: local test report artifacts must not be tracked for public launch",
        ".coverage: local test report artifacts must not be tracked for public launch",
        ".coverage.public-risk: local test report artifacts must not be tracked for public launch",
        "coverage.xml: local test report artifacts must not be tracked for public launch",
        "coverage-final.json: local test report artifacts must not be tracked for public launch",
        "clover.xml: local test report artifacts must not be tracked for public launch",
        "lcov.info: local test report artifacts must not be tracked for public launch",
        "crates/fathom-core/coverage.lcov: local test report artifacts must not be tracked for public launch",
        "junit.xml: local test report artifacts must not be tracked for public launch",
        "crates/fathom-core/test-output.junit.xml: local test report artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local test report artifacts")
    allowed_notebook_artifact_gitignore = "\n".join(sorted(required_notebook_artifact_gitignore_patterns)) + "\n"
    if gitignore_notebook_artifact_failures(allowed_notebook_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local notebook artifact ignore patterns")
    notebook_artifact_gitignore_failures = gitignore_notebook_artifact_failures(
        allowed_notebook_artifact_gitignore.replace(".nbhistory\n", "")
    )
    if notebook_artifact_gitignore_failures != [
        ".gitignore: missing local notebook artifact ignore patterns: .nbhistory"
    ]:
        raise AssertionError("public risk self-test did not reject missing local notebook artifact ignore patterns")
    notebook_artifact_failures = tracked_notebook_artifact_file_failures(
        tracked_paths=[
            "notebooks/.ipynb_checkpoints/demo-checkpoint.ipynb",
            ".jupyter/jupyter_notebook_config.py",
            ".nbhistory",
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
        ".jupyter/jupyter_notebook_config.py: local notebook runtime/config artifacts must not be tracked for public launch",
        ".nbhistory: local notebook runtime/config artifacts must not be tracked for public launch",
        "notebooks/with-output.ipynb: notebook execution outputs must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local notebook artifacts")
    allowed_doc_build_artifact_gitignore = "\n".join(sorted(required_doc_build_artifact_gitignore_patterns)) + "\n"
    if gitignore_doc_build_artifact_failures(allowed_doc_build_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local documentation build artifact ignore patterns")
    doc_build_artifact_gitignore_failures = gitignore_doc_build_artifact_failures(
        allowed_doc_build_artifact_gitignore.replace("*.fdb_latexmk\n", "")
    )
    if doc_build_artifact_gitignore_failures != [
        ".gitignore: missing local documentation build artifact ignore patterns: *.fdb_latexmk"
    ]:
        raise AssertionError("public risk self-test did not reject missing local documentation build artifact ignore patterns")
    doc_build_artifact_failures = tracked_doc_build_artifact_file_failures(
        tracked_paths=[
            "docs/paper.aux",
            "docs/paper.bbl",
            "docs/paper.blg",
            "docs/paper.fdb_latexmk",
            "docs/paper.fls",
            "docs/paper.synctex.gz",
            "docs/_minted-paper/default.pygstyle",
            "docs/paper.tex",
            "docs/public-launch-checklist.md",
        ],
    )
    if doc_build_artifact_failures != [
        "docs/paper.aux: local documentation build artifacts must not be tracked for public launch",
        "docs/paper.bbl: local documentation build artifacts must not be tracked for public launch",
        "docs/paper.blg: local documentation build artifacts must not be tracked for public launch",
        "docs/paper.fdb_latexmk: local documentation build artifacts must not be tracked for public launch",
        "docs/paper.fls: local documentation build artifacts must not be tracked for public launch",
        "docs/paper.synctex.gz: local documentation build artifacts must not be tracked for public launch",
        "docs/_minted-paper/default.pygstyle: local documentation build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local documentation build artifacts")
    rust_artifact_failures = tracked_rust_artifact_file_failures(
        tracked_paths=[
            ".cargo/registry/cache/index.crate",
            ".cargo/git/checkouts/fathom-local/HEAD",
            ".cargo/credentials.toml",
            ".cargo/config.toml",
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
        ".cargo/registry/cache/index.crate: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        ".cargo/git/checkouts/fathom-local/HEAD: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        ".cargo/credentials.toml: Rust/Cargo cache/build artifacts must not be tracked for public launch",
        ".cargo/config.toml: Rust/Cargo cache/build artifacts must not be tracked for public launch",
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
    native_build_artifact_failures = tracked_native_build_artifact_file_failures(
        tracked_paths=[
            "cmake-build-debug/CMakeCache.txt",
            "CMakeFiles/CMakeOutput.log",
            "native/CMakeFiles/rules.ninja",
            "native/CMakeCache.txt",
            "native/cmake_install.cmake",
            "compile_commands.json",
            "CMakeLists.txt",
            "docs/research/runtime-safety-policy.md",
            "crates/fathom-core/src/lib.rs",
        ],
    )
    if native_build_artifact_failures != [
        "cmake-build-debug/CMakeCache.txt: native/CMake build artifacts must not be tracked for public launch",
        "CMakeFiles/CMakeOutput.log: native/CMake build artifacts must not be tracked for public launch",
        "native/CMakeFiles/rules.ninja: native/CMake build artifacts must not be tracked for public launch",
        "native/CMakeCache.txt: native/CMake build artifacts must not be tracked for public launch",
        "native/cmake_install.cmake: native/CMake build artifacts must not be tracked for public launch",
        "compile_commands.json: native/CMake build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked native/CMake build artifacts")
    allowed_native_build_artifact_gitignore = "\n".join(sorted(required_native_build_artifact_gitignore_patterns)) + "\n"
    if gitignore_native_build_artifact_failures(allowed_native_build_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local native/CMake build artifact ignore patterns")
    native_build_artifact_gitignore_failures = gitignore_native_build_artifact_failures(
        allowed_native_build_artifact_gitignore.replace("compile_commands.json\n", "")
    )
    if native_build_artifact_gitignore_failures != [
        ".gitignore: missing local native/CMake build artifact ignore patterns: compile_commands.json"
    ]:
        raise AssertionError("public risk self-test did not reject missing local native/CMake build artifact ignore patterns")
    package_artifact_failures = tracked_package_artifact_file_failures(
        tracked_paths=[
            "artifacts/public-contract-smoke-summary.json",
            "dist/fathom.zip",
            "release/fathom",
            "releases/fathom-macos.dmg",
            "releases/fathom.pkg",
            "python/fathom-0.1.0-py3-none-any.whl",
            "python/fathom.egg",
            "jvm/fathom-cli.jar",
            "jvm/fathom-webapp.war",
            "jvm/fathom-enterprise.ear",
            "Fathom.app/Contents/Info.plist",
            "snapshots/fathom.tar",
            "snapshots/fathom.tar.gz",
            "snapshots/fathom.tar.bz2",
            "snapshots/fathom.tar.xz",
            "snapshots/fathom.tgz",
            "snapshots/fathom.7z",
            "snapshots/fathom.rar",
            "snapshots/fathom.tar.zst",
            "installers/fathom.deb",
            "installers/fathom.rpm",
            "installers/fathom.msi",
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
        "jvm/fathom-cli.jar: release/package artifacts must not be tracked for public launch",
        "jvm/fathom-webapp.war: release/package artifacts must not be tracked for public launch",
        "jvm/fathom-enterprise.ear: release/package artifacts must not be tracked for public launch",
        "Fathom.app/Contents/Info.plist: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.gz: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.bz2: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.xz: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tgz: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.7z: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.rar: release/package artifacts must not be tracked for public launch",
        "snapshots/fathom.tar.zst: release/package artifacts must not be tracked for public launch",
        "installers/fathom.deb: release/package artifacts must not be tracked for public launch",
        "installers/fathom.rpm: release/package artifacts must not be tracked for public launch",
        "installers/fathom.msi: release/package artifacts must not be tracked for public launch",
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
    allowed_experiment_tracking_artifact_gitignore = "\n".join(
        sorted(required_experiment_tracking_artifact_gitignore_patterns)
    ) + "\n"
    if gitignore_experiment_tracking_artifact_failures(allowed_experiment_tracking_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local ML experiment/tracking artifact ignore patterns")
    experiment_tracking_artifact_gitignore_failures = gitignore_experiment_tracking_artifact_failures(
        allowed_experiment_tracking_artifact_gitignore.replace("/mlruns/\n", "")
    )
    if experiment_tracking_artifact_gitignore_failures != [
        ".gitignore: missing local ML experiment/tracking artifact ignore patterns: /mlruns/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local ML experiment/tracking artifact ignore patterns")
    experiment_tracking_artifact_failures = tracked_experiment_tracking_artifact_file_failures(
        tracked_paths=[
            ".wandb/run-20260602_174300-fathom/files/config.yaml",
            "wandb/latest-run/files/output.log",
            "mlruns/0/meta.yaml",
            "lightning_logs/version_0/events.out.tfevents.123456.local",
            "runs/events.out.tfevents.123456.local",
            "docs/research/performance-strategy.md",
            "docs/api/public-contract.json",
        ],
    )
    if experiment_tracking_artifact_failures != [
        ".wandb/run-20260602_174300-fathom/files/config.yaml: local ML experiment/tracking artifacts must not be tracked for public launch",
        "wandb/latest-run/files/output.log: local ML experiment/tracking artifacts must not be tracked for public launch",
        "mlruns/0/meta.yaml: local ML experiment/tracking artifacts must not be tracked for public launch",
        "lightning_logs/version_0/events.out.tfevents.123456.local: local ML experiment/tracking artifacts must not be tracked for public launch",
        "runs/events.out.tfevents.123456.local: local ML experiment/tracking artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local ML experiment/tracking artifacts")
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
    allowed_deployment_platform_artifact_gitignore = "\n".join(
        sorted(required_deployment_platform_artifact_gitignore_patterns)
    ) + "\n"
    if gitignore_deployment_platform_artifact_failures(allowed_deployment_platform_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local deployment platform artifact ignore patterns")
    deployment_platform_artifact_gitignore_failures = gitignore_deployment_platform_artifact_failures(
        allowed_deployment_platform_artifact_gitignore.replace("/.vercel/\n", "")
    )
    if deployment_platform_artifact_gitignore_failures != [
        ".gitignore: missing local deployment platform artifact ignore patterns: /.vercel/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local deployment platform artifact ignore patterns")
    deployment_platform_artifact_failures = tracked_deployment_platform_artifact_file_failures(
        tracked_paths=[
            ".vercel/project.json",
            ".vercel/output/config.json",
            ".netlify/state.json",
            ".netlify/functions-internal/manifest.json",
            "netlify.toml",
            "vercel.json",
            "docs/api/public-contract.json",
        ],
    )
    if deployment_platform_artifact_failures != [
        ".vercel/project.json: local deployment platform artifacts must not be tracked for public launch",
        ".vercel/output/config.json: local deployment platform artifacts must not be tracked for public launch",
        ".netlify/state.json: local deployment platform artifacts must not be tracked for public launch",
        ".netlify/functions-internal/manifest.json: local deployment platform artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local deployment platform artifacts")
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
    allowed_nix_artifact_gitignore = "\n".join(sorted(required_nix_artifact_gitignore_patterns)) + "\n"
    if gitignore_nix_artifact_failures(allowed_nix_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Nix build result artifact ignore patterns")
    nix_artifact_gitignore_failures = gitignore_nix_artifact_failures(
        allowed_nix_artifact_gitignore.replace("/result-*\n", "")
    )
    if nix_artifact_gitignore_failures != [
        ".gitignore: missing local Nix build result artifact ignore patterns: /result-*"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Nix build result artifact ignore patterns")
    nix_artifact_failures = tracked_nix_artifact_file_failures(
        tracked_paths=[
            "result",
            "result/bin/fathom",
            "result-frontend/index.html",
            "result-docs/share/doc/fathom/index.html",
            "docs/result-analysis.md",
            "frontend/src/result-card.jsx",
        ],
    )
    if nix_artifact_failures != [
        "result: local Nix build result artifacts must not be tracked for public launch",
        "result/bin/fathom: local Nix build result artifacts must not be tracked for public launch",
        "result-frontend/index.html: local Nix build result artifacts must not be tracked for public launch",
        "result-docs/share/doc/fathom/index.html: local Nix build result artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local Nix build result artifacts")
    allowed_bazel_artifact_gitignore = "\n".join(sorted(required_bazel_artifact_gitignore_patterns)) + "\n"
    if gitignore_bazel_artifact_failures(allowed_bazel_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Bazel build artifact ignore patterns")
    bazel_artifact_gitignore_failures = gitignore_bazel_artifact_failures(
        allowed_bazel_artifact_gitignore.replace("/bazel-testlogs/\n", "")
    )
    if bazel_artifact_gitignore_failures != [
        ".gitignore: missing local Bazel build artifact ignore patterns: /bazel-testlogs/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Bazel build artifact ignore patterns")
    bazel_artifact_failures = tracked_bazel_artifact_file_failures(
        tracked_paths=[
            "bazel-bin/fathom",
            "bazel-out/darwin-fastbuild/bin/app",
            "bazel-testlogs/tests/test.log",
            "bazel-fathom/external/repo/file",
            "bazel-overview.md",
            "bazel-tools/BUILD",
            "BUILD.bazel",
            "MODULE.bazel",
            "tools/bazel/BUILD",
        ],
    )
    if bazel_artifact_failures != [
        "bazel-bin/fathom: local Bazel build artifacts must not be tracked for public launch",
        "bazel-out/darwin-fastbuild/bin/app: local Bazel build artifacts must not be tracked for public launch",
        "bazel-testlogs/tests/test.log: local Bazel build artifacts must not be tracked for public launch",
        "bazel-fathom/external/repo/file: local Bazel build artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local Bazel build artifacts")
    allowed_swiftpm_artifact_gitignore = "\n".join(sorted(required_swiftpm_artifact_gitignore_patterns)) + "\n"
    if gitignore_swiftpm_artifact_failures(allowed_swiftpm_artifact_gitignore):
        raise AssertionError("public risk self-test rejected complete local Swift Package Manager artifact ignore patterns")
    swiftpm_artifact_gitignore_failures = gitignore_swiftpm_artifact_failures(
        allowed_swiftpm_artifact_gitignore.replace("/.swiftpm/\n", "")
    )
    if swiftpm_artifact_gitignore_failures != [
        ".gitignore: missing local Swift Package Manager artifact ignore patterns: /.swiftpm/"
    ]:
        raise AssertionError("public risk self-test did not reject missing local Swift Package Manager artifact ignore patterns")
    swiftpm_artifact_failures = tracked_swiftpm_artifact_file_failures(
        tracked_paths=[
            ".build/debug/Fathom",
            ".swiftpm/xcode/package.xcworkspace/contents.xcworkspacedata",
            "Package.swift",
            "Sources/Fathom/main.swift",
        ],
    )
    if swiftpm_artifact_failures != [
        ".build/debug/Fathom: local Swift Package Manager artifacts must not be tracked for public launch",
        ".swiftpm/xcode/package.xcworkspace/contents.xcworkspacedata: local Swift Package Manager artifacts must not be tracked for public launch",
    ]:
        raise AssertionError("public risk self-test did not reject tracked local Swift Package Manager artifacts")
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
    gitmodules_failures = tracked_gitmodules_source_failures(
        tracked_paths=[
            ".gitmodules",
            "docs/api/public-contract.json",
        ],
        texts={
            ".gitmodules": (
                '[submodule "safe"]\n'
                "  path = vendor/safe\n"
                "  url = https://github.com/example/safe.git\n"
                '[submodule "relative"]\n'
                "  path = vendor/private-relative\n"
                "  url = ../private-relative.git\n"
                '[submodule "ssh"]\n'
                "  path = vendor/private-ssh\n"
                "  url = git@github.com:example/private.git\n"
                '[submodule "auth"]\n'
                "  path = vendor/private-auth\n"
                "  url = https://user:token@example.invalid/private.git\n"
                '[submodule "home"]\n'
                "  path = vendor/private-home\n"
                "  url = /Users/example/private.git\n"
            ),
            "docs/api/public-contract.json": '{"source":"../ignored.json"}\n',
        },
    )
    if gitmodules_failures != [
        ".gitmodules:6: local or relative submodule URL in .gitmodules: url = ../private-relative.git",
        ".gitmodules:9: SSH submodule URL in .gitmodules: url = git@github.com:example/private.git",
        ".gitmodules:12: authenticated submodule URL in .gitmodules: url = https://user:token@example.invalid/private.git",
        ".gitmodules:15: personal home path in Git submodule metadata: url = /Users/example/private.git",
        ".gitmodules:15: local or relative submodule URL in .gitmodules: url = /Users/example/private.git",
    ]:
        raise AssertionError("public risk self-test did not reject local/private Git submodule metadata sources")
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
            "yarn.lock": (
                '"safe@npm:^1.0.0":\n'
                '  resolution: "safe@npm:1.0.0"\n'
                '"portal-helper@portal:../private-helper":\n'
                '  resolution: "portal-helper@portal:../private-helper"\n'
                '"patched@patch:patched@npm%3A1.0.0#./.yarn/patches/patched.patch":\n'
                '  resolution: "patched@patch:patched@npm%3A1.0.0#./.yarn/patches/patched.patch"\n'
            ),
            "docs/api/public-contract.json": '{"source":"file:../ignored.json"}\n',
        },
    )
    if lockfile_failures != [
        'Cargo.lock:6: local file dependency source in lockfile: path = "../private-helper"',
        'Cargo.lock:9: SSH dependency source in lockfile: source = "git+ssh://git@github.com/example/private#abc123"',
        'package-lock.json:1: local file dependency source in lockfile: {"packages":{"node_modules/local":{"resolved":"file:../local.tgz"},"node_modules/private":{"resolved":"https://user:token@example.invalid/private.tgz"}}}',
        'package-lock.json:1: authenticated dependency URL in lockfile: {"packages":{"node_modules/local":{"resolved":"file:../local.tgz"},"node_modules/private":{"resolved":"https://user:token@example.invalid/private.tgz"}}}',
        "pnpm-lock.yaml:3: personal home path in dependency lockfile: resolution: {tarball: /Users/example/private.tgz}",
        'yarn.lock:3: local file dependency source in lockfile: "portal-helper@portal:../private-helper":',
        'yarn.lock:4: local file dependency source in lockfile: resolution: "portal-helper@portal:../private-helper"',
        'yarn.lock:5: local file dependency source in lockfile: "patched@patch:patched@npm%3A1.0.0#./.yarn/patches/patched.patch":',
        'yarn.lock:6: local file dependency source in lockfile: resolution: "patched@patch:patched@npm%3A1.0.0#./.yarn/patches/patched.patch"',
    ]:
        raise AssertionError("public risk self-test did not reject local/private dependency lockfile sources")
    print("public risk scan self-test passed")

if "--self-test" in sys.argv[1:]:
    self_test()
    raise SystemExit(0)

failures = scan_items(tracked_items())
failures.extend(tracked_dependency_lock_source_failures())
failures.extend(tracked_gitmodules_source_failures())
failures.extend(tracked_large_file_failures())
failures.extend(tracked_git_lfs_pointer_failures())
failures.extend(tracked_blocked_file_failures())
failures.extend(tracked_credential_file_failures())
failures.extend(tracked_cloud_credential_file_failures())
failures.extend(tracked_kubernetes_credential_file_failures())
failures.extend(tracked_workspace_context_failures())
failures.extend(tracked_command_history_file_failures())
failures.extend(tracked_local_ci_artifact_file_failures())
failures.extend(gitignore_workspace_context_failures())
failures.extend(gitignore_command_history_failures())
failures.extend(gitignore_local_ci_artifact_failures())
failures.extend(gitignore_credential_failures())
failures.extend(gitignore_cloud_credential_failures())
failures.extend(gitignore_kubernetes_credential_failures())
failures.extend(gitignore_os_metadata_failures())
failures.extend(gitignore_editor_artifact_failures())
failures.extend(gitignore_ide_artifact_failures())
failures.extend(gitignore_model_artifact_failures())
failures.extend(gitignore_experiment_tracking_artifact_failures())
failures.extend(gitignore_container_artifact_failures())
failures.extend(gitignore_deployment_platform_artifact_failures())
failures.extend(gitignore_infra_state_failures())
failures.extend(gitignore_nix_artifact_failures())
failures.extend(gitignore_bazel_artifact_failures())
failures.extend(gitignore_swiftpm_artifact_failures())
failures.extend(gitignore_mobile_build_failures())
failures.extend(gitignore_screen_capture_failures())
failures.extend(gitignore_media_capture_failures())
failures.extend(gitignore_rust_artifact_failures())
failures.extend(gitignore_native_build_artifact_failures())
failures.extend(gitignore_package_artifact_failures())
failures.extend(gitignore_backup_artifact_failures())
failures.extend(gitignore_diagnostic_artifact_failures())
failures.extend(gitignore_python_artifact_failures())
failures.extend(gitignore_python_env_artifact_failures())
failures.extend(gitignore_ruby_bundle_artifact_failures())
failures.extend(gitignore_php_composer_artifact_failures())
failures.extend(gitignore_r_artifact_failures())
failures.extend(gitignore_go_artifact_failures())
failures.extend(gitignore_jvm_dependency_artifact_failures())
failures.extend(gitignore_gradle_artifact_failures())
failures.extend(gitignore_frontend_artifact_failures())
failures.extend(gitignore_local_cache_artifact_failures())
failures.extend(gitignore_temp_artifact_failures())
failures.extend(gitignore_test_report_artifact_failures())
failures.extend(gitignore_notebook_artifact_failures())
failures.extend(gitignore_doc_build_artifact_failures())
failures.extend(gitignore_runtime_artifact_failures())
failures.extend(tracked_runtime_artifact_file_failures())
failures.extend(tracked_diagnostic_artifact_file_failures())
failures.extend(tracked_python_artifact_file_failures())
failures.extend(tracked_python_env_artifact_file_failures())
failures.extend(tracked_ruby_bundle_artifact_file_failures())
failures.extend(tracked_php_composer_artifact_file_failures())
failures.extend(tracked_r_artifact_file_failures())
failures.extend(tracked_go_artifact_file_failures())
failures.extend(tracked_jvm_dependency_artifact_file_failures())
failures.extend(tracked_gradle_artifact_file_failures())
failures.extend(tracked_frontend_artifact_file_failures())
failures.extend(tracked_local_cache_artifact_file_failures())
failures.extend(tracked_temp_artifact_file_failures())
failures.extend(tracked_rust_artifact_file_failures())
failures.extend(tracked_native_build_artifact_file_failures())
failures.extend(tracked_package_artifact_file_failures())
failures.extend(tracked_backup_artifact_file_failures())
failures.extend(tracked_model_artifact_file_failures())
failures.extend(tracked_experiment_tracking_artifact_file_failures())
failures.extend(tracked_container_artifact_file_failures())
failures.extend(tracked_deployment_platform_artifact_file_failures())
failures.extend(tracked_infra_state_file_failures())
failures.extend(tracked_nix_artifact_file_failures())
failures.extend(tracked_bazel_artifact_file_failures())
failures.extend(tracked_swiftpm_artifact_file_failures())
failures.extend(tracked_mobile_build_file_failures())
failures.extend(tracked_screen_capture_file_failures())
failures.extend(tracked_media_capture_file_failures())
failures.extend(tracked_test_report_artifact_file_failures())
failures.extend(tracked_notebook_artifact_file_failures())
failures.extend(tracked_doc_build_artifact_file_failures())
failures.extend(tracked_symlink_failures())
if failures:
    print("Public risk scan failed:", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print("public risk scan passed")
PY
