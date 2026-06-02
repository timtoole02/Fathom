# Public launch checklist

Use this checklist when validating a clean public Fathom checkout or preparing a launch-facing handoff. It is scoped to the current local, no-download public contract plus optional networked backend acceptance evidence.

## 1. Clean clone and install

```bash
git clone https://github.com/timtoole02/Fathom/ fathom
cd fathom
npm --prefix frontend ci
cargo test -q
```

This confirms the ordinary Rust and frontend toolchains are present, with frontend dependencies resolved from the checked-in lockfile. It does not download catalog model fixtures or enable the non-default ONNX Runtime feature.

## 2. No-download verification gates

Run these before treating docs/API contract changes as launch-ready:

```bash
git diff --check
python3 -m py_compile \
  examples/api/openai-sdk.py \
  examples/api/python-no-deps.py \
  scripts/api_client_examples_regression.py \
  scripts/backend_acceptance_artifact_qa.py \
  scripts/bench_backend.py \
  scripts/ci_static_policy.py \
  scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py \
  scripts/public_api_contract_qa.py \
  scripts/public_contract_smoke_artifact_qa.py \
  scripts/qwen25_optional_api_acceptance_artifact_qa.py \
  scripts/reference_llama3_tokenizer_ids.py \
  scripts/smollm2_optional_api_acceptance_artifact_qa.py
python3 scripts/ci_static_policy.py
python3 scripts/ci_static_policy.py --self-test
python3 scripts/api_client_examples_regression.py
python3 scripts/api_client_examples_regression.py --self-test
python3 scripts/public_api_contract_qa.py
python3 scripts/public_api_contract_qa.py --self-test
python3 scripts/public_contract_smoke_artifact_qa.py
python3 scripts/backend_acceptance_artifact_qa.py
python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py
python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py
python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py
bash -n examples/api/curl-quickstart.sh
bash -n scripts/public_risk_scan.sh
bash -n scripts/public_api_contract_smoke.sh
bash -n scripts/backend_acceptance_smoke.sh
bash -n scripts/minilm_embeddings_optional_api_acceptance_smoke.sh
bash -n scripts/smollm2_optional_api_acceptance_smoke.sh
bash -n scripts/qwen25_optional_api_acceptance_smoke.sh
bash -n scripts/smoke.sh
bash -n scripts/start-backend.sh
bash -n scripts/start.sh
bash -n scripts/stop.sh
bash scripts/public_risk_scan.sh --self-test
bash scripts/public_risk_scan.sh
npm --prefix frontend run build
npm --prefix frontend run qa:copy
```

For the real backend no-download contract check, run:

```bash
bash scripts/public_api_contract_smoke.sh
```

That smoke starts `fathom-server` with isolated temporary state/model/log directories and checks the public `/v1` routing/refusal boundary from [`docs/api/public-contract.json`](api/public-contract.json) and [`docs/api/v1-contract.md`](api/v1-contract.md), including JSON refusals for unsupported `/v1` routes and methods. It does not install catalog models, download fixtures, enable ONNX features, call providers, or prove model quality.

To keep a share-safe pass/fail summary for release handoff, set `FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR` to a directory you control:

```bash
FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR=public-contract-artifacts \
  bash scripts/public_api_contract_smoke.sh
```

The optional `public-contract-smoke-summary.json` and `.md` files include commit, manifest path/name/status, endpoint checks, boundary checks, and scope caveats only. They intentionally omit local temp paths, server log tails, request secrets, and model/provider payloads. Validate them offline before sharing with `python3 scripts/public_contract_smoke_artifact_qa.py public-contract-artifacts`.

The static QA also requires root `.gitattributes` text-normalization metadata so public diffs stay stable across platforms while binary artifact extensions remain marked as binary.

## 3. Backend/API quick smoke

For a backend-only manual pass:

```bash
bash scripts/start-backend.sh
BASE=http://127.0.0.1:8180
curl -fsS "$BASE/v1/health" | python3 -m json.tool
curl -fsS "$BASE/v1/models" | python3 -m json.tool
bash scripts/stop.sh
```

Use [`docs/api/backend-only-quickstart.md`](api/backend-only-quickstart.md) for catalog install examples, TinyStories chat, MiniLM embeddings, retrieval, and expected refusals. Those catalog demos require network access for pinned fixture downloads. The backend API has no built-in authentication and is intended for loopback development; do not expose it directly to the internet or an untrusted LAN without your own access controls and a [`SECURITY.md`](../SECURITY.md) review.
For a compact refusal/non-claim checklist before Phase 16 work, see [`docs/api/refusal-boundary-matrix.md`](api/refusal-boundary-matrix.md).

## 4. Optional networked acceptance smoke

When you need fuller backend evidence and network access is available:

```bash
FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh
```

Review `summary.md` first, then `summary.json` and the named JSON artifacts. Keep `summary.local.json` private unless you have reviewed it; local paths and runner-specific details belong there. Before sharing artifacts publicly, run `bash scripts/public_risk_scan.sh` and manually inspect logs/full payloads for local paths or request text. The scanner checks tracked-file privacy patterns including macOS, Linux, and Windows home/profile paths, secret-token/private-key/cloud API-key patterns, dependency lockfiles for local file/path, SSH-only, authenticated URL, secret, and private local source entries, public overclaim patterns, oversized tracked files, tracked Git LFS pointer files that can hide external artifact downloads, tracked OS/platform metadata files such as `.DS_Store`, `Thumbs.db`, `desktop.ini`, `__MACOSX/`, `._*`, `.AppleDouble`, `.fseventsd/`, `.Spotlight-V100/`, `.TemporaryItems/`, `.Trashes/`, `.LSOverride`, and `.localized`, root `.gitignore` coverage for local OS/platform metadata files, tracked editor backup/swap files and patch/diff files, root `.gitignore` coverage for local editor backup/swap files and patch/diff files, tracked IDE workspace/config artifacts such as `.zed/`, root `.gitignore` coverage for local IDE workspace/config artifacts, tracked credential/config filenames including SSH private-key filenames, `.ssh/` directories, direnv config/state, generic secret material paths, and Java/Android/Apple signing key material, root `.gitignore` coverage for local credential/config files including SSH private-key filenames, `.ssh/`, direnv config/state, generic secret material patterns, and Java/Android/Apple signing key material patterns, tracked local cloud SDK credential/config files, root `.gitignore` coverage for local cloud SDK credential/config files, tracked Kubernetes credential/config files, root `.gitignore` coverage for local Kubernetes credential/config files, tracked workspace/personal agent context files including local assistant state such as `.codex/`, `.claude/`, `.continue/`, `.cursor/`, `.windsurf/`, and `.aider.*`, root `.gitignore` coverage for personal workspace context including local assistant state, tracked local shell/REPL command history files, root `.gitignore` coverage for local shell/REPL command history files, tracked local runtime/artifact detail files including PID/process marker files, DuckDB databases/WAL files, and Redis RDB snapshots, root `.gitignore` coverage for local runtime/artifact detail files, tracked local log/trace/profiling/debug-output artifacts including crash dumps, root `.gitignore` coverage for local log/trace/profiling/debug-output artifacts including crash dumps, tracked Python cache/build artifacts including Hypothesis example databases, root `.gitignore` coverage for local Python cache/build artifacts, tracked Python virtualenv/dependency artifacts, root `.gitignore` coverage for local Python virtualenv/dependency artifacts, tracked frontend/Node cache/build artifacts including root-level `dist/` or `build/` output, package-manager caches/stores such as `.npm/`, `.pnpm-store/`, and selected `.yarn/` cache/state paths, root `.gitignore` coverage for local frontend/Node cache/build artifacts, tracked local cache artifacts such as `.cache/`, root `.gitignore` coverage for local cache artifacts, tracked local test report/coverage artifacts including Playwright/browser-test report directories, Python coverage data files, and LCOV outputs, root `.gitignore` coverage for local test report/coverage artifacts including Playwright/browser-test report directories, Python coverage data files, and LCOV outputs, tracked local notebook checkpoint artifacts, tracked notebook execution outputs, root `.gitignore` coverage for local notebook artifacts, tracked Rust/Cargo cache/build artifacts, root `.gitignore` coverage for local Rust/Cargo cache/build artifacts, tracked release/package artifacts including common archive and installer formats, root `.gitignore` coverage for local release/package artifacts, tracked backup/dump artifacts, root `.gitignore` coverage for local backup/dump artifacts, tracked local model/checkpoint artifacts, root `.gitignore` coverage for local model/checkpoint artifacts, tracked local Docker/container artifacts, root `.gitignore` coverage for local Docker/container artifacts, tracked local deployment platform artifacts such as `.vercel/` and `.netlify/`, root `.gitignore` coverage for local deployment platform artifacts, tracked local Terraform/OpenTofu state artifacts, root `.gitignore` coverage for local infrastructure state artifacts, tracked local mobile/Xcode/Android build artifacts, root `.gitignore` coverage for local mobile/Xcode/Android build artifacts, tracked local mobile/Xcode/Android signing/provisioning artifacts, root `.gitignore` coverage for local mobile/Xcode/Android signing/provisioning artifacts, tracked local screenshot/screen-recording artifacts, root `.gitignore` coverage for local screenshot/screen-recording artifacts, tracked local audio/video capture/export artifacts, root `.gitignore` coverage for local audio/video capture/export artifacts, and tracked symlinks that escape the repository or resolve only to missing/untracked local targets; it is not a complete privacy audit.

The backend acceptance artifact summaries reject local paths, secret markers, and request/payload text in shareable `summary.json` and `summary.md`; keep full JSON artifacts and logs on the manual review path before publishing.

The browser-test artifact guard also blocks Cypress screenshot, video, and download output directories in addition to Playwright report directories, with matching root `.gitignore` coverage.

## What this launch currently proves

- The documented no-download public `/v1` contract routes and refusal envelopes work against the real backend process.
- Default CI stays offline with respect to model downloads, networked acceptance smoke, and non-default ONNX feature tests.
- Offline artifact QA covers optional public-contract smoke summaries, backend acceptance smoke success/failure summaries with share-safety checks, and MiniLM, SmolLM2, and Qwen2.5 optional API acceptance artifact schemas; the optional backend acceptance smoke itself remains networked and only produces current pinned-fixture evidence when downloads succeed.
- The current launch evidence snapshot is recorded in [`public-launch-evidence.md`](public-launch-evidence.md).

## What this launch does not prove

- It is not production readiness, performance capacity, model quality, or legal/license advice.
- It is not full OpenAI API parity; streaming chat and many OpenAI endpoints are unsupported.
- It does not prove external provider proxying; saved external entries are metadata placeholders and chat is refused.
- It does not prove general GGUF runtime, tokenizer execution, dequantization, generation, ONNX chat, PyTorch `.bin` loading, or arbitrary SafeTensors/Hugging Face execution.

## Troubleshooting

- **Missing tools:** install Rust (`cargo`), Node.js/npm, `curl`, and `python3`; Vite requires Node `20.19+` or `22.12+`.
- **Port conflicts:** stop old runs with `bash scripts/stop.sh`, or set `FATHOM_PORT` for backend-only runs. The contract smoke chooses a temporary local port automatically.
- **First build is slow:** `scripts/start.sh` and `scripts/start-backend.sh` build the Rust backend in release mode; the first run can take a few minutes.
- **Model download/network errors:** no-download gates should still pass without network. Catalog demos and `scripts/backend_acceptance_smoke.sh` need network access to fetch pinned fixtures.
- **Logs:** normal runs write backend logs under `~/.fathom/logs`; isolated smoke scripts print their temporary artifact/log locations while running and clean them unless configured to keep artifacts.
