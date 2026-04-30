# Public launch evidence snapshot

This snapshot records the current public launch verification state for Fathom. It is evidence for the narrow local public contract only; it is not production-readiness, performance, quality, legal/license, full OpenAI parity, streaming, external proxying, or broad model-runtime evidence.

## Snapshot

- Launch baseline commit: `a32505eadac6539865d224a8b4195656003a0032` (`Close out public launch readiness phase`)
- Current evidence commit: `0655fc073af296f7687a04708053a30b9570fcdf` (`Record MiniLM embeddings API acceptance evidence`)
- Scope: no-download public `/v1` contract, public launch checklist/evidence snapshot, manifest-driven smoke, sanitized public-contract smoke artifacts, offline artifact QA, CI/static policy wiring, and Phase 16 narrow catalog-backed optional API/runtime evidence for MiniLM embeddings, SmolLM2 135M Instruct, and Qwen2.5 0.5B Instruct.
- Fresh-clone QA: passed for the launch baseline and earlier evidence commits. For `128550a`, focused fresh-clone QA passed after reducing cold-build pressure with `CARGO_BUILD_JOBS=1` and shared local target artifacts. Later optional API evidence commits are covered by focused static/public-risk/artifact QA gates and preserved opt-in local artifact QA, not a new full fresh-clone run.

## Gates represented by the snapshot

The fresh-clone QA for this snapshot verified:

- exact `origin/main` checkout and clean initial/final status;
- no repo-local `AGENTS.md`;
- no tracked files larger than 1 MiB;
- public risk scan and focused sensitive/overclaim review;
- JSON, Python, shell, static QA, API-example regression, and artifact-QA gates;
- `cargo fmt --all --check` and `cargo test -q`;
- manifest-driven public contract smoke in default and artifact modes;
- public-contract smoke artifact summaries accepted by `scripts/public_contract_smoke_artifact_qa.py`;
- frontend install/build/copy QA.

## What this evidence proves

- The documented no-download public `/v1` routing/refusal contract works against the real backend process in isolated empty state.
- Public-contract smoke summary artifacts have an offline QA gate for schema, pass/fail semantics, coverage, caveats, and share-safety.
- External OpenAI-compatible entries remain metadata placeholders; activation and `/v1/chat/completions` refuse with `external_proxy_not_implemented` and no provider call.
- Default CI remains scoped to local/offline gates and does not run model downloads, networked acceptance smoke, or non-default ONNX feature tests.
- The optional Qwen2.5 0.5B Instruct catalog demo is pinned to a specific revision with exact file sizes, SHA256 hashes, Apache-2.0 metadata, local runtime-smoke evidence, and opt-in API acceptance evidence captured outside default CI; see `docs/benchmarks/2026-04-29-qwen25-local-runtime-smoke.md` and `docs/benchmarks/2026-04-29-qwen25-api-acceptance.md`.
- The optional SmolLM2 135M Instruct catalog demo has opt-in API acceptance evidence for catalog install, `/v1/models`, cold/warm non-streaming chat metrics, and streaming refusal; see `docs/benchmarks/2026-04-29-smollm2-api-acceptance.md`.
- The optional MiniLM SafeTensors embedding demo has opt-in API acceptance evidence for catalog install, `/api/embedding-models`, `/v1/models` exclusion, float `/v1/embeddings`, `encoding_format: "base64"` refused with `invalid_request`, and chat refusal; see `docs/benchmarks/2026-04-29-minilm-embeddings-api-acceptance.md`.

## What this evidence does not prove

- It does not prove broad model download success, generation quality, embedding quality, latency, throughput, production readiness, legal/license suitability, or security hardening for exposed deployments.
- It does not prove full OpenAI API parity, streaming chat, real external provider proxying, GGUF runtime/tokenizer execution/generation, ONNX chat/general execution, PyTorch `.bin` loading, or arbitrary SafeTensors/Hugging Face execution.
- Optional networked backend acceptance smoke and large-model runtime artifacts remain separate and should be reviewed before sharing artifacts publicly.
