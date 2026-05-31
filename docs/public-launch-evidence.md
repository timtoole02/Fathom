# Public launch evidence snapshot

This snapshot records the current public launch verification state for Fathom. It is evidence for the narrow local public contract only; it is not production-readiness, performance, quality, legal/license, full OpenAI parity, streaming, external proxying, or broad model-runtime evidence.

## Snapshot

- Launch baseline commit: `a32505eadac6539865d224a8b4195656003a0032` (`Close out public launch readiness phase`)
- Latest no-download refusal evidence commit: `1b930d4ed2411fe72e66d45acc757318b649d730` (`Prove embeddings malformed JSON refusal`)
- Latest optional API evidence commit: `0655fc073af296f7687a04708053a30b9570fcdf` (`Record MiniLM embeddings API acceptance evidence`)
- Latest optional artifact QA CI wiring commit: `e9195bc7462999284960f5631d3a74aa5391bffc` (`Wire optional artifact QA into CI`)
- Latest public-contract QA hardening commit: `1fd97d676ced1c1da4efa91abfc2a9d9b1130e5d` (`Guard OpenAI SDK example regression`)
- Scope: no-download public `/v1` contract, public launch checklist/evidence snapshot, manifest-driven smoke, sanitized public-contract smoke artifacts, public-contract smoke duplicate-entry schema guards, public-contract smoke Markdown/status/proof-scope row consistency, failed public-contract smoke endpoint/boundary drift guards, malformed `/v1` JSON request body refusal smoke, unsupported `/v1` route/method public-contract refusal smoke, synthetic PyTorch `.bin` public-contract refusal smoke, unsupported ONNX chat/general public-contract refusal smoke, unverified SafeTensors/Hugging Face public-contract refusal smoke, metadata-only GGUF public-contract refusal smoke, offline public-contract and backend acceptance artifact QA, manifest shape validation, manifest-to-`/v1` docs boundary coverage, public refusal boundary status/code/request-hint coverage, local API no-auth/loopback security-note coverage, tracked Python syntax coverage guard, public security/privacy reporting copy guard, OpenAI SDK example offline request/response regression, API contract issue-template privacy checks, blank public issue disablement checks, bug/security issue-template privacy checks, PR template truthfulness/privacy checks, CI token-permissions guard, public overclaim scanner self-test coverage, production-readiness/legal-license public overclaim examples, offline MiniLM/SmolLM2/Qwen2.5 optional API acceptance artifact QA self-tests, CI/static policy wiring, and Phase 16 narrow catalog-backed optional API/runtime evidence for MiniLM embeddings, SmolLM2 135M Instruct, and Qwen2.5 0.5B Instruct.
- Fresh-clone QA: passed for the launch baseline and earlier evidence commits. For `128550a`, focused fresh-clone QA passed after reducing cold-build pressure with `CARGO_BUILD_JOBS=1` and shared local target artifacts. Later optional API evidence, optional artifact-QA CI wiring, no-download refusal evidence, and public-contract QA hardening commits are covered by focused static/public-risk/artifact QA gates and preserved opt-in local artifact QA, not a new full fresh-clone run.

## Gates represented by the snapshot

The referenced fresh-clone QA gates verified:

- exact `origin/main` checkout and clean initial/final status;
- no repo-local `AGENTS.md`;
- no tracked files larger than 1 MiB;
- public risk scan and focused sensitive/overclaim review;
- JSON, Python, shell, static QA, API-example regression, and artifact-QA gates;
- `cargo fmt --all --check` and `cargo test -q`;
- manifest-driven public contract smoke in default and artifact modes;
- public-contract smoke artifact summaries accepted by `scripts/public_contract_smoke_artifact_qa.py`;
- manifest shape validation and manifest-to-`/v1` docs boundary coverage in `scripts/public_api_contract_qa.py`;
- MiniLM, SmolLM2, and Qwen2.5 optional API acceptance artifact schemas covered by offline self-tests in default CI;
- frontend install/build/copy QA.

## What this evidence proves

- The documented no-download public `/v1` routing/refusal contract works against the real backend process in isolated empty state.
- Public-contract smoke summary artifacts have an offline QA gate for schema, pass/fail semantics, coverage, caveats, and share-safety.
- Public-contract smoke summary artifacts reject duplicate endpoint, boundary, and deferred-boundary entries so summary coverage cannot be inflated by repeated rows.
- Public-contract smoke summary artifacts keep Markdown endpoint/boundary/deferred/status/proof-scope rows aligned with JSON evidence rows.
- The offline public API contract QA gate validates manifest shape, keeps manifest boundary names covered by the `/v1` docs and refusal matrix, and requires request hints for status/code refusal boundaries to be exposed in the refusal matrix.
- The offline public API contract QA gate keeps the local API no-auth, loopback-only, internet/untrusted-LAN exposure, and `SECURITY.md` review warnings attached to the README, backend quickstart, `/v1` contract, and public launch checklist.
- The offline public API contract QA gate keeps every tracked Python helper covered by the documented and CI `python3 -m py_compile` syntax gate, including local oracle helpers that are not run during no-download launch QA.
- The offline API client example regression gate runs the OpenAI SDK example through a dependency-free local `openai` stub and checks both request routing/body shape and the JSON response payloads printed by the example, including opt-in float embeddings.
- The public API contract QA self-test exercises synthetic refused/unsupported public overclaim examples for streaming, external proxying, OpenAI parity, SafeTensors blanket execution, base64 embeddings, production readiness, and legal/license suitability, plus allowed caveated examples.
- Backend acceptance artifact QA and optional MiniLM, SmolLM2, and Qwen2.5 API acceptance artifact QA self-tests run offline in default CI as schema/caveat checks only; they do not download models, start the backend, or add runtime proof.
- External OpenAI-compatible entries remain metadata placeholders; activation and `/v1/chat/completions` refuse with `external_proxy_not_implemented` and no provider call.
- PyTorch `.bin` artifacts remain blocked: the no-download public contract smoke registers a tiny synthetic local `.bin` artifact, confirms it is excluded from `/v1/models`, and confirms chat refuses with `501 not_implemented` without deserializing pickle bytes or faking inference.
- Unsupported ONNX chat/general model execution remains refused: the no-download public contract smoke registers a tiny synthetic local `.onnx` artifact, confirms it is excluded from `/v1/models`, and confirms chat refuses with `501 not_implemented` without enabling ONNX Runtime, loading the graph, or faking inference. The feature-gated MiniLM ONNX embedding path remains separate.
- Unverified SafeTensors/Hugging Face packages remain refused: the no-download public contract smoke registers a tiny synthetic local HF-style `.safetensors` package, confirms it is excluded from `/v1/models`, and confirms chat refuses with `501 not_implemented` without loading weights or faking inference. This does not broaden the verified runnable SafeTensors/HF lanes.
- GGUF metadata-only packages remain refused: the no-download public contract smoke registers a tiny synthetic local `.gguf` file, confirms it is metadata-only and excluded from `/v1/models`, and confirms chat refuses with `501 not_implemented` without making a GGUF runtime, tokenizer execution, or generation claim, runtime weight loading claim, dequantization/kernels claim, or faking inference.
- Unsupported `/v1` routes and methods remain predictable refusals: the no-download public contract smoke checks `POST /v1/responses` and `GET /v1/chat/completions` return standard JSON error envelopes for unsupported `/v1` routes and methods, without implying OpenAI Responses API support or broader method support.
- Malformed `/v1` JSON request bodies remain predictable refusals: the no-download public contract smoke checks malformed JSON on `POST /v1/chat/completions` and `POST /v1/embeddings` return `400 invalid_request` in the standard JSON error envelope, without emitting fake chat `choices` or embedding `data`.
- Default CI remains scoped to local/offline gates and does not run model downloads, networked acceptance smoke, or non-default ONNX feature tests.
- The optional Qwen2.5 0.5B Instruct catalog demo is pinned to a specific revision with exact file sizes, SHA256 hashes, Apache-2.0 metadata, local runtime-smoke evidence, and opt-in API acceptance evidence captured outside default CI; see `docs/benchmarks/2026-04-29-qwen25-local-runtime-smoke.md` and `docs/benchmarks/2026-04-29-qwen25-api-acceptance.md`.
- The optional SmolLM2 135M Instruct catalog demo has opt-in API acceptance evidence for catalog install, `/v1/models`, cold/warm non-streaming chat metrics, and streaming refusal; see `docs/benchmarks/2026-04-29-smollm2-api-acceptance.md`.
- The optional MiniLM SafeTensors embedding demo has opt-in API acceptance evidence for catalog install, `/api/embedding-models`, `/v1/models` exclusion, float `/v1/embeddings`, `encoding_format: "base64"` refused with `invalid_request`, and chat refusal; see `docs/benchmarks/2026-04-29-minilm-embeddings-api-acceptance.md`.

## What this evidence does not prove

- It does not prove broad model download success, generation quality, embedding quality, latency, throughput, production readiness, legal/license suitability, or security hardening for exposed deployments.
- It does not prove full OpenAI API parity, streaming chat, real external provider proxying, GGUF runtime/tokenizer execution/generation, ONNX chat/general execution, PyTorch `.bin` loading, or arbitrary SafeTensors/Hugging Face execution.
- Optional networked backend acceptance smoke and large-model runtime artifacts remain separate and should be reviewed before sharing artifacts publicly.
