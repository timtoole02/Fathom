# Fathom Roadmap

_Last updated: 2026-04-27_

Fathom's current wedge: **Drop in a model. Fathom tells you what it is, what can run it, what cannot, why, and gives you one clean API to use it.**

This roadmap tracks the phases already completed, the next phase, and the completion criteria for each phase. It is intentionally conservative: checked phases are things that have been implemented and verified, not promises of broad model support.

## Legend

- `[x]` completed and verified
- `[~]` partially complete / ongoing hardening
- `[ ]` planned
- `➡️` next recommended phase

## Phase roadmap

### [x] Phase 0 — ForgeLocal public sandbox foundation

**What was worked on**

- Hardened install/start/stop UX across macOS/Linux/Windows scripts.
- Improved model search, download cancel behavior, disk-space guardrails, and catalog size visibility.
- Established the product discipline Fathom inherited: truthful state, runnable app, clear install path, and no fake readiness.

**Completion criteria**

- Public repo install/start/stop flow is usable by outside testers.
- App can be started, stopped, and smoke-tested from scripts.
- UI does not fake model/download/runtime state.

**Status:** complete as ForgeLocal foundation work; future ForgeLocal release installers and Windows CUDA validation remain separate later work.

---

### [x] Phase 1 — Fathom scaffold and backend-first product shape

**What was worked on**

- Created the Fathom repo structure with `crates/fathom-core`, `crates/fathom-server`, scripts, docs, and frontend.
- Established backend-first architecture: `fathom-core` and `fathom-server` are useful without the UI.
- Set default local ports: backend `8180`, frontend `4185`.

**Completion criteria**

- Backend starts independently.
- Frontend can talk to backend.
- Repo has clear scripts/docs for local development.

**Status:** complete.

---

### [x] Phase 2 — Clean public repo posture

**What was worked on**

- Reset to clean public Git history and pushed normal public commits from the clean repo.
- Preserved old local history only as local backup, not public history.
- Added public risk scanning and intake/contributing/security guidance.
- Added issue/PR templates that avoid asking for secrets, local paths, model stores, or exploit details.

**Completion criteria**

- `origin/main` reflects clean public history.
- Public risk scan passes.
- Repo contains no repo-local `AGENTS.md`.
- Templates and docs are safe for public GitHub use.

**Status:** complete; every launch-facing change should continue to run public-risk checks.

---

### [x] Phase 3 — Model registry, catalog, and transactional state

**What was worked on**

- Made model-state mutations transactional and persistence-backed.
- Added atomic/recoverable `models.json` behavior and corrupt-state recovery warnings.
- Hardened catalog install lifecycle: stage, verify, inspect, promote, then register.
- Added download cancel and failure-state truthfulness.

**Completion criteria**

- Failed persistence does not publish fake in-memory model state.
- Corrupt state is preserved/recovered safely.
- Failed installs do not destroy previous valid installs or leave misleading readiness.

**Status:** complete for current public scope.

---

### [x] Phase 4 — Narrow real local chat runtimes

**What was worked on**

- Implemented verified local SafeTensors/Hugging Face chat lanes for narrow supported architectures.
- Kept unsupported architectures detected/planned/blocked instead of pretending to run them.
- Added chat-template support only for tested patterns.
- Added process-local runtime/cache metrics without claiming batching/GPU/performance guarantees.

**Completion criteria**

- `/v1/chat/completions` produces real local tokens only for verified runnable chat models.
- Unsupported model families return structured refusal instead of fake output.
- `/v1/models` lists only chat/generation-runnable local models.

**Status:** complete for current narrow families; future runtime expansion must pass the same truthfulness bar.

---

### [x] Phase 5 — Metadata-only GGUF boundary

**What was worked on**

- Added GGUF metadata/provenance inspection and bounded internal readiness scaffolding.
- Added pinned real GGUF fixture evidence for metadata/parity testing only.
- Kept GGUF out of `/v1/models` and refused chat/generation.
- Documented blockers: tokenizer execution, runtime weight loading, dequantization, kernels, architecture runtime, KV cache, sampling, generation.

**Completion criteria**

- GGUF can be inspected safely as metadata/provenance.
- No public/runtime GGUF inference, tokenizer execution, or generation is claimed.
- Chat attempts are refused truthfully.

**Status:** complete for metadata-only public scope; GGUF runtime remains deferred.

---

### [x] Phase 6 — Embeddings and retrieval boundary

**What was worked on**

- Added narrow `/v1/embeddings` float-only adapter for verified local embedding runtimes.
- Kept embedding-only models out of `/v1/models` and chat selection.
- Added MiniLM SafeTensors embedding path and feature-gated pinned ONNX MiniLM embedding path.
- Added ONNX embedding safety preflight: pinned filename/size, path containment, external-data/custom-op/plugin refusal.

**Completion criteria**

- `/v1/embeddings` returns real float vectors only for verified embedding runtimes.
- `encoding_format: "base64"` is refused.
- Embedding-only models are not chat models.
- ONNX remains embedding-only, feature-gated, and non-default.

**Status:** complete for current narrow embedding scope; non-embedding ONNX execution remains deferred.

---

### [x] Phase 7 — Runtime and artifact safety hardening

**What was worked on**

- Blocked PyTorch `.bin` trusted-import/execution paths.
- Added package path-containment checks so runnable artifacts must canonicalize under the registered package root.
- Added structured not-implemented/blocked refusals for unsafe or unsupported paths.

**Completion criteria**

- No `torch.load`, `pickle.load`, or trusted `.bin` import path exists.
- Runnable local artifacts cannot escape package root through symlinks/path tricks.
- Unsafe formats are blocked with clear user-facing errors.

**Status:** complete for current public threat model.

---

### [x] Phase 8 — API error consistency and frontend refusal UX

**What was worked on**

- Standardized legacy API failures into JSON `{ error: { message, type, code, param } }` envelopes.
- Added frontend API error parsing so users see readable backend messages instead of raw JSON.
- Improved frontend refusal copy for embedding-only, metadata-only, blocked, planned, and unsupported entries.

**Completion criteria**

- Legacy and `/v1` failure paths return structured errors.
- Frontend reads `error.message` and preserves meaningful refusal details.
- UI does not imply unsupported models are runnable.

**Status:** complete for current routes and UI surfaces.

---

### [x] Phase 9 — Catalog license visibility and audit trail

**What was worked on**

- Added catalog license status and acknowledgement gating for unknown/restrictive/noncommercial entries.
- Persisted installed manifest license audit fields.
- Added UI and docs copy that records facts without giving legal advice.
- Added acceptance evidence proving refusal before download/staging when acknowledgement is required.

**Completion criteria**

- Catalog entries expose license status and warning fields.
- Acknowledgement-required installs require explicit acknowledgement.
- Installed manifests preserve license audit facts.
- Docs avoid legal approval/compliance claims.

**Status:** complete for visibility/gating; not legal review.

---

### [x] Phase 10 — Backend acceptance evidence and artifact QA

**What was worked on**

- Added optional networked backend acceptance smoke with isolated state/model dirs and named JSON artifacts.
- Added share-safe success/failure summaries.
- Added offline artifact QA to validate summary shape, caveats, legal-copy boundaries, and failed-run diagnostics.
- Added evidence for license gating, manifest audit fields, embeddings, stream refusal, GGUF metadata-only refusal, and external placeholder refusal.

**Completion criteria**

- Optional smoke produces readable artifacts and caveats.
- Failed smoke runs produce diagnostics without pretending to pass.
- Offline QA runs in default CI without starting Fathom or downloading models.
- Generated summaries do not leak local paths or overclaim support.

**Status:** complete for current acceptance evidence scope.

---

### [x] Phase 11 — API client examples and public contract clarity

**What was worked on**

- Added minimal cURL, dependency-free Python, OpenAI SDK, and `.http` examples.
- Added fake-loopback regression so examples stay narrow and CI-safe.
- Added `docs/api/v1-contract.md` as the source of truth for the supported `/v1` surface.
- Added `docs/api/public-contract.json` and offline `scripts/public_api_contract_qa.py` to keep docs/examples/README/CI aligned.

**Completion criteria**

- Examples use only documented supported `/v1` shapes.
- CI proves examples without SDK dependencies, model downloads, or external calls.
- Public contract QA guards against streaming/full parity/external proxy/arbitrary model execution overclaims.

**Status:** complete as of `cc2d93d Add public API contract QA`; fresh-clone QA was started for that commit.

---

### [x] Phase 12 — Default CI launch posture

**What was worked on**

- Removed mandatory `onnx-embeddings-ort` feature tests from default PR/push CI because that feature can trigger native ONNX Runtime binary/download behavior.
- Added CI static policy to prevent default CI from running ONNX feature tests or networked backend acceptance smoke.
- Kept optional/manual gates documented separately.

**Completion criteria**

- Default CI does not download models or native ONNX Runtime binaries.
- Default CI does not run networked backend acceptance smoke.
- CI static policy fails if those boundaries regress.

**Status:** complete.

---

### [x] Phase 13 — External API placeholder truthfulness

**What was worked on**

- Changed saved external OpenAI-compatible entries into metadata-only placeholders.
- Activation refuses external entries with `external_proxy_not_implemented`.
- `/v1/models` and `/v1/health` remain local-chat-only.
- Chat paths refuse external placeholders instead of proxying or faking responses.
- UI labels external entries as placeholders and excludes them from next-chat selection.
- Acceptance evidence proves metadata-only/refusal behavior without calling providers.

**Completion criteria**

- External entries can be saved as local metadata with key-configured flag only.
- No API key values are returned in artifacts/API responses.
- External entries do not appear in `/v1/models` or make generation ready.
- Activation/chat refuse with structured errors and no fake choices.

**Status:** complete.

---

### [x] Phase 14 — No-download real-backend public contract smoke

**What was worked on**

- Added `scripts/public_api_contract_smoke.sh`, a small real-backend smoke that starts `fathom-server` with isolated temporary state/model/log directories.
- Exercised the public `/v1` contract over real HTTP without model downloads, ONNX features, catalog installs, or external provider calls.
- Verified empty-state health/model listing, streaming refusal, missing chat model refusal, base64 embedding refusal, unknown embedding-model refusal, and external placeholder exclusion/refusal.
- Wired the smoke into default CI alongside the offline public contract QA and CI static policy.

**Completion criteria**

- Smoke is deterministic, offline/no-download, and uses isolated temp dirs.
- It starts and stops the real server cleanly.
- It verifies standard error envelopes over HTTP.
- It is wired into default CI without replacing or broadening the optional networked backend acceptance smoke.

**Status:** complete as of `caf52c4 Add public API contract smoke`. Fresh-clone QA for that commit completed before this roadmap refresh; Phase 14 did not add runtime/model-support scope.

---

### ➡️ [ ] Phase 15 — Public launch readiness bundle

**What will be worked on next**

- Keep README, backend quickstart, client examples, and the public launch checklist newcomer-friendly and aligned with the narrow launch contract.
- Confirm fresh-clone QA for the latest launch-facing commit after push.
- Ensure GitHub Actions stays green and policy-protected: no default model downloads, no networked backend acceptance smoke, and no non-default ONNX feature tests.
- Keep public-risk scan, focused sensitive checks, and tracked-large-file checks clean.

**Completion criteria**

- [`docs/public-launch-checklist.md`](docs/public-launch-checklist.md) explains clean clone/install, no-download gates, backend/API quick smoke, optional networked acceptance artifacts, boundaries, and troubleshooting.
- README and backend quickstart link the checklist from the launch/API verification path.
- `scripts/public_api_contract_qa.py` guards the checklist/linkage against public contract drift.
- Fresh clone passes documented gates for the launch-facing commit.
- Public docs do not broaden claims around production readiness, performance, legal/license advice, full OpenAI parity, streaming, external proxying, GGUF/ONNX/PyTorch, or arbitrary SafeTensors/Hugging Face execution.

---

### [ ] Phase 16 — Future runtime expansion lanes

These are intentionally **not** next until the public launch contract is stable.

**Candidate future lanes**

- More SafeTensors/HF architectures, one verified family at a time.
- Broader tokenizer/chat-template support only with fixtures and refusal tests.
- GGUF execution only after real loaders/kernels, tokenizer parity, KV cache, and sampling exist.
- ONNX only beyond embeddings if a safe, narrow, verified runtime path exists.
- External provider proxying only with explicit product decision, credential handling, opt-in behavior, and no local-runtime confusion.
- Streaming only after a real streaming implementation and API contract update.

**Completion criteria**

- Each new runtime family has real load/generate/embed behavior.
- `/v1/models`, `/v1/health`, UI readiness, docs, and tests all reflect the exact supported scope.
- Unsupported paths fail closed with structured errors.
- No model-support claim lands without tests, docs, and fresh-clone verification.

---

## Current next step

➡️ **Phase 15: Public launch readiness bundle.**

This is the next best launch-confidence step because Phase 14 is complete and the repo now needs one newcomer-friendly, checklist-backed path for clean clone, no-download verification, backend/API smoke, optional acceptance artifacts, and scoped launch boundaries. Phase 16 runtime expansion remains deferred until the launch contract is stable.

## Always-on completion gates for public-facing phases

Before calling a public-facing phase complete, prefer these gates when relevant:

```bash
git diff --check
cargo fmt --all --check
cargo test -q
python3 scripts/ci_static_policy.py
python3 scripts/public_api_contract_qa.py
python3 scripts/api_client_examples_regression.py
python3 scripts/backend_acceptance_artifact_qa.py
bash -n scripts/backend_acceptance_smoke.sh
bash scripts/public_risk_scan.sh
npm --prefix frontend run build
npm --prefix frontend run qa:copy
```

For major public-facing changes, also run fresh-clone QA against `origin/main` and confirm:

- exact commit checkout;
- clean initial/final status;
- no repo-local `AGENTS.md`;
- no tracked files over 1 MiB;
- focused sensitive-string grep clean;
- default CI remains offline/no-download/no non-default ONNX feature;
- docs do not broaden GGUF, ONNX, PyTorch `.bin`, arbitrary SafeTensors, external proxying, streaming, full OpenAI parity, performance, or legal claims.
