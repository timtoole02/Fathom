# Contributing

Fathom is early local-inference infrastructure. Contributions are welcome when they preserve the central rule: be truthful about what actually runs.

## Truthfulness principles

- Do not fake inference, embeddings, readiness, installs, downloads, benchmark results, or runtime availability.
- A model should appear runnable only after the relevant loader/runtime path has validated the package and can produce real output for that task.
- If support is partial, say exactly what is partial: format, family, fixture, feature flag, endpoint, and known blockers.
- Prefer clear refusals over optimistic fallbacks. Unsupported packages should explain why they are blocked, planned, metadata-only, or unavailable.
- Public docs, UI copy, API responses, tests, and benchmarks should all tell the same story.

## Contribution boundaries

Unless a change includes real implementation and verification, do not add or imply:

- broad OpenAI API parity;
- streaming chat support;
- general GGUF inference, tokenizer execution, dequantization, or generation;
- no claims of ONNX chat or general ONNX model support unless a verified implementation and docs land with it;
- PyTorch `.bin` / pickle loading;
- execution of arbitrary Hugging Face or SafeTensors packages;
- GPU, batching, shared-session memory, production-throughput, or quality claims;
- fake vectors, fake completions, placeholder runtimes, or mock installs presented as real.

Narrow support is fine when it is honest. For example, a pinned fixture, specific model family, specific file layout, or feature-gated runtime can be documented as such.

## Verification gates

GitHub Actions runs the core public gates on pull requests and pushes to `main`: Rust formatting and tests, frontend install/build/copy QA, script and Python syntax checks, and the public-risk scan.

Run the smallest gate that proves your change, and prefer the full set before opening a release-facing PR.

Common gates:

```bash
git diff --check
cargo fmt --all --check
cargo test -q
cargo test -q --features onnx-embeddings-ort
npm --prefix frontend run build
npm --prefix frontend run qa:copy
bash -n scripts/public_risk_scan.sh
bash scripts/public_risk_scan.sh
```

For backend/API changes, also consider the networked acceptance smoke locally or in a targeted manual CI run; it is intentionally not part of the default public CI path:

```bash
bash -n scripts/backend_acceptance_smoke.sh
FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh
```

For docs-only changes, `git diff --check`, script syntax checks, and `bash scripts/public_risk_scan.sh` are usually the minimum useful gates.

## Docs and benchmark caveats

Benchmark and smoke evidence should say what it proves and what it does not prove.

Include relevant context such as commit, model id or fixture, feature flags, backend port only if needed, warm/cold server state, request shape, and whether the evidence came from tiny/random fixtures. Avoid turning local fixture smoke into throughput, quality, production-capacity, GPU, or broad model-support claims.

When documenting runtime metrics, keep timing caveats clear: server-side non-streaming timing is not client-observed streaming latency.

## Privacy and public-risk scan expectations

Before committing public-facing text or generated artifacts:

- remove personal names, hostnames, usernames, local absolute paths, private temp paths, credentials, and prompt/document text that should not be public;
- inspect JSON artifacts and logs manually before sharing them;
- run `bash scripts/public_risk_scan.sh` from the repo root;
- keep the scanner itself free of literal private strings where possible.

The risk scan is not a substitute for judgment. If an artifact contains local state, logs, prompts, or user-provided documents, review it manually even if the scan passes.
