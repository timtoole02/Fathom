# Public launch evidence snapshot

This snapshot records the current public launch verification state for Fathom. It is evidence for the narrow local public contract only; it is not production-readiness, performance, quality, legal/license, full OpenAI parity, streaming, external proxying, or broad model-runtime evidence.

## Snapshot

- Commit: `a32505eadac6539865d224a8b4195656003a0032` (`Close out public launch readiness phase`)
- Scope: no-download public `/v1` contract, public launch checklist, public launch evidence snapshot, manifest-driven smoke, sanitized public-contract smoke artifacts, offline artifact QA, and CI/static policy wiring.
- Fresh-clone QA: passed for `origin/main` at the commit above.

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

## What this evidence does not prove

- It does not prove model download success, generation quality, embedding quality, latency, throughput, production readiness, legal/license suitability, or security hardening for exposed deployments.
- It does not prove full OpenAI API parity, streaming chat, real external provider proxying, GGUF runtime/tokenizer execution/generation, ONNX chat/general execution, PyTorch `.bin` loading, or arbitrary SafeTensors/Hugging Face execution.
- Optional networked backend acceptance smoke remains separate and should be reviewed before sharing artifacts publicly.
