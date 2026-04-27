## Summary

-

## Scope and truthfulness boundary

- [ ] This PR does not fake inference, embeddings, installs, downloads, readiness, benchmark results, or runtime availability.
- [ ] Any new/changed support is described narrowly by format, family, task, feature flag, fixture, endpoint, and known blockers.
- [ ] This PR does not imply broad GGUF runtime/tokenizer/generation support, ONNX chat/general ONNX support, PyTorch `.bin` loading, arbitrary SafeTensors execution, streaming/full OpenAI parity, GPU support, batching, or performance claims unless implemented and verified here.

## Privacy and artifact review

- [ ] Public text, logs, screenshots, fixtures, and generated artifacts were reviewed for private prompts, credentials, usernames, hostnames, absolute local paths, and model-store details.
- [ ] Attached evidence uses synthetic/share-safe prompts and minimal local context.
- [ ] `bash scripts/public_risk_scan.sh` was run when public-facing text or artifacts changed.

## Docs, UI, and API consistency

- [ ] README, CONTRIBUTING, SECURITY, API docs, UI copy, and tests tell the same capability story where relevant.
- [ ] `/v1` behavior changes are reflected in `docs/api/v1-contract.md` when relevant.
- [ ] User-facing copy distinguishes runnable, planned, blocked, metadata-only, and unavailable states truthfully.

## Gates run

- [ ] `git diff --check`
- [ ] `cargo fmt --all --check`
- [ ] `cargo test -q`
- [ ] `cargo test -q --features onnx-embeddings-ort`
- [ ] `npm --prefix frontend run build`
- [ ] `npm --prefix frontend run qa:copy`
- [ ] `bash -n scripts/public_risk_scan.sh`
- [ ] `bash scripts/public_risk_scan.sh`
- [ ] Not applicable / explained below

## Manual or release-facing checks

- [ ] For backend/API changes, I considered the optional networked acceptance smoke. It remains non-default CI and should be run only as a targeted manual check when appropriate: `FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh`.
- [ ] Release-facing claims include exact evidence and caveats: commit, feature flags, model or fixture, request shape, warm/cold state, and what the evidence does not prove.

## Notes for reviewers

-
