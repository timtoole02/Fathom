## Summary

-

## Scope and truthfulness boundary

- [ ] This PR does not fake inference, embeddings, installs, downloads, readiness, benchmark results, or runtime availability.
- [ ] Any new/changed support is described narrowly by format, family, task, feature flag, fixture, endpoint, and known blockers.
- [ ] This PR does not imply broad GGUF runtime/tokenizer/generation support, ONNX chat/general ONNX support, PyTorch `.bin` loading, arbitrary SafeTensors execution, streaming/full OpenAI parity, GPU support, batching, or performance claims unless implemented and verified here.

## Privacy and artifact review

- [ ] Public text, logs, screenshots, fixtures, and generated artifacts were reviewed for private prompts, credentials, usernames, hostnames, absolute local paths, and model-store details.
- [ ] Attached evidence uses synthetic/share-safe prompts and minimal local context.
- [ ] `bash scripts/public_risk_scan.sh` was run when public-facing text or artifacts changed; this is a guardrail, not a complete privacy audit.

## Docs, UI, and API consistency

- [ ] README, CONTRIBUTING, SECURITY, API docs, UI copy, and tests tell the same capability story where relevant.
- [ ] `/v1` behavior changes are reflected in `docs/api/v1-contract.md` when relevant.
- [ ] User-facing copy distinguishes runnable, planned, blocked, metadata-only, and unavailable states truthfully.

## Gates run

- [ ] `git diff --check`
- [ ] Python syntax gate from `docs/public-launch-checklist.md` was run: `python3 -m py_compile ...`
- [ ] Shell syntax gate from `docs/public-launch-checklist.md` was run: `bash -n ...`
- [ ] `python3 scripts/api_client_examples_regression.py`
- [ ] `python3 scripts/api_client_examples_regression.py --self-test`
- [ ] `python3 scripts/public_api_contract_qa.py`
- [ ] `python3 scripts/public_api_contract_qa.py --self-test`
- [ ] `python3 scripts/public_contract_smoke_artifact_qa.py`
- [ ] `python3 scripts/backend_acceptance_artifact_qa.py`
- [ ] `python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py`
- [ ] `python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py`
- [ ] `python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py`
- [ ] `python3 scripts/ci_static_policy.py`
- [ ] `python3 scripts/ci_static_policy.py --self-test`
- [ ] `cargo fmt --all --check`
- [ ] `cargo test -q`
- [ ] `npm --prefix frontend run build`
- [ ] `npm --prefix frontend run qa:copy`
- [ ] `bash -n scripts/public_risk_scan.sh`
- [ ] `bash scripts/public_risk_scan.sh --self-test`
- [ ] `bash scripts/public_risk_scan.sh`
- [ ] Not applicable / explained below

## Manual or release-facing checks

- [ ] For backend/API changes, I considered the optional networked acceptance smoke. It remains non-default CI and should be run only as a targeted manual check when appropriate: `FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh`.
- [ ] For ONNX embedding changes, I considered the targeted/manual feature gate: `cargo test -q --features onnx-embeddings-ort`.
- [ ] Release-facing claims include exact evidence and caveats: commit, feature flags, model or fixture, request shape, warm/cold state, and what the evidence does not prove.

## Notes for reviewers

-
