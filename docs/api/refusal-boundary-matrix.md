# Public refusal boundary matrix

This matrix is a Phase 16 prep artifact for Fathom's narrow public API contract. It records which unsupported boundaries are actively checked by no-download gates, which require optional fixture/networked evidence, and which remain explicit non-claims.

It is not a runtime expansion plan. It keeps these areas refused or unclaimed: streamed chat responses, external proxy behavior, GGUF execution, public/runtime GGUF tokenizer execution, non-embedding ONNX execution, PyTorch `.bin`, arbitrary SafeTensors/Hugging Face execution, and full OpenAI API parity.

## No-download checked boundaries

These boundaries are exercised by `scripts/public_api_contract_smoke.sh` against a real isolated backend and summarized by `scripts/public_contract_smoke_artifact_qa.py`.

| Boundary | Expected behavior | Evidence |
| --- | --- | --- |
| Streamed chat-completion requests | `501 not_implemented`, no `choices` | public contract smoke |
| Base64 embeddings | `400 invalid_request`, no embedding `data` | public contract smoke |
| Missing chat model | `400 model_not_found`, no `choices` | public contract smoke |
| Unknown embedding model | `404 embedding_model_not_found`, no embedding `data` | public contract smoke |
| External placeholder chat or activation | `501 external_proxy_not_implemented`, no provider call, no fake response | public contract smoke and server tests |
| Embedding models in `/v1/models` | excluded from chat/generation model listing | public contract smoke empty-state exclusion |

## Optional fixture/networked evidence boundaries

These boundaries need registered/downloaded model state or catalog fixtures, so they are intentionally outside default no-download CI. They are covered by optional backend acceptance evidence when that smoke is run locally with network access.

| Boundary | Expected behavior | Evidence |
| --- | --- | --- |
| GGUF metadata-only chat attempts | `501 not_implemented`; metadata/readiness only, no public/runtime tokenizer execution and no generation | optional backend acceptance smoke |
| PyTorch `.bin` execution | `501 not_implemented`; blocked because pickle artifacts can execute code | server tests and optional acceptance-style refusal evidence |
| Unsupported ONNX chat or general ONNX model execution | `501 not_implemented`; ONNX embeddings remain separate and narrow | docs/static contract boundary |
| Unverified SafeTensors/Hugging Face model execution | `501 not_implemented`; only explicitly verified local lanes are runnable | docs/static contract boundary and runtime tests for supported lanes |

## Non-claim boundaries

| Boundary | Expected behavior | Evidence |
| --- | --- | --- |
| Full OpenAI API parity | not claimed | docs/static contract QA |
| Production readiness, performance, quality, legal/license suitability | not claimed | public launch checklist and evidence caveats |
| Real external provider proxying | not implemented; external entries are metadata placeholders | public contract smoke and docs/static QA |

## Updating this matrix

When adding a future Phase 16 capability, update this matrix in the same change as:

1. the runtime or refusal implementation;
2. `/v1` docs and `docs/api/public-contract.json`;
3. tests or smoke evidence proving the new behavior;
4. static QA guardrails that prevent unsupported boundaries from silently becoming claims.

If the capability is not implemented and verified, keep it listed as a refusal or non-claim boundary.
