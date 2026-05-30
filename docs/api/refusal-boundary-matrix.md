# Public refusal boundary matrix

This matrix is a Phase 16 prep artifact for Fathom's narrow public API contract. It records which unsupported boundaries are actively checked by no-download gates, which require optional fixture/networked evidence, and which remain explicit non-claims.

It is not a runtime expansion plan. It keeps these areas refused or unclaimed: streamed chat responses, external proxy behavior, GGUF execution, public/runtime GGUF tokenizer execution, non-embedding ONNX execution, PyTorch `.bin`, arbitrary SafeTensors/Hugging Face execution, and full OpenAI API parity.

## No-download checked boundaries

These boundaries are exercised by `scripts/public_api_contract_smoke.sh` against a real isolated backend and summarized by `scripts/public_contract_smoke_artifact_qa.py`.

| Boundary | Request hint | Expected behavior | Evidence |
| --- | --- | --- | --- |
| Streamed chat-completion requests | `stream: true` | `501 not_implemented`, no `choices` | public contract smoke |
| Base64 embeddings | `encoding_format: base64` | `400 invalid_request`, no embedding `data` | public contract smoke |
| Missing chat model | unknown local chat model id | `400 model_not_found`, no `choices` | public contract smoke |
| Malformed `/v1` JSON request body | malformed JSON body on /v1/chat/completions or /v1/embeddings | `400 invalid_request`, standard JSON error envelope, no `choices` or embedding `data` | public contract smoke |
| Unknown embedding model | unknown local embedding model id | `404 embedding_model_not_found`, no embedding `data` | public contract smoke |
| External placeholder chat or activation | external placeholder activation or chat model id | `501 external_proxy_not_implemented`, no provider call, no fake response | public contract smoke and server tests |
| Embedding models in `/v1/models` | n/a; exclusion boundary | excluded from chat/generation model listing | public contract smoke empty-state exclusion |
| PyTorch `.bin` execution | PyTorch .bin model id in /v1/chat/completions | `501 not_implemented`; blocked because pickle artifacts can execute code, no fake response | public contract smoke with a tiny synthetic local artifact and server tests |
| Unsupported ONNX chat or general ONNX model execution | unsupported ONNX model id in /v1/chat/completions | `501 not_implemented`; ONNX embeddings remain separate and narrow, no fake response | public contract smoke with a tiny synthetic local artifact and server tests |
| Unverified SafeTensors/Hugging Face model execution | unverified SafeTensors/Hugging Face model id in /v1/chat/completions | `501 not_implemented`; only explicitly verified local lanes are runnable, no fake response | public contract smoke with a tiny synthetic local HF-style SafeTensors package |
| GGUF metadata-only chat attempts | metadata-only GGUF model id in /v1/chat/completions | `501 not_implemented`; metadata/readiness only, no public/runtime tokenizer execution, no runtime weight loading, no dequantization/kernels, and no generation | public contract smoke with a tiny synthetic metadata-only local GGUF file, plus optional backend acceptance smoke |
| Unsupported `/v1` endpoint | POST /v1/responses | `404 not_found`; unsupported OpenAI-style endpoints stay outside the narrow public contract but still return the standard JSON error envelope | public contract smoke and server tests |
| Unsupported `/v1` method | GET /v1/chat/completions | `405 method_not_allowed`; known `/v1` paths reject unsupported HTTP methods with the standard JSON error envelope | public contract smoke and server tests |

## Optional fixture/networked evidence boundaries

No manifest boundaries currently require optional fixture/networked evidence. Optional backend acceptance smoke still records catalog-backed GGUF metadata-only evidence, but the public no-download smoke now covers the GGUF refusal contract with a tiny synthetic local file.

## Non-claim boundaries

| Boundary | Request hint | Expected behavior | Evidence |
| --- | --- | --- | --- |
| Full OpenAI API parity | n/a; non-claim boundary | not claimed | docs/static contract QA |
| Production readiness, performance, quality, legal/license suitability | n/a; non-claim boundary | not claimed | public launch checklist and evidence caveats |
| Real external provider proxying | external placeholder activation or chat model id | not implemented; external entries are metadata placeholders | public contract smoke and docs/static QA |

## Updating this matrix

When adding a future Phase 16 capability, update this matrix in the same change as:

1. the runtime or refusal implementation;
2. `/v1` docs and `docs/api/public-contract.json`;
3. tests or smoke evidence proving the new behavior;
4. static QA guardrails that prevent unsupported boundaries from silently becoming claims.

If the capability is not implemented and verified, keep it listed as a refusal or non-claim boundary.
