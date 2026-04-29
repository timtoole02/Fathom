# Optional MiniLM embeddings API acceptance flow

This is an opt-in local acceptance flow for the pinned `hf-sentence-transformers-all-minilm-l6-v2-safetensors` catalog demo. It exercises Fathom's default-build SafeTensors MiniLM embedding path and narrow `/v1/embeddings` adapter while keeping the public `/v1` contract unchanged.

Do not add this flow to default CI. It downloads or reuses about 91 MB of model files. Treat the output as embedding API evidence only, not as an embedding quality, retrieval quality, latency, throughput, production-readiness, legal, arbitrary Hugging Face, ONNX chat, or full OpenAI parity claim.

## What this flow should prove

- The catalog entry is still pinned to `sentence-transformers/all-MiniLM-L6-v2` at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`.
- Catalog installation verifies expected file sizes, SHA256 hashes, Apache-2.0 metadata, and manifest fields before the model becomes available.
- `/api/embedding-models` includes the installed MiniLM embedding model.
- `/v1/models` excludes the embedding-only model from chat/generation listings.
- `/v1/embeddings` returns real finite 384-dimensional float vectors for multiple inputs with `fathom.runtime: candle-bert-embeddings`.
- `encoding_format: "base64"` remains refused with `400 invalid_request`.
- Chat against the embedding-only model remains refused with no fake `choices`.

## One-command optional smoke

```bash
FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE=1 \
FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_KEEP_ARTIFACTS=1 \
bash scripts/minilm_embeddings_optional_api_acceptance_smoke.sh
```

The script starts an isolated server, installs only the pinned MiniLM SafeTensors catalog entry, writes health/install/embedding-models/`/v1/models`/float-embeddings/base64-refusal/chat-refusal artifacts plus `summary.json` and `summary.md`, then runs offline artifact QA. It is intentionally not part of default CI.

Validate preserved artifacts later with:

```bash
python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py /path/to/artifacts
```

With no arguments, the artifact QA runs a dependency-free synthetic self-test.

## Isolated directories

The harness uses isolated directories by default and supports overrides:

- `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ROOT`
- `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_MODELS_DIR`
- `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_STATE_DIR`
- `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_LOG_DIR`
- `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ARTIFACT_DIR`
- `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_PORT`

Set `FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_KEEP_ARTIFACTS=1` when you want to review or preserve generated evidence.

## Share-safe artifact review

Before copying generated artifacts into checked-in docs or public issue comments:

- Replace local paths with generic placeholders.
- Remove text inputs that should not be public.
- Keep only summarized fields needed to prove install, embedding model listing, `/v1/models` exclusion, vector shape, metadata, and refusal behavior.
- Run `bash scripts/public_risk_scan.sh` after adding any sanitized evidence.

## Boundaries preserved

This flow does not claim public/runtime GGUF tokenizer execution, GGUF inference, ONNX chat/general execution, PyTorch `.bin`, external provider proxying, arbitrary SafeTensors/HF execution, streaming, or full OpenAI API parity. Embedding-only models remain excluded from `/v1/models`.
