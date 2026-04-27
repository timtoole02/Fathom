# Backend-only quickstart

Use this when you want Fathom as a local API/runtime layer without the React frontend.

## Start only the API server

```bash
git clone https://github.com/timtoole02/Fathom/ fathom
cd fathom
cargo test
bash scripts/start-backend.sh
```

The backend listens on `http://127.0.0.1:8180` by default and writes logs to `~/.fathom/logs/server.log`.

Useful environment overrides:

```bash
FATHOM_PORT=19180 bash scripts/start-backend.sh
FATHOM_MODELS_DIR=/tmp/fathom-models FATHOM_STATE_DIR=/tmp/fathom-state bash scripts/start-backend.sh
FATHOM_FEATURES=onnx-embeddings-ort bash scripts/start-backend.sh
```

Stop it with:

```bash
bash scripts/stop.sh
```

`scripts/stop.sh` also stops the frontend if it was started by `scripts/start.sh`, but `scripts/start-backend.sh` never starts the frontend.

The exact narrow `/v1` contract is documented in [`v1-contract.md`](v1-contract.md). Minimal client examples for cURL, dependency-free Python, the OpenAI Python SDK, and `.http` editors live in [`client-examples.md`](client-examples.md) and [`../../examples/api/`](../../examples/api/). For local threat-model, model artifact safety, and reporting guidance, see [`../../SECURITY.md`](../../SECURITY.md); for contribution boundaries and verification gates, see [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Optional backend acceptance smoke

For a fuller backend-only acceptance pass, run:

```bash
FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh
```

This optional script starts only `fathom-server` on an alternate port with isolated state/model directories, downloads pinned catalog fixtures, captures JSON artifacts, and verifies health/runtime/capabilities, catalog license metadata and acknowledgement gating, TinyStories chat, MiniLM embeddings, explicit-vector retrieval, retrieval-context chat metadata, and GGUF metadata-only refusal. It is networked and intentionally not part of default CI or benchmark evidence.

### Reading acceptance artifacts

Set `FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1` to keep the temporary artifact directory printed at the end of the run. Start with `summary.md` for the human-readable index and `summary.json` for machine-readable metadata such as repo commit, base URL, port, fixture model ids, relative state/model/log directories, timestamps, and named checks. Full local paths are isolated in `summary.local.json` for the runner; review that file, server logs, and full JSON payloads before sharing artifacts publicly because they may contain local paths or request text.

Useful evidence files:

- `00-corrupt-state-runtime.json`, `00-corrupt-state-files.json`, and `02b-api-dashboard-after-corrupt-recovery.json` show corrupt model-state recovery.
- `03b-api-models-catalog-license-metadata.json`, `03c-catalog-license-install-refusal.json`, and `03d-catalog-license-refusal-model-dir.json` show catalog license metadata visibility, acknowledgement-required refusal for a non-permissive entry, and refusal before download/staging in the isolated model directory. This is gating evidence only, not legal review or license-compatibility advice.
- `06-chat-non-stream.json` shows the pinned TinyStories SafeTensors/HF fixture returning a real non-streaming chat completion; `07-chat-stream-refusal.json` shows streaming refusal.
- `10-v1-embeddings-minilm.json` shows `POST /v1/embeddings` returning an OpenAI-style list with one finite 384-dimensional float vector from the pinned MiniLM SafeTensors runtime.
- `10b-v1-embeddings-base64-refusal.json` shows `encoding_format: "base64"` refused with `invalid_request`; only float embeddings are supported.
- `10-minilm-embed.json` shows the older `/api/embedding-models/:id/embed` endpoint returning a finite normalized 384-dimensional vector.
- `13-retrieval-search.json` and `14-chat-with-retrieval.json` show explicit-vector retrieval and retrieval-context chat metadata.
- `15-install-gguf-metadata-only.json`, `16-v1-models-after-gguf.json`, and `17-chat-gguf-refusal.json` show pinned GGUF metadata-only registration, `/v1/models` exclusion, and chat refusal.

If artifacts were created in the script's default temp directory, they are rebuildable and safe to remove after you have copied anything you need. Be more careful with custom `FATHOM_ACCEPTANCE_MODELS_DIR`, `FATHOM_ACCEPTANCE_STATE_DIR`, or `FATHOM_ACCEPTANCE_ARTIFACT_DIR` values: delete only the acceptance-run directories you intentionally created, not a shared/user model store or source-of-truth state. Run `bash scripts/public_risk_scan.sh` before public handoff to catch tracked-file privacy regressions and uncaveated support claims.

## Health and runtime discovery

For supported `/v1` JSON shapes and unsupported OpenAI features, see [`v1-contract.md`](v1-contract.md).

```bash
BASE=http://127.0.0.1:8180
curl -fsS "$BASE/v1/health" | python3 -m json.tool
curl -fsS "$BASE/api/runtime" | python3 -m json.tool
curl -fsS "$BASE/api/capabilities" | python3 -m json.tool
```

## Catalog and model APIs

List downloadable catalog entries. Each item includes `license`, `license_status`, `license_acknowledgement_required`, and `license_warning` so clients can show license visibility before install:

```bash
curl -fsS "$BASE/api/models/catalog" | python3 -m json.tool
```

Permissive catalog entries install with the existing `{repo_id, filename}` body. Entries marked unknown or restrictive/non-commercial are refused with `catalog_license_ack_required` unless the client includes `"accept_license": true` after the user explicitly acknowledges the listed status. This acknowledgement gate is a visibility/safety guardrail; it is not legal advice or a compatibility determination for your intended use.

Install the small trained TinyStories GPT-2 demo through the same verified catalog path used by the UI:

```bash
curl -fsS "$BASE/api/models/catalog/install" \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"vijaymohan/gpt2-tinystories-from-scratch-10m","filename":"model.safetensors"}' \
  | python3 -m json.tool
```

List runnable OpenAI-style models:

```bash
curl -fsS "$BASE/v1/models" | python3 -m json.tool
```

Only models Fathom has actually validated as runnable appear in `/v1/models`.

Install the tiny pinned GGUF metadata fixture when you want to verify real GGUF provenance and inspection without claiming inference:

```bash
curl -fsS "$BASE/api/models/catalog/install" \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"aladar/llama-2-tiny-random-GGUF","filename":"llama-2-tiny-random.gguf"}' \
  | python3 -m json.tool
```

That install records repo/revision/license/size/SHA256 in `fathom-download-manifest.json` and registers as `metadata_only`. Fathom may derive bounded tensor metadata, privately retain bounded tokenizer metadata for narrow synthetic GPT-2/BPE and Llama/SentencePiece GGUF shapes, and keep private fixture-scoped Llama/SentencePiece encode/decode parity helpers and internal payload-readiness facts, but public/runtime GGUF tokenizer execution, runtime weight loading, general dequantization, quantized kernels, architecture runtime, and generation are still not supported. The fixture remains excluded from `/v1/models`; `/v1/chat/completions` refuses it rather than running tokenizer-only or fake generation.

Optional live parser smoke for the same 1.75 MB file:

```bash
cargo test -q -p fathom-server live_pinned_gguf_fixture_download_inspects_metadata_only -- --ignored
```

## Chat completions

Use the model id returned by `/v1/models`:

```bash
curl -fsS "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors",
    "messages": [{"role": "user", "content": "Once upon a time, a small robot"}],
    "max_tokens": 32,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95
  }' | python3 -m json.tool
```

`stream: true` is rejected today with a truthful `501`; use non-streaming completions. If a runnable SafeTensors/HF chat package includes `chat_template` metadata, Fathom renders only a small tested set of patterns today: ChatML/Qwen-style, `[INST] ... [/INST]`, and Gemma `<start_of_turn>` user/model turns. Other HF templates are rejected with `chat_template_not_supported` rather than guessed. Verified GPT-2/TinyStories, Llama/SmolLM2, Qwen2, Phi, Mistral, and Gemma CPU/F32 Candle chat lanes report `runtime_cache_hit`, `runtime_cache_lookup_ms`, `runtime_residency`, and `runtime_family`; warm residency keeps model state process-local while prompt tokens, KV cache, sampling, and generated text stay request-local. Qwen2, Phi, Mistral, and Gemma residency is serialized mutable model reuse with explicit KV reset, not parallel batching or shared session memory. Conversation records can persist these optional server-side metrics so the chat UI can show cold/warm runtime rows for assistant replies.

## Retrieval indexes with explicit vectors

Fathom can persist and search caller-supplied vectors without embedding inference:

```bash
curl -fsS "$BASE/api/retrieval-indexes" \
  -H "Content-Type: application/json" \
  -d '{"id":"notes","embedding_model_id":"external-demo-3d","embedding_dimension":3}' \
  | python3 -m json.tool

curl -fsS "$BASE/api/retrieval-indexes/notes/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk": {
      "id": "rust-1",
      "document_id": "dev-notes",
      "text": "Rust ownership keeps memory safety explicit.",
      "byte_start": 0,
      "byte_end": 44
    },
    "vector": [0.9, 0.1, 0.0]
  }' | python3 -m json.tool

curl -fsS "$BASE/api/retrieval-indexes/notes/search" \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.85,0.15,0.0],"top_k":1,"metric":"cosine"}' \
  | python3 -m json.tool
```

Opt-in retrieval context can be inserted into chat by adding `fathom.retrieval` with an explicit `query_vector` to `/v1/chat/completions`.

## Embedding endpoint

Ordinary builds include a default Candle/SafeTensors MiniLM embedding lane for the verified `sentence-transformers/all-MiniLM-L6-v2` fixture. Install it, then list embedding models to copy the returned `id`:

```bash
curl -fsS "$BASE/api/models/catalog/install" \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"sentence-transformers/all-MiniLM-L6-v2","filename":"model.safetensors"}' \
  | python3 -m json.tool

curl -fsS "$BASE/api/embedding-models" | python3 -m json.tool
```

Call the embed endpoint with that installed model id:

```bash
EMBED_MODEL_ID=sentence-transformers-all-minilm-l6-v2-model-safetensors

curl -fsS "$BASE/api/embedding-models/$EMBED_MODEL_ID/embed" \
  -H "Content-Type: application/json" \
  -d '{"input":["Rust ownership keeps memory safety explicit."],"normalize":true}' \
  | python3 -m json.tool

curl -fsS "$BASE/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$EMBED_MODEL_ID\",\"input\":[\"Rust ownership keeps memory safety explicit.\"],\"encoding_format\":\"float\"}" \
  | python3 -m json.tool
```

The SafeTensors lane returns finite 384-dimensional vectors and stays out of `/v1/models` because it is not a chat/generation model. The `/v1/embeddings` adapter is intentionally narrow: string or array input, float embeddings only, and no fake vectors for chat, GGUF, PyTorch `.bin`, or unsupported packages. See [`v1-contract.md`](v1-contract.md) for the request/response envelope.

Fathom also has a pinned ONNX MiniLM embedding fixture. ONNX inference is real but non-default because it brings ONNX Runtime binaries; build/start with the feature when you want that lane:

```bash
FATHOM_FEATURES=onnx-embeddings-ort bash scripts/start-backend.sh
```

Then install the ONNX fixture and call the same embed endpoint with its installed id:

```bash
curl -fsS "$BASE/api/models/catalog/install" \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"nixiesearch/all-MiniLM-L6-v2-onnx","filename":"model_quantized.onnx"}' \
  | python3 -m json.tool
```

## Backend benchmarking

With a backend already running, the dependency-free benchmark harness can time any runnable `/v1/models` chat entries it finds:

```bash
python3 scripts/bench_backend.py --runs 3 --warmups 1 --cache-phase-report --markdown /tmp/fathom-bench.md
```

To include embeddings, install a runnable embedding fixture and pass its id explicitly. The SafeTensors MiniLM fixture works in ordinary builds; the ONNX fixture requires `FATHOM_FEATURES=onnx-embeddings-ort`:

```bash
python3 scripts/bench_backend.py \
  --embedding-model "$EMBED_MODEL_ID" \
  --runs 3 \
  --warmups 1 \
  --markdown /tmp/fathom-bench.md
```

The harness reports JSON to stdout and can write Markdown. `--cache-phase-report` labels the first successful same-process chat request as `cold_candidate` and later same-model requests as `warm_candidate`, and summarizes `runtime_cache_hit`, `runtime_residency`, `runtime_family`, and `runtime_cache_lookup_ms` in JSON/Markdown. It does not restart the backend, download models, or make unsupported fixtures runnable, so treat numbers as machine/model/build-specific evidence rather than broad speed claims. Current six-family local fixture smoke evidence is checked in at [`docs/benchmarks/2026-04-26-local-apple-silicon-candle-cache-six-family.md`](../benchmarks/2026-04-26-local-apple-silicon-candle-cache-six-family.md).
