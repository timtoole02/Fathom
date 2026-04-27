# Fathom

Fathom is an early Rust-native local inference sandbox for making local model artifacts understandable and runnable without pretending everything works already.

The long-term goal is a local runtime/router that can inspect common model packages, explain what can run them, choose the right backend lane, and expose one OpenAI-compatible API. Today, that means small but real custom Rust SafeTensors/Hugging Face chat and embedding paths plus a bounded retrieval API.

## What works today

| Area | Status |
| --- | --- |
| Local chat generation | Real generation for narrow custom Rust SafeTensors/HF lanes: GPT-2-style models, Llama/Llama-style tied-embedding packages, Qwen2 fixtures, Phi fixtures, Mistral fixtures, and Gemma fixtures. |
| Best first demo | **TinyStories GPT-2 10M** from the Models page. Small, trained, and fast enough for visible local text checks. |
| Larger chat-tuned demo | **SmolLM2 135M Instruct**. Validates the tied-embedding Llama-style lane with a real instruct package; larger than the first-run demo. |
| Runtime API | OpenAI-style `GET /v1/models`, `POST /v1/chat/completions`, and narrow `POST /v1/embeddings` for models Fathom actually considers runnable for that task. |
| Capability reporting | `/api/capabilities`, model inspection, backend-lane summaries, and UI copy distinguish runnable, metadata-readable, planned, blocked, and unsupported states. GGUF files can expose bounded header/key-value metadata, tensor descriptors, validated internal payload ranges, tokenizer/architecture compatibility hints, internal bounded tokenizer metadata retention for narrow synthetic GPT-2/BPE and Llama/SentencePiece shapes, private fixture-scoped Llama/SentencePiece encode/decode parity helpers, and internal synthetic payload decode checks without claiming public/runtime tokenizer execution or inference. |
| Catalog installs | Hugging Face catalog downloads are pinned to exact revisions, checked for expected file sizes, verified with per-file SHA256, license-visible in `/api/models/catalog`, and recorded in `fathom-download-manifest.json`. Catalog entries with unknown or restrictive/non-commercial license status require explicit acknowledgement before install; permissive entries keep the existing `{repo_id, filename}` request shape. The catalog includes one tiny real GGUF fixture (`aladar/llama-2-tiny-random-GGUF` at revision `8d5321916486e1d33c46b16990e8da6567785769`) strictly for metadata/provenance inspection. |
| Local embeddings | Default-build Candle/SafeTensors embeddings for the verified `sentence-transformers/all-MiniLM-L6-v2` fixture; vectors are mean-pooled, optionally L2-normalized, and excluded from `/v1/models` chat listings. |
| Retrieval | Persistent vector indexes with caller-supplied or verified local embedding vectors, vector search, and opt-in retrieval context insertion for chat via `fathom.retrieval`. |
| ONNX embeddings | Feature-gated real ONNX Runtime embeddings for the pinned MiniLM fixture via `onnx-embeddings-ort`; vectors are mean-pooled from `last_hidden_state` and optionally L2-normalized. |

## What does **not** work yet

Fathom does **not** currently claim general model support. In particular:

- No native GGUF runtime yet. The pinned `aladar/llama-2-tiny-random-GGUF` catalog fixture verifies real GGUF provenance and metadata inspection only; GGUF header/key-value metadata, tensor-descriptor inspection, validated payload-range derivation, tokenizer/architecture compatibility hints, internal bounded tokenizer metadata retention for narrow synthetic GPT-2/BPE and Llama/SentencePiece shapes, private fixture-scoped Llama/SentencePiece encode/decode parity helpers, and private synthetic F32/F16/Q8_0/Q4_0 payload decode tests are metadata/readiness groundwork, not public/runtime tokenizer execution, runtime weight loading, general dequantization, quantized kernels, architecture runtime, generation, or inference.
- No ONNX chat/LLM runtime yet.
- No default ONNX embedding inference in ordinary builds; real ONNX embeddings require the non-default `onnx-embeddings-ort` feature and are currently limited to the pinned MiniLM ONNX fixture path.
- No PyTorch `.bin`, MLX, CoreML, TensorRT, or arbitrary Hugging Face model execution.
- No fake fallbacks: if a model is known but not runnable, Fathom should return a clear error instead of placeholder text.

Supported today means narrow: a compatible HF package must include `config.json`, tokenizer files, and `model.safetensors`, then pass the lane-specific loader checks for a current custom Rust runtime path. Chat models must pass generation checks; embedding models must pass embedding-loader, pooling, and vector-shape checks. For packages with Hugging Face `chat_template` metadata, Fathom renders only a small tested set of local patterns today: ChatML/Qwen-style, `[INST] ... [/INST]`, and Gemma `<start_of_turn>` user/model turns; unsupported templates are refused with `chat_template_not_supported` instead of guessed.

## Prerequisites

- Rust toolchain with `cargo` (`rustup` is recommended)
- Node.js for the frontend toolchain (Node 22 LTS recommended; Vite requires Node `20.19+` or `22.12+`)
- `npm`, `curl`, `python3`, and a network connection for catalog downloads

## Quick start

```bash
git clone https://github.com/timtoole02/Fathom/ fathom
cd fathom
npm --prefix frontend install
cargo test
bash scripts/start.sh
```

Then open:

- UI: http://127.0.0.1:4185
- Health: http://127.0.0.1:8180/v1/health
- Capabilities: http://127.0.0.1:8180/api/capabilities

The first `scripts/start.sh` run builds the Rust backend in release mode, so it may take a few minutes. Logs are written to `~/.fathom/logs`.

Stop Fathom with:

```bash
bash scripts/stop.sh
```

## Backend-only quick start

Fathom can run as a standalone API layer without the frontend:

```bash
bash scripts/start-backend.sh
```

This starts only `fathom-server` on `http://127.0.0.1:8180`; it does not require `npm install` and does not launch Vite. Use `FATHOM_PORT`, `FATHOM_MODELS_DIR`, and `FATHOM_STATE_DIR` to isolate API tests. To enable the non-default ONNX embedding runtime lane, start with:

```bash
FATHOM_FEATURES=onnx-embeddings-ort bash scripts/start-backend.sh
```

The narrow `/v1` contract is documented in [`docs/api/v1-contract.md`](docs/api/v1-contract.md). Backend/API examples for health, capabilities, catalog install, `/v1/models`, `/v1/chat/completions`, explicit-vector retrieval, and embedding endpoints live in [`docs/api/backend-only-quickstart.md`](docs/api/backend-only-quickstart.md). Minimal client examples for cURL, dependency-free Python, the OpenAI Python SDK, and `.http` editors live in [`docs/api/client-examples.md`](docs/api/client-examples.md). Security posture and local threat-model notes live in [`SECURITY.md`](SECURITY.md); contribution truthfulness and verification expectations live in [`CONTRIBUTING.md`](CONTRIBUTING.md). For a fuller optional networked backend-only acceptance pass with isolated state/model directories, run `FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh`; it includes catalog license metadata/acknowledgement-gating evidence before the first fixture download. Before sharing generated artifacts publicly, run `bash scripts/public_risk_scan.sh` and review logs/full JSON payloads for local paths or request text.

## Try local generation in the UI

1. Start Fathom with `bash scripts/start.sh`.
2. Open http://127.0.0.1:4185.
3. Open the Models page.
4. Download **TinyStories GPT-2 10M** for the fastest human-visible demo, or **SmolLM2 135M Instruct** for a larger chat-tuned Llama-style demo.
5. Wait for the package to save into Fathom-managed storage, verify, and register.
6. Select/load it, then send a short prompt.

The tiny random GPT-2, Llama, Qwen2, Phi, Mistral, and Gemma SafeTensors catalog entries are useful for backend smoke tests, but their output can be gibberish or whitespace. Use TinyStories or SmolLM2 when you want a demo that looks more like language. The tiny random GGUF catalog fixture is different: it is metadata/provenance-only and will not appear in `/v1/models`.

## Try local generation by API

For the exact supported `/v1` request/response shapes, see [`docs/api/v1-contract.md`](docs/api/v1-contract.md). For minimal copy-paste client examples, see [`docs/api/client-examples.md`](docs/api/client-examples.md) and the scripts in [`examples/api/`](examples/api/).

List catalog entries:

```bash
curl -fsS http://127.0.0.1:8180/api/models/catalog | python3 -m json.tool
```

Catalog entries expose `license_status`, `license_acknowledgement_required`, and `license_warning`. Unknown or restrictive/non-commercial entries require explicit acknowledgement before install; that gate is visibility/refusal evidence, not legal review or license-compatibility advice.

Install TinyStories GPT-2 10M through the same catalog path used by the UI:

```bash
curl -fsS http://127.0.0.1:8180/api/models/catalog/install \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"vijaymohan/gpt2-tinystories-from-scratch-10m","filename":"model.safetensors"}' | python3 -m json.tool
```

List runnable OpenAI-style models:

```bash
curl -fsS http://127.0.0.1:8180/v1/models | python3 -m json.tool
```

Only models Fathom has actually validated as runnable appear in `/v1/models`. The pinned GGUF fixture can be installed for metadata/provenance inspection, including private bounded retention of narrow Llama/SentencePiece-shaped tokenizer metadata and private fixture-scoped Llama/SentencePiece encode/decode parity helpers, but it remains metadata-only, excluded from `/v1/models`, and `/v1/chat/completions` refuses it rather than attempting public/runtime tokenizer execution or generation:

```bash
curl -fsS http://127.0.0.1:8180/api/models/catalog/install \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"aladar/llama-2-tiny-random-GGUF","filename":"llama-2-tiny-random.gguf"}' | python3 -m json.tool
```

Its `fathom-download-manifest.json` records the exact Hugging Face repo, revision, MIT license, 1,750,560-byte size, and SHA256 verification.

Use the model id returned by `/v1/models`. For TinyStories GPT-2 10M today that id is usually `vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors`:

```bash
curl -fsS http://127.0.0.1:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors",
    "messages": [{"role": "user", "content": "Once upon a time, a robot learned to help people by"}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

`stream: true` is rejected for now. Non-streaming chat completions include Fathom metadata such as runtime lane and timing/token-rate metrics when generation runs, and assistant messages can persist/display those server-side metrics in the chat UI. Current generation metrics include `model_load_ms`, `generation_ms`, `total_ms`, `tokens_per_second`, `ttft_ms`, `prefill_ms`, `decode_ms`, `prefill_tokens_per_second`, `decode_tokens_per_second`, `runtime_cache_hit`, `runtime_cache_lookup_ms`, `runtime_residency`, and `runtime_family`. For the narrow GPT-2/TinyStories, verified Llama/SmolLM2, verified Qwen2, verified Phi, verified Mistral, and verified Gemma Candle lanes, Fathom can keep CPU/F32 model runtime state process-local across requests and reports `cold_loaded` vs `warm_reused`; KV cache, token buffers, sampling, and generated text remain request-local. Llama-family, Qwen2, Phi, Mistral, and Gemma cached generation are guarded by a per-entry lock, and Qwen2/Phi/Mistral/Gemma clear model-internal KV before and after each request, so this is serialized warm residency, not parallel batching or shared session memory. TTFT/prefill/decode are server-side non-streaming measurements, not client-observed streaming latency.

## Backend benchmarking

Use the dependency-free benchmark harness against a running backend to capture timings for any runnable chat models exposed by `/v1/models`:

```bash
python3 scripts/bench_backend.py --runs 3 --warmups 1 --cache-phase-report --markdown /tmp/fathom-bench.md
```

To include embedding timings, install a runnable embedding fixture and pass its embedding model id with `--embedding-model`. The default-build SafeTensors MiniLM fixture works without extra features; the ONNX fixture requires `FATHOM_FEATURES=onnx-embeddings-ort`.

If you want coarse RSS snapshots, pass the running `fathom-server` PID:

```bash
python3 scripts/bench_backend.py --pid "$(pgrep -n fathom-server)" --runs 3 --warmups 1
```

The harness prints JSON and can write Markdown. It records wall-clock latency, Fathom generation metrics (`model_load_ms`, TTFT/prefill/decode timing, prefill/decode token rates, and runtime cache fields when returned), embedding metrics (`tokenization_ms`, `inference_ms`, `pooling_ms`, `total_ms` when returned by the runtime), response usage/finish reason, embedding dimensions, and optional coarse `ps` RSS readings. With `--cache-phase-report`, the first successful same-process chat request for each model is labeled `cold_candidate` and later same-model requests are labeled `warm_candidate`, making GPT-2/TinyStories, Llama/SmolLM2, Qwen2, Phi, Mistral, and Gemma process-local cache behavior easy to inspect. The script does not restart the server, download models, or mark unsupported fixtures as runnable, so cold/warm evidence should note the server start state and fixture paths; see [`docs/benchmarks/2026-04-26-local-apple-silicon-candle-cache-six-family.md`](docs/benchmarks/2026-04-26-local-apple-silicon-candle-cache-six-family.md) for checked-in six-family local fixture smoke evidence.

## Retrieval index API

Fathom has a small developer retrieval surface for explicit caller-supplied vectors. It can:

- create/list persistent vector indexes
- add chunks with vectors
- search by explicit query vector
- optionally insert matching snippets into `/v1/chat/completions` through `fathom.retrieval`

By default retrieval APIs do **not** crawl documents or silently infer vectors during search/chat calls. They expect caller-supplied vectors. Fathom can generate vectors through `/api/embedding-models/:id/embed` for verified embedding packages: the default-build SafeTensors MiniLM fixture, and the pinned ONNX MiniLM fixture when built with `onnx-embeddings-ort`. Context assembly remains explicit/opt-in.

Current safety limits keep this developer surface small and predictable:

- max embedding dimension: 8,192
- max chunks per index: 50,000
- max chunk text: 20,000 characters
- max search results: 100
- max inserted chat retrieval context: 16,000 characters

Create an index, add a chunk, and search it:

```bash
curl -fsS http://127.0.0.1:8180/api/retrieval-indexes \
  -H "Content-Type: application/json" \
  -d '{"id":"notes","embedding_model_id":"external-demo-3d","embedding_dimension":3}' | python3 -m json.tool

curl -fsS http://127.0.0.1:8180/api/retrieval-indexes/notes/chunks \
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

curl -fsS http://127.0.0.1:8180/api/retrieval-indexes/notes/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.85,0.15,0.0],"top_k":1,"metric":"cosine"}' | python3 -m json.tool
```

Opt into retrieval during chat completion:

```bash
curl -fsS http://127.0.0.1:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors",
    "messages": [{"role": "user", "content": "Use the notes if relevant: why does Rust care about ownership?"}],
    "max_tokens": 32,
    "fathom.retrieval": {
      "index_id": "notes",
      "query_vector": [0.85, 0.15, 0.0],
      "top_k": 1,
      "metric": "cosine",
      "max_context_chars": 1000
    }
  }' | python3 -m json.tool
```

Fathom searches the existing index, prepends a bounded system message containing matched snippets, and returns retrieval metadata under `fathom.retrieval` in the response.

## SafeTensors MiniLM embeddings

Fathom includes a default-build catalog fixture for `sentence-transformers/all-MiniLM-L6-v2` at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. The catalog metadata pins `config.json`, `tokenizer.json`, `model.safetensors`, SentenceTransformers pooling metadata, expected file sizes, and SHA256 digests.

Install the fixture through the catalog before calling the embed endpoint:

```bash
curl -fsS http://127.0.0.1:8180/api/models/catalog/install \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"sentence-transformers/all-MiniLM-L6-v2","filename":"model.safetensors"}' | python3 -m json.tool

curl -fsS http://127.0.0.1:8180/api/embedding-models | python3 -m json.tool
```

Use the installed embedding model id with the embed endpoint:

```bash
EMBED_MODEL_ID=sentence-transformers-all-minilm-l6-v2-model-safetensors

curl -fsS http://127.0.0.1:8180/api/embedding-models/$EMBED_MODEL_ID/embed \
  -H "Content-Type: application/json" \
  -d '{"input":["Rust ownership keeps memory safety explicit."],"normalize":true}' \
  | python3 -m json.tool
```

This lane loads the verified BertModel/MiniLM SafeTensors package through Candle, tokenizes with `tokenizer.json`, mean-pools `last_hidden_state` over the attention mask, optionally L2-normalizes, and returns finite 384-dimensional vectors. It is embedding/retrieval-only and stays out of `/v1/models`. The same verified runtime is also available through the narrow OpenAI-style `POST /v1/embeddings` adapter with `input` as a string or array of strings and `encoding_format` omitted or set to `float`; base64, chat models, GGUF, PyTorch `.bin`, and unsupported packages are refused rather than faked.

## ONNX embeddings (`onnx-embeddings-ort`)

Fathom includes a pinned catalog fixture for `nixiesearch/all-MiniLM-L6-v2-onnx` at revision `1e6ba950da2d9627f0e297996bd2bdb5fdb521cc`. The catalog metadata pins `config.json`, `tokenizer.json`, and `model_quantized.onnx` (~23.7 MB total, with expected sizes and SHA256 digests).

ONNX embedding inference is real but intentionally **non-default** because it brings in ONNX Runtime binaries. Build or test with the feature when you want this lane:

```bash
cargo test --features onnx-embeddings-ort
FATHOM_PORT=8180 cargo run -p fathom-server --features onnx-embeddings-ort
```

Install the fixture through the catalog before calling the embed endpoint:

```bash
curl -fsS http://127.0.0.1:8180/api/models/catalog/install \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"nixiesearch/all-MiniLM-L6-v2-onnx","filename":"model_quantized.onnx"}' | python3 -m json.tool

curl -fsS http://127.0.0.1:8180/api/embedding-models | python3 -m json.tool
```

With that feature enabled, `POST /api/embedding-models/:id/embed` and the narrow `POST /v1/embeddings` adapter load `tokenizer.json`, tokenize one or more inputs with `input_ids`, `attention_mask`, and `token_type_ids` when the graph expects them, run `model_quantized.onnx` through ORT, mean-pool `last_hidden_state` over the attention mask, and return finite 384-dimensional vectors for the installed MiniLM model id. The legacy `/api/embedding-models/:id/embed` endpoint can optionally L2-normalize; the OpenAI-style adapter reports its normalization choice in `fathom.normalize`. Ordinary builds still return a truthful `501 embedding_runtime_unavailable` instead of fake vectors.

This remains narrow: it is verified for the pinned MiniLM embedding fixture, not arbitrary ONNX graphs, ONNX chat, ONNX LLM generation, or general ONNX model execution. ONNX embedding packages stay out of `/v1/models`.

## Storage and provenance

By default, Fathom stores model packages and state under `~/.fathom`:

- models: `~/.fathom/models`
- state: `~/.fathom/state`
- logs: `~/.fathom/logs`
- retrieval indexes: `~/.fathom/state/retrieval-indexes/*.json`

Catalog installs record provenance next to the installed package in `fathom-download-manifest.json`: repo id, pinned revision, source URL, license, install time, file sizes, SHA256 digests, and verification status. The same manifest is exposed in model metadata.

For isolated tests, set `FATHOM_MODELS_DIR` and `FATHOM_STATE_DIR` before starting the server.

## Verification

Run the standard smoke check:

```bash
bash scripts/smoke.sh
```

The smoke script runs Rust tests, builds the frontend, starts backend/frontend on temporary ports, checks health/capabilities, and verifies that non-runnable fixtures are not exposed as runnable models.

If you have already downloaded TinyStories GPT-2 10M, DistilGPT-2, SmolLM2, or another compatible package for a verified GPT-2/Llama/Qwen2/Phi/Mistral/Gemma lane, the smoke script can also register it and verify real `/v1/chat/completions` output. You can point it at a model directory explicitly:

```bash
FATHOM_SMOKE_RUNNABLE_MODEL_DIR="$HOME/.fathom/models/vijaymohan-gpt2-tinystories-from-scratch-10m" bash scripts/smoke.sh
```

## Project shape

- `crates/fathom-core`: artifact detection, capability reports, chat-template handling, custom Rust generation lanes, context strategy advice, and vector-index primitives.
- `crates/fathom-server`: HTTP API, catalog installation, OpenAI-compatible responses, retrieval-index endpoints, model registry/state, and frontend-facing data.
- `frontend`: local dashboard for chat, model catalog, runtime/capability status, memory/retrieval status, and API guidance.
- `scripts`: start/stop/smoke helpers for local development.

## Name

**Fathom**: a runtime that can look into different model containers and understand what they are before deciding how to run them.
