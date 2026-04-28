# Fathom `/v1` API contract

Fathom exposes a deliberately small OpenAI-style `/v1` surface for local integration tests. This is not full OpenAI API parity: endpoints only report or run models that Fathom has already validated for the specific local task.

Base URL in local development: `http://127.0.0.1:8180`.

Security note: Fathom's local API has no built-in authentication and is intended for loopback development, not direct internet exposure. See [`../../SECURITY.md`](../../SECURITY.md) before proxying, tunneling, or sharing logs/artifacts.

## Standard error envelope

All `/v1` application errors use this JSON shape:

```json
{
  "error": {
    "message": "Human-readable explanation.",
    "type": "invalid_request",
    "code": "invalid_request",
    "param": null
  }
}
```

Common status/code pairs include:

- `400 invalid_request` for malformed or unsupported request values.
- `400 model_not_found` when a required model id is missing, or when chat completion cannot resolve a requested/local active model.
- `404 embedding_model_not_found` when `/v1/embeddings` names a model Fathom does not know.
- `400 invalid_model` when a runnable model record is inconsistent.
- `501 not_implemented` when the requested feature or model family is intentionally not runnable.
- `501 chat_template_not_supported` for unsupported Hugging Face chat-template patterns.
- `501 embedding_runtime_unavailable` when an embedding package is recognized but the required runtime is not compiled/enabled.
- `500 generation_error` or `500 embedding_error` for runtime failures.

## `GET /v1/health`

Checks that the local Fathom API is responding and whether any chat-generation model is currently validated as runnable.

Response:

```json
{
  "ok": true,
  "engine": "fathom",
  "generation_ready": true
}
```

`generation_ready` is `true` when at least one local chat model is runnable. Embedding-only models and connected external API placeholders do not make chat generation ready.

## `GET /v1/models`

Lists runnable chat/generation models only.

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors",
      "object": "model",
      "created": 0,
      "owned_by": "fathom",
      "fathom": {
        "provider_kind": "local",
        "status": "available",
        "capability_status": "Runnable",
        "capability_summary": "Real local generation is available through a verified backend lane.",
        "backend_lanes": ["safetensors-hf"]
      }
    }
  ]
}
```

Important boundaries:

- Embedding-only models are excluded from `/v1/models` because they are not chat/generation models.
- Connected external OpenAI-compatible entries are excluded because Fathom only persists their metadata/key-local configuration today; external chat proxying is not implemented.
- Metadata-only GGUF packages are excluded.
- Blocked PyTorch `.bin`, unsupported packages, and planned-but-not-runnable formats are excluded.

## `POST /v1/chat/completions`

Runs one non-streaming local chat completion against a validated runnable local SafeTensors/Hugging Face chat model. Requests naming a connected external placeholder return a structured not-implemented error instead of proxying or faking a reply.

Request:

```json
{
  "model": "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors",
  "messages": [
    {"role": "user", "content": "Once upon a time"}
  ],
  "max_tokens": 32,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 40,
  "stream": false,
  "fathom.retrieval": {
    "index_id": "notes",
    "query_vector": [0.85, 0.15, 0.0],
    "top_k": 1,
    "metric": "cosine",
    "max_context_chars": 1000
  }
}
```

Fields:

- `model` is optional only when a runnable active model is already selected in Fathom state.
- `messages` must be non-empty. Supported roles are the roles accepted by the verified prompt renderer for the selected model/template.
- `stream` must be omitted or `false`; `true` returns `501 not_implemented`.
- `fathom.retrieval` is Fathom-specific and optional. It inserts explicit-vector retrieval context from an existing retrieval index; it does not crawl documents or infer vectors automatically.

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "vijaymohan-gpt2-tinystories-from-scratch-10m-model-safetensors",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 32,
    "total_tokens": 44
  },
  "fathom": {
    "runtime": "candle-safetensors-hf",
    "metrics": {
      "model_load_ms": 0,
      "generation_ms": 80,
      "total_ms": 81,
      "tokens_per_second": 400.0,
      "runtime_cache_hit": true,
      "runtime_cache_lookup_ms": 0,
      "runtime_residency": "warm_reused",
      "runtime_family": "gpt2"
    },
    "retrieval": null
  }
}
```

`fathom.metrics` is server-side Fathom metadata. Exact keys may vary by runtime and old conversations may not have all fields.

## `POST /v1/embeddings`

Returns OpenAI-style float embeddings for verified local MiniLM embedding runtimes only.

Request:

```json
{
  "model": "sentence-transformers-all-minilm-l6-v2-model-safetensors",
  "input": ["Rust ownership keeps memory safety explicit."],
  "encoding_format": "float"
}
```

Fields:

- `model` is required and must refer to an installed, verified embedding model.
- `input` may be one string or an array of strings.
- `encoding_format` may be omitted or set to `float`.
- `encoding_format: "base64"` is refused with `400 invalid_request`.

Response:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456],
      "index": 0
    }
  ],
  "model": "sentence-transformers-all-minilm-l6-v2-model-safetensors",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  },
  "fathom": {
    "runtime": "candle-bert-embeddings",
    "embedding_dimension": 384,
    "metrics": {
      "tokenization_ms": 1,
      "inference_ms": 10,
      "pooling_ms": 0,
      "total_ms": 11
    },
    "normalize": false,
    "scope": "verified local embedding runtime only"
  }
}
```

The feature-gated ONNX MiniLM runtime reports `fathom.runtime: "onnx-embeddings-ort"` when built with `onnx-embeddings-ort` and the pinned ONNX package validates. Ordinary builds refuse that runtime truthfully rather than returning fake vectors.

## Explicitly unsupported OpenAI features

Fathom intentionally does not support these OpenAI API features today:

- Streaming chat completions.
- Base64 embeddings.
- Tools, function calling, or structured tool invocation.
- Image, audio, or multimodal inputs/outputs.
- Fine-tuning, files, batches, assistants, responses API, or arbitrary model execution.
- Full OpenAI API parity.
- Turning embedding models into chat/generation models or listing them in `/v1/models`.
- No native GGUF chat/inference, no ONNX chat/LLM generation, no PyTorch `.bin` execution, and no arbitrary SafeTensors/Hugging Face execution.
