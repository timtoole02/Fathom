# API client examples

These examples call a running Fathom backend as a local API service. They are intentionally small and focus on the public HTTP surface, not runtime internals. The exact narrow `/v1` request/response contract is documented in [`v1-contract.md`](v1-contract.md).

In CI, the cURL, standard-library Python, SDK, and `.http` examples are checked with a fake loopback server/static assertions so request shapes do not drift. That contract test does not prove real downloads, model loading, generation, or embedding runtime behavior; the separate backend acceptance smoke remains the real-backend gate.

Start the backend first:

```bash
bash scripts/start-backend.sh
```

By default, the examples install and use this tiny SafeTensors/Hugging Face chat fixture because it is small and useful for fast real API smoke tests:

- `repo_id`: `echarlaix/tiny-random-PhiForCausalLM`
- `filename`: `model.safetensors`
- expected model id: `echarlaix-tiny-random-phiforcausallm-model-safetensors`

The model is random and tiny, so generated text may be gibberish or whitespace. Use it to verify that install, `/v1/models`, and non-streaming `/v1/chat/completions` are wired up. Use a trained runnable model from the catalog when you want more language-like output.

## Environment variables

All examples default to `http://127.0.0.1:8180` and can be configured with:

```bash
export FATHOM_BASE_URL=http://127.0.0.1:8180
export FATHOM_MODEL_ID=echarlaix-tiny-random-phiforcausallm-model-safetensors
export FATHOM_PROMPT='Say hello from Fathom.'
export FATHOM_MAX_TOKENS=24
```

The install examples also accept `FATHOM_REPO_ID` and `FATHOM_FILENAME`. The default example model is permissive and does not send a license acknowledgement. If you point an example at a catalog entry whose `/api/models/catalog` metadata says `license_acknowledgement_required: true`, acknowledge that status in your own client flow and include `"accept_license": true` on the install request.

Optional embeddings smoke examples are disabled by default so the quickstarts do not download an extra model. To opt in:

```bash
export FATHOM_RUN_EMBEDDINGS=1
export FATHOM_EMBEDDING_MODEL_ID=sentence-transformers-all-minilm-l6-v2-model-safetensors
export FATHOM_EMBEDDING_REPO_ID=sentence-transformers/all-MiniLM-L6-v2
export FATHOM_EMBEDDING_FILENAME=model.safetensors
export FATHOM_EMBEDDING_INPUT='Rust ownership keeps memory safety explicit.'
```

That path installs the pinned MiniLM SafeTensors fixture if needed and calls `POST /v1/embeddings` with `encoding_format: "float"`. It verifies the narrow local embedding lane only: OpenAI-style response shape, float vectors, the `candle-bert-embeddings` runtime, and 384-dimensional MiniLM output. It does not make embedding models chat-runnable or list them in `/v1/models`.

## cURL quickstart

```bash
bash examples/api/curl-quickstart.sh
```

This dependency-light script checks health, installs the pinned tiny Phi fixture if needed, lists runnable chat models, then sends one non-streaming chat request. With `FATHOM_RUN_EMBEDDINGS=1`, it also installs the pinned MiniLM embedding fixture and sends one float `/v1/embeddings` request.

## Python, no third-party dependencies

```bash
python3 examples/api/python-no-deps.py
```

This version uses only the Python standard library. With `FATHOM_RUN_EMBEDDINGS=1`, it also exercises the optional MiniLM `/v1/embeddings` path.

## OpenAI Python SDK

```bash
python3 -m pip install openai
python3 examples/api/openai-sdk.py
```

The SDK example points the OpenAI client at `FATHOM_BASE_URL + /v1`. It still uses Fathom's local backend and only demonstrates a non-streaming chat completion by default. With `FATHOM_RUN_EMBEDDINGS=1`, it installs the pinned MiniLM fixture through Fathom's local catalog endpoint and calls `client.embeddings.create(..., encoding_format="float")` against `/v1/embeddings`.

## `.http` file

Open [`examples/api/fathom.http`](../../examples/api/fathom.http) in an editor that supports REST Client-style `.http` requests and run the requests top to bottom. The MiniLM install and embeddings requests are optional; skip them if you do not want the extra model download.

## Current boundaries

For the full supported `/v1` endpoint list and standard error envelope, see [`v1-contract.md`](v1-contract.md).

- `stream: true` is not supported by Fathom's chat completions today; use non-streaming requests.
- `/v1/embeddings` currently supports the verified local MiniLM embedding runtime with float vectors only; `encoding_format: "base64"` is refused with `invalid_request`.
- Embedding models are not chat/generation models and remain excluded from `/v1/models`.
- GGUF packages are metadata/provenance-only right now and are not chat inference or embedding models.
- ONNX chat and general ONNX model execution are not supported.
- PyTorch `.bin` and arbitrary SafeTensors execution are not supported.
- Fathom does not claim arbitrary Hugging Face or full OpenAI API parity; `/v1/models` only lists models the local backend has validated as chat-runnable.
