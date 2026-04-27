# API client examples

These examples call a running Fathom backend as a local API service. They are intentionally small and focus on the public HTTP surface, not runtime internals.

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

The install examples also accept `FATHOM_REPO_ID` and `FATHOM_FILENAME`.

## cURL quickstart

```bash
bash examples/api/curl-quickstart.sh
```

This dependency-light script checks health, installs the pinned tiny Phi fixture if needed, lists runnable chat models, then sends one non-streaming chat request.

## Python, no third-party dependencies

```bash
python3 examples/api/python-no-deps.py
```

This version uses only the Python standard library.

## OpenAI Python SDK

```bash
python3 -m pip install openai
python3 examples/api/openai-sdk.py
```

The SDK example points the OpenAI client at `FATHOM_BASE_URL + /v1`. It still uses Fathom's local backend and only demonstrates a non-streaming chat completion.

## `.http` file

Open [`examples/api/fathom.http`](../../examples/api/fathom.http) in an editor that supports REST Client-style `.http` requests and run the requests top to bottom.

## Current boundaries

- `stream: true` is not supported by Fathom's chat completions today; use non-streaming requests.
- GGUF packages are metadata/provenance-only right now and are not chat inference models.
- ONNX chat is not supported.
- Fathom does not claim arbitrary Hugging Face or OpenAI API parity; `/v1/models` only lists models the local backend has validated as runnable.
