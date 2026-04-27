#!/usr/bin/env python3
"""Fathom chat completion example using the OpenAI Python SDK.

Install the optional SDK first:
  python3 -m pip install openai

Requires a running Fathom backend and an installed runnable chat model. The
pinned tiny random Phi fixture is the default model id used here; its output may
be gibberish because it is a small random smoke-test fixture.

Set FATHOM_RUN_EMBEDDINGS=1 to also install the pinned MiniLM embedding fixture
through Fathom's local catalog endpoint and call the OpenAI-style
/v1/embeddings route with encoding_format="float". Embedding models are not chat
models and remain excluded from /v1/models.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from openai import OpenAI

ROOT_URL = os.environ.get("FATHOM_BASE_URL", "http://127.0.0.1:8180").rstrip("/")
MODEL_ID = os.environ.get(
    "FATHOM_MODEL_ID", "echarlaix-tiny-random-phiforcausallm-model-safetensors"
)
PROMPT = os.environ.get("FATHOM_PROMPT", "Say hello from a local Fathom API smoke test.")
MAX_TOKENS = int(os.environ.get("FATHOM_MAX_TOKENS", "24"))
RUN_EMBEDDINGS = os.environ.get("FATHOM_RUN_EMBEDDINGS") == "1"
EMBEDDING_MODEL_ID = os.environ.get(
    "FATHOM_EMBEDDING_MODEL_ID", "sentence-transformers-all-minilm-l6-v2-model-safetensors"
)
EMBEDDING_REPO_ID = os.environ.get(
    "FATHOM_EMBEDDING_REPO_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_FILENAME = os.environ.get("FATHOM_EMBEDDING_FILENAME", "model.safetensors")
EMBEDDING_INPUT = os.environ.get(
    "FATHOM_EMBEDDING_INPUT", "Rust ownership keeps memory safety explicit."
)


def install_embedding_fixture_if_requested() -> None:
    if not RUN_EMBEDDINGS:
        return
    data = json.dumps(
        {"repo_id": EMBEDDING_REPO_ID, "filename": EMBEDDING_FILENAME}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{ROOT_URL}/api/models/catalog/install",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} installing embedding fixture: {body}") from exc


client = OpenAI(
    base_url=f"{ROOT_URL}/v1",
    api_key=os.environ.get("FATHOM_API_KEY", "fathom-local"),
)

completion = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=MAX_TOKENS,
)

print(completion.model_dump_json(indent=2))

install_embedding_fixture_if_requested()
if RUN_EMBEDDINGS:
    embeddings = client.embeddings.create(
        model=EMBEDDING_MODEL_ID,
        input=EMBEDDING_INPUT,
        encoding_format="float",
    )
    print(embeddings.model_dump_json(indent=2))
