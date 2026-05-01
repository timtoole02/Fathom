#!/usr/bin/env python3
"""Dependency-free Fathom API smoke example using urllib.

Requires a running Fathom backend. Fathom has no built-in authentication; keep
FATHOM_BASE_URL on loopback unless you have added your own access controls. By
default this installs and calls the tiny random Phi SafeTensors fixture; output
may be gibberish because the model is a small random test fixture.

Set FATHOM_RUN_EMBEDDINGS=1 to additionally install the pinned MiniLM
SafeTensors embedding fixture and call /v1/embeddings. That opt-in path downloads
an extra local model and only demonstrates float embeddings from Fathom's
verified local embedding runtime.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("FATHOM_BASE_URL", "http://127.0.0.1:8180").rstrip("/")
MODEL_ID = os.environ.get(
    "FATHOM_MODEL_ID", "echarlaix-tiny-random-phiforcausallm-model-safetensors"
)
REPO_ID = os.environ.get("FATHOM_REPO_ID", "echarlaix/tiny-random-PhiForCausalLM")
FILENAME = os.environ.get("FATHOM_FILENAME", "model.safetensors")
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


def request(method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} for {path}: {body}") from exc


def show(title: str, value: dict) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    show("Fathom health", request("GET", "/v1/health"))
    show(
        "Install pinned tiny Phi SafeTensors fixture, if not already present",
        request(
            "POST",
            "/api/models/catalog/install",
            {"repo_id": REPO_ID, "filename": FILENAME},
        ),
    )
    show("Runnable chat models", request("GET", "/v1/models"))
    show(
        "Non-streaming chat completion",
        request(
            "POST",
            "/v1/chat/completions",
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
            },
        ),
    )
    if RUN_EMBEDDINGS:
        show(
            "Install pinned MiniLM SafeTensors embedding fixture, if not already present",
            request(
                "POST",
                "/api/models/catalog/install",
                {"repo_id": EMBEDDING_REPO_ID, "filename": EMBEDDING_FILENAME},
            ),
        )
        show(
            "Float embeddings from verified local MiniLM runtime",
            request(
                "POST",
                "/v1/embeddings",
                {
                    "model": EMBEDDING_MODEL_ID,
                    "input": EMBEDDING_INPUT,
                    "encoding_format": "float",
                },
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
