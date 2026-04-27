#!/usr/bin/env python3
"""Dependency-free Fathom API smoke example using urllib.

Requires a running Fathom backend. By default this installs and calls the tiny
random Phi SafeTensors fixture; output may be gibberish because the model is a
small random test fixture.
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
