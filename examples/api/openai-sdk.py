#!/usr/bin/env python3
"""Fathom chat completion example using the OpenAI Python SDK.

Install the optional SDK first:
  python3 -m pip install openai

Requires a running Fathom backend and an installed runnable chat model. The
pinned tiny random Phi fixture is the default model id used here; its output may
be gibberish because it is a small random smoke-test fixture.
"""

from __future__ import annotations

import os

from openai import OpenAI

ROOT_URL = os.environ.get("FATHOM_BASE_URL", "http://127.0.0.1:8180").rstrip("/")
MODEL_ID = os.environ.get(
    "FATHOM_MODEL_ID", "echarlaix-tiny-random-phiforcausallm-model-safetensors"
)
PROMPT = os.environ.get("FATHOM_PROMPT", "Say hello from a local Fathom API smoke test.")
MAX_TOKENS = int(os.environ.get("FATHOM_MAX_TOKENS", "24"))

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
