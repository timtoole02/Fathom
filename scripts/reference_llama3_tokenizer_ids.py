#!/usr/bin/env python3
"""Dump pinned llama.cpp tokenizer ID references for the local Llama 3 GGUF.

This is a reproducible oracle helper for Fathom/Camelid tokenizer parity work.
It intentionally calls llama.cpp's `llama-tokenize` instead of any Fathom/Camelid
implementation so the resulting IDs can be used as an external reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROMPTS = [
    {
        "label": "quick_brown_fox",
        "text": "The quick brown fox jumps over the lazy dog.",
    },
    {
        "label": "begin_text_hello_hows_it_going",
        "text": "<|begin_of_text|>hello how's it going?",
    },
]

DEFAULT_MODEL = Path("/Volumes/SSK Drive/Camelid/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf")


def parse_ids(output: str) -> list[int]:
    value = json.loads(output.strip())
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"llama-tokenize did not emit a JSON integer array: {output!r}")
    return value


def run_tokenize(binary: Path, model: Path, prompt: str, *, no_bos_no_special: bool) -> list[int]:
    command = [
        str(binary),
        "-m",
        str(model),
        "-p",
        prompt,
        "--ids",
        "--log-disable",
    ]
    if no_bos_no_special:
        command.extend(["--no-bos", "--no-parse-special"])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_ids(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-tokenize", required=True, type=Path, help="Path to llama.cpp llama-tokenize binary")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path, help="Path to Llama 3 GGUF")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    rows = []
    for prompt in PROMPTS:
        rows.append(
            {
                "label": prompt["label"],
                "text": prompt["text"],
                "default_ids": run_tokenize(args.llama_tokenize, args.model, prompt["text"], no_bos_no_special=False),
                "no_bos_no_special_ids": run_tokenize(args.llama_tokenize, args.model, prompt["text"], no_bos_no_special=True),
            }
        )

    payload = {
        "model": str(args.model),
        "model_size_bytes": args.model.stat().st_size,
        "reference_tool": str(args.llama_tokenize),
        "prompts": rows,
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
