# Llama 3 tokenizer parity reference (llama.cpp)

Date: 2026-04-28 22:27 America/Los_Angeles

Purpose: pin a bit-for-bit external tokenizer oracle before any Camelid/Fathom Llama 3 tokenizer implementation is blessed.

## Reference inputs

- Model: `/Volumes/SSK Drive/Camelid/models/Meta-Llama-3-8B-Instruct-Q8_0.gguf`
- Model size: `8,540,770,880` bytes
- Reference repo: `/Users/timtoole/.openclaw/workspace/projects/llama.cpp`
- Reference revision: `665abc6`
- Reference binary: `/Users/timtoole/.openclaw/workspace/projects/llama.cpp/build-debug-callback/bin/llama-tokenize`
- Reproduction script: `scripts/reference_llama3_tokenizer_ids.py`

## Expected token IDs

### `"The quick brown fox jumps over the lazy dog."`

- llama.cpp default mode: `[128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13]`
- `--no-bos --no-parse-special`: `[791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13]`

### `"<|begin_of_text|>hello how's it going?"`

- llama.cpp default mode: `[128000, 128000, 15339, 1268, 596, 433, 2133, 30]`
- `--no-bos --no-parse-special`: `[27, 91, 7413, 3659, 4424, 91, 29, 15339, 1268, 596, 433, 2133, 30]`

Note: default mode adds BOS `128000`; when the input already begins with `<|begin_of_text|>`, llama.cpp parses that special token too, yielding two leading `128000` IDs.

## Evidence

Validated with:

```bash
cmake --build /Users/timtoole/.openclaw/workspace/projects/llama.cpp/build-debug-callback --target llama-tokenize -j 4
scripts/reference_llama3_tokenizer_ids.py \
  --llama-tokenize /Users/timtoole/.openclaw/workspace/projects/llama.cpp/build-debug-callback/bin/llama-tokenize \
  --pretty
```

The Fathom test suite now includes pinned non-ignored constants plus an ignored external-oracle test (`gguf_tokenizer_llama3_encode_goldens_match_llama_cpp_when_configured`) that compares these IDs against llama.cpp when `FATHOM_GGUF_LLAMA3_FIXTURE` and `LLAMA_TOKENIZE_BIN` are available.

Status: Camelid/Fathom Llama 3 tokenizer is **not blessed** by this artifact; these are reference IDs only. Any implementation must match these exactly before promotion.
