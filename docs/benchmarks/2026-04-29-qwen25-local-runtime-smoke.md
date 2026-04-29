# Qwen2.5 0.5B Instruct local runtime smoke evidence

This is optional local evidence for the pinned catalog entry `hf-qwen-qwen2-5-0-5b-instruct`. It is not default CI, not a first-run requirement, not a quality/performance claim, and not broad Qwen or arbitrary Hugging Face support.

## Scope

- Model repo: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- License metadata: `apache-2.0`
- Fathom lane: existing custom Rust/Candle `Qwen2ForCausalLM` SafeTensors/HF lane
- Template lane: existing ChatML/Qwen-style renderer
- Public contract: unchanged; default public CI remains no-download/offline for model fixtures.

## Verified catalog files

The optional demo was checked locally against these expected sizes and SHA256 hashes before being added to the catalog:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 659 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| `generation_config.json` | 242 | `e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6` |
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `model.safetensors` | 988,097,824 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| `tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | 7,305 | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |

Total expected catalog bytes: `999,597,690`.

## Runtime smoke

A local live runtime smoke was run against the verified files with:

```bash
FATHOM_QWEN2_FIXTURE=/path/to/verified/qwen2.5-0.5b-instruct \
  cargo test -q -p fathom-core qwen2_runtime_cache_live_fixture_reports_cold_then_warm_when_configured -- --nocapture
```

Result:

- Test result: `1 passed; 0 failed`
- Filtered tests: `115`
- Rust test duration: `17.84s`
- Maximum resident set size from `/usr/bin/time -l`: `3,285,057,536` bytes

This smoke proves the pinned files can load through the existing Qwen2 lane and exercise cold/warm runtime cache reporting. It does not prove generation quality, latency, throughput, production capacity, legal suitability, or broad Qwen-family compatibility.

## Boundaries preserved

- Qwen2.5 0.5B Instruct remains a larger optional demo, not a default/required model.
- Tied-output Qwen2 checkpoints are accepted only when `tie_word_embeddings: true`; untied checkpoints missing `lm_head.weight` still fail closed.
- `/v1/models` lists only validated local chat-runnable models after installation/inspection.
- `stream:true`, external placeholders, GGUF chat, ONNX chat/general execution, PyTorch `.bin`, unsupported templates, and arbitrary SafeTensors/Hugging Face execution remain refused or unclaimed.
- GGUF remains metadata-only; no public/runtime GGUF tokenizer execution, weight loading, generation, or inference is claimed here.
