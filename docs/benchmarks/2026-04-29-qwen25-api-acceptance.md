# Qwen2.5 optional API acceptance evidence

This is optional local API evidence for the pinned catalog entry `hf-qwen-qwen2-5-0-5b-instruct`. It is not default CI, not a first-run requirement, not a quality/performance claim, and not broad Qwen-family or arbitrary Hugging Face support.

## Scope

- Model repo: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- License metadata: `apache-2.0`
- Fathom lane: existing custom Rust/Candle `Qwen2ForCausalLM` SafeTensors/HF lane
- Catalog id: `hf-qwen-qwen2-5-0-5b-instruct`
- Installed `/v1` model id: `qwen-qwen2-5-0-5b-instruct-model-safetensors`
- Template lane: existing ChatML/Qwen-style renderer
- Public contract: unchanged; default public CI remains no-download/offline for model fixtures.
- Evidence commit: `cf5c380bd523aa605012904b6408a38cb2920ef2`

## Verified catalog files

The optional demo was installed through the catalog path and verified against these expected sizes and SHA256 hashes:

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

## Optional API acceptance smoke

The opt-in harness was run locally with preserved artifacts:

```bash
FATHOM_QWEN25_ACCEPTANCE=1 \
FATHOM_QWEN25_ACCEPTANCE_KEEP_ARTIFACTS=1 \
bash scripts/qwen25_optional_api_acceptance_smoke.sh
```

Artifact QA was then run against the preserved artifact directory:

```bash
python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py /path/to/artifacts
```

Result:

- Harness result: `passed`
- Artifact QA result: `passed`
- Installed model id: `qwen-qwen2-5-0-5b-instruct-model-safetensors`
- `/v1/models` included the model only after catalog install/verification.
- Two same-process non-streaming chat calls returned real local assistant content.
- Cold chat metrics included `runtime_family: qwen2`, `runtime_residency: cold_loaded`, `runtime_cache_hit: false`, and `model_load_ms: 6277`.
- Warm chat metrics included `runtime_family: qwen2`, `runtime_residency: warm_reused`, `runtime_cache_hit: true`, and `model_load_ms: 0`.
- `stream: true` was refused with `501 not_implemented` and no fake `choices`.

## Boundaries preserved

- Qwen2.5 0.5B Instruct remains a larger optional demo, not a default/required model.
- This proves only the pinned package through an existing verified SafeTensors/HF Qwen2 code path.
- `/v1/models` lists only validated local chat-runnable models after installation/inspection.
- `stream:true`, external placeholders, GGUF chat, ONNX chat/general execution, PyTorch `.bin`, unsupported templates, and arbitrary SafeTensors/Hugging Face execution remain refused or unclaimed.
- GGUF remains metadata-only; no public/runtime GGUF tokenizer execution, weight loading, generation, or inference is claimed here.
