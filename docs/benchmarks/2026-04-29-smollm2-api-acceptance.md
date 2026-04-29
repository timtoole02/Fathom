# SmolLM2 optional API acceptance evidence

This is optional local API evidence for the pinned catalog entry `hf-huggingfacetb-smollm2-135m-instruct`. It is not default CI, not a first-run requirement, not a quality/performance claim, and not broad SmolLM2, Llama-family, or arbitrary Hugging Face support.

## Scope

- Model repo: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- License metadata: `apache-2.0`
- Fathom lane: existing custom Rust/Candle `LlamaForCausalLM` SafeTensors/HF lane
- Catalog id: `hf-huggingfacetb-smollm2-135m-instruct`
- Installed `/v1` model id: `huggingfacetb-smollm2-135m-instruct-model-safetensors`
- Template lane: strict Llama-3 header-turn renderer
- Public contract: unchanged; default public CI remains no-download/offline for model fixtures.
- Evidence commit: `0a3c8d0f556321b295a76d1f95d1d0d4073433bd`

## Verified catalog files

The optional demo was installed through the catalog path and verified against these expected sizes and SHA256 hashes:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 861 | `8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a` |
| `generation_config.json` | 132 | `87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013` |
| `model.safetensors` | 269,060,552 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| `special_tokens_map.json` | 655 | `2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52` |
| `tokenizer.json` | 2,104,556 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `tokenizer_config.json` | 3,764 | `4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df` |

Total expected catalog bytes: `271,170,520`.

## Optional API acceptance smoke

The opt-in harness was run locally with preserved artifacts:

```bash
FATHOM_SMOLLM2_ACCEPTANCE=1 \
FATHOM_SMOLLM2_ACCEPTANCE_KEEP_ARTIFACTS=1 \
bash scripts/smollm2_optional_api_acceptance_smoke.sh
```

Artifact QA was then run against the preserved artifact directory:

```bash
python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py /path/to/artifacts
```

Result:

- Harness result: `passed`
- Artifact QA result: `passed`
- Installed model id: `huggingfacetb-smollm2-135m-instruct-model-safetensors`
- `/v1/models` included the model only after catalog install/verification.
- Two same-process non-streaming chat calls returned real local assistant content.
- Cold chat metrics included `runtime_family: llama`, `runtime_residency: cold_loaded`, `runtime_cache_hit: false`, and `model_load_ms: 1142`.
- Warm chat metrics included `runtime_family: llama`, `runtime_residency: warm_reused`, `runtime_cache_hit: true`, and `model_load_ms: 0`.
- `stream: true` was refused with `501 not_implemented` and no fake `choices`.

## Boundaries preserved

- SmolLM2 135M Instruct remains a larger optional demo, not a default/required model.
- This proves only the pinned package through an existing verified SafeTensors/HF Llama-style code path.
- `/v1/models` lists only validated local chat-runnable models after installation/inspection.
- `stream:true`, external placeholders, GGUF chat, ONNX chat/general execution, PyTorch `.bin`, unsupported templates, and arbitrary SafeTensors/Hugging Face execution remain refused or unclaimed.
- GGUF remains metadata-only; no public/runtime GGUF tokenizer execution, weight loading, generation, or inference is claimed here.
