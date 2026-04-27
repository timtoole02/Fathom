# SafeTensors/HF First Backend Plan

Date: 2026-04-26

Status update: this document is now historical planning context. The first
SafeTensors/HF milestone described here has been implemented in a narrow,
truthful form: real Candle CPU/F32 chat lanes exist for GPT-2/TinyStories,
Llama/SmolLM2, Qwen2, Phi, Mistral, and Gemma fixture/package shapes, plus a
default-build SafeTensors MiniLM embedding lane. Keep this page as design
rationale, not as the current delivery board. Current public behavior remains
architecture/package-gated; arbitrary `.safetensors` files are not supported.

Backend-only acceptance evidence for commit `93a2039` is captured outside the
product repo in private workspace acceptance notes. Public handoff artifacts should
prefer share-safe summaries and keep runner-local paths isolated from checked-in docs.

## Original recommendation

Build Fathom's first real non-GGUF runnable lane on **Candle + Hugging Face `tokenizers` + `safetensors` + explicit HF package scanning**.

This should be a separate backend adapter crate, e.g. `crates/fathom-backend-candle`, behind a feature flag at first. Keep `fathom-core` responsible for neutral capability contracts and package inspection; keep Candle types out of the core public API.

Initial dependency set:

- `candle-core`, `candle-nn`, `candle-transformers`: tensor runtime, NN layers, transformer model implementations.
- `safetensors`: direct secure tensor loading and indexed/sharded file support.
- `tokenizers`: `tokenizer.json` loading, encode/decode, special tokens.
- `hf-hub`: optional repo/file resolver and cache integration; do not require network for local folder registration.
- `minijinja`: render HF chat templates from `tokenizer_config.json` when present.
- `serde`, `serde_json`: parse `config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`.

Short version: **Candle is the best first lane because it is Rust-native, already centered on the HF artifact layout, directly reads SafeTensors, has transformer examples, and can prove the full loop without forcing conversion to GGUF or ONNX.**

## Why Candle first

| Candidate | Fit for first SafeTensors/HF backend | Strengths | Main problems |
| --- | --- | --- | --- |
| Candle | Best | Rust-native, HF-maintained, SafeTensors-first, includes many transformer model implementations, CPU/CUDA/Metal-ish ecosystem path, examples for generation | API/model coverage changes; not as battle-hardened as llama.cpp/vLLM; Fathom must wrap capabilities carefully |
| Burn | Not first | Clean Rust DL framework, good long-term option for training/custom graphs, multiple backend abstraction | Pretrained HF causal-LM inference path is less direct; fewer ready model loaders/generation examples; would require more Fathom-owned model code up front |
| `ort` / ONNX Runtime | Later ONNX lane | Production-grade graph runtime, strong hardware/provider story, good for exported ONNX artifacts | Not a direct SafeTensors/HF folder runtime; requires conversion/export or ONNX artifacts; generation loop and KV-cache contracts vary by export |
| `tract` | Later narrow ONNX/edge lane | Pure Rust, useful for small static graphs | Transformer LLM coverage/performance and dynamic generation are not the fastest path |
| Custom minimal runtime | Not first | Maximum control and clean-room ownership | Too much scope: kernels, attention, KV cache, tensor ops, model-specific loading, numerics, sampling, tests. High risk before product proves a first real path |

## Product boundary

For this lane, Fathom should say:

- **Runnable via Candle backend** only for architecture + dtype + tokenizer combinations actually exercised.
- **Metadata readable** for HF folders where Fathom can parse `config.json`, `tokenizer.json`, and SafeTensors headers but cannot run the architecture yet.
- **Blocked / needs alternate lane** for quantization or architectures Candle/Fathom cannot execute.

Do not label all `.safetensors` as supported. SafeTensors is a container, not a runtime contract.

## First runnable slice

Target the smallest causal-LM path that proves real tokens:

1. Local HF folder scanner
   - Detect `config.json`.
   - Detect `tokenizer.json`, `tokenizer_config.json`, optional `generation_config.json`.
   - Detect `model.safetensors` or `model.safetensors.index.json` + shards.
   - Read SafeTensors headers and shard map without loading all tensors.
2. Capability classifier
   - Map `config.model_type` / `architectures[]` to backend support.
   - Start with GPT-2 family or Llama-family tiny random model, then add Mistral/Phi/Gemma only after fixture coverage.
   - Report unsupported dtype/quantization explicitly.
3. Tokenizer layer
   - Use `tokenizers::Tokenizer` for `tokenizer.json`.
   - Read BOS/EOS/PAD and added token config from tokenizer files.
   - Decode incrementally enough for streaming `/v1/chat/completions`.
4. Chat/template layer
   - If `tokenizer_config.json.chat_template` exists, render with `minijinja` using an HF-compatible message schema.
   - If no chat template exists, use plain completion prompt path and mark chat formatting as generic, not model-native.
5. Candle generation runtime
   - Load weights with Candle var builders from SafeTensors.
   - Run prefill + decode with KV cache where the Candle model implementation supports it.
   - Implement Fathom-owned sampling wrapper: greedy first, then temperature/top-p/top-k with seeded deterministic tests.
   - Stop on EOS, max tokens, or OpenAI stop strings.
6. Server/API integration
   - `/v1/models` includes only models classified runnable by the Candle backend as runnable local models.
   - `/v1/chat/completions` rejects metadata-only models with a concrete reason.
   - Stream later; non-streaming first is acceptable if truthfully reported.

## Suggested implementation shape

```text
crates/
  fathom-core/
    capability contracts, model package metadata, neutral generation structs
  fathom-backend-candle/
    CandleBackend
    HfCausalLmPackage
    HfTokenizer
    ChatTemplateRenderer
    CandleModel enum/adapters
  fathom-server/
    runtime registry, OpenAI-compatible API glue
```

Core traits will need to become more specific than the current `InferenceRuntime::generate(prompt)` surface:

- `inspect_package(path) -> PackageCapabilityReport`
- `load_model(package, device, load_options) -> LoadedModelHandle`
- `tokenize/messages_to_prompt`
- `generate(request) -> token stream or complete response`
- `unload(handle)`

Keep the backend registry explicit: Fathom should be able to explain “Candle backend can run GPT-2 SafeTensors on CPU, but this Llama variant is metadata-only because architecture X/tokenizer Y is not wired yet.”

## Test model recommendations

Use tiny random models for CI and smoke tests. They are not quality tests; they are functional-path tests.

Primary first fixture:

- `Intel/tiny-random-gpt2`
  - HF tags indicate `transformers`, `safetensors`, `gpt2`, `text-generation`.
  - About 6.96M F32 parameters, unsharded SafeTensors.
  - Good first target because GPT-2 causal LM is simpler than modern chat Llama variants and does not require gated weights.

Secondary chat-template fixtures after base path works:

- `llamafactory/tiny-random-Llama-3`
  - Tiny random Llama-3-style instruct fixture with SafeTensors and chat-template metadata.
  - Useful to verify chat template rendering, not output quality.
- `yujiepan/llama-2-tiny-random` or `stas/tiny-random-llama-2`
  - Tiny Llama-family fixtures for Llama config/tokenizer compatibility.

Do not use large real instruct models for CI. Add an optional local smoke script for a small real model only after the tiny random path passes.

## Acceptance criteria for first backend milestone

A model is `Runnable` in the SafeTensors/HF lane only when all are true:

1. Fathom parses the local HF package and reports architecture, tokenizer, weight files, dtype, and backend lane.
2. The Candle backend loads the weights without conversion.
3. The tokenizer encodes a prompt and decodes generated token IDs.
4. `/v1/chat/completions` returns real generated tokens for the fixture.
5. Tests cover package scanning, tokenization, model load, one deterministic greedy generation, and API error behavior for unsupported SafeTensors packages.
6. UI/API distinguishes runnable from metadata-only packages.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Candle model coverage does not match arbitrary HF repos | Users may expect every SafeTensors model to run | Capability map must be architecture-specific; start with one family; label everything else metadata-only |
| Chat template compatibility differs from Python Transformers | Chat models may get malformed prompts | Use `minijinja`; fixture-test template rendering against known expected strings where possible |
| Tokenizer variants: SentencePiece, Tekken, custom tokenizers | Some popular models cannot tokenize through `tokenizer.json` alone | First milestone supports `tokenizer.json`; mark SentencePiece/custom tokenizer paths metadata-only until implemented |
| Quantized SafeTensors variants | Loading may fail or produce wrong results | Detect dtype/quantization metadata; only mark F32/F16/BF16 supported after exercised; block unknown quant schemes |
| Apple performance disappoints on CPU | Demo may be slow | Use tiny fixtures for CI; add Accelerate/Metal path as performance follow-up, not correctness gate |
| Dependency/API churn | Backend may need updates | Isolate in `fathom-backend-candle`; pin versions; keep core interfaces stable |
| ONNX may be faster for some models | Candle first may not be universal | Treat ONNX as a separate backend lane, not a replacement for direct HF/SafeTensors support |
| Random tiny fixtures do not prove quality | Generated text will be nonsense | Document fixtures as functional tests only; add optional quality smoke with a small real public model later |

## Follow-up lanes

After the first Candle milestone:

1. Expand Candle architecture matrix: GPT-2 -> Llama -> Mistral/Phi/Gemma, based on Candle support and available tiny fixtures.
2. Add streaming token responses and per-message performance metrics.
3. Add ONNX lane with `ort` for exported graph artifacts.
4. Revisit Burn only if Fathom needs training/custom graph ownership or Candle blocks key architectures.
5. Start native GGUF lane separately with parser/quantization work; do not mix it into Candle support claims.

## Sources checked

- Candle documentation: features include SafeTensors loading, language model examples, CPU/CUDA/WASM support, and many included transformer architectures: <https://huggingface.github.io/candle/>
- Candle crate docs show `candle-examples` depends on `candle-core`, `candle-transformers`, `safetensors`, `hf-hub`, `tokenizers`, and `minijinja`, including a chat template module: <https://docs.rs/candle-examples/latest/candle_examples/chat_template/index.html>
- HF model page for `Intel/tiny-random-gpt2` lists `transformers`, `safetensors`, `gpt2`, `text-generation`, and about 6.96M F32 parameters: <https://huggingface.co/Intel/tiny-random-gpt2>
- HF chat template docs describe the Jinja-style chat template mechanism used by tokenizer configs: <https://huggingface.co/docs/transformers/chat_templating>
- Existing Fathom docs: `docs/research/model-format-landscape.md` and `docs/architecture/rust-runtime-architecture.md`.
