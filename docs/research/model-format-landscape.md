# Model Format Landscape - Fathom

Fathom’s core product promise is not just “open files.” A model becomes runnable only when the runtime can handle all of these layers:

1. container format
2. tensor dtype / quantization scheme
3. architecture graph
4. tokenizer and chat template
5. sampler / generation loop
6. device backend and kernels
7. test fixtures and truthful capability reporting

## Initial support matrix

| Format / artifact | What it contains | Main missing pieces | Likely Rust lane | Initial Fathom status |
| --- | --- | --- | --- | --- |
| GGUF | self-describing metadata, bounded tokenizer metadata fields, quantized tensors | public/runtime tokenizer execution, runtime weight loading, dequant/kernels, architecture runtime, generation | custom parser first; evaluate GGUF crates before adopting | metadata readable / metadata-only / not runnable |
| SafeTensors | safe tensor container only | config, tokenizer, architecture runtime | `safetensors` + Candle-backed custom Rust lanes | narrow runnable for validated chat/embedding packages; otherwise metadata-readable/planned/blocked |
| `model.safetensors.index.json` | sharded SafeTensors map | shard resolver, HF repo layout, architecture/tokenizer/runtime gate | `safetensors` + `hf-hub` style resolver | metadata readable unless a validated runnable lane accepts the full package |
| PyTorch `.bin` | legacy weights, often pickle | security policy; no blind pickle loading | prefer SafeTensors alternative; treat as trusted import only | detected / blocked |
| ONNX | graph plus weights | tokenizer, text-generation wrappers, operator/device support | `ort` for production; `tract` for pure-Rust/edge | detected / planned |
| MLX | Apple-focused model layout | MLX conventions, Apple backend | evaluate MLX file conventions; may bridge through SafeTensors/NPZ | detected / planned |
| CoreML | Apple runtime package | platform runtime bridge | macOS-specific backend | detected / planned |
| TensorRT plan/engine | NVIDIA-specific compiled engine | exact GPU/CUDA/TensorRT compatibility | backend adapter, not portable core | detected / planned |
| tokenizer.json | tokenizer | chat template integration | Hugging Face `tokenizers` | metadata readable |
| sentencepiece `.model` | tokenizer | sentencepiece implementation/bindings | evaluate Rust sentencepiece options | metadata readable |
| config.json | architecture metadata | map model_type/architectures to backend | native JSON parser | metadata readable |

## Backend strategy

- **Core stays Rust and capability-driven.** Backends must declare exactly what they can load and run.
- **Do not pretend all formats are equal.** GGUF is a runtime format. SafeTensors is a weight container. ONNX is a graph. TensorRT is a hardware-specific compiled plan.
- **Avoid silent conversion as the default path.** Conversion can become an optional tool, but Fathom’s promise is direct understanding where practical.
- **Start with one real text model path.** The first real backend should prove the full loop: load → tokenize → forward → sample → stream/return tokens → tests.

## Candidate Rust ecosystem to evaluate

- `safetensors`: safe tensor container parsing.
- `tokenizers`: mature Rust tokenizer library used by Hugging Face.
- Candle: Rust ML framework from Hugging Face; attractive for SafeTensors/Transformers first backend.
- `ort`: Rust bindings to ONNX Runtime; strongest pragmatic ONNX path.
- `tract`: pure-Rust ONNX/TensorFlow inference; useful for smaller/edge graph models.
- GGUF parser crates: useful references to evaluate, but Fathom should avoid blindly outsourcing its core product identity.
- Native GGUF lane: Fathom safely reads the GGUF magic/header, bounded key/value metadata, bounded tensor descriptors (names, shapes, GGML type tags, offsets, safe element counts, and known byte estimates), validated internal payload ranges, tokenizer/architecture compatibility hints, and privately retained bounded tokenizer metadata for narrow synthetic GPT-2/BPE and Llama/SentencePiece shapes when present. The catalog includes one pinned tiny real GGUF metadata/provenance fixture (`aladar/llama-2-tiny-random-GGUF` / `llama-2-tiny-random.gguf`, revision `8d5321916486e1d33c46b16990e8da6567785769`, MIT, 1,750,560 bytes, SHA256 `81f226c62d28ed4a1a9b9fa080fcd9f0cc40e0f9d5680036583ff98fbcd035cb`). Internal tokenizer retention, private fixture-scoped Llama/SentencePiece encode/decode parity helpers, and private synthetic F32/F16/Q8_0/Q4_0 payload decode tests exist as readiness groundwork, but this lane is still metadata-only: it does not expose public/runtime tokenizer execution, load runtime weights, provide general GGUF dequantization or quantized kernels, map an architecture runtime, expose GGUF in `/v1/models`, or generate.

## First implementation recommendation

Historical note: the SafeTensors/HF recommendation has now produced narrow real Candle lanes for validated GPT-2/TinyStories, Llama/SmolLM2, Qwen2, Phi, Mistral, Gemma, and MiniLM embedding packages. The strategic rule still holds: SafeTensors is a container, not a blanket support claim, so unvalidated packages remain metadata-readable, planned, blocked, or unsupported.

Build toward two lanes in parallel:

1. **Format intelligence lane**: robust package scanner for HF directories, single-file GGUF, ONNX, CoreML, TensorRT, MLX markers.
2. **First runnable lane**: SafeTensors + tokenizer + config using a Rust ML backend such as Candle, because it aligns with the “no GGUF conversion” product promise.

GGUF should still be supported, but making SafeTensors work first differentiates Fathom from llama.cpp-style workflows.
