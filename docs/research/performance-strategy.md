# Fathom Performance Strategy

Date: 2026-04-26
Owner lane: Fathom Performance / Benchmark Agent

## Executive recommendation

Treat performance as a product truth signal, not a vanity number. For the first SafeTensors/HF backend, Fathom should ship with measured latency and throughput for the exact backend/model/device combination that is runnable, while keeping broad speed claims out of the UI/API until they are backed by repeatable benchmarks.

Recommended first path:

1. **Candle first, behind explicit device/backend capability flags.** Start with CPU and Apple Metal on a local Apple Silicon macOS development host; leave CUDA as a declared Linux/NVIDIA lane until a CUDA host verifies it.
2. **Measure load, tokenizer, prefill, decode, and memory separately.** Do not collapse everything into one `tokens/sec` number.
3. **Use tiny public fixtures for CI, optional real-model smoke for local demos.** Tiny random models catch regressions in the loop; real models establish user-facing expectations.
4. **Compare against llama.cpp/Ollama only when artifacts and hardware are honest.** Same model family, same prompt/output counts, same quantization or clearly labeled different precision.
5. **Prefer simple harnesses before optimization.** A small Rust benchmark CLI plus JSON output will be more useful than premature kernel work.

## Performance principles

- **Truth first:** every measurement must include model id/path, parameter count when known, precision/quantization, backend, device, thread count, prompt tokens, generated tokens, batch settings, KV-cache setting, build profile, git commit, OS, CPU/GPU/RAM, and thermal/power notes when relevant.
- **Separate cold and warm paths:** cold load includes metadata parse, mmap/open, weight transfer, tokenizer load, and first graph setup. Warm generation starts from an already-loaded model.
- **Measure user latency and engine throughput:** TTFT matters to chat feel; output tokens/sec matters to streaming feel; load time and resident memory matter to local-product trust.
- **No unsupported claims:** if Candle can run Llama on Metal but Fathom has only wired GPT-2 on CPU, Fathom reports only GPT-2/CPU as measured.
- **Shareable evidence hygiene:** checked-in benchmark notes should use generic host labels and home-relative paths rather than usernames, hostnames, or absolute local directories.
- **Regression gates start as warnings:** early gates should catch 2x regressions and gross memory blowups without blocking useful architecture iteration.

## Backend/device strategy

### Candle CPU

Candle is a good first runtime because it is Rust-native, Hugging Face-maintained, has examples for many transformer families, supports SafeTensors, and exposes CPU/CUDA/Metal-oriented device selection. The Candle README describes it as a minimalist Rust ML framework focused on performance and GPU support, with command-line examples for LLaMA, Mistral, Gemma, Phi, Qwen, GPT-style models, and quantized lanes.

Practical CPU plan:

- Build the first harness on `Device::Cpu` because it is the most portable and easiest to test in CI.
- Compile release builds for all speed measurements; debug numbers are invalid for benchmark docs.
- Record `RAYON_NUM_THREADS`, physical/logical CPU count, and whether Candle is using default CPU kernels, Accelerate, or MKL.
- On macOS/Apple Silicon, evaluate Candle's optional Accelerate path only after the plain CPU path is stable. Report it as `cpu-accelerate`, not generic CPU.
- Avoid CPU micro-optimizations until the harness proves where time is going: tokenizer, load, prefill, decode, or sampler.

Risks:

- Candle model implementations and APIs can move quickly. Pin crate versions when adding the backend and include them in benchmark metadata.
- CPU performance on F32/F16 SafeTensors tiny models will not predict quantized llama.cpp performance. Label the precision.

### Candle Metal

Metal is the most relevant acceleration lane for a local Apple Silicon macOS development host. Candle exposes Metal-related crates/features and examples, but Fathom should treat Metal support as a separate capability gate because model coverage, dtype support, kernel coverage, and memory pressure can differ from CPU.

Practical Metal plan:

- Add Metal only after the CPU path has a repeatable correctness and benchmark baseline.
- Gate by OS, architecture, device initialization success, supported dtype, and successful fixture generation.
- Measure first-token warmup separately; GPU command-pipeline initialization can distort the first request.
- Track both host resident memory and GPU/device allocation where available. If GPU allocation is not directly observable, state that clearly and use process RSS plus backend logs as a proxy.
- Prefer one loaded model per process for milestone 1; multi-model residency and eviction can wait.

Risks:

- Metal speed can be excellent for larger matmuls but overhead may dominate tiny fixtures. Do not use tiny fixtures for product speed claims.
- F16/BF16 support and tensor transfer behavior must be verified per architecture.

### Candle CUDA

CUDA should be a future measured lane, not assumed support from a Mac-only development loop. Candle documents CUDA device selection and has CUDA dependencies/features, but Fathom should not claim CUDA readiness until a Linux/NVIDIA host runs the same harness.

Practical CUDA plan:

- Define the metadata now: GPU model, driver, CUDA toolkit/runtime, Candle feature set, VRAM, power cap, and whether NCCL/multi-GPU is in use.
- Keep first CUDA tests single-GPU and single-model.
- Measure host-to-device weight load separately from generation.
- Add CUDA gates only in an optional benchmark profile or dedicated CI runner; do not burden standard local tests.

### ONNX Runtime and future GGUF lanes

ONNX Runtime can be a strong later backend for exported ONNX artifacts, especially where provider acceleration is mature. It is not the first SafeTensors/HF path because it requires ONNX artifacts or conversion and a separate generation/KV-cache contract.

GGUF/llama.cpp-style performance is an important baseline, not a source-code shortcut. Fathom should stay clean-room and compare against llama.cpp/Ollama only as external executables with explicit model files and command lines.

## Tokenizer performance strategy

Use Hugging Face `tokenizers` for the first HF lane. Its README states that it is Rust-native, production-oriented, and can tokenize a GB of text in under 20 seconds on a server CPU. For interactive chat, tokenization will usually be smaller than model decode time, but it can dominate tiny-model benchmarks and high-throughput batch tests.

Implementation guidance:

- Load `tokenizer.json` once per loaded model; never reload it per request.
- Measure tokenizer load time and encode time separately from model load/generation.
- For a single chat request, use normal `encode`; for benchmark batches, use `encode_batch` so tokenizer parallelism is visible.
- Record `TOKENIZERS_PARALLELISM` and `RAYON_NUM_THREADS`. Avoid nested parallelism surprises by making benchmark mode explicit.
- Decode incrementally for streaming, but benchmark both:
  - per-token decode cost for streaming, and
  - full-output decode cost for non-stream responses.
- Chat-template rendering should be measured as prompt-format time, not tokenizer time. It is usually small, but malformed templates can become latency traps.

Metrics to expose internally:

- tokenizer_load_ms
- chat_template_render_ms
- encode_ms
- input_tokens
- decode_text_ms or incremental_decode_ms
- tokenizer_errors/blockers

## SafeTensors mmap / zero-copy strategy

SafeTensors is attractive because it is safe relative to pickle and designed for fast, zero-copy/lazy access. The SafeTensors README describes the format as safe and fast/zero-copy, with a JSON header followed by contiguous tensor bytes; it explicitly disallows holes in the byte buffer. Candle's docs expose `MmapedSafetensors`, and the dependency list includes `memmap2` and `safetensors`.

Practical loading plan:

- During package inspection, parse only headers and shard indexes. Do not allocate all tensors just to classify capability.
- During model load, memory-map SafeTensors files where the backend allows it.
- Validate file size, header size, offsets, dtype, shapes, shard paths, and duplicate/unknown tensor behavior before mapping/using tensors.
- Keep the mmap object alive for as long as borrowed tensor views need it. Treat lifetime management as part of the backend design, not a helper detail.
- Separate measurements:
  - metadata_scan_ms: JSON/config/index/header parsing without tensor materialization
  - mmap_open_ms: map/open SafeTensors files
  - tensor_bind_ms: constructing Candle variables/model from mapped tensors
  - device_transfer_ms: CPU-to-GPU transfer when applicable
  - total_load_ms
- For GPU lanes, be honest: mmap can avoid extra CPU copies and speed open time, but weights still need to reach device memory before GPU inference.

Memory metrics:

- model_file_bytes and shard_count
- expected_tensor_bytes by dtype/shape
- process RSS before load, after mmap, after model construction, after first generation
- peak RSS during load when measurable
- GPU/device memory when backend/platform exposes it; otherwise mark unavailable

## Generation loop, batching, and KV cache

For chat workloads, the key split is **prefill** then **decode**:

- Prefill processes the input prompt and populates attention/KV state.
- Decode generates one or more output tokens using the existing KV cache.

Candle has `candle_nn::kv_cache::KvCache` with methods such as `append`, `current_seq_len`, `reset`, and accessors for key/value caches. Fathom should use model-family implementations that support KV cache before claiming acceptable chat performance.

Milestone 1 generation policy:

- Require KV cache for any real chat-speed claim. If the first tiny GPT-2 path runs without cache, label benchmark as `no_kv_cache` and do not compare it to optimized llama.cpp generation.
- Greedy decoding first for deterministic benchmark gates; temperature/top-p/top-k benchmarks can follow after sampler tests.
- Use fixed prompt fixtures and fixed max-new-token counts.
- Explicitly separate:
  - prompt_tokens_per_sec / prefill_ms
  - time_to_first_token_ms
  - decode_tokens_per_sec
  - end_to_end_ms
- Do not batch unrelated user requests in milestone 1. Add single-request correctness first.

Future batching plan:

1. **Phase A: no batching, one loaded model, one request at a time.** Establish correctness and latency.
2. **Phase B: queued requests with honest busy/loading states.** Measure queue wait separately.
3. **Phase C: prefill batching.** Batch prompts of similar lengths; record padding waste.
4. **Phase D: decode batching / continuous batching.** Only after KV-cache layout and scheduler ownership are clear.
5. **Phase E: multi-model residency and eviction.** Measure load/unload and memory pressure, not just tok/sec.

## Metrics definitions

Use these names consistently in benchmark JSON and docs:

| Metric | Definition | Notes |
| --- | --- | --- |
| `metadata_scan_ms` | Time to inspect package/config/tokenizer headers without model load | Should stay cheap enough for UI/catalog use. |
| `model_load_ms` | Time from load request start to backend model ready | Include tokenizer load only if `include_tokenizer_load=true`; otherwise report separately. |
| `tokenizer_load_ms` | Time to parse/build tokenizer object | Once per loaded model. |
| `prompt_format_ms` | Time to render chat template / build final prompt string | Separate from encode. |
| `encode_ms` | Time to tokenize final prompt | Include `input_tokens`. |
| `prefill_ms` | Time for first forward over input tokens | Equivalent to prompt-processing latency. |
| `prompt_tokens_per_sec` | `input_tokens / prefill_ms` | Report only when input token count is meaningful. |
| `ttft_ms` | Request start or generation start to first emitted token, depending on scope | Always include `ttft_scope`: `request`, `loaded_model`, or `decode`. |
| `decode_ms` | Time to generate output tokens after first token path begins | Exclude prompt formatting/encode. |
| `decode_tokens_per_sec` | Generated output tokens divided by decode time | Output-only tokens/sec. |
| `end_to_end_ms` | HTTP request start to response complete | User-visible non-stream latency. |
| `rss_delta_mb` | Process RSS increase after load/generate | Include measurement method. |
| `peak_rss_mb` | Peak process RSS during run | Best effort; can be unavailable. |
| `tokens_generated` | Number of output tokens actually produced | Required for all throughput metrics. |

TTFT scopes matter:

- `request`: includes HTTP, routing, prompt formatting, tokenization, prefill, sampler, first output serialization.
- `loaded_model`: starts after request is routed to an already-loaded model.
- `decode`: starts after prefill and measures first next-token step. Useful for engine profiling but not user latency.

## Benchmark harness design

Add a small benchmark CLI once the Candle backend exists. Suggested shape:

```text
cargo run --release --bin fathom-bench -- \
  --model /path/to/hf-folder \
  --backend candle \
  --device cpu|metal|cuda:0 \
  --prompt-file benches/prompts/chat-short.txt \
  --max-new-tokens 128 \
  --repetitions 5 \
  --warmup 1 \
  --output-json target/fathom-bench.json
```

Harness responsibilities:

- Load model once and record cold load metrics.
- Run one warmup generation that is excluded from aggregate stats but retained in raw records.
- Run N measured repetitions with deterministic greedy decoding by default.
- Emit JSON Lines or JSON with raw per-run metrics and aggregate mean/min/max/stddev/p50/p95.
- Include full benchmark context: git commit, build profile, OS, architecture, CPU/GPU, RAM, backend crate versions, env vars, model metadata, prompt hash, and command line.
- Never download models implicitly. If a model path is missing, fail with instructions.
- Support `--no-network` as the default posture.

Recommended files later:

```text
benches/prompts/chat-short.txt
benches/prompts/chat-medium.txt
benches/prompts/completion-short.txt
scripts/bench-local.sh
docs/benchmarks/YYYY-MM-DD-<machine>.md
```

CI strategy:

- CI should run tiny fixtures for correctness and gross regressions only.
- Do not require large public model downloads in CI.
- Keep local real-model benchmark scripts opt-in.

## Fair baselines vs llama.cpp and Ollama

llama.cpp's `llama-bench` reports prompt-processing (`pp`), text-generation (`tg`), and combined (`pg`) tests, with output formats including JSON/CSV/markdown. That split is the right inspiration for Fathom's harness.

Fair-comparison rules:

1. **Same artifact class when possible:** compare HF SafeTensors F16/BF16 Candle against an equivalent precision baseline if one exists. Do not compare F32 tiny GPT-2 to Q4 GGUF and imply engine superiority/inferiority.
2. **Same model family and size:** if exact weights differ due to conversion, document the conversion source, quantization, and tool versions.
3. **Same prompt/output counts:** align prompt token count and generated token count. Record tokenizer differences if token counts diverge.
4. **Same hardware state:** same machine, plugged in, similar thermal state, no heavy background jobs, same OS power mode where possible.
5. **Separate load from generation:** Ollama may keep models resident; llama.cpp commands may cold-load; Fathom must report both cold and warm to avoid misleading comparisons.
6. **No code borrowing:** external tools are baselines only; Fathom remains clean-room.

Suggested baseline commands to document when applicable:

```bash
# llama.cpp prompt/decode benchmark; use exact local path and record llama.cpp commit/version.
llama-bench -m /path/to/model.gguf -p 512 -n 128 -r 5 -o json

# Ollama API end-to-end smoke; record whether model was already loaded.
time curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"MODEL","prompt":"...","stream":false,"options":{"num_predict":128,"temperature":0}}'
```

Ollama comparisons should be labeled API/product-level, not engine-level, unless internals and residency are controlled.

## First benchmark matrix

### CI / tiny fixture matrix

Purpose: correctness and regression smoke, not public speed claims.

- Model: selected tiny public SafeTensors/HF causal LM from the implemented narrow runnable lanes, or generated fixture when useful for deterministic regression tests.
- Backend/device: Candle CPU.
- Prompt: 16-64 token completion prompt.
- Output: 8-32 greedy tokens.
- Gate type: must produce real tokens and avoid gross regressions.

### Local Apple Silicon macOS development matrix

Purpose: demo readiness and Apple Silicon optimization direction.

- Model A: tiny fixture, CPU and Metal if available.
- Model B: smallest useful real causal LM supported by Fathom's implemented architecture.
- Prompts:
  - short chat: ~128 input tokens / 64 output tokens
  - medium chat: ~512 input tokens / 128 output tokens
- Repetitions: 1 warmup + 5 measured.
- Metrics: full set above.

### Future baseline matrix

Purpose: understand practical gap vs established local tools.

- llama.cpp: same-family GGUF where available, `llama-bench pp512 tg128 pg512,128` style.
- Ollama: same-family model if present, warmed and cold API calls labeled separately.
- Fathom: same prompt/output, CPU and Metal/CUDA lanes as verified.

## Optimization roadmap

1. **Instrumentation before tuning:** add timing spans around scan, load, tokenizer, prompt render, prefill, decode, sampler, HTTP serialization.
2. **Correct KV cache:** ensure model-family implementation uses cache for decode; benchmark with and without cache only for validation.
3. **Device specialization:** CPU baseline first, then Metal on Apple Silicon, CUDA on verified host.
4. **Memory discipline:** mmap headers/weights where possible; avoid per-request tokenizer/model allocation; avoid prompt string copies where easy.
5. **Sampler simplicity:** greedy and seeded sampling first; optimize after correctness.
6. **Warm residency:** keep loaded model handles and tokenizers alive; expose loading/loaded/busy truthfully.
7. **Batching later:** add queue metrics before actual batching; then prefill batching, then decode batching.
8. **Profile real bottlenecks:** use Instruments on macOS, `time`/RSS sampling, and backend spans before changing architecture.

## Sources consulted

- Hugging Face Candle README: <https://raw.githubusercontent.com/huggingface/candle/main/README.md>
- Candle `MmapedSafetensors` docs: <https://docs.rs/candle-core/latest/candle_core/safetensors/struct.MmapedSafetensors.html>
- Candle `KvCache` docs: <https://docs.rs/candle-nn/latest/candle_nn/kv_cache/struct.KvCache.html>
- Hugging Face Tokenizers README: <https://raw.githubusercontent.com/huggingface/tokenizers/main/README.md>
- Hugging Face SafeTensors README: <https://raw.githubusercontent.com/huggingface/safetensors/main/README.md>
- llama.cpp `llama-bench` docs: <https://www.mintlify.com/ggml-org/llama.cpp/api/tools/llama-bench>
- Existing Fathom docs: `docs/research/safetensors-first-backend-plan.md`, `docs/research/runtime-safety-policy.md`, and checked-in benchmark evidence under `docs/benchmarks/`.
