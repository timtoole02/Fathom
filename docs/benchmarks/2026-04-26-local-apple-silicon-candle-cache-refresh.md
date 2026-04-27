# Fathom Candle runtime cache refresh evidence

This is local fixture smoke evidence, not a throughput, quality, or production-capacity claim. The GPT-2, Llama, Qwen2, Mistral, and Gemma fixtures used here are tiny/random or smoke-scale packages intended to verify Fathom cache instrumentation and process-local residency semantics.


Path note: model paths in this checked-in evidence are intentionally home-relative so the document can be read without exposing a local username or host-specific directory layout.

## Environment

- Commit under test: `9dd34ea Cache Gemma Candle runtime state`
- Host: local Apple Silicon macOS machine (`arm64`, Darwin 25.3.0)
- Backend URL: `http://127.0.0.1:8180`
- Build/start command: `bash scripts/stop.sh && bash scripts/start.sh`
- Benchmark command: `python3 scripts/bench_backend.py --model runnable-gpt2-smoke --model llama-runtime-cache-smoke --model yujiepan-qwen2-tiny-random-model-safetensors --model sanchit-gandhi-tiny-random-mistralforcausallm-1-layer-model-safetensors --model fxmarty-tiny-random-gemmaforcausallm-model-safetensors --runs 1 --warmups 1 --max-tokens 8 --temperature 0.8 --top-k 40 --top-p 1.0 --prompt 'Write three words about a green turtle.' --cache-phase-report --pid "$PID" --markdown /tmp/fathom-cache-refresh-bench.md`
- Cache phase semantics: `cold_candidate` is the first successful request observed by the benchmark process in the current server process; `warm_candidate` is later same-model traffic.
- RSS caveat: `rss_*_kib` is sampled with `ps` around each request when `--pid` is supplied and is only a coarse process-level reading.
- Tiny-random caveat: sampled decoding is used here so the random Gemma fixture produces non-empty output; do not treat text quality or token rates as representative.

## Fixtures

- `runnable-gpt2-smoke` → `~/.fathom/models/distilbert-distilgpt2` (`runtime_family=gpt2`)
- `llama-runtime-cache-smoke` → `~/.fathom/models/stas-tiny-random-llama-2` (`runtime_family=llama`)
- `yujiepan-qwen2-tiny-random-model-safetensors` → `~/.fathom/models/yujiepan-qwen2-tiny-random` (`runtime_family=qwen2`)
- `sanchit-gandhi-tiny-random-mistralforcausallm-1-layer-model-safetensors` → `~/.fathom/models/sanchit-gandhi-tiny-random-mistralforcausallm-1-layer` (`runtime_family=mistral`)
- `fxmarty-tiny-random-gemmaforcausallm-model-safetensors` → `~/.fathom/models/fxmarty-tiny-random-gemmaforcausallm` (`runtime_family=gemma`)

## Results

# Fathom backend benchmark

- Timestamp: `2026-04-27T02:50:56.998575+00:00`
- Server: `http://127.0.0.1:8180`
- Runs: 1 measured, 1 warmup

- Cache phase report: enabled (`cold_candidate` is the first successful request observed by this process; `warm_candidate` is later same-model traffic).

## Chat generation

### `runnable-gpt2-smoke`
- Wall ms median: `80.346`
- Model load ms median: `0`
- TTFT/prefill ms median: `18`
- Decode ms median: `60`
- Tokens/sec median: `100.82243761003588`
- Prefill tokens/sec median: `794.0937214821091`
- Decode tokens/sec median: `115.78293603990171`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'gpt2': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 323.818 | 205 | 21 | 95.5355063680386 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 80.346 | 0 | 18 | 100.82243761003588 |

### `llama-runtime-cache-smoke`
- Wall ms median: `2.609`
- Model load ms median: `0`
- TTFT/prefill ms median: `0`
- Decode ms median: `1`
- Tokens/sec median: `4245.5318430808975`
- Prefill tokens/sec median: `121956.04830726588`
- Decode tokens/sec median: `5586.7749862525425`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'llama': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 4.929 | 0 | 0 | 3793.5673426961735 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 2.609 | 0 | 0 | 4245.5318430808975 |

### `yujiepan-qwen2-tiny-random-model-safetensors`
- Wall ms median: `22.538`
- Model load ms median: `0`
- TTFT/prefill ms median: `2`
- Decode ms median: `18`
- Tokens/sec median: `368.2972659820757`
- Prefill tokens/sec median: `5609.523849591837`
- Decode tokens/sec median: `370.9731133526367`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'qwen2': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 118.435 | 1 | 3 | 365.4017857550671 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 22.538 | 0 | 2 | 368.2972659820757 |

### `sanchit-gandhi-tiny-random-mistralforcausallm-1-layer-model-safetensors`
- Wall ms median: `5.327`
- Model load ms median: `0`
- TTFT/prefill ms median: `0`
- Decode ms median: `3`
- Tokens/sec median: `1720.414568298523`
- Prefill tokens/sec median: `26614.069379921755`
- Decode tokens/sec median: `1761.5788578325332`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'mistral': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 30.501 | 1 | 0 | 1642.2467989017475 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 5.327 | 0 | 0 | 1720.414568298523 |

### `fxmarty-tiny-random-gemmaforcausallm-model-safetensors`
- Wall ms median: `42.504`
- Model load ms median: `0`
- TTFT/prefill ms median: `5`
- Decode ms median: `36`
- Tokens/sec median: `191.99404972041108`
- Prefill tokens/sec median: `2700.9993697668137`
- Decode tokens/sec median: `193.82818925317943`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'gemma': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 308.946 | 4 | 5 | 191.14082352737108 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 42.504 | 0 | 5 | 191.99404972041108 |

Note: chat TTFT/prefill/decode timings are server-side Fathom metrics from a non-streaming request, not client-observed streaming latency.
