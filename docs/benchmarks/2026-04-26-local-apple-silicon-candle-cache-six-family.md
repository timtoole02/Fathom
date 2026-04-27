# Fathom six-family Candle runtime cache evidence

This is local fixture smoke evidence, not a throughput, quality, production-capacity, or representative performance claim. The GPT-2, Llama, Qwen2, Phi, Mistral, and Gemma fixtures used here are tiny/random or smoke-scale packages intended to verify Fathom cache instrumentation, process-local runtime residency semantics, and the metrics now persisted for chat replies.


Path note: model paths in this checked-in evidence are intentionally home-relative so the document can be read without exposing a local username or host-specific directory layout.

## Environment

- Commit under test: `4caf72d5eaedeba6b8188a5eb6a074a92e431f1e` (`4caf72d Show runtime cache metrics in chat`)
- Host: local Apple Silicon macOS machine (`arm64`, `Darwin 25.3.0`)
- Backend URL: `http://127.0.0.1:8180`
- Build/start command: `bash scripts/stop.sh && bash scripts/start.sh`
- Server restart state: backend and frontend were stopped and restarted immediately before this benchmark; the first successful benchmark request for each model hit a fresh server-process runtime cache.
- Benchmark command:

```bash
PID=$(cat ~/.fathom/run/server.pid)
python3 scripts/bench_backend.py \
  --model runnable-gpt2-smoke \
  --model llama-runtime-cache-smoke \
  --model yujiepan-qwen2-tiny-random-model-safetensors \
  --model echarlaix-tiny-random-phiforcausallm-model-safetensors \
  --model sanchit-gandhi-tiny-random-mistralforcausallm-1-layer-model-safetensors \
  --model fxmarty-tiny-random-gemmaforcausallm-model-safetensors \
  --runs 1 \
  --warmups 1 \
  --max-tokens 8 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 1.0 \
  --prompt 'Write three words about a green turtle.' \
  --cache-phase-report \
  --pid "$PID" \
  --markdown /tmp/fathom-six-family-cache-bench.md
```

- Cache phase semantics: `cold_candidate` is the first successful request observed by the benchmark process in the current server process; `warm_candidate` is later same-model traffic.
- RSS caveat: `rss_*_kib` is sampled with `ps` around each request when `--pid` is supplied and is only a coarse process-level reading.
- Timing caveat: chat TTFT/prefill/decode timings are server-side Fathom metrics from non-streaming requests, not client-observed streaming latency.
- Tiny-random caveat: sampled decoding is used so tiny/random fixtures, especially Gemma, produce non-empty output; do not treat text quality or token rates as representative.
- Scope caveat: this does not claim batching, shared/session memory, GPU/Metal/CUDA, GGUF inference, ONNX chat, arbitrary SafeTensors support, or production throughput.

## Fixtures

- `runnable-gpt2-smoke` → `~/.fathom/models/distilbert-distilgpt2` (`runtime_family=gpt2`)
- `llama-runtime-cache-smoke` → `~/.fathom/models/stas-tiny-random-llama-2` (`runtime_family=llama`)
- `yujiepan-qwen2-tiny-random-model-safetensors` → `~/.fathom/models/yujiepan-qwen2-tiny-random` (`runtime_family=qwen2`)
- `echarlaix-tiny-random-phiforcausallm-model-safetensors` → `~/.fathom/models/echarlaix-tiny-random-phiforcausallm` (`runtime_family=phi`)
- `sanchit-gandhi-tiny-random-mistralforcausallm-1-layer-model-safetensors` → `~/.fathom/models/sanchit-gandhi-tiny-random-mistralforcausallm-1-layer` (`runtime_family=mistral`)
- `fxmarty-tiny-random-gemmaforcausallm-model-safetensors` → `~/.fathom/models/fxmarty-tiny-random-gemmaforcausallm` (`runtime_family=gemma`)

## Chat-visible metrics note

Commit `4caf72d` persists optional assistant-message runtime metrics from generation responses, including `runtime_cache_hit`, `runtime_residency`, `runtime_family`, `runtime_cache_lookup_ms`, `model_load_ms`, `generation_ms`, and `total_ms`. The chat UI can display compact cold/warm rows such as `Warm runtime · Phi · 0 ms load · 7 ms total`, but these rows are still server-side non-streaming timing fields from one local generation response, not aggregate analytics or production performance evidence.

## Results

# Fathom backend benchmark

- Timestamp: `2026-04-27T03:15:31.662428+00:00`
- Server: `http://127.0.0.1:8180`
- Runs: 1 measured, 1 warmup

- Cache phase report: enabled (`cold_candidate` is the first successful request observed by this process; `warm_candidate` is later same-model traffic).

## Chat generation

### `runnable-gpt2-smoke`
- Wall ms median: `200.539`
- Model load ms median: `0`
- TTFT/prefill ms median: `59`
- Decode ms median: `138`
- Tokens/sec median: `40.30874808062338`
- Prefill tokens/sec median: `252.05865122508317`
- Decode tokens/sec median: `50.37488811827305`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'gpt2': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 1397.914 | 966 | 187 | 24.583441874350758 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 200.539 | 0 | 59 | 40.30874808062338 |

### `llama-runtime-cache-smoke`
- Wall ms median: `4.198`
- Model load ms median: `0`
- TTFT/prefill ms median: `1`
- Decode ms median: `0`
- Tokens/sec median: `3356.5269344503854`
- Prefill tokens/sec median: `45493.72806921664`
- Decode tokens/sec median: `10132.078885471323`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'llama': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 17.63 | 1 | 4 | 670.9532753171218 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 4.198 | 0 | 1 | 3356.5269344503854 |

### `yujiepan-qwen2-tiny-random-model-safetensors`
- Wall ms median: `90.136`
- Model load ms median: `0`
- TTFT/prefill ms median: `47`
- Decode ms median: `40`
- Tokens/sec median: `90.79035878568031`
- Prefill tokens/sec median: `335.0606968734774`
- Decode tokens/sec median: `173.42812980976592`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'qwen2': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 345.381 | 12 | 22 | 74.58288890089052 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 90.136 | 0 | 47 | 90.79035878568031 |

### `echarlaix-tiny-random-phiforcausallm-model-safetensors`
- Wall ms median: `6.752`
- Model load ms median: `0`
- TTFT/prefill ms median: `1`
- Decode ms median: `3`
- Tokens/sec median: `1552.8327065388814`
- Prefill tokens/sec median: `13162.890773319832`
- Decode tokens/sec median: `2203.596584425294`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'phi': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 8.52 | 1 | 1 | 2530.210715948424 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 6.752 | 0 | 1 | 1552.8327065388814 |

### `sanchit-gandhi-tiny-random-mistralforcausallm-1-layer-model-safetensors`
- Wall ms median: `13.79`
- Model load ms median: `0`
- TTFT/prefill ms median: `5`
- Decode ms median: `4`
- Tokens/sec median: `877.8627378212457`
- Prefill tokens/sec median: `3580.0711757928316`
- Decode tokens/sec median: `1713.4986239382122`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'mistral': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 64.422 | 10 | 3 | 1022.5492561976711 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 13.79 | 0 | 5 | 877.8627378212457 |

### `fxmarty-tiny-random-gemmaforcausallm-model-safetensors`
- Wall ms median: `79.206`
- Model load ms median: `0`
- TTFT/prefill ms median: `15`
- Decode ms median: `61`
- Tokens/sec median: `103.73219759370605`
- Prefill tokens/sec median: `977.230528681716`
- Decode tokens/sec median: `113.31964442820987`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'gemma': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 1215.618 | 55 | 10 | 88.92108128434987 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 79.206 | 0 | 15 | 103.73219759370605 |

Note: chat TTFT/prefill/decode timings are server-side Fathom metrics from a non-streaming request, not client-observed streaming latency.
