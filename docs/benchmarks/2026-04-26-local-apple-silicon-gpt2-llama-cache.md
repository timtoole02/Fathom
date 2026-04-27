# Fathom GPT-2/Llama runtime cache smoke evidence

This is a local fixture smoke, not a throughput claim. The GPT-2 and Llama fixtures are tiny/random or smoke-scale packages intended to verify Fathom cache instrumentation and process-local residency semantics, not generation quality.


Path note: model paths in this checked-in evidence are intentionally home-relative so the document can be read without exposing a local username or host-specific directory layout.

## Environment

- Commit: `a5f4679`
- Host: local Apple Silicon macOS machine (`arm64`, Darwin 25.3.0)
- Rust: `rustc 1.94.1 (e408947bf 2026-03-25)`
- Backend URL: `http://127.0.0.1:8180`
- Build/start command: `bash scripts/stop.sh && bash scripts/start.sh`
- Benchmark command: `python3 scripts/bench_backend.py --model runnable-gpt2-smoke --model llama-runtime-cache-smoke --runs 1 --warmups 1 --max-tokens 8 --cache-phase-report --pid "$PID" --markdown docs/benchmarks/2026-04-26-local-apple-silicon-gpt2-llama-cache.md`
- Prompt: `Write one short paragraph about a small robot learning to garden.`
- Cache phase semantics: `cold_candidate` is the first successful request observed by the benchmark process after this backend restart; `warm_candidate` is later same-model traffic in the same server process.
- RSS caveat: `rss_*_kib` is sampled with `ps` around each request when `--pid` is supplied and is only a coarse process-level reading.

## Fixtures

- `runnable-gpt2-smoke` → `~/.fathom/models/distilbert-distilgpt2` (`runtime_family=gpt2`)
- `llama-runtime-cache-smoke` → `~/.fathom/models/stas-tiny-random-llama-2` (`runtime_family=llama`)

## Results

### Harness output

- Timestamp: `2026-04-27T02:10:44.229426+00:00`
- Server: `http://127.0.0.1:8180`
- Runs: 1 measured, 1 warmup

- Cache phase report: enabled (`cold_candidate` is the first successful request observed by this process; `warm_candidate` is later same-model traffic).

## Chat generation

### `runnable-gpt2-smoke`
- Wall ms median: `76.854`
- Model load ms median: `0`
- TTFT/prefill ms median: `19`
- Decode ms median: `55`
- Tokens/sec median: `105.39559578154127`
- Prefill tokens/sec median: `952.3928871289114`
- Decode tokens/sec median: `125.1010861455015`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'gpt2': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 1115.38 | 995 | 33 | 89.44456323800499 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 76.854 | 0 | 19 | 105.39559578154127 |

### `llama-runtime-cache-smoke`
- Wall ms median: `3.058`
- Model load ms median: `0`
- TTFT/prefill ms median: `0`
- Decode ms median: `1`
- Tokens/sec median: `3358.5222502099077`
- Prefill tokens/sec median: `111707.44323802643`
- Decode tokens/sec median: `5042.0180172912405`
- Runtime cache hits: `{'True': 1}`
- Runtime residency: `{'warm_reused': 1}`
- Runtime family: `{'llama': 1}`
- Runtime cache lookup ms median: `0`

| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| cold_candidate | 1 | `{'cold_loaded': 1}` | `{'False': 1}` | 6.139 | 1 | 1 | 3338.3171626640938 |
| warm_candidate | 1 | `{'warm_reused': 1}` | `{'True': 1}` | 3.058 | 0 | 0 | 3358.5222502099077 |

Note: chat TTFT/prefill/decode timings are server-side Fathom metrics from a non-streaming request, not client-observed streaming latency.
