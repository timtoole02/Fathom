#!/usr/bin/env python3
"""Lightweight backend-only benchmark harness for a running Fathom server.

The harness intentionally uses only Python's standard library. It benchmarks the
OpenAI-compatible chat endpoint and runnable embedding endpoints when installed
fixtures are available, then prints JSON (and optionally Markdown) evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def request_json(base_url: str, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any], float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = resp.read().decode("utf-8")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return resp.status, json.loads(payload), elapsed_ms
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"error": {"message": payload}}
        return exc.code, parsed, elapsed_ms
    except urllib.error.URLError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return 0, {"error": {"message": f"could not connect to Fathom server: {exc}"}}, elapsed_ms


def maybe_rss_kib(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return int(output.splitlines()[-1].strip())
    except (IndexError, ValueError):
        return None


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def metric_values(samples: list[dict[str, Any]], name: str) -> list[float]:
    return [
        sample["metrics"][name]
        for sample in samples
        if sample.get("metrics") and sample["metrics"].get(name) is not None
    ]


def metric_value(sample: dict[str, Any], name: str) -> Any:
    metrics = sample.get("metrics")
    if not metrics:
        return None
    return metrics.get(name)


def counts(values: list[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        result[key] = result.get(key, 0) + 1
    return result


def runtime_cache_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.get("http_status") == 200]
    return {
        "runtime_cache_hit": counts([sample.get("runtime_cache_hit") for sample in successful]),
        "runtime_residency": counts([sample.get("runtime_residency") for sample in successful]),
        "runtime_family": counts([sample.get("runtime_family") for sample in successful]),
        "runtime_cache_lookup_ms": summarize(
            [
                sample["runtime_cache_lookup_ms"]
                for sample in successful
                if sample.get("runtime_cache_lookup_ms") is not None
            ]
        ),
    }


def cache_phase_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("cold_candidate", "warm_candidate"):
        phase_samples = [sample for sample in samples if sample.get("cache_phase") == phase and sample.get("http_status") == 200]
        phases[phase] = {
            "samples": len(phase_samples),
            "wall_ms": summarize([sample["wall_ms"] for sample in phase_samples]),
            "model_load_ms": summarize(metric_values(phase_samples, "model_load_ms")),
            "generation_ms": summarize(metric_values(phase_samples, "generation_ms")),
            "ttft_ms": summarize(metric_values(phase_samples, "ttft_ms")),
            "tokens_per_second": summarize(metric_values(phase_samples, "tokens_per_second")),
            "runtime_cache": runtime_cache_summary(phase_samples),
        }
    return phases


def pick_models(base_url: str, requested: list[str]) -> list[str]:
    if requested:
        return requested
    status, payload, _ = request_json(base_url, "GET", "/v1/models")
    if status != 200:
        raise SystemExit(f"Could not list /v1/models: HTTP {status} {payload}")
    return [item["id"] for item in payload.get("data", [])]


def pick_embedding_models(base_url: str, requested: list[str]) -> list[str]:
    if requested:
        return requested
    status, payload, _ = request_json(base_url, "GET", "/api/embedding-models")
    if status != 200:
        return []
    return [item["id"] for item in payload.get("items", []) if item.get("runnable")]


def bench_chat(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    successful_requests = 0
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    for run in range(args.warmups + args.runs):
        rss_before = maybe_rss_kib(args.pid)
        status, payload, wall_ms = request_json(args.base_url, "POST", "/v1/chat/completions", body)
        rss_after = maybe_rss_kib(args.pid)
        sample = {
            "run": run + 1,
            "warmup": run < args.warmups,
            "http_status": status,
            "wall_ms": round(wall_ms, 3),
            "rss_before_kib": rss_before,
            "rss_after_kib": rss_after,
        }
        if status == 200:
            sample["usage"] = payload.get("usage")
            sample["metrics"] = payload.get("fathom", {}).get("metrics")
            sample["runtime_cache_hit"] = metric_value(sample, "runtime_cache_hit")
            sample["runtime_residency"] = metric_value(sample, "runtime_residency")
            sample["runtime_family"] = metric_value(sample, "runtime_family")
            sample["runtime_cache_lookup_ms"] = metric_value(sample, "runtime_cache_lookup_ms")
            if args.cache_phase_report:
                sample["cache_phase"] = "cold_candidate" if successful_requests == 0 else "warm_candidate"
            successful_requests += 1
            sample["finish_reason"] = payload.get("choices", [{}])[0].get("finish_reason")
        else:
            sample["error"] = payload.get("error", payload)
        samples.append(sample)
    measured = [s for s in samples if not s["warmup"] and s["http_status"] == 200]
    return {
        "model": model_id,
        "samples": samples,
        "summary": {
            "wall_ms": summarize([s["wall_ms"] for s in measured]),
            "model_load_ms": summarize(metric_values(measured, "model_load_ms")),
            "generation_ms": summarize(metric_values(measured, "generation_ms")),
            "ttft_ms": summarize(metric_values(measured, "ttft_ms")),
            "prefill_ms": summarize(metric_values(measured, "prefill_ms")),
            "decode_ms": summarize(metric_values(measured, "decode_ms")),
            "tokens_per_second": summarize(metric_values(measured, "tokens_per_second")),
            "prefill_tokens_per_second": summarize(metric_values(measured, "prefill_tokens_per_second")),
            "decode_tokens_per_second": summarize(metric_values(measured, "decode_tokens_per_second")),
            "runtime_cache": runtime_cache_summary(measured),
            "cache_phases": cache_phase_summary(samples) if args.cache_phase_report else None,
        },
    }


def bench_embedding(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    body = {"input": args.embed_input, "normalize": True}
    for run in range(args.warmups + args.runs):
        rss_before = maybe_rss_kib(args.pid)
        status, payload, wall_ms = request_json(args.base_url, "POST", f"/api/embedding-models/{model_id}/embed", body)
        rss_after = maybe_rss_kib(args.pid)
        sample = {
            "run": run + 1,
            "warmup": run < args.warmups,
            "http_status": status,
            "wall_ms": round(wall_ms, 3),
            "rss_before_kib": rss_before,
            "rss_after_kib": rss_after,
        }
        if status == 200:
            sample["embedding_dimension"] = payload.get("embedding_dimension")
            sample["metrics"] = payload.get("fathom", {}).get("metrics")
        else:
            sample["error"] = payload.get("error", payload)
        samples.append(sample)
    measured = [s for s in samples if not s["warmup"] and s["http_status"] == 200]
    return {
        "model": model_id,
        "samples": samples,
        "summary": {
            "wall_ms": summarize([s["wall_ms"] for s in measured]),
            "embedding_total_ms": summarize(metric_values(measured, "total_ms")),
            "tokenization_ms": summarize(metric_values(measured, "tokenization_ms")),
            "inference_ms": summarize(metric_values(measured, "inference_ms")),
            "pooling_ms": summarize(metric_values(measured, "pooling_ms")),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Fathom backend benchmark", "", f"- Timestamp: `{result['timestamp']}`", f"- Server: `{result['server']}`", f"- Runs: {result['runs']} measured, {result['warmups']} warmup", ""]
    if result.get("cache_phase_report"):
        lines += [
            "- Cache phase report: enabled (`cold_candidate` is the first successful request observed by this process; `warm_candidate` is later same-model traffic).",
            "",
        ]
    if result["chat"]:
        lines += ["## Chat generation", ""]
        for item in result["chat"]:
            runtime_cache = item["summary"].get("runtime_cache", {})
            lines.append(f"### `{item['model']}`")
            lines.append(f"- Wall ms median: `{item['summary']['wall_ms']['median']}`")
            lines.append(f"- Model load ms median: `{item['summary']['model_load_ms']['median']}`")
            lines.append(f"- TTFT/prefill ms median: `{item['summary']['ttft_ms']['median']}`")
            lines.append(f"- Decode ms median: `{item['summary']['decode_ms']['median']}`")
            lines.append(f"- Tokens/sec median: `{item['summary']['tokens_per_second']['median']}`")
            lines.append(f"- Prefill tokens/sec median: `{item['summary']['prefill_tokens_per_second']['median']}`")
            lines.append(f"- Decode tokens/sec median: `{item['summary']['decode_tokens_per_second']['median']}`")
            lines.append(f"- Runtime cache hits: `{runtime_cache.get('runtime_cache_hit', {})}`")
            lines.append(f"- Runtime residency: `{runtime_cache.get('runtime_residency', {})}`")
            lines.append(f"- Runtime family: `{runtime_cache.get('runtime_family', {})}`")
            lines.append(f"- Runtime cache lookup ms median: `{runtime_cache.get('runtime_cache_lookup_ms', {}).get('median')}`")
            if result.get("cache_phase_report"):
                lines += ["", "| Phase | Samples | Residency | Cache hits | Wall ms median | Model load ms median | TTFT ms median | Tokens/sec median |", "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
                for phase, summary in item["summary"].get("cache_phases", {}).items():
                    phase_cache = summary.get("runtime_cache", {})
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                phase,
                                str(summary.get("samples")),
                                f"`{phase_cache.get('runtime_residency', {})}`",
                                f"`{phase_cache.get('runtime_cache_hit', {})}`",
                                str(summary.get("wall_ms", {}).get("median")),
                                str(summary.get("model_load_ms", {}).get("median")),
                                str(summary.get("ttft_ms", {}).get("median")),
                                str(summary.get("tokens_per_second", {}).get("median")),
                            ]
                        )
                        + " |"
                    )
            lines.append("")
    if result["embeddings"]:
        lines += ["## Embeddings", ""]
        for item in result["embeddings"]:
            lines.append(f"### `{item['model']}`")
            lines.append(f"- Wall ms median: `{item['summary']['wall_ms']['median']}`")
            lines.append(f"- Runtime total ms median: `{item['summary']['embedding_total_ms']['median']}`")
            lines.append(f"- Tokenization ms median: `{item['summary']['tokenization_ms']['median']}`")
            lines.append(f"- Inference ms median: `{item['summary']['inference_ms']['median']}`")
            lines.append(f"- Pooling ms median: `{item['summary']['pooling_ms']['median']}`")
            lines.append("")
    lines.append("Note: chat TTFT/prefill/decode timings are server-side Fathom metrics from a non-streaming request, not client-observed streaming latency.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a running Fathom backend without extra dependencies.")
    parser.add_argument("--base-url", default=os.environ.get("FATHOM_BASE_URL", "http://127.0.0.1:8180"))
    parser.add_argument("--model", dest="models", action="append", default=[], help="Chat model id to benchmark. Repeatable. Defaults to all /v1/models.")
    parser.add_argument("--embedding-model", dest="embedding_models", action="append", default=[], help="Embedding model id to benchmark. Repeatable. Defaults to runnable /api/embedding-models.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt", default="Write one short paragraph about a small robot learning to garden.")
    parser.add_argument("--embed-input", action="append", default=["Fathom measures local retrieval embeddings."])
    parser.add_argument("--pid", type=int, default=None, help="Optional fathom-server PID for RSS samples via ps.")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path.")
    parser.add_argument(
        "--cache-phase-report",
        action="store_true",
        help="Label the first successful same-process chat request as cold_candidate and later same-model requests as warm_candidate, then summarize runtime cache fields by phase.",
    )
    args = parser.parse_args()

    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be >= 1 and --warmups must be >= 0")

    chat_models = pick_models(args.base_url, args.models)
    embedding_models = pick_embedding_models(args.base_url, args.embedding_models)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": args.base_url,
        "runs": args.runs,
        "warmups": args.warmups,
        "cache_phase_report": args.cache_phase_report,
        "memory_note": "rss_*_kib fields are coarse process RSS snapshots from ps when --pid is supplied; otherwise null.",
        "chat": [bench_chat(args, model) for model in chat_models],
        "embeddings": [bench_embedding(args, model) for model in embedding_models],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(result))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
