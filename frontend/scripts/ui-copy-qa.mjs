#!/usr/bin/env node
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { buildAssistantMetricSummary, buildAssistantMetricTitle, buildRuntimeCacheAnalytics, formatRuntimeFamily, formatRuntimeResidency } from '../src/lib/formatters.js'

const root = new URL('..', import.meta.url).pathname

const checks = [
  {
    file: 'src/views/MemoryView.jsx',
    snippets: [
      'SafeTensors embedding inference available',
      '/api/embedding-models/:id/embed',
      'fathom.retrieval',
    ],
  },
  {
    file: 'src/views/SystemView.jsx',
    snippets: [
      'Retrieval API',
      'Embedding lane',
      'explicit-vector retrieval',
    ],
  },
  {
    file: 'src/views/ModelsView.jsx',
    snippets: [
      'ONNX embedding',
      'Not a chat model',
      'local-embeddings-retrieval',
      'Installing…',
      'Downloading and verifying…',
    ],
  },
  {
    file: 'src/hooks/useDashboardData.js',
    snippets: [
      'Hugging Face model installed and inspected',
      'Could not install that Hugging Face model',
    ],
  },
]

const failures = []

for (const check of checks) {
  const path = join(root, check.file)
  const content = readFileSync(path, 'utf8')
  for (const snippet of check.snippets) {
    if (!content.includes(snippet)) {
      failures.push(`${check.file} is missing: ${snippet}`)
    }
  }
}

const distIndex = join(root, 'dist/index.html')
if (existsSync(distIndex)) {
  const html = readFileSync(distIndex, 'utf8')
  if (!html.includes('assets/index-')) {
    failures.push('dist/index.html does not reference a built asset bundle')
  }
}

const warmSummary = buildAssistantMetricSummary({
  role: 'assistant',
  runtime_residency: 'warm_reused',
  runtime_family: 'phi',
  model_load_ms: 0,
  total_ms: 41,
  tokens_out_per_sec: 12.3,
  runtime_cache_hit: true,
  runtime_cache_lookup_ms: 0,
  generation_ms: 41,
})
const coldSummary = buildAssistantMetricSummary({
  role: 'assistant',
  runtime_residency: 'cold_loaded',
  runtime_family: 'gemma',
  model_load_ms: 63,
  total_ms: 107,
  tokens_out_per_sec: 8.1,
  runtime_cache_hit: false,
})
const oldSummary = buildAssistantMetricSummary({
  role: 'assistant',
  tokens_out_per_sec: 9.5,
})
const title = buildAssistantMetricTitle({ role: 'assistant', runtime_cache_hit: true, runtime_cache_lookup_ms: 2, generation_ms: 39 })
const runtimeAnalytics = buildRuntimeCacheAnalytics([
  { role: 'assistant', runtime_residency: 'cold_loaded', runtime_cache_hit: false, runtime_family: 'gpt2', runtime_cache_lookup_ms: 4, model_load_ms: 20, total_ms: 50, tokens_out_per_sec: 10 },
  { role: 'assistant', runtime_residency: 'warm_reused', runtime_cache_hit: true, runtime_family: 'gpt2', runtime_cache_lookup_ms: 2, model_load_ms: 0, total_ms: 25, tokens_out_per_sec: 20 },
  { role: 'assistant', runtime_residency: 'warm_reused', runtime_cache_hit: true, runtime_family: 'phi', model_load_ms: 0, total_ms: 7, tokens_out_per_sec: 12 },
  { role: 'assistant', tokens_out_per_sec: 9.5 },
  { role: 'assistant', runtime_residency: 'not_cached', runtime_cache_hit: false, total_ms: 14 },
  { role: 'user', content: 'ignore me' },
])
const analyticsCopy = readFileSync(join(root, 'src/views/AnalyticsView.jsx'), 'utf8')
const ggufCopyFiles = [
  join(root, '../README.md'),
  join(root, '../docs/research/model-format-landscape.md'),
  join(root, '../docs/api/backend-only-quickstart.md'),
  join(root, '../crates/fathom-server/src/main.rs'),
]
const ggufCopy = ggufCopyFiles.map((path) => readFileSync(path, 'utf8')).join('\n')

if (formatRuntimeResidency('warm_reused') !== 'Warm runtime') failures.push('warm_reused should render as Warm runtime')
if (formatRuntimeResidency('cold_loaded') !== 'Cold load') failures.push('cold_loaded should render as Cold load')
if (formatRuntimeResidency('not_cached') !== null) failures.push('not_cached should not imply a cache state in compact chat copy')
if (formatRuntimeFamily('gpt2') !== 'GPT-2') failures.push('gpt2 should render as GPT-2')
if (formatRuntimeFamily('qwen2') !== 'Qwen2') failures.push('qwen2 should render as Qwen2')
if (warmSummary !== 'Warm runtime · Phi · 0 ms load · 41 ms total · 12.3 tok/s') failures.push(`unexpected warm runtime summary: ${warmSummary}`)
if (coldSummary !== 'Cold load · Gemma · 63 ms load · 107 ms total · 8.1 tok/s') failures.push(`unexpected cold runtime summary: ${coldSummary}`)
if (oldSummary !== '9.5 tok/s') failures.push(`old token-rate-only messages should still render safely, got: ${oldSummary}`)
if (!title.includes('Server-side non-streaming generation metrics')) failures.push('metric title must scope timing as server-side non-streaming')
if (runtimeAnalytics.summary.assistantReplies !== 5) failures.push(`runtime analytics should count five assistant replies, got ${runtimeAnalytics.summary.assistantReplies}`)
if (runtimeAnalytics.summary.warm !== 2) failures.push(`runtime analytics should count two warm replies, got ${runtimeAnalytics.summary.warm}`)
if (runtimeAnalytics.summary.cold !== 1) failures.push(`runtime analytics should count one cold reply, got ${runtimeAnalytics.summary.cold}`)
if (runtimeAnalytics.summary.unknown !== 2) failures.push(`runtime analytics should keep old/not_cached replies unknown, got ${runtimeAnalytics.summary.unknown}`)
const gpt2Row = runtimeAnalytics.families.find((row) => row.key === 'gpt2')
if (!gpt2Row || gpt2Row.warm !== 1 || gpt2Row.cold !== 1 || Math.round(gpt2Row.avgTotalMs) !== 38 || Math.round(gpt2Row.avgLoadMs) !== 10 || Math.round(gpt2Row.avgLookupMs) !== 3 || Math.round(gpt2Row.avgTokenRate) !== 15) {
  failures.push(`runtime family analytics should average GPT-2 warm/cold metrics, got ${JSON.stringify(gpt2Row)}`)
}
if (!analyticsCopy.includes('Replies without those fields stay unknown; they are not treated as cache misses.')) failures.push('analytics copy must keep old/external replies out of cache-miss counts')
if (!analyticsCopy.includes('persisted assistant message metrics only')) failures.push('analytics copy must scope runtime analytics to persisted assistant metrics')
if (!ggufCopy.includes('private synthetic F32/F16/Q8_0/Q4_0 payload decode tests') && !ggufCopy.includes('internal synthetic payload decode tests cover F32/F16/Q8_0/Q4_0 readiness')) failures.push('GGUF copy must scope Q8_0/Q4_0 payload decode as internal synthetic readiness groundwork')
if (!ggufCopy.includes('narrow synthetic GPT-2/BPE and Llama/SentencePiece')) failures.push('GGUF copy must mention bounded internal tokenizer metadata retention for both narrow GGUF tokenizer shapes')
if (!ggufCopy.includes('excluded from `/v1/models`')) failures.push('GGUF copy must keep metadata-only fixtures excluded from /v1/models')
if (!ggufCopy.includes('private fixture-scoped Llama/SentencePiece encode/decode parity helpers')) failures.push('GGUF copy must mention private fixture-scoped Llama/SentencePiece encode/decode parity helpers as readiness groundwork')
if (!ggufCopy.includes('public/runtime tokenizer execution') && !ggufCopy.includes('public/runtime GGUF tokenizer execution')) failures.push('GGUF copy must explicitly scope missing tokenizer execution to public/runtime surfaces')
for (const text of [warmSummary, coldSummary, title, analyticsCopy]) {
  if (/session memory|instant|client ttft|streaming ttft|production throughput|batching|\bgpu\b|gguf chat|onnx chat/i.test(text)) {
    failures.push(`runtime metric copy overclaims: ${text.match(/session memory|instant|client ttft|streaming ttft|production throughput|batching|\bgpu\b|gguf chat|onnx chat/i)?.[0]}`)
  }
}
for (const text of [ggufCopy]) {
  if (/GGUF (runtime|inference|chat|generation) (is|are) (ready|supported|available)|GGUF weights loaded|broad GGUF dequantization|general GGUF dequantization supported|GGUF tokenizer support|public GGUF tokenizer support|Llama tokenizer implemented|SentencePiece support/i.test(text)) {
    failures.push(`GGUF copy overclaims: ${text.match(/GGUF (runtime|inference|chat|generation) (is|are) (ready|supported|available)|GGUF weights loaded|broad GGUF dequantization|general GGUF dequantization supported|GGUF tokenizer support|public GGUF tokenizer support|Llama tokenizer implemented|SentencePiece support/i)?.[0]}`)
  }
}

if (failures.length) {
  console.error('UI capability copy QA failed:')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log('UI capability copy QA passed.')
