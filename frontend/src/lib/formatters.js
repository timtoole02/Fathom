export function formatRate(value) {
  if (value === null || value === undefined) return '—'
  return `${Number(value).toFixed(1)} tok/s`
}

export function formatDate(value) {
  if (!value) return ''
  return new Date(value).toLocaleString()
}

export function formatSidebarDate(value) {
  if (!value) return ''
  const date = new Date(value)
  const now = new Date()
  const sameDay = date.toDateString() === now.toDateString()
  if (sameDay) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  }
  return date.toLocaleDateString([], { month: 'numeric', day: 'numeric' })
}

export function formatHistoryDate(value) {
  if (!value) return ''
  const date = new Date(value)
  const now = new Date()
  const sameDay = date.toDateString() === now.toDateString()
  if (sameDay) {
    return `Today, ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export function formatBytes(value) {
  if (value === null || value === undefined) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = Number(value)
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size.toFixed(size >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

export function formatCompactNumber(value) {
  if (value === null || value === undefined) return '0'
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value))
}

export function formatPreview(value, maxLength = 120) {
  if (!value) return 'No messages yet'
  const normalized = String(value).replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength - 1)}…`
}

export function clampText(value, maxLength = 72) {
  if (!value) return ''
  const normalized = String(value).replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength - 1)}…`
}

export function formatRuntimeResidency(value) {
  switch (value) {
    case 'warm_reused':
      return 'Warm runtime'
    case 'cold_loaded':
      return 'Cold load'
    case 'not_cached':
      return null
    default:
      return null
  }
}

export function formatRuntimeFamily(value) {
  const labels = {
    gpt2: 'GPT-2',
    'gpt-2': 'GPT-2',
    llama: 'Llama',
    qwen2: 'Qwen2',
    phi: 'Phi',
    mistral: 'Mistral',
    gemma: 'Gemma',
  }
  if (!value) return null
  const key = String(value).toLowerCase()
  return labels[key] || String(value)
}

export function formatMillis(value) {
  if (value === null || value === undefined) return null
  return `${Math.round(Number(value))} ms`
}

export function safeMetricAverage(values) {
  const finite = values.map(Number).filter(Number.isFinite)
  if (!finite.length) return null
  return finite.reduce((sum, value) => sum + value, 0) / finite.length
}

export function hasMetricValue(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value))
}

export function getRuntimeCacheState(message) {
  if (!message || message.role !== 'assistant') return 'unknown'
  const residency = message.runtime_residency
  if (residency === 'warm_reused' || message.runtime_cache_hit === true) return 'warm'
  if (residency === 'cold_loaded') return 'cold'
  if (message.runtime_cache_hit === false && residency && residency !== 'not_cached') return 'cold'
  return 'unknown'
}

export function buildRuntimeCacheAnalytics(messages = []) {
  const familyOrder = ['gpt2', 'llama', 'qwen2', 'phi', 'mistral', 'gemma']
  const familyRows = new Map()
  const summary = {
    assistantReplies: 0,
    warm: 0,
    cold: 0,
    unknown: 0,
    measured: 0,
  }

  for (const message of messages) {
    if (!message || message.role !== 'assistant') continue
    summary.assistantReplies += 1
    const state = getRuntimeCacheState(message)
    if (state === 'warm') summary.warm += 1
    if (state === 'cold') summary.cold += 1
    if (state === 'unknown') summary.unknown += 1
    if (state === 'warm' || state === 'cold') summary.measured += 1

    const familyLabel = formatRuntimeFamily(message.runtime_family)
    if (!familyLabel) continue
    const familyKey = String(message.runtime_family).toLowerCase()
    const row = familyRows.get(familyKey) || {
      key: familyKey,
      family: familyLabel,
      replies: 0,
      warm: 0,
      cold: 0,
      unknown: 0,
      totalMs: [],
      loadMs: [],
      lookupMs: [],
      tokenRates: [],
    }
    row.replies += 1
    row[state] += 1
    if (hasMetricValue(message.total_ms)) row.totalMs.push(Number(message.total_ms))
    if (hasMetricValue(message.model_load_ms)) row.loadMs.push(Number(message.model_load_ms))
    if (hasMetricValue(message.runtime_cache_lookup_ms)) row.lookupMs.push(Number(message.runtime_cache_lookup_ms))
    if (hasMetricValue(message.tokens_out_per_sec)) row.tokenRates.push(Number(message.tokens_out_per_sec))
    familyRows.set(familyKey, row)
  }

  const families = [...familyRows.values()]
    .map((row) => ({
      ...row,
      avgTotalMs: safeMetricAverage(row.totalMs),
      avgLoadMs: safeMetricAverage(row.loadMs),
      avgLookupMs: safeMetricAverage(row.lookupMs),
      avgTokenRate: safeMetricAverage(row.tokenRates),
    }))
    .sort((left, right) => {
      const leftKnown = familyOrder.indexOf(left.key)
      const rightKnown = familyOrder.indexOf(right.key)
      if (leftKnown !== rightKnown) {
        if (leftKnown === -1) return 1
        if (rightKnown === -1) return -1
        return leftKnown - rightKnown
      }
      return right.replies - left.replies
    })

  return { summary, families }
}

export function buildAssistantMetricSummary(message) {
  if (!message || message.role !== 'assistant') return null

  const parts = []
  const residency = formatRuntimeResidency(message.runtime_residency)
  const family = formatRuntimeFamily(message.runtime_family)
  const load = formatMillis(message.model_load_ms)
  const total = formatMillis(message.total_ms)

  if (residency) parts.push(residency)
  if (family) parts.push(family)
  if (load) parts.push(`${load} load`)
  if (total) parts.push(`${total} total`)
  if (message.tokens_out_per_sec !== null && message.tokens_out_per_sec !== undefined) {
    parts.push(formatRate(message.tokens_out_per_sec))
  }

  if (!parts.length) return null
  return parts.join(' · ')
}

export function buildAssistantMetricTitle(message) {
  if (!message || message.role !== 'assistant') return ''
  const details = ['Server-side non-streaming generation metrics for this reply.']
  if (message.runtime_cache_hit !== null && message.runtime_cache_hit !== undefined) {
    details.push(`Runtime cache hit: ${message.runtime_cache_hit ? 'yes' : 'no'}.`)
  }
  const lookup = formatMillis(message.runtime_cache_lookup_ms)
  if (lookup) details.push(`Runtime cache lookup: ${lookup}.`)
  const generation = formatMillis(message.generation_ms)
  if (generation) details.push(`Generation: ${generation}.`)
  return details.join(' ')
}
