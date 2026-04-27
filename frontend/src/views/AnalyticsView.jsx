import { buildRuntimeCacheAnalytics, formatCompactNumber, formatDate, formatMillis, formatRate } from '../lib/formatters'

function startOfDay(date) {
  const next = new Date(date)
  next.setHours(0, 0, 0, 0)
  return next
}

function dayKey(date) {
  return startOfDay(date).toISOString().slice(0, 10)
}

function labelDay(date) {
  return new Intl.DateTimeFormat(undefined, { weekday: 'short' }).format(date)
}

function safeAverage(values) {
  if (!values.length) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function conversationLabel(conversation) {
  const raw = conversation?.title?.trim()
  return raw && raw.toLowerCase() !== 'new conversation' ? raw : 'Untitled chat'
}

export default function AnalyticsView({ conversations, models, runtime }) {
  const now = new Date()
  const sevenDays = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(now)
    date.setDate(now.getDate() - (6 - index))
    return {
      key: dayKey(date),
      label: labelDay(date),
      date,
      prompts: 0,
      replies: 0,
    }
  })
  const dayMap = new Map(sevenDays.map((day) => [day.key, day]))

  const modelStats = new Map()
  let totalMessages = 0
  let totalAssistantReplies = 0
  let totalUserPrompts = 0
  let activeToday = 0
  let latestActivityAt = null

  for (const conversation of conversations) {
    const updatedAt = conversation?.updated_at ? new Date(conversation.updated_at) : null
    if (updatedAt && startOfDay(updatedAt).getTime() === startOfDay(now).getTime()) {
      activeToday += 1
    }
    if (updatedAt && (!latestActivityAt || updatedAt > latestActivityAt)) {
      latestActivityAt = updatedAt
    }

    const modelId = conversation.model_id || 'unknown'
    const base = modelStats.get(modelId) || {
      id: modelId,
      name: models.find((model) => model.id === modelId)?.name || modelId,
      prompts: 0,
      replies: 0,
      conversations: new Set(),
      lastUsedAt: null,
      outRates: [],
      inRates: [],
    }

    base.conversations.add(conversation.id)

    for (const message of conversation.messages || []) {
      totalMessages += 1
      const createdAt = message?.created_at ? new Date(message.created_at) : updatedAt
      if (createdAt && (!base.lastUsedAt || createdAt > base.lastUsedAt)) {
        base.lastUsedAt = createdAt
      }
      if (createdAt && (!latestActivityAt || createdAt > latestActivityAt)) {
        latestActivityAt = createdAt
      }

      const bucket = createdAt ? dayMap.get(dayKey(createdAt)) : null
      if (message.role === 'user') {
        totalUserPrompts += 1
        base.prompts += 1
        if (bucket) bucket.prompts += 1
      }
      if (message.role === 'assistant') {
        totalAssistantReplies += 1
        base.replies += 1
        if (bucket) bucket.replies += 1
        if (message.tokens_out_per_sec !== null && message.tokens_out_per_sec !== undefined) {
          base.outRates.push(Number(message.tokens_out_per_sec))
        }
        if (message.tokens_in_per_sec !== null && message.tokens_in_per_sec !== undefined) {
          base.inRates.push(Number(message.tokens_in_per_sec))
        }
      }
    }

    modelStats.set(modelId, base)
  }

  const modelRows = [...modelStats.values()]
    .map((row) => ({
      ...row,
      conversationCount: row.conversations.size,
      avgOutRate: safeAverage(row.outRates),
      avgInRate: safeAverage(row.inRates),
    }))
    .sort((left, right) => {
      if (right.replies !== left.replies) return right.replies - left.replies
      return (right.prompts + right.replies) - (left.prompts + left.replies)
    })

  const totalTrackedEvents = Math.max(1, totalAssistantReplies + totalUserPrompts)
  const topModels = modelRows.slice(0, 5)
  const readyModels = models.filter((model) => model.status === 'ready' && model.model_path).length
  const averageReplyRate = safeAverage(modelRows.flatMap((row) => row.outRates))
  const busiestDay = [...sevenDays].sort((left, right) => (right.prompts + right.replies) - (left.prompts + left.replies))[0]
  const runtimeCacheAnalytics = buildRuntimeCacheAnalytics(conversations.flatMap((conversation) => conversation.messages || []))
  const runtimeSummary = runtimeCacheAnalytics.summary
  const runtimeMeasuredLabel = runtimeSummary.assistantReplies
    ? `${runtimeSummary.measured} of ${runtimeSummary.assistantReplies} assistant replies include cache residency metrics`
    : 'No assistant replies stored yet'
  const runtimeUnknownLabel = runtimeSummary.unknown === 0
    ? 'No unknown replies in stored conversations'
    : `${runtimeSummary.unknown} old or external replies have no cache residency metric`
  const recentThreads = [...conversations]
    .sort((left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime())
    .slice(0, 4)

  return (
    <section className="view-stack analytics-view view-shell">
      <div className="panel panel-hero analytics-hero">
        <div className="view-hero-copy">
          <p className="panel-kicker">Internal analytics</p>
          <h2>How your local models are actually being used</h2>
          <p className="hero-summary">A calmer internal view of prompts, replies, activity patterns, and which local models are doing the work. Everything here is computed from Fathom’s own local conversation and message telemetry.</p>
        </div>
        <div className="analytics-hero-aside">
          <div className={`status-pill ${runtime?.loaded_now ? 'ready' : runtime?.ready ? 'ready' : 'warm'}`}>
            {runtime?.loaded_now ? 'Runtime loaded now' : runtime?.ready ? 'Runtime ready on demand' : 'Runtime needs a ready model'}
          </div>
          <div className="analytics-hero-note">
            <span>Latest activity</span>
            <strong>{latestActivityAt ? formatDate(latestActivityAt.toISOString()) : 'No chat activity yet'}</strong>
          </div>
        </div>
      </div>

      <div className="analytics-stat-grid">
        <div className="analytics-stat-card panel">
          <span>Total conversations</span>
          <strong>{formatCompactNumber(conversations.length)}</strong>
          <small>{activeToday} active today</small>
        </div>
        <div className="analytics-stat-card panel">
          <span>User prompts</span>
          <strong>{formatCompactNumber(totalUserPrompts)}</strong>
          <small>{formatCompactNumber(totalMessages)} total stored messages</small>
        </div>
        <div className="analytics-stat-card panel">
          <span>Assistant replies</span>
          <strong>{formatCompactNumber(totalAssistantReplies)}</strong>
          <small>{averageReplyRate ? `${formatRate(averageReplyRate)} average reply speed` : 'Reply speed will appear after assistant messages land'}</small>
        </div>
        <div className="analytics-stat-card panel">
          <span>Models ready now</span>
          <strong>{readyModels}</strong>
          <small>{runtime?.loaded_now ? `${runtime?.active_model_id} loaded` : 'Load happens on demand'}</small>
        </div>
      </div>

      <section className="panel panel-section analytics-panel analytics-runtime-panel">
        <div className="section-heading">
          <div>
            <p className="panel-kicker">Runtime cache analytics</p>
            <h2>Warm and cold residency from stored replies</h2>
          </div>
          <p className="model-summary">Counts use persisted assistant message metrics only. Replies without those fields stay unknown; they are not treated as cache misses.</p>
        </div>
        <div className="analytics-runtime-grid">
          <div className="analytics-stat-card analytics-runtime-card">
            <span>Warm runtime replies</span>
            <strong>{formatCompactNumber(runtimeSummary.warm)}</strong>
            <small>{runtimeMeasuredLabel}</small>
          </div>
          <div className="analytics-stat-card analytics-runtime-card">
            <span>Cold load replies</span>
            <strong>{formatCompactNumber(runtimeSummary.cold)}</strong>
            <small>Cold means the persisted reply reported a real cold load.</small>
          </div>
          <div className="analytics-stat-card analytics-runtime-card">
            <span>Unknown cache data</span>
            <strong>{formatCompactNumber(runtimeSummary.unknown)}</strong>
            <small>{runtimeUnknownLabel}</small>
          </div>
        </div>
        <div className="analytics-runtime-table">
          <div className="analytics-runtime-table-head analytics-runtime-table-row">
            <span>Family</span>
            <span>Replies</span>
            <span>Warm / cold</span>
            <span>Avg total</span>
            <span>Avg load</span>
            <span>Avg lookup</span>
            <span>Avg output rate</span>
          </div>
          {runtimeCacheAnalytics.families.length === 0 && <div className="empty-state">No persisted runtime-family metrics yet. Send local chat replies with a cached Candle model and this table will fill in.</div>}
          {runtimeCacheAnalytics.families.map((family) => (
            <div key={family.key} className="analytics-runtime-table-row analytics-runtime-table-data">
              <strong>{family.family}</strong>
              <span>{family.replies}</span>
              <span>{family.warm} warm / {family.cold} cold{family.unknown ? ` · ${family.unknown} unknown` : ''}</span>
              <span>{formatMillis(family.avgTotalMs) || '—'}</span>
              <span>{formatMillis(family.avgLoadMs) || '—'}</span>
              <span>{formatMillis(family.avgLookupMs) || '—'}</span>
              <span>{family.avgTokenRate !== null && family.avgTokenRate !== undefined ? formatRate(family.avgTokenRate) : '—'}</span>
            </div>
          ))}
        </div>
      </section>

      <div className="analytics-grid analytics-grid-primary">
        <section className="panel panel-section analytics-panel">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Model leaderboard</p>
              <h2>Which models are carrying the workload</h2>
            </div>
            <p className="model-summary">Sorted by assistant replies so you can see real usage, not just what is installed.</p>
          </div>
          <div className="analytics-leaderboard">
            {topModels.length === 0 && <div className="empty-state">No model usage yet. Send a few local chats and this board will fill in.</div>}
            {topModels.map((model) => {
              const share = ((model.prompts + model.replies) / totalTrackedEvents) * 100
              return (
                <article key={model.id} className="analytics-model-row">
                  <div className="analytics-model-row-head">
                    <div>
                      <strong>{model.name}</strong>
                      <span>{model.id}</span>
                    </div>
                    <div className="analytics-model-row-metrics">
                      <span>{model.replies} replies</span>
                      <span>{model.avgOutRate ? `${formatRate(model.avgOutRate)} avg out` : 'No speed data yet'}</span>
                    </div>
                  </div>
                  <div className="analytics-bar-track" aria-hidden="true">
                    <div className="analytics-bar-fill" style={{ width: `${Math.max(8, Math.min(100, share))}%` }} />
                  </div>
                  <div className="analytics-model-row-foot">
                    <small>{model.conversationCount} conversations</small>
                    <small>{model.lastUsedAt ? `Last used ${formatDate(model.lastUsedAt.toISOString())}` : 'No recent activity'}</small>
                  </div>
                </article>
              )
            })}
          </div>
        </section>

        <section className="panel panel-section analytics-panel">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Weekly flow</p>
              <h2>Prompt and reply volume</h2>
            </div>
            <p className="model-summary">A lightweight seven day view so you can spot bursts and quiet periods without a heavy chart library.</p>
          </div>
          <div className="analytics-volume-chart">
            {sevenDays.map((day) => {
              const total = day.prompts + day.replies
              const maxTotal = Math.max(...sevenDays.map((entry) => entry.prompts + entry.replies), 1)
              const height = total === 0 ? 10 : Math.max(18, (total / maxTotal) * 100)
              return (
                <div key={day.key} className="analytics-volume-day">
                  <div className="analytics-volume-bars">
                    <div className="analytics-volume-bar analytics-volume-bar-prompts" style={{ height: `${Math.max(10, (day.prompts / maxTotal) * 100)}%` }} />
                    <div className="analytics-volume-bar analytics-volume-bar-replies" style={{ height: `${height}%` }} />
                  </div>
                  <strong>{day.label}</strong>
                  <span>{total} total</span>
                </div>
              )
            })}
          </div>
          <div className="analytics-inline-stats">
            <div className="history-metric-pill">
              <span>Busiest day</span>
              <strong>{busiestDay ? `${busiestDay.label} · ${busiestDay.prompts + busiestDay.replies} events` : 'No activity yet'}</strong>
            </div>
            <div className="history-metric-pill">
              <span>Reply share</span>
              <strong>{totalTrackedEvents ? `${Math.round((totalAssistantReplies / totalTrackedEvents) * 100)}% assistant` : 'No activity yet'}</strong>
            </div>
          </div>
        </section>
      </div>

      <div className="analytics-grid analytics-grid-secondary">
        <section className="panel panel-section analytics-panel">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Model details</p>
              <h2>Usage, speed, and recency by model</h2>
            </div>
            <p className="model-summary">This is the deeper internal table for comparing real usage across the local model library.</p>
          </div>
          <div className="analytics-table">
            <div className="analytics-table-head analytics-table-row">
              <span>Model</span>
              <span>Conversations</span>
              <span>Prompts</span>
              <span>Replies</span>
              <span>Avg out</span>
              <span>Last used</span>
            </div>
            {modelRows.length === 0 && <div className="empty-state">No model usage to compare yet.</div>}
            {modelRows.map((model) => (
              <div key={model.id} className="analytics-table-row analytics-table-row-data">
                <strong title={model.id}>{model.name}</strong>
                <span>{model.conversationCount}</span>
                <span>{model.prompts}</span>
                <span>{model.replies}</span>
                <span>{model.avgOutRate ? formatRate(model.avgOutRate) : '—'}</span>
                <span>{model.lastUsedAt ? formatDate(model.lastUsedAt.toISOString()) : '—'}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel panel-section analytics-panel">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Recent threads</p>
              <h2>Where usage has been happening most recently</h2>
            </div>
            <p className="model-summary">Helpful for quickly seeing which chats drove the latest model activity.</p>
          </div>
          <div className="analytics-thread-list">
            {recentThreads.length === 0 && <div className="empty-state">No conversations yet.</div>}
            {recentThreads.map((conversation) => {
              const messageCount = conversation.messages?.length || 0
              const assistantReplies = conversation.messages?.filter((message) => message.role === 'assistant').length || 0
              const modelName = models.find((model) => model.id === conversation.model_id)?.name || conversation.model_id || 'No model recorded'
              return (
                <article key={conversation.id} className="analytics-thread-card">
                  <div>
                    <strong>{conversationLabel(conversation)}</strong>
                    <span>{modelName}</span>
                  </div>
                  <div className="analytics-thread-card-meta">
                    <small>{messageCount} messages</small>
                    <small>{assistantReplies} replies</small>
                    <small>{formatDate(conversation.updated_at)}</small>
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      </div>
    </section>
  )
}
