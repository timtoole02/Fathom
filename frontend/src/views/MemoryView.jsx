import { useEffect, useMemo, useState } from 'react'
import { clampText, formatDate } from '../lib/formatters'

export default function MemoryView({
  memories,
  memorySearch,
  setMemorySearch,
  selectedConversation,
  latestAssistantMessage,
  contextStrategy,
  saveToMemory,
  createMemory,
  updateMemory,
  deleteMemory,
  setTab,
}) {
  const [scopeFilter, setScopeFilter] = useState('all')
  const [showPinnedOnly, setShowPinnedOnly] = useState(false)
  const [newMemory, setNewMemory] = useState({ title: '', scope: 'General', body: '' })
  const [editingId, setEditingId] = useState(null)
  const [editDraft, setEditDraft] = useState({ title: '', scope: '', body: '' })
  const [pendingDeleteId, setPendingDeleteId] = useState(null)
  const [busyAction, setBusyAction] = useState('')
  const [retrievalStatus, setRetrievalStatus] = useState({ indexes: [], summary: '', embeddingSummary: '', embeddingStatus: '', embeddingRuntimeLane: '', embeddingModels: [] })

  useEffect(() => {
    let cancelled = false

    async function loadRetrievalStatus() {
      try {
        const [indexesRes, embeddingRes] = await Promise.all([
          fetch('/api/retrieval-indexes'),
          fetch('/api/embedding-models'),
        ])
        const indexesData = indexesRes.ok ? await indexesRes.json() : null
        const embeddingData = embeddingRes.ok ? await embeddingRes.json() : null
        if (cancelled) return
        setRetrievalStatus({
          indexes: indexesData?.items || [],
          summary: indexesData?.summary || 'Retrieval index status is unavailable right now.',
          embeddingSummary: embeddingData?.retrieval?.summary || 'Embedding inference status is unavailable right now.',
          embeddingStatus: embeddingData?.retrieval?.status || '',
          embeddingRuntimeLane: embeddingData?.retrieval?.runtime_lane || '',
          embeddingModels: embeddingData?.items || [],
        })
      } catch {
        if (cancelled) return
        setRetrievalStatus({ indexes: [], summary: 'Retrieval index status is unavailable right now.', embeddingSummary: 'Embedding inference status is unavailable right now.', embeddingStatus: '', embeddingRuntimeLane: '', embeddingModels: [] })
      }
    }

    loadRetrievalStatus()
    const interval = window.setInterval(loadRetrievalStatus, 5000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  const searchableMemories = useMemo(() => {
    if (!memorySearch.trim()) return memories
    const q = memorySearch.toLowerCase()
    return memories.filter((memory) =>
      memory.title.toLowerCase().includes(q)
      || memory.body.toLowerCase().includes(q)
      || memory.scope.toLowerCase().includes(q),
    )
  }, [memories, memorySearch])

  const availableScopes = useMemo(
    () => ['all', ...new Set(memories.map((memory) => memory.scope).filter(Boolean))],
    [memories],
  )

  const visibleMemories = useMemo(() => searchableMemories.filter((memory) => {
    if (showPinnedOnly && !memory.pinned) return false
    if (scopeFilter !== 'all' && memory.scope !== scopeFilter) return false
    return true
  }), [scopeFilter, searchableMemories, showPinnedOnly])

  const pinnedCount = memories.filter((memory) => memory.pinned).length
  const defaultEmbeddingAvailable = retrievalStatus.embeddingStatus === 'vector_index_ready_default_minilm_embedding_inference_available'
    || retrievalStatus.embeddingStatus === 'vector_index_ready_default_minilm_and_onnx_embedding_inference_available'
  const onnxEmbeddingAvailable = retrievalStatus.embeddingStatus === 'vector_index_ready_default_minilm_and_onnx_embedding_inference_available'
  const embeddingStatusTitle = defaultEmbeddingAvailable ? (onnxEmbeddingAvailable ? 'SafeTensors + ONNX embedding inference available' : 'SafeTensors embedding inference available') : 'Embedding inference unavailable'
  const embeddingChipCopy = defaultEmbeddingAvailable ? (onnxEmbeddingAvailable ? 'explicit vectors + SafeTensors/ONNX embed API' : 'explicit vectors + SafeTensors embed API') : 'caller-supplied vectors only'
  const latestChatLabel = clampText(selectedConversation?.title?.trim() || 'Current chat', 40)
  const canSaveLatestReply = Boolean(selectedConversation && latestAssistantMessage?.content)

  const handleCreateMemory = async () => {
    if (busyAction) return
    setBusyAction('create')
    const saved = await createMemory(newMemory)
    if (saved) {
      setNewMemory({ title: '', scope: newMemory.scope || 'General', body: '' })
    }
    setBusyAction('')
  }

  const startEditing = (memory) => {
    setEditingId(memory.id)
    setEditDraft({ title: memory.title, scope: memory.scope, body: memory.body })
    setPendingDeleteId(null)
  }

  const cancelEditing = () => {
    if (busyAction) return
    setEditingId(null)
    setEditDraft({ title: '', scope: '', body: '' })
  }

  const handleSaveEdit = async (memoryId) => {
    if (busyAction) return
    setBusyAction(`edit:${memoryId}`)
    const saved = await updateMemory(memoryId, editDraft)
    if (saved) cancelEditing()
    setBusyAction('')
  }

  const handleTogglePin = async (memory) => {
    if (busyAction) return
    setBusyAction(`pin:${memory.id}`)
    await updateMemory(memory.id, { pinned: !memory.pinned }, { successMessage: memory.pinned ? 'Memory unpinned.' : 'Memory pinned.' })
    setBusyAction('')
  }

  const handleCopy = async (memory) => {
    try {
      await navigator.clipboard.writeText(`${memory.title}\n\n${memory.body}`)
    } catch {
      // noop; updateMemory/createMemory flows already carry user-facing notices for real actions
    }
  }

  const handleDelete = async (memoryId) => {
    if (busyAction) return

    if (pendingDeleteId !== memoryId) {
      setPendingDeleteId(memoryId)
      return
    }

    setBusyAction(`delete:${memoryId}`)
    const deleted = await deleteMemory(memoryId, { successMessage: 'Memory removed from local memory.' })
    if (deleted) {
      if (editingId === memoryId) cancelEditing()
      setPendingDeleteId(null)
    }
    setBusyAction('')
  }

  return (
    <section className="view-stack memory-view view-shell">
      <div className="panel panel-hero view-hero">
        <div className="view-hero-copy">
          <p className="panel-kicker">Memory layer</p>
          <h2>Useful context you can actually manage</h2>
          <p className="hero-summary">Capture durable notes, keep the important ones pinned, and clean up stale context without leaving the app.</p>
        </div>
        <div className="view-hero-stats memory-hero-stats">
          <div className="context-chip context-chip-emphasis">
            <span>Total memories</span>
            <strong>{memories.length}</strong>
            <small>{visibleMemories.length} visible right now</small>
          </div>
          <div className="context-chip">
            <span>Pinned</span>
            <strong>{pinnedCount}</strong>
            <small>{availableScopes.length - 1} scopes represented</small>
          </div>
        </div>
      </div>

      <section className="panel panel-section memory-strategy-panel">
        <div className="panel-header-row panel-header-row-wide">
          <div>
            <p className="panel-kicker">Retrieval runtime</p>
            <h2>Explicit-vector retrieval indexes are available</h2>
            <p className="hero-summary">{retrievalStatus.summary} Opt-in chat retrieval is available through the OpenAI-compatible request extension <code>fathom.retrieval</code> when the caller supplies a query vector. {defaultEmbeddingAvailable ? `The embedding endpoint can generate vectors for verified MiniLM packages through the default SafeTensors lane${onnxEmbeddingAvailable ? ' and the feature-gated ONNX lane' : ''}; automatic document ingestion is still explicit and opt-in.` : 'Embedding inference is unavailable in this build.'}</p>
          </div>
          <div className="context-chip context-chip-emphasis">
            <span>Vector indexes</span>
            <strong>{retrievalStatus.indexes.length}</strong>
            <small>{embeddingChipCopy}</small>
          </div>
        </div>
        <div className="memory-context-card">
          <div className="memory-context-meta">
            <strong>{embeddingStatusTitle}</strong>
            <span>{retrievalStatus.embeddingSummary}</span>
          </div>
          <ul className="memory-strategy-caveats">
            <li>Create/list indexes: <code>GET/POST /api/retrieval-indexes</code></li>
            <li>Add/search explicit vectors: <code>/api/retrieval-indexes/:id/chunks</code> and <code>/api/retrieval-indexes/:id/search</code></li>
            <li>Generate embeddings when available: <code>POST /api/embedding-models/:id/embed</code> for verified MiniLM text-embedding packages.</li>
            <li>Opt-in chat context: send <code>fathom.retrieval</code> with <code>index_id</code>, <code>query_vector</code>, and optional <code>top_k</code>.</li>
          </ul>
          {retrievalStatus.indexes.length > 0 && (
            <div className="memory-context-meta">
              <strong>Current indexes</strong>
              <span>{retrievalStatus.indexes.slice(0, 3).map((index) => `${index.id} · ${index.embedding_dimension}d · ${index.chunk_count} chunks`).join(' / ')}</span>
            </div>
          )}
          {retrievalStatus.embeddingModels.length > 0 && (
            <div className="memory-context-meta">
              <strong>Detected embedding packages</strong>
              <span>{retrievalStatus.embeddingModels.slice(0, 3).map((model) => `${model.name || model.id} · ${model.status?.capability_status || 'metadata'}`).join(' / ')}</span>
            </div>
          )}
        </div>
      </section>

      {contextStrategy && (
        <section className="panel panel-section memory-strategy-panel">
          <div className="panel-header-row panel-header-row-wide">
            <div>
              <p className="panel-kicker">Context strategy advisor</p>
              <h2>{contextStrategy.label}</h2>
              <p className="hero-summary">{contextStrategy.summary}</p>
            </div>
            <div className="context-chip context-chip-emphasis">
              <span>Budget</span>
              <strong>{contextStrategy.max_context_tokens ? `${contextStrategy.max_context_tokens.toLocaleString()} tok` : 'Unknown'}</strong>
              <small>reserve {contextStrategy.reserve_output_tokens} for output</small>
            </div>
          </div>
          <div className="memory-context-card">
            <div className="memory-context-meta">
              <strong>{contextStrategy.needs_retrieval ? 'Plan for retrieval/context assembly' : 'Inline context is acceptable'}</strong>
              <span>
                Chunk ~{contextStrategy.recommended_chunk_tokens} tokens, overlap ~{contextStrategy.recommended_overlap_tokens}, retrieve top {contextStrategy.top_k}.
              </span>
            </div>
            {contextStrategy.caveats?.length > 0 && (
              <ul className="memory-strategy-caveats">
                {contextStrategy.caveats.slice(0, 3).map((caveat) => <li key={caveat}>{caveat}</li>)}
              </ul>
            )}
          </div>
        </section>
      )}

      <div className="memory-layout-grid">
        <section className="panel panel-section memory-capture-panel">
          <div className="panel-header-row panel-header-row-wide">
            <div>
              <p className="panel-kicker">Quick capture</p>
              <h2>Add a memory from here</h2>
              <p className="hero-summary">Save a fact, decision, preference, or useful note without needing to go back into chat first.</p>
            </div>
          </div>

          <div className="memory-form-grid">
            <input
              value={newMemory.title}
              onChange={(event) => setNewMemory((current) => ({ ...current, title: event.target.value }))}
              placeholder="Short title"
            />
            <input
              value={newMemory.scope}
              onChange={(event) => setNewMemory((current) => ({ ...current, scope: event.target.value }))}
              placeholder="Scope"
            />
            <textarea
              className="memory-form-textarea"
              value={newMemory.body}
              onChange={(event) => setNewMemory((current) => ({ ...current, body: event.target.value }))}
              placeholder="What should Fathom remember?"
              rows={5}
            />
            <div className="memory-inline-actions">
              <button className="primary-button" onClick={handleCreateMemory} disabled={busyAction === 'create'}>
                {busyAction === 'create' ? 'Saving…' : 'Save memory'}
              </button>
            </div>
          </div>
        </section>

        <section className="panel panel-section memory-capture-panel">
          <div className="panel-header-row panel-header-row-wide">
            <div>
              <p className="panel-kicker">Current chat</p>
              <h2>Pull useful replies into memory</h2>
              <p className="hero-summary">When a reply is worth keeping, save it here and turn the Memory page into a real working set instead of a dead archive.</p>
            </div>
          </div>

          <div className="memory-context-card">
            <div className="memory-context-meta">
              <strong title={selectedConversation?.title || 'No chat selected'}>{selectedConversation ? latestChatLabel : 'No chat selected'}</strong>
              <span>{canSaveLatestReply ? 'Latest assistant reply is ready to save.' : 'Open a conversation with an assistant reply to save it here.'}</span>
            </div>
            <div className="memory-inline-actions">
              <button className="ghost-button" onClick={() => setTab('chat')}>Open chat</button>
              <button className="primary-button" onClick={saveToMemory} disabled={!canSaveLatestReply}>Save latest reply</button>
            </div>
          </div>
        </section>
      </div>

      <div className="panel panel-section memory-toolbar-panel">
        <div className="panel-header-row panel-header-row-wide memory-toolbar-row">
          <div>
            <p className="panel-kicker">Browse</p>
            <h2>Search, filter, pin, and clean up</h2>
            <p className="hero-summary">Find what matters fast, then keep the memory set tidy from the same page.</p>
          </div>
          <div className="memory-toolbar-controls">
            <div className="storage-badge">{visibleMemories.length} results</div>
            <input className="memory-search" value={memorySearch} onChange={(e) => setMemorySearch(e.target.value)} placeholder="Search memories" />
            <select className="memory-scope-filter" value={scopeFilter} onChange={(event) => setScopeFilter(event.target.value)}>
              {availableScopes.map((scope) => (
                <option key={scope} value={scope}>{scope === 'all' ? 'All scopes' : scope}</option>
              ))}
            </select>
            <button className={`ghost-button ${showPinnedOnly ? 'memory-filter-active' : ''}`} onClick={() => setShowPinnedOnly((current) => !current)}>
              {showPinnedOnly ? 'Pinned only' : 'Show pinned only'}
            </button>
          </div>
        </div>
      </div>

      <div className="memory-grid memory-grid-polished memory-grid-productive">
        {visibleMemories.length === 0 && (
          <div className="empty-state">
            {memories.length === 0
              ? 'No memories saved yet. Add one above or save a useful assistant reply from chat.'
              : 'No memories matched those filters. Try a broader search or clear the pinned/scope filter.'}
          </div>
        )}

        {visibleMemories.map((memory) => {
          const isEditing = editingId === memory.id
          const isDeleting = busyAction === `delete:${memory.id}`
          const isPinning = busyAction === `pin:${memory.id}`
          const isSavingEdit = busyAction === `edit:${memory.id}`

          return (
            <article key={memory.id} className={`memory-card memory-card-polished memory-card-productive ${memory.pinned ? 'memory-card-pinned' : ''}`}>
              <div className="memory-topline memory-topline-productive">
                <div className="memory-title-block">
                  <strong>{memory.title}</strong>
                  <div className="memory-meta-row">
                    <span className="memory-scope-pill">{memory.scope}</span>
                    {memory.pinned && <div className="pin-badge">Pinned</div>}
                  </div>
                </div>
                <div className="memory-card-actions-top">
                  <button className="ghost-button subtle-action" onClick={() => handleTogglePin(memory)} disabled={Boolean(busyAction)}>
                    {isPinning ? 'Saving…' : memory.pinned ? 'Unpin' : 'Pin'}
                  </button>
                  <button className="ghost-button subtle-action" onClick={() => handleCopy(memory)}>Copy</button>
                </div>
              </div>

              {isEditing ? (
                <div className="memory-edit-stack">
                  <input value={editDraft.title} onChange={(event) => setEditDraft((current) => ({ ...current, title: event.target.value }))} placeholder="Title" />
                  <input value={editDraft.scope} onChange={(event) => setEditDraft((current) => ({ ...current, scope: event.target.value }))} placeholder="Scope" />
                  <textarea className="memory-form-textarea" value={editDraft.body} onChange={(event) => setEditDraft((current) => ({ ...current, body: event.target.value }))} rows={6} placeholder="Memory body" />
                  <div className="memory-inline-actions">
                    <button className="primary-button" onClick={() => handleSaveEdit(memory.id)} disabled={isSavingEdit}>{isSavingEdit ? 'Saving…' : 'Save changes'}</button>
                    <button className="ghost-button" onClick={cancelEditing} disabled={isSavingEdit}>Cancel</button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="memory-body-copy">{memory.body}</p>
                  <div className="memory-card-footer memory-card-footer-productive">
                    <div className="history-metric-pill">
                      <span>Updated</span>
                      <strong>{formatDate(memory.updated_at) || 'Unknown'}</strong>
                    </div>
                    <div className="memory-inline-actions memory-inline-actions-end">
                      <button className="ghost-button" onClick={() => startEditing(memory)} disabled={Boolean(busyAction)}>Edit</button>
                      <button className={`ghost-button ${pendingDeleteId === memory.id ? 'danger-button' : 'history-delete-button'}`} onClick={() => handleDelete(memory.id)} disabled={isDeleting || (Boolean(busyAction) && pendingDeleteId !== memory.id)}>
                        {isDeleting ? 'Deleting…' : pendingDeleteId === memory.id ? 'Confirm delete' : 'Delete'}
                      </button>
                    </div>
                  </div>
                </>
              )}
            </article>
          )
        })}
      </div>
    </section>
  )
}
