import { useEffect, useMemo, useState } from 'react'
import { formatApiErrorDetails, readApiErrorDetails } from '../lib/apiErrors'
import { formatBytes, formatCompactNumber } from '../lib/formatters'
import { canLoadIntoRuntime, describeModelState, getModelStatusLabel, hasPlannedCapability, isEmbeddingOnlyModel, isExternalModel, isRunnableModel } from '../lib/modelState'

const FILTERS = [
  { key: 'all', label: 'Everything' },
  { key: 'installed', label: 'Ready locally' },
  { key: 'external', label: 'Connected APIs' },
  { key: 'imported', label: 'Imported' },
  { key: 'downloading', label: 'Downloading' },
  { key: 'attention', label: 'Needs attention' },
]

const CATALOG_PAGE_SIZE = 18

function getGroupKey(model, runtime) {
  if (runtime?.active_model_id === model.id) return 'installed'
  if (isExternalModel(model)) return model.status === 'ready' ? 'external' : 'attention'
  if (model.status === 'failed' || ((model.status === 'ready' || model.status === 'registered') && !model.model_path)) return 'attention'
  if (model.status === 'ready' && model.model_path) return 'installed'
  if (model.status === 'registered') return 'imported'
  if (model.status === 'downloading' || model.status === 'canceling') return 'downloading'
  if (model.status === 'not_installed') return 'catalog'
  return 'attention'
}

function formatModelMeta(model) {
  if (isExternalModel(model)) {
    return [model.source || 'External API', model.runtime_model_name || 'Remote model'].filter(Boolean).join(' · ')
  }
  return [model.size_gb ? `${model.size_gb} GB` : null, model.quant || null, model.engine || null].filter(Boolean).join(' · ')
}

function formatModelOrigin(model) {
  if (isExternalModel(model)) return 'Connected through an external OpenAI-compatible API.'
  if (model.status === 'registered') return 'Imported from a local model artifact and waiting for a real backend that can load it.'
  if (model.status === 'downloading') return 'Downloading into Fathom-managed storage.'
  if (model.status === 'canceling') return 'Stopping the download and cleaning up the partial file.'
  if (model.hf_repo) return 'Downloaded from Hugging Face into Fathom-managed storage.'
  return 'Stored locally on this Mac.'
}

function formatDownloadCopy(model) {
  if (model.status === 'canceling') return 'Canceling…'
  if (model.status === 'downloading' && model.bytes_downloaded) {
    return `${formatBytes(model.bytes_downloaded)} / ${formatBytes(model.total_bytes)}`
  }
  return `${model.progress || 0}%`
}

function findCatalogMatch(models, item) {
  return models.find((model) => model.hf_repo === item.repo_id && model.hf_filename === item.filename)
}

function isRecommendedNonGgufDemo(item) {
  return item?.repo_id === 'distilbert/distilgpt2' && item?.filename === 'model.safetensors'
}

function isOnnxEmbeddingFixture(item) {
  return item?.repo_id === 'nixiesearch/all-MiniLM-L6-v2-onnx' || item?.filename?.toLowerCase().endsWith('.onnx')
}

function isEmbeddingOnlyFixture(item) {
  return isOnnxEmbeddingFixture(item) || item?.repo_id === 'sentence-transformers/all-MiniLM-L6-v2'
}

function formatRemoteMeta(item) {
  return [
    item.license ? item.license.toUpperCase() : 'Public repo',
    item.size_bytes ? `${formatBytes(item.size_bytes)} download` : 'Size checks before download',
    `${formatCompactNumber(item.downloads)} downloads`,
    `${formatCompactNumber(item.likes)} likes`,
  ].join(' · ')
}

function formatCatalogTitle(item) {
  const title = item.title || item.name || item.repo_id || 'Untitled model'
  const suffix = ` (${item.quant})`
  return title.endsWith(suffix) ? title.slice(0, -suffix.length) : title
}

function getNextStepCopy(model, { active, selected, runnable } = {}) {
  if (!model) return 'Pick a model for the next chat or load one now.'
  if (active) return 'Already loaded and ready to answer immediately.'
  if (model.status === 'downloading') return 'Wait for the download to finish before loading it, or cancel it if it is the wrong model.'
  if (model.status === 'canceling') return 'Fathom is stopping this download.'
  if (model.status === 'failed') return isExternalModel(model) ? 'Reconnect the API details to make it usable again.' : 'Retry the download to finish setting it up locally.'
  if (isEmbeddingOnlyModel(model) && model.model_path) return 'Ready for embedding/retrieval calls. It is intentionally excluded from chat selection and runtime loading.'
  if (model.status === 'registered') return 'Load it once to confirm the file path and move it into the ready set.'
  if (isExternalModel(model) && selected) return 'Chosen for the next chat. Fathom will call its connected API on send.'
  if (isExternalModel(model) && runnable) return 'Connected and ready whenever you want it.'
  if (runnable && selected) return 'Chosen for your next chat, or load it now for immediate use.'
  if (runnable) return 'Ready locally, choose it for the next chat or load it now.'
  return 'Download it before you can use it.'
}

function getCapabilityChecklist(model, { runnable = false, canLoad = false } = {}) {
  const external = isExternalModel(model)
  const detected = Boolean(model && model.status !== 'not_installed')
  const metadataReadable = Boolean(external || model?.capability_summary || model?.backend_lanes?.length)
  const laneClaimed = Boolean(external || model?.backend_lanes?.length)

  return [
    { label: 'Detected', done: detected, detail: detected ? 'Known to Fathom' : 'Catalog only' },
    { label: 'Metadata', done: metadataReadable, detail: metadataReadable ? 'Capability reported' : 'Not inspected yet' },
    { label: 'Lane', done: laneClaimed, detail: laneClaimed ? (external ? 'External API' : 'Backend mapped') : 'No backend lane yet' },
    { label: 'Runnable', done: runnable || isEmbeddingOnlyModel(model), detail: runnable ? 'Chat-ready now' : isEmbeddingOnlyModel(model) ? 'Embedding-ready only' : hasPlannedCapability(model) ? 'Backend planned' : canLoad ? 'First load required' : 'Not runnable yet' },
  ]
}

function hasText(model, needle) {
  const haystack = [model?.capability_status, model?.capability_summary, model?.task, model?.install_error, ...(model?.backend_lanes || [])]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
  return haystack.includes(needle)
}

function getNonRunnableCallouts(model) {
  if (!model || isExternalModel(model) || isRunnableModel(model)) return []

  const callouts = []
  const status = (model.capability_status || '').toString().toLowerCase()
  const lanes = model.backend_lanes || []
  const file = `${model.hf_filename || ''} ${model.model_path || ''}`.toLowerCase()

  if (isEmbeddingOnlyModel(model) || hasText(model, 'embedding-only') || model.task === 'text_embedding') {
    callouts.push('Embedding-only: usable for verified MiniLM vector/retrieval calls, not chat or /v1/models.')
  }

  if (status === 'metadata_only' || lanes.includes('gguf-native') || hasText(model, 'metadata-only') || hasText(model, 'gguf')) {
    callouts.push('GGUF metadata-only: inspected for metadata/readiness; native GGUF chat is not implemented.')
  }

  if (file.endsWith('.bin') || hasText(model, 'pytorch') || hasText(model, 'pickle')) {
    callouts.push('PyTorch .bin blocked: Fathom does not load pickle/PyTorch weight files.')
  }

  if (status === 'planned' || status === 'unsupported' || hasPlannedCapability(model) || hasText(model, 'planned') || hasText(model, 'unsupported')) {
    callouts.push('Backend not runnable: this lane is planned or unsupported until a real verified loader exists.')
  }

  if (hasText(model, 'chat_template_not_supported') || hasText(model, 'template not supported') || hasText(model, 'unsupported template')) {
    callouts.push('Chat template unsupported: Fathom refuses unknown templates instead of guessing prompt format.')
  }

  return [...new Set(callouts)].slice(0, 3)
}

function NonRunnableCallouts({ model }) {
  const callouts = getNonRunnableCallouts(model)
  if (!callouts.length) return null

  return (
    <div className="models-card-copy-stack" aria-label="Non-runnable model details">
      {callouts.map((callout) => <p key={callout} className="model-summary">{callout}</p>)}
    </div>
  )
}

function CapabilityChecklist({ model, runnable, canLoad }) {
  const checkpoints = getCapabilityChecklist(model, { runnable, canLoad })

  return (
    <div className="capability-checklist" aria-label="Capability checkpoints">
      {checkpoints.map((item) => (
        <div key={item.label} className={`capability-checkpoint ${item.done ? 'ready' : ''}`}>
          <span>{item.label}</span>
          <strong>{item.detail}</strong>
        </div>
      ))}
    </div>
  )
}

function statusTone(model) {
  if (model.status === 'ready') return 'ready'
  if (model.status === 'downloading' || model.status === 'canceling' || model.status === 'registered') return 'warm'
  return ''
}

function normalizeSortText(value) {
  return (value || '').toString().trim().toLowerCase()
}

function compareModelsByName(left, right) {
  return normalizeSortText(left.name).localeCompare(normalizeSortText(right.name), undefined, {
    numeric: true,
    sensitivity: 'base',
  }) || normalizeSortText(left.id).localeCompare(normalizeSortText(right.id), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function compareCatalogItemsByTitle(left, right) {
  return normalizeSortText(formatCatalogTitle(left)).localeCompare(normalizeSortText(formatCatalogTitle(right)), undefined, {
    numeric: true,
    sensitivity: 'base',
  }) || normalizeSortText(left.repo_id).localeCompare(normalizeSortText(right.repo_id), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

export default function ModelsView({
  runtime,
  registerForm,
  setRegisterForm,
  externalForm,
  setExternalForm,
  registerModel,
  connectExternalModel,
  models,
  selectedModelId,
  setSelectedModelId,
  activateModel,
  installModel,
  installCatalogModel,
  cancelModelDownload,
}) {
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [showImportAdvanced, setShowImportAdvanced] = useState(false)
  const [catalogItems, setCatalogItems] = useState([])
  const [catalogNextCursor, setCatalogNextCursor] = useState(null)
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [catalogLoadingMore, setCatalogLoadingMore] = useState(false)
  const [catalogError, setCatalogError] = useState('')
  const [pendingCatalogInstalls, setPendingCatalogInstalls] = useState(() => new Set())
  const [catalogLicenseAcknowledgements, setCatalogLicenseAcknowledgements] = useState(() => new Set())

  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(async () => {
      setCatalogLoading(true)
      setCatalogError('')
      try {
        const params = new URLSearchParams({ limit: String(CATALOG_PAGE_SIZE) })
        if (query.trim()) params.set('query', query.trim())
        const res = await fetch(`/api/models/catalog?${params.toString()}`, { signal: controller.signal })
        if (!res.ok) throw new Error(formatApiErrorDetails(await readApiErrorDetails(res, 'Could not load the model catalog.'), 'Could not load the model catalog.'))
        const data = await res.json()
        setCatalogItems(data.items || [])
        setCatalogNextCursor(data.next_cursor || null)
      } catch (error) {
        if (error.name === 'AbortError') return
        setCatalogError(error.message || 'Could not load the model catalog.')
        setCatalogItems([])
        setCatalogNextCursor(null)
      } finally {
        if (!controller.signal.aborted) setCatalogLoading(false)
      }
    }, query.trim() ? 250 : 0)

    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [query])

  const catalogInstallKey = (item) => item?.catalog_id || `${item?.repo_id || ''}:${item?.filename || ''}`

  const installCatalogItem = async (item) => {
    const key = catalogInstallKey(item)
    const accepted = catalogLicenseAcknowledgements.has(key)
    setPendingCatalogInstalls((current) => new Set(current).add(key))
    try {
      return await installCatalogModel({ ...item, accept_license: accepted })
    } finally {
      setPendingCatalogInstalls((current) => {
        const next = new Set(current)
        next.delete(key)
        return next
      })
    }
  }

  const loadMoreCatalog = async () => {
    if (!catalogNextCursor || catalogLoadingMore) return
    setCatalogLoadingMore(true)
    setCatalogError('')
    try {
      const params = new URLSearchParams({ limit: String(CATALOG_PAGE_SIZE), cursor: catalogNextCursor })
      if (query.trim()) params.set('query', query.trim())
      const res = await fetch(`/api/models/catalog?${params.toString()}`)
      if (!res.ok) throw new Error(formatApiErrorDetails(await readApiErrorDetails(res, 'Could not load more catalog results.'), 'Could not load more catalog results.'))
      const data = await res.json()
      setCatalogItems((current) => {
        const existing = new Set(current.map((item) => item.catalog_id))
        return [...current, ...(data.items || []).filter((item) => !existing.has(item.catalog_id))]
      })
      setCatalogNextCursor(data.next_cursor || null)
    } catch (error) {
      setCatalogError(error.message || 'Could not load more catalog results.')
    } finally {
      setCatalogLoadingMore(false)
    }
  }

  const filteredModels = useMemo(() => {
    const q = query.trim().toLowerCase()
    return models.filter((model) => {
      const groupKey = getGroupKey(model, runtime)
      const matchesFilter = statusFilter === 'all' || groupKey === statusFilter
      if (!matchesFilter) return false
      if (!q) return true
      return [
        model.name,
        model.id,
        model.quant,
        model.engine,
        model.source,
        model.hf_repo,
        model.hf_filename,
        model.runtime_model_name,
      ].filter(Boolean).some((value) => value.toLowerCase().includes(q))
    }).sort(compareModelsByName)
  }, [models, query, runtime, statusFilter])

  const groupedModels = useMemo(() => {
    const groups = {
      installed: [],
      external: [],
      imported: [],
      downloading: [],
      attention: [],
      catalog: [],
    }
    filteredModels.forEach((model) => {
      groups[getGroupKey(model, runtime)]?.push(model)
    })
    return groups
  }, [filteredModels, runtime])

  const counts = useMemo(() => ({
    installed: models.filter((model) => getGroupKey(model, runtime) === 'installed').length,
    external: models.filter((model) => getGroupKey(model, runtime) === 'external').length,
    downloading: models.filter((model) => getGroupKey(model, runtime) === 'downloading').length,
    imported: models.filter((model) => getGroupKey(model, runtime) === 'imported').length,
    attention: models.filter((model) => getGroupKey(model, runtime) === 'attention').length,
  }), [models, runtime])

  const selectedLocalModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) || null,
    [models, selectedModelId],
  )

  const activeLocalModel = useMemo(
    () => models.find((model) => model.id === runtime?.active_model_id) || null,
    [models, runtime?.active_model_id],
  )

  const selectedRunnable = isRunnableModel(selectedLocalModel)
  const readyModels = [...groupedModels.installed, ...groupedModels.external].sort(compareModelsByName)
  const setupModels = [...groupedModels.imported, ...groupedModels.downloading, ...groupedModels.attention].sort(compareModelsByName)
  const discoverCatalogItems = useMemo(
    () => catalogItems.filter((item) => {
      const localMatch = findCatalogMatch(models, item)
      if (!localMatch) return true
      return localMatch.status === 'failed' || localMatch.status === 'not_installed'
    }).sort(compareCatalogItemsByTitle),
    [catalogItems, models],
  )

  return (
    <section className="view-stack models-view view-shell-wide">
      <div className="panel models-toolbar-panel">
        <div className="models-toolbar-top">
          <label className="models-search-field">
            <span>Search models</span>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by name, repo, quant, source, or file" />
          </label>
          <label className="models-filter-field">
            <span>Show</span>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              {FILTERS.map((filter) => (
                <option key={filter.key} value={filter.key}>{filter.label}</option>
              ))}
            </select>
          </label>
        </div>
        <p className="model-summary">Models are sorted A→Z by name. Search filters both your local library and the model catalog.</p>
        <div className="models-summary-strip" aria-label="Model summary">
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'installed' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'installed' ? 'all' : 'installed')}>
            <span>Ready locally</span>
            <strong>{counts.installed}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'external' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'external' ? 'all' : 'external')}>
            <span>Connected APIs</span>
            <strong>{counts.external}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'downloading' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'downloading' ? 'all' : 'downloading')}>
            <span>Downloading</span>
            <strong>{counts.downloading}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'imported' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'imported' ? 'all' : 'imported')}>
            <span>Imported</span>
            <strong>{counts.imported}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'attention' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'attention' ? 'all' : 'attention')}>
            <span>Needs attention</span>
            <strong>{counts.attention}</strong>
          </button>
        </div>
      </div>

      <section className="panel models-section-panel">
        <div className="models-section-heading">
          <div>
            <p className="panel-kicker">Ready to use</p>
            <h3>{readyModels.length === 0 ? 'Nothing ready yet' : `${readyModels.length} ready ${readyModels.length === 1 ? 'model' : 'models'}`}</h3>
          </div>
          <p className="model-summary">These models are ready for a chat right now, either locally or through a connected API.</p>
        </div>

        {readyModels.length === 0 ? (
          <div className="empty-state">No ready models yet. Download, import, or connect one below.</div>
        ) : (
          <div className="models-card-grid">
            {readyModels.map((model) => {
              const runnable = isRunnableModel(model)
              const canLoad = canLoadIntoRuntime(model)
              const external = isExternalModel(model)
              const active = runtime?.active_model_id === model.id
              const selected = selectedModelId === model.id

              return (
                <article key={model.id} className={`model-card models-model-card ${active ? 'active-model-card' : ''} ${selected ? 'selected-model-card' : ''}`}>
                  <div className="models-card-head">
                    <div className="models-card-title">
                      <strong>{model.name}</strong>
                      <span>{formatModelMeta(model)}</span>
                    </div>
                    <div className={`status-pill ${statusTone(model)}`}>{getModelStatusLabel(model)}</div>
                  </div>

                  <div className="models-card-tags">
                    {active && <div className="pin-badge">Loaded now</div>}
                    {selected && <div className="pin-badge">Next chat</div>}
                    {external && <div className="pin-badge">API connected</div>}
                    {model.model_path && !external && <div className="pin-badge">Saved locally</div>}
                  </div>

                  <div className="models-card-copy-stack">
                    <p className="model-summary">{describeModelState(model)}</p>
                    <p className="model-summary">{formatModelOrigin(model)}</p>
                    {model.capability_summary && <p className="model-summary">Capability: {model.capability_summary}</p>}
                  </div>

                  <CapabilityChecklist model={model} runnable={runnable} canLoad={canLoad} />
                  <NonRunnableCallouts model={model} />

                  {model.backend_lanes?.length > 0 && (
                    <div className="models-card-tags">
                      {model.backend_lanes.map((lane) => <div key={lane} className="pin-badge">{lane}</div>)}
                    </div>
                  )}

                  {model.install_error && <p className="library-error-copy">{model.install_error}</p>}

                  <div className="models-card-actions">
                    <button className="ghost-button" onClick={() => setSelectedModelId(model.id)}>{selected ? 'Chosen for next chat' : 'Use for next chat'}</button>
                    {canLoad && !external && <button className="primary-button" onClick={() => activateModel(model.id)}>{active ? 'Loaded now' : 'Load now'}</button>}
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </section>

      <section className="panel models-section-panel models-catalog-panel-clean">
        <div className="models-section-heading models-section-heading-catalog">
          <div>
            <p className="panel-kicker">Download new models</p>
            <h3>Public model format lanes</h3>
          </div>
          <p className="model-summary">Downloads come from Hugging Face. Fathom’s verified local chat lanes are narrow: TinyStories/DistilGPT-2-style GPT-2, tiny Llama including tied embeddings, tiny Qwen2, tiny Phi, tiny Mistral, and tiny Gemma through the custom Rust SafeTensors runtime. MiniLM embedding packages belong to the separate local-embeddings-retrieval lane and are not chat models.</p>
        </div>

        {catalogError && <p className="library-error-copy">{catalogError}</p>}

        {catalogLoading && discoverCatalogItems.length === 0 ? (
          <div className="empty-state">Loading Hugging Face catalog entries…</div>
        ) : discoverCatalogItems.length === 0 ? (
          <div className="empty-state">No catalog model formats matched that search.</div>
        ) : (
          <>
            <div className="models-card-grid models-catalog-grid-clean">
              {discoverCatalogItems.map((item) => {
                const localMatch = findCatalogMatch(models, item)
                const runnable = isRunnableModel(localMatch)
                const canLoad = canLoadIntoRuntime(localMatch)
                const active = runtime?.active_model_id === localMatch?.id
                const selected = selectedModelId === localMatch?.id
                const installKey = catalogInstallKey(item)
                const installing = pendingCatalogInstalls.has(installKey)
                const licenseAckRequired = Boolean(item.license_acknowledgement_required)
                const licenseAcknowledged = catalogLicenseAcknowledgements.has(installKey)
                const installDisabled = installing || (licenseAckRequired && !licenseAcknowledged)
                const licenseStatus = item.license_status || (item.license === 'unknown' ? 'unknown' : 'permissive')

                return (
                  <article key={item.catalog_id} className={`model-card models-model-card models-catalog-card-clean ${active ? 'active-model-card' : ''} ${selected ? 'selected-model-card' : ''} ${installing ? 'is-installing' : ''}`} aria-busy={installing ? 'true' : undefined}>
                    <div className="models-card-head">
                      <div className="models-card-title">
                        <strong>{formatCatalogTitle(item)}</strong>
                        <span>{formatRemoteMeta(item)}</span>
                      </div>
                      {installing ? <div className="status-pill warm">Installing…</div> : localMatch ? <div className={`status-pill ${statusTone(localMatch)}`}>{getModelStatusLabel(localMatch)}</div> : null}
                    </div>

                    <div className="models-card-tags">
                      {isRecommendedNonGgufDemo(item) && <div className="pin-badge">Recommended non-GGUF demo</div>}
                      {isOnnxEmbeddingFixture(item) && <div className="pin-badge">ONNX embedding</div>}
                      {item?.repo_id === 'sentence-transformers/all-MiniLM-L6-v2' && <div className="pin-badge">SafeTensors embedding</div>}
                      {isEmbeddingOnlyFixture(item) && <div className="pin-badge">Not a chat model</div>}
                      {active && <div className="pin-badge">Loaded now</div>}
                      {selected && <div className="pin-badge">Next chat</div>}
                      {installing && <div className="pin-badge">Downloading and verifying…</div>}
                      {localMatch?.model_path && <div className="pin-badge">Saved locally</div>}
                      <div className="pin-badge">License: {item.license || 'unknown'}</div>
                      {licenseAckRequired && <div className="pin-badge">License acknowledgement required</div>}
                    </div>

                    <dl className="models-definition-grid">
                      <div>
                        <dt>Repo</dt>
                        <dd title={item.repo_id}>{item.repo_id}</dd>
                      </div>
                      <div>
                        <dt>File</dt>
                        <dd title={item.filename}>{item.filename}</dd>
                      </div>
                      <div>
                        <dt>Download size</dt>
                        <dd>{item.size_bytes ? formatBytes(item.size_bytes) : 'Checked before download'}</dd>
                      </div>
                      <div>
                        <dt>License status</dt>
                        <dd>{licenseStatus}</dd>
                      </div>
                    </dl>

                    {item.description && <p className="model-summary">{item.description}</p>}
                    {isRecommendedNonGgufDemo(item) && <p className="model-summary">Start here to test Fathom’s first real non-GGUF local path: Hugging Face SafeTensors loaded by Fathom’s custom Rust backend.</p>}
                    {item?.repo_id === 'sentence-transformers/all-MiniLM-L6-v2' && <p className="model-summary">This is for retrieval experiments through the default Candle/SafeTensors MiniLM embedding lane. It returns local vectors, but it must not appear as a chat/generation model.</p>}
                    {isOnnxEmbeddingFixture(item) && <p className="model-summary">This is for retrieval experiments through the feature-gated ONNX MiniLM embedding lane. It can expose embeddings when ONNX Runtime is compiled, but it must not appear as a chat/generation model.</p>}
                    <p className="model-summary">Download source: Hugging Face. Runtime promise: files are saved and inspected, but no fake generation is enabled.</p>
                    {item.license_warning && <p className="library-error-copy">{item.license_warning}</p>}
                    {licenseAckRequired && (
                      <label className="models-license-ack">
                        <input
                          type="checkbox"
                          checked={licenseAcknowledged}
                          onChange={(event) => {
                            setCatalogLicenseAcknowledgements((current) => {
                              const next = new Set(current)
                              if (event.target.checked) next.add(installKey)
                              else next.delete(installKey)
                              return next
                            })
                          }}
                          disabled={installing}
                        />
                        <span>I reviewed the listed license status and want to download this catalog entry.</span>
                      </label>
                    )}
                    {localMatch?.capability_summary && <p className="model-summary">Capability: {localMatch.capability_summary}</p>}
                    {localMatch && <CapabilityChecklist model={localMatch} runnable={runnable} canLoad={canLoad} />}
                    {localMatch && <NonRunnableCallouts model={localMatch} />}
                    {localMatch?.backend_lanes?.length > 0 && (
                      <div className="models-card-tags">
                        {localMatch.backend_lanes.map((lane) => <div key={lane} className="pin-badge">{lane}</div>)}
                      </div>
                    )}
                    {localMatch?.install_error && <p className="library-error-copy">{localMatch.install_error}</p>}

                    <div className="models-card-actions">
                      {(!localMatch || localMatch.status === 'not_installed' || localMatch.status === 'failed') && <button className="primary-button" onClick={() => installCatalogItem(item)} disabled={installDisabled}>{installing ? 'Downloading and verifying…' : licenseAckRequired && !licenseAcknowledged ? 'Acknowledge license to download' : isRecommendedNonGgufDemo(item) ? 'Download SafeTensors demo' : isEmbeddingOnlyFixture(item) ? 'Download embedding model' : 'Download from Hugging Face'}</button>}
                      {(localMatch?.status === 'downloading' || localMatch?.status === 'canceling') && <button className="ghost-button" onClick={() => cancelModelDownload(localMatch.id)} disabled={localMatch.status === 'canceling' || installing}>{localMatch.status === 'canceling' ? 'Canceling…' : 'Cancel download'}</button>}
                      {runnable && <button className="ghost-button" onClick={() => setSelectedModelId(localMatch.id)} disabled={installing}>{selected ? 'Chosen for next chat' : 'Use for next chat'}</button>}
                      {canLoad && <button className="primary-button" onClick={() => activateModel(localMatch.id)} disabled={installing}>{active ? 'Loaded now' : localMatch.status === 'registered' ? 'Load now and confirm file' : 'Load now'}</button>}
                    </div>
                  </article>
                )
              })}
            </div>

            {catalogNextCursor && (
              <div className="library-load-more-row">
                <button className="ghost-button" onClick={loadMoreCatalog} disabled={catalogLoadingMore}>
                  {catalogLoadingMore ? 'Loading more models…' : 'Load more models'}
                </button>
              </div>
            )}
          </>
        )}
      </section>

      <div className="models-setup-grid">
        <div className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Import a local model artifact</p>
              <h3>Bring in a model you already downloaded</h3>
            </div>
            <p className="model-summary">Keep the first step simple. Fathom can generate the internal ID, then confirm the file on first load.</p>
          </div>

          <div className="models-form-stack">
            <div className="composer-actions import-grid">
              <input value={registerForm.name} onChange={(e) => setRegisterForm((form) => ({ ...form, name: e.target.value }))} placeholder="Model name" />
              <input value={registerForm.model_path} onChange={(e) => setRegisterForm((form) => ({ ...form, model_path: e.target.value }))} placeholder="/path/to/model folder, .safetensors, .gguf, .onnx…" />
            </div>

            <button className="ghost-button subtle-action import-advanced-toggle" onClick={() => setShowImportAdvanced((current) => !current)}>
              {showImportAdvanced ? 'Hide advanced options' : 'Show advanced options'}
            </button>

            {showImportAdvanced && (
              <div className="composer-actions import-grid import-grid-advanced">
                <input value={registerForm.id} onChange={(e) => setRegisterForm((form) => ({ ...form, id: e.target.value }))} placeholder="Internal model ID (optional)" />
                <input value={registerForm.runtime_model_name} onChange={(e) => setRegisterForm((form) => ({ ...form, runtime_model_name: e.target.value }))} placeholder="Runtime name override (optional)" />
              </div>
            )}

            <div className="import-callout-row">
              <p className="model-summary">After import, Fathom inspects the artifact and backend lane. Load is offered only when a real runnable backend lane exists.</p>
              <button className="primary-button" onClick={registerModel}>Import local model</button>
            </div>
          </div>
        </div>

        <div className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Connect a hosted model</p>
              <h3>Link an external OpenAI-compatible API</h3>
            </div>
            <p className="model-summary">Add the provider label, base URL, key, and remote model name. Fathom keeps the key local and verifies the connection before marking it ready.</p>
          </div>

          <div className="models-form-stack">
            <div className="composer-actions import-grid">
              <input value={externalForm.name} onChange={(e) => setExternalForm((form) => ({ ...form, name: e.target.value }))} placeholder="Display name" />
              <input value={externalForm.model_name} onChange={(e) => setExternalForm((form) => ({ ...form, model_name: e.target.value }))} placeholder="Remote model name" />
            </div>
            <div className="composer-actions import-grid import-grid-advanced">
              <input value={externalForm.source} onChange={(e) => setExternalForm((form) => ({ ...form, source: e.target.value }))} placeholder="Provider label" />
              <input value={externalForm.api_base} onChange={(e) => setExternalForm((form) => ({ ...form, api_base: e.target.value }))} placeholder="https://api.openai.com/v1" />
            </div>
            <div className="composer-actions import-grid import-grid-advanced">
              <input value={externalForm.id} onChange={(e) => setExternalForm((form) => ({ ...form, id: e.target.value }))} placeholder="Internal model ID (optional)" />
              <input type="password" value={externalForm.api_key} onChange={(e) => setExternalForm((form) => ({ ...form, api_key: e.target.value }))} placeholder="API key" autoComplete="off" />
            </div>
            <div className="import-callout-row">
              <p className="model-summary">Tip: for ChatGPT-style APIs, use the full API base, usually something like <code>https://api.openai.com/v1</code>.</p>
              <button className="primary-button" onClick={connectExternalModel}>Connect external model</button>
            </div>
          </div>
        </div>
      </div>

      {setupModels.length > 0 && (
        <section className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Still needs setup</p>
              <h3>{setupModels.length} model{setupModels.length === 1 ? '' : 's'} still need attention</h3>
            </div>
            <p className="model-summary">This is the short list of models that are still importing, downloading, or need a fix before they can be used confidently.</p>
          </div>

          <div className="models-card-grid">
            {setupModels.map((model) => {
              const runnable = isRunnableModel(model)
              const canLoad = canLoadIntoRuntime(model)
              const external = isExternalModel(model)
              const selected = selectedModelId === model.id
              const active = runtime?.active_model_id === model.id

              return (
                <article key={model.id} className={`model-card models-model-card ${active ? 'active-model-card' : ''} ${selected ? 'selected-model-card' : ''}`}>
                  <div className="models-card-head">
                    <div className="models-card-title">
                      <strong>{model.name}</strong>
                      <span>{formatModelMeta(model)}</span>
                    </div>
                    <div className={`status-pill ${statusTone(model)}`}>{getModelStatusLabel(model)}</div>
                  </div>

                  <div className="models-card-copy-stack">
                    <p className="model-summary">{describeModelState(model)}</p>
                    <p className="model-summary">{formatModelOrigin(model)}</p>
                    {model.capability_summary && <p className="model-summary">Capability: {model.capability_summary}</p>}
                  </div>

                  <CapabilityChecklist model={model} runnable={runnable} canLoad={canLoad} />
                  <NonRunnableCallouts model={model} />

                  {model.backend_lanes?.length > 0 && (
                    <div className="models-card-tags">
                      {model.backend_lanes.map((lane) => <div key={lane} className="pin-badge">{lane}</div>)}
                    </div>
                  )}

                  {!external && (model.status === 'downloading' || model.status === 'canceling' || model.progress) && (
                    <div className="progress-wrap">
                      <div className="progress-bar"><div style={{ width: `${model.progress || 0}%` }} /></div>
                      <small>{formatDownloadCopy(model)}</small>
                    </div>
                  )}

                  {model.install_error && <p className="library-error-copy">{model.install_error}</p>}

                  <div className="models-card-actions">
                    {!external && (model.status === 'not_installed' || model.status === 'failed') && <button className="primary-button" onClick={() => installModel(model.id)}>{model.status === 'failed' ? 'Retry download' : 'Download'}</button>}
                    {!external && (model.status === 'downloading' || model.status === 'canceling') && <button className="ghost-button" onClick={() => cancelModelDownload(model.id)} disabled={model.status === 'canceling'}>{model.status === 'canceling' ? 'Canceling…' : 'Cancel download'}</button>}
                    {!external && !model.model_path && (model.status === 'ready' || model.status === 'registered') && <button className="ghost-button" disabled>Re-import with a local file path</button>}
                    {runnable && <button className="ghost-button" onClick={() => setSelectedModelId(model.id)}>{selected ? 'Chosen for next chat' : 'Use for next chat'}</button>}
                    {canLoad && !external && <button className="primary-button" onClick={() => activateModel(model.id)}>{active ? 'Loaded now' : model.status === 'registered' ? 'Load now and confirm file' : 'Load now'}</button>}
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      )}
    </section>
  )
}
