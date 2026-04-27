import { describeModelState } from '../lib/modelState'

const CAPABILITY_STAGES = [
  { label: 'Detected', detail: 'Fathom recognizes the artifact or API connection.' },
  { label: 'Metadata-readable', detail: 'Config, tokenizer, or format metadata can be inspected safely.' },
  { label: 'Backend lane', detail: 'A concrete runtime lane has claimed responsibility.' },
  { label: 'Runnable', detail: 'The lane can produce real responses now.' },
]

function runtimeReadinessLabel(runtime) {
  if (runtime?.loaded_now) return 'Loaded and answering now'
  if (runtime?.ready) return 'Ready to load on demand'
  return 'Waiting for a ready model'
}

function normalizeCapabilityStatus(status) {
  return (status || 'Unknown').replace(/([a-z])([A-Z])/g, '$1 $2')
}

function laneTone(status) {
  if (status === 'Runnable') return 'ready'
  if (status === 'Blocked' || status === 'Unsupported') return 'blocked'
  return 'warm'
}

function runtimeWarningText(warning) {
  if (warning?.type === 'model_state_recovered') {
    return warning.message || 'Recovered from unreadable model registry; the corrupt file was preserved for inspection.'
  }
  return warning?.message || null
}

export default function SystemView({ runtime, selectedModel }) {
  const runtimePill = runtimeReadinessLabel(runtime)
  const selectedModelName = selectedModel?.name || 'No next-chat model selected'
  const apiBase = runtime?.api_base || 'Local API unavailable'
  const runtimeWarnings = (runtime?.warnings || []).map(runtimeWarningText).filter(Boolean)

  return (
    <section className="view-stack system-layout-single view-shell">
      <div className="panel panel-hero system-hero system-hero-separated">
        <div className="view-hero-copy">
          <p className="panel-kicker">System</p>
          <h2>Runtime, readiness, and local API access</h2>
          <p className="hero-summary">This is the operational view. It keeps runtime health, model readiness, and developer connection details in one calmer place while Library stays focused on browsing and setup.</p>
        </div>
        <div className="view-hero-stats system-hero-pills system-hero-pills-polished">
          <div className={`status-pill ${runtime?.ready ? 'ready' : 'warm'}`}>{runtimePill}</div>
          <div className="status-pill">{apiBase}</div>
        </div>
      </div>

      <div className="runtime-grid runtime-grid-polished">
        <div className="panel panel-section">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Runtime</p>
              <h2>What Fathom can do right now</h2>
            </div>
            <p className="model-summary">A quick operational summary of the local engine, loaded model, and what is ready to wake up on demand.</p>
          </div>
          <div className="runtime-stat-grid">
            <div className="runtime-stat"><span>Runtime state</span><strong>{runtime?.loaded_now ? 'Loaded now' : runtime?.ready ? 'Ready to load' : 'Not ready yet'}</strong></div>
            <div className="runtime-stat"><span>Local engine</span><strong>{runtime?.engine || 'Fathom runtime'}</strong></div>
            <div className="runtime-stat"><span>Loaded model</span><strong>{runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}</strong></div>
            <div className="runtime-stat"><span>Ready models</span><strong>{runtime?.ready_model_count ?? 0} local</strong></div>
            <div className="runtime-stat"><span>Retrieval API</span><strong>Explicit vectors</strong></div>
            <div className="runtime-stat"><span>Embedding lane</span><strong>MiniLM float vectors; not chat-runnable</strong></div>
            <div className="runtime-stat"><span>GPU support</span><strong>{runtime?.llama_server_installed ? 'Backend-specific' : 'Planned backend lane'}</strong></div>
            <div className="runtime-stat"><span>Next chat selection</span><strong>{selectedModelName}</strong></div>
            <div className="runtime-stat"><span>API base</span><strong>{apiBase}</strong></div>
          </div>
        </div>

        <div className="panel panel-section">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Capability</p>
              <h2>What Fathom is handling locally</h2>
            </div>
            <p className="model-summary">A plain-language snapshot of what is already available without reaching outside this machine.</p>
          </div>
          <div className="activity-feed activity-feed-polished">
            {runtimeWarnings.map((warning) => <div key={warning} className="activity-item runtime-warning-item">{warning}</div>)}
            <div className="activity-item">Persistent conversations are already available from local storage.</div>
            <div className="activity-item">Saved memory remains on-device and can be recalled in later chats.</div>
            <div className="activity-item">The retrieval surface supports explicit-vector retrieval indexes plus opt-in <code>fathom.retrieval</code> chat context; embedding generation is limited to verified MiniLM packages through the default SafeTensors lane and optional ONNX runtime feature.</div>
            <div className="activity-item">GPU backend support will be reported by Fathom backends as they come online; the current verified local lane is a custom Rust SafeTensors runtime, not a llama.cpp wrapper.</div>
            <div className="activity-item">Current next-chat model state: {describeModelState(selectedModel)}</div>
            <div className="activity-item">Fathom separates detected/imported/runnable so it never claims a model can run until a backend lane proves it.</div>
            <div className="activity-item">The OpenAI-compatible local API is exposed at {apiBase}.</div>
          </div>
        </div>

        <div className="panel panel-section">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Backend lanes</p>
              <h2>Traffic controller map</h2>
            </div>
            <p className="model-summary">These are Fathom’s runtime lanes: what each lane recognizes, what is planned, and what is actually runnable.</p>
          </div>
          <div className="capability-stage-row" aria-label="Capability checkpoints">
            {CAPABILITY_STAGES.map((stage) => (
              <div key={stage.label} className="capability-stage-card">
                <strong>{stage.label}</strong>
                <span>{stage.detail}</span>
              </div>
            ))}
          </div>
          <div className="backend-lane-grid">
            {(runtime?.backend_lanes || []).map((lane) => (
              <article key={lane.id} className="backend-lane-card">
                <div className="backend-lane-card-head">
                  <strong>{lane.name}</strong>
                  <div className={`status-pill ${laneTone(lane.status)}`}>{normalizeCapabilityStatus(lane.status)}</div>
                </div>
                <p className="model-summary">{lane.summary}</p>
                {lane.formats?.length > 0 && (
                  <div className="backend-lane-detail">
                    <span>Recognizes</span>
                    <div className="models-card-tags">
                      {lane.formats.map((format) => <div key={format} className="pin-badge">{format}</div>)}
                    </div>
                  </div>
                )}
                {lane.blockers?.length > 0 && (
                  <div className="backend-lane-detail">
                    <span>{lane.status === 'Blocked' ? 'Blocked by' : 'Before runnable'}</span>
                    <ul>
                      {lane.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
                    </ul>
                  </div>
                )}
              </article>
            ))}
            {(!runtime?.backend_lanes || runtime.backend_lanes.length === 0) && <div className="activity-item">Backend lane reporting is unavailable.</div>}
          </div>
        </div>
      </div>

      <section className="panel api-panel panel-section">
        <div className="panel-header-row panel-header-row-wide">
          <div>
            <p className="panel-kicker">Developer</p>
            <h2>OpenAI-compatible local API</h2>
            <p className="hero-summary">The local /v1 surface is online for integration checks. Chat completions stays truthful: it only generates for models that validate against the current narrow custom Rust SafeTensors lanes. Embeddings are separate float-vector endpoints for verified MiniLM packages and are not listed as chat-runnable models.</p>
          </div>
          <div className={`status-pill ${runtime?.loaded_now ? 'ready' : 'warm'}`}>{runtime?.loaded_now ? 'Local generation loaded' : 'No fake local generation'}</div>
        </div>
        <div className="api-grid api-grid-polished">
          <div className="api-card">
            <strong>Chat completions</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/chat/completions · verified custom Rust SafeTensors models only` : 'Unavailable until the local API is running'}</code>
          </div>
          <div className="api-card">
            <strong>Models</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/models` : 'Unavailable until the local API is running'}</code>
          </div>
          <div className="api-card">
            <strong>Retrieval API</strong>
            <code>{runtime?.api_base ? '/api/retrieval-indexes · explicit vectors only' : 'Unavailable until the local API is running'}</code>
          </div>
          <div className="api-card">
            <strong>Embeddings</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/embeddings · verified MiniLM float vectors only` : 'Unavailable until the local API is running'}</code>
          </div>
          <div className="api-card">
            <strong>Health</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/health` : 'Unavailable until the local API is running'}</code>
          </div>
          <div className="api-card wide api-card-code">
            <strong>Truthful API check</strong>
            <pre>{runtime?.api_base ? `curl ${runtime.api_base}/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "${selectedModel?.id}",
    "messages": [{"role": "user", "content": "Hello from Fathom"}]
  }'` : 'Start the local runtime to test the truthful API surface.'}</pre>
          </div>
        </div>
      </section>
    </section>
  )
}
