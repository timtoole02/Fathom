import { describeModelState } from '../lib/modelState'

function runtimeWarningText(warning) {
  if (warning?.type === 'model_state_recovered') {
    return warning.message || 'Recovered from unreadable model registry; the corrupt file was preserved for inspection.'
  }
  return warning?.message || null
}

export default function RuntimeView({ runtime, selectedModel }) {
  const runtimeWarnings = (runtime?.warnings || []).map(runtimeWarningText).filter(Boolean)

  return (
    <section className="runtime-grid">
      <div className="panel">
        <p className="panel-kicker">Runtime</p>
        <h2>Truthful local execution</h2>
        <div className="runtime-stat-grid">
          <div className="runtime-stat"><span>Backend</span><strong>{runtime?.backend}</strong></div>
          <div className="runtime-stat"><span>Inference engine</span><strong>{runtime?.engine || 'Fathom'}</strong></div>
          <div className="runtime-stat"><span>API base</span><strong>{runtime?.api_base}</strong></div>
          <div className="runtime-stat"><span>Loaded model</span><strong>{runtime?.active_model_id || 'none'}</strong></div>
        </div>
      </div>
      <div className="panel">
        <p className="panel-kicker">Readiness</p>
        <h2>Operational checklist</h2>
        <div className="activity-feed">
          {runtimeWarnings.map((warning) => <div key={warning} className="activity-item runtime-warning-item">{warning}</div>)}
          <div className="activity-item">Persistent conversations loaded from local SQLite store</div>
          <div className="activity-item">Memory index is loaded from backend state</div>
          <div className="activity-item">Current loaded model id: {runtime?.active_model_id || 'none'}</div>
          <div className="activity-item">Selected model state: {describeModelState(selectedModel)}</div>
          <div className="activity-item">/v1 compatibility surface is online at {runtime?.api_base}; chat completions only runs models that are truly runnable through the current narrow custom Rust SafeTensors lanes.</div>
        </div>
      </div>
    </section>
  )
}
