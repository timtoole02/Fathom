export default function ApiView({ runtime, selectedModel }) {
  return (
    <section className="panel api-panel">
      <div className="panel-header-row">
        <div>
          <p className="panel-kicker">Connectivity</p>
          <h2>OpenAI-compatible local API surface</h2>
        </div>
        <div className={`status-pill ${runtime?.ready ? 'ready' : 'warm'}`}>{runtime?.loaded_now ? 'Local generation loaded' : 'No fake generation'}</div>
      </div>
      <div className="api-grid">
        <div className="api-card">
          <strong>Chat completions</strong>
          <code>{runtime?.api_base}/chat/completions · truthful generation for verified custom Rust SafeTensors models only</code>
        </div>
        <div className="api-card">
          <strong>Model listing</strong>
          <code>{runtime?.api_base}/models · chat-runnable models only</code>
        </div>
        <div className="api-card">
          <strong>Embeddings</strong>
          <code>{runtime?.api_base}/embeddings · verified MiniLM float vectors only</code>
        </div>
        <div className="api-card">
          <strong>Health check</strong>
          <code>{runtime?.api_base}/health</code>
        </div>
        <div className="api-card wide">
          <strong>Truthful API check</strong>
          <pre>{`curl ${runtime?.api_base}/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "${selectedModel?.id}",
    "messages": [{"role": "user", "content": "Hello from Fathom"}]
  }'`}</pre>
        </div>
      </div>
    </section>
  )
}
