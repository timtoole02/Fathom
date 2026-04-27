export default function ApiView({ runtime, selectedModel }) {
  const apiBase = runtime?.api_base || '/v1'

  return (
    <section className="panel api-panel">
      <div className="panel-header-row">
        <div>
          <p className="panel-kicker">Connectivity</p>
          <h2>OpenAI-compatible local API surface</h2>
        </div>
        <div className={`status-pill ${runtime?.ready ? 'ready' : 'warm'}`}>{runtime?.loaded_now ? 'Local generation loaded' : 'No fake generation'}</div>
      </div>
      <p className="model-summary">Fathom exposes a narrow local <code>/v1</code> surface. It is compatible where implemented, not a full OpenAI clone: <code>stream: true</code> is refused because streaming is not implemented, base64 embeddings are refused, chat requests for embedding/GGUF/PyTorch/unsupported models are refused, and unknown model IDs return a JSON error envelope.</p>
      <div className="api-grid">
        <div className="api-card">
          <strong>Chat completions</strong>
          <code>{apiBase}/chat/completions · non-streaming chat for verified custom Rust SafeTensors models only</code>
        </div>
        <div className="api-card">
          <strong>Model listing</strong>
          <code>{apiBase}/models · chat-runnable models only; embeddings and metadata-only packages stay out</code>
        </div>
        <div className="api-card">
          <strong>Embeddings</strong>
          <code>{apiBase}/embeddings · verified MiniLM float vectors only; base64 is refused</code>
        </div>
        <div className="api-card">
          <strong>Health check</strong>
          <code>{apiBase}/health</code>
        </div>
        <div className="api-card wide">
          <strong>Truthful API check</strong>
          <pre>{`curl ${apiBase}/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "${selectedModel?.id || 'chat-runnable-model-id'}",
    "messages": [{"role": "user", "content": "Hello from Fathom"}],
    "stream": false
  }'`}</pre>
        </div>
      </div>
    </section>
  )
}
