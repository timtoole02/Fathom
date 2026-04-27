import { buildAssistantMetricSummary, buildAssistantMetricTitle, clampText, formatDate, formatRate } from '../lib/formatters'
import { describeModelState, getModelStatusLabel, isRunnableModel } from '../lib/modelState'

const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

export default function ChatWorkspace({
  selectedConversation,
  selectedModel,
  selectedModelId,
  setSelectedModelId,
  models,
  runtime,
  latestAssistantMessage,
  pendingConversation,
  composer,
  setComposer,
  saveToMemory,
  sendMessage,
  sending,
  selectedModelRunnable,
  setTab,
}) {
  const visibleMessages = (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message))
  const pendingPrompt = (sending ? composer.trim() : '') || (pendingConversation?.content || '')
  const isFreshThread = !selectedConversation || (visibleMessages.length === 0 && !pendingPrompt)
  const latestVisibleAssistantMessage = [...visibleMessages].reverse().find((message) => message.role === 'assistant') || latestAssistantMessage
  const starterPrompts = [
    'The best part of local AI is',
    'In one short paragraph, explain Rust.',
    'A tiny inference engine should',
    'Today I learned that',
    'The next practical step is',
  ]

  const handleComposerKeyDown = async (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (!sending && selectedModelRunnable) {
        await sendMessage()
      }
    }
  }

  const rawConversationTitle = selectedConversation?.title?.trim()
  const hasCustomConversationTitle = Boolean(rawConversationTitle && rawConversationTitle.toLowerCase() !== 'new conversation')
  const conversationLabel = clampText(hasCustomConversationTitle ? rawConversationTitle : 'Untitled chat', 30)
  const lastUpdated = selectedConversation?.updated_at ? formatDate(selectedConversation.updated_at) : null
  const speedLabel = latestVisibleAssistantMessage?.tokens_out_per_sec !== null && latestVisibleAssistantMessage?.tokens_out_per_sec !== undefined
    ? formatRate(latestVisibleAssistantMessage.tokens_out_per_sec)
    : 'Waiting for first reply'
  const runnableModels = models.filter((model) => isRunnableModel(model))
  const hasRunnableChoices = runnableModels.length > 0
  const modelPickerTitle = selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what Fathom should use for this chat.'
  const selectedModelMeta = !selectedModelRunnable
    ? describeModelState(selectedModel)
    : runtime?.loaded_now && runtime?.active_model_id === selectedModelId
      ? 'Loaded now'
      : isFreshThread
        ? 'Ready to chat'
        : speedLabel

  const renderModelPicker = () => {
    if (!hasRunnableChoices) {
      return (
        <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>
          Choose model
        </button>
      )
    }

    return (
      <label className="composer-model-picker" title={modelPickerTitle}>
        <span className="composer-tool-label">Model</span>
        <select
          className="composer-model-select"
          aria-label="Choose model for chat"
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={sending}
        >
          {runnableModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </label>
    )
  }

  return (
    <section className={`chat-layout chat-layout-gemini view-stack ${isFreshThread ? 'chat-layout-empty' : ''}`}>
      {selectedConversation && (
        <div className="mobile-conversation-bar" aria-label="Conversation navigation">
          <button className="ghost-button mobile-conversation-trigger" onClick={() => setTab('history')}>
            <span>Conversations</span>
            <strong title={rawConversationTitle || 'Untitled chat'}>{conversationLabel}</strong>
          </button>
          <div className="mobile-conversation-status">
            {lastUpdated ? `Updated ${lastUpdated}` : 'Current thread'}
          </div>
        </div>
      )}

      <div className={`chat-canvas ${isFreshThread ? 'chat-canvas-empty' : ''}`}>
        {isFreshThread ? (
          <div className="chat-empty-shell chat-empty-shell-gemini">
            <div className="chat-empty-stage">
              <div className="chat-empty-hero chat-empty-hero-gemini">
                <p className="chat-empty-greeting">New chat</p>
                <h2>{selectedModelRunnable ? 'What should we work on today?' : 'Choose a runnable model first'}</h2>
              </div>

              <div className="composer composer-gemini composer-gemini-stage">
                <textarea className="composer-input composer-input-gemini composer-input-gemini-stage" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={2} placeholder={selectedModelRunnable ? 'Ask me anything' : 'Pick a ready model first, then start your chat'} disabled={sending || !selectedModelRunnable} />
                <div className="composer-gemini-footer composer-gemini-footer-stage">
                  <div className="composer-gemini-tools composer-gemini-tools-stage">
                    {renderModelPicker()}
                    <span className="composer-meta-pill">{selectedModelMeta}</span>
                    {selectedModelRunnable && <span className="composer-meta-pill composer-meta-pill-advice">Small base models do best with short completion prompts.</span>}
                    {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>Open Library</button>}
                  </div>
                  <div className="composer-gemini-actions composer-gemini-actions-stage">
                    <button className="primary-button composer-send-button" onClick={sendMessage} disabled={sending || !selectedModelRunnable}>{sending ? 'Sending…' : 'Send'}</button>
                  </div>
                </div>
              </div>

              {selectedModelRunnable && (
                <div className="chat-starter-chips chat-starter-chips-centered" aria-label="Conversation ideas">
                  {starterPrompts.map((prompt) => (
                    <button key={prompt} type="button" className="chat-starter-chip chat-starter-chip-stage" onClick={() => setComposer(prompt)}>
                      {prompt}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <>
            {!selectedModelRunnable && (
              <div className="setup-card setup-card-inline setup-card-gemini">
                <div>
                  <p className="panel-kicker">Before you chat</p>
                  <h2>Choose a runnable model</h2>
                  <p className="hero-summary">{describeModelState(selectedModel)}</p>
                </div>
                <div className="composer-actions single-action-row">
                  <button className="primary-button" onClick={() => setTab('library')}>Open Library</button>
                </div>
              </div>
            )}

            <div className="chat-thread chat-thread-gemini">
              {visibleMessages.length === 0 && !pendingPrompt && <div className="empty-state empty-state-chat">Pick a ready model, then send the first message when you’re ready.</div>}
              {visibleMessages.map((message) => {
                const metricSummary = buildAssistantMetricSummary(message)
                const metricTitle = buildAssistantMetricTitle(message)

                return (
                  <article key={message.id} className={`message-row message-row-gemini ${message.role}`}>
                    <div className={`message-bubble message-bubble-gemini ${message.role}`}>
                      {message.role === 'assistant' && metricSummary && (
                        <div className="message-heading message-heading-clean" title={metricTitle}>
                          <span className="message-micro-meta">{metricSummary}</span>
                        </div>
                      )}
                      <p>{message.content}</p>
                    </div>
                  </article>
                )
              })}
              {pendingPrompt && (
                <>
                  <article className="message-row message-row-gemini user pending">
                    <div className="message-bubble message-bubble-gemini user pending">
                      <p>{pendingPrompt}</p>
                    </div>
                  </article>
                  <article className="message-row message-row-gemini assistant pending">
                    <div className="message-thinking-loader" aria-hidden="true">
                      <span className="message-thinking-pacman message-thinking-pacman-top" />
                      <span className="message-thinking-pacman message-thinking-pacman-bottom" />
                      <span className="message-thinking-pellet message-thinking-pellet-1" />
                      <span className="message-thinking-pellet message-thinking-pellet-2" />
                      <span className="message-thinking-pellet message-thinking-pellet-3" />
                    </div>
                    <div className="message-bubble message-bubble-gemini assistant pending">
                      <div className="message-heading message-heading-clean">
                        <span className="message-micro-meta">Thinking…</span>
                      </div>
                      <p className="message-placeholder-copy">Working on it…</p>
                    </div>
                  </article>
                </>
              )}
            </div>
          </>
        )}
      </div>

      {!isFreshThread && (
        <div className="composer composer-gemini composer-gemini-floating">
          <textarea className="composer-input composer-input-gemini" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={3} placeholder={selectedModelRunnable ? 'Ask me anything' : 'Pick a ready model first, then start your chat'} disabled={sending || !selectedModelRunnable} />
          <div className="composer-gemini-footer">
            <div className="composer-gemini-tools">
              {renderModelPicker()}
              <span className="composer-meta-pill">{selectedModelMeta}</span>
              {selectedModelRunnable && <span className="composer-meta-pill composer-meta-pill-advice">Use short completion-style prompts for DistilGPT-2-class demos.</span>}
              {selectedModelRunnable && <button className="ghost-button subtle-action" onClick={saveToMemory} disabled={sending}>Save to memory</button>}
            </div>
            <div className="composer-gemini-actions">
              {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button" onClick={() => setTab('library')}>Open Library</button>}
              <button className="primary-button composer-send-button" onClick={sendMessage} disabled={sending || !selectedModelRunnable}>{sending ? 'Sending…' : 'Send'}</button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
