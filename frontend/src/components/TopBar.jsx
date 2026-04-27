import { clampText, formatPreview, formatSidebarDate } from '../lib/formatters'
import { describeModelState, getModelStatusLabel, isRunnableModel } from '../lib/modelState'

const titles = {
  chat: 'Chat',
  library: 'Models',
  analytics: 'Analytics',
  history: 'History',
  memory: 'Memory',
  system: 'System',
}

const isMeaningfulConversationMessage = (message) =>
  typeof message?.content === 'string' && message.content.trim() && !message.content.startsWith('Conversation created.')

const navItems = [
  { id: 'chat', label: 'Chat' },
  { id: 'library', label: 'Models' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'history', label: 'History' },
  { id: 'memory', label: 'Memory' },
  { id: 'system', label: 'System' },
]

export default function TopBar({ tab, setTab, selectedConversation, runtime, theme, setTheme, selectedModelId, setSelectedModelId, models }) {
  const rawConversationTitle = selectedConversation?.title?.trim()
  const hasCustomConversationTitle = Boolean(rawConversationTitle && rawConversationTitle.toLowerCase() !== 'new conversation')
  const latestConversationMessage = [...(selectedConversation?.messages || [])].reverse().find((message) => isMeaningfulConversationMessage(message))
  const untitledConversationLabel = selectedConversation
    ? `${formatPreview(latestConversationMessage?.content, 42)} · ${formatSidebarDate(selectedConversation.updated_at) || 'New chat'}`
    : 'Ready when you are'
  const heading = tab === 'chat'
    ? (hasCustomConversationTitle ? clampText(rawConversationTitle, 72) : '')
    : titles[tab] || 'Fathom'
  const activeModel = models.find((model) => model.id === runtime?.active_model_id)
  const selectedModel = models.find((model) => model.id === selectedModelId)
  const activeModelLabel = activeModel?.name || 'Nothing loaded now'
  const selectedModelLabel = selectedModel?.name || 'Nothing chosen for next chat'
  const selectedModelSummary = selectedModel ? describeModelState(selectedModel) : 'Choose the model you want Fathom to use next.'

  if (tab === 'chat') {
    return (
      <header className="topbar topbar-chat">
        <div className="topbar-chat-row">
          <div className="topbar-chat-brand">Fathom</div>
          <div className="topbar-chat-center" title={hasCustomConversationTitle ? rawConversationTitle : untitledConversationLabel}>
            {hasCustomConversationTitle ? clampText(rawConversationTitle, 64) : untitledConversationLabel}
          </div>
          <div className="topbar-chat-actions">
            <button className="topbar-chat-icon" aria-label="Toggle color theme" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? '◐' : '◑'}
            </button>
          </div>
        </div>
        <div className="mobile-nav" aria-label="Primary navigation">
          {navItems.map((item) => (
            <button key={item.id} className={`mobile-nav-item ${tab === item.id ? 'active' : ''}`} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
      </header>
    )
  }

  return (
    <header className="topbar topbar-page">
      <div className="topbar-page-row">
        <div className="topbar-chat-brand">Fathom</div>
        <div className="topbar-chat-center topbar-page-center" title={heading}>{heading}</div>
        <div className="topbar-chat-actions">
          <label className="topbar-chat-picker" title={selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what new chats should use next.'}>
            <select className="topbar-select topbar-select-chat" aria-label="Use for next chat" value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
              {models.map((model) => {
                const runnable = isRunnableModel(model)
                return (
                  <option key={model.id} value={model.id} disabled={!runnable}>
                    {model.name}
                  </option>
                )
              })}
            </select>
          </label>
          <button className="topbar-chat-icon" aria-label="Toggle color theme" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? '◐' : '◑'}
          </button>
        </div>
      </div>
      {tab !== 'library' && (
        <div className="topbar-status-strip" aria-label="Model status">
          <div className={`status-pill topbar-status-pill ${runtime?.loaded_now ? 'ready' : runtime?.ready ? 'warm' : ''}`} title={activeModelLabel}>
            <span className="topbar-status-label">Loaded now</span>
            <strong>{clampText(activeModelLabel, 32)}</strong>
          </div>
          <div className="status-pill topbar-status-pill topbar-status-pill-wide" title={selectedModelSummary}>
            <span className="topbar-status-label">Next chat</span>
            <strong>{clampText(selectedModelLabel, 36)}</strong>
          </div>
        </div>
      )}
      <div className="mobile-nav" aria-label="Primary navigation">
        {navItems.map((item) => (
          <button key={item.id} className={`mobile-nav-item ${tab === item.id ? 'active' : ''}`} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
    </header>
  )
}
