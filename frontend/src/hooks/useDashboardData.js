import { useEffect, useMemo, useRef, useState } from 'react'
import { formatApiErrorDetails, readApiErrorDetails } from '../lib/apiErrors'
import { isExternalModel, isRunnableModel } from '../lib/modelState'

const TAB_STORAGE_KEY = 'fathom.activeTab'
const SELECTED_CONVERSATION_STORAGE_KEY = 'fathom.selectedConversationId'
const SELECTED_MODEL_STORAGE_KEY = 'fathom.selectedModelId'
const PENDING_CHAT_STORAGE_KEY = 'fathom.pendingChat'
const VALID_TABS = new Set(['chat', 'library', 'analytics', 'history', 'memory', 'system'])
const NEW_CHAT_SENTINEL = '__new__'

function getInitialTab() {
  if (typeof window === 'undefined') return 'chat'
  const saved = window.localStorage.getItem(TAB_STORAGE_KEY)
  return saved && VALID_TABS.has(saved) ? saved : 'chat'
}

function getInitialPendingChat() {
  if (typeof window === 'undefined') return null

  try {
    const saved = window.localStorage.getItem(PENDING_CHAT_STORAGE_KEY)
    return saved ? JSON.parse(saved) : null
  } catch {
    window.localStorage.removeItem(PENDING_CHAT_STORAGE_KEY)
    return null
  }
}

function getInitialConversationId() {
  if (typeof window === 'undefined') return null

  const pendingChat = getInitialPendingChat()
  if (pendingChat?.conversationId) return pendingChat.conversationId

  return window.localStorage.getItem(SELECTED_CONVERSATION_STORAGE_KEY) || null
}

function getInitialModelId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY) || ''
}

function slugifyModelId(value) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function buildChatSuccessNotice({ modelType, memoryContext, recovered = false }) {
  const base = recovered
    ? (modelType === 'external' ? 'External placeholder refusal recovered after refresh.' : 'Local reply recovered after refresh.')
    : (modelType === 'external' ? 'External placeholder returned without local generation.' : 'Local response generated successfully.')

  if (!memoryContext?.used_count) return base

  const titles = (memoryContext.memories || []).slice(0, 2).map((memory) => memory.title).filter(Boolean)
  const titleSuffix = titles.length ? ` ${titles.join(', ')}.` : ''
  return `${base} Used ${memoryContext.used_count} saved ${memoryContext.used_count === 1 ? 'memory' : 'memories'}${titleSuffix}`
}

async function readApiErrorNotice(response, fallback) {
  return formatApiErrorDetails(await readApiErrorDetails(response, fallback), fallback)
}

export function useDashboardData({ showNotice, clearNotice }) {
  const [dashboard, setDashboard] = useState(null)
  const [tab, setTab] = useState(getInitialTab)
  const [selectedConversationId, setSelectedConversationId] = useState(getInitialConversationId)
  const [selectedModelId, setSelectedModelId] = useState(getInitialModelId)
  const [search, setSearch] = useState('')
  const [memorySearch, setMemorySearch] = useState('')
  const [composer, setComposer] = useState('')
  const [newChatTitle, setNewChatTitle] = useState('')
  const [sending, setSending] = useState(false)
  const [pendingChat, setPendingChat] = useState(getInitialPendingChat)
  const [resumeAttempted, setResumeAttempted] = useState(false)
  const resumeTimerRef = useRef(null)
  const [registerForm, setRegisterForm] = useState({
    id: '',
    name: '',
    model_path: '',
    runtime_model_name: '',
  })
  const [externalForm, setExternalForm] = useState({
    id: '',
    name: '',
    source: 'OpenAI',
    api_base: 'https://api.openai.com/v1',
    api_key: '',
    model_name: '',
  })

  const loadDashboard = async ({ silent = false } = {}) => {
    try {
      const res = await fetch('/api/dashboard')
      if (!res.ok) throw new Error(await readApiErrorNotice(res, 'Could not load Fathom state.'))
      const data = await res.json()
      setDashboard(data)
      if (!silent) clearNotice()
      setSelectedConversationId((current) => {
        if (current === NEW_CHAT_SENTINEL) return current
        if (!data.conversations.length) return null
        if (current && data.conversations.some((conversation) => conversation.id === current)) return current
        return data.conversations[0].id
      })
      setSelectedModelId((current) => {
        const runnableModels = data.models.filter((model) => isRunnableModel(model))
        if (!runnableModels.length) return ''
        if (current && runnableModels.some((model) => model.id === current)) return current
        if (data.runtime?.loaded_now && data.runtime?.active_model_id && runnableModels.some((model) => model.id === data.runtime.active_model_id)) {
          return data.runtime.active_model_id
        }
        return runnableModels[0]?.id || ''
      })
    } catch (error) {
      if (!silent) showNotice(error.message || 'Could not load Fathom state.', 'error')
    }
  }

  useEffect(() => {
    loadDashboard()
    const interval = setInterval(() => loadDashboard({ silent: true }), 2500)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined' || !VALID_TABS.has(tab)) return
    window.localStorage.setItem(TAB_STORAGE_KEY, tab)
  }, [tab])

  useEffect(() => {
    if (typeof window === 'undefined') return

    if (!selectedConversationId) {
      window.localStorage.removeItem(SELECTED_CONVERSATION_STORAGE_KEY)
      return
    }

    window.localStorage.setItem(SELECTED_CONVERSATION_STORAGE_KEY, selectedConversationId)
  }, [selectedConversationId])

  useEffect(() => {
    if (typeof window === 'undefined') return

    if (!selectedModelId) {
      window.localStorage.removeItem(SELECTED_MODEL_STORAGE_KEY)
      return
    }

    window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, selectedModelId)
  }, [selectedModelId])

  useEffect(() => {
    if (typeof window === 'undefined') return

    if (!pendingChat) {
      window.localStorage.removeItem(PENDING_CHAT_STORAGE_KEY)
      return
    }

    window.localStorage.setItem(PENDING_CHAT_STORAGE_KEY, JSON.stringify(pendingChat))
  }, [pendingChat])

  useEffect(() => {
    if (!pendingChat) {
      setResumeAttempted(false)
      if (resumeTimerRef.current) {
        window.clearTimeout(resumeTimerRef.current)
        resumeTimerRef.current = null
      }
    }
  }, [pendingChat])

  useEffect(() => () => {
    if (resumeTimerRef.current) {
      window.clearTimeout(resumeTimerRef.current)
      resumeTimerRef.current = null
    }
  }, [])

  const conversations = dashboard?.conversations || []
  const memories = dashboard?.memories || []
  const models = dashboard?.models || []
  const runtime = dashboard?.runtime

  useEffect(() => {
    if (!pendingChat) return

    const matchingConversation = conversations.find((conversation) => conversation.id === pendingChat.conversationId)
    if (!matchingConversation) return

    const hasPersistedUserMessage = matchingConversation.messages.some((message) =>
      message.role === 'user' && typeof message.content === 'string' && message.content.trim() === pendingChat.content,
    )
    const hasRealAssistantReply = matchingConversation.messages.some((message) =>
      message.role === 'assistant' && typeof message.content === 'string' && !message.content.startsWith('Conversation created.'),
    )

    if (hasPersistedUserMessage || hasRealAssistantReply) {
      setPendingChat(null)
    }
  }, [conversations, pendingChat])

  const selectedConversation = useMemo(() => {
    if (selectedConversationId === NEW_CHAT_SENTINEL) return null
    return conversations.find((conversation) => conversation.id === selectedConversationId) || conversations[0] || null
  }, [conversations, selectedConversationId])

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) || models.find((model) => isRunnableModel(model)) || null,
    [models, selectedModelId],
  )

  const selectedModelRunnable = isRunnableModel(selectedModel)
  const pendingConversation = pendingChat?.conversationId && selectedConversation?.id === pendingChat.conversationId ? pendingChat : null

  const filteredConversations = useMemo(() => {
    if (!search.trim()) return conversations
    const q = search.toLowerCase()
    return conversations.filter((conversation) =>
      conversation.title.toLowerCase().includes(q)
      || conversation.messages.some((message) => message.content.toLowerCase().includes(q)),
    )
  }, [conversations, search])

  const filteredMemories = useMemo(() => {
    if (!memorySearch.trim()) return memories
    const q = memorySearch.toLowerCase()
    return memories.filter((memory) =>
      memory.title.toLowerCase().includes(q)
      || memory.body.toLowerCase().includes(q)
      || memory.scope.toLowerCase().includes(q),
    )
  }, [memories, memorySearch])

  const latestAssistantMessage = useMemo(
    () => [...(selectedConversation?.messages || [])].reverse().find((message) => message.role === 'assistant'),
    [selectedConversation],
  )

  const createConversationRecord = async ({ manualTitle = '', silent = false } = {}) => {
    const res = await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: manualTitle || null, model_id: selectedModelId || models[0]?.id }),
    })
    if (!res.ok) {
      throw new Error(await readApiErrorNotice(res, 'Could not create the conversation.'))
    }
    const conversation = await res.json()
    await loadDashboard({ silent: true })
    setSelectedConversationId(conversation.id)
    setTab('chat')
    setNewChatTitle('')
    if (!silent) {
      showNotice(manualTitle ? 'Conversation created locally.' : 'Conversation created. Fathom will name it from the chat.', 'success')
    }
    return conversation
  }

  const createConversation = async () => {
    try {
      const manualTitle = newChatTitle.trim()
      await createConversationRecord({ manualTitle })
    } catch (error) {
      showNotice(error.message || 'Could not create the conversation.', 'error')
    }
  }

  const resumePendingMessage = async (pending) => {
    const resumeModelId = pending?.modelId || selectedModelId
    const resumeModel = models.find((model) => model.id === resumeModelId) || selectedModel

    if (!pending?.conversationId || !pending?.content || !resumeModel || !isRunnableModel(resumeModel)) {
      setPendingChat(null)
      return
    }

    setSending(true)

    try {
      if (!isExternalModel(resumeModel) && (!runtime?.loaded_now || runtime?.active_model_id !== resumeModelId)) {
        const activateRes = await fetch(`/api/models/${resumeModelId}/activate`, { method: 'POST' })
        if (!activateRes.ok) {
          throw new Error(await readApiErrorNotice(activateRes, 'Could not reload the selected model after refresh.'))
        }
      }

      const res = await fetch(`/api/conversations/${pending.conversationId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: pending.content, model_id: resumeModelId }),
      })

      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Could not resume the pending chat after refresh.'))
      }

      const updatedConversation = await res.json()
      setPendingChat(null)
      await loadDashboard({ silent: true })
      setSelectedConversationId(updatedConversation.id)
      showNotice(buildChatSuccessNotice({
        modelType: isExternalModel(resumeModel) ? 'external' : 'local',
        memoryContext: updatedConversation.memory_context,
        recovered: true,
      }), 'success')
    } catch (error) {
      setPendingChat(null)
      showNotice(error.message || 'Could not resume the pending chat after refresh.', 'error')
      await loadDashboard({ silent: true })
    } finally {
      setSending(false)
    }
  }

  const sendMessage = async () => {
    if (!composer.trim()) return
    if (!selectedModelRunnable) {
      showNotice('Choose a model that is actually ready before sending a prompt.', 'error')
      return
    }

    const messageContent = composer.trim()

    setSending(true)
    showNotice(isExternalModel(selectedModel) ? 'External API proxying is not implemented in Fathom yet…' : 'Running local inference…', 'info')

    try {
      let conversationId = selectedConversation?.id

      if (!conversationId) {
        const conversation = await createConversationRecord({ silent: true })
        conversationId = conversation.id
      }

      setSelectedConversationId(conversationId)
      setPendingChat({
        conversationId,
        content: messageContent,
        modelId: selectedModelId,
        resumeAfter: Date.now() + 5000,
      })

      if (!isExternalModel(selectedModel) && (!runtime?.loaded_now || runtime?.active_model_id !== selectedModelId)) {
        const activateRes = await fetch(`/api/models/${selectedModelId}/activate`, { method: 'POST' })
        if (!activateRes.ok) {
          throw new Error(await readApiErrorNotice(activateRes, 'Could not load the selected model.'))
        }
      }

      const res = await fetch(`/api/conversations/${conversationId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: messageContent, model_id: selectedModelId }),
      })

      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Local inference failed.'))
      }

      const updatedConversation = await res.json()
      setComposer('')
      setPendingChat(null)
      await loadDashboard({ silent: true })
      setSelectedConversationId(updatedConversation.id)
      showNotice(buildChatSuccessNotice({
        modelType: isExternalModel(selectedModel) ? 'external' : 'local',
        memoryContext: updatedConversation.memory_context,
      }), 'success')
    } catch (error) {
      setPendingChat(null)
      showNotice(error.message || (isExternalModel(selectedModel) ? 'The connected model request failed.' : 'Local inference failed.'), 'error')
      await loadDashboard({ silent: true })
    } finally {
      setSending(false)
    }
  }

  const renameConversation = async (id, nextTitle) => {
    const trimmedTitle = nextTitle.trim()
    if (!trimmedTitle) {
      showNotice('Conversation title cannot be empty.', 'error')
      return false
    }

    try {
      const res = await fetch(`/api/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: trimmedTitle }),
      })
      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Could not rename that conversation.'))
      }
      await loadDashboard({ silent: true })
      showNotice('Conversation title updated.', 'success')
      return true
    } catch (error) {
      showNotice(error.message || 'Could not rename that conversation.', 'error')
      return false
    }
  }

  const deleteConversation = async (id) => {
    try {
      const res = await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Could not delete that conversation.'))
      }
      if (selectedConversationId === id) {
        setSelectedConversationId(null)
      }
      await loadDashboard({ silent: true })
      showNotice('Conversation deleted.', 'success')
      return true
    } catch (error) {
      showNotice(error.message || 'Could not delete that conversation.', 'error')
      return false
    }
  }

  const showNewChatLanding = () => {
    setTab('chat')
    setSelectedConversationId(NEW_CHAT_SENTINEL)
    setComposer('')
    setPendingChat(null)
  }

  useEffect(() => {
    if (!pendingChat || sending || resumeAttempted) return

    const matchingConversation = conversations.find((conversation) => conversation.id === pendingChat.conversationId)
    if (!matchingConversation) return

    const hasPersistedUserMessage = matchingConversation.messages.some((message) =>
      message.role === 'user' && typeof message.content === 'string' && message.content.trim() === pendingChat.content,
    )
    const hasRealAssistantReply = matchingConversation.messages.some((message) =>
      message.role === 'assistant' && typeof message.content === 'string' && !message.content.startsWith('Conversation created.'),
    )

    if (hasPersistedUserMessage || hasRealAssistantReply || resumeTimerRef.current) return

    const waitMs = Math.max(0, (pendingChat.resumeAfter || (Date.now() + 5000)) - Date.now())
    resumeTimerRef.current = window.setTimeout(() => {
      resumeTimerRef.current = null
      setResumeAttempted(true)
      void resumePendingMessage(pendingChat)
    }, waitMs)
  }, [conversations, pendingChat, resumeAttempted, sending])

  const createMemory = async ({ title, body, scope = 'General' }) => {
    try {
      const res = await fetch('/api/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, body, scope }),
      })
      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Could not save that memory.'))
      }
      await loadDashboard({ silent: true })
      setTab('memory')
      showNotice('Memory saved locally.', 'success')
      return true
    } catch (error) {
      showNotice(error.message || 'Could not save that memory.', 'error')
      return false
    }
  }

  const updateMemory = async (id, changes, { successMessage = 'Memory updated.' } = {}) => {
    try {
      const res = await fetch(`/api/memories/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(changes),
      })
      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Could not update that memory.'))
      }
      await loadDashboard({ silent: true })
      if (successMessage) showNotice(successMessage, 'success')
      return true
    } catch (error) {
      showNotice(error.message || 'Could not update that memory.', 'error')
      return false
    }
  }

  const deleteMemory = async (id, { successMessage = 'Memory deleted.' } = {}) => {
    try {
      const res = await fetch(`/api/memories/${id}`, { method: 'DELETE' })
      if (!res.ok) {
        throw new Error(await readApiErrorNotice(res, 'Could not delete that memory.'))
      }
      await loadDashboard({ silent: true })
      if (successMessage) showNotice(successMessage, 'success')
      return true
    } catch (error) {
      showNotice(error.message || 'Could not delete that memory.', 'error')
      return false
    }
  }

  const saveToMemory = async () => {
    const latestAssistant = [...(selectedConversation?.messages || [])].reverse().find((message) => message.role === 'assistant')
    if (!latestAssistant) {
      showNotice('There is no assistant reply to save yet.', 'error')
      return
    }

    const conversationTitle = selectedConversation?.title?.trim() || 'Current chat'

    await createMemory({
      title: `Saved from ${conversationTitle}`,
      body: latestAssistant.content,
      scope: 'Conversation',
    })
  }

  const installModel = async (id) => {
    const res = await fetch('/api/models/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    })

    if (!res.ok) {
      showNotice(await readApiErrorNotice(res, 'Could not start the model download.'), 'error')
      return
    }

    await loadDashboard({ silent: true })
    showNotice('Model download started. Progress will update here as bytes land on disk.', 'success')
  }

  const installCatalogModel = async ({ repo_id, filename, accept_license, accepted_license }) => {
    const body = { repo_id, filename }
    if (accept_license || accepted_license) body.accept_license = true

    const res = await fetch('/api/models/catalog/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      showNotice(await readApiErrorNotice(res, 'Could not install that Hugging Face model.'), 'error')
      return false
    }

    const model = await res.json().catch(() => null)
    await loadDashboard({ silent: true })
    const capability = model?.capability_summary ? ` Capability: ${model.capability_summary}` : ''
    showNotice(`Hugging Face model installed and inspected. Fathom will enable generation only if it matches a verified custom Rust backend lane.${capability}`, 'success')
    return true
  }

  const cancelModelDownload = async (id) => {
    const res = await fetch(`/api/models/${id}/cancel`, { method: 'POST' })

    if (!res.ok) {
      showNotice(await readApiErrorNotice(res, 'Could not cancel that download.'), 'error')
      return false
    }

    await loadDashboard({ silent: true })
    showNotice('Download cancellation acknowledged.', 'success')
    return true
  }

  const activateModel = async (id) => {
    const res = await fetch(`/api/models/${id}/activate`, { method: 'POST' })
    if (!res.ok) {
      showNotice(await readApiErrorNotice(res, 'Could not activate that model.'), 'error')
      return
    }

    await loadDashboard({ silent: true })
    setSelectedModelId(id)
    showNotice('Model loaded into the local runtime.', 'success')
  }

  const connectExternalModel = async () => {
    const name = externalForm.name.trim()
    const modelName = externalForm.model_name.trim()
    const apiBase = externalForm.api_base.trim()
    const apiKey = externalForm.api_key.trim()
    const source = externalForm.source.trim() || 'External API'
    const derivedId = externalForm.id.trim() || slugifyModelId(`${source}-${modelName || name}`)

    if (!name || !modelName || !apiBase || !apiKey) {
      showNotice('Add a display name, API base, API key, and remote model name before connecting an external model.', 'error')
      return
    }

    const res = await fetch('/api/models/external', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: derivedId,
        name,
        source,
        api_base: apiBase,
        api_key: apiKey,
        model_name: modelName,
      }),
    })

    if (!res.ok) {
      showNotice(await readApiErrorNotice(res, 'Could not connect that external model.'), 'error')
      return
    }

    await loadDashboard({ silent: true })
    setExternalForm({
      id: '',
      name: '',
      source: source,
      api_base: apiBase,
      api_key: '',
      model_name: '',
    })
    showNotice('External API details saved locally as a connected placeholder. Chat proxying is not implemented yet.', 'success')
  }

  const registerModel = async () => {
    const name = registerForm.name.trim()
    const modelPath = registerForm.model_path.trim()
    const derivedId = registerForm.id.trim() || slugifyModelId(name || modelPath.split('/').pop()?.replace(/\.gguf$/i, '') || '')

    if (!name || !modelPath) {
      showNotice('Add a model name and local model artifact path before importing.', 'error')
      return
    }

    if (!derivedId) {
      showNotice('Fathom could not create a model ID from that name. Add one in advanced options.', 'error')
      return
    }

    const res = await fetch('/api/models/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: derivedId,
        name,
        model_path: modelPath,
        runtime_model_name: registerForm.runtime_model_name.trim() || derivedId,
      }),
    })

    if (!res.ok) {
      showNotice(await readApiErrorNotice(res, 'Could not register that local model.'), 'error')
      return
    }

    await loadDashboard({ silent: true })
    setSelectedModelId(derivedId)
    setRegisterForm({ id: '', name: '', model_path: '', runtime_model_name: '' })
    showNotice('Local model imported. Fathom will only offer chat when a real runnable backend lane exists.', 'success')
  }

  return {
    dashboard,
    tab,
    setTab,
    selectedConversationId,
    setSelectedConversationId,
    selectedModelId,
    setSelectedModelId,
    search,
    setSearch,
    memorySearch,
    setMemorySearch,
    composer,
    setComposer,
    newChatTitle,
    setNewChatTitle,
    sending,
    registerForm,
    setRegisterForm,
    externalForm,
    setExternalForm,
    conversations,
    memories,
    models,
    runtime,
    selectedConversation,
    selectedModel,
    selectedModelRunnable,
    filteredConversations,
    filteredMemories,
    latestAssistantMessage,
    pendingConversation,
    createConversation,
    showNewChatLanding,
    sendMessage,
    saveToMemory,
    createMemory,
    updateMemory,
    deleteMemory,
    renameConversation,
    deleteConversation,
    installModel,
    installCatalogModel,
    cancelModelDownload,
    activateModel,
    registerModel,
    connectExternalModel,
    loadDashboard,
  }
}
