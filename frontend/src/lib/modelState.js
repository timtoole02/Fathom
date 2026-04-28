export function isExternalModel(model) {
  return model?.provider_kind === 'external'
}

export function normalizeCapabilityStatus(status) {
  return (status || '').toString().trim().toLowerCase()
}

export function isEmbeddingOnlyModel(model) {
  const task = normalizeCapabilityStatus(model?.task)
  if (task === 'text_embedding' || task === 'textembedding') return true
  if (task && task !== 'text_generation' && task !== 'textgeneration') return false
  const lanes = model?.backend_lanes || []
  const summary = (model?.capability_summary || '').toString().toLowerCase()
  return lanes.includes('local-embeddings-retrieval')
    && (summary.includes('embedding-only') || summary.includes('text-embedding') || summary.includes('not chat/generation'))
}

export function hasRunnableCapability(model) {
  if (isExternalModel(model)) return false
  if (isEmbeddingOnlyModel(model)) return false
  return normalizeCapabilityStatus(model?.capability_status) === 'runnable'
}

export function hasPlannedCapability(model) {
  return normalizeCapabilityStatus(model?.capability_status) === 'planned'
}

export function isRunnableModel(model) {
  if (!model || model.status !== 'ready') return false
  if (isExternalModel(model)) return hasRunnableCapability(model)
  return Boolean(model.model_path && hasRunnableCapability(model))
}

export function canLoadIntoRuntime(model) {
  return Boolean(model && !isExternalModel(model) && (model.status === 'ready' || model.status === 'registered') && model.model_path && hasRunnableCapability(model))
}

export function getModelStatusLabel(model) {
  if (!model) return 'No model selected'
  if (model.status === 'downloading') return 'Downloading'
  if (model.status === 'canceling') return 'Canceling download'
  if (model.status === 'failed') return 'Needs attention'
  if (isExternalModel(model) && model.status === 'ready') return 'Connected API placeholder'
  if (isEmbeddingOnlyModel(model) && model.model_path) return 'Embedding-ready, not chat'
  if (hasPlannedCapability(model) && model.model_path) return 'Downloaded, backend planned'
  if (model.status === 'registered') return 'Imported, first load pending'
  if (model.status === 'ready' && model.model_path) return 'Ready locally'
  if (model.status === 'ready') return 'Catalog entry needs file path'
  return 'Not downloaded yet'
}

export function describeModelState(model) {
  if (!model) return 'Choose a model to decide what you want to use for the next chat.'
  if (model.status === 'downloading') return 'This model is still downloading to local storage, so it cannot be used yet.'
  if (model.status === 'canceling') return 'Fathom is stopping the download and cleaning up the partial file.'
  if (model.status === 'failed') return isExternalModel(model) ? 'This API connection needs attention before Fathom can keep its metadata.' : 'The last download did not finish. Retry to continue or start again if needed.'
  if (isExternalModel(model) && model.status === 'ready') return 'Connected external OpenAI-compatible metadata only. Fathom stores the endpoint and key locally, but chat proxying is not implemented yet, so it stays out of next-chat selection.'
  if (isEmbeddingOnlyModel(model) && model.model_path) return 'Ready for local embedding and retrieval experiments. It is not a chat model and will stay out of next-chat selection.'
  if (hasPlannedCapability(model) && model.model_path) return 'Downloaded into Fathom-managed storage, but this backend lane is still planned. It is not chat-runnable until the real generation backend is wired.'
  if (model.status === 'registered') return 'The file is listed locally, but Fathom still needs to load it once before it is ready to chat.'
  if (model.status === 'ready' && model.model_path) return 'Ready to use. You can keep it selected for your next chat or load it into the runtime now.'
  if (model.status === 'ready') return 'This entry exists in the catalog, but it still needs a local model file before it can run here.'
  return 'This model is listed in the catalog, but it is not downloaded locally yet.'
}
