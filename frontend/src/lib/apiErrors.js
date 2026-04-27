function getNonEmptyString(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function scrubLocalDetails(value) {
  const message = getNonEmptyString(value)
  if (!message) return null

  return message
    .replace(/\b\/?(?:Users|private\/tmp|tmp|var\/folders)\/[^\s"'`),;]+/gi, '[local path]')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeStatus(response) {
  const status = Number(response?.status)
  return Number.isFinite(status) && status > 0 ? status : null
}

function labelFromDetails({ type, code, status } = {}) {
  const normalizedCode = getNonEmptyString(code)?.replace(/[_-]+/g, ' ')
  if (normalizedCode) return normalizedCode

  const normalizedType = getNonEmptyString(type)?.replace(/[_-]+/g, ' ')
  if (normalizedType) return normalizedType

  if (status === 400) return 'Request refused'
  if (status === 404) return 'Not found'
  if (status === 409) return 'State conflict'
  if (status === 422) return 'Request not supported'
  if (status && status >= 500) return 'Backend error'
  return 'Request failed'
}

function parseApiErrorBody(body, fallback, response) {
  const status = normalizeStatus(response)
  const fallbackMessage = getNonEmptyString(fallback) || 'Request failed.'
  const trimmedBody = getNonEmptyString(body)

  if (!trimmedBody) {
    return {
      message: fallbackMessage,
      safeMessage: scrubLocalDetails(fallbackMessage),
      type: null,
      code: null,
      param: null,
      status,
      label: labelFromDetails({ status }),
    }
  }

  try {
    const parsed = JSON.parse(trimmedBody)
    const envelope = parsed?.error && typeof parsed.error === 'object' ? parsed.error : null
    const message = getNonEmptyString(envelope?.message)
      || getNonEmptyString(parsed?.message)
      || getNonEmptyString(parsed)
      || fallbackMessage
    const type = getNonEmptyString(envelope?.type)
    const code = getNonEmptyString(envelope?.code)
    const param = getNonEmptyString(envelope?.param)

    return {
      message,
      safeMessage: scrubLocalDetails(message) || fallbackMessage,
      type,
      code,
      param,
      status,
      label: labelFromDetails({ type, code, status }),
    }
  } catch {
    const message = trimmedBody || fallbackMessage
    return {
      message,
      safeMessage: scrubLocalDetails(message) || fallbackMessage,
      type: null,
      code: null,
      param: null,
      status,
      label: labelFromDetails({ status }),
    }
  }
}

export async function readApiErrorDetails(response, fallback = 'Request failed.') {
  try {
    return parseApiErrorBody(await response.text(), fallback, response)
  } catch {
    const status = normalizeStatus(response)
    const message = getNonEmptyString(fallback) || 'Request failed.'
    return {
      message,
      safeMessage: scrubLocalDetails(message) || message,
      type: null,
      code: null,
      param: null,
      status,
      label: labelFromDetails({ status }),
    }
  }
}

export async function readApiErrorMessage(response, fallback = 'Request failed.') {
  const details = await readApiErrorDetails(response, fallback)
  return details.message
}

export function formatApiErrorDetails(details, fallback = 'Request failed.') {
  const label = getNonEmptyString(details?.label) || 'Request failed'
  const message = scrubLocalDetails(details?.message) || scrubLocalDetails(details?.safeMessage) || scrubLocalDetails(fallback) || 'Request failed.'
  return `${label}: ${message}`
}
