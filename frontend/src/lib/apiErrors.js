function getNonEmptyString(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

export async function readApiErrorMessage(response, fallback = 'Request failed.') {
  let body = ''

  try {
    body = await response.text()
  } catch {
    return fallback
  }

  const trimmedBody = getNonEmptyString(body)
  if (!trimmedBody) return fallback

  try {
    const parsed = JSON.parse(trimmedBody)
    const envelopeMessage = getNonEmptyString(parsed?.error?.message)
    if (envelopeMessage) return envelopeMessage

    const topLevelMessage = getNonEmptyString(parsed?.message)
    if (topLevelMessage) return topLevelMessage

    const stringBody = getNonEmptyString(parsed)
    if (stringBody) return stringBody
  } catch {
    // Plain-text API failures are valid fallback details.
  }

  return trimmedBody
}
