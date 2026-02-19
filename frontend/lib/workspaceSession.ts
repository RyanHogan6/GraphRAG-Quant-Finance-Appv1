const STORAGE_KEY = 'karga_session_id'

export function getSessionId(): string {
  if (typeof window === 'undefined') return ''
  let id = localStorage.getItem(STORAGE_KEY)
  if (!id || id.length < 8) {
    id = crypto.randomUUID?.() ?? `sess_${Date.now()}_${Math.random().toString(36).slice(2, 12)}`
    localStorage.setItem(STORAGE_KEY, id)
  }
  return id
}
