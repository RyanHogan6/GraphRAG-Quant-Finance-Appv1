'use client'

import { useState, useEffect } from 'react'
import { api } from '@/lib/api'

type Alert = { id: string; name: string; workspace_id: string; created_at: number }
type Notification = { id: string; alert_id: string; title: string; body: string; created_at: number; read: boolean }
type WorkspaceItem = { id: string; name: string; question: string }

export default function AlertsModal({
  onClose,
  onEvaluate,
}: {
  onClose: () => void
  onEvaluate?: () => void
}) {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [workspaces, setWorkspaces] = useState<WorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [newName, setNewName] = useState('')
  const [newWorkspaceId, setNewWorkspaceId] = useState('')
  const [running, setRunning] = useState(false)

  const load = async () => {
    setError(null)
    try {
      const [a, n, w] = await Promise.all([
        api.getAlerts(),
        api.getNotifications(),
        api.getWorkspaceHeaders(),
      ])
      setAlerts(Array.isArray(a) ? a : [])
      setNotifications(Array.isArray(n) ? n : [])
      setWorkspaces(Array.isArray(w) ? w : [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const handleAdd = async () => {
    if (!newName.trim() || !newWorkspaceId) return
    setAdding(true)
    setError(null)
    try {
      await api.createAlert({ name: newName.trim(), workspace_id: newWorkspaceId })
      setNewName('')
      setNewWorkspaceId('')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create alert')
    } finally {
      setAdding(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await api.deleteAlert(id)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete')
    }
  }

  const handleRunNow = async () => {
    setRunning(true)
    setError(null)
    try {
      await api.evaluateAlerts()
      await load()
      onEvaluate?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Run failed')
    } finally {
      setRunning(false)
    }
  }

  const handleMarkRead = async (id: string) => {
    try {
      await api.markNotificationRead(id)
      setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)))
    } catch {
      // ignore
    }
  }

  const workspaceName = (id: string) => workspaces.find((w) => w.id === id)?.name || id

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 w-full max-w-lg max-h-[85vh] flex flex-col shadow-xl" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-sm font-bold text-gold mb-3">Alerts</h3>
        {error && <p className="text-red-400 text-xs mb-2">{error}</p>}
        {loading ? (
          <p className="text-gray-400 text-sm">Loading…</p>
        ) : (
          <>
            <div className="mb-4">
              <div className="text-xs font-semibold text-gray-400 mb-2">Alert rules (notify when workspace has results)</div>
              <div className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Alert name"
                  className="flex-1 bg-dark-900 border border-gold/20 rounded px-2 py-1.5 text-sm text-white placeholder-gray-500"
                />
                <select
                  value={newWorkspaceId}
                  onChange={(e) => setNewWorkspaceId(e.target.value)}
                  className="bg-dark-900 border border-gold/20 rounded px-2 py-1.5 text-sm text-white"
                >
                  <option value="">Select workspace</option>
                  {workspaces.map((w) => (
                    <option key={w.id} value={w.id}>{w.name}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={handleAdd}
                  disabled={adding || !newName.trim() || !newWorkspaceId}
                  className="px-2 py-1.5 text-xs bg-gold/20 text-gold rounded border border-gold/40 disabled:opacity-50"
                >
                  Add
                </button>
              </div>
              <ul className="space-y-1 max-h-32 overflow-y-auto">
                {alerts.map((a) => (
                  <li key={a.id} className="flex items-center justify-between gap-2 bg-dark-900 rounded border border-gold/20 p-2 text-xs">
                    <span className="text-white truncate">{a.name}</span>
                    <span className="text-gray-500 truncate flex-1 text-right">{workspaceName(a.workspace_id)}</span>
                    <button type="button" onClick={() => handleDelete(a.id)} className="text-red-400 hover:text-red-300 shrink-0">Delete</button>
                  </li>
                ))}
                {alerts.length === 0 && <li className="text-gray-500 text-xs">No alert rules. Add one above.</li>}
              </ul>
              <button
                type="button"
                onClick={handleRunNow}
                disabled={running || alerts.length === 0}
                className="mt-2 px-3 py-1.5 text-xs bg-gold/20 text-gold rounded border border-gold/40 disabled:opacity-50"
              >
                {running ? 'Running…' : 'Run now'}
              </button>
            </div>
            <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
              <div className="text-xs font-semibold text-gray-400 mb-2">Notifications</div>
              <ul className="overflow-y-auto space-y-1 flex-1 min-h-0">
                {notifications.length === 0 ? (
                  <li className="text-gray-500 text-xs">No notifications yet. Run alerts to check for new results.</li>
                ) : (
                  notifications.map((n) => (
                    <li
                      key={n.id}
                      className={`rounded border p-2 text-xs ${n.read ? 'bg-dark-900/50 border-gray-700 text-gray-500' : 'bg-dark-900 border-gold/20 text-gray-200'}`}
                    >
                      <div className="font-medium">{n.title}</div>
                      <div className="text-gray-500 mt-0.5">{n.body}</div>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-[10px] text-gray-600">{new Date(n.created_at * 1000).toLocaleString()}</span>
                        {!n.read && (
                          <button type="button" onClick={() => handleMarkRead(n.id)} className="text-gold/80 hover:text-gold text-[10px]">
                            Mark read
                          </button>
                        )}
                      </div>
                    </li>
                  ))
                )}
              </ul>
            </div>
          </>
        )}
        <div className="mt-3 pt-3 border-t border-gold/20 flex justify-end">
          <button type="button" onClick={onClose} className="px-3 py-1.5 text-sm text-gray-400 hover:text-white rounded border border-gray-600">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
