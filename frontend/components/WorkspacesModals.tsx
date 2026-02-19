'use client'

import { useState, useEffect } from 'react'
import { api } from '@/lib/api'

type WorkspaceItem = { id: string; name: string; type: string; question: string; created_at: number; updated_at: number }

export function SaveWorkspaceModal({
  question,
  forcedPlanAql,
  onClose,
  onSaved,
}: {
  question: string
  forcedPlanAql?: string
  onClose: () => void
  onSaved: () => void
}) {
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const short = question.slice(0, 50)
    setName(short + (question.length > 50 ? '…' : ''))
  }, [question])

  const handleSave = async () => {
    const trimmed = name.trim()
    if (!trimmed) {
      setError('Enter a name')
      return
    }
    setSaving(true)
    setError(null)
    try {
      await api.createWorkspace({
        name: trimmed,
        type: forcedPlanAql ? 'builder' : 'nl',
        question,
        forced_plan_aql: forcedPlanAql,
      })
      onSaved()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 w-full max-w-md shadow-xl" onClick={e => e.stopPropagation()}>
        <h3 className="text-sm font-bold text-gold mb-3">Save workspace</h3>
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="Workspace name"
          className="w-full bg-dark-900 border border-gold/20 rounded px-3 py-2 text-sm text-white placeholder-gray-500 mb-3"
        />
        {error && <p className="text-red-400 text-xs mb-2">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 text-sm text-gray-400 hover:text-white rounded border border-gray-600">
            Cancel
          </button>
          <button type="button" onClick={handleSave} disabled={saving} className="px-3 py-1.5 text-sm bg-gold text-dark-900 rounded font-medium disabled:opacity-50">
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function MyWorkspacesModal({
  onClose,
  onRun,
}: {
  onClose: () => void
  onRun: (payload: { results: any[]; analysis: string; follow_up_questions?: string[]; query_plan?: any; metadata?: any }, question: string) => void
}) {
  const [list, setList] = useState<WorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)
  const [runningId, setRunningId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getWorkspaceHeaders()
      setList(Array.isArray(data) ? data : [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
      setList([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const handleRun = async (id: string) => {
    const w = list.find(x => x.id === id)
    const question = w?.question ?? ''
    setRunningId(id)
    setError(null)
    try {
      const data = await api.runWorkspace(id)
      onRun(data, question)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Run failed')
    } finally {
      setRunningId(null)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await api.deleteWorkspace(id)
      setList(prev => prev.filter(w => w.id !== id))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed')
    }
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 w-full max-w-lg max-h-[80vh] flex flex-col shadow-xl" onClick={e => e.stopPropagation()}>
        <h3 className="text-sm font-bold text-gold mb-3">My workspaces</h3>
        {error && <p className="text-red-400 text-xs mb-2">{error}</p>}
        {loading ? (
          <p className="text-gray-400 text-sm">Loading…</p>
        ) : list.length === 0 ? (
          <p className="text-gray-400 text-sm">No saved workspaces. Run a query and use &quot;Save&quot; to add one.</p>
        ) : (
          <ul className="overflow-y-auto space-y-2 flex-1 min-h-0">
            {list.map((w) => (
              <li key={w.id} className="flex items-center justify-between gap-2 bg-dark-900 rounded border border-gold/20 p-2">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-white truncate">{w.name}</div>
                  <div className="text-xs text-gray-500 truncate">{w.question}</div>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() => handleRun(w.id)}
                    disabled={runningId !== null}
                    className="px-2 py-1 text-xs bg-gold/20 text-gold rounded border border-gold/40 hover:bg-gold/30 disabled:opacity-50"
                  >
                    {runningId === w.id ? 'Running…' : 'Run'}
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDelete(w.id)}
                    className="px-2 py-1 text-xs text-gray-400 hover:text-red-400 rounded border border-gray-600 hover:border-red-500"
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
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
