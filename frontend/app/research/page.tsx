'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'

export default function ResearchPage() {
  const [platform, setPlatform] = useState<'kalshi' | 'polymarket'>('kalshi')
  const [markets, setMarkets] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    api.getResearchMarkets(platform, undefined, 80)
      .then((data: any) => {
        if (!cancelled) setMarkets(data?.markets ?? [])
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || 'Failed to load markets')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [platform])

  return (
    <div className="min-h-screen bg-dark-900 text-gray-100">
      <div className="container mx-auto px-4 md:px-6 py-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-gold font-mono">Market Research</h1>
            <p className="text-gray-400 mt-1">Prediction market workups: macro, options, SEC sentiment, contracts, and synthesis.</p>
            <Link href="/research/backtest" className="text-gold/80 hover:text-gold text-sm mt-2 inline-block">Backtest →</Link>
          </div>
          <div className="flex rounded-lg border border-gold/20 bg-dark-800 p-1">
            <button
              onClick={() => setPlatform('kalshi')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${platform === 'kalshi' ? 'bg-gold/20 text-gold border border-gold/40' : 'text-gray-400 hover:text-gold'}`}
            >
              Kalshi
            </button>
            <button
              onClick={() => setPlatform('polymarket')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${platform === 'polymarket' ? 'bg-gold/20 text-gold border border-gold/40' : 'text-gray-400 hover:text-gold'}`}
            >
              Polymarket
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg border border-red-500/30 bg-red-500/10 text-red-200">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-16">
            <div className="inline-block w-8 h-8 border-2 border-gold/40 border-t-gold rounded-full animate-spin" />
            <span className="ml-3 text-gray-400">Loading markets…</span>
          </div>
        ) : (
          <div className="space-y-3">
            {markets.length === 0 && !error && (
              <p className="text-gray-400">No markets found. Try the other platform.</p>
            )}
            {markets.map((m) => {
              const yesPct = m.yes_probability != null ? (typeof m.yes_probability === 'number' ? m.yes_probability * 100 : m.yes_probability) : null
              return (
                <Link
                  key={m.id}
                  href={`/research/market/${encodeURIComponent(m.id)}?platform=${platform}`}
                  className="block rounded-lg border border-gold/20 bg-dark-800 p-4 hover:border-gold/40 hover:bg-dark-700/50 transition-all"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-100 truncate">{m.question || 'No question'}</p>
                      <div className="flex flex-wrap gap-2 mt-1 text-xs text-gray-500">
                        {m.category && <span>{m.category}</span>}
                        {m.end_date && <span>End: {m.end_date}</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-4 shrink-0">
                      {yesPct != null && (
                        <span className="text-gold font-mono font-medium">{Number(yesPct).toFixed(0)}% Yes</span>
                      )}
                      {(m.volume != null || m.open_interest != null) && (
                        <span className="text-gray-400 text-sm">
                          Vol: {(m.volume ?? m.open_interest ?? 0).toLocaleString()}
                        </span>
                      )}
                      <span className="text-gold/70 text-sm">View workup →</span>
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
