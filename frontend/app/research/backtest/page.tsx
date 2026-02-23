'use client'

import { useState } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'

export default function ResearchBacktestPage() {
  const [platform, setPlatform] = useState<'kalshi' | 'polymarket'>('polymarket')
  const [marketId, setMarketId] = useState('')
  const [resolutionDate, setResolutionDate] = useState('')
  const [lookbackDays, setLookbackDays] = useState(30)
  const [signals, setSignals] = useState({ macro: true, options: true, sec: false, contracts: false })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const runBacktest = async () => {
    if (!resolutionDate) {
      setError('Set resolution date (YYYY-MM-DD)')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await api.runResearchBacktest({
        platform,
        resolution_date: resolutionDate,
        lookback_days: lookbackDays,
        market_id: marketId || undefined,
        signals,
      })
      setResult(data)
    } catch (e) {
      setError((e as Error).message || 'Backtest failed')
    } finally {
      setLoading(false)
    }
  }

  const toggleSignal = (k: keyof typeof signals) => {
    setSignals((s) => ({ ...s, [k]: !s[k] }))
  }

  return (
    <div className="min-h-screen bg-dark-900 text-gray-100">
      <div className="container mx-auto px-4 md:px-6 py-8">
        <div className="mb-6">
          <Link href="/research" className="text-gold/80 hover:text-gold text-sm font-medium">← Back to Research</Link>
        </div>
        <h1 className="text-2xl font-bold text-gold font-mono mb-2">Prediction Market Backtester</h1>
        <p className="text-gray-400 text-sm mb-8">
          Overlay macro, options, SEC, and contract signals on market probability over a lookback window; see lead/lag and correlations.
        </p>

        <div className="rounded-lg border border-gold/20 bg-dark-800 p-6 mb-8 max-w-2xl">
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Platform</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setPlatform('polymarket')}
                  className={`px-3 py-1.5 rounded text-sm ${platform === 'polymarket' ? 'bg-gold/20 text-gold border border-gold/40' : 'text-gray-400 border border-gray-600'}`}
                >
                  Polymarket
                </button>
                <button
                  type="button"
                  onClick={() => setPlatform('kalshi')}
                  className={`px-3 py-1.5 rounded text-sm ${platform === 'kalshi' ? 'bg-gold/20 text-gold border border-gold/40' : 'text-gray-400 border border-gray-600'}`}
                >
                  Kalshi
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Market ID (optional; for Polymarket fetches probability from DB)</label>
              <input
                type="text"
                value={marketId}
                onChange={(e) => setMarketId(e.target.value)}
                placeholder="e.g. 516938"
                className="w-full px-3 py-2 rounded bg-dark-700 border border-gold/20 text-gray-100 placeholder-gray-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Resolution date (end of window) *</label>
              <input
                type="date"
                value={resolutionDate}
                onChange={(e) => setResolutionDate(e.target.value)}
                className="w-full px-3 py-2 rounded bg-dark-700 border border-gold/20 text-gray-100"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Lookback days</label>
              <input
                type="number"
                min={7}
                max={90}
                value={lookbackDays}
                onChange={(e) => setLookbackDays(Number(e.target.value) || 30)}
                className="w-24 px-3 py-2 rounded bg-dark-700 border border-gold/20 text-gray-100"
              />
            </div>
            <div>
              <span className="block text-sm text-gray-400 mb-2">Signals to overlay</span>
              <div className="flex flex-wrap gap-3">
                {(['macro', 'options', 'sec', 'contracts'] as const).map((k) => (
                  <label key={k} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={signals[k]}
                      onChange={() => toggleSignal(k)}
                      className="rounded border-gold/40 text-gold bg-dark-700"
                    />
                    <span className="text-sm capitalize">{k}</span>
                  </label>
                ))}
              </div>
            </div>
            <button
              type="button"
              onClick={runBacktest}
              disabled={loading}
              className="px-4 py-2 rounded bg-gold/20 text-gold border border-gold/40 hover:bg-gold/30 disabled:opacity-50"
            >
              {loading ? 'Running…' : 'Run backtest'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg border border-red-500/30 bg-red-500/10 text-red-200">{error}</div>
        )}

        {result && (
          <div className="space-y-8">
            <div className="rounded-lg border border-gold/20 bg-dark-800 overflow-hidden">
              <h3 className="px-4 py-3 border-b border-gold/20 text-gold font-mono">Lead / Lag</h3>
              <div className="p-4 overflow-x-auto">
                {result.lead_lag && Object.keys(result.lead_lag).length > 0 ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-400 border-b border-gold/20">
                        <th className="text-left py-2">Signal</th>
                        <th className="text-left py-2">Best lag (days)</th>
                        <th className="text-left py-2">Summary</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.lead_lag).map(([sig, v]: [string, any]) => (
                        <tr key={sig} className="border-b border-gray-700">
                          <td className="py-2 capitalize">{sig}</td>
                          <td className="py-2 text-gold">{v.best_lead_lag != null ? v.best_lead_lag : '—'}</td>
                          <td className="py-2 text-gray-300">{v.summary || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-gray-500">No probability series or signal data for lead/lag (e.g. provide market_id for Polymarket or probability_series for Kalshi).</p>
                )}
              </div>
            </div>

            {result.probability_series?.length > 0 && (
              <div className="rounded-lg border border-gold/20 bg-dark-800 overflow-hidden">
                <h3 className="px-4 py-3 border-b border-gold/20 text-gold font-mono">Probability series ({result.probability_series.length} points)</h3>
                <div className="p-4 overflow-x-auto max-h-64 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-400 border-b border-gold/20">
                        <th className="text-left py-1">Date</th>
                        <th className="text-right py-1">Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.probability_series.slice(0, 30).map((row: any, i: number) => (
                        <tr key={i} className="border-b border-gray-700">
                          <td className="py-1">{row.date}</td>
                          <td className="py-1 text-right text-gold">{(row.probability * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {result.probability_series.length > 30 && (
                    <p className="text-gray-500 text-xs mt-2">Showing first 30 of {result.probability_series.length}</p>
                  )}
                </div>
              </div>
            )}

            {result.signal_series && (result.signal_series.macro?.length > 0 || result.signal_series.options?.length > 0) && (
              <div className="rounded-lg border border-gold/20 bg-dark-800 overflow-hidden">
                <h3 className="px-4 py-3 border-b border-gold/20 text-gold font-mono">Signal series</h3>
                <div className="p-4 text-sm text-gray-400">
                  Macro: {result.signal_series.macro?.length ?? 0} rows · Options: {result.signal_series.options?.length ?? 0} rows · Contracts: {result.signal_series.contracts?.length ?? 0} rows
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
