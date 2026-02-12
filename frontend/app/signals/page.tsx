'use client'

import { useState, useEffect } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface ContractMomentumRow {
  ticker: string
  company?: string
  sector?: string
  contract_momentum_90d: number
  award_count_90d: number
  momentum_rank?: number
}

interface OptionsFilingRow {
  ticker: string
  options_filing_events: number
  unusual_activity_count: number
  max_unusual_ratio: number
}

interface CentralityRow {
  ticker: string
  company?: string
  sector?: string
  contract_degree_centrality: number
}

interface AllSignalsResponse {
  contract_momentum_90d: ContractMomentumRow[]
  options_filing_convergence: OptionsFilingRow[]
  contract_centrality: CentralityRow[]
}

interface BacktestResponse {
  signal?: string
  start_date?: string
  end_date?: string
  observations?: number
  rebalance_dates?: number
  rank_ic?: number
  hit_rate?: number
  top_quintile_avg_return?: number
  bottom_quintile_avg_return?: number
  spread?: number
  sharpe_like?: number
  point_in_time?: string
  error?: string
}

function SignalsTable<T extends Record<string, unknown>>({
  title,
  rows,
  columns,
  loading,
}: {
  title: string
  rows: T[]
  columns: { key: keyof T; label: string; format?: (v: unknown) => string }[]
  loading: boolean
}) {
  if (loading) {
    return (
      <div className="bg-dark-800 border border-gold/20 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-gold mb-4">{title}</h2>
        <div className="text-gray-400">Loading…</div>
      </div>
    )
  }
  if (!rows?.length) {
    return (
      <div className="bg-dark-800 border border-gold/20 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-gold mb-4">{title}</h2>
        <div className="text-gray-500">No data</div>
      </div>
    )
  }
  return (
    <div className="bg-dark-800 border border-gold/20 rounded-lg p-6 overflow-x-auto">
      <h2 className="text-xl font-semibold text-gold mb-4">{title}</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gold/30">
            {columns.map((col) => (
              <th key={String(col.key)} className="text-left py-2 px-3 text-gold font-medium">
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-white/5 hover:bg-dark-700/50">
              {columns.map((col) => {
                const v = row[col.key]
                const display = col.format ? col.format(v) : (v != null ? String(v) : '—')
                return (
                  <td key={String(col.key)} className="py-2 px-3 text-gray-300">
                    {display}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function SignalsPage() {
  const [allData, setAllData] = useState<AllSignalsResponse | null>(null)
  const [backtestData, setBacktestData] = useState<BacktestResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [backtestLoading, setBacktestLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [backtestError, setBacktestError] = useState<string | null>(null)

  const contractLimit = 20
  const optionsLimit = 15
  const centralityLimit = 20
  const backtestStart = '2024-01-01'
  const backtestEnd = '2024-12-31'

  useEffect(() => {
    const fetchAll = async () => {
      try {
        setLoading(true)
        setError(null)
        const res = await fetch(
          `${API_BASE}/api/signals/all?contract_limit=${contractLimit}&options_limit=${optionsLimit}&centrality_limit=${centralityLimit}`
        )
        if (!res.ok) throw new Error(`Signals failed: ${res.status}`)
        const data = await res.json()
        setAllData(data)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load signals')
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [])

  useEffect(() => {
    const fetchBacktest = async () => {
      try {
        setBacktestLoading(true)
        setBacktestError(null)
        const res = await fetch(
          `${API_BASE}/api/signals/backtest?start_date=${backtestStart}&end_date=${backtestEnd}`
        )
        if (!res.ok) throw new Error(`Backtest failed: ${res.status}`)
        const data = await res.json()
        setBacktestData(data)
      } catch (e) {
        setBacktestError(e instanceof Error ? e.message : 'Failed to load backtest')
      } finally {
        setBacktestLoading(false)
      }
    }
    fetchBacktest()
  }, [])

  return (
    <div className="container mx-auto px-4 md:px-6 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gold mb-2">Signal Dashboard</h1>
        <p className="text-gray-500">
          Alpha signals from the graph: contract momentum, options–filing convergence, contract centrality, and backtest.
        </p>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-300">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:gap-8">
        <SignalsTable<ContractMomentumRow>
          title="Contract momentum (90d)"
          rows={allData?.contract_momentum_90d ?? []}
          loading={loading}
          columns={[
            { key: 'ticker', label: 'Ticker' },
            { key: 'company', label: 'Company' },
            { key: 'sector', label: 'Sector' },
            {
              key: 'contract_momentum_90d',
              label: '90d total',
              format: (v) => (typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'),
            },
            { key: 'award_count_90d', label: 'Awards' },
            { key: 'momentum_rank', label: 'Rank' },
          ]}
        />

        <SignalsTable<OptionsFilingRow>
          title="Options–filing convergence"
          rows={allData?.options_filing_convergence ?? []}
          loading={loading}
          columns={[
            { key: 'ticker', label: 'Ticker' },
            { key: 'options_filing_events', label: 'Events' },
            { key: 'unusual_activity_count', label: 'Unusual count' },
            {
              key: 'max_unusual_ratio',
              label: 'Max unusual ratio',
              format: (v) => (typeof v === 'number' ? v.toFixed(2) : '—'),
            },
          ]}
        />

        <SignalsTable<CentralityRow>
          title="Contract centrality"
          rows={allData?.contract_centrality ?? []}
          loading={loading}
          columns={[
            { key: 'ticker', label: 'Ticker' },
            { key: 'company', label: 'Company' },
            { key: 'sector', label: 'Sector' },
            {
              key: 'contract_degree_centrality',
              label: 'Degree',
              format: (v) => (typeof v === 'number' ? v.toLocaleString() : '—'),
            },
          ]}
        />

        <div className="bg-dark-800 border border-gold/20 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gold mb-4">Backtest (contract momentum vs forward returns)</h2>
          {backtestLoading && <div className="text-gray-400">Loading…</div>}
          {backtestError && (
            <div className="text-red-300 mb-2">{backtestError}</div>
          )}
          {backtestData?.error && (
            <div className="text-amber-300">{backtestData.error}</div>
          )}
          {!backtestLoading && backtestData && !backtestData.error && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Rank IC</div>
                <div className="text-gold font-mono">
                  {backtestData.rank_ic != null ? backtestData.rank_ic.toFixed(4) : '—'}
                </div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Hit rate</div>
                <div className="text-gold font-mono">
                  {backtestData.hit_rate != null ? (backtestData.hit_rate * 100).toFixed(2) + '%' : '—'}
                </div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Observations</div>
                <div className="text-gold font-mono">{backtestData.observations ?? '—'}</div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Rebalance dates</div>
                <div className="text-gold font-mono">{backtestData.rebalance_dates ?? '—'}</div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Top quintile avg return</div>
                <div className="text-gold font-mono">
                  {backtestData.top_quintile_avg_return != null
                    ? (backtestData.top_quintile_avg_return * 100).toFixed(2) + '%'
                    : '—'}
                </div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Bottom quintile avg return</div>
                <div className="text-gold font-mono">
                  {backtestData.bottom_quintile_avg_return != null
                    ? (backtestData.bottom_quintile_avg_return * 100).toFixed(2) + '%'
                    : '—'}
                </div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Spread</div>
                <div className="text-gold font-mono">
                  {backtestData.spread != null ? (backtestData.spread * 100).toFixed(2) + '%' : '—'}
                </div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Sharpe-like</div>
                <div className="text-gold font-mono">
                  {backtestData.sharpe_like != null ? backtestData.sharpe_like.toFixed(4) : '—'}
                </div>
              </div>
            </div>
          )}
          {!backtestLoading && backtestData && !backtestData.error && backtestData.point_in_time && (
            <p className="text-gray-500 text-xs mt-4">{backtestData.point_in_time}</p>
          )}
        </div>
      </div>
    </div>
  )
}
