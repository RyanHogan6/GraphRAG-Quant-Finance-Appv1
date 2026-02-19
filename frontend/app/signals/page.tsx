'use client'

import { useState, useEffect, useMemo, useRef } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const GOLD = '#D4AF37'
const GOLD_LIGHT = 'rgba(212, 175, 55, 0.4)'

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
  mean_return?: number
  return_volatility?: number
  total_return?: number
  max_drawdown?: number
  point_in_time?: string
  error?: string
}

function formatNum(v: number): string {
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

type ColumnConfig<T> = {
  key: keyof T
  label: string
  format?: (v: unknown) => string
  isRank?: boolean
  isNumeric?: boolean
  barOfMax?: boolean
}

function SignalsTable<T extends object>({
  title,
  rows,
  columns,
  loading,
  noDataExplanation,
  maxForBar,
}: {
  title: string
  rows: T[]
  columns: ColumnConfig<T>[]
  loading: boolean
  noDataExplanation?: string
  maxForBar?: number
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
        {noDataExplanation && (
          <p className="text-gray-500 text-sm mt-2 max-w-md">{noDataExplanation}</p>
        )}
      </div>
    )
  }
  return (
    <div className="bg-dark-800 border border-gold/20 rounded-lg p-6 overflow-x-auto">
      <h2 className="text-xl font-semibold text-gold mb-4 border-b border-gold/30 pb-2">{title}</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b-2 border-gold/40">
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={`py-2 px-3 text-gold font-semibold ${col.isNumeric ? 'text-right' : 'text-left'}`}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={`border-b border-white/5 hover:bg-gold/5 transition-colors ${i % 2 === 1 ? 'bg-dark-700/30' : ''}`}
            >
              {columns.map((col) => {
                const v = row[col.key]
                const display = col.format ? col.format(v) : (v != null ? String(v) : '—')
                const isNum = col.isNumeric || col.barOfMax
                return (
                  <td
                    key={String(col.key)}
                    className={`py-2 px-3 text-gray-300 border-l border-white/5 first:border-l-gold/30 first:border-l-2 ${isNum ? 'text-right' : ''}`}
                  >
                    {col.isRank && typeof v === 'number' ? (
                      <span className="inline-flex items-center justify-center min-w-[2rem] px-2 py-0.5 rounded-full bg-gold/20 text-gold text-xs font-mono border border-gold/40">
                        #{v}
                      </span>
                    ) : col.barOfMax && typeof v === 'number' && maxForBar != null && maxForBar > 0 ? (
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-1.5 bg-dark-700 rounded overflow-hidden">
                          <div
                            className="h-full bg-gold/60 rounded"
                            style={{ width: `${Math.min(100, (v / maxForBar) * 100)}%` }}
                          />
                        </div>
                        <span className="tabular-nums">{display}</span>
                      </div>
                    ) : (
                      display
                    )}
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

function BarChart({
  labels,
  values,
  valueLabel,
  height = 220,
}: {
  labels: string[]
  values: number[]
  valueLabel: string
  height?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  useEffect(() => {
    if (!canvasRef.current || labels.length === 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height
    const pad = { top: 20, right: 20, bottom: 36, left: 52 }
    const chartW = w - pad.left - pad.right
    const chartH = h - pad.top - pad.bottom
    const maxVal = Math.max(...values, 1)
    const barW = Math.max(8, (chartW / labels.length) * 0.7)

    ctx.clearRect(0, 0, w, h)
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.15)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (chartH / 4) * i
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
    }
    ctx.strokeStyle = GOLD_LIGHT
    ctx.beginPath()
    ctx.moveTo(pad.left, pad.top)
    ctx.lineTo(pad.left, h - pad.bottom)
    ctx.stroke()
    ctx.fillStyle = '#999'
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const v = maxVal * (1 - i / 4)
      const y = pad.top + (chartH / 4) * i
      ctx.fillText(formatNum(v), pad.left - 6, y + 3)
    }
    labels.forEach((label, i) => {
      const x = pad.left + (i + 0.5) * (chartW / labels.length) - barW / 2
      const barH = (values[i] / maxVal) * chartH
      const y = pad.top + chartH - barH
      ctx.fillStyle = hovered === i ? GOLD : GOLD_LIGHT
      ctx.fillRect(x, y, barW, barH)
    })
    ctx.fillStyle = '#999'
    ctx.font = '9px monospace'
    ctx.textAlign = 'center'
    labels.forEach((label, i) => {
      const x = pad.left + (i + 0.5) * (chartW / labels.length)
      const short = label.length > 10 ? label.slice(0, 8) + '…' : label
      ctx.fillText(short, x, h - pad.bottom + 14)
    })
  }, [labels, values, hovered, height])

  return (
    <div className="relative">
      <div className="text-[10px] text-gold/80 uppercase tracking-wider mb-1 font-semibold">{valueLabel}</div>
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-gold/20 bg-black/20"
        style={{ height }}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          const pad = 52
          const chartW = rect.width - 72
          const i = Math.floor(((e.clientX - rect.left - pad) / chartW) * labels.length)
          setHovered(i >= 0 && i < labels.length ? i : null)
        }}
        onMouseLeave={() => setHovered(null)}
      />
      {hovered !== null && hovered < labels.length && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded bg-dark-800 border border-gold/30 text-xs text-gold shadow-lg z-10">
          {labels[hovered]}: {formatNum(values[hovered])}
        </div>
      )}
    </div>
  )
}

function SectorPieChart({ sectorTotals }: { sectorTotals: { sector: string; total: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const total = useMemo(() => sectorTotals.reduce((s, x) => s + x.total, 0), [sectorTotals])
  const [hovered, setHovered] = useState<number | null>(null)

  useEffect(() => {
    if (!canvasRef.current || sectorTotals.length === 0 || total <= 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height
    const cx = w / 2
    const cy = h / 2 - 10
    const r = Math.min(w, h) / 2 - 24

    let start = -Math.PI / 2
    const colors = [GOLD, GOLD_LIGHT, 'rgba(212, 175, 55, 0.6)', 'rgba(212, 175, 55, 0.3)', 'rgba(212, 175, 55, 0.2)']
    sectorTotals.forEach((s, i) => {
      const slice = (s.total / total) * 2 * Math.PI
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.arc(cx, cy, r, start, start + slice)
      ctx.closePath()
      ctx.fillStyle = hovered === i ? GOLD : colors[i % colors.length]
      ctx.fill()
      ctx.strokeStyle = 'rgba(0,0,0,0.3)'
      ctx.lineWidth = 1
      ctx.stroke()
      start += slice
    })
    ctx.fillStyle = '#999'
    ctx.font = '9px monospace'
    ctx.textAlign = 'center'
    sectorTotals.forEach((s, i) => {
      const pct = ((s.total / total) * 100).toFixed(0)
      const short = s.sector.length > 12 ? s.sector.slice(0, 10) + '…' : s.sector
      ctx.fillText(`${short} ${pct}%`, cx, cy + r + 16 + i * 12)
    })
  }, [sectorTotals, total, hovered])

  return (
    <div className="relative">
      <div className="text-[10px] text-gold/80 uppercase tracking-wider mb-1 font-semibold">Contract momentum by sector</div>
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-gold/20 bg-black/20"
        style={{ height: 200 }}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          const cx = rect.width / 2
          const cy = rect.height / 2 - 10
          const r = Math.min(rect.width, rect.height) / 2 - 24
          const dx = e.clientX - rect.left - cx
          const dy = e.clientY - rect.top - cy
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist > r) {
            setHovered(null)
            return
          }
          let angle = Math.atan2(dy, dx) + Math.PI / 2
          if (angle < 0) angle += 2 * Math.PI
          let cum = 0
          for (let i = 0; i < sectorTotals.length; i++) {
            cum += (sectorTotals[i].total / total) * 2 * Math.PI
            if (angle <= cum) {
              setHovered(i)
              return
            }
          }
          setHovered(null)
        }}
        onMouseLeave={() => setHovered(null)}
      />
    </div>
  )
}

function ScatterChart({
  points,
  xLabel,
  yLabel,
}: {
  points: { ticker: string; x: number; y: number }[]
  xLabel: string
  yLabel: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  const xs = useMemo(() => points.map((p) => p.x), [points])
  const ys = useMemo(() => points.map((p) => p.y), [points])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1

  useEffect(() => {
    if (!canvasRef.current || points.length === 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height
    const pad = { top: 16, right: 16, bottom: 28, left: 48 }
    const chartW = w - pad.left - pad.right
    const chartH = h - pad.top - pad.bottom

    const toX = (x: number) => pad.left + ((x - minX) / rangeX) * chartW
    const toY = (y: number) => pad.top + chartH - ((y - minY) / rangeY) * chartH

    ctx.clearRect(0, 0, w, h)
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.15)'
    ctx.lineWidth = 1
    for (let i = 1; i <= 4; i++) {
      const y = pad.top + (chartH / 4) * i
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
    }
    for (let i = 1; i <= 4; i++) {
      const x = pad.left + (chartW / 4) * i
      ctx.beginPath()
      ctx.moveTo(x, pad.top)
      ctx.lineTo(x, h - pad.bottom)
      ctx.stroke()
    }
    ctx.strokeStyle = GOLD_LIGHT
    ctx.beginPath()
    ctx.moveTo(pad.left, pad.top)
    ctx.lineTo(pad.left, h - pad.bottom)
    ctx.moveTo(pad.left, h - pad.bottom)
    ctx.lineTo(w - pad.right, h - pad.bottom)
    ctx.stroke()
    ctx.fillStyle = '#999'
    ctx.font = '9px monospace'
    ctx.textAlign = 'right'
    ctx.fillText(formatNum(minY), pad.left - 4, pad.top + 3)
    ctx.fillText(formatNum(maxY), pad.left - 4, h - pad.bottom + 3)
    ctx.textAlign = 'left'
    ctx.fillText(formatNum(minX), pad.left, h - pad.bottom + 14)
    ctx.fillText(formatNum(maxX), w - pad.right - 2, h - pad.bottom + 14)

    points.forEach((p, i) => {
      const px = toX(p.x)
      const py = toY(p.y)
      ctx.beginPath()
      ctx.arc(px, py, hovered === i ? 6 : 4, 0, 2 * Math.PI)
      ctx.fillStyle = hovered === i ? GOLD : GOLD_LIGHT
      ctx.fill()
      ctx.strokeStyle = 'rgba(0,0,0,0.2)'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }, [points, minX, maxX, minY, maxY, rangeX, rangeY, hovered])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const pad = { left: 48, right: 16, top: 16, bottom: 28 }
    const chartW = rect.width - pad.left - pad.right
    const chartH = rect.height - pad.top - pad.bottom
    const mx = e.clientX - rect.left - pad.left
    const my = e.clientY - rect.top - pad.top
    const dataX = minX + (mx / chartW) * rangeX
    const dataY = maxY - (my / chartH) * rangeY
    let best = -1
    let bestD = 999999
    points.forEach((p, i) => {
      const dx = p.x - dataX
      const dy = p.y - dataY
      const d = dx * dx + dy * dy
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    setHovered(best)
  }

  return (
    <div className="relative">
      <div className="text-[10px] text-gold/80 uppercase tracking-wider mb-1 font-semibold">
        Momentum vs centrality (by ticker)
      </div>
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-gold/20 bg-black/20"
        style={{ height: 220 }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHovered(null)}
      />
      {hovered !== null && hovered < points.length && (
        <div className="absolute top-0 right-0 px-2 py-1 rounded bg-dark-800 border border-gold/30 text-xs text-gold shadow-lg z-10">
          {points[hovered].ticker}: {xLabel}={formatNum(points[hovered].x)}, {yLabel}={formatNum(points[hovered].y)}
        </div>
      )}
    </div>
  )
}

export default function SignalsPage() {
  const [allData, setAllData] = useState<AllSignalsResponse | null>(null)
  const [backtestData, setBacktestData] = useState<BacktestResponse | null>(null)
  const [backtestFallbackNote, setBacktestFallbackNote] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [backtestLoading, setBacktestLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [backtestError, setBacktestError] = useState<string | null>(null)

  const [corrTickerA, setCorrTickerA] = useState('AAPL')
  const [corrTickerB, setCorrTickerB] = useState('MSFT')
  const [corrWindowDays, setCorrWindowDays] = useState(90)
  const [corrResult, setCorrResult] = useState<{ correlation: number; p_value: number; n_observations: number; date_range?: { min: string; max: string } } | null>(null)
  const [corrLoading, setCorrLoading] = useState(false)
  const [corrError, setCorrError] = useState<string | null>(null)

  const contractLimit = 20
  const optionsLimit = 15
  const centralityLimit = 20
  const backtestStart2024 = '2024-01-01'
  const backtestEnd2024 = '2024-12-31'
  const backtestStart2023 = '2023-01-01'
  const backtestEnd2023 = '2023-12-31'

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
    let cancelled = false
    const run = async (start: string, end: string, fallbackNote: string | null): Promise<void> => {
      try {
        setBacktestLoading(true)
        setBacktestError(null)
        setBacktestFallbackNote(fallbackNote)
        const res = await fetch(`${API_BASE}/api/signals/backtest?start_date=${start}&end_date=${end}`)
        if (!res.ok) throw new Error(`Backtest failed: ${res.status}`)
        const data = await res.json()
        if (cancelled) return
        setBacktestData(data)
        if (data.error && start === backtestStart2024 && end === backtestEnd2024) {
          await run(backtestStart2023, backtestEnd2023, '2024 had insufficient data; showing 2023.')
          return
        }
      } catch (e) {
        if (cancelled) return
        setBacktestError(e instanceof Error ? e.message : 'Failed to load backtest')
        if (start === backtestStart2024) {
          await run(backtestStart2023, backtestEnd2023, '2024 failed; showing 2023.')
          return
        }
      } finally {
        if (!cancelled) setBacktestLoading(false)
      }
    }
    run(backtestStart2024, backtestEnd2024, null)
    return () => {
      cancelled = true
    }
  }, [])

  const momentumWithShare = useMemo(() => {
    const raw = allData?.contract_momentum_90d
    const rows = Array.isArray(raw) ? raw.filter((r): r is ContractMomentumRow => r != null && r.ticker != null) : []
    const total = rows.reduce((s, r) => s + (r.contract_momentum_90d || 0), 0)
    if (total <= 0) return rows
    return rows.map((r) => ({
      ...r,
      share_of_total_pct: (r.contract_momentum_90d / total) * 100,
    }))
  }, [allData?.contract_momentum_90d])

  const centralityWithShare = useMemo(() => {
    const raw = allData?.contract_centrality
    const rows = Array.isArray(raw) ? raw.filter((r): r is CentralityRow => r != null && r.ticker != null) : []
    const total = rows.reduce((s, r) => s + (r.contract_degree_centrality || 0), 0)
    if (total <= 0) return rows
    return rows.map((r) => ({
      ...r,
      share_of_total_pct: (r.contract_degree_centrality / total) * 100,
    }))
  }, [allData?.contract_centrality])

  const sectorTotals = useMemo(() => {
    const map = new Map<string, number>()
    momentumWithShare.forEach((r) => {
      const sec = r.sector || 'Other'
      map.set(sec, (map.get(sec) ?? 0) + r.contract_momentum_90d)
    })
    return Array.from(map.entries())
      .map(([sector, total]) => ({ sector, total }))
      .sort((a, b) => b.total - a.total)
      .slice(0, 6)
  }, [momentumWithShare])

  const scatterPoints = useMemo(() => {
    const momentum = Array.isArray(allData?.contract_momentum_90d) ? allData.contract_momentum_90d.filter((r): r is ContractMomentumRow => r != null && r.ticker != null) : []
    const centrality = Array.isArray(allData?.contract_centrality) ? allData.contract_centrality.filter((r): r is CentralityRow => r != null && r.ticker != null) : []
    const byTicker = new Map<string, { x: number; y: number }>()
    momentum.forEach((r) => byTicker.set(r.ticker, { x: r.contract_momentum_90d, y: 0 }))
    centrality.forEach((r) => {
      const prev = byTicker.get(r.ticker)
      if (prev) prev.y = r.contract_degree_centrality
      else byTicker.set(r.ticker, { x: 0, y: r.contract_degree_centrality })
    })
    return Array.from(byTicker.entries())
      .filter(([, v]) => v.x > 0 && v.y > 0)
      .map(([ticker, v]) => ({ ticker, x: v.x, y: v.y }))
      .slice(0, 80)
  }, [allData?.contract_momentum_90d, allData?.contract_centrality])

  const maxMomentum = useMemo(
    () => Math.max(...(momentumWithShare.map((r) => r.contract_momentum_90d) || [1]), 1),
    [momentumWithShare]
  )
  const maxCentrality = useMemo(
    () => Math.max(...(centralityWithShare.map((r) => r.contract_degree_centrality) || [1]), 1),
    [centralityWithShare]
  )

  const optionsFilingRows = useMemo(() => {
    const raw = allData?.options_filing_convergence
    return Array.isArray(raw) ? raw.filter((r): r is OptionsFilingRow => r != null && r.ticker != null) : []
  }, [allData?.options_filing_convergence])

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
        {/* Contract momentum: chart + table */}
        <div className="space-y-4">
          {!loading && (allData?.contract_momentum_90d?.length ?? 0) > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4">
                <BarChart
                  labels={(allData?.contract_momentum_90d ?? []).filter((r): r is ContractMomentumRow => r != null && r.ticker != null).slice(0, 10).map((r) => r.ticker)}
                  values={(allData?.contract_momentum_90d ?? []).filter((r): r is ContractMomentumRow => r != null && r.ticker != null).slice(0, 10).map((r) => r.contract_momentum_90d)}
                  valueLabel="90d contract total (top 10)"
                  height={220}
                />
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4">
                <SectorPieChart sectorTotals={sectorTotals} />
              </div>
            </div>
          )}
          <SignalsTable<ContractMomentumRow & { share_of_total_pct?: number }>
            title="Contract momentum (90d)"
            rows={momentumWithShare}
            loading={loading}
            noDataExplanation={undefined}
            maxForBar={maxMomentum}
            columns={[
              { key: 'ticker', label: 'Ticker' },
              { key: 'company', label: 'Company' },
              { key: 'sector', label: 'Sector' },
              {
                key: 'contract_momentum_90d',
                label: '90d total',
                format: (v) => (typeof v === 'number' ? formatNum(v) : '—'),
                isNumeric: true,
                barOfMax: true,
              },
              {
                key: 'share_of_total_pct',
                label: 'Share',
                format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : '—'),
                isNumeric: true,
              },
              { key: 'award_count_90d', label: 'Awards', isNumeric: true },
              { key: 'momentum_rank', label: 'Rank', isRank: true },
            ]}
          />
        </div>

        {/* Options–filing: chart + table */}
        <div className="space-y-4">
          {!loading && (allData?.options_filing_convergence?.length ?? 0) > 0 && (
            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 max-w-2xl">
              <BarChart
                labels={(allData?.options_filing_convergence ?? []).filter((r): r is OptionsFilingRow => r != null && r.ticker != null).slice(0, 10).map((r) => r.ticker)}
                values={(allData?.options_filing_convergence ?? []).filter((r): r is OptionsFilingRow => r != null && r.ticker != null).slice(0, 10).map((r) => r.options_filing_events)}
                valueLabel="Options–filing events (top 10)"
                height={200}
              />
            </div>
          )}
          <SignalsTable<OptionsFilingRow>
            title="Options–filing convergence"
            rows={optionsFilingRows}
            loading={loading}
            noDataExplanation="Requires OPTIONS_BEFORE_FILING edges between options activity and SEC filings."
            columns={[
              { key: 'ticker', label: 'Ticker' },
              { key: 'options_filing_events', label: 'Events', isNumeric: true },
              { key: 'unusual_activity_count', label: 'Unusual count', isNumeric: true },
              {
                key: 'max_unusual_ratio',
                label: 'Max unusual ratio',
                format: (v) => (typeof v === 'number' ? v.toFixed(2) : '—'),
                isNumeric: true,
              },
            ]}
          />
        </div>

        {/* Centrality: chart + table + scatter */}
        <div className="space-y-4">
          {!loading && (allData?.contract_centrality?.length ?? 0) > 0 && (
            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4">
              <BarChart
                labels={(allData?.contract_centrality ?? []).filter((r): r is CentralityRow => r != null && r.ticker != null).slice(0, 10).map((r) => r.ticker)}
                values={(allData?.contract_centrality ?? []).filter((r): r is CentralityRow => r != null && r.ticker != null).slice(0, 10).map((r) => r.contract_degree_centrality)}
                valueLabel="Contract degree centrality (top 10)"
                height={220}
              />
            </div>
          )}
          {scatterPoints.length > 0 && (
            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4">
              <ScatterChart
                points={scatterPoints}
                xLabel="90d momentum"
                yLabel="centrality"
              />
            </div>
          )}
          <SignalsTable<CentralityRow & { share_of_total_pct?: number }>
            title="Contract centrality"
            rows={centralityWithShare}
            loading={loading}
            noDataExplanation={undefined}
            maxForBar={maxCentrality}
            columns={[
              { key: 'ticker', label: 'Ticker' },
              { key: 'company', label: 'Company' },
              { key: 'sector', label: 'Sector' },
              {
                key: 'contract_degree_centrality',
                label: 'Degree',
                format: (v) => (typeof v === 'number' ? formatNum(v) : '—'),
                isNumeric: true,
                barOfMax: true,
              },
              {
                key: 'share_of_total_pct',
                label: 'Share',
                format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : '—'),
                isNumeric: true,
              },
            ]}
          />
        </div>

        {/* Backtest */}
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gold mb-4 border-b border-gold/30 pb-2">
            Backtest (contract momentum vs forward returns)
          </h2>
          {backtestFallbackNote && (
            <p className="text-amber-300/90 text-sm mb-3">{backtestFallbackNote}</p>
          )}
          {backtestLoading && <div className="text-gray-400">Loading…</div>}
          {backtestError && (
            <div className="text-red-300 mb-2">{backtestError}</div>
          )}
          {backtestData?.error && (
            <>
              <div className="text-amber-300">{backtestData.error}</div>
              <p className="text-gray-500 text-sm mt-2">
                Requires award and market data in the selected date range. Try 2023 if 2024 has no data.
              </p>
            </>
          )}
          {!backtestLoading && backtestData && !backtestData.error && (
            <>
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
                <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                  <div className="text-gray-500 mb-1">Mean return</div>
                  <div className="text-gold font-mono">
                    {backtestData.mean_return != null ? (backtestData.mean_return * 100).toFixed(2) + '%' : '—'}
                  </div>
                </div>
                <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                  <div className="text-gray-500 mb-1">Volatility</div>
                  <div className="text-gold font-mono">
                    {backtestData.return_volatility != null ? (backtestData.return_volatility * 100).toFixed(2) + '%' : '—'}
                  </div>
                </div>
                <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                  <div className="text-gray-500 mb-1">Total return</div>
                  <div className="text-gold font-mono">
                    {backtestData.total_return != null ? (backtestData.total_return * 100).toFixed(2) + '%' : '—'}
                  </div>
                </div>
                <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                  <div className="text-gray-500 mb-1">Max drawdown</div>
                  <div className="text-gold font-mono">
                    {backtestData.max_drawdown != null ? (backtestData.max_drawdown * 100).toFixed(2) + '%' : '—'}
                  </div>
                </div>
              </div>
              <p className="text-gray-500 text-xs mt-4">
                {backtestData.point_in_time}
              </p>
              <p className="text-gray-400 text-sm mt-2 max-w-2xl">
                Rank IC &gt; 0 means higher contract momentum tended to predict higher forward returns; hit rate is the share of observations with positive forward return. Use: rank companies by 90d contract momentum; consider top quintile for the forward period.
              </p>
            </>
          )}
        </div>

        {/* Correlation */}
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-6 mt-6">
          <h2 className="text-xl font-semibold text-gold mb-4 border-b border-gold/30 pb-2">
            Correlation (pairwise time series)
          </h2>
          <p className="text-gray-400 text-sm mb-4">
            Compare two series (e.g. two tickers’ close price). Aligned by date; Pearson correlation and p-value.
          </p>
          <div className="flex flex-wrap items-end gap-3 mb-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Series A (ticker)</label>
              <input
                type="text"
                value={corrTickerA}
                onChange={(e) => setCorrTickerA(e.target.value.toUpperCase())}
                className="bg-dark-900 border border-gold/20 rounded px-3 py-1.5 text-sm text-white w-24"
                placeholder="AAPL"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Series B (ticker)</label>
              <input
                type="text"
                value={corrTickerB}
                onChange={(e) => setCorrTickerB(e.target.value.toUpperCase())}
                className="bg-dark-900 border border-gold/20 rounded px-3 py-1.5 text-sm text-white w-24"
                placeholder="MSFT"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Window (days)</label>
              <input
                type="number"
                value={corrWindowDays}
                onChange={(e) => setCorrWindowDays(parseInt(e.target.value, 10) || 90)}
                min={7}
                max={730}
                className="bg-dark-900 border border-gold/20 rounded px-3 py-1.5 text-sm text-white w-20"
              />
            </div>
            <button
              type="button"
              onClick={async () => {
                setCorrLoading(true)
                setCorrError(null)
                setCorrResult(null)
                try {
                  const res = await fetch(`${API_BASE}/api/analyze/correlation`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      series_a: { collection: 'MarketData', field: 'close', filter: { ticker: corrTickerA } },
                      series_b: { collection: 'MarketData', field: 'close', filter: { ticker: corrTickerB } },
                      window_days: corrWindowDays,
                      method: 'pearson',
                    }),
                  })
                  if (!res.ok) {
                    const err = await res.json().catch(() => ({}))
                    throw new Error(err.detail || res.statusText)
                  }
                  const data = await res.json()
                  setCorrResult(data)
                } catch (e) {
                  setCorrError(e instanceof Error ? e.message : 'Correlation failed')
                } finally {
                  setCorrLoading(false)
                }
              }}
              disabled={corrLoading || !corrTickerA.trim() || !corrTickerB.trim()}
              className="px-4 py-1.5 text-sm bg-gold/20 text-gold rounded border border-gold/40 hover:bg-gold/30 disabled:opacity-50"
            >
              {corrLoading ? 'Running…' : 'Run'}
            </button>
          </div>
          {corrError && <p className="text-red-300 text-sm mb-2">{corrError}</p>}
          {corrResult && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">Correlation</div>
                <div className="text-gold font-mono">{corrResult.correlation.toFixed(4)}</div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">p-value</div>
                <div className="text-gold font-mono">{corrResult.p_value.toFixed(4)}</div>
              </div>
              <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                <div className="text-gray-500 mb-1">N observations</div>
                <div className="text-gold font-mono">{corrResult.n_observations}</div>
              </div>
              {corrResult.date_range && (
                <div className="bg-dark-700/50 rounded-lg p-3 border border-gold/10">
                  <div className="text-gray-500 mb-1">Date range</div>
                  <div className="text-gold font-mono text-xs">{corrResult.date_range.min} to {corrResult.date_range.max}</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
