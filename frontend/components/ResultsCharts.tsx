'use client'

import { useMemo, useRef, useEffect, useState } from 'react'
import TimeSeriesChart from './TimeSeriesChart'
import OHLCChart from './OHLCChart'
import PositioningCOTChart from './PositioningCOTChart'
import ProbabilityTimelineChart from './ProbabilityTimelineChart'
import ResultSummaryStrip from './ResultSummaryStrip'
import { computeResultMetrics } from '@/lib/computeResultMetrics'
import type { DisplayFamily } from '@/lib/displayFamily'

const GOLD = '#D4AF37'
const GOLD_LIGHT = 'rgba(212, 175, 55, 0.4)'
const GOLD_DIM = 'rgba(212, 175, 55, 0.15)'

/** Infer label column (for categories) and numeric columns from flat result rows */
function inferColumns(data: Record<string, any>[]): {
  labelKey: string | null
  numericKeys: string[]
  dateKey: string | null
  hasOHLC: boolean
  ohlcDateKey: string | null
} {
  if (!data.length) return { labelKey: null, numericKeys: [], dateKey: null, hasOHLC: false, ohlcDateKey: null }
  const keys = Object.keys(data[0]).filter(k => !k.startsWith('_'))
  const labelCandidates = ['SYMBOL', 'symbol', 'ticker', 'Ticker', 'company', 'Company', 'commodity', 'name', 'title', 'market_ticker', 'contract_symbol', 'contract_code']
  const dateCandidates = ['date', 'Date', 'filing_date', 'start_date', 'as_of_date', 'report_date', 'week_ending']
  const skipNumeric = new Set(['_key', '_id', '_rev', 'outcome_index', 'year', 'month', 'day_of_week', 'day_of_month'])
  let labelKey: string | null = null
  let dateKey: string | null = null
  const numericKeys: string[] = []
  for (const k of keys) {
    if (labelCandidates.includes(k) || (k.length <= 10 && keys.length > 5 && typeof data[0][k] === 'string')) {
      if (!labelKey && typeof data[0][k] === 'string') labelKey = k
    }
    if (dateCandidates.includes(k)) dateKey = k
    const val = data[0][k]
    if (typeof val === 'number' && !skipNumeric.has(k) && !Number.isNaN(val)) numericKeys.push(k)
  }
  if (!labelKey && keys.length > 0) {
    const firstStr = keys.find(k => typeof data[0][k] === 'string' && !dateCandidates.includes(k))
    if (firstStr) labelKey = firstStr
  }
  // OHLC: open, high, low, close (case-insensitive for common DB fields)
  const hasOpen = keys.some(k => k.toLowerCase() === 'open')
  const hasHigh = keys.some(k => k.toLowerCase() === 'high')
  const hasLow = keys.some(k => k.toLowerCase() === 'low')
  const hasClose = keys.some(k => k.toLowerCase() === 'close')
  const ohlcDateKey = dateCandidates.find(k => keys.includes(k)) ?? null
  const hasOHLC = hasOpen && hasHigh && hasLow && hasClose && !!ohlcDateKey
  return { labelKey, numericKeys: numericKeys.slice(0, 8), dateKey, hasOHLC, ohlcDateKey }
}

/** Bar chart via canvas: category (labelKey) vs one numeric series */
function BarChartCanvas({
  labels,
  values,
  valueLabel,
  color = GOLD,
  height = 220
}: {
  labels: string[]
  values: number[]
  valueLabel: string
  color?: string
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
    const gap = chartW / labels.length - barW

    ctx.clearRect(0, 0, w, h)
    // Grid
    ctx.strokeStyle = GOLD_DIM
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (chartH / 4) * i
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
    }
    // Y-axis
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
      ctx.fillText(v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}K` : v.toFixed(0), pad.left - 6, y + 3)
    }
    // Bars
    labels.forEach((label, i) => {
      const x = pad.left + (i + 0.5) * (chartW / labels.length) - barW / 2
      const barH = (values[i] / maxVal) * chartH
      const y = pad.top + chartH - barH
      const isHover = hovered === i
      ctx.fillStyle = isHover ? GOLD : GOLD_LIGHT
      ctx.fillRect(x, y, barW, barH)
    })
    // X labels
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
          {labels[hovered]}: {typeof values[hovered] === 'number' ? values[hovered].toLocaleString(undefined, { maximumFractionDigits: 2 }) : values[hovered]}
        </div>
      )}
    </div>
  )
}

interface ResultsChartsProps {
  data: any[]
  maxRows?: number
  /** When ohlc_candlestick, prefer OHLC chart when data has open/high/low/close; also drives computeResultMetrics. */
  displayFamily?: DisplayFamily
}

/** Normalize OHLC keys (OPEN/CLOSE vs open/close) */
function getOHLC(row: Record<string, any>, dateKey: string) {
  const d = row[dateKey] != null ? String(row[dateKey]).slice(0, 10) : ''
  const o = row.open ?? row.OPEN
  const h = row.high ?? row.HIGH
  const l = row.low ?? row.LOW
  const c = row.close ?? row.CLOSE
  const v = row.volume ?? row.VOLUME
  return { date: d, open: Number(o), high: Number(h), low: Number(l), close: Number(c), volume: typeof v === 'number' ? v : undefined }
}

export default function ResultsCharts({ data, maxRows = 20, displayFamily }: ResultsChartsProps) {
  const inferred = useMemo(() => inferColumns(data), [data])
  const displayData = useMemo(() => data.slice(0, maxRows).filter(row => {
    const v = inferred.labelKey ? row[inferred.labelKey] : null
    return v != null && v !== ''
  }), [data, maxRows, inferred.labelKey])

  const { labelKey, numericKeys, dateKey, hasOHLC, ohlcDateKey } = inferred
  const hasDate = dateKey && displayData.every(r => r[dateKey] != null)
  const hasCategories = labelKey && displayData.length > 0 && displayData.length <= 30

  // OHLC data for candlestick (single series by date; multi-commodity could be filtered by first ticker/commodity)
  const ohlcData = useMemo(() => {
    if (!hasOHLC || !ohlcDateKey) return null
    const sorted = [...displayData].sort((a, b) => String(a[ohlcDateKey]).localeCompare(String(b[ohlcDateKey])))
    return sorted.map(r => getOHLC(r, ohlcDateKey)).filter(r => !Number.isNaN(r.open + r.high + r.low + r.close))
  }, [displayData, hasOHLC, ohlcDateKey])

  // Time series: one or more series when we have date + numeric (commodities: price + inventory = two charts)
  const timeSeries = useMemo(() => {
    if (!hasDate || !dateKey || !numericKeys.length) return null
    const sorted = [...displayData].sort((a, b) => String(a[dateKey]).localeCompare(String(b[dateKey])))
    const dates = sorted.map(r => String(r[dateKey]).slice(0, 10))
    const uniqDates = Array.from(new Set(dates)).sort()
    if (uniqDates.length < 2) return null
    const numKey = numericKeys.find(k => ['close', 'Close', 'OPEN', 'open', 'volume', 'VOLUME'].includes(k)) || numericKeys[0]
    const values = uniqDates.map(d => {
      const row = sorted.find(r => String(r[dateKey]).slice(0, 10) === d)
      return row ? Number(row[numKey]) : 0
    })
    return { dates: uniqDates, values, label: numKey }
  }, [displayData, dateKey, numericKeys, hasDate])

  // Second time series for commodities demo: inventory / stocks (different scale than price)
  const timeSeries2 = useMemo(() => {
    if (!timeSeries || !dateKey || !numericKeys.length) return null
    const inventoryKeys = ['inventory_million_barrels', 'crude_stocks', 'total_stocks', 'stocks_change', 'value']
    const secondKey = numericKeys.find(k => inventoryKeys.includes(k) || k.toLowerCase().includes('inventory') || k.toLowerCase().includes('stock'))
    if (!secondKey || secondKey === timeSeries.label) return null
    const sorted = [...displayData].sort((a, b) => String(a[dateKey]).localeCompare(String(b[dateKey])))
    const uniqDates = timeSeries.dates
    const values = uniqDates.map(d => {
      const row = sorted.find(r => String(r[dateKey]).slice(0, 10) === d)
      return row != null && row[secondKey] != null ? Number(row[secondKey]) : 0
    })
    return { dates: uniqDates, values, label: secondKey.replace(/_/g, ' ') }
  }, [displayData, dateKey, numericKeys, timeSeries])

  if (displayData.length === 0) return null

  const useOHLC = (displayFamily === 'ohlc_candlestick' || hasOHLC) && ohlcData && ohlcData.length >= 2
  const firstKeys = Object.keys(displayData[0] || {})
  const hasCOTShape = firstKeys.includes('Open_Interest_All') || firstKeys.includes('Commercial_Positions_Long_All')
  const useCOT = (displayFamily === 'positioning_cot' || hasCOTShape) && displayData.length >= 2 && displayData.some((r: any) => r.Open_Interest_All != null || r.Commercial_Positions_Long_All != null)
  const hasProbTimeline = firstKeys.some((k: string) => k === 'yes_price' || k === 'yes_probability') && firstKeys.some((k: string) => k === 'datetime' || k === 'date' || k === 'timestamp')
  const useProbTimeline = (displayFamily === 'probability_timeline' || hasProbTimeline) && displayData.length >= 2
  const hasAnyChart = numericKeys.length > 0 && (useOHLC || useCOT || useProbTimeline || timeSeries || (hasCategories && labelKey))

  const summaryMetrics = useMemo(() => {
    const family = (displayFamily ?? (useOHLC ? 'ohlc_candlestick' : hasCategories ? 'company_screener' : 'generic')) as DisplayFamily
    return computeResultMetrics(displayData, family)
  }, [displayData, displayFamily, useOHLC, hasCategories])

  return (
    <div className="space-y-6">
      {summaryMetrics.length > 0 && (
        <ResultSummaryStrip metrics={summaryMetrics} />
      )}
      {!hasAnyChart && (
        <p className="text-sm text-gray-500 italic">Select the Table tab for raw data. Charts appear when results have a category column (e.g. SYMBOL, ticker) and numeric columns.</p>
      )}
      {/* Metric strip: first numeric column summary (skip when OHLC, COT, or probability have their own) */}
      {numericKeys.length > 0 && !useOHLC && !useCOT && !useProbTimeline && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {numericKeys.slice(0, 4).map(key => {
            const nums = displayData.map(r => r[key]).filter((v): v is number => typeof v === 'number' && !Number.isNaN(v))
            const avg = nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0
            const max = nums.length ? Math.max(...nums) : 0
            return (
              <div key={key} className="rounded-lg border border-gold/20 bg-dark-800/50 px-3 py-2">
                <div className="text-[10px] text-gold/70 uppercase tracking-wider truncate">{key.replace(/_/g, ' ')}</div>
                <div className="text-sm font-semibold text-white tabular-nums">
                  {max >= 1e6 ? `${(avg / 1e6).toFixed(2)}M` : max >= 1e3 ? `${avg.toLocaleString(undefined, { maximumFractionDigits: 1 })}` : avg.toFixed(2)}
                </div>
                <div className="text-[10px] text-gray-500">avg of {nums.length} rows</div>
              </div>
            )
          })}
        </div>
      )}

      {/* OHLC / Candlestick when data has open, high, low, close */}
      {useOHLC && (
        <div className="rounded-lg border border-gold/20 bg-black/20 p-3">
          <OHLCChart
            data={ohlcData!}
            title={displayData[0]?.commodity ?? displayData[0]?.ticker ?? displayData[0]?.symbol ? `OHLC` : undefined}
            ticker={displayData[0]?.ticker ?? displayData[0]?.commodity ?? displayData[0]?.symbol}
            showVolume={displayData.some(r => (r.volume ?? r.VOLUME) != null)}
          />
        </div>
      )}

      {/* COT / Positioning when data has Open_Interest_All, Commercial_*, Noncommercial_* */}
      {useCOT && (
        <div className="rounded-lg border border-gold/20 bg-black/20 p-3">
          <PositioningCOTChart
            data={displayData}
            title={displayData[0]?.Market_and_Exchange_Names ? `COT: ${displayData[0].Market_and_Exchange_Names}` : 'COT Positioning'}
            showOI={displayData.some((r: any) => r.Open_Interest_All != null)}
          />
        </div>
      )}

      {/* Probability timeline (yes_price / yes_probability over datetime) */}
      {useProbTimeline && (
        <div className="rounded-lg border border-gold/20 bg-black/20 p-3">
          <ProbabilityTimelineChart
            data={displayData}
            title="Market probability"
            showNoPrice={displayData.some((r: any) => r.no_price != null)}
          />
        </div>
      )}

      {/* Time series when we have dates (price) and not using OHLC or probability timeline */}
      {timeSeries && !useOHLC && !useProbTimeline && (
        <div className="rounded-lg border border-gold/20 bg-black/20 p-3">
          <TimeSeriesChart
            dates={timeSeries.dates}
            values={timeSeries.values}
            label={timeSeries.label}
          />
        </div>
      )}
      {/* Second time series for commodities: inventory / stocks (e.g. crude price + EIA inventory) */}
      {timeSeries2 && !useProbTimeline && (
        <div className="rounded-lg border border-gold/20 bg-black/20 p-3">
          <h4 className="text-xs text-gold/80 font-semibold uppercase tracking-wider mb-2">{timeSeries2.label}</h4>
          <TimeSeriesChart
            dates={timeSeries2.dates}
            values={timeSeries2.values}
            label={timeSeries2.label}
          />
        </div>
      )}

      {/* Bar chart when we have categories + numeric */}
      {hasCategories && labelKey && numericKeys.length > 0 && !timeSeries && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {numericKeys.slice(0, 2).map(numKey => (
            <div key={numKey} className="rounded-lg border border-gold/20 bg-black/20 p-3">
              <BarChartCanvas
                labels={displayData.map(r => String(r[labelKey] ?? ''))}
                values={displayData.map(r => Number(r[numKey]) || 0)}
                valueLabel={numKey.replace(/_/g, ' ')}
                height={220}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
