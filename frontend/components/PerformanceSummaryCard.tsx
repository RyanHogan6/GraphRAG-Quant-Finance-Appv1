'use client'

import { useMemo } from 'react'

export interface PerformanceSummaryCardProps {
  dates: string[]
  values: number[]
  volumes?: number[]
  ticker?: string
  label?: string
}

function computeStats(dates: string[], values: number[], volumes?: number[]) {
  if (!values?.length) return null
  const first = values[0]
  const last = values[values.length - 1]
  const change = last - first
  const changePct = first ? (change / first) * 100 : 0
  const high = Math.max(...values)
  const low = Math.min(...values)
  const avg = values.reduce((a, b) => a + b, 0) / values.length
  const volatility = avg ? ((high - low) / avg) * 100 : 0
  const avgVolume = volumes?.length
    ? volumes.reduce((a, b) => a + b, 0) / volumes.length
    : 0
  return {
    periodStart: dates[0] || 'N/A',
    periodEnd: dates[dates.length - 1] || 'N/A',
    tradingDays: dates.length,
    first,
    last,
    change,
    changePct,
    high,
    low,
    avg,
    volatility,
    avgVolume,
  }
}

function buildInsights(stats: NonNullable<ReturnType<typeof computeStats>>): string[] {
  const out: string[] = []
  if (stats.changePct > 10) {
    out.push(`Strong upward trend with ${stats.changePct.toFixed(1)}% gain over the period`)
  } else if (stats.changePct < -10) {
    out.push(`Significant decline of ${stats.changePct.toFixed(1)}% over the period`)
  } else {
    out.push(`Relatively stable price movement (${stats.changePct >= 0 ? '+' : ''}${stats.changePct.toFixed(1)}% change)`)
  }
  if (stats.volatility > 15) {
    out.push(`High volatility with ${stats.volatility.toFixed(1)}% price range`)
  } else if (stats.volatility < 5) {
    out.push(`Low volatility with ${stats.volatility.toFixed(1)}% price range`)
  }
  return out
}

export default function PerformanceSummaryCard({
  dates,
  values,
  volumes,
  ticker,
  label,
}: PerformanceSummaryCardProps) {
  const stats = useMemo(
    () => computeStats(dates, values, volumes),
    [dates, values, volumes]
  )
  const insights = useMemo(
    () => (stats ? buildInsights(stats) : []),
    [stats]
  )

  if (!stats) return null

  const title = ticker ? `${ticker} Stock Performance` : (label || 'Stock Performance')
  const changeIndicator = stats.change >= 0 ? '📈' : '📉'

  return (
    <div className="bg-dark-900/40 border border-gold/20 rounded-xl p-4 shadow-xl backdrop-blur-sm">
      <h3 className="text-sm md:text-base font-bold text-gold uppercase tracking-wider mb-3">
        {title}
      </h3>
      <div className="text-[10px] md:text-xs text-gray-400 mb-4">
        Period: {stats.periodStart} to {stats.periodEnd} ({stats.tradingDays} trading days)
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <div className="text-[10px] font-semibold text-gold uppercase tracking-wider mb-2">
            Price Summary
          </div>
          <ul className="space-y-1 text-xs text-gray-300">
            <li><span className="text-gray-500">Starting Price:</span> ${stats.first.toFixed(2)}</li>
            <li><span className="text-gray-500">Ending Price:</span> ${stats.last.toFixed(2)}</li>
            <li>
              <span className="text-gray-500">Change:</span>{' '}
              <span className={stats.change >= 0 ? 'text-green-400' : 'text-red-400'}>
                {changeIndicator} ${stats.change >= 0 ? '+' : ''}{stats.change.toFixed(2)} ({stats.changePct >= 0 ? '+' : ''}{stats.changePct.toFixed(2)}%)
              </span>
            </li>
            <li><span className="text-gray-500">High:</span> ${stats.high.toFixed(2)}</li>
            <li><span className="text-gray-500">Low:</span> ${stats.low.toFixed(2)}</li>
            <li><span className="text-gray-500">Average:</span> ${stats.avg.toFixed(2)}</li>
          </ul>
        </div>

        <div>
          <div className="text-[10px] font-semibold text-gold uppercase tracking-wider mb-2">
            Trading Activity
          </div>
          <ul className="space-y-1 text-xs text-gray-300">
            <li><span className="text-gray-500">Average Daily Volume:</span> {stats.avgVolume.toLocaleString('en-US', { maximumFractionDigits: 0 })} shares</li>
            <li><span className="text-gray-500">Total Days:</span> {stats.tradingDays}</li>
          </ul>
        </div>

        <div>
          <div className="text-[10px] font-semibold text-gold uppercase tracking-wider mb-2">
            Key Insights
          </div>
          <ul className="space-y-1.5 text-xs text-gray-300">
            {insights.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
