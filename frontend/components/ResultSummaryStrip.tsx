'use client'

import type { ResultMetric } from '@/lib/computeResultMetrics'

interface ResultSummaryStripProps {
  metrics: ResultMetric[]
  className?: string
}

export default function ResultSummaryStrip({ metrics, className = '' }: ResultSummaryStripProps) {
  if (!metrics?.length) return null

  return (
    <div className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 p-3 rounded-lg border border-gold/10 bg-dark-800/50 ${className}`}>
      {metrics.slice(0, 6).map((m, i) => (
        <div key={i}>
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold truncate" title={m.label}>
            {m.label}
          </div>
          <div
            className={`text-sm font-bold font-mono truncate ${
              m.status === 'good' ? 'text-emerald-400' : m.status === 'bad' ? 'text-red-400' : 'text-white'
            }`}
            title={String(m.value)}
          >
            {typeof m.value === 'number' ? m.value.toLocaleString() : m.value}
          </div>
        </div>
      ))}
    </div>
  )
}
