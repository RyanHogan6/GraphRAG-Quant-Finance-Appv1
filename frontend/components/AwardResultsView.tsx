'use client'

import ResultsTable from './ResultsTable'

interface AwardResultsViewProps {
  results: any[]
  maxRows?: number
}

/** Award list: cards or table with recipient_name, award_amount_float, start_date, awarding_agency, ticker */
export default function AwardResultsView({ results, maxRows = 50 }: AwardResultsViewProps) {
  const formatAmount = (val: any) => {
    if (val == null || Number.isNaN(Number(val))) return '—'
    const n = Number(val)
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
    if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
    if (n >= 1e3) return `$${(n / 1e3).toFixed(0)}K`
    return `$${n.toLocaleString()}`
  }

  const displayResults = results.slice(0, maxRows)

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center gap-2 border-b border-gold/20 pb-3">
        <span className="text-2xl">🏛️</span>
        <h3 className="text-lg font-bold text-white">Government Awards</h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {displayResults.map((row, idx) => (
          <div
            key={row._key ?? idx}
            className="rounded-lg border border-gold/20 bg-dark-800/80 p-4 hover:border-gold/40 transition-colors"
          >
            <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">
              {row.awarding_agency ?? row.recipient ?? row.recipient_name ?? '—'}
            </div>
            <div className="text-lg font-semibold text-gold mb-1">
              {formatAmount(row.award_amount_float ?? row.award_amount ?? row.amount)}
            </div>
            <div className="text-sm text-gray-300">
              {row.start_date ?? row.award_date ?? row.date ?? '—'}
              {row.ticker && (
                <span className="ml-2 text-gray-500 font-mono">{row.ticker}</span>
              )}
            </div>
          </div>
        ))}
      </div>
      {results.length > maxRows && (
        <p className="text-xs text-gray-500">Showing {maxRows} of {results.length} awards</p>
      )}
      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-gold hover:text-gold/80 font-semibold">
          View as table ({results.length} rows)
        </summary>
        <div className="mt-2 overflow-hidden">
          <ResultsTable data={results} maxRows={maxRows} />
        </div>
      </details>
    </div>
  )
}
