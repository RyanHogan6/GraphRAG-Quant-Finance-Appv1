'use client'

interface SECFilingsListViewProps {
  results: any[]
  maxRows?: number
}

const SEC_EDGAR_BASE = 'https://www.sec.gov/cgi-bin/browse-edgar'

/** SEC filings list: type, filing_date, ticker, avg_finbert; card layout + link to SEC viewer */
export default function SECFilingsListView({ results, maxRows = 50 }: SECFilingsListViewProps) {
  const displayResults = results.slice(0, maxRows)

  const secUrl = (row: any) => {
    const ticker = row.ticker ?? row.company
    if (ticker) return `${SEC_EDGAR_BASE}?action=getcompany&company=${encodeURIComponent(ticker)}`
    return null
  }

  const sentimentColor = (score: number | null | undefined) => {
    if (score == null || Number.isNaN(score)) return 'text-gray-400'
    if (score < -0.2) return 'text-red-400'
    if (score > 0.2) return 'text-green-400'
    return 'text-gray-300'
  }

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center gap-2 border-b border-gold/20 pb-3">
        <span className="text-2xl">📄</span>
        <h3 className="text-lg font-bold text-white">SEC Filings</h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {displayResults.map((row, idx) => {
          const url = secUrl(row)
          const score = row.avg_finbert ?? row.finbert_score
          return (
            <div
              key={row._key ?? row.accession ?? idx}
              className="rounded-lg border border-gold/20 bg-dark-800/80 p-4 hover:border-gold/40 transition-colors"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className="font-mono text-sm font-semibold text-gold">
                    {row.type ?? row.filing_type ?? 'Filing'}
                  </span>
                  {row.ticker && (
                    <span className="ml-2 text-xs text-gray-500 font-mono">{row.ticker}</span>
                  )}
                </div>
                {score != null && !Number.isNaN(Number(score)) && (
                  <span className={`text-sm font-medium ${sentimentColor(Number(score))}`}>
                    {(Number(score)).toFixed(2)}
                  </span>
                )}
              </div>
              <div className="text-sm text-gray-400 mt-1">
                {row.filing_date ?? row.date ?? '—'}
              </div>
              {url && (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block mt-2 text-xs text-gold hover:text-gold/80 font-semibold"
                >
                  Open on SEC EDGAR →
                </a>
              )}
            </div>
          )
        })}
      </div>
      {results.length > maxRows && (
        <p className="text-xs text-gray-500">Showing {maxRows} of {results.length} filings</p>
      )}
    </div>
  )
}
