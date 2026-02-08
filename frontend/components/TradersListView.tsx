'use client'

import ResultsTable from './ResultsTable'

interface TradersListViewProps {
  results: any[]
  maxRows?: number
}

function shortAddress(addr: string | null | undefined): string {
  if (!addr || typeof addr !== 'string') return '—'
  if (addr.length <= 14) return addr
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`
}

/** Polymarket traders: table or compact cards with address (short), total_profit, total_volume, is_whale */
export default function TradersListView({ results, maxRows = 50 }: TradersListViewProps) {
  const displayResults = results.slice(0, maxRows)

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center gap-2 border-b border-gold/20 pb-3">
        <span className="text-2xl">🐋</span>
        <h3 className="text-lg font-bold text-white">Polymarket Traders</h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {displayResults.map((row, idx) => (
          <div
            key={row._key ?? row.address ?? idx}
            className="rounded-lg border border-gold/20 bg-dark-800/80 p-4 hover:border-gold/40 transition-colors"
          >
            <div className="font-mono text-sm text-gray-300 mb-2">
              {shortAddress(row.address ?? row.trader_address)}
            </div>
            <div className="flex flex-wrap gap-3 text-xs">
              {row.total_profit != null && (
                <span className={Number(row.total_profit) >= 0 ? 'text-green-400' : 'text-red-400'}>
                  P/L: ${Number(row.total_profit).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              )}
              {row.total_volume != null && (
                <span className="text-gray-400">
                  Vol: ${Number(row.total_volume).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              )}
              {row.is_whale && (
                <span className="text-gold font-semibold">Whale</span>
              )}
            </div>
          </div>
        ))}
      </div>
      {results.length > maxRows && (
        <p className="text-xs text-gray-500">Showing {maxRows} of {results.length} traders</p>
      )}
      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-gold hover:text-gold/80 font-semibold">
          View as table
        </summary>
        <div className="mt-2 overflow-hidden">
          <ResultsTable data={results} maxRows={maxRows} />
        </div>
      </details>
    </div>
  )
}
