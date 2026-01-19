interface ResultsTableProps {
  data: any[]
  maxRows?: number
}

export default function ResultsTable({ data, maxRows = 10 }: ResultsTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="text-gray-500 text-center py-4">
        No results found
      </div>
    )
  }

  // Get all unique keys from all objects
  const allKeys = Array.from(
    new Set(data.flatMap(obj => Object.keys(obj)))
  ).filter(key => !key.startsWith('_')) // Filter out internal ArangoDB keys

  // Limit displayed rows
  const displayData = data.slice(0, maxRows)
  const hasMore = data.length > maxRows

  // Format value for display
  const formatValue = (value: any): any => {
    if (value === null || value === undefined) return '-'
    if (typeof value === 'boolean') return value ? '✓' : '✗'

    // Handle nested arrays (e.g., sec_filings, awards)
    if (Array.isArray(value)) {
      return (
        <div className="max-h-40 overflow-y-auto space-y-2">
          {value.map((item: any, i: number) => (
            <div key={i} className="text-xs bg-dark-800 p-2 rounded border border-white/5">
              {/* Recursive or specific rendering */}
              {typeof item === 'object' ? (
                <div className="space-y-1">
                  {Object.entries(item).map(([k, v]) => {
                    if (k.startsWith('_') || k === 'description_embedding') return null // Skip internal/bulky
                    if (k === 'top_sentences' && Array.isArray(v)) {
                      return (
                        <div key={k} className="pl-2 border-l border-gold/20 mt-1">
                          <div className="text-[10px] text-gold uppercase">Top Sentences:</div>
                          {v.map((s: any, si: number) => (
                            <div key={si} className="text-[10px] text-gray-400 mt-1">
                              <span className={s.score > 0 ? 'text-green-400' : 'text-red-400'}>
                                ({typeof s.score === 'number' ? s.score.toFixed(2) : s.score})
                              </span> {s.text?.substring(0, 100)}...
                            </div>
                          ))}
                        </div>
                      )
                    }
                    return (
                      <div key={k} className="flex gap-1">
                        <span className="text-gray-500 font-mono">{k}:</span>
                        <span className="text-gray-300 truncate">{String(v).substring(0, 50)}</span>
                      </div>
                    )
                  })}
                </div>
              ) : String(item)}
            </div>
          ))}
        </div>
      )
    }

    if (typeof value === 'object') return JSON.stringify(value)
    if (typeof value === 'number') {
      return value.toLocaleString()
    }
    const str = String(value)
    return str.length > 200 ? str.substring(0, 200) + '...' : str
  }

  return (
    <div className="my-4">
      <div className="overflow-x-auto rounded-lg border border-gold/30">
        <table className="min-w-full">
          <thead className="bg-gold/10 border-b border-gold/30">
            <tr>
              {allKeys.map((key) => (
                <th
                  key={key}
                  className="px-4 py-2 text-left text-xs font-semibold text-gold uppercase tracking-wider"
                >
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gold/20">
            {displayData.map((row, idx) => (
              <tr key={idx} className="hover:bg-gold/5 transition-colors">
                {allKeys.map((key) => (
                  <td
                    key={key}
                    className="px-4 py-2 text-sm text-gray-300 max-w-xs align-top"
                  >
                    {formatValue(row[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {hasMore && (
        <div className="text-center mt-3 text-sm text-gray-500">
          Showing {maxRows} of {data.length} results
        </div>
      )}
    </div>
  )
}
