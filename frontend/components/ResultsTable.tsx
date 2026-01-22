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

  // Filter out low-signal records (more than 1 null/NaN)
  const filteredData = data.filter(row => {
    const values = Object.values(row).filter(v => !Array.isArray(v) && typeof v !== 'object')
    const nullCount = values.filter(v =>
      v === null ||
      v === undefined ||
      v === '' ||
      v === 'N/A' ||
      (typeof v === 'number' && isNaN(v))
    ).length
    return nullCount <= 1
  })

  // Limit displayed rows
  const displayData = filteredData.slice(0, maxRows)
  const hasMore = filteredData.length > maxRows

  // Format value for display
  const formatValue = (value: any): any => {
    if (value === null || value === undefined) return '-'
    if (typeof value === 'boolean') return value ? '✓' : '✗'

    // Handle nested arrays (e.g., sec_filings, awards)
    if (Array.isArray(value)) {
      return (
        <div className="max-h-40 overflow-y-auto space-y-2 min-w-[200px]">
          {value.map((item: any, i: number) => (
            <div key={i} className="text-xs bg-dark-800 p-2 rounded border border-white/5">
              {/* Recursive or specific rendering */}
              {typeof item === 'object' ? (
                <div className="space-y-1">
                  {Object.entries(item).map(([k, v]) => {
                    if (k.startsWith('_') || k === 'description_embedding') return null // Skip internal/bulky
                    // Special rendering for SEC sentences
                    if (k === 'top_sentences' && Array.isArray(v)) {
                      return (
                        <div key={k} className="pl-2 border-l border-gold/20 mt-1">
                          <div className="text-[10px] text-gold uppercase font-semibold">Top Sentences:</div>
                          {v.map((s: any, si: number) => (
                            <div key={si} className="text-[10px] text-gray-400 mt-1 leading-tight">
                              <span className={s.score > 0 ? 'text-green-400' : 'text-red-400'}>
                                ({typeof s.score === 'number' ? s.score.toFixed(2) : s.score})
                              </span> {s.text?.substring(0, 100)}...
                            </div>
                          ))}
                        </div>
                      )
                    }

                    // Generic rendering for other fields
                    const displayVal = typeof v === 'object' ? JSON.stringify(v).substring(0, 30) + '...' : String(v)
                    return (
                      <div key={k} className="flex gap-1 overflow-hidden">
                        <span className="text-gray-500 font-mono flex-shrink-0">{k}:</span>
                        <span className="text-gray-300 truncate">{displayVal}</span>
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

    if (typeof value === 'object') return <pre className="text-[10px] bg-dark-800 p-1 rounded max-h-20 overflow-auto">{JSON.stringify(value, null, 2)}</pre>
    if (typeof value === 'number') {
      return value.toLocaleString()
    }
    const str = String(value)
    // More aggressive truncation for table cells
    return str.length > 100 ? str.substring(0, 100) + '...' : str
  }

  return (
    <div className="my-2 md:my-4 w-full overflow-hidden">
      <div className="overflow-x-auto rounded-lg border border-gold/30 bg-black/20 scrollbar-thin scrollbar-thumb-gold/20">
        <table className="w-full border-collapse">
          <thead className="bg-gold/10 border-b border-gold/30">
            <tr>
              {allKeys.map((key) => (
                <th
                  key={key}
                  className="px-2 py-1.5 md:px-3 md:py-2 text-left text-[9px] md:text-[10px] font-bold text-gold uppercase tracking-wider whitespace-nowrap"
                >
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gold/10">
            {displayData.map((row, idx) => (
              <tr key={idx} className="hover:bg-gold/5 transition-colors">
                {allKeys.map((key) => (
                  <td
                    key={key}
                    className="px-2 py-1.5 md:px-3 md:py-2 text-[10px] md:text-xs text-gray-300 align-top max-w-[150px] md:max-w-[200px] break-words"
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
