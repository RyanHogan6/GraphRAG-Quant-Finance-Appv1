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
  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return '-'
    if (typeof value === 'boolean') return value ? '✓' : '✗'
    if (typeof value === 'object') return JSON.stringify(value)
    if (typeof value === 'number') {
      // Format large numbers with commas
      return value.toLocaleString()
    }
    // Truncate long strings
    const str = String(value)
    return str.length > 100 ? str.substring(0, 100) + '...' : str
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
                    className="px-4 py-2 text-sm text-gray-300 max-w-xs"
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
