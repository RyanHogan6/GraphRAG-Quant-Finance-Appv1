/**
 * Convert an array of result objects to CSV string (flattens nested values).
 */
function escapeCSV(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return String(value)
  const s = typeof value === 'object' ? JSON.stringify(value) : String(value)
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

export function resultsToCSV(results: any[]): string {
  if (!results?.length) return ''
  const keys = Array.from(
    new Set(results.flatMap((obj) => Object.keys(obj).filter((k) => !k.startsWith('_'))))
  )
  const header = keys.join(',')
  const rows = results.map((row) =>
    keys.map((k) => escapeCSV(row[k])).join(',')
  )
  return [header, ...rows].join('\r\n')
}

export function downloadCSV(csv: string, filename: string = 'karga-export.csv'): void {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
