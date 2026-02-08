'use client'

import { useState } from 'react'

interface SECSentencesViewProps {
  results: any[]
  maxRows?: number
}

/** SEC sentences: text + finbert_score; sentiment styling, expandable text */
export default function SECSentencesView({ results, maxRows = 100 }: SECSentencesViewProps) {
  const displayResults = results.slice(0, maxRows)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  const toggle = (idx: number) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const sentimentColor = (score: number | null | undefined) => {
    if (score == null || Number.isNaN(score)) return 'border-gray-500/50 bg-gray-500/10'
    if (score < -0.2) return 'border-red-500/50 bg-red-500/10'
    if (score > 0.2) return 'border-green-500/50 bg-green-500/10'
    return 'border-gray-500/50 bg-gray-500/10'
  }

  const sentimentText = (score: number | null | undefined) => {
    if (score == null || Number.isNaN(score)) return 'neutral'
    if (score < -0.2) return 'negative'
    if (score > 0.2) return 'positive'
    return 'neutral'
  }

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center gap-2 border-b border-gold/20 pb-3">
        <span className="text-2xl">💬</span>
        <h3 className="text-lg font-bold text-white">SEC Sentence Sentiment</h3>
      </div>
      <div className="space-y-2 max-h-[60vh] overflow-y-auto">
        {displayResults.map((row, idx) => {
          const text = row.text ?? row.sentence ?? row.content ?? ''
          const score = row.finbert_score ?? row.sentiment
          const isExpanded = expanded.has(idx)
          const showExpand = text.length > 120
          return (
            <div
              key={row._key ?? idx}
              className={`rounded-lg border p-3 ${sentimentColor(score)}`}
            >
              <div className="flex items-start gap-2">
                <span className="shrink-0 text-xs font-semibold text-gray-400 w-16">
                  {sentimentText(score)}
                </span>
                <span className="shrink-0 text-xs font-mono text-gray-500">
                  {(score != null && !Number.isNaN(Number(score))) ? Number(score).toFixed(2) : '—'}
                </span>
                <p className="text-sm text-gray-200 flex-1 min-w-0">
                  {showExpand ? (isExpanded ? text : `${text.slice(0, 120)}...`) : text}
                  {showExpand && (
                    <button
                      type="button"
                      onClick={() => toggle(idx)}
                      className="ml-1 text-gold hover:text-gold/80 text-xs font-semibold"
                    >
                      {isExpanded ? 'Less' : 'More'}
                    </button>
                  )}
                </p>
              </div>
            </div>
          )
        })}
      </div>
      {results.length > maxRows && (
        <p className="text-xs text-gray-500">Showing {maxRows} of {results.length} sentences</p>
      )}
    </div>
  )
}
