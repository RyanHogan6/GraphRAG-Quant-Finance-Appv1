'use client'

/**
 * Evidence Chain View - Shows the graph path that supports an answer.
 * Renders when the backend returns decomposed narrative results (metadata.decomposed + evidence_paths).
 * Nodes = data sources (collections), edges = relationships (edge_path).
 */

export interface EvidencePathStep {
  label: string
  edge_path?: string[]
  collections?: string[]
  result_count?: number
}

interface EvidenceChainViewProps {
  evidencePaths: EvidencePathStep[]
  className?: string
}

function formatLabel(label: string): string {
  const map: Record<string, string> = {
    contracts: 'Gov Contracts',
    sec_filings: 'SEC Filings',
    options_flow: 'Options Flow',
    market_data: 'Market Data',
    macro: 'Macro / Economic'
  }
  return map[label] || label.replace(/_/g, ' ')
}

export default function EvidenceChainView({ evidencePaths, className = '' }: EvidenceChainViewProps) {
  if (!evidencePaths || evidencePaths.length === 0) return null

  return (
    <div className={`rounded-lg border border-amber-500/30 bg-amber-950/20 p-4 ${className}`}>
      <div className="text-xs font-semibold uppercase tracking-wider text-amber-400/90 mb-3">
        Evidence chain (graph path)
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {evidencePaths.map((step, i) => (
          <span key={step.label} className="flex items-center gap-2">
            <span
              className="inline-flex items-center gap-1.5 rounded-md bg-dark-800/80 px-2.5 py-1 text-sm text-gray-200 border border-gray-600/50"
              title={step.collections?.join(' → ') || step.label}
            >
              <span className="text-amber-400/90">{formatLabel(step.label)}</span>
              {typeof step.result_count === 'number' && (
                <span className="text-gray-500 text-xs">({step.result_count})</span>
              )}
            </span>
            {step.edge_path && step.edge_path.length > 0 && (
              <span className="text-gray-500 text-xs font-mono">
                {step.edge_path.join(' → ')}
              </span>
            )}
            {i < evidencePaths.length - 1 && (
              <span className="text-gray-500 mx-0.5" aria-hidden="true">→</span>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}
