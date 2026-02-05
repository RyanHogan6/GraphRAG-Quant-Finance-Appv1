'use client'

import { useMemo } from 'react'
import SectorComparison from './SectorComparison'

interface PeerAnalysisProps {
  companies: any[]
}

/**
 * Peer Analysis - Multi-company comparison (may be cross-sector)
 * Delegates to SectorComparison for now, but can be customized
 */
export default function PeerAnalysis({ companies }: PeerAnalysisProps) {
  // For now, use SectorComparison component
  // Future: Add cross-sector specific features like sector breakdown

  return (
    <div>
      <div className="mb-4 px-4 py-2 bg-blue-900/20 border border-blue-500/30 rounded-lg">
        <div className="text-xs text-blue-400">
          💡 Comparing companies across different sectors
        </div>
      </div>
      <SectorComparison companies={companies} />
    </div>
  )
}
