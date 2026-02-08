'use client'

import { useState } from 'react'
import MarketCard from './MarketCard'
import MarketDetailModal from './MarketDetailModal'
import { Market } from '@/lib/types'

interface PredictionMarketsListViewProps {
  results: any[]
  maxRows?: number
}

/** Map flat result row to Market type for MarketCard */
function rowToMarket(row: any): Market {
  const yesP = row.yes_probability ?? row.yes_prob ?? row.probability ?? 50
  return {
    id: row._key ?? row.market_slug ?? row.slug ?? String(Math.random()),
    question: row.question ?? row.title ?? row.market_question ?? String(row._key ?? ''),
    icon: '📊',
    yes_prob: yesP,
    no_prob: row.no_probability ?? row.no_prob ?? (100 - yesP),
    category: row.category ?? row.group ?? 'Market',
    volume_24h: row.volume_24h ?? row.volume ?? 0,
    liquidity: row.liquidity ?? 0,
    end_date: row.end_date_iso ?? row.end_date ?? '',
    outcomes: row.outcomes,
    traders: row.traders ?? 0,
  }
}

/** Prediction markets list: cards with question, yes_probability, volume_24h, category (reuse MarketCard) */
export default function PredictionMarketsListView({ results, maxRows = 30 }: PredictionMarketsListViewProps) {
  const [selectedMarket, setSelectedMarket] = useState<Market | null>(null)
  const displayResults = results.slice(0, maxRows)

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center gap-2 border-b border-gold/20 pb-3">
        <span className="text-2xl">📊</span>
        <h3 className="text-lg font-bold text-white">Prediction Markets</h3>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {displayResults.map((row, idx) => (
          <MarketCard
            key={row._key ?? row.market_slug ?? idx}
            market={rowToMarket(row)}
            onClick={() => setSelectedMarket(rowToMarket(row))}
          />
        ))}
      </div>
      {results.length > maxRows && (
        <p className="text-xs text-gray-500">Showing {maxRows} of {results.length} markets</p>
      )}
      {selectedMarket && (
        <MarketDetailModal
          market={selectedMarket}
          onClose={() => setSelectedMarket(null)}
        />
      )}
    </div>
  )
}
