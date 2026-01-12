import { Market } from '@/lib/mockData'

interface MarketCardProps {
  market: Market
  onClick?: () => void
}

export default function MarketCard({ market, onClick }: MarketCardProps) {
  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(1)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(0)}k`
    return `$${volume}`
  }

  // Detect market type from question/category
  const detectMarketType = () => {
    const q = market.question.toLowerCase()
    const cat = market.category?.toLowerCase() || ''

    // Sports categories
    if (cat.includes('sport') || cat.includes('nba') || cat.includes('nfl') ||
        cat.includes('nhl') || cat.includes('mlb') || q.includes(' vs ') ||
        q.includes('win the') && (q.includes('game') || q.includes('match'))) {
      return 'sports'
    }

    // Multiple choice - has outcomes array
    if (market.outcomes && market.outcomes.length > 2) {
      return 'multiple'
    }

    // Binary yes/no
    return 'binary'
  }

  const marketType = detectMarketType()

  // Determine which side to highlight (higher probability)
  const leadingSide = market.yes_prob > market.no_prob ? 'yes' : 'no'
  const leadingProb = Math.max(market.yes_prob, market.no_prob)

  return (
    <div
      className="relative bg-dark-800 border border-gold/20 rounded-lg p-4 hover:border-gold/40 transition-all cursor-pointer group"
      onClick={onClick}
    >
      {/* Top: Question + Probability Badge */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 pr-3">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            {market.category}
          </div>
          <h3 className="text-sm text-gray-100 font-medium leading-tight group-hover:text-gold transition-colors">
            {market.question}
          </h3>
        </div>

        {/* Circular Probability Badge (like Polymarket) */}
        <div className="flex flex-col items-center shrink-0">
          <div className={`
            w-14 h-14 rounded-full flex items-center justify-center border-2
            ${leadingSide === 'yes' ? 'border-green-500/50 bg-green-500/10' : 'border-red-500/50 bg-red-500/10'}
          `}>
            <div className="text-center">
              <div className={`text-lg font-bold ${leadingSide === 'yes' ? 'text-green-400' : 'text-red-400'}`}>
                {leadingProb}%
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-600 mt-1">chance</div>
        </div>
      </div>

      {/* Middle: Outcomes based on type */}
      {marketType === 'multiple' && market.outcomes ? (
        /* Multiple Choice: List of options */
        <div className="space-y-2 mb-4">
          {market.outcomes.slice(0, 3).map((outcome, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between bg-dark-700/50 border border-gold/10 rounded px-3 py-2 hover:bg-dark-700 transition-all"
            >
              <span className="text-xs text-gray-300">{outcome.name}</span>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-semibold text-gold">{outcome.prob}%</span>
                <div className="flex space-x-1">
                  <span className="text-xs text-green-400">Yes</span>
                  <span className="text-xs text-red-400">No</span>
                </div>
              </div>
            </div>
          ))}
          {market.outcomes.length > 3 && (
            <div className="text-xs text-gray-500 text-center">
              +{market.outcomes.length - 3} more options
            </div>
          )}
        </div>
      ) : (
        /* Binary: Simple Yes/No buttons without percentages */
        <div className="grid grid-cols-2 gap-2 mb-4">
          <button className="bg-green-500/10 hover:bg-green-500/20 border border-green-500/30 hover:border-green-500/50 rounded-lg py-3 px-4 transition-all">
            <div className="text-center">
              <div className="text-sm font-medium text-green-400">{market.outcome_yes || 'Yes'}</div>
            </div>
          </button>
          <button className="bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 hover:border-red-500/50 rounded-lg py-3 px-4 transition-all">
            <div className="text-center">
              <div className="text-sm font-medium text-red-400">{market.outcome_no || 'No'}</div>
            </div>
          </button>
        </div>
      )}

      {/* Bottom: Stats */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center space-x-3 text-gray-500">
          <div title="24h Volume">
            <span className="text-gold">{formatVolume(market.volume_24h)}</span>
            <span className="ml-1">Vol.</span>
          </div>
          {market.traders > 0 && (
            <div className="flex items-center space-x-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
              </svg>
              <span>{market.traders.toLocaleString()}</span>
            </div>
          )}
        </div>
        <div className="text-gray-600">
          {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        </div>
      </div>
    </div>
  )
}
