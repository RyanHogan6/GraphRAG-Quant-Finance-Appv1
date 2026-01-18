import { Market } from '@/lib/mockData'

interface MarketCardProps {
  market: Market
  onClick?: () => void
}

export default function MarketCard({ market, onClick }: MarketCardProps) {
  const formatVolume = (volume: number) => {
    if (volume == null || isNaN(volume) || volume === 0) return '$0'
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(1)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(0)}k`
    return `$${volume.toFixed(2)}`
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
      className="relative bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all cursor-pointer group"
      onClick={onClick}
    >
      {/* Top: Question + Probability Badge */}
      <div className="flex items-start justify-between mb-5">
        <div className="flex-1 pr-4">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-2 font-semibold">
            {market.category}
          </div>
          <h3 className="text-base text-gray-100 font-semibold leading-snug group-hover:text-gold transition-colors">
            {market.question}
          </h3>
        </div>

        {/* Circular Probability Badge (like Polymarket) */}
        <div className="flex flex-col items-center shrink-0">
          <div className={`
            w-16 h-16 rounded-full flex items-center justify-center border-2
            ${leadingSide === 'yes' ? 'border-green-400/60 bg-green-500/15' : 'border-red-400/60 bg-red-500/15'}
          `}>
            <div className="text-center">
              <div className={`text-xl font-bold ${leadingSide === 'yes' ? 'text-green-300' : 'text-red-300'}`}>
                {leadingProb}%
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500 mt-1.5 font-medium">chance</div>
        </div>
      </div>

      {/* Middle: Outcomes based on type */}
      {marketType === 'multiple' && market.outcomes ? (
        /* Multiple Choice: List of options */
        <div className="space-y-2.5 mb-5">
          {market.outcomes.slice(0, 3).map((outcome, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between bg-dark-700/60 border border-gold/10 rounded-lg px-4 py-3 hover:bg-dark-700 hover:border-gold/20 transition-all"
            >
              <span className="text-sm text-gray-200 font-medium">{outcome.name}</span>
              <div className="flex items-center space-x-3">
                <span className="text-base font-bold text-gold">{outcome.prob}%</span>
                <div className="flex space-x-1.5">
                  <span className="text-xs text-green-300 font-medium">Yes</span>
                  <span className="text-xs text-red-300 font-medium">No</span>
                </div>
              </div>
            </div>
          ))}
          {market.outcomes.length > 3 && (
            <div className="text-xs text-gray-400 text-center font-medium">
              +{market.outcomes.length - 3} more options
            </div>
          )}
        </div>
      ) : (
        /* Binary: Team names with probabilities (like Polymarket) */
        <div className="grid grid-cols-2 gap-3 mb-5">
          <button className="bg-green-500/10 hover:bg-green-500/20 border border-green-500/40 hover:border-green-500/60 rounded-lg py-4 px-4 transition-all">
            <div className="text-center">
              <div className="text-xs text-green-300/80 mb-1.5 font-semibold uppercase tracking-wide">{market.outcome_yes || 'Yes'}</div>
              <div className="text-xl font-bold text-green-300">{market.yes_prob}%</div>
            </div>
          </button>
          <button className="bg-red-500/10 hover:bg-red-500/20 border border-red-500/40 hover:border-red-500/60 rounded-lg py-4 px-4 transition-all">
            <div className="text-center">
              <div className="text-xs text-red-300/80 mb-1.5 font-semibold uppercase tracking-wide">{market.outcome_no || 'No'}</div>
              <div className="text-xl font-bold text-red-300">{market.no_prob}%</div>
            </div>
          </button>
        </div>
      )}

      {/* Bottom: Stats */}
      <div className="flex items-center justify-between text-sm pt-2 border-t border-gold/10">
        <div className="flex items-center space-x-4 text-gray-400">
          <div title="24h Volume" className="flex items-center">
            <span className="text-gold font-semibold">{formatVolume(market.volume_24h)}</span>
            <span className="ml-1.5 font-medium">Vol.</span>
          </div>
          {market.traders > 0 && (
            <div className="flex items-center space-x-1.5">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
              </svg>
              <span className="font-medium">{market.traders.toLocaleString()}</span>
            </div>
          )}
        </div>
        <div className="text-gray-400 font-medium">
          {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        </div>
      </div>
    </div>
  )
}
