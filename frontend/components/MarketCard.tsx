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

  return (
    <div className="market-card group relative" onClick={onClick}>
      {/* Header with Category */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-sm text-gray-100 leading-tight group-hover:text-gold transition-colors mb-2">
            {market.question}
          </h3>
        </div>
        <div className="category-badge ml-2 shrink-0">
          {market.category}
        </div>
      </div>

      {/* Outcomes or Yes/No */}
      {market.outcomes && market.outcomes.length > 0 ? (
        <div className="grid grid-cols-2 gap-2 mb-3">
          {market.outcomes.map((outcome, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between bg-dark-700 border border-gold/20 rounded px-3 py-2 hover:border-gold/40 transition-all"
            >
              <span className="text-xs text-gray-300">{outcome.name}</span>
              <span className="text-sm font-semibold text-gold">{outcome.prob}%</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="flex items-center justify-center bg-green-500/10 border border-green-500/30 rounded py-3 hover:bg-green-500/20 transition-all cursor-pointer">
            <div className="text-center">
              <div className="text-xs text-green-400 mb-1">Yes</div>
              <div className="text-lg font-bold text-green-400">{market.yes_prob}%</div>
            </div>
          </div>
          <div className="flex items-center justify-center bg-red-500/10 border border-red-500/30 rounded py-3 hover:bg-red-500/20 transition-all cursor-pointer">
            <div className="text-center">
              <div className="text-xs text-red-400 mb-1">No</div>
              <div className="text-lg font-bold text-red-400">{market.no_prob}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1" title="24h Volume">
            <span className="text-gold">Vol:</span>
            <span>{formatVolume(market.volume_24h)}</span>
          </div>
          <div className="flex items-center space-x-1" title="Liquidity">
            <span className="text-gold">Liq:</span>
            <span>{formatVolume(market.liquidity)}</span>
          </div>
          <div className="flex items-center space-x-1" title="Traders">
            <span className="text-gold">Traders:</span>
            <span>{market.traders}</span>
          </div>
        </div>
        <div className="text-gray-600">
          {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        </div>
      </div>

      {/* Hover tooltip */}
      {market.description && (
        <div className="absolute left-0 right-0 -bottom-2 opacity-0 group-hover:opacity-100 group-hover:bottom-[-60px] transition-all pointer-events-none z-10">
          <div className="bg-dark-700 border border-gold/30 rounded-lg p-3 text-xs text-gray-400 shadow-xl">
            {market.description}
          </div>
        </div>
      )}
    </div>
  )
}
