import { Market } from '@/lib/mockData'

interface MarketDetailModalProps {
  market: Market
  onClose: () => void
}

export default function MarketDetailModal({ market, onClose }: MarketDetailModalProps) {
  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(2)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(0)}k`
    return `$${volume}`
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-dark-800 border border-gold/30 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="border-b border-gold/20 p-6">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="category-badge mb-3">
                {market.category}
              </div>
              <h2 className="text-2xl font-bold text-gray-100 mb-2">
                {market.question}
              </h2>
              {market.description && (
                <p className="text-sm text-gray-400">{market.description}</p>
              )}
            </div>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gold transition-colors ml-4 text-2xl"
            >
              ×
            </button>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-4 gap-4 p-6 border-b border-gold/20">
          <div>
            <div className="text-xs text-gray-500 mb-1">Volume (24h)</div>
            <div className="text-lg font-bold text-gold">{formatVolume(market.volume_24h)}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500 mb-1">Liquidity</div>
            <div className="text-lg font-bold text-gold">{formatVolume(market.liquidity)}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500 mb-1">Traders</div>
            <div className="text-lg font-bold text-gold">{market.traders.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500 mb-1">End Date</div>
            <div className="text-lg font-bold text-gold">
              {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
            </div>
          </div>
        </div>

        {/* Outcomes */}
        <div className="p-6">
          <h3 className="text-sm font-semibold text-gold mb-4">OUTCOMES</h3>

          {market.outcomes && market.outcomes.length > 0 ? (
            <div className="space-y-2">
              {market.outcomes.map((outcome, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-dark-700 border border-gold/20 rounded-lg p-4 hover:border-gold/40 transition-all"
                >
                  <span className="text-gray-200">{outcome.name}</span>
                  <div className="flex items-center space-x-4">
                    <span className="text-2xl font-bold text-gold">{outcome.prob}%</span>
                    <button className="px-4 py-2 bg-gold/10 border border-gold/30 rounded text-gold hover:bg-gold/20 transition-all text-sm">
                      Trade
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-6 hover:bg-green-500/20 transition-all cursor-pointer">
                <div className="text-center">
                  <div className="text-xs text-green-400 mb-2">YES</div>
                  <div className="text-4xl font-bold text-green-400 mb-4">{market.yes_prob}%</div>
                  <button className="w-full px-4 py-2 bg-green-500/20 border border-green-500/40 rounded text-green-400 hover:bg-green-500/30 transition-all text-sm">
                    Buy Yes
                  </button>
                </div>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 hover:bg-red-500/20 transition-all cursor-pointer">
                <div className="text-center">
                  <div className="text-xs text-red-400 mb-2">NO</div>
                  <div className="text-4xl font-bold text-red-400 mb-4">{market.no_prob}%</div>
                  <button className="w-full px-4 py-2 bg-red-500/20 border border-red-500/40 rounded text-red-400 hover:bg-red-500/30 transition-all text-sm">
                    Buy No
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="border-t border-gold/20 p-6 bg-dark-700/50">
          <div className="text-xs text-gray-500">
            Market ID: {market.id} • Category: {market.category}
          </div>
        </div>
      </div>
    </div>
  )
}
