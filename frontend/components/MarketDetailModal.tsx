import { Market } from '@/lib/mockData'
import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import ProbabilityChart from './ProbabilityChart'

interface MarketDetailModalProps {
  market: Market
  onClose: () => void
}

export default function MarketDetailModal({ market, onClose }: MarketDetailModalProps) {
  const [fullMarketData, setFullMarketData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [showTraders, setShowTraders] = useState(false)
  const [traders, setTraders] = useState<any[]>([])
  const [loadingTraders, setLoadingTraders] = useState(false)

  // Check if this is a Polymarket market (has trader data available)
  const isPolymarket = market.id && !market.id.toString().startsWith('kalshi')

  useEffect(() => {
    // Fetch full market details when modal opens
    async function fetchDetails() {
      try {
        setLoading(true)
        const details = await api.getMarketDetail(market.id)
        setFullMarketData(details)
      } catch (error) {
        console.error('Failed to fetch market details:', error)
        setFullMarketData(market) // Fallback to basic market data
      } finally {
        setLoading(false)
      }
    }
    fetchDetails()
  }, [market.id])

  // Fetch top traders when toggle is enabled
  useEffect(() => {
    if (showTraders && isPolymarket) {
      async function fetchTraders() {
        try {
          setLoadingTraders(true)
          const traderData = await api.getWhales(20)
          // Filter traders for this specific market (would need backend support)
          setTraders(traderData.slice(0, 10))
        } catch (error) {
          console.error('Failed to fetch traders:', error)
          setTraders([])
        } finally {
          setLoadingTraders(false)
        }
      }
      fetchTraders()
    }
  }, [showTraders, isPolymarket])

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(2)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(2)}k`
    return `$${volume.toFixed(2)}`
  }

  const formatPercent = (value: number) => {
    if (value === null || value === undefined) return 'N/A'
    return `${(value * 100).toFixed(2)}%`
  }

  const displayData = fullMarketData || market

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
        {loading ? (
          <div className="p-6 text-center text-gray-500">
            <div className="animate-pulse">Loading market details...</div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-4 gap-4 p-6 border-b border-gold/20">
              <div>
                <div className="text-xs text-gray-500 mb-1">Volume (24h)</div>
                <div className="text-lg font-bold text-gold">{formatVolume(displayData.volume_24h)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Liquidity</div>
                <div className="text-lg font-bold text-gold">{formatVolume(displayData.liquidity || 0)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Traders</div>
                <div className="text-lg font-bold text-gold">{(displayData.trader_count || displayData.traders || 0).toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">End Date</div>
                <div className="text-lg font-bold text-gold">
                  {new Date(displayData.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </div>
              </div>
            </div>

            {/* Additional Metrics */}
            {displayData && (displayData.volume || displayData.liquidity_24h || displayData.spread) && (
              <div className="p-6 border-b border-gold/20">
                <h3 className="text-sm font-semibold text-gold mb-4">MARKET METRICS</h3>
                <div className="grid grid-cols-3 gap-4">
                  {displayData.volume && (
                    <div className="bg-dark-700 border border-gold/10 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Total Volume</div>
                      <div className="text-sm font-semibold text-gray-200">{formatVolume(displayData.volume)}</div>
                    </div>
                  )}
                  {displayData.spread !== undefined && (
                    <div className="bg-dark-700 border border-gold/10 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Spread</div>
                      <div className="text-sm font-semibold text-gray-200">{formatPercent(displayData.spread)}</div>
                    </div>
                  )}
                  {displayData.liq_vol_ratio && (
                    <div className="bg-dark-700 border border-gold/10 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Liq/Vol Ratio</div>
                      <div className="text-sm font-semibold text-gray-200">{displayData.liq_vol_ratio.toFixed(2)}</div>
                    </div>
                  )}
                  {displayData.price_momentum && (
                    <div className="bg-dark-700 border border-gold/10 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Price Momentum</div>
                      <div className="text-sm font-semibold text-gray-200">{formatPercent(displayData.price_momentum)}</div>
                    </div>
                  )}
                  {displayData.volume_momentum && (
                    <div className="bg-dark-700 border border-gold/10 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Volume Momentum</div>
                      <div className="text-sm font-semibold text-gray-200">{formatPercent(displayData.volume_momentum)}</div>
                    </div>
                  )}
                  {displayData.turnover_ratio && (
                    <div className="bg-dark-700 border border-gold/10 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Turnover Ratio</div>
                      <div className="text-sm font-semibold text-gray-200">{displayData.turnover_ratio.toFixed(2)}</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {/* Probability Chart */}
        {!loading && (
          <div className="p-6 border-b border-gold/20">
            <ProbabilityChart
              yesProb={displayData.yes_prob}
              noProb={displayData.no_prob}
              marketData={displayData}
            />
          </div>
        )}

        {/* Trader View Toggle (Polymarket only) */}
        {!loading && isPolymarket && (
          <div className="p-6 border-b border-gold/20">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gold">TOP TRADERS</h3>
              <button
                onClick={() => setShowTraders(!showTraders)}
                className={`px-4 py-2 rounded-lg border transition-all text-sm font-semibold ${
                  showTraders
                    ? 'bg-gold/20 text-gold border-gold/40'
                    : 'bg-dark-700 text-gray-400 border-gold/20 hover:border-gold/40'
                }`}
              >
                {showTraders ? 'Hide' : 'Show'} Traders
              </button>
            </div>

            {showTraders && (
              <div>
                {loadingTraders ? (
                  <div className="text-center text-gray-500 py-8">
                    <div className="animate-pulse">Loading trader data...</div>
                  </div>
                ) : traders.length > 0 ? (
                  <div className="space-y-2">
                    {traders.map((trader, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between bg-dark-700 border border-gold/10 rounded-lg p-4 hover:border-gold/30 transition-all"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="text-gold font-semibold">#{idx + 1}</div>
                          <div>
                            <div className="text-sm text-gray-200">{trader.address?.slice(0, 6)}...{trader.address?.slice(-4)}</div>
                            <div className="text-xs text-gray-500">
                              {trader.markets_traded || 0} markets
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-semibold text-gold">
                            ${(trader.total_volume / 1000000).toFixed(2)}M
                          </div>
                          <div className="text-xs text-gray-500">volume</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    No trader data available
                  </div>
                )}
              </div>
            )}
          </div>
        )}

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
          <div className="text-xs text-gray-500 space-y-1">
            <div>Market ID: {displayData.id || displayData.market_id}</div>
            <div>Category: {displayData.category}</div>
            {displayData.condition_id && <div>Condition ID: {displayData.condition_id}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
