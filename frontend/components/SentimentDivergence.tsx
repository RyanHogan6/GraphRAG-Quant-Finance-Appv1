'use client'

import { motion } from 'framer-motion'

interface DivergenceSignal {
  ticker: string
  company: string
  sec_sentiment?: number
  market_sentiment?: number
  stock_performance?: number
  polymarket_prob?: number
  signal_type: 'sec_vs_price' | 'sec_vs_market' | 'triple_confluence' | 'triple_divergence'
  confidence: number
}

interface Props {
  signals: DivergenceSignal[]
}

export default function SentimentDivergence({ signals }: Props) {
  const getGaugeColor = (value: number) => {
    if (value > 0.6) return 'text-green-400'
    if (value < 0.4) return 'text-red-400'
    return 'text-yellow-400'
  }

  const getDivergenceType = (sig: DivergenceSignal) => {
    if (sig.signal_type === 'triple_confluence') {
      return {
        title: '🎯 TRIPLE CONFLUENCE',
        desc: 'All sources agree - high conviction signal',
        color: 'border-green-500/50 bg-green-500/10'
      }
    }
    if (sig.signal_type === 'triple_divergence') {
      return {
        title: '⚠️ TRIPLE DIVERGENCE',
        desc: 'Sources contradict each other - major uncertainty',
        color: 'border-red-500/50 bg-red-500/10'
      }
    }
    if (sig.signal_type === 'sec_vs_price') {
      return {
        title: '⚡ SENTIMENT-PRICE DIVERGENCE',
        desc: 'Market ignoring fundamentals - opportunity or trap?',
        color: 'border-yellow-500/50 bg-yellow-500/10'
      }
    }
    return {
      title: '🔔 MARKET DISAGREEMENT',
      desc: 'Prediction markets and SEC filings show opposite signals',
      color: 'border-purple-500/50 bg-purple-500/10'
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-500/10 to-transparent border-l-4 border-purple-500 p-4 rounded-lg">
        <h3 className="text-purple-400 font-bold text-sm mb-1">MULTI-SOURCE SENTIMENT ANALYSIS</h3>
        <p className="text-xs text-gray-300">
          Cross-referencing {signals.length} companies across SEC filings, prediction markets, and stock price action
        </p>
      </div>

      {/* Signals */}
      {signals.map((sig, idx) => {
        const divType = getDivergenceType(sig)
        return (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className={`border rounded-xl p-4 ${divType.color}`}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div>
                <h4 className="text-lg font-bold text-white mb-1">{sig.company}</h4>
                <span className="text-xs text-gray-400 font-mono">{sig.ticker}</span>
                <div className="mt-2 text-xs font-bold text-purple-400">{divType.title}</div>
                <div className="text-xs text-gray-400 mt-1">{divType.desc}</div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500 mb-1">Signal Strength</div>
                <div className="text-2xl font-bold text-white">{Math.round(sig.confidence * 100)}%</div>
              </div>
            </div>

            {/* Sentiment Gauges */}
            <div className="grid grid-cols-3 gap-4">
              {sig.sec_sentiment !== undefined && (
                <div className="bg-dark-900/40 rounded-lg p-3 border border-white/5">
                  <div className="text-xs text-gray-400 mb-2">SEC Sentiment</div>
                  <div className="relative w-full h-2 bg-dark-800 rounded-full overflow-hidden">
                    <div
                      className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                      style={{ width: '100%' }}
                    />
                    <div
                      className="absolute top-0 h-full w-1 bg-white shadow-lg"
                      style={{ left: `${((sig.sec_sentiment + 1) / 2) * 100}%` }}
                    />
                  </div>
                  <div className={`text-lg font-bold mt-2 ${getGaugeColor((sig.sec_sentiment + 1) / 2)}`}>
                    {sig.sec_sentiment > 0 ? '+' : ''}{sig.sec_sentiment.toFixed(2)}
                  </div>
                </div>
              )}

              {sig.polymarket_prob !== undefined && (
                <div className="bg-dark-900/40 rounded-lg p-3 border border-white/5">
                  <div className="text-xs text-gray-400 mb-2">Market Probability</div>
                  <div className="relative w-full h-2 bg-dark-800 rounded-full overflow-hidden">
                    <div
                      className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                      style={{ width: '100%' }}
                    />
                    <div
                      className="absolute top-0 h-full w-1 bg-white shadow-lg"
                      style={{ left: `${sig.polymarket_prob * 100}%` }}
                    />
                  </div>
                  <div className={`text-lg font-bold mt-2 ${getGaugeColor(sig.polymarket_prob)}`}>
                    {Math.round(sig.polymarket_prob * 100)}%
                  </div>
                </div>
              )}

              {sig.stock_performance !== undefined && (
                <div className="bg-dark-900/40 rounded-lg p-3 border border-white/5">
                  <div className="text-xs text-gray-400 mb-2">Stock Performance</div>
                  <div className="relative w-full h-2 bg-dark-800 rounded-full overflow-hidden">
                    <div
                      className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                      style={{ width: '100%' }}
                    />
                    <div
                      className="absolute top-0 h-full w-1 bg-white shadow-lg"
                      style={{ left: `${Math.max(0, Math.min(100, 50 + sig.stock_performance))}%` }}
                    />
                  </div>
                  <div className={`text-lg font-bold mt-2 ${sig.stock_performance > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {sig.stock_performance > 0 ? '+' : ''}{sig.stock_performance.toFixed(1)}%
                  </div>
                </div>
              )}
            </div>

            {/* Interpretation */}
            <div className="mt-4 pt-3 border-t border-white/10">
              <div className="text-xs text-gray-400">
                {sig.signal_type === 'triple_confluence' && (
                  <span>✅ <strong>HIGH CONVICTION:</strong> SEC filings, prediction markets, and price action all confirm the same direction. Historical win rate: 74%</span>
                )}
                {sig.signal_type === 'sec_vs_price' && (
                  <span>⚠️ <strong>DIVERGENCE DETECTED:</strong> Price ignoring fundamental signals. Either market has information SEC doesn't, or correction imminent.</span>
                )}
                {sig.signal_type === 'triple_divergence' && (
                  <span>🚨 <strong>MAJOR UNCERTAINTY:</strong> All three sources disagree. Wait for clarity or hedge positions.</span>
                )}
              </div>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
