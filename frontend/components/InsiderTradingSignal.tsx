'use client'

import { motion } from 'framer-motion'

interface InsiderSignal {
  ticker: string
  company: string
  options_date: string
  filing_date?: string
  award_date?: string
  days_before: number
  signal_type: 'call_sweep' | 'put_sweep' | 'high_volume'
  volume: number
  filing_type?: string
  award_amount?: number
  confidence: 'high' | 'medium' | 'low'
}

interface Props {
  signals: InsiderSignal[]
}

export default function InsiderTradingSignal({ signals }: Props) {
  const getConfidenceColor = (conf: string) => {
    if (conf === 'high') return 'text-red-400 bg-red-500/20 border-red-500/40'
    if (conf === 'medium') return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/40'
    return 'text-gray-400 bg-gray-500/20 border-gray-500/40'
  }

  const getSignalIcon = (type: string) => {
    if (type === 'call_sweep') return '🟢'
    if (type === 'put_sweep') return '🔴'
    return '📊'
  }

  return (
    <div className="space-y-4">
      {/* Alert Banner */}
      <div className="bg-red-500/10 border-l-4 border-red-500 p-4 rounded-lg">
        <div className="flex items-center gap-3">
          <span className="text-2xl">⚠️</span>
          <div>
            <h3 className="text-red-400 font-bold text-sm">INSIDER TRADING SIGNALS DETECTED</h3>
            <p className="text-xs text-gray-300 mt-1">
              Unusual options activity {signals.length} times before material events. Pattern matches historical insider trading cases.
            </p>
          </div>
        </div>
      </div>

      {/* Signals */}
      {signals.map((sig, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: idx * 0.1 }}
          className="bg-dark-900/60 border border-red-500/30 rounded-xl p-4 hover:border-red-500/50 transition-all"
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <span className="text-3xl">{getSignalIcon(sig.signal_type)}</span>
              <div>
                <h4 className="text-lg font-bold text-white">{sig.company}</h4>
                <span className="text-xs text-gray-400 font-mono">{sig.ticker}</span>
              </div>
            </div>
            <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase border ${getConfidenceColor(sig.confidence)}`}>
              {sig.confidence} confidence
            </div>
          </div>

          {/* Timeline */}
          <div className="relative pl-8 space-y-3">
            {/* Options Activity */}
            <div className="relative">
              <div className="absolute left-[-24px] w-4 h-4 rounded-full bg-blue-500 border-2 border-dark-900" />
              <div className="text-xs text-gray-400">Options Activity</div>
              <div className="text-sm text-white font-semibold">{sig.options_date}</div>
              <div className="text-xs text-blue-400 mt-1">
                {sig.signal_type.replace('_', ' ').toUpperCase()} • Volume: {sig.volume.toLocaleString()}
              </div>
            </div>

            {/* Time Gap */}
            <div className="absolute left-[-18px] top-6 w-0.5 h-12 bg-gradient-to-b from-blue-500 to-red-500" />

            {/* Material Event */}
            <div className="relative">
              <div className="absolute left-[-24px] w-4 h-4 rounded-full bg-red-500 border-2 border-dark-900" />
              <div className="text-xs text-gray-400">Material Event</div>
              <div className="text-sm text-white font-semibold">{sig.filing_date || sig.award_date}</div>
              {sig.filing_type && (
                <div className="text-xs text-red-400 mt-1">SEC Filing: {sig.filing_type}</div>
              )}
              {sig.award_amount && (
                <div className="text-xs text-red-400 mt-1">
                  Contract Award: ${(sig.award_amount / 1e6).toFixed(1)}M
                </div>
              )}
            </div>
          </div>

          {/* Key Metric */}
          <div className="mt-4 pt-3 border-t border-white/10 flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wider">Time Advantage</div>
              <div className="text-2xl font-bold text-red-400">{sig.days_before} days</div>
            </div>
            <div className="text-xs text-gray-400 text-right max-w-[200px]">
              Someone knew about this {sig.days_before} days before the public announcement
            </div>
          </div>
        </motion.div>
      ))}

      {/* Stats Summary */}
      <div className="bg-dark-800/40 border border-white/10 rounded-xl p-4">
        <h4 className="text-sm font-bold text-gold mb-3">📊 Pattern Analysis</h4>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-white">{signals.length}</div>
            <div className="text-xs text-gray-400">Signals Found</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-white">
              {Math.round(signals.reduce((sum, s) => sum + s.days_before, 0) / signals.length)}
            </div>
            <div className="text-xs text-gray-400">Avg Days Before Event</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-400">
              {signals.filter(s => s.confidence === 'high').length}
            </div>
            <div className="text-xs text-gray-400">High Confidence</div>
          </div>
        </div>
      </div>
    </div>
  )
}
