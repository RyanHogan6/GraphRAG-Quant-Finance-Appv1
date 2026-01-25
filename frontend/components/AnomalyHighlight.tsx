'use client'

import { motion } from 'framer-motion'

interface Anomaly {
  type: 'volume' | 'positioning' | 'sentiment' | 'correlation'
  severity: 'extreme' | 'high' | 'moderate'
  description: string
  metric_value: number
  percentile: number
  historical_context: string
}

interface Props {
  anomalies: Anomaly[]
  title?: string
}

export default function AnomalyHighlight({ anomalies, title }: Props) {
  const getSeverityColor = (severity: string) => {
    if (severity === 'extreme') return 'border-red-500 bg-red-500/10 text-red-400'
    if (severity === 'high') return 'border-orange-500 bg-orange-500/10 text-orange-400'
    return 'border-yellow-500 bg-yellow-500/10 text-yellow-400'
  }

  const getIcon = (type: string) => {
    if (type === 'volume') return '📊'
    if (type === 'positioning') return '⚖️'
    if (type === 'sentiment') return '🎭'
    return '🔗'
  }

  return (
    <div className="space-y-3">
      {title && (
        <div className="flex items-center gap-2 mb-4">
          <span className="text-xl">🚨</span>
          <h3 className="text-lg font-bold text-gold">{title}</h3>
        </div>
      )}

      {anomalies.map((anomaly, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: idx * 0.1 }}
          className={`border-2 rounded-xl p-4 ${getSeverityColor(anomaly.severity)}`}
        >
          <div className="flex items-start gap-3">
            <span className="text-3xl">{getIcon(anomaly.type)}</span>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-bold uppercase tracking-wider">
                  {anomaly.type} Anomaly
                </div>
                <div className="px-2 py-1 bg-white/10 rounded text-xs font-mono">
                  {anomaly.percentile}th percentile
                </div>
              </div>

              <p className="text-white font-semibold mb-2">{anomaly.description}</p>

              <div className="flex items-center gap-4 text-xs">
                <div>
                  <span className="text-gray-400">Current: </span>
                  <span className="font-bold">{anomaly.metric_value.toLocaleString()}</span>
                </div>
                <div className="text-gray-400">|</div>
                <div className="text-gray-400 italic">{anomaly.historical_context}</div>
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  )
}
