'use client'

interface SentimentIndicator {
  label: string
  score: number  // -100 to +100
  status: 'bearish' | 'neutral' | 'bullish'
  description: string
  factors: { name: string, value: string, impact: 'bullish' | 'bearish' | 'neutral' }[]
}

interface SentimentIndicatorsProps {
  marketData: any[]
  secFilings: any[]
  optionsFlow: any[]
  xbrlData: any[]
}

/**
 * Three quick sentiment indicators at the top of company analysis
 * Shows Technical Momentum, Fundamental Health, and Sentiment Signal
 */
export default function SentimentIndicators({
  marketData,
  secFilings,
  optionsFlow,
  xbrlData
}: SentimentIndicatorsProps) {
  const latest = marketData[0] || {}
  const latestOptions = optionsFlow[0] || {}
  const latestXbrl = xbrlData[0] || {}

  // Calculate Technical Momentum (-100 to +100)
  const technicalMomentum = (): SentimentIndicator => {
    let score = 0
    const factors: SentimentIndicator['factors'] = []

    // RSI score (-50 to +50)
    if (latest.rsi != null) {
      const rsiScore = latest.rsi < 30 ? -50 : latest.rsi > 70 ? 50 : 0
      score += rsiScore
      factors.push({
        name: 'RSI',
        value: latest.rsi.toFixed(1),
        impact: latest.rsi < 30 ? 'bearish' : latest.rsi > 70 ? 'bullish' : 'neutral'
      })
    }

    // MACD score (-25 to +25)
    if (latest.macd != null) {
      const macdScore = latest.macd > 0 ? 25 : -25
      score += macdScore
      factors.push({
        name: 'MACD',
        value: latest.macd.toFixed(2),
        impact: latest.macd > 0 ? 'bullish' : 'bearish'
      })
    }

    // Price vs SMA20 (-25 to +25)
    if (latest.close != null && latest.sma_20 != null) {
      const pctFromSMA = ((latest.close / latest.sma_20 - 1) * 100)
      const smaScore = pctFromSMA > 5 ? 25 : pctFromSMA < -5 ? -25 : 0
      score += smaScore
      factors.push({
        name: 'vs SMA20',
        value: `${pctFromSMA.toFixed(1)}%`,
        impact: pctFromSMA > 5 ? 'bullish' : pctFromSMA < -5 ? 'bearish' : 'neutral'
      })
    }

    const status = score < -30 ? 'bearish' : score > 30 ? 'bullish' : 'neutral'
    const label = score < -30 ? 'Oversold' : score > 30 ? 'Overbought' : 'Neutral'

    return {
      label,
      score,
      status,
      description: `Technical indicators suggest ${label.toLowerCase()} conditions with score of ${score.toFixed(0)}`,
      factors
    }
  }

  // Calculate Fundamental Health (-100 to +100)
  const fundamentalHealth = (): SentimentIndicator => {
    let score = 0
    const factors: SentimentIndicator['factors'] = []

    // Debt-to-Equity (-30 to +30)
    if (latest.debtToEquity != null) {
      const deScore = latest.debtToEquity < 0.5 ? 30 : latest.debtToEquity > 1.5 ? -30 : 0
      score += deScore
      factors.push({
        name: 'D/E Ratio',
        value: latest.debtToEquity.toFixed(2) + 'x',
        impact: latest.debtToEquity < 0.5 ? 'bullish' : latest.debtToEquity > 1.5 ? 'bearish' : 'neutral'
      })
    }

    // Free Cash Flow (-30 to +30)
    if (latest.freeCashflow != null) {
      const fcfScore = latest.freeCashflow > 1e9 ? 30 : latest.freeCashflow < 0 ? -30 : 0
      score += fcfScore
      factors.push({
        name: 'FCF',
        value: latest.freeCashflow >= 1e9 ? `$${(latest.freeCashflow / 1e9).toFixed(2)}B` : `$${Number(latest.freeCashflow).toLocaleString('en-US', { maximumFractionDigits: 0 })}`,
        impact: latest.freeCashflow > 1e9 ? 'bullish' : latest.freeCashflow < 0 ? 'bearish' : 'neutral'
      })
    }

    // Profit Margin (-20 to +20)
    if (latest.profitMargins != null) {
      const marginScore = latest.profitMargins > 0.15 ? 20 : latest.profitMargins < 0.05 ? -20 : 0
      score += marginScore
      factors.push({
        name: 'Margin',
        value: `${(latest.profitMargins * 100).toFixed(1)}%`,
        impact: latest.profitMargins > 0.15 ? 'bullish' : latest.profitMargins < 0.05 ? 'bearish' : 'neutral'
      })
    }

    // ROE (-20 to +20)
    if (latest.returnOnEquity != null) {
      const roeScore = latest.returnOnEquity > 0.20 ? 20 : latest.returnOnEquity < 0.10 ? -20 : 0
      score += roeScore
      factors.push({
        name: 'ROE',
        value: `${(latest.returnOnEquity * 100).toFixed(1)}%`,
        impact: latest.returnOnEquity > 0.20 ? 'bullish' : latest.returnOnEquity < 0.10 ? 'bearish' : 'neutral'
      })
    }

    const status = score < -30 ? 'bearish' : score > 30 ? 'bullish' : 'neutral'
    const label = score < -30 ? 'Distressed' : score > 30 ? 'Healthy' : 'Stable'

    return {
      label,
      score,
      status,
      description: `Fundamentals indicate ${label.toLowerCase()} financial position with score of ${score.toFixed(0)}`,
      factors
    }
  }

  // Calculate Sentiment Signal (-100 to +100)
  const sentimentSignal = (): SentimentIndicator => {
    let score = 0
    const factors: SentimentIndicator['factors'] = []

    // SEC FinBERT sentiment (-50 to +50)
    if (secFilings.length > 0) {
      const avgFinbert = secFilings.reduce((sum, f) => sum + (f.avg_finbert || 0), 0) / secFilings.length
      const finbertScore = avgFinbert * 250  // Scale to -50 to +50 range
      score += finbertScore
      factors.push({
        name: 'SEC Sentiment',
        value: avgFinbert > 0 ? `+${avgFinbert.toFixed(2)}` : avgFinbert.toFixed(2),
        impact: avgFinbert > 0.1 ? 'bullish' : avgFinbert < -0.1 ? 'bearish' : 'neutral'
      })
    }

    // Options Put/Call Ratio (-30 to +30)
    if (latestOptions.put_call_volume_ratio != null) {
      const pcRatio = latestOptions.put_call_volume_ratio
      const pcScore = pcRatio < 0.7 ? 30 : pcRatio > 1.3 ? -30 : 0
      score += pcScore
      factors.push({
        name: 'P/C Ratio',
        value: pcRatio.toFixed(2),
        impact: pcRatio < 0.7 ? 'bullish' : pcRatio > 1.3 ? 'bearish' : 'neutral'
      })
    }

    // Unusual Options Activity (-20 to +20)
    if (latestOptions.unusual_call_activity != null && latestOptions.unusual_put_activity != null) {
      const activityScore = latestOptions.unusual_call_activity ? 20 :
                           latestOptions.unusual_put_activity ? -20 : 0
      score += activityScore
      if (latestOptions.unusual_call_activity || latestOptions.unusual_put_activity) {
        factors.push({
          name: 'Options Flow',
          value: latestOptions.unusual_call_activity ? 'Unusual Calls' : 'Unusual Puts',
          impact: latestOptions.unusual_call_activity ? 'bullish' : 'bearish'
        })
      } else {
        factors.push({
          name: 'Options Flow',
          value: 'Normal',
          impact: 'neutral'
        })
      }
    }

    const status = score < -30 ? 'bearish' : score > 30 ? 'bullish' : 'neutral'
    const label = score < -30 ? 'Bearish' : score > 30 ? 'Bullish' : 'Mixed'

    return {
      label,
      score,
      status,
      description: `Sentiment signals are ${label.toLowerCase()} with score of ${score.toFixed(0)}`,
      factors
    }
  }

  const technical = technicalMomentum()
  const fundamental = fundamentalHealth()
  const sentiment = sentimentSignal()

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'bullish': return 'from-green-600 to-green-800'
      case 'bearish': return 'from-red-600 to-red-800'
      default: return 'from-yellow-600 to-yellow-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'bullish': return '🟢'
      case 'bearish': return '🔴'
      default: return '🟡'
    }
  }

  const getScoreBarColor = (score: number) => {
    if (score < -30) return 'bg-red-500'
    if (score > 30) return 'bg-green-500'
    return 'bg-yellow-500'
  }

  const indicators = [
    { title: 'Technical Momentum', data: technical },
    { title: 'Fundamental Health', data: fundamental },
    { title: 'Sentiment Signal', data: sentiment }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
      {indicators.map(({ title, data }) => (
        <div
          key={title}
          className={`bg-gradient-to-br ${getStatusColor(data.status)} rounded-xl p-4 border border-white/10 shadow-xl`}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{getStatusIcon(data.status)}</span>
              <span className="text-xs font-bold text-white uppercase tracking-wide">
                {title}
              </span>
            </div>
            <div className="text-xl font-bold text-white">
              {data.label}
            </div>
          </div>

          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-white/80">Score</span>
              <span className="text-sm font-mono font-bold text-white">
                {data.score > 0 ? '+' : ''}{data.score.toFixed(0)}
              </span>
            </div>
            <div className="w-full h-2 bg-black/30 rounded-full overflow-hidden">
              <div
                className={`h-full ${getScoreBarColor(data.score)} transition-all duration-500`}
                style={{ width: `${Math.abs(data.score)}%` }}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            {data.factors.map((factor, idx) => (
              <div key={idx} className="flex items-center justify-between text-xs">
                <span className="text-white/70">{factor.name}:</span>
                <div className="flex items-center gap-1.5">
                  <span className="font-mono font-semibold text-white">{factor.value}</span>
                  <span className="text-[10px]">
                    {factor.impact === 'bullish' ? '↑' : factor.impact === 'bearish' ? '↓' : '→'}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-3 pt-3 border-t border-white/20 text-[10px] text-white/60 italic">
            {data.description}
          </div>
        </div>
      ))}
    </div>
  )
}
