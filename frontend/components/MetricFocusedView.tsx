'use client'

import { useMemo } from 'react'
import TimeSeriesChart from './TimeSeriesChart'

interface MetricFocusedViewProps {
  metric: string
  data: any[]
}

/**
 * Metric-Focused View - Deep dive into a specific financial metric
 * Shows big number, historical trend, breakdown, and context
 */
export default function MetricFocusedView({ metric, data }: MetricFocusedViewProps) {
  const company = data[0]
  const marketData = company?.MarketData || []
  const latest = marketData[0] || {}

  // Extract metric value and details based on metric type
  const metricDetails = useMemo(() => {
    const lowerMetric = metric.toLowerCase()

    if (lowerMetric.includes('p/e') || lowerMetric.includes('price to earnings')) {
      return {
        name: 'P/E Ratio',
        value: latest.trailingPE || latest.forwardPE,
        unit: 'x',
        sectorAvg: 25, // Would come from backend in production
        status: (latest.trailingPE || 0) < 20 ? 'Fair' : (latest.trailingPE || 0) < 30 ? 'Moderate' : 'High',
        icon: null,
        breakdown: [
          { label: 'Trailing P/E', value: latest.trailingPE?.toFixed(2) || 'N/A' },
          { label: 'Forward P/E', value: latest.forwardPE?.toFixed(2) || 'N/A' },
          { label: 'Price', value: `$${latest.close?.toFixed(2) || 'N/A'}` },
          { label: 'EPS (TTM)', value: `$${latest.trailingEps?.toFixed(2) || 'N/A'}` }
        ],
        historicalKey: 'trailingPE'
      }
    }

    if (lowerMetric.includes('debt') && lowerMetric.includes('equity')) {
      return {
        name: 'Debt-to-Equity Ratio',
        value: latest.debtToEquity,
        unit: 'x',
        sectorAvg: 0.65,
        status: (latest.debtToEquity || 0) < 0.5 ? 'Low Leverage' : (latest.debtToEquity || 0) < 1 ? 'Moderate' : 'High Leverage',
        icon: null,
        breakdown: [
          { label: 'Total Debt', value: latest.totalDebt ? `$${(latest.totalDebt / 1e9).toFixed(2)}B` : 'N/A' },
          { label: 'Long-term Debt', value: 'N/A' },
          { label: 'Current Debt', value: 'N/A' },
          { label: 'Total Equity', value: 'N/A' }
        ],
        historicalKey: 'debtToEquity'
      }
    }

    if (lowerMetric.includes('current ratio')) {
      return {
        name: 'Current Ratio',
        value: latest.currentRatio,
        unit: 'x',
        sectorAvg: 1.5,
        status: (latest.currentRatio || 0) > 1.5 ? 'Healthy' : (latest.currentRatio || 0) > 1 ? 'Adequate' : 'Weak',
        icon: null,
        breakdown: [
          { label: 'Current Assets', value: 'N/A' },
          { label: 'Current Liabilities', value: 'N/A' },
          { label: 'Quick Ratio', value: latest.quickRatio?.toFixed(2) || 'N/A' }
        ],
        historicalKey: 'currentRatio'
      }
    }

    if (lowerMetric.includes('free cash flow') || lowerMetric.includes('fcf')) {
      return {
        name: 'Free Cash Flow',
        value: latest.freeCashflow,
        unit: '$',
        sectorAvg: null,
        status: (latest.freeCashflow || 0) > 0 ? 'Positive' : 'Negative',
        icon: null,
        breakdown: [
          { label: 'Operating Cash Flow', value: latest.operatingCashflow ? `$${(latest.operatingCashflow / 1e9).toFixed(2)}B` : 'N/A' },
          { label: 'Free Cash Flow', value: latest.freeCashflow ? `$${(latest.freeCashflow / 1e9).toFixed(2)}B` : 'N/A' }
        ],
        historicalKey: 'freeCashflow'
      }
    }

    if (lowerMetric.includes('roe') || lowerMetric.includes('return on equity')) {
      return {
        name: 'Return on Equity (ROE)',
        value: latest.returnOnEquity,
        unit: '%',
        sectorAvg: 0.15,
        status: (latest.returnOnEquity || 0) > 0.20 ? 'Excellent' : (latest.returnOnEquity || 0) > 0.15 ? 'Good' : 'Fair',
        icon: null,
        breakdown: [
          { label: 'Net Income', value: 'N/A' },
          { label: 'Shareholder Equity', value: 'N/A' },
          { label: 'ROA', value: latest.returnOnAssets ? `${(latest.returnOnAssets * 100).toFixed(1)}%` : 'N/A' }
        ],
        historicalKey: 'returnOnEquity'
      }
    }

    // Default generic metric
    return {
      name: metric,
      value: null,
      unit: '',
      sectorAvg: null,
      status: 'N/A',
      icon: null,
      breakdown: [],
      historicalKey: null
    }
  }, [metric, latest])

  // Prepare historical data for chart
  const chartData = useMemo(() => {
    if (!metricDetails.historicalKey) return null

    const sorted = [...marketData]
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .filter(d => d[metricDetails.historicalKey] != null)

    if (sorted.length === 0) return null

    return {
      dates: sorted.map(d => d.date),
      values: sorted.map(d => {
        const val = d[metricDetails.historicalKey]
        // Convert to appropriate scale
        if (metricDetails.unit === '%') return val * 100
        if (metricDetails.unit === '$' && val > 1e9) return val / 1e9
        return val
      }),
      label: metricDetails.name,
      ticker: company.ticker
    }
  }, [marketData, metricDetails, company.ticker])

  const formatValue = (value: number | null | undefined) => {
    if (value == null) return 'N/A'
    if (metricDetails.unit === '%') return `${(value * 100).toFixed(1)}%`
    if (metricDetails.unit === '$') {
      if (value > 1e12) return `$${(value / 1e12).toFixed(2)}T`
      if (value > 1e9) return `$${(value / 1e9).toFixed(2)}B`
      if (value > 1e6) return `$${(value / 1e6).toFixed(2)}M`
      return `$${value.toFixed(2)}`
    }
    return `${value.toFixed(2)}${metricDetails.unit}`
  }

  const getStatusColor = (status: string) => {
    if (status.includes('Excellent') || status.includes('Healthy') || status.includes('Positive') || status.includes('Low'))
      return 'from-green-600 to-green-800'
    if (status.includes('Good') || status.includes('Fair') || status.includes('Adequate') || status.includes('Moderate'))
      return 'from-yellow-600 to-yellow-800'
    return 'from-red-600 to-red-800'
  }

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header */}
      <div className="border-b border-gold/20 pb-4">
        <div className="flex items-center gap-2 mb-2">
          {metricDetails.icon ? <span className="text-3xl">{metricDetails.icon}</span> : <span className="w-3 h-3 rounded-full bg-gold/70" />}
          <h2 className="text-2xl md:text-3xl font-bold text-white">{metricDetails.name}</h2>
        </div>
        <p className="text-sm text-gray-400">
          {company.company} ({company.ticker}) | {company.sector}
        </p>
      </div>

      {/* Big Number Card */}
      <div className={`bg-gradient-to-br ${getStatusColor(metricDetails.status)} rounded-xl p-6 border border-white/10 shadow-2xl`}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-xs text-white/80 uppercase tracking-wide mb-1">Current Value</div>
            <div className="text-5xl font-bold text-white">
              {formatValue(metricDetails.value)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-white/80 uppercase tracking-wide mb-1">Status</div>
            <div className="text-2xl font-bold text-white">{metricDetails.status}</div>
          </div>
        </div>

        {metricDetails.sectorAvg != null && (
          <div className="pt-4 border-t border-white/20">
            <div className="flex items-center justify-between text-sm text-white/80">
              <span>Sector Average:</span>
              <span className="font-mono font-bold">
                {metricDetails.unit === '%'
                  ? `${(metricDetails.sectorAvg * 100).toFixed(1)}%`
                  : `${metricDetails.sectorAvg.toFixed(2)}${metricDetails.unit}`}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Historical Trend */}
      {chartData && (
        <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
          <h3 className="text-xs font-bold text-gold uppercase tracking-widest mb-3">
            Historical Trend (5 Years)
          </h3>
          <div className="h-64 w-full">
            <TimeSeriesChart
              dates={chartData.dates}
              values={chartData.values}
              label={chartData.label}
              ticker={chartData.ticker}
            />
          </div>
        </div>
      )}

      {/* Breakdown */}
      {metricDetails.breakdown.length > 0 && (
        <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl">
          <h3 className="text-xs font-bold text-gold uppercase tracking-widest mb-4">
            Breakdown
          </h3>
          <div className="space-y-3">
            {metricDetails.breakdown.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between py-2 border-b border-white/5">
                <span className="text-sm text-gray-400">{item.label}</span>
                <span className="text-sm font-mono font-bold text-white">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Context */}
      <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl">
        <h3 className="text-xs font-bold text-gold uppercase tracking-widest mb-3">
          Context & Analysis
        </h3>
        <p className="text-sm text-gray-300 leading-relaxed">
          {company.company}'s {metricDetails.name.toLowerCase()} of{' '}
          <span className="font-bold text-white">{formatValue(metricDetails.value)}</span>
          {metricDetails.sectorAvg != null && (
            <>
              {' '}is{' '}
              {metricDetails.value! > metricDetails.sectorAvg ? 'above' : 'below'} the sector average of{' '}
              {metricDetails.unit === '%'
                ? `${(metricDetails.sectorAvg * 100).toFixed(1)}%`
                : `${metricDetails.sectorAvg.toFixed(2)}${metricDetails.unit}`}
            </>
          )}
          . This indicates a <span className="font-bold text-gold">{metricDetails.status.toLowerCase()}</span> position
          relative to industry peers.
        </p>
      </div>
    </div>
  )
}
