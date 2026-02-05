'use client'

import { useMemo } from 'react'
import TimeSeriesChart from './TimeSeriesChart'

interface TimeSeriesViewProps {
  data: any[]
  chartData?: {
    type?: string
    dates: string[]
    values: number[]
    label: string
    ticker?: string
  }
}

/**
 * Time Series View - Chart-first presentation for historical analysis
 * Shows large chart with statistical summary and key observations
 */
export default function TimeSeriesView({ data, chartData }: TimeSeriesViewProps) {
  // Calculate statistics
  const statistics = useMemo(() => {
    if (!chartData?.values || chartData.values.length === 0) return null

    const values = chartData.values
    const firstValue = values[0]
    const lastValue = values[values.length - 1]
    const change = lastValue - firstValue
    const changePct = (change / firstValue) * 100

    // Calculate volatility (standard deviation)
    const returns = values.slice(1).map((v, i) => (v - values[i]) / values[i])
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    const volatility = Math.sqrt(variance) * 100

    // Find peaks and troughs
    const max = Math.max(...values)
    const min = Math.min(...values)
    const maxIdx = values.indexOf(max)
    const minIdx = values.indexOf(min)

    return {
      firstValue,
      lastValue,
      change,
      changePct,
      volatility,
      max,
      min,
      maxDate: chartData.dates[maxIdx],
      minDate: chartData.dates[minIdx],
      dataPoints: values.length
    }
  }, [chartData])

  // If we have multiple series (comparison), extract them
  const isComparison = data.length > 1 && data.every(d => d.MarketData || d.futures_prices)

  const comparisonData = useMemo(() => {
    if (!isComparison) return null

    const series = data.slice(0, 3).map((item, idx) => {
      const timeData = item.MarketData || item.futures_prices || []
      const sorted = [...timeData].sort((a, b) =>
        new Date(a.date).getTime() - new Date(b.date).getTime()
      )

      const colors = ['#D4AF37', '#3B82F6', '#10B981']

      return {
        dates: sorted.map(d => d.date),
        values: sorted.map(d => d.close),
        label: item.ticker || item.commodity || `Series ${idx + 1}`,
        color: colors[idx],
        ticker: item.ticker || item.commodity
      }
    })

    // Calculate correlation if 2 series
    let correlation = null
    if (series.length === 2) {
      const s1 = series[0].values
      const s2 = series[1].values
      const minLen = Math.min(s1.length, s2.length)

      const mean1 = s1.slice(0, minLen).reduce((sum, v) => sum + v, 0) / minLen
      const mean2 = s2.slice(0, minLen).reduce((sum, v) => sum + v, 0) / minLen

      const cov = s1.slice(0, minLen).reduce((sum, v, i) =>
        sum + (v - mean1) * (s2[i] - mean2), 0
      ) / minLen

      const std1 = Math.sqrt(
        s1.slice(0, minLen).reduce((sum, v) => sum + Math.pow(v - mean1, 2), 0) / minLen
      )
      const std2 = Math.sqrt(
        s2.slice(0, minLen).reduce((sum, v) => sum + Math.pow(v - mean2, 2), 0) / minLen
      )

      correlation = cov / (std1 * std2)
    }

    return { series, correlation }
  }, [data, isComparison])

  const formatValue = (val: number) => {
    if (Math.abs(val) > 1e9) return `$${(val / 1e9).toFixed(2)}B`
    if (Math.abs(val) > 1e6) return `$${(val / 1e6).toFixed(2)}M`
    if (Math.abs(val) > 1000) return `$${(val / 1000).toFixed(2)}K`
    return `$${val.toFixed(2)}`
  }

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header */}
      <div className="border-b border-gold/20 pb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-3xl">📈</span>
          <h2 className="text-2xl md:text-3xl font-bold text-white">
            {isComparison ? 'Comparative Time Series Analysis' : 'Time Series Analysis'}
          </h2>
        </div>
        <p className="text-sm text-gray-400">
          {chartData?.label || 'Historical trend analysis'}
        </p>
      </div>

      {/* Main Chart */}
      <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
        <div className="h-96 w-full">
          {comparisonData ? (
            <TimeSeriesChart
              series={comparisonData.series}
              dates={comparisonData.series[0]?.dates || []}
              values={comparisonData.series[0]?.values || []}
              label="Comparison"
            />
          ) : chartData ? (
            <TimeSeriesChart
              dates={chartData.dates}
              values={chartData.values}
              label={chartData.label}
              ticker={chartData.ticker}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              No chart data available
            </div>
          )}
        </div>
      </div>

      {/* Statistics Grid */}
      {statistics && !isComparison && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-dark-900/40 border border-gold/10 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Change</div>
            <div className={`text-2xl font-bold ${statistics.changePct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {statistics.changePct >= 0 ? '+' : ''}{statistics.changePct.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {formatValue(statistics.change)}
            </div>
          </div>

          <div className="bg-dark-900/40 border border-gold/10 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Volatility</div>
            <div className="text-2xl font-bold text-yellow-400">
              {statistics.volatility.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Std. dev of returns
            </div>
          </div>

          <div className="bg-dark-900/40 border border-gold/10 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Peak</div>
            <div className="text-2xl font-bold text-white">
              {formatValue(statistics.max)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {new Date(statistics.maxDate).toLocaleDateString()}
            </div>
          </div>

          <div className="bg-dark-900/40 border border-gold/10 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-1">Trough</div>
            <div className="text-2xl font-bold text-white">
              {formatValue(statistics.min)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {new Date(statistics.minDate).toLocaleDateString()}
            </div>
          </div>
        </div>
      )}

      {/* Comparison Statistics */}
      {comparisonData && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {comparisonData.series.map((s, idx) => {
            const firstVal = s.values[0]
            const lastVal = s.values[s.values.length - 1]
            const change = ((lastVal - firstVal) / firstVal) * 100

            return (
              <div key={idx} className="bg-dark-900/40 border border-gold/10 rounded-lg p-4">
                <div className="text-xs text-gray-400 mb-1">{s.label}</div>
                <div className={`text-2xl font-bold ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {change >= 0 ? '+' : ''}{change.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {formatValue(firstVal)} → {formatValue(lastVal)}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Correlation (if comparing 2 series) */}
      {comparisonData?.correlation != null && (
        <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl">
          <h3 className="text-xs font-bold text-gold uppercase tracking-widest mb-3">
            Correlation Analysis
          </h3>
          <div className="flex items-center gap-4">
            <div className="text-4xl font-bold text-white">
              {comparisonData.correlation.toFixed(2)}
            </div>
            <div className="text-sm text-gray-300">
              {Math.abs(comparisonData.correlation) > 0.7 ? 'Strong' :
               Math.abs(comparisonData.correlation) > 0.4 ? 'Moderate' : 'Weak'}
              {' '}
              {comparisonData.correlation > 0 ? 'positive' : 'negative'} correlation between{' '}
              {comparisonData.series[0].label} and {comparisonData.series[1].label}
            </div>
          </div>
        </div>
      )}

      {/* Key Observations */}
      <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl">
        <h3 className="text-xs font-bold text-gold uppercase tracking-widest mb-4">
          Key Observations
        </h3>
        <div className="space-y-2">
          {statistics && !isComparison && (
            <>
              <div className="flex items-start gap-2">
                <span className={statistics.changePct >= 0 ? 'text-green-400' : 'text-red-400'}>•</span>
                <span className="text-sm text-gray-300">
                  Overall {statistics.changePct >= 0 ? 'gain' : 'loss'} of{' '}
                  <span className="font-bold">{Math.abs(statistics.changePct).toFixed(1)}%</span>{' '}
                  over the period ({statistics.dataPoints} data points)
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-400">•</span>
                <span className="text-sm text-gray-300">
                  Volatility of <span className="font-bold">{statistics.volatility.toFixed(1)}%</span>{' '}
                  indicates {statistics.volatility > 25 ? 'high' : statistics.volatility > 15 ? 'moderate' : 'low'}{' '}
                  price fluctuations
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-400">•</span>
                <span className="text-sm text-gray-300">
                  Peaked at <span className="font-bold">{formatValue(statistics.max)}</span> on{' '}
                  {new Date(statistics.maxDate).toLocaleDateString()}, representing a{' '}
                  {((statistics.max - statistics.firstValue) / statistics.firstValue * 100).toFixed(1)}% rise from start
                </span>
              </div>
            </>
          )}

          {comparisonData && (
            <>
              {comparisonData.series.map((s, idx) => {
                const change = ((s.values[s.values.length - 1] - s.values[0]) / s.values[0]) * 100
                return (
                  <div key={idx} className="flex items-start gap-2">
                    <span className={change >= 0 ? 'text-green-400' : 'text-red-400'}>•</span>
                    <span className="text-sm text-gray-300">
                      {s.label}: {change >= 0 ? '+' : ''}{change.toFixed(1)}% change
                    </span>
                  </div>
                )
              })}
              {comparisonData.correlation != null && (
                <div className="flex items-start gap-2">
                  <span className="text-purple-400">•</span>
                  <span className="text-sm text-gray-300">
                    {Math.abs(comparisonData.correlation) > 0.7 ? 'Strong' : 'Moderate'}{' '}
                    {comparisonData.correlation > 0 ? 'positive correlation' : 'inverse relationship'} detected
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button className="px-4 py-2 bg-dark-800 border border-gold/30 rounded-lg text-sm text-gold hover:bg-gold/10 transition-all">
          📥 Download Data CSV
        </button>
        <button className="px-4 py-2 bg-dark-800 border border-gold/30 rounded-lg text-sm text-gold hover:bg-gold/10 transition-all">
          📊 Share Chart
        </button>
      </div>
    </div>
  )
}
