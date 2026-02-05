'use client'

import { useMemo, useState } from 'react'
import TimeSeriesChart from './TimeSeriesChart'

interface SectorComparisonProps {
  companies: any[]
}

/**
 * Sector Comparison - Shows multiple companies from same sector
 * with normalized performance chart and key metrics table
 */
export default function SectorComparison({ companies }: SectorComparisonProps) {
  const [sortBy, setSortBy] = useState<'marketCap' | 'pe' | 'roe' | 'debt'>('marketCap')

  // Extract sector from first company
  const sector = companies[0]?.sector || 'Unknown Sector'

  // Prepare normalized chart data (all start at 100)
  const chartData = useMemo(() => {
    const series = companies
      .filter(c => c.MarketData && c.MarketData.length > 0)
      .map((company, idx) => {
        const marketData = company.MarketData
        const sorted = [...marketData].sort((a, b) =>
          new Date(a.date).getTime() - new Date(b.date).getTime()
        )

        // Get last 6 months
        const sixMonthsAgo = new Date()
        sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6)
        const filtered = sorted.filter(d => new Date(d.date) >= sixMonthsAgo)

        if (filtered.length === 0) return null

        // Normalize to 100 at start
        const baseValue = filtered[0].close
        const normalized = filtered.map(d => ({
          date: d.date,
          value: (d.close / baseValue) * 100
        }))

        const colors = ['#D4AF37', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

        return {
          dates: normalized.map(d => d.date),
          values: normalized.map(d => d.value),
          label: company.ticker,
          color: colors[idx % colors.length],
          ticker: company.ticker
        }
      })
      .filter(s => s !== null)

    return series
  }, [companies])

  // Calculate key metrics for comparison
  const companyMetrics = useMemo(() => {
    return companies.map(company => {
      const latest = company.MarketData?.[0] || {}

      // Calculate market cap
      const marketCap = latest.close && latest.volume
        ? latest.close * (latest.volume * 1000) // Rough approximation
        : null

      return {
        ticker: company.ticker,
        company: company.company,
        price: latest.close,
        pe: latest.trailingPE || latest.forwardPE,
        roe: latest.returnOnEquity,
        debt: latest.debtToEquity,
        marketCap: marketCap,
        volume: latest.volume,
        performance6M: (chartData.find(s => s?.ticker === company.ticker)?.values?.slice(-1)[0] ?? 100) - 100
      }
    }).sort((a, b) => {
      if (sortBy === 'marketCap') return (b.marketCap || 0) - (a.marketCap || 0)
      if (sortBy === 'pe') return (a.pe || 999) - (b.pe || 999)
      if (sortBy === 'roe') return (b.roe || 0) - (a.roe || 0)
      if (sortBy === 'debt') return (a.debt || 0) - (b.debt || 0)
      return 0
    })
  }, [companies, chartData, sortBy])

  // Find leaders and laggards
  const topPerformer = companyMetrics.reduce((max, c) =>
    c.performance6M > max.performance6M ? c : max
  , companyMetrics[0])

  const bestROE = companyMetrics.reduce((max, c) =>
    (c.roe || 0) > (max.roe || 0) ? c : max
  , companyMetrics[0])

  const mostLeveraged = companyMetrics.reduce((max, c) =>
    (c.debt || 0) > (max.debt || 0) ? c : max
  , companyMetrics[0])

  const formatMarketCap = (cap: number | null) => {
    if (!cap) return 'N/A'
    if (cap > 1e12) return `$${(cap / 1e12).toFixed(2)}T`
    if (cap > 1e9) return `$${(cap / 1e9).toFixed(2)}B`
    if (cap > 1e6) return `$${(cap / 1e6).toFixed(2)}M`
    return `$${cap.toLocaleString()}`
  }

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header */}
      <div className="border-b border-gold/20 pb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-3xl">🏢</span>
          <h2 className="text-2xl md:text-3xl font-bold text-white">{sector} Sector Comparison</h2>
        </div>
        <p className="text-sm text-gray-400">
          Analyzing {companies.length} companies in the {sector} sector
        </p>
      </div>

      {/* Normalized Performance Chart */}
      {chartData.length > 0 && (
        <div className="bg-dark-900/40 border border-gold/10 rounded-xl p-4 shadow-xl backdrop-blur-sm">
          <h3 className="text-xs font-bold text-gold uppercase tracking-widest mb-3">
            Relative Performance (Last 6 Months)
          </h3>
          <div className="text-xs text-gray-500 mb-3 italic">
            All series normalized to 100 at start date for comparison
          </div>
          <div className="h-80 w-full">
            <TimeSeriesChart
              series={chartData}
              dates={chartData[0]?.dates || []}
              values={chartData[0]?.values || []}
              label="Sector Comparison"
            />
          </div>
        </div>
      )}

      {/* Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 border border-green-500/30 rounded-lg p-4">
          <div className="text-xs text-green-400 mb-1">🏆 Best Performer (6M)</div>
          <div className="text-xl font-bold text-white">{topPerformer?.ticker}</div>
          <div className="text-sm text-green-300">
            {topPerformer?.performance6M > 0 ? '+' : ''}{topPerformer?.performance6M.toFixed(1)}%
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 border border-blue-500/30 rounded-lg p-4">
          <div className="text-xs text-blue-400 mb-1">📈 Highest ROE</div>
          <div className="text-xl font-bold text-white">{bestROE?.ticker}</div>
          <div className="text-sm text-blue-300">
            {((bestROE?.roe || 0) * 100).toFixed(1)}% return on equity
          </div>
        </div>

        <div className="bg-gradient-to-br from-yellow-600/20 to-yellow-800/20 border border-yellow-500/30 rounded-lg p-4">
          <div className="text-xs text-yellow-400 mb-1">⚠️ Most Leveraged</div>
          <div className="text-xl font-bold text-white">{mostLeveraged?.ticker}</div>
          <div className="text-sm text-yellow-300">
            {(mostLeveraged?.debt || 0).toFixed(2)}x debt-to-equity
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-dark-900/40 border border-gold/10 rounded-xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-gold/10">
          <h3 className="text-xs font-bold text-gold uppercase tracking-widest">
            Key Metrics Comparison
          </h3>
        </div>

        {/* Sort buttons */}
        <div className="px-4 py-2 bg-dark-800/50 border-b border-gold/10 flex gap-2 flex-wrap">
          <span className="text-xs text-gray-400">Sort by:</span>
          {[
            { key: 'marketCap', label: 'Market Cap' },
            { key: 'pe', label: 'P/E Ratio' },
            { key: 'roe', label: 'ROE' },
            { key: 'debt', label: 'Debt/Equity' }
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setSortBy(key as any)}
              className={`px-2 py-1 text-xs rounded transition-all ${
                sortBy === key
                  ? 'bg-gold text-dark-900 font-bold'
                  : 'bg-dark-800 text-gray-400 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-dark-800/50 border-b border-gold/10">
                <th className="text-left px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">Company</th>
                <th className="text-right px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">Price</th>
                <th className="text-right px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">6M %</th>
                <th className="text-right px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">P/E</th>
                <th className="text-right px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">ROE</th>
                <th className="text-right px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">Debt/Eq</th>
                <th className="text-right px-4 py-3 text-xs font-bold text-gold uppercase tracking-wide">Mkt Cap</th>
              </tr>
            </thead>
            <tbody>
              {companyMetrics.map((metric, idx) => (
                <tr
                  key={metric.ticker}
                  className="border-b border-white/5 hover:bg-gold/5 transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {idx === 0 && <span className="text-lg">🏆</span>}
                      <div>
                        <div className="text-sm font-bold text-white">{metric.ticker}</div>
                        <div className="text-xs text-gray-500 truncate max-w-xs">
                          {metric.company}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-white font-mono">
                    ${metric.price?.toFixed(2) || 'N/A'}
                  </td>
                  <td className={`px-4 py-3 text-right text-sm font-bold ${
                    metric.performance6M > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {metric.performance6M > 0 ? '+' : ''}{metric.performance6M.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-300 font-mono">
                    {metric.pe?.toFixed(1) || 'N/A'}x
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-300 font-mono">
                    {metric.roe ? `${(metric.roe * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-300 font-mono">
                    {metric.debt?.toFixed(2) || 'N/A'}x
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-300 font-mono">
                    {formatMarketCap(metric.marketCap)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer actions */}
      <div className="flex justify-center">
        <button className="px-6 py-2 bg-dark-800 border border-gold/30 rounded-lg text-sm text-gold hover:bg-gold/10 transition-all">
          Select Company for Deep Dive →
        </button>
      </div>
    </div>
  )
}
