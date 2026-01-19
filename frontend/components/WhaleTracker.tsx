'use client'

import { useState, useEffect, useMemo } from 'react'
import { api } from '@/lib/api'

interface Whale {
  address: string
  volume: number
  profit: number
  trades: number
  activity: string
  profit_ratio: number
  win_rate?: number
}

export default function WhaleTracker() {
  const [whales, setWhales] = useState<Whale[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [displayLimit, setDisplayLimit] = useState(20)

  useEffect(() => {
    const fetchWhales = async () => {
      try {
        setLoading(true)
        const data = await api.getWhales(50) // Top 50 whales
        setWhales(data)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch whales:', err)
        setError('Failed to load whale traders')
      } finally {
        setLoading(false)
      }
    }

    fetchWhales()
  }, [])

  const formatVolume = (volume: number) => {
    if (volume == null || isNaN(volume) || volume === 0) return '$0'
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(2)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(0)}k`
    return `$${volume.toFixed(0)}`
  }

  const formatProfit = (profit: number) => {
    if (profit == null || isNaN(profit)) return '$0'
    const prefix = profit >= 0 ? '+' : ''
    if (Math.abs(profit) >= 1000000) return `${prefix}$${(profit / 1000000).toFixed(2)}M`
    if (Math.abs(profit) >= 1000) return `${prefix}$${(profit / 1000).toFixed(0)}k`
    return `${prefix}$${profit.toFixed(0)}`
  }

  const shortenAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`
  }

  // Filter whales based on search
  const filteredWhales = useMemo(() => {
    if (!searchQuery) return whales
    const query = searchQuery.toLowerCase()
    return whales.filter(whale =>
      whale.address.toLowerCase().includes(query) ||
      whale.activity?.toLowerCase().includes(query)
    )
  }, [whales, searchQuery])

  // Calculate metrics
  const totalVolume = useMemo(() =>
    filteredWhales.reduce((sum, w) => sum + (w.volume || 0), 0)
  , [filteredWhales])

  const totalProfit = useMemo(() =>
    filteredWhales.reduce((sum, w) => sum + (w.profit || 0), 0)
  , [filteredWhales])

  const avgROI = useMemo(() => {
    if (filteredWhales.length === 0) return 0
    return (filteredWhales.reduce((sum, w) => sum + (w.profit_ratio || 0), 0) / filteredWhales.length) * 100
  }, [filteredWhales])

  const totalTrades = useMemo(() =>
    filteredWhales.reduce((sum, w) => sum + (w.trades || 0), 0)
  , [filteredWhales])

  if (loading) {
    return (
      <div className="text-center py-8 md:py-12">
        <div className="text-base md:text-lg text-gray-400">Loading whale traders...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 md:p-6">
        <div className="text-red-400 font-semibold text-sm md:text-base mb-2">Error Loading Whales</div>
        <div className="text-red-300 text-xs md:text-sm">{error}</div>
        <button
          onClick={() => window.location.reload()}
          className="mt-3 px-3 py-1.5 md:px-4 md:py-2 bg-red-500/20 border border-red-500/40 rounded-lg text-red-300 text-xs md:text-sm hover:bg-red-500/30"
        >
          Retry
        </button>
      </div>
    )
  }

  if (whales.length === 0) {
    return (
      <div className="text-center py-8 md:py-12 bg-dark-800 border border-gold/20 rounded-lg">
        <div className="text-base md:text-lg text-gray-400 mb-2">No whale traders found</div>
        <div className="text-xs md:text-sm text-gray-600">Check back later for whale activity</div>
      </div>
    )
  }

  return (
    <div className="space-y-4 md:space-y-5">
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 md:gap-3">
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-xs md:text-sm mb-1">Total Whales</div>
          <div className="text-xl md:text-3xl font-bold text-gold">{filteredWhales.length}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-xs md:text-sm mb-1">Combined Volume</div>
          <div className="text-xl md:text-3xl font-bold text-gold">{formatVolume(totalVolume)}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-xs md:text-sm mb-1">Total Profit</div>
          <div className={`text-xl md:text-3xl font-bold ${totalProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatProfit(totalProfit)}
          </div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-xs md:text-sm mb-1">Avg ROI</div>
          <div className={`text-xl md:text-3xl font-bold ${avgROI >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {avgROI.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by address or activity..."
          className="w-full bg-dark-800 border border-gold/30 rounded-lg px-3 py-2 md:px-4 md:py-2.5 text-sm md:text-base text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-3 md:right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gold transition-colors text-sm"
          >
            ✕
          </button>
        )}
      </div>

      {searchQuery && (
        <div className="text-xs md:text-sm text-gray-400">
          Found <span className="text-gold font-semibold">{filteredWhales.length}</span> whales matching "{searchQuery}"
        </div>
      )}

      {/* Card View - 2 cols mobile, 4 cols desktop */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 md:gap-3 md:hidden mb-4 md:mb-6">
        {filteredWhales.slice(0, displayLimit).map((whale, index) => (
          <div
            key={whale.address}
            className="bg-gradient-to-br from-amber-900/10 to-yellow-900/5 border border-gold/30 rounded-lg p-3 hover:border-gold/50 hover:from-amber-900/15 hover:to-yellow-900/10 transition-all"
          >
            {/* Rank and Address */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-1.5">
                <span className={`text-base font-bold ${
                  index === 0 ? 'text-yellow-400' :
                  index === 1 ? 'text-amber-300' :
                  index === 2 ? 'text-orange-400' :
                  'text-amber-200/70'
                }`}>
                  #{index + 1}
                </span>
                {index < 3 && (
                  <span className="text-sm">
                    {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                  </span>
                )}
              </div>
            </div>

            {/* Address */}
            <div className="font-mono text-gold font-semibold text-xs mb-3">
              {shortenAddress(whale.address)}
            </div>

            {/* Volume and Profit */}
            <div className="grid grid-cols-2 gap-1.5 mb-2">
              <div className="bg-amber-900/20 border border-gold/20 rounded p-2">
                <div className="text-xs text-amber-300/60 mb-0.5">Volume</div>
                <div className="text-xs font-bold text-amber-100">{formatVolume(whale.volume)}</div>
              </div>
              <div className="bg-amber-900/20 border border-gold/20 rounded p-2">
                <div className="text-xs text-amber-300/60 mb-0.5">Profit</div>
                <div className={`text-xs font-bold ${whale.profit >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                  {formatProfit(whale.profit)}
                </div>
              </div>
            </div>

            {/* ROI and Trades */}
            <div className="flex flex-col gap-1 text-xs pt-2 border-t border-gold/20">
              <div className="flex items-center justify-between">
                <span className="text-amber-300/60">ROI:</span>
                <span className={`font-semibold ${whale.profit_ratio >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                  {(whale.profit_ratio * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-amber-300/60">Trades:</span>
                <span className="text-amber-200/80">{whale.trades.toLocaleString()}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Desktop Table View */}
      <div className="hidden md:block bg-dark-700 border border-gold/20 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full align-middle">
            <div className="overflow-hidden">
              <table className="min-w-full divide-y divide-gold/10">
                <thead className="bg-dark-800 border-b border-gold/20">
                  <tr>
                    <th className="px-2 md:px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Rank</th>
                    <th className="px-2 md:px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Trader</th>
                    <th className="px-2 md:px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Volume</th>
                    <th className="hidden md:table-cell px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Profit</th>
                    <th className="hidden lg:table-cell px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">ROI %</th>
                    <th className="hidden lg:table-cell px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Trades</th>
                    <th className="hidden md:table-cell px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">Activity</th>
                  </tr>
                </thead>
            <tbody className="divide-y divide-gold/10">
              {filteredWhales.slice(0, displayLimit).map((whale, index) => (
              <tr
                key={whale.address}
                className={`hover:bg-dark-800/70 transition-colors ${
                  index % 2 === 0 ? 'bg-dark-800/20' : 'bg-gold/5'
                }`}
              >
                {/* Rank */}
                <td className="px-2 md:px-4 py-3">
                  <div className="flex items-center space-x-1 md:space-x-2">
                    <span className={`text-sm md:text-lg font-bold ${
                      index === 0 ? 'text-yellow-400' :
                      index === 1 ? 'text-gray-400' :
                      index === 2 ? 'text-orange-600' :
                      'text-gray-500'
                    }`}>
                      #{index + 1}
                    </span>
                    {index < 3 && (
                      <span className="text-base md:text-lg">
                        {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                      </span>
                    )}
                  </div>
                </td>

                {/* Trader Address */}
                <td className="px-2 md:px-4 py-3">
                  <div className="font-mono text-gold font-semibold text-xs md:text-sm">
                    {shortenAddress(whale.address)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1 md:hidden">
                    {formatVolume(whale.volume)}
                  </div>
                </td>

                {/* Volume */}
                <td className="hidden md:table-cell px-2 md:px-4 py-3 text-right">
                  <div className="text-xs md:text-sm text-white font-semibold whitespace-nowrap">
                    {formatVolume(whale.volume)}
                  </div>
                </td>

                {/* Profit */}
                <td className="hidden md:table-cell px-4 py-3 text-right">
                  <div className={`text-xs md:text-sm font-semibold whitespace-nowrap ${
                    whale.profit >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatProfit(whale.profit)}
                  </div>
                </td>

                {/* Profit Ratio */}
                <td className="hidden lg:table-cell px-4 py-3 text-right">
                  <div className={`text-xs md:text-sm font-semibold whitespace-nowrap ${
                    whale.profit_ratio >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(whale.profit_ratio * 100).toFixed(2)}%
                  </div>
                </td>

                {/* Trades */}
                <td className="hidden lg:table-cell px-4 py-3 text-right">
                  <div className="text-xs md:text-sm text-white font-semibold whitespace-nowrap">
                    {whale.trades.toLocaleString()}
                  </div>
                </td>

                {/* Activity */}
                <td className="hidden md:table-cell px-4 py-3">
                  <span className={`text-xs px-2 py-1 rounded whitespace-nowrap ${
                    whale.activity === 'high' || whale.activity === 'High' ? 'bg-green-500/20 text-green-400' :
                    whale.activity === 'medium' || whale.activity === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {whale.activity && whale.activity !== 'nan' ? whale.activity : 'Unknown'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
            </div>
          </div>
        </div>
      </div>

    {/* Load More */}
    {filteredWhales.length > displayLimit && (
      <div className="text-center mt-4 md:mt-5">
        <button
          onClick={() => setDisplayLimit(prev => prev + 20)}
          className="px-6 py-2 md:px-8 md:py-3 bg-gold/10 border border-gold/30 rounded-lg text-sm md:text-base text-gold hover:bg-gold/20 hover:border-gold/50 transition-all"
        >
          Load More Whales (showing {displayLimit} of {filteredWhales.length})
        </button>
      </div>
    )}
  </div>
  )
}
