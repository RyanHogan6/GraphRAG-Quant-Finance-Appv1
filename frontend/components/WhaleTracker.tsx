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
      <div className="text-center py-16">
        <div className="text-xl text-gray-400">Loading whale traders...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
        <div className="text-red-400 font-semibold mb-2">Error Loading Whales</div>
        <div className="text-red-300">{error}</div>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-red-500/20 border border-red-500/40 rounded-lg text-red-300 hover:bg-red-500/30"
        >
          Retry
        </button>
      </div>
    )
  }

  if (whales.length === 0) {
    return (
      <div className="text-center py-16 bg-dark-800 border border-gold/20 rounded-lg">
        <div className="text-xl text-gray-400 mb-2">No whale traders found</div>
        <div className="text-sm text-gray-600">Check back later for whale activity</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-sm mb-1">Total Whales</div>
          <div className="text-3xl font-bold text-gold">{filteredWhales.length}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-sm mb-1">Combined Volume</div>
          <div className="text-3xl font-bold text-gold">{formatVolume(totalVolume)}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-sm mb-1">Total Profit</div>
          <div className={`text-3xl font-bold ${totalProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatProfit(totalProfit)}
          </div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-400 text-sm mb-1">Avg ROI</div>
          <div className={`text-3xl font-bold ${avgROI >= 0 ? 'text-green-400' : 'text-red-400'}`}>
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
          className="w-full bg-dark-800 border border-gold/30 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gold transition-colors"
          >
            ✕
          </button>
        )}
      </div>

      {searchQuery && (
        <div className="text-sm text-gray-400">
          Found <span className="text-gold font-semibold">{filteredWhales.length}</span> whales matching "{searchQuery}"
        </div>
      )}

      {/* Whale Table */}
      <div className="bg-dark-700 border border-gold/20 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-dark-800 border-b border-gold/20">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Rank</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Trader</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Total Volume</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Profit</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">ROI %</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Trades</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Activity</th>
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
                <td className="px-4 py-3">
                  <div className="flex items-center space-x-2">
                    <span className={`text-lg font-bold ${
                      index === 0 ? 'text-yellow-400' :
                      index === 1 ? 'text-gray-400' :
                      index === 2 ? 'text-orange-600' :
                      'text-gray-500'
                    }`}>
                      #{index + 1}
                    </span>
                    {index < 3 && (
                      <span className="text-lg">
                        {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                      </span>
                    )}
                  </div>
                </td>

                {/* Trader Address */}
                <td className="px-4 py-3">
                  <div className="font-mono text-gold font-semibold text-sm">
                    {shortenAddress(whale.address)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {whale.activity || 'Unknown'} activity
                  </div>
                </td>

                {/* Volume */}
                <td className="px-4 py-3 text-right">
                  <div className="text-sm text-white font-semibold">
                    {formatVolume(whale.volume)}
                  </div>
                </td>

                {/* Profit */}
                <td className="px-4 py-3 text-right">
                  <div className={`text-sm font-semibold ${
                    whale.profit >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatProfit(whale.profit)}
                  </div>
                </td>

                {/* Profit Ratio */}
                <td className="px-4 py-3 text-right">
                  <div className={`text-sm font-semibold ${
                    whale.profit_ratio >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(whale.profit_ratio * 100).toFixed(2)}%
                  </div>
                </td>

                {/* Trades */}
                <td className="px-4 py-3 text-right">
                  <div className="text-sm text-white font-semibold">
                    {whale.trades.toLocaleString()}
                  </div>
                </td>

                {/* Activity */}
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-1 rounded ${
                    whale.activity === 'high' || whale.activity === 'High' ? 'bg-green-500/20 text-green-400' :
                    whale.activity === 'medium' || whale.activity === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {whale.activity || 'Unknown'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>

    {/* Load More */}
    {filteredWhales.length > displayLimit && (
      <div className="text-center mt-6">
        <button
          onClick={() => setDisplayLimit(prev => prev + 20)}
          className="px-8 py-3 bg-gold/10 border border-gold/30 rounded-lg text-gold hover:bg-gold/20 hover:border-gold/50 transition-all"
        >
          Load More Whales (showing {displayLimit} of {filteredWhales.length})
        </button>
      </div>
    )}
  </div>
  )
}
