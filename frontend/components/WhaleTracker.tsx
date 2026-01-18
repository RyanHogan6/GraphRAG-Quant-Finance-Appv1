'use client'

import { useState, useEffect } from 'react'
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
    <div className="space-y-4">
      {/* Whale Leaderboard Header - Hidden on mobile */}
      <div className="hidden md:block bg-dark-800 border border-gold/30 rounded-lg p-4">
        <div className="grid grid-cols-6 gap-4 text-sm text-gray-500 font-semibold">
          <div>Rank</div>
          <div>Trader</div>
          <div className="text-right">Total Volume</div>
          <div className="text-right">Profit</div>
          <div className="text-right">Profit %</div>
          <div className="text-right">Trades</div>
        </div>
      </div>

      {/* Whale List */}
      {whales.map((whale, index) => (
        <div
          key={whale.address}
          className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-4 hover:border-gold/40 transition-all cursor-pointer"
        >
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 md:gap-4 items-center">
            {/* Rank - Full width on mobile */}
            <div className="flex items-center space-x-2 col-span-2 md:col-span-1">
              <div className={`text-xl md:text-2xl font-bold ${
                index === 0 ? 'text-yellow-400' :
                index === 1 ? 'text-gray-400' :
                index === 2 ? 'text-orange-600' :
                'text-gray-500'
              }`}>
                #{index + 1}
              </div>
              {index < 3 && (
                <span className="text-xl md:text-2xl">
                  {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                </span>
              )}
              <div className="md:hidden flex-1">
                <div className="font-mono text-gold font-semibold text-sm">
                  {shortenAddress(whale.address)}
                </div>
                <div className="text-xs text-gray-500">
                  {whale.activity || 'Unknown'}
                </div>
              </div>
            </div>

            {/* Trader Address - Desktop only */}
            <div className="hidden md:block">
              <div className="font-mono text-gold font-semibold">
                {shortenAddress(whale.address)}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {whale.activity || 'Unknown'} activity
              </div>
            </div>

            {/* Volume */}
            <div className="text-right">
              <div className="text-white font-semibold">
                {formatVolume(whale.volume)}
              </div>
              <div className="text-xs text-gray-500 mt-1">Total volume</div>
            </div>

            {/* Profit */}
            <div className="text-right">
              <div className={`font-semibold ${
                whale.profit >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatProfit(whale.profit)}
              </div>
              <div className="text-xs text-gray-500 mt-1">Net profit</div>
            </div>

            {/* Profit Ratio */}
            <div className="text-right">
              <div className={`font-semibold ${
                whale.profit_ratio >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {(whale.profit_ratio * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">ROI</div>
            </div>

            {/* Trades */}
            <div className="text-right">
              <div className="text-white font-semibold">
                {whale.trades.toLocaleString()}
              </div>
              <div className="text-xs text-gray-500 mt-1">Total trades</div>
            </div>
          </div>

          {/* Whale Badge */}
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-2xl">🐋</span>
              <span className="text-xs text-gold font-semibold">WHALE TRADER</span>
            </div>
            <button className="text-xs text-gray-400 hover:text-gold transition-colors">
              View Details →
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
