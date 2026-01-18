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
            </tr>
          </thead>
          <tbody className="divide-y divide-gold/10">
            {whales.map((whale, index) => (
              <tr
                key={whale.address}
                className="hover:bg-dark-800/50 transition-colors"
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
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
