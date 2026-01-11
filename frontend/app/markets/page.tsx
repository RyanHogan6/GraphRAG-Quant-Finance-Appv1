'use client'

import { useState, useMemo, useEffect } from 'react'
import MarketCard from '@/components/MarketCard'
import MarketDetailModal from '@/components/MarketDetailModal'
import { Market } from '@/lib/mockData'
import { api } from '@/lib/api'

export default function MarketsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [sortBy, setSortBy] = useState<'volume' | 'probability' | 'traders'>('volume')
  const [markets, setMarkets] = useState<Market[]>([])
  const [categories, setCategories] = useState<Array<{category: string, count: number}>>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedMarket, setSelectedMarket] = useState<Market | null>(null)

  // Fetch markets and categories on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [marketsData, categoriesData] = await Promise.all([
          api.getMarkets({ limit: 100 }),
          api.getCategories()
        ])

        // Map backend data to frontend format
        const formattedMarkets = marketsData.map((m: any) => ({
          id: m.id || m._key,
          question: m.question,
          icon: '',
          category: m.category || 'Other',
          yes_prob: m.yes_prob,
          no_prob: m.no_prob,
          volume_24h: m.volume_24h,
          liquidity: m.liquidity || 0,
          end_date: m.end_date || '',
          traders: m.traders || 0,
        }))

        setMarkets(formattedMarkets)
        setCategories(categoriesData)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch markets:', err)
        setError('Failed to load markets. Please try again.')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  // Filter and sort markets
  const filteredMarkets = useMemo(() => {
    let filtered = markets

    // Category filter
    if (selectedCategory !== 'All') {
      filtered = filtered.filter(m => m.category === selectedCategory)
    }

    // Natural language search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(m =>
        m.question.toLowerCase().includes(query) ||
        m.category.toLowerCase().includes(query) ||
        (m.description && m.description.toLowerCase().includes(query))
      )
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      if (sortBy === 'volume') return b.volume_24h - a.volume_24h
      if (sortBy === 'probability') return b.yes_prob - a.yes_prob
      if (sortBy === 'traders') return b.traders - a.traders
      return 0
    })

    return filtered
  }, [markets, selectedCategory, searchQuery, sortBy])

  const totalVolume = filteredMarkets.reduce((sum, m) => sum + m.volume_24h, 0)
  const avgProbability = filteredMarkets.reduce((sum, m) => sum + m.yes_prob, 0) / (filteredMarkets.length || 1)

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(2)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(0)}k`
    return `$${volume}`
  }

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gold mb-2">Prediction Markets</h1>
        <p className="text-gray-500">Live market data from Polymarket & Kalshi</p>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-500 text-sm mb-1">Active Markets</div>
          <div className="text-3xl font-bold text-gold">{filteredMarkets.length}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-500 text-sm mb-1">24h Volume</div>
          <div className="text-3xl font-bold text-gold">{formatVolume(totalVolume)}</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-500 text-sm mb-1">Avg Probability</div>
          <div className="text-3xl font-bold text-gold">{avgProbability.toFixed(0)}%</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
          <div className="text-gray-500 text-sm mb-1">Categories</div>
          <div className="text-3xl font-bold text-gold">{categories.length}</div>
        </div>
      </div>

      {/* Natural Language Search */}
      <div className="mb-6">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search markets... (e.g., 'Trump', 'crypto', 'election')"
            className="search-input"
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
          <div className="mt-2 text-sm text-gray-500">
            Found <span className="text-gold font-semibold">{filteredMarkets.length}</span> markets matching "{searchQuery}"
          </div>
        )}
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-16">
          <div className="text-xl text-gray-400">Loading markets...</div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 mb-8">
          <div className="text-red-400 font-semibold mb-2">Error</div>
          <div className="text-red-300">{error}</div>
        </div>
      )}

      {/* Filters */}
      {!loading && !error && (
        <div className="flex items-center justify-between mb-8">
          {/* Category Filter */}
          <div className="flex items-center space-x-2 overflow-x-auto pb-2">
            <button
              onClick={() => setSelectedCategory('All')}
              className={`px-4 py-2 rounded-lg border transition-all whitespace-nowrap ${
                selectedCategory === 'All'
                  ? 'bg-gold/20 text-gold border-gold/40'
                  : 'bg-dark-800 text-gray-400 border-gold/20 hover:border-gold/40'
              }`}
            >
              All ({markets.length})
            </button>
            {categories.map((cat) => (
              <button
                key={cat.category}
                onClick={() => setSelectedCategory(cat.category)}
                className={`px-4 py-2 rounded-lg border transition-all whitespace-nowrap ${
                  selectedCategory === cat.category
                    ? 'bg-gold/20 text-gold border-gold/40'
                    : 'bg-dark-800 text-gray-400 border-gold/20 hover:border-gold/40'
                }`}
              >
                {cat.category} ({cat.count})
              </button>
            ))}
        </div>

          {/* Sort */}
          <div className="flex items-center space-x-2 ml-4">
            <span className="text-gray-500 text-sm whitespace-nowrap">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-dark-800 border border-gold/30 rounded-lg px-3 py-2 text-gray-200 focus:outline-none focus:border-gold/60"
            >
              <option value="volume">Volume</option>
              <option value="probability">Probability</option>
              <option value="traders">Traders</option>
            </select>
          </div>
        </div>
      )}

      {/* Markets Grid */}
      {!loading && !error && filteredMarkets.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
          {filteredMarkets.map((market) => (
            <MarketCard
              key={market.id}
              market={market}
              onClick={() => setSelectedMarket(market)}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-16 bg-dark-800 border border-gold/20 rounded-lg">
          <div className="text-xl text-gray-400 mb-2">No markets found</div>
          <div className="text-sm text-gray-600">Try adjusting your search or filters</div>
        </div>
      )}

      {/* Load More */}
      {filteredMarkets.length > 0 && (
        <div className="text-center">
          <button className="px-8 py-3 bg-gold/10 border border-gold/30 rounded-lg text-gold hover:bg-gold/20 hover:border-gold/50 transition-all">
            Load More Markets
          </button>
        </div>
      )}

      {/* Market Detail Modal */}
      {selectedMarket && (
        <MarketDetailModal
          market={selectedMarket}
          onClose={() => setSelectedMarket(null)}
        />
      )}
    </div>
  )
}
