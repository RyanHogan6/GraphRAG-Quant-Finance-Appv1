'use client'

import { useState, useMemo, useEffect } from 'react'
import MarketCard from '@/components/MarketCard'
import MarketDetailModal from '@/components/MarketDetailModal'
import WhaleTracker from '@/components/WhaleTracker'
import { Market } from '@/lib/mockData'
import { api } from '@/lib/api'

export default function MarketsPage() {
  const [platform, setPlatform] = useState<'polymarket' | 'kalshi'>('polymarket')
  const [view, setView] = useState<'markets' | 'traders'>('markets')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [sortBy, setSortBy] = useState<'volume' | 'probability' | 'traders'>('volume')
  const [markets, setMarkets] = useState<Market[]>([])
  const [categories, setCategories] = useState<Array<{category: string, count: number}>>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedMarket, setSelectedMarket] = useState<Market | null>(null)
  const [displayLimit, setDisplayLimit] = useState(100)
  const [hasMore, setHasMore] = useState(true)

  // Fetch markets and categories when platform changes
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        // Use FEATURED endpoint for diverse, curated markets
        const [marketsData, categoriesData] = await Promise.all([
          api.getFeaturedMarkets(100, platform),
          api.getCategories(platform)
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
          outcome_yes: m.outcome_yes,
          outcome_no: m.outcome_no,
          outcomes: m.outcomes,
        }))

        setMarkets(formattedMarkets)
        setCategories(categoriesData)
        setHasMore(marketsData.length >= 100)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch markets:', err)
        setError('Failed to load markets. Please try refreshing the page.')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [platform])

  // Load more markets - continues using featured endpoint for diversity
  const handleLoadMore = async () => {
    try {
      setLoadingMore(true)
      const newLimit = displayLimit + 100
      // Continue using featured endpoint to maintain category diversity
      const marketsData = await api.getFeaturedMarkets(newLimit)

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
        outcome_yes: m.outcome_yes,
        outcome_no: m.outcome_no,
        outcomes: m.outcomes,
      }))

      setMarkets(formattedMarkets)
      setDisplayLimit(newLimit)
      // If we got fewer markets than requested, we've reached the end
      setHasMore(marketsData.length >= newLimit)
    } catch (err) {
      console.error('Failed to load more markets:', err)
    } finally {
      setLoadingMore(false)
    }
  }

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

      {/* Platform and View Toggles */}
      <div className="flex items-center justify-between mb-8">
        {/* Platform Toggle */}
        <div className="flex items-center space-x-2 bg-dark-800 border border-gold/30 rounded-lg p-1">
          <button
            onClick={() => {
              setPlatform('polymarket')
              setView('markets') // Reset to markets when switching platform
            }}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              platform === 'polymarket'
                ? 'bg-gold text-dark-900'
                : 'text-gray-400 hover:text-gold'
            }`}
          >
            Polymarket
          </button>
          <button
            onClick={() => {
              setPlatform('kalshi')
              setView('markets') // Reset to markets when switching platform
            }}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              platform === 'kalshi'
                ? 'bg-gold text-dark-900'
                : 'text-gray-400 hover:text-gold'
            }`}
          >
            Kalshi
          </button>
        </div>

        {/* View Toggle - Only show for Polymarket */}
        {platform === 'polymarket' && (
          <div className="flex items-center space-x-2 bg-dark-800 border border-gold/30 rounded-lg p-1">
            <button
              onClick={() => setView('markets')}
              className={`px-6 py-2 rounded-lg font-semibold transition-all ${
                view === 'markets'
                  ? 'bg-gold text-dark-900'
                  : 'text-gray-400 hover:text-gold'
              }`}
            >
              Markets
            </button>
            <button
              onClick={() => setView('traders')}
              className={`px-6 py-2 rounded-lg font-semibold transition-all flex items-center space-x-2 ${
                view === 'traders'
                  ? 'bg-gold text-dark-900'
                  : 'text-gray-400 hover:text-gold'
              }`}
            >
              <span>🐋</span>
              <span>Whale Tracker</span>
            </button>
          </div>
        )}
      </div>

      {/* Conditional Content: Markets or Whale Tracker */}
      {view === 'traders' ? (
        /* Whale Tracker View */
        <WhaleTracker />
      ) : (
        /* Markets View */
        <>
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
              <div className="text-red-400 font-semibold mb-2">Error Loading Markets</div>
              <div className="text-red-300">{error}</div>
              <button
                onClick={() => window.location.reload()}
                className="mt-4 px-4 py-2 bg-red-500/20 border border-red-500/40 rounded-lg text-red-300 hover:bg-red-500/30"
              >
                Retry
              </button>
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
            !loading && !error && (
              <div className="text-center py-16 bg-dark-800 border border-gold/20 rounded-lg">
                <div className="text-xl text-gray-400 mb-2">No markets found</div>
                <div className="text-sm text-gray-600">Try adjusting your search or filters</div>
              </div>
            )
          )}

          {/* Load More */}
          {!loading && hasMore && (
            <div className="text-center">
              <button
                onClick={handleLoadMore}
                disabled={loadingMore}
                className="px-8 py-3 bg-gold/10 border border-gold/30 rounded-lg text-gold hover:bg-gold/20 hover:border-gold/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loadingMore ? 'Loading...' : `Load More Markets (${markets.length} loaded)`}
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
        </>
      )}
    </div>
  )
}
