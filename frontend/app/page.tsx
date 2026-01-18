'use client'

import { useState, useRef, useEffect, useMemo } from 'react'
import { motion, useScroll, useTransform, useInView } from 'framer-motion'
import MarkdownRenderer from '@/components/MarkdownRenderer'
import ResultsTable from '@/components/ResultsTable'
import GraphVisualization from '@/components/GraphVisualization'
import MarketCard from '@/components/MarketCard'
import MarketDetailModal from '@/components/MarketDetailModal'
import ScrollToTop from '@/components/ScrollToTop'
import AnimatedLogo from '@/components/AnimatedLogo'
import WhaleTracker from '@/components/WhaleTracker'
import { Market } from '@/lib/mockData'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  results?: any[]
  useMarkdown?: boolean
  followUpQuestions?: string[]
  webContext?: {
    sources?: string[]
    citations?: Array<{number: number, url: string, referenced: boolean}>
  }
  metadata?: {
    tickers?: string[]
    companies?: string[]
    collections?: string[]
    queryIntent?: string
    resultCount?: number
  }
  queryPlan?: {
    aql_query?: string
    bind_vars?: any
    intent?: string
    explanation?: string
  }
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Welcome to KARGA. Ask me anything about financial markets, SEC filings, prediction markets, or run complex queries across our knowledge graph.',
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showAdvancedMode, setShowAdvancedMode] = useState(false)

  // Refs for scroll animations
  const heroRef = useRef(null)
  const whyGraphsRef = useRef(null)
  const statsRef = useRef(null)
  const graphVizRef = useRef(null)

  const isWhyGraphsInView = useInView(whyGraphsRef, { once: true, amount: 0.3 })
  const isStatsInView = useInView(statsRef, { once: true, amount: 0.3 })
  const isGraphVizInView = useInView(graphVizRef, { once: true, amount: 0.2 })

  const { scrollYProgress } = useScroll()
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.8])

  // Grid pattern opacity that increases as you scroll
  const gridOpacity1 = useTransform(scrollYProgress, [0, 0.2], [0.08, 0.15])
  const gridOpacity2 = useTransform(scrollYProgress, [0.2, 0.4], [0.15, 0.22])
  const gridOpacity3 = useTransform(scrollYProgress, [0.4, 0.6], [0.22, 0.28])

  // State for collection browser
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [collectionData, setCollectionData] = useState<any[]>([])
  const [isLoadingData, setIsLoadingData] = useState(false)
  const [searchFilter, setSearchFilter] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [collections, setCollections] = useState<Array<{name: string, count: number, description: string}>>([])
  const [loadingCollections, setLoadingCollections] = useState(true)
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({})

  // Collection name translations
  const collectionDisplayNames: Record<string, string> = {
    'Company': 'S&P 500 Companies',
    'MarketData': 'Stock Prices',
    'Award': 'Government Contract Awards',
    'EconomicData': 'FRED Economic Indicators',
    'commodity_positions': 'Commodities & Futures',
    'prediction_markets_polymarket': 'Polymarket',
    'prediction_markets_kalshi': 'Kalshi',
    'sec_sentences': '10-K Sentiment Analysis',
  }

  // Collection icons mapping
  const collectionIcons: Record<string, React.ReactNode> = {
    'Company': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
      </svg>
    ),
    'MarketData': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
      </svg>
    ),
    'Award': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    'EconomicData': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    'commodity_positions': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    'prediction_markets_polymarket': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    'prediction_markets_kalshi': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    'sec_sentences': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  }

  // State for markets section
  const [selectedPlatform, setSelectedPlatform] = useState<'polymarket' | 'kalshi'>('polymarket')
  const [polymarketView, setPolymarketView] = useState<'markets' | 'whales'>('markets') // Toggle between markets and whale tracker
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [sortBy, setSortBy] = useState<'volume' | 'probability' | 'traders'>('volume')
  const [markets, setMarkets] = useState<Market[]>([])
  const [categories, setCategories] = useState<Array<{category: string, count: number}>>([])
  const [loadingMarkets, setLoadingMarkets] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [marketError, setMarketError] = useState<string | null>(null)
  const [selectedMarket, setSelectedMarket] = useState<Market | null>(null)
  const [displayLimit, setDisplayLimit] = useState(12)
  const [hasMore, setHasMore] = useState(true)

  // Fetch collections metadata on mount
  useEffect(() => {
    const fetchCollections = async () => {
      try {
        setLoadingCollections(true)
        const { api } = await import('@/lib/api')
        const collectionsData = await api.getCollections()
        // Filter out sec_sections and only show collections we want
        const filteredCollections = collectionsData.filter((c: any) =>
          c.name !== 'sec_sections' && collectionDisplayNames[c.name]
        )
        setCollections(filteredCollections)
      } catch (error) {
        console.error('Failed to fetch collections:', error)
        // Fallback to static data if API fails
        setCollections([
          { name: 'Company', count: 852, description: 'S&P 500 companies with fundamentals' },
          { name: 'MarketData', count: 2100000, description: 'Daily OHLCV + technical indicators' },
          { name: 'Award', count: 500000, description: 'Federal contracts with embeddings' },
          { name: 'sec_sentences', count: 890000, description: 'Filing sentences with FinBERT scores' },
          { name: 'prediction_markets_polymarket', count: 12968, description: 'Polymarket prediction data' },
          { name: 'prediction_markets_kalshi', count: 5432, description: 'Kalshi event contracts' },
          { name: 'EconomicData', count: 8900, description: 'Macro indicators & rates' },
        ])
      } finally {
        setLoadingCollections(false)
      }
    }

    fetchCollections()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const currentInput = input
    setInput('')
    setIsLoading(true)

    // Create placeholder for streaming assistant message
    const assistantMessage: Message = {
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      useMarkdown: true,
      results: []
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      // Prepare conversation history (last 6 messages, excluding current)
      const conversationHistory = messages.slice(-6).map(msg => ({
        role: msg.role,
        content: msg.content,
        metadata: msg.metadata  // Include extracted entities
      }))

      // Use streaming endpoint
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_BASE_URL}/api/query/execute-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentInput,
          conversation_history: conversationHistory
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''
      let contentBuffer = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)

            if (data === '[DONE]') {
              setIsLoading(false)
              break
            }

            try {
              const parsed = JSON.parse(data)

              if (parsed.type === 'progress') {
                // Update with progress message
                setMessages((prev) => {
                  const newMessages = [...prev]
                  newMessages[newMessages.length - 1].content = `*${parsed.message}*${parsed.details ? ` (${parsed.details})` : ''}`
                  return newMessages
                })
              } else if (parsed.type === 'content_start') {
                // Clear progress, start content
                contentBuffer = ''
                setMessages((prev) => {
                  const newMessages = [...prev]
                  newMessages[newMessages.length - 1].content = ''
                  return newMessages
                })
              } else if (parsed.type === 'content_chunk') {
                // Append content chunk
                contentBuffer += parsed.chunk
                console.log('[STREAM] Content chunk received, total length:', contentBuffer.length)
                setMessages((prev) => {
                  const newMessages = [...prev]
                  newMessages[newMessages.length - 1].content = contentBuffer
                  return newMessages
                })
              } else if (parsed.type === 'complete') {
                // Add final data - PRESERVE EXISTING CONTENT
                setMessages((prev) => {
                  const newMessages = [...prev]
                  const lastMsg = newMessages[newMessages.length - 1]

                  // IMPORTANT: Preserve the content that was streamed
                  // Only update if content is missing (shouldn't happen)
                  if (!lastMsg.content || lastMsg.content.trim() === '') {
                    lastMsg.content = contentBuffer || 'No analysis available'
                  }

                  // Store follow-up questions separately (render as buttons)
                  if (parsed.follow_up_questions && parsed.follow_up_questions.length > 0) {
                    lastMsg.followUpQuestions = parsed.follow_up_questions
                  }

                  // Store web context (sources and citations)
                  if (parsed.web_context) {
                    lastMsg.webContext = parsed.web_context
                  }

                  // Store query plan for advanced mode
                  if (parsed.query_plan) {
                    lastMsg.queryPlan = parsed.query_plan
                  }

                  // Extract and store metadata from results
                  lastMsg.results = parsed.results
                  if (parsed.results && parsed.results.length > 0) {
                    const tickers = new Set<string>()
                    const companies = new Set<string>()

                    parsed.results.forEach((result: any) => {
                      if (result.ticker) tickers.add(result.ticker)
                      if (result.company) companies.add(result.company)
                      if (result.recipient_name) companies.add(result.recipient_name)
                      if (result.recipient) companies.add(result.recipient)
                    })

                    lastMsg.metadata = {
                      tickers: Array.from(tickers),
                      companies: Array.from(companies),
                      collections: parsed.query_plan?.collections || [],
                      queryIntent: parsed.query_intent,
                      resultCount: parsed.count || parsed.results.length
                    }
                  }

                  return newMessages
                })
                setIsLoading(false)
              } else if (parsed.type === 'error') {
                // Handle error
                setMessages((prev) => {
                  const newMessages = [...prev]
                  newMessages[newMessages.length - 1].content = `❌ Error: ${parsed.message}`
                  return newMessages
                })
                setIsLoading(false)
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data:', data, parseError)
            }
          }
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const newMessages = [...prev]
        newMessages[newMessages.length - 1].content = `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`
        return newMessages
      })
      setIsLoading(false)
    }
  }

  const suggestedQuestions = [
    'Show me the top 10 largest government contracts',
    'What are the most active Polymarket prediction markets?',
    'Find contracts related to artificial intelligence',
    'What markets are whale traders betting on?',
    'Show me Apple stock data for the last 30 days',
    'Find SEC filings with negative sentiment about cybersecurity',
  ]

  // Debounce search input (500ms delay)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchFilter)
    }, 500)

    return () => clearTimeout(timer)
  }, [searchFilter])

  // Fetch collection data when selected or search changes
  useEffect(() => {
    const fetchCollectionData = async () => {
      if (!selectedCollection) {
        setCollectionData([])
        setSearchFilter('')
        setDebouncedSearch('')
        setSortColumn(null)
        setColumnFilters({})
        return
      }

      setIsLoadingData(true)
      try {
        const { api } = await import('@/lib/api')
        // Pass search to server for full database search
        // Using max limit of 500 (backend constraint)
        const response = await api.browseCollection(
          selectedCollection,
          500,
          debouncedSearch || undefined
        )
        // Handle multiple response formats: {documents: [...]}, {data: [...]}, or [...]
        const data = Array.isArray(response)
          ? response
          : (response.documents || response.data || [])
        console.log('Collection data received:', data.length, 'items', 'Search:', debouncedSearch)

        // Auto-detect date column and set as default sort (only on first load)
        if (data.length > 0 && !debouncedSearch) {
          const firstItem = data[0]
          const dateColumns = Object.keys(firstItem).filter(key =>
            key.toLowerCase().includes('date') ||
            key.toLowerCase().includes('time') ||
            key === 'timestamp'
          )

          // Prefer 'date' column, otherwise use first date-like column
          const defaultSortCol = dateColumns.find(col => col === 'date') || dateColumns[0]
          if (defaultSortCol) {
            setSortColumn(defaultSortCol)
            setSortDirection('desc') // Most recent first
          } else {
            setSortColumn(null)
          }
        }

        setCollectionData(data)
      } catch (error) {
        console.error('Failed to fetch collection data:', error)
        setCollectionData([])
      } finally {
        setIsLoadingData(false)
      }
    }

    fetchCollectionData()
  }, [selectedCollection, debouncedSearch])

  // Filter and sort data (search is now server-side, only column filters are client-side)
  const filteredData = useMemo(() => {
    let result = [...collectionData]

    // Apply column-specific filters (client-side for quick refinement)
    Object.entries(columnFilters).forEach(([column, filterValue]) => {
      if (filterValue) {
        const filterLower = filterValue.toLowerCase()
        result = result.filter((item) =>
          String(item[column]).toLowerCase().includes(filterLower)
        )
      }
    })

    // Apply sorting (client-side for instant feedback)
    if (sortColumn) {
      result.sort((a, b) => {
        const aVal = a[sortColumn]
        const bVal = b[sortColumn]

        // Handle null/undefined
        if (aVal == null && bVal == null) return 0
        if (aVal == null) return 1
        if (bVal == null) return -1

        // Sort by type
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
        }

        // Date sorting
        if (sortColumn.toLowerCase().includes('date') || sortColumn.toLowerCase().includes('time')) {
          const dateA = new Date(aVal).getTime()
          const dateB = new Date(bVal).getTime()
          if (!isNaN(dateA) && !isNaN(dateB)) {
            return sortDirection === 'asc' ? dateA - dateB : dateB - dateA
          }
        }

        // String sorting
        const strA = String(aVal).toLowerCase()
        const strB = String(bVal).toLowerCase()
        const comparison = strA.localeCompare(strB)
        return sortDirection === 'asc' ? comparison : -comparison
      })
    }

    return result
  }, [collectionData, columnFilters, sortColumn, sortDirection])

  // Debug logging
  useEffect(() => {
    console.log('Collection data state:', {
      collectionDataLength: collectionData.length,
      filteredDataLength: filteredData.length,
      selectedCollection,
      isLoadingData,
      firstItem: collectionData[0]
    })
  }, [collectionData, filteredData, selectedCollection, isLoadingData])

  // Prefetch markets on app load (background, both platforms)
  useEffect(() => {
    const prefetchMarkets = async () => {
      try {
        const { api } = await import('@/lib/api')
        const { marketCache } = await import('@/lib/marketCache')
        // Clear cache on mount to ensure fresh data
        marketCache.clear()
        // Prefetch Polymarket in background (most common)
        api.getFeaturedMarkets(200, 'polymarket').catch(() => {})
        api.getCategories('polymarket').catch(() => {})
      } catch (err) {
        // Silent fail for prefetch
      }
    }
    prefetchMarkets()
  }, [])

  // Fetch markets and categories when platform changes
  useEffect(() => {
    const fetchMarketsData = async () => {
      try {
        setLoadingMarkets(true)
        setSelectedCategory('All')
        setSearchQuery('')
        const { api } = await import('@/lib/api')

        // Fetch markets for both Polymarket and Kalshi
        const [marketsData, categoriesData] = await Promise.all([
          api.getFeaturedMarkets(200, selectedPlatform),
          api.getCategories(selectedPlatform)
        ])

        // Filter out markets with past end dates
        const now = new Date()
        const activeMarkets = marketsData.filter((m: any) => {
          if (!m.end_date) return true
          const endDate = new Date(m.end_date)
          return endDate > now
        })

        // Sort by volume and take top markets
        const sortedMarkets = activeMarkets.sort((a: any, b: any) => (b.volume_24h || 0) - (a.volume_24h || 0))

        const formattedMarkets = sortedMarkets.map((m: any) => ({
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
        setDisplayLimit(12)
        setHasMore(formattedMarkets.length > 12)
        setMarketError(null)
      } catch (err: any) {
        console.error('Failed to fetch markets:', err)
        const errorMsg = err?.message?.includes('500')
          ? 'Server error loading markets. The data might be temporarily unavailable. Please try refreshing.'
          : err?.message?.includes('timeout')
          ? 'Request timed out. The server might be slow. Please try again.'
          : 'Failed to load markets. Please try refreshing the page.'
        setMarketError(errorMsg)
        setMarkets([])
        setCategories([])
      } finally {
        setLoadingMarkets(false)
      }
    }

    fetchMarketsData()
  }, [selectedPlatform])

  // Load more markets - increase display limit
  const handleLoadMore = () => {
    const newLimit = Math.min(displayLimit + 12, 50)
    setDisplayLimit(newLimit)
    setHasMore(newLimit < Math.min(markets.length, 50))
  }

  // Filter and sort markets
  const filteredMarkets = useMemo(() => {
    let filtered = markets

    if (selectedCategory !== 'All') {
      filtered = filtered.filter(m => m.category === selectedCategory)
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(m =>
        m.question.toLowerCase().includes(query) ||
        m.category.toLowerCase().includes(query) ||
        (m.description && m.description.toLowerCase().includes(query))
      )
    }

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
    <div className="relative z-10">
      {/* Hero Section */}
      <motion.section
        ref={heroRef}
        style={{ opacity, scale }}
        className="min-h-screen snap-start flex flex-col items-center justify-center px-6 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />

        {/* Smooth gradient overlay at bottom for transition */}
        <div className="absolute bottom-0 left-0 right-0 h-64 bg-gradient-to-b from-transparent via-dark-900/50 to-dark-900 pointer-events-none" />

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
          className="text-center z-10 px-4"
        >
          {/* Animated KARGA Logo */}
          <div className="mb-8 md:mb-16">
            <AnimatedLogo />
          </div>
          <p className="text-xl sm:text-2xl md:text-3xl lg:text-4xl text-gray-200 mb-4 md:mb-6 font-light leading-tight">
            Financial Intelligence Powered by Knowledge Graphs
          </p>
          <p className="text-base sm:text-lg md:text-xl lg:text-2xl text-gray-400 max-w-3xl mx-auto font-light leading-relaxed">
            Ask natural language questions. Get insights from millions of connected data points.
          </p>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
          className="absolute bottom-10 z-10"
        >
          <div className="flex flex-col items-center">
            <p className="text-sm text-gray-500 mb-2">Scroll to explore</p>
            <motion.div
              animate={{ y: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="w-6 h-10 border-2 border-gold/30 rounded-full p-2"
            >
              <motion.div className="w-1.5 h-1.5 bg-gold rounded-full mx-auto" />
            </motion.div>
          </div>
        </motion.div>
      </motion.section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Why Graphs Section */}
      <section
        ref={whyGraphsRef}
        className="snap-start flex flex-col items-center justify-center px-6 py-16 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity1 }}
        />
        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isWhyGraphsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-10"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gold mb-4">Why Knowledge Graphs?</h2>
            <p className="text-lg text-gray-400 max-w-3xl mx-auto">
              Traditional databases see data in silos. Graphs see connections.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: 'Connected Data',
                description: 'Every company links to market data, government contracts, SEC filings, and prediction markets',
                delay: 0.2,
                icon: (
                  <svg className="w-10 h-10 text-gold mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                ),
              },
              {
                title: 'Semantic Search',
                description: 'Find contracts mentioning "Iran" or "cybersecurity" using AI embeddings, not just keywords',
                delay: 0.4,
                icon: (
                  <svg className="w-10 h-10 text-gold mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                  </svg>
                ),
              },
              {
                title: 'Hybrid Intelligence',
                description: 'Combine historical database queries with real-time web search for complete context',
                delay: 0.6,
                icon: (
                  <svg className="w-10 h-10 text-gold mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                ),
              },
            ].map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 50 }}
                animate={isWhyGraphsInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.8, delay: item.delay }}
                className="bg-dark-800 border border-gold/20 rounded-lg p-6 hover:border-gold/40 transition-all"
              >
                {item.icon}
                <h3 className="text-xl font-semibold text-gold mb-3">{item.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Chat Interface Section */}
      <section id="query" className="min-h-screen snap-start flex items-center justify-center px-6 py-12 relative overflow-hidden">
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity2 }}
        />
        <div className="container mx-auto max-w-7xl relative z-10">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.8 }}
            className="mb-8 text-center"
          >
            <h2 className="text-2xl md:text-4xl font-bold text-gold mb-2">AI Query Interface</h2>
            <p className="text-gray-500">Natural language queries over financial knowledge graph</p>
          </motion.div>

          {/* Chat Container */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.8 }}
            className="bg-dark-800 border border-gold/20 rounded-lg shadow-xl mb-6"
          >
            {/* Messages */}
            <div className="h-[500px] md:h-[700px] overflow-y-auto p-4 md:p-6 space-y-4">
              {messages.map((message, idx) => (
                <div
                  key={idx}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-gold/20 border border-gold/40 text-gray-100'
                        : 'bg-dark-700 border border-gold/20 text-gray-300'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className="text-xs font-semibold text-gray-500 uppercase">
                        {message.role === 'user' ? 'You' : 'AI'}
                      </div>
                      <div className="flex-1">
                        <div className="text-base mb-2 leading-relaxed">
                          {message.useMarkdown ? (
                            <MarkdownRenderer content={message.content} />
                          ) : (
                            message.content
                          )}
                        </div>
                        {showAdvancedMode && message.queryPlan && message.queryPlan.aql_query && (
                          <details className="mt-3 mb-3">
                            <summary className="cursor-pointer text-xs text-purple-400 hover:text-purple-300 font-semibold">
                              🔧 Query Plan (Advanced)
                            </summary>
                            <div className="mt-2 bg-dark-900 border border-purple-500/30 rounded p-3">
                              <div className="text-xs text-gray-400 mb-2">
                                <strong className="text-purple-400">Intent:</strong> {message.queryPlan.intent || 'N/A'}
                                {message.queryPlan.explanation && (
                                  <div className="mt-1">
                                    <strong className="text-purple-400">Strategy:</strong> {message.queryPlan.explanation}
                                  </div>
                                )}
                              </div>
                              <div className="text-xs text-gray-500 font-mono mb-1">AQL Query:</div>
                              <pre className="text-xs bg-black/50 p-2 rounded overflow-x-auto text-green-400 border border-green-500/20">
                                {message.queryPlan.aql_query}
                              </pre>
                              {message.queryPlan.bind_vars && Object.keys(message.queryPlan.bind_vars).length > 0 && (
                                <div className="mt-2">
                                  <div className="text-xs text-gray-500 font-mono mb-1">Bind Variables:</div>
                                  <pre className="text-xs bg-black/50 p-2 rounded overflow-x-auto text-blue-400 border border-blue-500/20">
                                    {JSON.stringify(message.queryPlan.bind_vars, null, 2)}
                                  </pre>
                                </div>
                              )}
                            </div>
                          </details>
                        )}
                        {message.results && message.results.length > 0 && !message.content.includes('|') && (
                          <details className="mt-3">
                            <summary className="cursor-pointer text-xs text-gold hover:text-gold/80 font-semibold">
                              View raw data table ({message.results.length} rows)
                            </summary>
                            <div className="mt-2">
                              <ResultsTable data={message.results} maxRows={20} />
                            </div>
                          </details>
                        )}
                        {message.webContext && (message.webContext.citations?.length || message.webContext.sources?.length) && (
                          <div className="mt-4 pt-3 border-t border-gold/20">
                            <div className="text-xs font-semibold text-gold mb-2">Sources:</div>
                            <div className="space-y-1">
                              {message.webContext.citations && message.webContext.citations.length > 0 ? (
                                message.webContext.citations.map((citation) => (
                                  <div key={citation.number} className="text-xs text-gray-400">
                                    <span className="text-gold font-mono">[{citation.number}]</span>{' '}
                                    <a
                                      href={citation.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-blue-400 hover:text-blue-300 hover:underline break-all"
                                    >
                                      {citation.url}
                                    </a>
                                  </div>
                                ))
                              ) : (
                                message.webContext.sources?.map((source, sIdx) => (
                                  <div key={sIdx} className="text-xs text-gray-400">
                                    <span className="text-gold font-mono">[{sIdx + 1}]</span>{' '}
                                    <a
                                      href={source}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-blue-400 hover:text-blue-300 hover:underline break-all"
                                    >
                                      {source}
                                    </a>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        )}
                        {message.followUpQuestions && message.followUpQuestions.length > 0 && (
                          <div className="mt-4">
                            <div className="text-xs font-semibold text-gold mb-2">Follow-up questions:</div>
                            <div className="flex flex-col gap-2">
                              {message.followUpQuestions.map((question, qIdx) => (
                                <button
                                  key={qIdx}
                                  onClick={() => setInput(question)}
                                  className="text-left text-xs bg-dark-800 border border-gold/30 rounded px-3 py-2 text-gray-300 hover:border-gold/60 hover:bg-dark-700 hover:text-gold transition-all"
                                >
                                  {question}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                        <div className="text-xs text-gray-600 mt-2">
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-dark-700 border border-gold/20 rounded-lg p-4">
                    <div className="flex items-center space-x-3">
                      <div className="text-xs font-semibold text-gray-500 uppercase">AI</div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="border-t border-gold/20 p-3 md:p-4">
              {/* Advanced Mode Toggle */}
              <div className="flex items-center justify-end mb-3">
                <label className="flex items-center cursor-pointer">
                  <span className="text-xs text-gray-400 mr-2">Advanced Mode</span>
                  <div className="relative">
                    <input
                      type="checkbox"
                      checked={showAdvancedMode}
                      onChange={(e) => setShowAdvancedMode(e.target.checked)}
                      className="sr-only"
                    />
                    <div className={`block w-10 h-6 rounded-full transition ${showAdvancedMode ? 'bg-purple-500' : 'bg-gray-600'}`}></div>
                    <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition ${showAdvancedMode ? 'transform translate-x-4' : ''}`}></div>
                  </div>
                </label>
              </div>
              <form onSubmit={handleSubmit} className="flex flex-col md:flex-row md:space-x-3 space-y-3 md:space-y-0">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit(e)
                    }
                  }}
                  placeholder="Ask a question about markets, companies, or contracts..."
                  rows={3}
                  className="flex-1 bg-dark-700 border border-gold/30 rounded-lg px-3 py-2 md:px-4 md:py-3 text-sm md:text-base text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20 resize-y min-h-[80px] max-h-[200px]"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="w-full md:w-auto px-6 py-3 bg-gold/20 border border-gold/40 rounded-lg text-gold hover:bg-gold/30 hover:border-gold/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold md:self-end"
                >
                  Send
                </button>
              </form>
            </div>
          </motion.div>

          {/* Suggested Questions */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h3 className="text-base md:text-lg font-semibold text-gold mb-4">Suggested Questions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {suggestedQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(question)}
                  className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-4 text-left text-xs md:text-sm text-gray-400 hover:border-gold/40 hover:text-gold transition-all"
                >
                  {question}
                </button>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Graph Architecture Visualization */}
      <section
        ref={graphVizRef}
        className="min-h-screen snap-start flex flex-col items-center justify-center px-6 py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-7xl mx-auto w-full relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isGraphVizInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-5xl font-bold text-gold mb-6">Graph Architecture</h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
              Everything connects to everything. Click nodes to explore how data flows through the knowledge graph.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isGraphVizInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <GraphVisualization />
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={isGraphVizInView ? { opacity: 1 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="mt-12 text-center"
          >
            <p className="text-gray-400 text-lg">
              Built on <span className="text-gold font-semibold">ArangoDB</span> with{' '}
              <span className="text-gold font-semibold">GPT-4</span> query planning and{' '}
              <span className="text-gold font-semibold">Perplexity</span> web search
            </p>
          </motion.div>
        </div>
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Prediction Markets Section */}
      <section
        id="markets"
        className="min-h-screen snap-start flex flex-col justify-center px-4 py-8 md:px-6 md:py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-7xl mx-auto w-full relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <h2 className="text-3xl md:text-5xl font-bold text-gold mb-2 text-center">Prediction Markets</h2>
            <p className="text-gray-400 text-center mb-6">Live market data from Polymarket & Kalshi</p>

            {/* Platform Toggle */}
            <div className="flex justify-center mb-4">
              <div className="inline-flex bg-dark-800 border border-gold/30 rounded-lg p-1">
                <button
                  onClick={() => setSelectedPlatform('polymarket')}
                  className={`px-6 py-2 rounded-lg transition-all font-semibold ${
                    selectedPlatform === 'polymarket'
                      ? 'bg-gold/20 text-gold border border-gold/40'
                      : 'text-gray-400 hover:text-gold'
                  }`}
                >
                  Polymarket
                </button>
                <button
                  onClick={() => setSelectedPlatform('kalshi')}
                  className={`px-6 py-2 rounded-lg transition-all font-semibold ${
                    selectedPlatform === 'kalshi'
                      ? 'bg-gold/20 text-gold border border-gold/40'
                      : 'text-gray-400 hover:text-gold'
                  }`}
                >
                  Kalshi
                </button>
              </div>
            </div>

            {/* Polymarket View Toggle (Markets vs Whales) */}
            {selectedPlatform === 'polymarket' && (
              <div className="flex justify-center mb-8">
                <div className="inline-flex bg-dark-700 border border-gold/20 rounded-lg p-1">
                  <button
                    onClick={() => setPolymarketView('markets')}
                    className={`px-5 py-2 rounded-lg transition-all text-sm font-semibold ${
                      polymarketView === 'markets'
                        ? 'bg-gold/20 text-gold border border-gold/40'
                        : 'text-gray-400 hover:text-gold'
                    }`}
                  >
                    Markets
                  </button>
                  <button
                    onClick={() => setPolymarketView('whales')}
                    className={`px-5 py-2 rounded-lg transition-all text-sm font-semibold ${
                      polymarketView === 'whales'
                        ? 'bg-gold/20 text-gold border border-gold/40'
                        : 'text-gray-400 hover:text-gold'
                    }`}
                  >
                    Whale Traders
                  </button>
                </div>
              </div>
            )}
          </motion.div>

          {/* Conditional Content: Markets or Whale Tracker */}
          {selectedPlatform === 'polymarket' && polymarketView === 'whales' ? (
            /* Whale Tracker View */
            <div className="mt-8">
              <WhaleTracker />
            </div>
          ) : (
            /* Markets View */
            <>
          {/* Metrics Cards */}
          <div className="grid grid-cols-4 gap-4 mb-8">
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
                <div className="text-gray-400 text-sm mb-1">Active Markets</div>
                <div className="text-3xl font-bold text-gold">{filteredMarkets.length}</div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
                <div className="text-gray-400 text-sm mb-1">24h Volume</div>
                <div className="text-3xl font-bold text-gold">{formatVolume(totalVolume)}</div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
                <div className="text-gray-400 text-sm mb-1">Avg Probability</div>
                <div className="text-3xl font-bold text-gold">{avgProbability.toFixed(0)}%</div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5 hover:border-gold/40 transition-all">
                <div className="text-gray-400 text-sm mb-1">Categories</div>
                <div className="text-3xl font-bold text-gold">{categories.length}</div>
              </div>
            </div>

          {/* Natural Language Search */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="mb-6"
          >
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search markets... (e.g., 'Trump', 'crypto', 'election')"
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
              <div className="mt-2 text-sm text-gray-400">
                Found <span className="text-gold font-semibold">{filteredMarkets.length}</span> markets matching "{searchQuery}"
              </div>
            )}
          </motion.div>

          {/* Loading State */}
          {loadingMarkets && (
            <div className="text-center py-16">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gold"></div>
              <p className="text-gray-400 mt-4">Loading markets...</p>
            </div>
          )}

          {/* Error State */}
          {marketError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 mb-8">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="text-red-400 font-semibold mb-2">Error Loading Markets</div>
                  <div className="text-red-300 mb-4">{marketError}</div>
                  <button
                    onClick={() => {
                      setMarketError(null)
                      setLoadingMarkets(true)
                      // Trigger refetch by toggling platform state
                      setSelectedPlatform(prev => prev === 'polymarket' ? 'polymarket' : 'polymarket')
                    }}
                    className="px-4 py-2 bg-red-500/20 border border-red-500/40 rounded-lg text-red-300 hover:bg-red-500/30 hover:border-red-500/60 transition-all font-semibold"
                  >
                    Retry
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Filters */}
          {!loadingMarkets && !marketError && (
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="flex items-center justify-between mb-8"
            >
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
                <span className="text-gray-400 text-sm whitespace-nowrap">Sort by:</span>
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
            </motion.div>
          )}

          {/* Markets Table */}
          {!loadingMarkets && !marketError && filteredMarkets.length > 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="bg-dark-700 border border-gold/20 rounded-lg overflow-hidden mb-12"
            >
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-dark-800 border-b border-gold/20">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Question</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Category</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">Yes</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">No</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Volume 24h</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Liquidity</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Confidence</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Days Left</th>
                    </tr>
                    {/* Column Filters */}
                    <tr className="bg-dark-800/50">
                      <th className="px-2 py-2">
                        <input
                          type="text"
                          placeholder="Filter question..."
                          className="w-full bg-dark-900 border border-gold/20 rounded px-2 py-1 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-gold/40"
                        />
                      </th>
                      <th className="px-2 py-2">
                        <input
                          type="text"
                          placeholder="Filter category..."
                          className="w-full bg-dark-900 border border-gold/20 rounded px-2 py-1 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-gold/40"
                        />
                      </th>
                      <th colSpan={6}></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gold/10">
                    {filteredMarkets.slice(0, displayLimit).map((market, index) => (
                      <tr
                        key={market.id}
                        className={`hover:bg-dark-800/70 transition-colors cursor-pointer ${
                          index % 2 === 0 ? 'bg-dark-800/20' : 'bg-gold/5'
                        }`}
                        onClick={() => setSelectedMarket(market)}
                      >
                        <td className="px-4 py-3 text-sm text-gray-200 max-w-md">
                          <div className="line-clamp-2">{market.question}</div>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-400 whitespace-nowrap">
                          {market.category}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className="text-sm font-semibold text-green-400">{market.yes_prob}%</span>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className="text-sm font-semibold text-red-400">{market.no_prob}%</span>
                        </td>
                        <td className="px-4 py-3 text-right text-sm text-gold font-medium">
                          {market.volume_24h >= 1000000
                            ? `$${(market.volume_24h / 1000000).toFixed(1)}M`
                            : market.volume_24h >= 1000
                            ? `$${(market.volume_24h / 1000).toFixed(0)}k`
                            : `$${market.volume_24h}`}
                        </td>
                        <td className="px-4 py-3 text-right text-sm text-gray-300">
                          {market.liquidity >= 1000000
                            ? `$${(market.liquidity / 1000000).toFixed(1)}M`
                            : market.liquidity >= 1000
                            ? `$${(market.liquidity / 1000).toFixed(0)}k`
                            : `$${Math.round(market.liquidity)}`}
                        </td>
                        <td className="px-4 py-3 text-right text-sm">
                          <span className={`${
                            (market.probability_confidence || 0) > 0.3 ? 'text-green-400' :
                            (market.probability_confidence || 0) > 0.15 ? 'text-yellow-400' :
                            'text-gray-500'
                          }`}>
                            {((market.probability_confidence || 0) * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right text-xs text-gray-400">
                          {market.days_until_end || Math.ceil((new Date(market.end_date).getTime() - Date.now()) / (1000 * 60 * 60 * 24))}d
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          ) : !loadingMarkets && !marketError ? (
            <div className="text-center py-16 bg-dark-800 border border-gold/20 rounded-lg">
              <div className="text-xl text-gray-400 mb-2">No markets found</div>
              <div className="text-sm text-gray-500">Try adjusting your search or filters</div>
            </div>
          ) : null}

          {/* Load More */}
          {!loadingMarkets && hasMore && filteredMarkets.length > displayLimit && (
            <div className="text-center">
              <button
                onClick={handleLoadMore}
                className="px-8 py-3 bg-gold/10 border border-gold/30 rounded-lg text-gold hover:bg-gold/20 hover:border-gold/50 transition-all"
              >
                Load More Markets (showing {displayLimit} of {Math.min(filteredMarkets.length, 50)})
              </button>
            </div>
          )}
              </>
            )}
        </div>

        {/* Market Detail Modal */}
        {selectedMarket && (
          <MarketDetailModal
            market={selectedMarket}
            onClose={() => setSelectedMarket(null)}
          />
        )}
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Data Universe - Collections Browser */}
      <section
        id="database"
        ref={statsRef}
        className="min-h-screen snap-start flex flex-col justify-center px-4 py-8 md:px-6 md:py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-7xl mx-auto w-full relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isStatsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-5xl font-bold text-gold mb-6">Data Universe</h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
              Explore all database collections and their connections
            </p>

            {/* Top Stats */}
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
                <div className="text-gray-400 text-sm mb-1">Collections</div>
                <div className="text-3xl font-bold text-gold">{collections.length}</div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
                <div className="text-gray-400 text-sm mb-1">Total Documents</div>
                <div className="text-3xl font-bold text-gold">
                  {(collections.reduce((sum, c) => sum + c.count, 0) / 1000000).toFixed(1)}M
                </div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
                <div className="text-gray-400 text-sm mb-1">Edge Collections</div>
                <div className="text-3xl font-bold text-gold">15</div>
              </div>
            </div>
          </motion.div>

          {/* Collections Grid */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={isStatsInView ? { opacity: 1 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h3 className="text-2xl font-semibold text-gold mb-6 text-center">Document Collections</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-8">
              {collections.map((collection, idx) => (
                <motion.button
                  key={collection.name}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={isStatsInView ? { opacity: 1, scale: 1 } : {}}
                  transition={{ duration: 0.6, delay: 0.3 + idx * 0.05 }}
                  onClick={() => setSelectedCollection(collection.name === selectedCollection ? null : collection.name)}
                  className={`bg-dark-800 border rounded-lg p-5 text-left transition-all ${
                    selectedCollection === collection.name
                      ? 'border-gold/60 ring-2 ring-gold/20'
                      : 'border-gold/20 hover:border-gold/40'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-10 h-10 rounded-lg bg-gold/10 border border-gold/30 flex items-center justify-center">
                      {collectionIcons[collection.name] || (
                        <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                        </svg>
                      )}
                    </div>
                  </div>
                  <div className="text-gold font-semibold mb-1 text-sm">
                    {collectionDisplayNames[collection.name] || collection.name}
                  </div>
                  <div className="text-xs text-gray-500 mb-2">
                    {collection.count.toLocaleString()} docs
                  </div>
                  <div className="text-xs text-gray-600">{collection.description}</div>
                </motion.button>
              ))}
            </div>
          </motion.div>

          {/* Collection Details */}
          {selectedCollection && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-dark-800 border border-gold/30 rounded-lg p-6 mt-6"
            >
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gold">
                    {collectionDisplayNames[selectedCollection] || selectedCollection}
                  </h3>
                  <p className="text-gray-400 text-sm mt-1">
                    {collections.find(c => c.name === selectedCollection)?.description}
                  </p>
                </div>
                <button
                  onClick={() => {
                    setSelectedCollection(null)
                    setSearchFilter('')
                  }}
                  className="text-gray-500 hover:text-gold transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-dark-700 border border-gold/20 rounded p-4">
                  <div className="text-xs text-gray-400 mb-1">Total Documents</div>
                  <div className="text-2xl font-semibold text-gold">
                    {collections.find(c => c.name === selectedCollection)?.count.toLocaleString()}
                  </div>
                </div>
                <div className="bg-dark-700 border border-gold/20 rounded p-4">
                  <div className="text-xs text-gray-400 mb-1">Loaded</div>
                  <div className="text-2xl font-semibold text-gold">
                    {filteredData.length}
                  </div>
                </div>
                <div className="bg-dark-700 border border-gold/20 rounded p-4">
                  <div className="text-xs text-gray-400 mb-1">Fields</div>
                  <div className="text-2xl font-semibold text-gold">
                    {collectionData.length > 0 ? Object.keys(collectionData[0]).length : '—'}
                  </div>
                </div>
              </div>

              {/* Search Filter - Server-side full database search */}
              <div className="mb-4 relative">
                <input
                  type="text"
                  placeholder="Search database (e.g., '2023', company name, keyword)..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full bg-dark-700 border border-gold/30 rounded-lg px-4 py-3 pr-10 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20"
                />
                {isLoadingData && searchFilter ? (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gold"></div>
                  </div>
                ) : searchFilter ? (
                  <button
                    onClick={() => setSearchFilter('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gold transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                ) : null}
              </div>
              {searchFilter && !isLoadingData && (
                <div className="mb-4 text-sm text-gray-400">
                  Found {collectionData.length} results (showing up to 500 records)
                </div>
              )}

              {/* Data Table */}
              <div className="bg-dark-700 border border-gold/20 rounded-lg overflow-hidden">
                {isLoadingData ? (
                  <div className="p-8 text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gold"></div>
                    <p className="text-gray-400 mt-4">Loading data...</p>
                  </div>
                ) : filteredData.length > 0 ? (
                  <>
                    <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-dark-800 sticky top-0 z-10">
                          {/* Column Headers - Sortable */}
                          <tr>
                            {Object.keys(filteredData[0])
                              .filter(key => !key.startsWith('_'))
                              .map((key) => (
                                <th
                                  key={key}
                                  className="px-4 py-3 text-left text-xs font-semibold text-gold uppercase tracking-wider border-b border-gold/20 cursor-pointer hover:bg-dark-700 transition-colors"
                                  onClick={() => {
                                    if (sortColumn === key) {
                                      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
                                    } else {
                                      setSortColumn(key)
                                      setSortDirection('desc')
                                    }
                                  }}
                                >
                                  <div className="flex items-center space-x-2">
                                    <span>{key}</span>
                                    {sortColumn === key && (
                                      <span className="text-gold">
                                        {sortDirection === 'asc' ? '↑' : '↓'}
                                      </span>
                                    )}
                                  </div>
                                </th>
                              ))}
                          </tr>
                          {/* Column Filters */}
                          <tr>
                            {Object.keys(filteredData[0])
                              .filter(key => !key.startsWith('_'))
                              .map((key) => (
                                <th key={key} className="px-2 py-2 border-b border-gold/10">
                                  <input
                                    type="text"
                                    placeholder="Filter..."
                                    value={columnFilters[key] || ''}
                                    onChange={(e) => {
                                      setColumnFilters(prev => ({
                                        ...prev,
                                        [key]: e.target.value
                                      }))
                                    }}
                                    className="w-full bg-dark-900 border border-gold/20 rounded px-2 py-1 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-gold/40"
                                    onClick={(e) => e.stopPropagation()}
                                  />
                                </th>
                              ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gold/10">
                          {filteredData.slice(0, 100).map((row, idx) => (
                            <tr key={idx} className={`hover:bg-dark-800/70 transition-colors ${
                              idx % 2 === 0 ? 'bg-dark-800/20' : 'bg-gold/5'
                            }`}>
                              {Object.entries(row)
                                .filter(([key]) => !key.startsWith('_'))
                                .map(([key, value]) => (
                                  <td key={key} className="px-4 py-3 text-gray-300 whitespace-nowrap">
                                    {typeof value === 'object' && value !== null ? (
                                      <span className="text-xs text-gray-500 italic">
                                        {Array.isArray(value) ? `Array[${value.length}]` : 'Object'}
                                      </span>
                                    ) : typeof value === 'number' ? (
                                      <span className="text-gold">{value.toLocaleString()}</span>
                                    ) : String(value).length > 50 ? (
                                      <span className="text-xs" title={String(value)}>
                                        {String(value).substring(0, 50)}...
                                      </span>
                                    ) : (
                                      String(value)
                                    )}
                                  </td>
                                ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {filteredData.length > 1000 && (
                      <div className="bg-dark-800 border-t border-gold/20 px-4 py-3 text-center text-sm text-gray-400">
                        Showing first 1,000 of {filteredData.length.toLocaleString()} results. Use search/filters to narrow down results.
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-8 text-center text-gray-400">
                    {searchFilter || Object.values(columnFilters).some(v => v) ? 'No matching records found' : 'No data available'}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* About Section */}
      <section
        id="about"
        className="min-h-screen snap-start flex flex-col justify-center px-4 py-8 md:px-6 md:py-12 relative overflow-hidden"
      >
        <div className="max-w-5xl mx-auto w-full relative z-10 px-2 md:px-0">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl md:text-5xl font-bold text-gold mb-8 text-center font-mono tracking-tight">About KARGA</h2>

            <div className="space-y-6 md:space-y-8">
              {/* Overview */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-8">
                <p className="text-lg md:text-xl text-gray-300 leading-relaxed mb-4 font-light">
                  KARGA combines <span className="text-gold font-semibold">Knowledge Graphs</span>, <span className="text-gold font-semibold">Retrieval Augmented Generation</span>, and <span className="text-gold font-semibold">Semantic Search</span> to provide AI-powered financial intelligence.
                </p>
                <p className="text-base md:text-lg text-gray-400 leading-relaxed font-light">
                  Ask questions in natural language, and GPT-4 generates precise database queries across interconnected financial data sources—no hallucinations, only real data.
                </p>
              </div>

              {/* Key Features Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <div className="text-3xl mb-3">📊</div>
                  <h3 className="text-lg font-semibold text-gold mb-2">Multi-Source Data</h3>
                  <p className="text-sm text-gray-400 font-light leading-relaxed">
                    S&P 500 stocks, government contracts, prediction markets, SEC filings, economic indicators, and commodities—all in one graph.
                  </p>
                </div>

                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <div className="text-3xl mb-3">🤖</div>
                  <h3 className="text-lg font-semibold text-gold mb-2">AI Query Generation</h3>
                  <p className="text-sm text-gray-400 font-light leading-relaxed">
                    GPT-4 converts your questions into optimized graph queries with semantic search, multi-hop traversals, and parallel execution.
                  </p>
                </div>

                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <div className="text-3xl mb-3">⚡</div>
                  <h3 className="text-lg font-semibold text-gold mb-2">Blazing Fast</h3>
                  <p className="text-sm text-gray-400 font-light leading-relaxed">
                    Query 2M+ documents with 50ms graph traversals. Parallel DB + web search delivers results in under 3 seconds.
                  </p>
                </div>
              </div>

              {/* Architecture Preview */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-gold mb-3 md:mb-4">How It Works</h3>
                <div className="space-y-2 md:space-y-3 text-xs md:text-sm text-gray-400 font-light">
                  <div className="flex items-start">
                    <span className="text-green-400 font-semibold mr-3 mt-1">1.</span>
                    <div>
                      <strong className="text-gray-300">Intent Detection:</strong> GPT-4 classifies your query (ticker vs. concept)
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-blue-400 font-semibold mr-3 mt-1">2.</span>
                    <div>
                      <strong className="text-gray-300">Query Planning:</strong> AI generates optimized AQL with semantic embeddings
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-purple-400 font-semibold mr-3 mt-1">3.</span>
                    <div>
                      <strong className="text-gray-300">Parallel Execution:</strong> Database query + web search run simultaneously
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-orange-400 font-semibold mr-3 mt-1">4.</span>
                    <div>
                      <strong className="text-gray-300">Synthesis:</strong> GPT-4 combines results with analysis and insights
                    </div>
                  </div>
                </div>
              </div>

              {/* CTA */}
              <div className="text-center">
                <a
                  href="/about"
                  className="inline-block px-6 md:px-8 py-3 md:py-4 bg-gold/20 border-2 border-gold/40 rounded-lg text-gold font-semibold hover:bg-gold/30 hover:border-gold/60 transition-all text-base md:text-lg"
                >
                  View Full Technical Deep-Dive →
                </a>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Scroll to Top Button */}
      <ScrollToTop />
    </div>
  )
}
