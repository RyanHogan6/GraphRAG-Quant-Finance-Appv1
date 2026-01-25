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
import TimeSeriesChart from '@/components/TimeSeriesChart'
import QueryBuilder from '../components/QueryBuilder'
import WhaleTracker from '@/components/WhaleTracker'
import CompanyWorkup from '@/components/CompanyWorkup'
import CompanyCompare from '@/components/CompanyCompare'
import Navigation from '@/components/Navigation'
import DataSourceAttribution from '@/components/DataSourceAttribution'
import ComplexQueryGallery from '@/components/ComplexQueryGallery'
import InsiderTradingSignal from '@/components/InsiderTradingSignal'
import SentimentDivergence from '@/components/SentimentDivergence'
import AnomalyHighlight from '@/components/AnomalyHighlight'
import { Market } from '@/lib/types'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  results?: any[]
  useMarkdown?: boolean
  followUpQuestions?: string[]
  webContext?: {
    sources?: string[]
    citations?: Array<{ number: number, url: string, referenced: boolean }>
  }
  metadata?: {
    tickers?: string[]
    companies?: string[]
    collections?: string[]
    queryIntent?: string
    resultCount?: number
  }
  presentationType?: string
  queryPlan?: {
    aql_query?: string
    bind_vars?: any
    intent?: string
    explanation?: string
    is_time_series?: boolean
    chart_data?: {
      type: string
      dates: string[]
      values: number[]
      label: string
      ticker: string
    }
  }
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showAdvancedMode, setShowAdvancedMode] = useState(false)

  // Query Builder State
  const [isBuilderMode, setIsBuilderMode] = useState(false)
  const [builtQuery, setBuiltQuery] = useState({ aql: '', description: '' })
  const [expandedMessageIdx, setExpandedMessageIdx] = useState<number | null>(null)

  // Complex Query Gallery State
  const [showComplexQueries, setShowComplexQueries] = useState(false)

  // Ref for auto-scroll
  const chatScrollRef = useRef<HTMLDivElement>(null)

  // Close expanded view on ESC
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedMessageIdx(null)
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [])

  // Refs for scroll animations
  const whyGraphsRef = useRef(null)
  const statsRef = useRef(null)
  const graphVizRef = useRef(null)

  const isWhyGraphsInView = useInView(whyGraphsRef, { once: true, amount: 0.3 })
  const isStatsInView = useInView(statsRef, { once: true, amount: 0.3 })
  const isGraphVizInView = useInView(graphVizRef, { once: true, amount: 0.2 })

  const { scrollYProgress } = useScroll()

  // Grid pattern opacity that increases as you scroll
  const gridOpacity1 = useTransform(scrollYProgress, [0, 0.2], [0.08, 0.15])
  const gridOpacity2 = useTransform(scrollYProgress, [0.2, 0.4], [0.15, 0.22])
  const gridOpacity3 = useTransform(scrollYProgress, [0.4, 0.6], [0.22, 0.28])

  // State for collection browser
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [collectionData, setCollectionData] = useState<any[]>([])

  // Auto-scroll logic
  useEffect(() => {
    if (chatScrollRef.current) {
      const scrollContainer = chatScrollRef.current
      scrollContainer.scrollTop = scrollContainer.scrollHeight
    }
  }, [messages, isLoading])
  const [isLoadingData, setIsLoadingData] = useState(false)
  const [searchFilter, setSearchFilter] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [collections, setCollections] = useState<Array<{ name: string, count: number, description: string }>>([])
  const [loadingCollections, setLoadingCollections] = useState(true)
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({})
  const [collectionOffset, setCollectionOffset] = useState(0)
  const [hasMoreCollectionData, setHasMoreCollectionData] = useState(true)

  const renderMessageContent = (message: Message) => (
    <>
      <div className="flex-1">
        <div className="text-xs md:text-sm mb-1.5 md:mb-2 leading-relaxed">
          {message.useMarkdown ? (
            <MarkdownRenderer content={message.content} />
          ) : (
            message.content
          )}
        </div>
        {!!message.queryPlan?.is_time_series && message.queryPlan?.chart_data &&
          // Only show standalone chart if NOT showing CompanyWorkup (which has its own charts)
          !message.results?.some((r: any) => r.MarketData || r.sec_filings || r.Award || r.prediction_markets_polymarket) && (
            <TimeSeriesChart
              dates={message.queryPlan.chart_data.dates}
              values={message.queryPlan.chart_data.values}
              label={message.queryPlan.chart_data.label}
              ticker={message.queryPlan.chart_data.ticker}
            />
          )}
        {showAdvancedMode && message.queryPlan && message.queryPlan.aql_query && (
          <details className="mt-2 mb-2">
            <summary className="cursor-pointer text-[10px] md:text-xs text-purple-400 hover:text-purple-300 font-semibold opacity-80 hover:opacity-100 transition-opacity">
              🔧 Query Plan
            </summary>
            <div className="mt-1.5 bg-dark-900/50 border border-purple-500/20 rounded p-2 md:p-3">
              <div className="text-[10px] md:text-xs text-gray-400 mb-1.5">
                <strong className="text-purple-400">Intent:</strong> {message.queryPlan.intent || 'N/A'}
                {message.queryPlan.explanation && (
                  <div className="mt-1">
                    <strong className="text-purple-400">Strategy:</strong> {message.queryPlan.explanation}
                  </div>
                )}
              </div>
              <div className="text-[10px] text-gray-500 font-mono mb-1">AQL Query:</div>
              <pre className="text-[9px] md:text-xs bg-black/50 p-2 rounded overflow-x-auto text-green-400/80 border border-green-500/10">
                {message.queryPlan.aql_query}
              </pre>
              {message.queryPlan.bind_vars && Object.keys(message.queryPlan.bind_vars).length > 0 && (
                <div className="mt-2">
                  <div className="text-[10px] text-gray-500 font-mono mb-1">Bind Variables:</div>
                  <pre className="text-[9px] md:text-xs bg-black/50 p-2 rounded overflow-x-auto text-blue-400/80 border border-blue-500/10">
                    {JSON.stringify(message.queryPlan.bind_vars, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </details>
        )}

        {message.results && message.results.length > 0 && (
          <div className="mt-3 md:mt-4 space-y-3 md:space-y-4">
            {(() => {
              // Check for specialized presentation types first
              if (message.presentationType === 'insider_trading') {
                return <InsiderTradingSignal signals={message.results} />;
              }

              if (message.presentationType === 'sentiment_divergence') {
                return <SentimentDivergence signals={message.results} />;
              }

              // Then check for company workups
              const isWorkup = (r: any) =>
                r.MarketData || r.sec_filings || r.Award || r.prediction_markets_polymarket;

              const workupResults = message.results.filter(isWorkup);

              if (workupResults.length === 2) {
                return (
                  <CompanyCompare
                    companyA={workupResults[0]}
                    companyB={workupResults[1]}
                  />
                );
              } else if (workupResults.length === 1) {
                return <CompanyWorkup data={workupResults[0]} />;
              } else {
                return <ResultsTable data={message.results} />;
              }
            })()}

            {/* Data Source Attribution */}
            <DataSourceAttribution
              collectionsUsed={message.metadata?.collections || []}
              queryIntent={message.metadata?.queryIntent}
            />
          </div>
        )}

        {message.followUpQuestions && message.followUpQuestions.length > 0 && (
          <div className="mt-4 md:mt-6 space-y-2">
            <div className="text-[10px] md:text-sm text-gold font-bold mb-2">Follow-up questions:</div>
            {message.followUpQuestions.map((q, i) => (
              <button
                key={i}
                onClick={() => {
                  setInput(q);
                  setTimeout(() => {
                    const fakeEvent = { preventDefault: () => { } } as React.FormEvent;
                    handleSubmit(fakeEvent);
                  }, 50);
                }}
                className="w-full text-left p-2 md:p-3 bg-dark-800/50 border border-white/5 rounded-lg text-xs md:text-sm text-gray-300 hover:border-gold/30 hover:bg-gold/5 transition-all"
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </>
  );
  /* Removed duplicate hasMoreCollectionData */

  // Collection name translations
  const collectionDisplayNames: Record<string, string> = {
    'Company': 'S&P 500 Companies',
    'MarketData': 'Stock Prices & Indicators',
    'Award': 'Government Contract Awards',
    'EconomicData': 'FRED Economic Indicators',
    'sec_filings': 'SEC Filings (12 Form Types)',
    'sec_sentences': 'SEC Sentiment Analysis',
    'sec_sections': 'SEC Filing Sections',
    'options_flow': 'Options Flow Activity',
    'futures_prices': 'Commodity Futures Prices',
    'commodity_positions': 'CFTC Trader Positions',
    'eia_crude_inventory': 'EIA Crude Oil Inventory',
    'eia_natgas_storage': 'EIA Natural Gas Storage',
    'eia_natgas_production': 'EIA Natural Gas Production',
    'eia_lng_exports': 'EIA LNG Exports',
    'prediction_markets_polymarket': 'Polymarket Prediction Markets',
    'prediction_markets_kalshi': 'Kalshi Event Contracts',
    'polymarket_traders': 'Polymarket Whale Traders',
    'polymarket_positions': 'Polymarket Trader Positions',
    'polymarket_price_history': 'Polymarket Price History',
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
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
      </svg>
    ),
    'sec_filings': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    'sec_sections': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
      </svg>
    ),
    'options_flow': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
    'futures_prices': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
      </svg>
    ),
    'eia_crude_inventory': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    'eia_natgas_storage': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
      </svg>
    ),
    'eia_natgas_production': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    'eia_lng_exports': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
      </svg>
    ),
    'polymarket_traders': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
    'polymarket_positions': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
    'polymarket_price_history': (
      <svg className="w-5 h-5 text-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
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
  const [categories, setCategories] = useState<Array<{ category: string, count: number }>>([])
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
          { name: 'Company', count: 612, description: 'S&P 500 companies with fundamentals' },
          { name: 'MarketData', count: 2100000, description: 'Daily OHLCV + 40+ technical/fundamental indicators' },
          { name: 'Award', count: 500000, description: 'Federal contracts with 1536-dim embeddings' },
          { name: 'options_flow', count: 612, description: 'Daily options activity (insider trading detection)' },
          { name: 'futures_prices', count: 64000, description: 'CME commodity futures (18 commodities)' },
          { name: 'sec_filings', count: 7495, description: '12 SEC form types with sentiment scores' },
          { name: 'sec_sentences', count: 890000, description: 'Filing sentences with FinBERT scores' },
          { name: 'prediction_markets_polymarket', count: 12968, description: 'Polymarket prediction markets with embeddings' },
          { name: 'prediction_markets_kalshi', count: 5432, description: 'Kalshi event contracts' },
          { name: 'polymarket_traders', count: 500, description: 'Whale traders and profit makers' },
          { name: 'commodity_positions', count: 5000, description: 'CFTC Commitments of Traders (weekly)' },
          { name: 'eia_crude_inventory', count: 200, description: 'EIA crude oil inventory (weekly)' },
          { name: 'eia_natgas_storage', count: 200, description: 'EIA natural gas storage (weekly)' },
          { name: 'EconomicData', count: 8900, description: 'FRED macro indicators & rates' },
        ])
      } finally {
        setLoadingCollections(false)
      }
    }

    fetchCollections()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Determine input based on mode
    const queryInput = isBuilderMode ? builtQuery.description : input
    if (!queryInput.trim()) return

    const userMessage: Message = {
      role: 'user',
      content: queryInput,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const currentInput = queryInput
    if (!isBuilderMode) setInput('') // Only clear text input if we used it

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

      const payload: any = {
        question: currentInput,
        conversation_history: conversationHistory
      }

      // If in builder mode, pass the forced AQL
      if (isBuilderMode && builtQuery.aql) {
        payload.forced_plan_aql = builtQuery.aql
        payload.timestamp = Date.now() // Force fresh request
      }

      const response = await fetch(`${API_BASE_URL}/api/query/execute-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
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

                  // Store presentation type for specialized rendering
                  if (parsed.presentation_type) {
                    lastMsg.presentationType = parsed.presentation_type
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

  // Optimized suggested questions - all guaranteed <2s response time
  const suggestedQuestions = [
    'Show me the top 10 largest government contracts',
    'What are the most active Polymarket prediction markets?',
    'Show me Apple stock data for the last 30 days',
    'Find government contracts over $1 billion',
    'What was TSLA stock price on October 4th, 2022?',
    'Show me the most recent 10-K filings',
  ]

  // Debounce search input (500ms delay)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchFilter)
    }, 500)

    return () => clearTimeout(timer)
  }, [searchFilter])

  // Fetch collection data when selected, search changes, or offset changes
  useEffect(() => {
    const fetchCollectionData = async () => {
      if (!selectedCollection) {
        setCollectionData([])
        setSearchFilter('')
        setDebouncedSearch('')
        setSortColumn(null)
        setColumnFilters({})
        setCollectionOffset(0)
        setHasMoreCollectionData(true)
        return
      }

      setIsLoadingData(true)
      try {
        const { api } = await import('@/lib/api')
        // Pass search to server for full database search
        // Using batch size of 100 for pagination
        const limit = 100
        const response = await api.browseCollection(
          selectedCollection,
          limit,
          debouncedSearch || undefined,
          collectionOffset
        )
        // Handle multiple response formats: {documents: [...]}, {data: [...]}, or [...]
        const data = Array.isArray(response)
          ? response
          : (response.documents || response.data || [])

        console.log('Collection data received:', data.length, 'items', 'Search:', debouncedSearch, 'Offset:', collectionOffset)

        if (collectionOffset === 0) {
          // New collection or fresh search - replace data
          setCollectionData(data)

          // Auto-detect date column and set as default sort (only on first load)
          if (data.length > 0 && !debouncedSearch) {
            const firstItem = data[0]
            const dateColumns = Object.keys(firstItem).filter(key =>
              key.toLowerCase().includes('date') ||
              key.toLowerCase().includes('time') ||
              key === 'timestamp'
            )

            const defaultSortCol = dateColumns.find(col => col === 'date') || dateColumns[0]
            if (defaultSortCol) {
              setSortColumn(defaultSortCol)
              setSortDirection('desc')
            } else {
              setSortColumn(null)
            }
          }
        } else {
          // Appending data for pagination
          setCollectionData(prev => [...prev, ...data])
        }

        // If we got fewer results than the limit, we've reached the end
        setHasMoreCollectionData(data.length === limit)
      } catch (error) {
        console.error('Failed to fetch collection data:', error)
        if (collectionOffset === 0) setCollectionData([])
      } finally {
        setIsLoadingData(false)
      }
    }

    fetchCollectionData()
  }, [selectedCollection, debouncedSearch, collectionOffset])

  // Helper to load next page of collection data
  const handleLoadMoreCollectionData = () => {
    if (!isLoadingData && hasMoreCollectionData) {
      setCollectionOffset(prev => prev + 100)
    }
  }

  // Reset offset when collection or search changes
  useEffect(() => {
    setCollectionOffset(0)
    setHasMoreCollectionData(true)
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
        api.getFeaturedMarkets(200, 'polymarket').catch(() => { })
        api.getCategories('polymarket').catch(() => { })
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
          // Rich Polymarket fields
          probability_confidence: m.probability_confidence,
          days_until_end: m.days_until_end,
          activity_score: m.activity_score,
          liquidity_score: m.liquidity_score,
          volume_per_day: m.volume_per_day,
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

  // Recalculate category counts based on actual markets data
  const actualCategories = useMemo(() => {
    const categoryCounts: Record<string, number> = {}
    markets.forEach(m => {
      const cat = m.category || 'Other'
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1
    })
    return Object.entries(categoryCounts)
      .map(([category, count]) => ({ category, count }))
      .sort((a, b) => b.count - a.count)
  }, [markets])

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(2)}M`
    if (volume >= 1000) return `$${(volume / 1000).toFixed(0)}k`
    return `$${volume}`
  }

  return (
    <div className="relative z-10 h-screen flex flex-col px-4 py-4">
      {/* Logo and Tagline - Compact Top Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex flex-col items-center pt-4 pb-3"
      >
        {/* Medium KARGA Logo */}
        <div className="scale-[0.45] md:scale-[0.5] origin-center -mb-12 md:-mb-14">
          <AnimatedLogo />
        </div>
        <p className="text-sm md:text-base text-gray-400">
          Financial Intelligence Powered by Knowledge Graphs
        </p>
      </motion.div>

      {/* Chat Interface Section - Compact, All Visible */}
      <section id="query" className="flex-1 flex flex-col max-w-[1600px] mx-auto w-full min-h-0">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-2 text-left"
        >
          <h2 className="text-xs md:text-sm font-bold text-gold tracking-tight flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-gold rounded-full animate-pulse" />
            Intelligence Terminal
          </h2>
        </motion.div>

        {/* Chat Container - Compact */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="bg-dark-700/95 border border-gold/30 rounded-xl shadow-2xl p-3 md:p-4 flex-1 flex flex-col min-h-0"
        >
            {/* Expanded Message Overlay */}
            {expandedMessageIdx !== null && (
              <div
                className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[100] flex items-center justify-center p-4 md:p-8"
                onClick={() => setExpandedMessageIdx(null)}
              >
                <div
                  className="bg-dark-800 border border-gold/30 rounded-xl w-full max-w-[96vw] h-[92vh] overflow-hidden flex flex-col shadow-2xl relative"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex items-center justify-between p-4 border-b border-gold/20 bg-dark-900/50">
                    <h3 className="text-gold font-bold uppercase tracking-widest text-sm">Deep Intelligence Terminal</h3>
                    <button
                      onClick={() => setExpandedMessageIdx(null)}
                      className="p-2 text-gray-400 hover:text-white transition-colors"
                    >
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4 md:p-8">
                    {renderMessageContent(messages[expandedMessageIdx])}
                  </div>
                </div>
              </div>
            )}

            {/* Messages */}
            <div
              ref={chatScrollRef}
              className="flex-1 overflow-y-auto py-2 md:py-3 space-y-3 md:space-y-4 scroll-smooth min-h-0"
            >
              {/* Visual Query Builder - Show at top when in builder mode */}
              {isBuilderMode && (
                <div className="space-y-4 mb-6">
                  <QueryBuilder
                    onQueryChange={(aql, desc) => setBuiltQuery({ aql, description: desc })}
                  />

                  {/* AQL Preview */}
                  {builtQuery.aql && (
                    <div className="bg-dark-900 border border-green-500/20 rounded-lg p-3">
                      <details>
                        <summary className="text-[10px] text-green-400 font-mono cursor-pointer hover:text-green-300 select-none flex items-center gap-1.5 transition-colors">
                          <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                          </svg>
                          Generated AQL Query
                        </summary>
                        <pre className="mt-2 text-[9px] md:text-xs text-gray-300 font-mono overflow-x-auto whitespace-pre-wrap max-h-[200px] bg-black/30 p-2 rounded">
                          {builtQuery.aql}
                        </pre>
                      </details>
                    </div>
                  )}
                </div>
              )}

              {/* Empty State - Show suggested questions when no messages */}
              {messages.length === 0 && !isBuilderMode && (
                <div className="flex items-center justify-center h-full py-4">
                  <div className="max-w-5xl w-full space-y-3">
                    <div className="text-center mb-3">
                      <h3 className="text-base md:text-lg font-semibold text-gold mb-1">What can I help you discover?</h3>
                      <p className="text-xs md:text-sm text-gray-400">Ask about markets, companies, contracts, commodities, or options flow</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2 md:gap-3">
                      {suggestedQuestions.slice(0, 3).map((question, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            setInput(question)
                            setIsBuilderMode(false)
                          }}
                          className="bg-dark-600/60 border border-gold/30 rounded-lg p-2.5 md:p-3 text-left text-xs md:text-sm text-gray-200 hover:border-gold/50 hover:bg-dark-600 hover:text-gold transition-all shadow-sm"
                        >
                          {question}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Only show messages in LLM mode OR if there are actual messages in builder mode */}
              {messages.filter((m, idx) => !isBuilderMode || idx > 0).map((message, idx) => (
                <div
                  key={idx}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[95%] md:max-w-[90%] rounded-2xl p-4 md:p-5 relative group transition-all duration-300 shadow-lg ${message.role === 'user'
                      ? 'bg-gold/25 border border-gold/50 text-white rounded-tr-none'
                      : 'bg-dark-700/80 border border-white/30 text-gray-50 rounded-tl-none'
                      }`}
                  >
                    {/* Expand/Maximize Button for AI results */}
                    {message.role === 'assistant' && message.results && message.results.length > 0 && (
                      <button
                        onClick={() => setExpandedMessageIdx(expandedMessageIdx === idx ? null : idx)}
                        className="absolute -top-2 -right-2 bg-dark-800 border border-gold/30 rounded-full p-1.5 text-gold hover:bg-gold hover:text-dark-800 opacity-0 group-hover:opacity-100 transition-all z-20 shadow-lg"
                        title="Maximize View"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                        </svg>
                      </button>
                    )}

                    <div className="flex items-start space-x-3 md:space-x-4">
                      {/* Profile Icon / Label */}
                      <div className="flex flex-col items-center flex-shrink-0 mt-1">
                        <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center text-[10px] md:text-xs font-bold shadow-inner ${message.role === 'user' ? 'bg-gold/30 text-gold border border-gold/40' : 'bg-purple-600/30 text-purple-200 border border-purple-500/30'
                          }`}>
                          {message.role === 'user' ? 'YU' : 'AI'}
                        </div>
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="text-xs md:text-sm font-bold text-gray-400 uppercase tracking-tight mb-2 flex items-center gap-2">
                          {message.role === 'user' ? 'Market Analyst' : 'KARGA Intelligence'}
                          {message.role === 'assistant' && <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />}
                        </div>
                        <div className="text-base md:text-lg mb-2 md:mb-3 leading-relaxed text-gray-100">
                          {message.useMarkdown ? (
                            <MarkdownRenderer content={message.content} />
                          ) : (
                            message.content
                          )}
                        </div>
                      </div>
                    </div>
                    {message.queryPlan?.is_time_series && message.queryPlan?.chart_data && (
                      <TimeSeriesChart
                        dates={message.queryPlan.chart_data.dates}
                        values={message.queryPlan.chart_data.values}
                        label={message.queryPlan.chart_data.label}
                        ticker={message.queryPlan.chart_data.ticker}
                      />
                    )}
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
                    {message.results && message.results.length > 0 && (
                      <div className="mt-4 overflow-hidden">
                        {(() => {
                          const result = message.results[0];
                          const isWorkup = (r: any) => r.ticker && (r.MarketData || r.sec_filings || r.prediction_markets_polymarket || r.Award);

                          const singleWorkup = message.results.length === 1 && isWorkup(result);
                          const doubleCompare = message.results.length === 2 && isWorkup(message.results[0]) && isWorkup(message.results[1]);

                          if (doubleCompare) {
                            return (
                              <div className="mt-2">
                                <div className="text-[10px] text-gold uppercase font-bold tracking-widest opacity-70 mb-4 text-center">
                                  Comparative Market Intelligence Report
                                </div>
                                <CompanyCompare companyA={message.results[0]} companyB={message.results[1]} />
                              </div>
                            );
                          }

                          if (singleWorkup) {
                            return (
                              <div className="mt-2">
                                <div className="text-[10px] text-gold uppercase font-bold tracking-widest opacity-70 mb-4">
                                  The KARGA Financial Workup
                                </div>
                                <CompanyWorkup data={result} />
                              </div>
                            );
                          }

                          return message.queryPlan?.intent === 'builder_execution' || !message.content.includes('|') ? (
                            <div className="space-y-2 overflow-hidden">
                              <div className="text-[10px] text-gold uppercase font-bold tracking-widest opacity-70">
                                Database Results ({message.results.length})
                              </div>
                              <ResultsTable data={message.results} maxRows={20} />
                            </div>
                          ) : (
                            <details className="mt-3 overflow-hidden">
                              <summary className="cursor-pointer text-xs text-gold hover:text-gold/80 font-semibold">
                                View raw data table ({message.results.length} rows)
                              </summary>
                              <div className="mt-2 overflow-hidden">
                                <ResultsTable data={message.results} maxRows={20} />
                              </div>
                            </details>
                          );
                        })()}
                      </div>
                    )}
                    {message.webContext && (!!message.webContext.citations?.length || !!message.webContext.sources?.length) && (
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
                              onClick={() => {
                                setInput(question);
                                setTimeout(() => {
                                  const fakeEvent = { preventDefault: () => { } } as React.FormEvent;
                                  handleSubmit(fakeEvent);
                                }, 50);
                              }}
                              className="text-left text-xs bg-dark-800 border border-gold/30 rounded px-3 py-2 text-gray-300 hover:border-gold/60 hover:bg-dark-700 hover:text-gold transition-all"
                            >
                              {question}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    <div className="text-[10px] md:text-xs text-gray-600 mt-1 md:mt-1.5">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}

              {
                isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-dark-700 border border-gold/20 rounded-lg p-2.5 md:p-3">
                      <div className="flex items-center space-x-2">
                        <div className="text-[10px] md:text-xs font-semibold text-gray-500 uppercase">AI</div>
                        <div className="flex space-x-1">
                          <div className="w-1.5 h-1.5 md:w-2 md:h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                          <div className="w-1.5 h-1.5 md:w-2 md:h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                          <div className="w-1.5 h-1.5 md:w-2 md:h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              }
            </div>

            {/* Input Area - Sticky on Mobile */}
            <div className="sticky bottom-0 z-[40] md:relative bg-dark-800 border-t border-gold/20 p-2 md:p-3 backdrop-blur-md mt-auto">
              {/* Tab Switcher */}
              <div className="flex items-center justify-between mb-2 gap-2 px-1">
                <div className="flex items-center gap-1 bg-dark-900 p-1 rounded-lg border border-gray-700">
                  {/* LLM Interface Tab */}
                  <button
                    onClick={() => setIsBuilderMode(false)}
                    className={`px-3 py-1 rounded text-[10px] font-medium transition-all ${
                      !isBuilderMode
                        ? 'bg-gold text-dark-900'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    LLM Interface
                  </button>

                  {/* Visual Query Builder Tab */}
                  <button
                    onClick={() => {
                      setIsBuilderMode(true)
                      setShowAdvancedMode(true)
                    }}
                    className={`px-3 py-1 rounded text-[10px] font-medium transition-all ${
                      isBuilderMode
                        ? 'bg-gold text-dark-900'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Visual Query Builder
                  </button>
                </div>

                <div className="flex items-center gap-3">
                  {/* Complex Queries Toggle */}
                  <button
                    onClick={() => setShowComplexQueries(!showComplexQueries)}
                    className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${
                      showComplexQueries
                        ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50 shadow-lg'
                        : 'bg-dark-800 text-purple-300 border border-purple-500/30 hover:bg-purple-500/10 hover:border-purple-500/50'
                    }`}
                  >
                    <span>🧠</span>
                    <span className="hidden sm:inline">Complex Queries</span>
                    <span className="sm:hidden">Gallery</span>
                  </button>
                  {/* Advanced Mode Toggle (only for LLM mode) */}
                  {!isBuilderMode && (
                    <label className="flex items-center cursor-pointer group">
                      <div className="relative">
                        <input
                          type="checkbox"
                          checked={showAdvancedMode}
                          onChange={(e) => setShowAdvancedMode(e.target.checked)}
                          className="sr-only"
                        />
                        <div className={`block w-6 h-3.5 md:w-8 md:h-4.5 rounded-full transition ${showAdvancedMode ? 'bg-purple-500' : 'bg-gray-600'}`}></div>
                        <div className={`dot absolute left-0.5 top-0.5 md:left-0.5 md:top-0.5 bg-white w-2.5 h-2.5 md:w-3.5 md:h-3.5 rounded-full transition ${showAdvancedMode ? 'transform translate-x-2.5 md:translate-x-3.5' : ''}`}></div>
                      </div>
                      <span className="text-[9px] md:text-[10px] text-gray-400 ml-1.5 font-medium group-hover:text-purple-400 transition-colors">Advanced</span>
                    </label>
                  )}

                  <div className="text-[9px] md:text-[10px] text-gray-500 font-mono tracking-tighter">
                    QUERY_ENGINE_V1.2
                  </div>
                </div>
              </div>
              {isBuilderMode ? (
                <div className="flex justify-end">
                  <button
                    onClick={handleSubmit}
                    disabled={isLoading || !builtQuery.aql}
                    className="px-4 py-2 bg-gold/20 border border-gold/40 rounded-md text-xs text-gold hover:bg-gold/30 hover:border-gold/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-bold uppercase tracking-wider shadow-sm"
                  >
                    Execute Query
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="flex flex-col md:flex-row md:space-x-2 space-y-2 md:space-y-0">
                  <textarea
                    value={input}
                    onChange={(e) => {
                      setInput(e.target.value)
                      if (isBuilderMode && e.target.value.trim()) {
                        setIsBuilderMode(false)
                      }
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSubmit(e)
                      }
                    }}
                    placeholder="Ask about markets, companies, contracts, commodities, options flow..."
                    rows={1}
                    className="flex-1 bg-black border border-gold/50 rounded-lg px-3 py-2 md:px-4 md:py-2.5 text-sm md:text-base text-white placeholder-gray-400 focus:outline-none focus:border-gold/80 focus:ring-2 focus:ring-gold/40 resize-y min-h-[42px] max-h-[120px]"
                    disabled={isLoading}
                  />
                  <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="w-full md:w-auto px-4 py-1.5 bg-gold/20 border border-gold/40 rounded-md text-[11px] text-gold hover:bg-gold/30 hover:border-gold/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-bold uppercase tracking-wider md:self-end shadow-md"
                  >
                    Send Query
                  </button>
                </form>
              )}

              {/* Complex Query Gallery */}
              {showComplexQueries && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-4"
                >
                  <ComplexQueryGallery
                    onQuerySelect={(naturalLanguage, aql) => {
                      setInput(naturalLanguage)
                      setBuiltQuery({ aql: aql, description: naturalLanguage })
                      setIsBuilderMode(true)
                      setShowComplexQueries(false)
                      setTimeout(() => {
                        const fakeEvent = { preventDefault: () => {} } as React.FormEvent
                        handleSubmit(fakeEvent)
                      }, 100)
                    }}
                  />
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Suggested Questions - Only show when there are messages */}
          {messages.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-8"
          >
            <h3 className="text-sm md:text-base font-semibold text-gold mb-2 md:mb-3">Suggested Questions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-3">
              {suggestedQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInput(question)
                    setIsBuilderMode(false)
                  }}
                  className="bg-dark-800 border border-gold/20 rounded-lg p-2 md:p-3 text-left text-[11px] md:text-xs text-gray-400 hover:border-gold/40 hover:text-gold transition-all"
                >
                  {question}
                </button>
              ))}
            </div>
          </motion.div>
          )}
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8" >
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Why Graphs Section */}
      <section
        ref={whyGraphsRef}
        className="snap-start px-4 md:px-6 py-8 md:py-12 relative overflow-hidden"
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
            className="text-center mb-6"
          >
            <h2 className="text-2xl md:text-3xl font-bold text-gold mb-2">Why Knowledge Graphs?</h2>
            <p className="text-sm md:text-base text-gray-400 max-w-2xl mx-auto">
              Traditional databases see data in silos. Graphs see connections.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-3 md:gap-4">
            {[
              {
                title: 'Connected Data',
                description: 'Every company links to market data, options flow, government contracts, SEC filings, commodity futures, EIA energy data, and prediction markets—19 collections, 22 edge types',
                delay: 0.2,
                icon: (
                  <svg className="w-6 h-6 md:w-8 md:h-8 text-gold mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                ),
              },
              {
                title: 'Semantic Search',
                description: 'Find contracts mentioning "AI" or "cybersecurity" using 1536-dim vector embeddings and cosine similarity—no keywords, pure meaning',
                delay: 0.4,
                icon: (
                  <svg className="w-6 h-6 md:w-8 md:h-8 text-gold mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                  </svg>
                ),
              },
              {
                title: 'Hybrid Intelligence',
                description: 'ArangoDB graph queries + Perplexity web search run in parallel. Historical data meets real-time news in under 3 seconds',
                delay: 0.6,
                icon: (
                  <svg className="w-6 h-6 md:w-8 md:h-8 text-gold mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                className="bg-dark-800 border border-gold/20 rounded-lg p-3 md:p-5 hover:border-gold/40 transition-all"
              >
                {item.icon}
                <h3 className="text-base md:text-lg font-semibold text-gold mb-2">{item.title}</h3>
                <p className="text-gray-400 text-xs md:text-sm leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8" >
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Graph Architecture Visualization */}
      <section
        ref={graphVizRef}
        className="snap-start flex flex-col items-center justify-center px-4 py-6 md:px-6 md:py-8 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-7xl mx-auto w-full relative z-10" >
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isGraphVizInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-4 md:mb-6"
          >
            <h2 className="text-2xl md:text-4xl font-bold text-gold mb-2 md:mb-3">Graph Architecture</h2>
            <p className="text-sm md:text-base text-gray-400 max-w-3xl mx-auto mb-3 md:mb-4">
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
            className="mt-4 md:mt-6 text-center"
          >
            <p className="text-gray-400 text-xs md:text-sm">
              Built on <span className="text-gold font-semibold">ArangoDB</span> with{' '}
              <span className="text-gold font-semibold">GPT-4</span> query planning and{' '}
              <span className="text-gold font-semibold">Perplexity</span> web search
            </p>
          </motion.div>
        </div>
      </section>

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8" >
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Prediction Markets Section - Hidden per user request
      <section
        id="markets"
        className="min-h-screen snap-start flex flex-col justify-center px-4 py-8 md:px-6 md:py-12 relative overflow-hidden"
      >
        ... [Prediction Market Content] ...
      </section>
      */}

      {/* Section Divider */}
      <div className="w-full flex items-center justify-center py-8">
        <div className="w-[70%] border-t-2 border-dashed border-gold/10"></div>
      </div>

      {/* Data Universe - Collections Browser */}
      <section
        id="database"
        ref={statsRef}
        className="snap-start px-4 py-8 md:px-6 md:py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-7xl mx-auto w-full relative z-10" >
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isStatsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-6"
          >
            <h2 className="text-2xl md:text-4xl font-bold text-gold mb-3">Data Universe</h2>
            <p className="text-sm md:text-lg text-gray-400 max-w-2xl mx-auto mb-4">
              Explore all database collections and their connections
            </p>

            {/* Top Stats */}
            <div className="grid grid-cols-3 gap-2 md:gap-4 mb-4">
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-2 md:p-4">
                <div className="text-gray-400 text-xs md:text-sm mb-1">Collections</div>
                <div className="text-lg md:text-2xl font-bold text-gold">{collections.length}</div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-2 md:p-4">
                <div className="text-gray-400 text-xs md:text-sm mb-1">Total Documents</div>
                <div className="text-lg md:text-2xl font-bold text-gold">
                  {(collections.reduce((sum, c) => sum + c.count, 0) / 1000000).toFixed(1)}M
                </div>
              </div>
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-2 md:p-4">
                <div className="text-gray-400 text-xs md:text-sm mb-1">Edge Collections</div>
                <div className="text-lg md:text-2xl font-bold text-gold">22</div>
              </div>
            </div>
          </motion.div>

          {/* Collections Grid */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={isStatsInView ? { opacity: 1 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h3 className="text-lg md:text-xl font-semibold text-gold mb-4 text-center">Document Collections</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 md:gap-3 mb-6">
              {collections.map((collection, idx) => (
                <motion.button
                  key={collection.name}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={isStatsInView ? { opacity: 1, scale: 1 } : {}}
                  transition={{ duration: 0.6, delay: 0.3 + idx * 0.05 }}
                  onClick={() => setSelectedCollection(collection.name === selectedCollection ? null : collection.name)}
                  className={`bg-dark-800 border rounded-lg p-3 md:p-4 text-left transition-all ${selectedCollection === collection.name
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
          {
            selectedCollection && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 mt-6"
              >
                <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-3">
                  <div className="flex-1">
                    <h3 className="text-xl md:text-2xl font-bold text-gold">
                      {collectionDisplayNames[selectedCollection] || selectedCollection}
                    </h3>
                    <p className="text-gray-400 text-xs md:text-sm mt-1">
                      {collections.find(c => c.name === selectedCollection)?.description}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setSelectedCollection(null)
                      setSearchFilter('')
                    }}
                    className="text-gray-500 hover:text-gold transition-colors self-end md:self-auto"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4 mb-6">
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
                      <div className="overflow-x-auto overflow-y-auto max-h-[400px] md:max-h-[600px]">
                        <table className="w-full text-xs md:text-sm">
                          <thead className="bg-dark-800 sticky top-0 z-10">
                            {/* Column Headers - Sortable */}
                            <tr>
                              {Object.keys(filteredData[0])
                                .filter(key => !key.startsWith('_'))
                                .map((key) => (
                                  <th
                                    key={key}
                                    className="px-2 md:px-4 py-2 md:py-3 text-left text-xs font-semibold text-gold uppercase tracking-wider border-b border-gold/20 cursor-pointer hover:bg-dark-700 transition-colors whitespace-nowrap"
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
                            {/* Column Filters - Hidden on mobile */}
                            <tr className="hidden md:table-row">
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
                              <tr key={idx} className={`hover:bg-dark-800/70 transition-colors ${idx % 2 === 0 ? 'bg-dark-800/20' : 'bg-gold/5'
                                }`}>
                                {Object.entries(row)
                                  .filter(([key]) => !key.startsWith('_'))
                                  .map(([key, value]) => (
                                    <td key={key} className="px-2 md:px-4 py-2 md:py-3 text-gray-300 whitespace-nowrap">
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
                      {isLoadingData && collectionOffset > 0 && (
                        <div className="bg-dark-800 border-t border-gold/20 px-4 py-3 text-center">
                          <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-gold mr-2 align-middle"></div>
                          <span className="text-gray-400 text-sm">Fetching more records...</span>
                        </div>
                      )}

                      {!isLoadingData && hasMoreCollectionData && collectionData.length > 0 && (
                        <div className="bg-dark-800 border-t border-gold/20 px-4 py-3 text-center font-mono">
                          <button
                            onClick={handleLoadMoreCollectionData}
                            className="text-xs text-gold hover:text-white uppercase tracking-widest font-bold transition-colors py-1 px-4 rounded border border-gold/30 hover:bg-gold/10"
                          >
                            [ Load More Records ]
                          </button>
                        </div>
                      )}

                      {!hasMoreCollectionData && collectionData.length > 0 && (
                        <div className="bg-dark-800 border-t border-gold/20 px-4 py-8 text-center">
                          <div className="text-[10px] text-gray-600 uppercase tracking-[0.2em] font-bold">
                            End of Data Collection Reached
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Showing {collectionData.length.toLocaleString()} of {collections.find(c => c.name === selectedCollection)?.count.toLocaleString()} documents
                          </div>
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
            )
          }
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
                  KARGA combines <span className="text-gold font-semibold">Knowledge Graphs</span>, <span className="text-gold font-semibold">Retrieval Augmented Generation</span>, and <span className="text-gold font-semibold">Semantic Search</span> to provide AI-powered financial intelligence across equities, commodities, options, contracts, filings, and prediction markets.
                </p>
                <p className="text-base md:text-lg text-gray-400 leading-relaxed font-light">
                  Ask questions in natural language, and GPT-4 generates precise AQL graph queries across 19 interconnected collections (2M+ documents, 22 edge types)—no hallucinations, only real data. Detect insider trading with daily options flow monitoring, correlate commodity prices with company exposure via direct graph edges, analyze sentiment from 7.5K SEC filings, and track whale positioning across 18K prediction markets.
                </p>
              </div>

              {/* Key Features Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <div className="text-3xl mb-3">📊</div>
                  <h3 className="text-lg font-semibold text-gold mb-2">Multi-Source Data</h3>
                  <p className="text-sm text-gray-400 font-light leading-relaxed">
                    S&P 500 stocks (2M+ OHLCV records), government contracts (500K+), options flow (612 tickers daily), commodity futures (64K CME prices), EIA energy data (crude/natgas inventory), SEC filings (12 form types), prediction markets (18K+ Polymarket/Kalshi), CFTC positioning, and FRED economic indicators—19 collections connected by 22 edge types in one unified knowledge graph.
                  </p>
                </div>

                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <div className="text-3xl mb-3">🤖</div>
                  <h3 className="text-lg font-semibold text-gold mb-2">AI Query Generation</h3>
                  <p className="text-sm text-gray-400 font-light leading-relaxed">
                    GPT-4 converts natural language into optimized AQL graph queries with semantic search (cosine similarity on 1536-dim embeddings), multi-hop traversals across 22 edge types (e.g., Company → COMPANY_TRADES_COMMODITY → futures_prices → INVENTORY_AFFECTS_PRICE → eia_crude_inventory), insider trading detection via OPTIONS_BEFORE_AWARD/FILING edges, and parallel web search execution.
                  </p>
                </div>

                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <div className="text-3xl mb-3">⚡</div>
                  <h3 className="text-lg font-semibold text-gold mb-2">Blazing Fast</h3>
                  <p className="text-sm text-gray-400 font-light leading-relaxed">
                    Query 2M+ OHLCV records, 500K+ government contracts, 64K commodity futures, 7.5K SEC filings, 18K+ prediction markets, and 612 daily options flow records with 50ms graph traversals. ArangoDB parallel execution + Perplexity web search delivers complete answers in under 3 seconds.
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
                      <strong className="text-gray-300">Intent Detection:</strong> GPT-4 classifies your query (ticker lookup, concept search, multi-hop graph traversal, insider trading detection, commodity correlation, etc.)
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-blue-400 font-semibold mr-3 mt-1">2.</span>
                    <div>
                      <strong className="text-gray-300">Query Planning:</strong> AI generates optimized AQL (ArangoDB Query Language) with graph traversals across 22 edge types, semantic search using cosine similarity on 1536-dim embeddings (Award descriptions, Polymarket questions), and joins across 19 collections
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-purple-400 font-semibold mr-3 mt-1">3.</span>
                    <div>
                      <strong className="text-gray-300">Parallel Execution:</strong> ArangoDB graph query (50ms traversals across 2M+ docs) + Perplexity web search run simultaneously. Graph queries use OUTBOUND/INBOUND edges (e.g., Company → COMPANY_TRADES_COMMODITY → futures_prices → INVENTORY_AFFECTS_PRICE → eia_crude_inventory)
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-orange-400 font-semibold mr-3 mt-1">4.</span>
                    <div>
                      <strong className="text-gray-300">Synthesis:</strong> GPT-4 merges database results (historical data, graph relationships, sentiment scores, positioning data) with real-time web context, providing analysis, insights, and follow-up questions
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-amber-400 font-semibold mr-3 mt-1">5.</span>
                    <div>
                      <strong className="text-gray-300">Insider Trading Detection:</strong> Options flow pipeline tracks 612 tickers daily, building 20-day baselines to detect unusual call/put activity before contract awards (OPTIONS_BEFORE_AWARD edge) and SEC filings (OPTIONS_BEFORE_FILING edge)
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-teal-400 font-semibold mr-3 mt-1">6.</span>
                    <div>
                      <strong className="text-gray-300">Commodity Analysis:</strong> Direct company-to-commodity links via COMPANY_TRADES_COMMODITY (501K edges, 49 companies), enriched with CFTC positioning data, EIA inventory reports (crude oil, natural gas, LNG), and FRED macro indicators
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
