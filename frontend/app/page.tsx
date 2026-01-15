'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, useScroll, useTransform, useInView } from 'framer-motion'
import MarkdownRenderer from '@/components/MarkdownRenderer'
import ResultsTable from '@/components/ResultsTable'
import GraphVisualization from '@/components/GraphVisualization'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  results?: any[]
  useMarkdown?: boolean
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Welcome to GraphRAG! Ask me anything about financial markets, SEC filings, prediction markets, or run complex queries across our knowledge graph.',
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

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

  const collections = [
    { name: 'Company', count: 852, icon: '🏢', description: 'S&P 500 companies with fundamentals' },
    { name: 'MarketData', count: 2100000, icon: '📊', description: 'Daily OHLCV + technical indicators' },
    { name: 'Award', count: 500000, icon: '🏛️', description: 'Federal contracts with embeddings' },
    { name: 'sec_filings', count: 15000, icon: '📄', description: 'SEC filings with sentiment' },
    { name: 'sec_sentences', count: 890000, icon: '📝', description: 'Filing sentences with FinBERT scores' },
    { name: 'prediction_markets_polymarket', count: 12968, icon: '🎲', description: 'Polymarket prediction data' },
    { name: 'prediction_markets_kalshi', count: 5432, icon: '🎯', description: 'Kalshi event contracts' },
    { name: 'EconomicData', count: 8900, icon: '💹', description: 'Macro indicators & rates' },
  ]

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

    try {
      const { api } = await import('@/lib/api')
      const response = await api.executeQuery(currentInput)

      let resultText = `**Query Results:** ${response.count} results in ${response.execution_time.toFixed(2)}s\n\n`

      if (response.analysis) {
        resultText += response.analysis
      }

      if (response.follow_up_questions && response.follow_up_questions.length > 0) {
        resultText += '\n\n**Follow-up questions:**\n'
        response.follow_up_questions.forEach((q: string) => {
          resultText += `- ${q}\n`
        })
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: resultText,
        timestamp: new Date(),
        results: response.results,
        useMarkdown: true,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const suggestedQuestions = [
    'What do prediction markets say about the 2024 election outcome?',
    'Show me defense contracts related to AI and cybersecurity',
    'Which tech stocks have the highest institutional ownership?',
    'Find government contracts mentioning China or Taiwan',
    'What are the biggest bets on Polymarket right now?',
    'Show me companies with recent SEC filings mentioning recession',
  ]

  // Fetch collection data when selected
  useEffect(() => {
    const fetchCollectionData = async () => {
      if (!selectedCollection) {
        setCollectionData([])
        setSearchFilter('')
        return
      }

      // Reset search when switching collections
      setSearchFilter('')
      setIsLoadingData(true)
      try {
        const { api } = await import('@/lib/api')
        const response = await api.browseCollection(selectedCollection, 100)
        setCollectionData(response.data || [])
      } catch (error) {
        console.error('Failed to fetch collection data:', error)
        setCollectionData([])
      } finally {
        setIsLoadingData(false)
      }
    }

    fetchCollectionData()
  }, [selectedCollection])

  // Filter data based on search
  const filteredData = collectionData.filter((item) => {
    if (!searchFilter) return true
    const searchLower = searchFilter.toLowerCase()
    return Object.values(item).some((value) =>
      String(value).toLowerCase().includes(searchLower)
    )
  })

  return (
    <div className="relative">
      {/* Hero Section */}
      <motion.section
        ref={heroRef}
        style={{ opacity, scale }}
        className="min-h-screen flex flex-col items-center justify-center px-6 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
          className="text-center z-10"
        >
          <h1 className="text-6xl md:text-8xl font-bold text-gold mb-6 tracking-tight">
            GraphRAG
          </h1>
          <p className="text-2xl md:text-3xl text-gray-300 mb-4">
            Financial Intelligence Powered by Knowledge Graphs
          </p>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            Ask natural language questions. Get insights from millions of connected data points.
          </p>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
          className="absolute bottom-10"
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

      {/* Why Graphs Section */}
      <section
        ref={whyGraphsRef}
        className="min-h-screen flex flex-col items-center justify-center px-6 py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity1 }}
        />
        <div className="max-w-6xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isWhyGraphsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl font-bold text-gold mb-6">Why Knowledge Graphs?</h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Traditional databases see data in silos. Graphs see connections.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: 'Connected Data',
                description: 'Every company links to market data, government contracts, SEC filings, and prediction markets',
                delay: 0.2,
              },
              {
                title: 'Semantic Search',
                description: 'Find contracts mentioning "Iran" or "cybersecurity" using AI embeddings, not just keywords',
                delay: 0.4,
              },
              {
                title: 'Hybrid Intelligence',
                description: 'Combine historical database queries with real-time web search for complete context',
                delay: 0.6,
              },
            ].map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 50 }}
                animate={isWhyGraphsInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.8, delay: item.delay }}
                className="bg-dark-800 border border-gold/20 rounded-lg p-8 hover:border-gold/40 transition-all"
              >
                <h3 className="text-2xl font-semibold text-gold mb-4">{item.title}</h3>
                <p className="text-gray-400 leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Chat Interface Section */}
      <section className="min-h-screen flex items-center justify-center px-6 py-12 relative overflow-hidden">
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity2 }}
        />
        <div className="container mx-auto max-w-6xl relative z-10">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.8 }}
            className="mb-8 text-center"
          >
            <h2 className="text-4xl font-bold text-gold mb-2">AI Query Interface</h2>
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
            <div className="h-[500px] overflow-y-auto p-6 space-y-4">
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
                        <div className="text-sm mb-2 leading-relaxed">
                          {message.useMarkdown ? (
                            <MarkdownRenderer content={message.content} />
                          ) : (
                            message.content
                          )}
                        </div>
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
                        <div className="text-xs text-gray-600">
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
            <div className="border-t border-gold/20 p-4">
              <form onSubmit={handleSubmit} className="flex space-x-3">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit(e)
                    }
                  }}
                  placeholder="Ask a question about markets, companies, or contracts... (Shift+Enter for new line)"
                  rows={4}
                  className="flex-1 bg-dark-700 border border-gold/30 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20 resize-y min-h-[100px] max-h-[300px]"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="px-6 py-3 bg-gold/20 border border-gold/40 rounded-lg text-gold hover:bg-gold/30 hover:border-gold/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold self-end"
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
            <h3 className="text-lg font-semibold text-gold mb-4">Suggested Questions</h3>
            <div className="grid grid-cols-2 gap-3">
              {suggestedQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(question)}
                  className="bg-dark-800 border border-gold/20 rounded-lg p-4 text-left text-sm text-gray-400 hover:border-gold/40 hover:text-gold transition-all"
                >
                  {question}
                </button>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Graph Architecture Visualization */}
      <section
        ref={graphVizRef}
        className="min-h-screen flex flex-col items-center justify-center px-6 py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-6xl mx-auto w-full relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isGraphVizInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-5xl font-bold text-gold mb-6">Graph Architecture</h2>
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

      {/* Data Universe - Collections Browser */}
      <section
        ref={statsRef}
        className="min-h-screen flex flex-col justify-center px-6 py-12 relative overflow-hidden"
      >
        {/* Animated background grid */}
        <motion.div
          className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"
          style={{ opacity: gridOpacity3 }}
        />
        <div className="max-w-6xl mx-auto w-full relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={isStatsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-5xl font-bold text-gold mb-6">Data Universe</h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
              Explore all database collections and their connections
            </p>

            {/* Top Stats */}
            <div className="grid grid-cols-4 gap-4 mb-8">
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
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
                <div className="text-gray-400 text-sm mb-1">Database</div>
                <div className="text-xl font-bold text-gold">QUANT_v3</div>
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
                  <div className="text-3xl mb-2">{collection.icon}</div>
                  <div className="text-gold font-semibold mb-1 text-sm">{collection.name}</div>
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
                  <h3 className="text-2xl font-bold text-gold">{selectedCollection}</h3>
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
                  <div className="text-xs text-gray-400 mb-1">Showing</div>
                  <div className="text-2xl font-semibold text-gold">
                    {filteredData.length} / 100
                  </div>
                </div>
                <div className="bg-dark-700 border border-gold/20 rounded p-4">
                  <div className="text-xs text-gray-400 mb-1">Fields</div>
                  <div className="text-2xl font-semibold text-gold">
                    {collectionData.length > 0 ? Object.keys(collectionData[0]).length : '—'}
                  </div>
                </div>
              </div>

              {/* Search Filter */}
              <div className="mb-4 relative">
                <input
                  type="text"
                  placeholder="Search across all fields..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full bg-dark-700 border border-gold/30 rounded-lg px-4 py-3 pr-10 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20"
                />
                {searchFilter && (
                  <button
                    onClick={() => setSearchFilter('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gold transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>

              {/* Data Table */}
              <div className="bg-dark-700 border border-gold/20 rounded-lg overflow-hidden">
                {isLoadingData ? (
                  <div className="p-8 text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gold"></div>
                    <p className="text-gray-400 mt-4">Loading data...</p>
                  </div>
                ) : filteredData.length > 0 ? (
                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-dark-800 sticky top-0">
                        <tr>
                          {Object.keys(filteredData[0])
                            .filter(key => !key.startsWith('_') || key === '_key')
                            .map((key) => (
                              <th
                                key={key}
                                className="px-4 py-3 text-left text-xs font-semibold text-gold uppercase tracking-wider border-b border-gold/20"
                              >
                                {key}
                              </th>
                            ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gold/10">
                        {filteredData.slice(0, 50).map((row, idx) => (
                          <tr key={idx} className="hover:bg-dark-800/50 transition-colors">
                            {Object.entries(row)
                              .filter(([key]) => !key.startsWith('_') || key === '_key')
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
                ) : (
                  <div className="p-8 text-center text-gray-400">
                    {searchFilter ? 'No matching records found' : 'No data available'}
                  </div>
                )}
              </div>

              <div className="mt-4 bg-dark-700 border border-gold/20 rounded-lg p-4">
                <div className="text-sm text-gray-300">
                  💡 Click on the <span className="text-gold font-semibold">{selectedCollection}</span> node in the Graph Architecture section above to see full schema details, sample data, and example AQL queries
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </section>
    </div>
  )
}
