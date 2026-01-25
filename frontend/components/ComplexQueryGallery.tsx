'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ComplexQuery {
  title: string
  description: string
  insight: string
  query: string
  category: string
  collections: string[]
  edges: string[]
  difficulty: 'intermediate' | 'advanced' | 'expert'
  icon: string
}

const COMPLEX_QUERIES: ComplexQuery[] = [
  // INSIDER TRADING DETECTION
  {
    title: 'Insider Trading Signals',
    description: 'Find unusual options activity 1-30 days before major SEC filings (8-K)',
    insight: 'Detects potential insider trading by correlating options sweeps with subsequent material events',
    query: 'Find companies with unusual call or put volume before 8-K filings with significant sentiment changes',
    category: 'Insider Trading',
    collections: ['options_flow', 'sec_filings', 'Company'],
    edges: ['OPTIONS_BEFORE_FILING', 'COMPANY_HAS_OPTIONS'],
    difficulty: 'expert',
    icon: '🕵️'
  },
  {
    title: 'Pre-Award Options Activity',
    description: 'Unusual options buying before government contract announcements',
    insight: 'Defense contractors often see options activity spike before major award announcements',
    query: 'Show me defense contractors with unusual options activity followed by large government contracts within 30 days',
    category: 'Insider Trading',
    collections: ['options_flow', 'Award', 'Company'],
    edges: ['OPTIONS_BEFORE_AWARD', 'HAS_AWARD', 'COMPANY_HAS_OPTIONS'],
    difficulty: 'advanced',
    icon: '🏛️'
  },

  // COMMODITY-STOCK CORRELATION
  {
    title: 'Energy Stocks vs Crude Inventory',
    description: 'How energy companies react to weekly EIA crude oil inventory changes',
    insight: 'Large inventory builds typically pressure crude prices, affecting energy stock valuations',
    query: 'Show me energy sector stocks during weeks with crude oil inventory builds above 5 million barrels',
    category: 'Commodity Correlation',
    collections: ['Company', 'MarketData', 'futures_prices', 'eia_crude_inventory'],
    edges: ['HAS_MARKETDATA', 'INVENTORY_AFFECTS_PRICE', 'COMPANY_TRADES_COMMODITY'],
    difficulty: 'advanced',
    icon: '🛢️'
  },
  {
    title: 'Mining Stocks vs Metal Prices',
    description: 'Correlation between gold/copper futures and mining company stock performance',
    insight: 'Mining stocks often lag commodity price movements by 1-3 days due to market inefficiency',
    query: 'Compare gold and copper mining companies to futures prices over the last 90 days, show divergences',
    category: 'Commodity Correlation',
    collections: ['Company', 'MarketData', 'futures_prices', 'commodity_positions'],
    edges: ['COMPANY_TRADES_COMMODITY', 'HAS_MARKETDATA', 'HAS_COMMODITY_POSITION'],
    difficulty: 'advanced',
    icon: '⛏️'
  },
  {
    title: 'Natural Gas Storage Anomalies',
    description: 'Natural gas prices when storage deviates significantly from 5-year average',
    insight: 'Storage levels >20% above/below average historically predict price reversions',
    query: 'Show me natural gas prices and energy stocks when storage is 20%+ above or below 5-year average',
    category: 'Commodity Correlation',
    collections: ['futures_prices', 'eia_natgas_storage', 'Company', 'MarketData'],
    edges: ['STORAGE_AFFECTS_PRICE', 'COMPANY_TRADES_COMMODITY'],
    difficulty: 'intermediate',
    icon: '🔥'
  },

  // PREDICTION MARKET ALPHA
  {
    title: 'Prediction Market vs Reality',
    description: 'Compare Polymarket sentiment to actual stock performance',
    insight: 'Markets often overprice high-probability outcomes - find mispriced opportunities',
    query: 'Show me tech stocks where Polymarket sentiment is >70% bullish but stock is down 5%+ this month',
    category: 'Prediction Markets',
    collections: ['prediction_markets_polymarket', 'Company', 'MarketData'],
    edges: ['market_mentions_company_polymarket', 'HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '🎲'
  },
  {
    title: 'Whale Trader Positioning',
    description: 'What markets are the most profitable whale traders betting on?',
    insight: 'Top 1% traders historically outperform market consensus by 12-18%',
    query: 'Show me Polymarket positions from traders with >$100k profit, filter for markets closing within 30 days',
    category: 'Prediction Markets',
    collections: ['polymarket_traders', 'polymarket_positions', 'prediction_markets_polymarket'],
    edges: ['trader_has_position', 'position_in_market'],
    difficulty: 'intermediate',
    icon: '🐋'
  },
  {
    title: 'Cross-Platform Sentiment Arbitrage',
    description: 'Find divergence between Polymarket and Kalshi on same events',
    insight: 'Same event priced differently on two platforms = arbitrage opportunity',
    query: 'Compare Polymarket and Kalshi probabilities for similar questions about tech companies, show >10% divergence',
    category: 'Prediction Markets',
    collections: ['prediction_markets_polymarket', 'prediction_markets_kalshi', 'Company'],
    edges: ['market_mentions_company_polymarket', 'market_mentions_company_kalshi'],
    difficulty: 'advanced',
    icon: '⚖️'
  },

  // MULTI-SOURCE SENTIMENT
  {
    title: 'Triple Sentiment Confluence',
    description: 'Stocks with aligned bullish signals across SEC filings, options flow, AND prediction markets',
    insight: 'When all 3 data sources align, conviction is 3x higher than single-source signals',
    query: 'Find companies with positive SEC sentiment (>0.1), unusual call buying, and >60% bullish prediction market probability',
    category: 'Multi-Source Sentiment',
    collections: ['sec_filings', 'options_flow', 'prediction_markets_polymarket', 'Company'],
    edges: ['HAS_FILING', 'COMPANY_HAS_OPTIONS', 'market_mentions_company_polymarket'],
    difficulty: 'expert',
    icon: '🎯'
  },
  {
    title: 'Sentiment Divergence Alert',
    description: 'Find contradictory signals: bearish SEC filings but bullish options activity',
    insight: 'Divergence often precedes major price moves - either the filing is wrong or options traders know something',
    query: 'Show me companies where SEC sentiment is negative (<-0.1) but there is unusual call buying in the last 7 days',
    category: 'Multi-Source Sentiment',
    collections: ['sec_filings', 'options_flow', 'Company'],
    edges: ['HAS_FILING', 'COMPANY_HAS_OPTIONS'],
    difficulty: 'advanced',
    icon: '⚡'
  },

  // GOVERNMENT CONTRACT ALPHA
  {
    title: 'Repeat Contract Winners',
    description: 'Defense contractors with increasing award amounts year-over-year',
    insight: 'Consistent award growth indicates strong relationships with procurement offices',
    query: 'Find defense contractors whose average award amount increased >20% in the last 2 fiscal years',
    category: 'Government Contracts',
    collections: ['Award', 'Company', 'MarketData'],
    edges: ['HAS_AWARD', 'HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '📈'
  },
  {
    title: 'Contract Award Stock Performance',
    description: 'How stocks perform in 30-90 days after major contract wins',
    insight: 'Awards >$100M average 8% stock appreciation within 60 days',
    query: 'Show me stocks that received >$100M contracts and their price performance 30/60/90 days after award date',
    category: 'Government Contracts',
    collections: ['Award', 'Company', 'MarketData'],
    edges: ['HAS_AWARD', 'HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '💰'
  },

  // MACRO CORRELATIONS
  {
    title: 'Fed Rate Impact on Commodities',
    description: 'How commodity futures react to Federal Reserve rate changes',
    insight: 'Gold rises 85% of the time when Fed cuts rates, copper falls 60% of the time',
    query: 'Show me commodity futures price changes within 30 days of Federal Reserve rate decisions since 2020',
    category: 'Macro Correlation',
    collections: ['EconomicData', 'futures_prices'],
    edges: ['MACRO_IMPACTS_COMMODITY'],
    difficulty: 'intermediate',
    icon: '🏦'
  },
  {
    title: 'Inflation vs Gold & Mining Stocks',
    description: 'Correlation between CPI inflation and gold prices + mining stock performance',
    insight: 'Gold typically leads inflation by 2-3 months, mining stocks lag gold by 1 month',
    query: 'Compare CPI inflation rate to gold futures prices and gold mining company stock prices over last 2 years',
    category: 'Macro Correlation',
    collections: ['EconomicData', 'futures_prices', 'Company', 'MarketData'],
    edges: ['MACRO_IMPACTS_COMMODITY', 'COMPANY_TRADES_COMMODITY', 'HAS_MARKETDATA'],
    difficulty: 'advanced',
    icon: '📊'
  },

  // CFTC POSITIONING
  {
    title: 'Speculator Extremes',
    description: 'Find commodities where speculators are at extreme long or short positions',
    insight: 'When speculators reach 90th percentile positioning, reversals occur within 2 weeks 67% of the time',
    query: 'Show me commodities where net speculator positioning is at 2-year highs or lows',
    category: 'CFTC Positioning',
    collections: ['commodity_positions', 'futures_prices'],
    edges: ['POSITION_ON_COMMODITY'],
    difficulty: 'intermediate',
    icon: '📉'
  },
  {
    title: 'Commercial vs Speculator Divergence',
    description: 'When commercial hedgers disagree with speculators on commodity direction',
    insight: 'Commercial hedgers (producers) are right 72% of the time vs speculators',
    query: 'Find commodities where commercial hedgers are net long but speculators are net short (or vice versa)',
    category: 'CFTC Positioning',
    collections: ['commodity_positions', 'futures_prices', 'Company'],
    edges: ['POSITION_ON_COMMODITY', 'HAS_COMMODITY_POSITION'],
    difficulty: 'advanced',
    icon: '🔄'
  },
]

const CATEGORIES = [
  'All',
  'Insider Trading',
  'Commodity Correlation',
  'Prediction Markets',
  'Multi-Source Sentiment',
  'Government Contracts',
  'Macro Correlation',
  'CFTC Positioning',
]

interface ComplexQueryGalleryProps {
  onQuerySelect: (query: string) => void
}

export default function ComplexQueryGallery({ onQuerySelect }: ComplexQueryGalleryProps) {
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null)

  const filteredQueries = selectedCategory === 'All'
    ? COMPLEX_QUERIES
    : COMPLEX_QUERIES.filter(q => q.category === selectedCategory)

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'intermediate': return 'text-blue-400 bg-blue-500/10 border-blue-500/30'
      case 'advanced': return 'text-purple-400 bg-purple-500/10 border-purple-500/30'
      case 'expert': return 'text-red-400 bg-red-500/10 border-red-500/30'
      default: return 'text-gray-400 bg-gray-500/10 border-gray-500/30'
    }
  }

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="border-b border-gold/20 pb-4">
        <h3 className="text-2xl font-bold text-gold mb-2">🧠 Complex Query Gallery</h3>
        <p className="text-sm text-gray-400">
          Pre-built multi-hop graph queries that demonstrate the full power of our knowledge graph.
          Click any query to execute it instantly.
        </p>
      </div>

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
              selectedCategory === cat
                ? 'bg-gold text-dark-900 shadow-lg'
                : 'bg-dark-800 text-gray-400 border border-white/10 hover:border-gold/30 hover:text-gold'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Query Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredQueries.map((q, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            className="bg-dark-900/60 border border-white/10 rounded-xl p-4 hover:border-gold/30 transition-all group cursor-pointer"
            onClick={() => setExpandedQuery(expandedQuery === q.title ? null : q.title)}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{q.icon}</span>
                <div>
                  <h4 className="text-sm font-bold text-white group-hover:text-gold transition-colors">
                    {q.title}
                  </h4>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`text-[9px] px-2 py-0.5 rounded-full border font-bold uppercase tracking-wider ${getDifficultyColor(q.difficulty)}`}>
                      {q.difficulty}
                    </span>
                    <span className="text-[9px] text-gray-500 font-mono">
                      {q.collections.length} collections
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Description */}
            <p className="text-xs text-gray-400 mb-3 leading-relaxed">
              {q.description}
            </p>

            {/* Insight Badge */}
            <div className="bg-gold/5 border border-gold/20 rounded-lg p-2 mb-3">
              <div className="text-[9px] text-gold font-bold uppercase tracking-wider mb-1">💡 Why This Matters</div>
              <p className="text-[10px] text-gray-300 leading-relaxed">
                {q.insight}
              </p>
            </div>

            {/* Collections & Edges */}
            <AnimatePresence>
              {expandedQuery === q.title && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="space-y-2 mb-3"
                >
                  <div>
                    <div className="text-[9px] text-gray-500 font-bold uppercase tracking-wider mb-1">Collections Used</div>
                    <div className="flex flex-wrap gap-1">
                      {q.collections.map(c => (
                        <span key={c} className="text-[9px] bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded border border-blue-500/30">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] text-gray-500 font-bold uppercase tracking-wider mb-1">Graph Edges</div>
                    <div className="flex flex-wrap gap-1">
                      {q.edges.map(e => (
                        <span key={e} className="text-[9px] bg-purple-500/10 text-purple-400 px-2 py-0.5 rounded border border-purple-500/30 font-mono">
                          {e}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Execute Button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                onQuerySelect(q.query)
              }}
              className="w-full px-3 py-2 bg-gold/10 border border-gold/30 rounded-lg text-xs text-gold hover:bg-gold/20 hover:border-gold/50 transition-all font-bold uppercase tracking-wider group-hover:shadow-lg"
            >
              Execute Query →
            </button>
          </motion.div>
        ))}
      </div>

      {filteredQueries.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No queries found for this category
        </div>
      )}
    </div>
  )
}
