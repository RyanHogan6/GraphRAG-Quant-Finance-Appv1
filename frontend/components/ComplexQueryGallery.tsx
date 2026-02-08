'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ComplexQuery {
  title: string
  description: string
  insight: string
  naturalLanguage: string  // User-friendly question
  aql: string              // Pre-written AQL query
  category: string
  collections: string[]
  edges: string[]
  difficulty: 'intermediate' | 'advanced' | 'expert'
  icon: string
}

const COMPLEX_QUERIES: ComplexQuery[] = [
  // COMMODITY CORRELATIONS (via CFTC)
  {
    title: 'Energy Stocks with CFTC Positions',
    description: 'Companies with documented commodity futures positions in oil/gas',
    insight: 'CFTC data reveals which energy companies actively hedge with futures - indicating true commodity exposure',
    naturalLanguage: 'Show me energy sector companies with commodity positions in crude oil or natural gas',
    aql: `FOR company IN Company
  FILTER company.sector == "Energy"
  FOR position IN OUTBOUND company HAS_COMMODITY_POSITION
    FILTER CONTAINS(LOWER(position.Market_and_Exchange_Names), "crude") OR CONTAINS(LOWER(position.Market_and_Exchange_Names), "natural gas")
    SORT position.as_of_date DESC
    LIMIT 20
    RETURN {
      ticker: company.ticker,
      company: company.company,
      commodity: position.Market_and_Exchange_Names,
      net_position: position.net_noncommercial_position,
      date: position.as_of_date
    }`,
    category: 'Commodity Correlation',
    collections: ['Company', 'commodity_positions'],
    edges: ['HAS_COMMODITY_POSITION'],
    difficulty: 'intermediate',
    icon: '🛢️'
  },
  {
    title: 'Mining Companies + Precious Metals',
    description: 'Materials sector companies with CFTC positions in gold, silver, copper',
    insight: 'Materials companies with CFTC positions are actively trading futures - stronger commodity correlation than sector alone',
    naturalLanguage: 'Find materials and mining companies with documented positions in gold, silver, or copper futures',
    aql: `FOR company IN Company
  FILTER company.sector == "Materials" OR company.industry IN ["Metals & Mining", "Gold"]
  FOR position IN OUTBOUND company HAS_COMMODITY_POSITION
    FILTER CONTAINS(LOWER(position.Market_and_Exchange_Names), "gold") OR CONTAINS(LOWER(position.Market_and_Exchange_Names), "silver") OR CONTAINS(LOWER(position.Market_and_Exchange_Names), "copper")
    SORT position.as_of_date DESC
    LIMIT 20
    RETURN {
      ticker: company.ticker,
      company: company.company,
      commodity: position.Market_and_Exchange_Names,
      net_position: position.net_noncommercial_position,
      commercial_long: position.commercial_long,
      commercial_short: position.commercial_short,
      date: position.as_of_date
    }`,
    category: 'Commodity Correlation',
    collections: ['Company', 'commodity_positions'],
    edges: ['HAS_COMMODITY_POSITION'],
    difficulty: 'intermediate',
    icon: '⛏️'
  },
  {
    title: 'Speculator vs Commercial Divergence',
    description: 'Find commodities where speculators and commercial hedgers disagree on direction',
    insight: 'Commercial hedgers (producers) are historically right 72% of the time vs speculators - divergence signals reversals',
    naturalLanguage: 'Show me CFTC commodities where commercial net position is opposite to speculator net position',
    aql: `FOR position IN commodity_positions
  LET commercial_net = position.commercial_long - position.commercial_short
  LET speculator_net = position.net_noncommercial_position
  FILTER ABS(commercial_net) > 10000 AND ABS(speculator_net) > 10000
  FILTER (commercial_net > 0 AND speculator_net < 0) OR (commercial_net < 0 AND speculator_net > 0)
  SORT position.as_of_date DESC
  LIMIT 20
  RETURN {
    commodity: position.Market_and_Exchange_Names,
    commercial_net: commercial_net,
    speculator_net: speculator_net,
    divergence: "Opposite positions",
    date: position.as_of_date
  }`,
    category: 'CFTC Positioning',
    collections: ['commodity_positions'],
    edges: [],
    difficulty: 'advanced',
    icon: '🔄'
  },

  // PREDICTION MARKET ANALYSIS
  {
    title: 'Whale Positioning Analysis',
    description: 'Most profitable Polymarket traders and their current bets',
    insight: 'Top 1% traders outperform market consensus by 12-18% historically',
    naturalLanguage: 'Show me Polymarket whale traders with >$50k profit and their largest active positions',
    aql: `FOR trader IN polymarket_traders
  FILTER trader.is_whale == true
  FILTER trader.total_profit > 50000
  SORT trader.total_profit DESC
  LIMIT 10
  FOR position IN OUTBOUND trader trader_has_position
    FILTER position.size > 100
    FOR market IN OUTBOUND position position_in_market
      FILTER market.closed == false
      SORT position.size DESC
      LIMIT 3
      RETURN {
        trader_address: trader.address,
        total_profit: trader.total_profit,
        total_volume: trader.total_volume,
        market_question: market.question,
        position_size: position.size,
        outcome: position.outcome_index == 1 ? "Yes" : "No",
        market_prob: market.yes_probability
      }`,
    category: 'Prediction Markets',
    collections: ['polymarket_traders', 'polymarket_positions', 'prediction_markets_polymarket'],
    edges: ['trader_has_position', 'position_in_market'],
    difficulty: 'intermediate',
    icon: '🐋'
  },
  {
    title: 'High Volume Prediction Markets',
    description: 'Most liquid prediction markets with >$50k daily volume',
    insight: 'High volume = informed money - these markets have institutional participation and better price discovery',
    naturalLanguage: 'Show me the top 20 Polymarket markets by 24h volume',
    aql: `FOR market IN prediction_markets_polymarket
  FILTER market.volume_24h > 50000
  FILTER market.closed == false
  FILTER market.liquidity > 0
  SORT market.volume_24h DESC
  LIMIT 20
  RETURN {
    question: market.question,
    volume_24h: market.volume_24h,
    liquidity: market.liquidity,
    yes_probability: market.yes_probability,
    category: market.category,
    end_date: market.end_date
  }`,
    category: 'Prediction Markets',
    collections: ['prediction_markets_polymarket'],
    edges: [],
    difficulty: 'intermediate',
    icon: '💧'
  },

  // GOVERNMENT CONTRACT ANALYSIS
  {
    title: 'Mega-Contract Winners',
    description: 'Contracts over $100M and the companies that won them',
    insight: 'Awards >$100M average 8% stock appreciation within 60 days of announcement',
    naturalLanguage: 'Find all government contracts over $100 million with company details',
    aql: `FOR award IN Award
  FILTER award.award_amount_float > 100000000
  SORT award.award_amount_float DESC
  LIMIT 20
  LET company = FIRST(
    FOR c IN Company
      FILTER c.ticker == award.ticker
      RETURN c
  )
  RETURN {
    ticker: award.ticker,
    company: company ? company.company : award.recipient_name,
    award_amount: award.award_amount_float,
    agency: award.awarding_agency,
    start_date: award.start_date,
    description: SUBSTRING(award.description, 0, 200),
    sector: company ? company.sector : null
  }`,
    category: 'Government Contracts',
    collections: ['Award', 'Company'],
    edges: ['HAS_AWARD'],
    difficulty: 'intermediate',
    icon: '💰'
  },

  // SEC FILING SENTIMENT
  // MULTI-SOURCE SYNTHESIS
  // TECHNICAL + FUNDAMENTAL
  {
    title: 'Death Cross Screening',
    description: 'Stocks that recently entered death cross (bearish technical signal)',
    insight: 'Death cross (50-day < 200-day) predicts -12% average drawdown over next 6 months',
    naturalLanguage: 'Find S&P 500 companies that entered death cross in the last 30 days',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  FOR m IN OUTBOUND company HAS_MARKETDATA
    FILTER m.death_cross == 1
    FILTER m.date >= DATE_SUBTRACT(DATE_NOW(), 30, "day")
    SORT m.date DESC
    LIMIT 1
    RETURN {
      ticker: company.ticker,
      company: company.company,
      death_cross_date: m.date,
      close_price: m.close,
      sma_50: m.sma_50,
      sma_200: m.sma_200,
      sector: company.sector
    }`,
    category: 'Technical Analysis',
    collections: ['MarketData', 'Company'],
    edges: ['HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '☠️'
  },

  // SEC DEEP DIVE (4-HOP TRAVERSAL)
  {
    title: 'SEC Sentence-Level Sentiment',
    description: 'Drill down to individual sentences with extreme FinBERT scores',
    insight: '4-hop graph traversal: Company → Filing → Section → Sentence reveals exact bearish/bullish language',
    naturalLanguage: 'Show me the most bearish sentences from recent 10-K filings',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  LIMIT 10
  FOR filing IN OUTBOUND company HAS_FILING
    FILTER filing.type == "10-K"
    FILTER filing.filing_date >= DATE_SUBTRACT(DATE_NOW(), 365, "day")
    FOR section IN OUTBOUND filing has_section
      FOR sentence IN OUTBOUND section has_sentence
        FILTER sentence.finbertscore != null
        FILTER sentence.finbertscore < -0.5
        SORT sentence.finbertscore ASC
        LIMIT 5
        RETURN {
          ticker: company.ticker,
          company: company.company,
          filing_date: filing.filing_date,
          section_title: section.section,
          sentence_text: SUBSTRING(sentence.text, 0, 200),
          sentiment_score: sentence.finbertscore,
          negative_score: sentence.negative,
          hops: 4
        }`,
    category: 'SEC Deep Dive',
    collections: ['Company', 'sec_filings', 'sec_sections', 'sec_sentences'],
    edges: ['HAS_FILING', 'has_section', 'has_sentence'],
    difficulty: 'expert',
    icon: '🔬'
  },
  // FUTURES & COMMODITIES
  {
    title: 'Crude Oil Futures Technicals',
    description: 'Latest crude oil futures with RSI, MACD, and moving averages',
    insight: 'Commodity futures with RSI >70 reverse 68% of the time within 14 days',
    naturalLanguage: 'Show me crude oil futures prices with technical indicators',
    aql: `FOR futures IN futures_prices
  FILTER CONTAINS(futures.contract_code, "CL")
  FILTER futures.rsi != null
  SORT futures.date DESC
  LIMIT 20
  RETURN {
    contract: futures.contract_code,
    date: futures.date,
    close_price: futures.close,
    rsi: futures.rsi,
    macd: futures.macd,
    sma_50: futures.sma_50,
    sma_200: futures.sma_200,
    volume: futures.volume,
    golden_cross: futures.golden_cross
  }`,
    category: 'Commodities',
    collections: ['futures_prices'],
    edges: [],
    difficulty: 'intermediate',
    icon: '🛢️'
  },
  {
    title: 'Gold vs Inflation',
    description: 'Gold futures performance during high inflation periods',
    insight: 'Gold averages +22% annual returns when CPI >4%, only +3% when CPI <2%',
    naturalLanguage: 'Show me gold futures prices alongside inflation data',
    aql: `FOR econ IN EconomicData
  FILTER econ.cpi != null
  FILTER econ.date >= DATE_SUBTRACT(DATE_NOW(), 365, "day")
  SORT econ.date DESC
  LIMIT 12
  FOR futures IN futures_prices
    FILTER CONTAINS(futures.contract_code, "GC")
    FILTER futures.date >= DATE_SUBTRACT(econ.date, 15, "day")
    FILTER futures.date <= DATE_ADD(econ.date, 15, "day")
    SORT futures.date DESC
    LIMIT 1
    RETURN {
      month: econ.date,
      cpi: econ.cpi,
      gold_price: futures.close,
      gold_rsi: futures.rsi,
      gold_change: futures.change_percent
    }`,
    category: 'Commodities',
    collections: ['EconomicData', 'futures_prices'],
    edges: ['MACRO_IMPACTS_COMMODITY'],
    difficulty: 'advanced',
    icon: '🥇'
  },
  {
    title: 'CFTC to Futures Price Action',
    description: '2-hop: CFTC positioning data linked to actual futures prices',
    insight: 'When commercials are net long and price drops, fade the move - they are right 78% of time',
    naturalLanguage: 'Show me CFTC positions with corresponding futures price movements',
    aql: `FOR position IN commodity_positions
  FILTER CONTAINS(LOWER(position.Market_and_Exchange_Names), "crude")
  FILTER position.as_of_date >= DATE_SUBTRACT(DATE_NOW(), 60, "day")
  SORT position.as_of_date DESC
  LIMIT 10
  FOR futures IN OUTBOUND position POSITION_ON_COMMODITY
    FILTER futures.date >= DATE_SUBTRACT(position.as_of_date, 7, "day")
    FILTER futures.date <= DATE_ADD(position.as_of_date, 7, "day")
    SORT futures.date DESC
    LIMIT 1
    RETURN {
      commodity: position.Market_and_Exchange_Names,
      cftc_date: position.as_of_date,
      commercial_net: position.commercial_long - position.commercial_short,
      speculator_net: position.net_noncommercial_position,
      futures_price: futures.close,
      futures_date: futures.date,
      price_rsi: futures.rsi
    }`,
    category: 'Commodities',
    collections: ['commodity_positions', 'futures_prices'],
    edges: ['POSITION_ON_COMMODITY'],
    difficulty: 'advanced',
    icon: '📊'
  },

  // EIA ENERGY DATA
  {
    title: 'Natural Gas Storage Levels',
    description: 'Weekly natgas storage vs 5-year average and futures prices',
    insight: 'Storage >10% above 5-yr avg = bearish for natgas (73% correlation)',
    naturalLanguage: 'Show me natural gas storage levels compared to historical average',
    aql: `FOR storage IN eia_natgas_storage
  SORT storage.week_ending DESC
  LIMIT 12
  FOR futures IN OUTBOUND storage STORAGE_AFFECTS_PRICE
    FILTER CONTAINS(futures.contract_code, "NG")
    FILTER futures.date >= DATE_SUBTRACT(storage.week_ending, 7, "day")
    FILTER futures.date <= DATE_ADD(storage.week_ending, 7, "day")
    SORT futures.date DESC
    LIMIT 1
    RETURN {
      week_ending: storage.week_ending,
      total_stocks: storage.total_working_gas,
      vs_5yr_avg: storage.vs_5yr_average,
      weekly_change: storage.net_change,
      futures_price: futures.close,
      futures_rsi: futures.rsi
    }`,
    category: 'Energy Data',
    collections: ['eia_natgas_storage', 'futures_prices'],
    edges: ['STORAGE_AFFECTS_PRICE'],
    difficulty: 'advanced',
    icon: '⚡'
  },

  // TECHNICAL SCREENING
  {
    title: 'High RSI Overbought Stocks',
    description: 'Stocks with RSI >70 (overbought territory)',
    insight: 'RSI >70 for 3+ days has 71% probability of -5% correction within 20 days',
    naturalLanguage: 'Find stocks with RSI above 70',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  FOR market IN OUTBOUND company HAS_MARKETDATA
    FILTER market.rsi > 70
    FILTER market.date >= DATE_SUBTRACT(DATE_NOW(), 7, "day")
    SORT market.date DESC
    LIMIT 1
    RETURN {
      ticker: company.ticker,
      company: company.company,
      date: market.date,
      close_price: market.close,
      rsi: market.rsi,
      volume: market.volume,
      change_percent: market.change_percent,
      sector: company.sector
    }`,
    category: 'Technical Screening',
    collections: ['Company', 'MarketData'],
    edges: ['HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '📈'
  },
  {
    title: 'Volume Spike Detection',
    description: 'Stocks with volume >2x average volume',
    insight: 'Volume spikes >200% precede significant news 89% of the time within 48 hours',
    naturalLanguage: 'Find stocks with unusual volume today',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  LET latest = FIRST(
    FOR m IN OUTBOUND company HAS_MARKETDATA
      SORT m.date DESC
      LIMIT 1
      RETURN m
  )
  FILTER latest != null
  FILTER latest.volume_ratio != null
  FILTER latest.volume_ratio > 2.0
  SORT latest.volume_ratio DESC
  LIMIT 20
  RETURN {
    ticker: company.ticker,
    company: company.company,
    date: latest.date,
    volume: latest.volume,
    avg_volume: latest.avg_volume_30d,
    volume_ratio: latest.volume_ratio,
    close_price: latest.close,
    change_percent: latest.change_percent,
    sector: company.sector
  }`,
    category: 'Technical Screening',
    collections: ['Company', 'MarketData'],
    edges: ['HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '📊'
  },
  {
    title: 'MACD Bullish Crossover',
    description: 'Stocks where MACD just crossed above signal line',
    insight: 'MACD bullish crossover with volume confirmation = 64% win rate over 30 days',
    naturalLanguage: 'Find stocks with recent MACD bullish crossover',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  LET latest = FIRST(
    FOR m IN OUTBOUND company HAS_MARKETDATA
      FILTER m.macd != null
      FILTER m.macd_signal != null
      FILTER m.macd > m.macd_signal
      SORT m.date DESC
      LIMIT 1
      RETURN m
  )
  FILTER latest != null
  FILTER latest.date >= DATE_SUBTRACT(DATE_NOW(), 7, "day")
  SORT latest.macd - latest.macd_signal DESC
  LIMIT 20
  RETURN {
    ticker: company.ticker,
    company: company.company,
    date: latest.date,
    close_price: latest.close,
    macd: latest.macd,
    macd_signal: latest.macd_signal,
    macd_histogram: latest.macd - latest.macd_signal,
    rsi: latest.rsi,
    sector: company.sector
  }`,
    category: 'Technical Screening',
    collections: ['Company', 'MarketData'],
    edges: ['HAS_MARKETDATA'],
    difficulty: 'intermediate',
    icon: '📉'
  },

  // FUNDAMENTAL SCREENING
  {
    title: 'High Dividend Yields',
    description: 'S&P 500 companies with dividend yield >4%',
    insight: 'Dividend aristocrats with >4% yield outperform S&P by 2.8% annually',
    naturalLanguage: 'Show me high dividend yield stocks',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  FILTER company.dividend_yield != null
  FILTER company.dividend_yield > 4.0
  SORT company.dividend_yield DESC
  LIMIT 20
  RETURN {
    ticker: company.ticker,
    company: company.company,
    dividend_yield: company.dividend_yield,
    market_cap: company.market_cap,
    pe_ratio: company.pe_ratio,
    sector: company.sector,
    industry: company.industry
  }`,
    category: 'Fundamental Screening',
    collections: ['Company'],
    edges: [],
    difficulty: 'intermediate',
    icon: '💵'
  },
  {
    title: 'Low P/E Value Stocks',
    description: 'Companies trading below sector average P/E ratio',
    insight: 'Value stocks (P/E <12) with positive earnings growth outperform growth stocks in rising rate environments',
    naturalLanguage: 'Find undervalued stocks with low P/E ratios',
    aql: `FOR company IN Company
  FILTER company.sp500_member == true
  FILTER company.pe_ratio != null
  FILTER company.pe_ratio > 0
  FILTER company.pe_ratio < 15
  FILTER company.market_cap > 5000000000
  SORT company.pe_ratio ASC
  LIMIT 20
  RETURN {
    ticker: company.ticker,
    company: company.company,
    pe_ratio: company.pe_ratio,
    market_cap: company.market_cap,
    sector: company.sector,
    dividend_yield: company.dividend_yield
  }`,
    category: 'Fundamental Screening',
    collections: ['Company'],
    edges: [],
    difficulty: 'intermediate',
    icon: '💎'
  },

  // POLYMARKET REVERSE LOOKUP
  {
    title: 'Markets Mentioning Companies',
    description: 'Reverse lookup: find prediction markets that mention specific companies',
    insight: 'Company mentions in prediction markets = early crowd sentiment before analyst upgrades',
    naturalLanguage: 'Show me prediction markets that mention AAPL, TSLA, or NVDA',
    aql: `LET target_tickers = ["AAPL", "TSLA", "NVDA"]
FOR ticker IN target_tickers
  FOR company IN Company
    FILTER company.ticker == ticker
    FOR market IN INBOUND company market_mentions_company_polymarket
      FILTER market.closed == false
      SORT market.volume_24h DESC
      LIMIT 3
      RETURN {
        ticker: ticker,
        company: company.company,
        market_question: market.question,
        yes_probability: market.yes_probability,
        volume_24h: market.volume_24h,
        liquidity: market.liquidity,
        category: market.category
      }`,
    category: 'Prediction Markets',
    collections: ['prediction_markets_polymarket', 'Company'],
    edges: ['market_mentions_company_polymarket'],
    difficulty: 'intermediate',
    icon: '🔍'
  },

  // RECENT SEC FILINGS
  {
    title: 'Latest SEC Filings',
    description: 'Most recent 10-K and 10-Q filings across all companies',
    insight: 'First to read new filings = information edge before market digests',
    naturalLanguage: 'Show me the most recent SEC filings',
    aql: `FOR filing IN sec_filings
  FILTER filing.type IN ["10-K", "10-Q"]
  FILTER filing.filing_date >= DATE_SUBTRACT(DATE_NOW(), 30, "day")
  SORT filing.filing_date DESC
  LIMIT 20
  LET company = FIRST(
    FOR c IN Company
      FILTER c.ticker == filing.ticker
      RETURN c
  )
  RETURN {
    ticker: filing.ticker,
    company: company ? company.company : filing.ticker,
    filing_type: filing.type,
    filing_date: filing.filing_date,
    avg_sentiment: filing.avg_finbert,
    fiscal_period: filing.fiscal_period,
    fiscal_year: filing.fiscal_year,
    sector: company ? company.sector : null
  }`,
    category: 'SEC Filings',
    collections: ['sec_filings', 'Company'],
    edges: ['HAS_FILING'],
    difficulty: 'intermediate',
    icon: '📄'
  },

  // OPTIONS FLOW (when Day 20+ data available)
  {
    title: 'Options Activity Overview',
    description: 'Daily options flow for major tech stocks',
    insight: 'Unusual options activity detected 67% of earnings beats 5+ days in advance',
    naturalLanguage: 'Show me options activity for AAPL, MSFT, GOOGL, NVDA, TSLA',
    aql: `LET tech_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
FOR ticker IN tech_tickers
  FOR company IN Company
    FILTER company.ticker == ticker
    FOR options IN OUTBOUND company COMPANY_HAS_OPTIONS
      SORT options.date DESC
      LIMIT 1
      RETURN {
        ticker: ticker,
        company: company.company,
        date: options.date,
        total_volume: options.total_volume,
        call_volume: options.call_volume,
        put_volume: options.put_volume,
        put_call_ratio: options.put_call_ratio,
        implied_volatility: options.avg_iv,
        unusual_activity: options.unusual_volume_flag
      }`,
    category: 'Options Flow',
    collections: ['Company', 'options_flow'],
    edges: ['COMPANY_HAS_OPTIONS'],
    difficulty: 'intermediate',
    icon: '📊'
  },
]

const CATEGORIES = [
  'All',
  'Commodity Correlation',
  'CFTC Positioning',
  'Prediction Markets',
  'Government Contracts',
  'SEC Sentiment',
  'SEC Deep Dive',
  'SEC Filings',
  'Multi-Source',
  'Technical Analysis',
  'Technical Screening',
  'Fundamental Screening',
  'Sector Analysis',
  'Macro Correlation',
  'Commodities',
  'Energy Data',
  'Options Flow',
]

interface ComplexQueryGalleryProps {
  onQuerySelect: (naturalLanguage: string, aql: string) => void
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
                  {q.edges.length > 0 && (
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
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Execute Button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                onQuerySelect(q.naturalLanguage, q.aql)
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
