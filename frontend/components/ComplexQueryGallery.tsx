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
    title: 'Cross-Platform Sentiment Check',
    description: 'Compare Polymarket and Kalshi probabilities for tech companies',
    insight: 'Same event priced differently on two platforms reveals arbitrage opportunities or information asymmetry',
    naturalLanguage: 'Compare sentiment for AAPL, MSFT, GOOGL across Polymarket and Kalshi markets',
    aql: `LET tickers = ["AAPL", "MSFT", "GOOGL"]
FOR ticker IN tickers
  FOR company IN Company
    FILTER company.ticker == ticker
    LET poly_markets = (
      FOR market IN INBOUND company market_mentions_company_polymarket
        FILTER market.closed == false
        RETURN {platform: "Polymarket", question: market.question, probability: market.yes_probability, volume: market.volume_24h}
    )
    LET kalshi_markets = (
      FOR market IN INBOUND company market_mentions_company_kalshi
        FILTER market.closed == false
        RETURN {platform: "Kalshi", question: market.title, probability: market.yes_probability, volume: market.volume}
    )
    RETURN {
      ticker: ticker,
      company: company.company,
      polymarket_count: LENGTH(poly_markets),
      polymarket_avg_prob: AVG(poly_markets[*].probability),
      kalshi_count: LENGTH(kalshi_markets),
      kalshi_avg_prob: AVG(kalshi_markets[*].probability),
      top_poly_market: FIRST(poly_markets),
      top_kalshi_market: FIRST(kalshi_markets)
    }`,
    category: 'Prediction Markets',
    collections: ['prediction_markets_polymarket', 'prediction_markets_kalshi', 'Company'],
    edges: ['market_mentions_company_polymarket', 'market_mentions_company_kalshi'],
    difficulty: 'advanced',
    icon: '⚖️'
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
    title: 'Defense Contract Winners',
    description: 'Top 10 defense contractors by total award value this fiscal year',
    insight: 'Concentration analysis - top 5 contractors capture 60% of DoD spending',
    naturalLanguage: 'Show me companies with the highest total government contract awards from defense agencies',
    aql: `FOR company IN Company
  LET awards = (
    FOR award IN OUTBOUND company HAS_AWARD
      FILTER award.awarding_agency LIKE "%Defense%" OR award.awarding_agency LIKE "%Navy%" OR award.awarding_agency LIKE "%Army%" OR award.awarding_agency LIKE "%Air Force%"
      RETURN award
  )
  FILTER LENGTH(awards) > 0
  LET total_value = SUM(awards[*].award_amount_float)
  SORT total_value DESC
  LIMIT 10
  RETURN {
    ticker: company.ticker,
    company: company.company,
    total_award_value: total_value,
    award_count: LENGTH(awards),
    avg_award: total_value / LENGTH(awards),
    sector: company.sector
  }`,
    category: 'Government Contracts',
    collections: ['Award', 'Company'],
    edges: ['HAS_AWARD'],
    difficulty: 'intermediate',
    icon: '🏛️'
  },
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
  {
    title: 'Most Bearish 10-K Filings',
    description: 'Companies with most negative sentiment in annual reports',
    insight: 'Extreme negative FinBERT scores (<-0.2) precede stock declines 65% of the time within 90 days',
    naturalLanguage: 'Show me the 15 most bearish 10-K filings from the last 2 years with sentiment scores',
    aql: `FOR filing IN sec_filings
  FILTER filing.type == "10-K"
  FILTER filing.avg_finbert != null
  FILTER filing.avg_finbert < -0.1
  FILTER filing.filing_date >= DATE_SUBTRACT(DATE_NOW(), 730, "day")
  SORT filing.avg_finbert ASC
  LIMIT 15
  LET company = FIRST(
    FOR c IN Company
      FILTER c.ticker == filing.ticker
      RETURN c
  )
  RETURN {
    ticker: filing.ticker,
    company: company ? company.company : filing.ticker,
    filing_date: filing.filing_date,
    sentiment_score: filing.avg_finbert,
    negative_score: filing.avg_negative,
    fiscal_year: filing.fiscal_year,
    sector: company ? company.sector : null
  }`,
    category: 'SEC Sentiment',
    collections: ['sec_filings', 'Company'],
    edges: ['HAS_FILING'],
    difficulty: 'intermediate',
    icon: '📉'
  },
  {
    title: 'Sentiment Flip Detection',
    description: 'Companies whose SEC sentiment changed significantly between consecutive 10-Ks',
    insight: 'Sentiment reversals >0.3 absolute change correlate with major business inflection points',
    naturalLanguage: 'Find companies where 10-K sentiment changed significantly between consecutive filings',
    aql: `FOR company IN Company
  LET filings = (
    FOR filing IN OUTBOUND company HAS_FILING
      FILTER filing.type == "10-K"
      FILTER filing.avg_finbert != null
      SORT filing.filing_date DESC
      LIMIT 2
      RETURN filing
  )
  FILTER LENGTH(filings) == 2
  LET sentiment_change = filings[0].avg_finbert - filings[1].avg_finbert
  FILTER ABS(sentiment_change) > 0.2
  SORT ABS(sentiment_change) DESC
  LIMIT 15
  RETURN {
    ticker: company.ticker,
    company: company.company,
    latest_sentiment: filings[0].avg_finbert,
    previous_sentiment: filings[1].avg_finbert,
    sentiment_change: sentiment_change,
    latest_date: filings[0].filing_date,
    previous_date: filings[1].filing_date,
    direction: sentiment_change > 0 ? "Improved" : "Deteriorated"
  }`,
    category: 'SEC Sentiment',
    collections: ['sec_filings', 'Company'],
    edges: ['HAS_FILING'],
    difficulty: 'advanced',
    icon: '🔄'
  },

  // MULTI-SOURCE SYNTHESIS
  {
    title: 'Defense + Prediction Market Confluence',
    description: 'Defense contractors with both large contracts AND bullish prediction market sentiment',
    insight: 'When govt contracts align with market sentiment, conviction is 3x stronger than single source',
    naturalLanguage: 'Show me defense contractors with large contracts AND bullish prediction market sentiment',
    aql: `FOR company IN Company
  LET awards = (
    FOR award IN OUTBOUND company HAS_AWARD
      FILTER award.awarding_agency LIKE "%Defense%"
      FILTER award.award_amount_float > 50000000
      RETURN award
  )
  FILTER LENGTH(awards) > 0
  LET markets = (
    FOR market IN INBOUND company market_mentions_company_polymarket
      FILTER market.closed == false
      FILTER market.yes_probability > 0.55
      RETURN market
  )
  FILTER LENGTH(markets) > 0
  LIMIT 10
  RETURN {
    ticker: company.ticker,
    company: company.company,
    total_awards: SUM(awards[*].award_amount_float),
    award_count: LENGTH(awards),
    market_count: LENGTH(markets),
    avg_market_prob: AVG(markets[*].yes_probability),
    top_market: FIRST(markets).question
  }`,
    category: 'Multi-Source',
    collections: ['Award', 'Company', 'prediction_markets_polymarket'],
    edges: ['HAS_AWARD', 'market_mentions_company_polymarket'],
    difficulty: 'advanced',
    icon: '🎯'
  },
  {
    title: 'SEC Sentiment + Stock Performance',
    description: 'Companies with negative SEC filings but strong recent stock performance (sentiment divergence)',
    insight: 'When price ignores negative filings, either: (1) market knows something or (2) correction imminent',
    naturalLanguage: 'Find companies with bearish SEC sentiment but stock up >5% in last 30 days',
    aql: `FOR company IN Company
  LET recent_filing = FIRST(
    FOR filing IN OUTBOUND company HAS_FILING
      FILTER filing.type IN ["10-K", "10-Q"]
      FILTER filing.avg_finbert != null
      FILTER filing.avg_finbert < -0.1
      SORT filing.filing_date DESC
      LIMIT 1
      RETURN filing
  )
  FILTER recent_filing != null
  LET market_data = (
    FOR m IN OUTBOUND company HAS_MARKETDATA
      FILTER m.date >= DATE_SUBTRACT(DATE_NOW(), 30, "day")
      SORT m.date ASC
      RETURN m
  )
  FILTER LENGTH(market_data) > 0
  LET first_price = FIRST(market_data).close
  LET last_price = LAST(market_data).close
  LET price_change = ((last_price - first_price) / first_price) * 100
  FILTER price_change > 5
  SORT price_change DESC
  LIMIT 15
  RETURN {
    ticker: company.ticker,
    company: company.company,
    filing_sentiment: recent_filing.avg_finbert,
    filing_date: recent_filing.filing_date,
    price_change_30d: ROUND(price_change * 100) / 100,
    current_price: last_price,
    divergence_signal: "Bearish filing + bullish price"
  }`,
    category: 'Multi-Source',
    collections: ['sec_filings', 'MarketData', 'Company'],
    edges: ['HAS_FILING', 'HAS_MARKETDATA'],
    difficulty: 'advanced',
    icon: '⚡'
  },

  // TECHNICAL + FUNDAMENTAL
  {
    title: 'Golden Cross with Awards',
    description: 'Stocks in golden cross pattern that also won recent government contracts',
    insight: 'Technical breakout + fundamental catalyst = 74% win rate over 90 days',
    naturalLanguage: 'Show me companies with golden cross that won contracts in last 60 days',
    aql: `FOR company IN Company
  LET latest_market = FIRST(
    FOR m IN OUTBOUND company HAS_MARKETDATA
      FILTER m.golden_cross == 1
      SORT m.date DESC
      LIMIT 1
      RETURN m
  )
  FILTER latest_market != null
  LET recent_awards = (
    FOR award IN OUTBOUND company HAS_AWARD
      FILTER award.start_date >= DATE_SUBTRACT(DATE_NOW(), 60, "day")
      RETURN award
  )
  FILTER LENGTH(recent_awards) > 0
  LIMIT 15
  RETURN {
    ticker: company.ticker,
    company: company.company,
    golden_cross_date: latest_market.date,
    close_price: latest_market.close,
    award_count: LENGTH(recent_awards),
    total_award_value: SUM(recent_awards[*].award_amount_float),
    latest_award: FIRST(recent_awards).awarding_agency
  }`,
    category: 'Multi-Source',
    collections: ['MarketData', 'Award', 'Company'],
    edges: ['HAS_MARKETDATA', 'HAS_AWARD'],
    difficulty: 'advanced',
    icon: '✨'
  },
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

  // MACRO CORRELATIONS
  {
    title: 'Fed Rate Changes Impact',
    description: 'S&P 500 performance during Fed rate decisions',
    insight: 'Tech stocks average -3% when Fed raises rates >25 bps, financials +5%',
    naturalLanguage: 'Show me how different sectors performed during recent Fed rate changes',
    aql: `FOR econ IN EconomicData
  FILTER econ.federal_funds_rate != null
  FILTER econ.date >= "2020-01-01"
  SORT econ.date DESC
  LIMIT 20
  LET sp500 = FIRST(
    FOR e IN EconomicData
      FILTER e.date == econ.date
      FILTER e.sandp_500_index != null
      RETURN e.sandp_500_index
  )
  RETURN {
    date: econ.date,
    fed_funds_rate: econ.federal_funds_rate,
    sp500_level: sp500,
    unemployment: econ.unemployment_rate,
    inflation: econ.cpi
  }`,
    category: 'Macro Correlation',
    collections: ['EconomicData', 'MarketData', 'Company'],
    edges: ['HAS_MARKETDATA'],
    difficulty: 'advanced',
    icon: '🏦'
  },
]

const CATEGORIES = [
  'All',
  'Commodity Correlation',
  'CFTC Positioning',
  'Prediction Markets',
  'Government Contracts',
  'SEC Sentiment',
  'Multi-Source',
  'Technical Analysis',
  'Macro Correlation',
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
