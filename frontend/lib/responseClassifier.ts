/**
 * Response Classification System
 * Routes queries to appropriate UI templates based on query intent and result structure
 */

export type QueryType =
  | 'company_deep_dive'      // Single company, comprehensive analysis
  | 'sector_comparison'      // Multiple companies in same sector
  | 'peer_analysis'          // Company vs specific peers
  | 'time_series'            // Historical trends, chart-first
  | 'metric_focused'         // Specific financial metric query
  | 'news_sentiment'         // SEC filings, sentiment analysis
  | 'insider_activity'       // Options flow, insider trading
  | 'general_query'          // Everything else

interface Message {
  role: string
  content: string
  results?: any[]
  presentationType?: string
  queryPlan?: {
    is_time_series?: boolean
    chart_data?: any
    intent?: string
  }
  metadata?: {
    tickers?: string[]
    companies?: string[]
    collections?: string[]
    queryIntent?: string
    data_types?: {
      has_options?: boolean
      has_sec_filings?: boolean
      has_form4_insider?: boolean
      has_awards?: boolean
      has_futures?: boolean
      has_commodities?: boolean
      has_eia_data?: boolean
      has_prediction_markets?: boolean
      has_market_data?: boolean
    }
  }
}

/**
 * Classify query based on results structure and metadata
 */
export function classifyQuery(message: Message): QueryType {
  const { metadata, results, queryPlan, presentationType, content } = message

  // Check backend presentation type hint first
  if (presentationType === 'insider_trading') return 'insider_activity'
  if (presentationType === 'sentiment_divergence') return 'news_sentiment'

  // If no results, treat as general query
  if (!results || results.length === 0) return 'general_query'

  // Check result count and data types (derive tickers from results if metadata missing)
  const tickers = metadata?.tickers?.length
    ? metadata.tickers
    : [...new Set((results || []).map((r: any) => r.ticker).filter(Boolean))] as string[]
  const hasSingleTicker = tickers.length === 1
  const hasMultipleTickers = tickers.length > 1
  const isTimeSeries = queryPlan?.is_time_series || false

  // Sector keyword detection
  const sectorKeywords = [
    'energy stocks', 'energy companies', 'energy sector',
    'tech companies', 'tech stocks', 'technology sector', 'tech sector',
    'financial sector', 'financial companies', 'banks',
    'healthcare companies', 'healthcare sector',
    'utilities', 'utility companies',
    'consumer discretionary', 'consumer staples',
    'industrials', 'industrial sector',
    'materials', 'materials sector',
    'real estate', 'reits'
  ]
  const hasSectorKeyword = sectorKeywords.some(kw =>
    content.toLowerCase().includes(kw)
  )

  // Screener / comparison-style query: multiple companies + filter criteria (dividend, FCF, large-cap, etc.)
  const screenerKeywords = [
    'dividend payers', 'dividend payers with', 'dividend stocks',
    'free cash flow', 'stable or growing', 'growing free cash flow',
    'large-cap', 'large cap', 'large cap dividend',
    'screen', 'screener', 'show me', 'find', 'list of',
    'companies with', 'stocks with', 'with stable', 'with growing'
  ]
  const hasScreenerKeyword = screenerKeywords.some(kw =>
    content.toLowerCase().includes(kw)
  )
  const isMultiCompanyScreener = results.length >= 3 &&
    hasMultipleTickers &&
    (hasScreenerKeyword || hasSectorKeyword)

  // Check if multiple companies are from same sector
  const allSameSector = (results: any[]): boolean => {
    if (results.length < 2) return false
    const sectors = results
      .map(r => r.sector)
      .filter(s => s != null)
    if (sectors.length < 2) return false
    return sectors.every(s => s === sectors[0])
  }

  // Single company with rich data → Deep dive
  if (hasSingleTicker && results.some(r =>
    r.MarketData?.length > 0 ||
    r.sec_filings?.length > 0 ||
    r.Award?.length > 0
  )) {
    return 'company_deep_dive'
  }

  // Multiple companies: sector comparison (same sector, sector keyword, or screener-style query)
  if ((hasMultipleTickers && allSameSector(results)) ||
      (hasMultipleTickers && hasSectorKeyword) ||
      isMultiCompanyScreener) {
    return 'sector_comparison'
  }

  // Multiple companies → Peer analysis
  if (hasMultipleTickers) {
    return 'peer_analysis'
  }

  // Time series data → Chart-first view
  if (isTimeSeries || queryPlan?.chart_data) {
    return 'time_series'
  }

  // Specific metric query detection
  if (isMetricQuery(content)) {
    return 'metric_focused'
  }

  // Default fallback
  return 'general_query'
}

/**
 * Detect if query is asking about a specific metric
 */
function isMetricQuery(content: string): boolean {
  const metricKeywords = [
    'what is', 'what\'s',
    'revenue', 'debt', 'equity', 'ratio',
    'p/e', 'pe ratio', 'price to earnings',
    'debt-to-equity', 'debt to equity',
    'current ratio', 'quick ratio',
    'free cash flow', 'fcf',
    'eps', 'earnings per share',
    'roe', 'return on equity',
    'roa', 'return on assets',
    'margin', 'profit margin', 'ebitda margin',
    'market cap', 'market capitalization'
  ]

  const lowerContent = content.toLowerCase()

  // Must have "what is" or similar question phrase
  const hasQuestionPhrase = lowerContent.includes('what is') ||
                           lowerContent.includes('what\'s') ||
                           lowerContent.includes('show me the')

  // And must have a metric keyword
  const hasMetricKeyword = metricKeywords.some(kw => lowerContent.includes(kw))

  // And should be a relatively short query (not a complex question)
  const isShortQuery = content.split(' ').length < 15

  return hasQuestionPhrase && hasMetricKeyword && isShortQuery
}

/**
 * Extract the specific metric being asked about
 */
export function extractMetric(content: string): string {
  const lowerContent = content.toLowerCase()

  // Map query patterns to metric names
  const metricMappings: { pattern: RegExp, metric: string }[] = [
    { pattern: /p\/e|pe ratio|price to earnings/i, metric: 'P/E Ratio' },
    { pattern: /debt[- ]?to[- ]?equity/i, metric: 'Debt-to-Equity' },
    { pattern: /current ratio/i, metric: 'Current Ratio' },
    { pattern: /quick ratio/i, metric: 'Quick Ratio' },
    { pattern: /free cash flow|fcf/i, metric: 'Free Cash Flow' },
    { pattern: /\beps\b|earnings per share/i, metric: 'EPS' },
    { pattern: /\broe\b|return on equity/i, metric: 'Return on Equity' },
    { pattern: /\broa\b|return on assets/i, metric: 'Return on Assets' },
    { pattern: /profit margin/i, metric: 'Profit Margin' },
    { pattern: /ebitda margin/i, metric: 'EBITDA Margin' },
    { pattern: /\brevenue\b/i, metric: 'Revenue' },
    { pattern: /market cap|market capitalization/i, metric: 'Market Cap' },
    { pattern: /\bdebt\b/i, metric: 'Total Debt' },
    { pattern: /\bequity\b/i, metric: 'Equity' }
  ]

  for (const { pattern, metric } of metricMappings) {
    if (pattern.test(content)) {
      return metric
    }
  }

  return 'Financial Metric'
}

/**
 * Get user-friendly description of query type
 */
export function getQueryTypeDescription(queryType: QueryType): string {
  const descriptions: Record<QueryType, string> = {
    company_deep_dive: 'Comprehensive company analysis with fundamentals, filings, and market data',
    sector_comparison: 'Sector-wide comparison showing relative performance and key metrics',
    peer_analysis: 'Peer-to-peer comparison across multiple companies',
    time_series: 'Historical trend analysis with chart visualization',
    metric_focused: 'Focused analysis of a specific financial metric',
    news_sentiment: 'SEC filings and sentiment analysis',
    insider_activity: 'Options flow and insider trading signals',
    general_query: 'General data query with flexible display'
  }

  return descriptions[queryType]
}
