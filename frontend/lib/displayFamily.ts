/**
 * Collection-aware result display: map message (metadata + results) to a display family
 * so the UI can render the appropriate component (CompanyWorkup, AwardResultsView, etc.)
 * Uses result-shape taxonomy (see resultShapeTaxonomy.ts) for OHLC, COT, probability timeline.
 */

import { getViewFamilyFromResults } from './resultShapeTaxonomy'

export type DisplayFamily =
  | 'company_workup'
  | 'company_enriched'
  | 'company_compare'
  | 'company_screener'
  | 'time_series'
  | 'ohlc_candlestick'
  | 'awards_list'
  | 'sec_filings_list'
  | 'sec_sentences'
  | 'prediction_markets_list'
  | 'polymarket_traders'
  | 'futures_commodities'
  | 'eia_energy'
  | 'economic_data'
  | 'options_flow_list'
  | 'positioning_cot'
  | 'probability_timeline'
  | 'generic'

export interface MessageForDisplay {
  results?: any[]
  metadata?: {
    collections_used?: string[]
    collections?: string[]
    data_types?: {
      has_options?: boolean
      has_sec_filings?: boolean
      has_awards?: boolean
      has_futures?: boolean
      has_commodities?: boolean
      has_eia_data?: boolean
      has_prediction_markets?: boolean
      has_market_data?: boolean
    }
    display_family?: DisplayFamily
  }
  queryPlan?: {
    intent?: string
    is_time_series?: boolean
    chart_data?: any
  }
}

function collectionsUsed(msg: MessageForDisplay): string[] {
  const meta = msg.metadata
  const list = meta?.collections_used ?? meta?.collections ?? []
  return Array.isArray(list) ? list.map((c: string) => String(c).toLowerCase()) : []
}

function dataTypes(msg: MessageForDisplay): Record<string, boolean> {
  return msg.metadata?.data_types ?? {}
}

function hasCollection(msg: MessageForDisplay, ...names: string[]): boolean {
  const used = collectionsUsed(msg)
  return names.some(n => used.includes(n.toLowerCase()))
}

function inferFromFirstRow(results: any[]): { keys: string[]; hasNested: (key: string) => boolean } {
  const first = results[0]
  if (!first || typeof first !== 'object') return { keys: [], hasNested: () => false }
  const keys = Object.keys(first).filter(k => !k.startsWith('_'))
  const hasNested = (key: string) => {
    const v = first[key]
    return Array.isArray(v) && v.length > 0 && typeof v[0] === 'object'
  }
  return { keys, hasNested }
}

export function getResultDisplayFamily(message: MessageForDisplay): DisplayFamily {
  const results = message.results ?? []
  const intent = message.queryPlan?.intent ?? ''
  const isBuilder = intent === 'builder_execution'

  // Backend-suggested family (when we add it)
  const suggested = message.metadata?.display_family
  if (suggested) return suggested

  // No results -> generic
  if (results.length === 0) return 'generic'

  const first = results[0]
  const { keys, hasNested } = inferFromFirstRow(results)
  const dt = dataTypes(message)

  // Company compare: exactly two workup-shaped results
  if (results.length === 2) {
    const workup = (r: any) => r?.ticker && (r?.MarketData?.length > 0 || r?.sec_filings?.length > 0 || r?.Award?.length > 0 || r?.prediction_markets_polymarket?.length > 0)
    if (workup(results[0]) && workup(results[1])) return 'company_compare'
  }

  // Single company workup (nested MarketData, sec_filings, Award, etc.)
  if (results.length === 1 && first?.ticker && (hasNested('MarketData') || hasNested('sec_filings') || hasNested('Award') || hasNested('prediction_markets_polymarket'))) {
    return 'company_workup'
  }

  // Single enriched (VQB): ticker + options_flow / Award / sec_filings / prediction_markets / futures_prices as arrays
  if (results.length === 1 && first?.ticker && isBuilder && (
    (Array.isArray(first.options_flow) && first.options_flow.length > 0) ||
    (Array.isArray(first.Award) && first.Award.length > 0) ||
    (Array.isArray(first.sec_filings) && first.sec_filings.length > 0) ||
    (Array.isArray(first.prediction_markets_polymarket) && first.prediction_markets_polymarket.length > 0) ||
    (Array.isArray(first.futures_prices) && first.futures_prices.length > 0)
  )) {
    return 'company_enriched'
  }

  // Result-shape taxonomy: OHLC vs time_series vs positioning vs probability
  const viewFamily = getViewFamilyFromResults(results)
  if (viewFamily === 'ohlc_candlestick') return 'ohlc_candlestick'
  if (viewFamily === 'positioning_cot') return 'positioning_cot'
  if (viewFamily === 'probability_timeline') return 'probability_timeline'

  // Time series: query plan hint or chart_data
  if (message.queryPlan?.is_time_series || message.queryPlan?.chart_data) return 'time_series'

  // Market data time series: flat rows with date + price/volume
  if (dt.has_market_data && keys.includes('date') && (keys.includes('close') || keys.includes('open'))) {
    return 'time_series'
  }

  // Awards list: Award collection, flat rows
  if (dt.has_awards || hasCollection(message, 'Award')) {
    if (results.every((r: any) => r.award_amount_float != null || r.recipient_name != null)) return 'awards_list'
  }

  // SEC filings list
  if (dt.has_sec_filings || hasCollection(message, 'sec_filings')) {
    if (keys.some(k => ['filing_date', 'type', 'avg_finbert', 'accession'].includes(k))) return 'sec_filings_list'
  }

  // SEC sentences: sentence-level
  if (hasCollection(message, 'sec_sentences') || (keys.includes('text') && keys.includes('finbert_score') && !keys.includes('filing_date'))) {
    return 'sec_sentences'
  }

  // Prediction markets list
  if (dt.has_prediction_markets || hasCollection(message, 'prediction_markets_polymarket', 'prediction_markets_kalshi')) {
    if (keys.some(k => ['question', 'yes_probability', 'volume_24h', 'title'].includes(k))) {
      if (keys.includes('address') && keys.some(k => ['total_profit', 'total_volume', 'is_whale'].includes(k))) return 'polymarket_traders'
      return 'prediction_markets_list'
    }
  }

  // Polymarket traders (address, total_profit, is_whale)
  if (hasCollection(message, 'polymarket_traders') || (keys.includes('address') && (keys.includes('total_profit') || keys.includes('total_volume')))) {
    return 'polymarket_traders'
  }

  // Futures / commodities (positioning_cot already handled above by shape)
  if (dt.has_futures || dt.has_commodities || hasCollection(message, 'futures_prices', 'commodity_positions')) {
    if (keys.some(k => ['SYMBOL', 'symbol', 'commodity', 'OPEN', 'CLOSE', 'VOLUME', 'close', 'volume'].includes(k))) return 'futures_commodities'
  }

  // EIA energy
  if (dt.has_eia_data || hasCollection(message, 'eia_crude_inventory', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng_exports')) {
    return 'eia_energy'
  }

  // Economic data
  if (hasCollection(message, 'EconomicData', 'economicdata')) {
    return 'economic_data'
  }

  // Options flow list
  if (dt.has_options || hasCollection(message, 'options_flow')) {
    if (keys.some(k => ['put_call_volume_ratio', 'ticker', 'unusual_total_activity'].includes(k))) return 'options_flow_list'
  }

  // Company screener: multiple rows with ticker + metrics, no nested arrays
  if (results.length >= 2 && first?.ticker && !hasNested('MarketData') && !hasNested('Award')) {
    return 'company_screener'
  }

  return 'generic'
}

/** Whether this family should show ResultsCharts (bar/time series) by default */
export function familyUsesCharts(family: DisplayFamily): boolean {
  return ['futures_commodities', 'eia_energy', 'economic_data', 'time_series', 'ohlc_candlestick', 'positioning_cot', 'probability_timeline'].includes(family)
}

/** For generic family only: show charts tab when result has category/date + numeric columns */
export function isGenericResultChartable(results: any[]): boolean {
  if (!results?.length) return false
  const first = results[0]
  if (!first || typeof first !== 'object') return false
  const keys = Object.keys(first).filter((k: string) => !k.startsWith('_'))
  const labelCandidates = ['SYMBOL', 'symbol', 'ticker', 'company', 'commodity', 'name', 'title']
  const dateCandidates = ['date', 'Date', 'filing_date', 'start_date', 'report_date', 'week_ending']
  const hasLabel = keys.some((k: string) => labelCandidates.includes(k) && typeof first[k] === 'string')
  const hasDate = keys.some((k: string) => dateCandidates.includes(k))
  const numericKeys = keys.filter((k: string) => typeof first[k] === 'number' && !Number.isNaN(first[k]))
  return (hasLabel || hasDate) && numericKeys.length > 0
}
