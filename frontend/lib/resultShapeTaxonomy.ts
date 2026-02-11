/**
 * Output-shape taxonomy for VQB & NL results.
 * Maps (source collection, result shape) to one of 8 view families for reusable viz.
 */

export type ViewFamily =
  | 'time_series'
  | 'ohlc_candlestick'
  | 'screener_leaderboard'
  | 'workup_enriched_entity'
  | 'sentiment_list'
  | 'structured_financials'
  | 'positioning_cot'
  | 'probability_timeline'
  | 'multi_series_macro'
  | 'generic'

/** VQB source keys (from GRAPH_SCHEMA) */
export const VQB_SOURCES = [
  'company', 'marketdata', 'economicdata',
  'sec', 'sec_sections', 'sec_sentences', 'sec_exhibits', 'sec_xbrl_data',
  'predictionmarkets', 'kalshi', 'polymarket_traders', 'polymarket_positions', 'polymarket_price_history',
  'futures', 'cftc', 'eia_crude', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng',
  'options', 'awards'
] as const

/** Map: source → default view family when flat (no enrichments) */
export const SOURCE_TO_VIEW_FAMILY: Record<string, ViewFamily> = {
  company: 'screener_leaderboard',
  marketdata: 'ohlc_candlestick',
  economicdata: 'multi_series_macro',
  sec: 'screener_leaderboard',
  sec_sections: 'screener_leaderboard',
  sec_sentences: 'sentiment_list',
  sec_exhibits: 'sentiment_list',
  sec_xbrl_data: 'structured_financials',
  predictionmarkets: 'screener_leaderboard',
  kalshi: 'screener_leaderboard',
  polymarket_traders: 'screener_leaderboard',
  polymarket_positions: 'screener_leaderboard',
  polymarket_price_history: 'probability_timeline',
  futures: 'ohlc_candlestick',
  cftc: 'positioning_cot',
  eia_crude: 'time_series',
  eia_natgas_storage: 'time_series',
  eia_natgas_production: 'time_series',
  eia_lng: 'time_series',
  options: 'screener_leaderboard',
  awards: 'screener_leaderboard'
}

/** Infer view family from flat result row keys (no metadata). Reusable for NL + VQB. */
export function inferViewFamilyFromShape(keys: string[], firstRow: Record<string, unknown>): ViewFamily {
  const has = (k: string) => keys.includes(k)
  const hasNum = (k: string) => typeof firstRow[k] === 'number' && !Number.isNaN(firstRow[k] as number)

  // OHLC: open, high, low, close (and usually date)
  if (has('open') && has('high') && has('low') && has('close') && (has('date') || has('report_date'))) {
    return 'ohlc_candlestick'
  }

  // Probability timeline: datetime/timestamp + yes_price or yes_probability (0–1)
  if ((has('yes_price') || has('yes_probability')) && (has('datetime') || has('date') || has('timestamp'))) {
    return 'probability_timeline'
  }

  // COT/positioning: commercial vs noncommercial, open interest
  if (has('Open_Interest_All') || (has('Commercial_Positions_Long_All') && has('Noncommercial_Positions_Long_All'))) {
    return 'positioning_cot'
  }

  // Multi-series macro: date + many numeric series (economic)
  const dateLike = ['date', 'Date', 'report_date'].some(k => has(k))
  const numericCount = keys.filter(k => hasNum(k)).length
  if (dateLike && numericCount >= 5 && (has('federal_funds_rate') || has('consumer_price_index_cpi') || has('unemployment_rate'))) {
    return 'multi_series_macro'
  }

  // Time series: date + one or more numeric (no OHLC)
  if (dateLike && (has('close') || has('crude_stocks') || has('total_stocks') || has('value'))) {
    if (has('open') && has('high') && has('low')) return 'ohlc_candlestick'
    return 'time_series'
  }

  // Sentiment list: text + score
  if (has('text') && (has('finbert_score') || has('score'))) {
    return 'sentiment_list'
  }

  // Structured financials: XBRL segments/debt/costs
  if (has('revenue_segments') || has('costs') || (has('debt') && typeof firstRow.debt === 'object')) {
    return 'structured_financials'
  }

  // Screener: entity label + metrics (ticker, company, question, commodity, etc.)
  const labelKeys = ['ticker', 'company', 'question', 'title', 'commodity', 'recipient_name', 'awarding_agency', 'Market_and_Exchange_Names']
  if (labelKeys.some(l => has(l)) && numericCount >= 1) {
    return 'screener_leaderboard'
  }

  return 'generic'
}

/** Get view family from result array (uses first row). */
export function getViewFamilyFromResults(results: any[]): ViewFamily {
  if (!results?.length) return 'generic'
  const first = results[0]
  if (!first || typeof first !== 'object') return 'generic'
  const keys = Object.keys(first).filter(k => !k.startsWith('_'))
  return inferViewFamilyFromShape(keys, first as Record<string, unknown>)
}
