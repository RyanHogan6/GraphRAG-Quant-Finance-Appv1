'use client'

import { useMemo } from 'react'

interface DataSourceAttributionProps {
  collectionsUsed?: string[]
  queryIntent?: string
  lastUpdated?: string
}

const dataSourceInfo: Record<string, { name: string; source: string; updateFrequency: string }> = {
  'Company': { name: 'S&P 500 Companies', source: 'S&P Dow Jones Indices', updateFrequency: 'Quarterly' },
  'MarketData': { name: 'Stock Prices', source: 'Yahoo Finance', updateFrequency: 'Daily (15min delay)' },
  'Award': { name: 'Government Contracts', source: 'USAspending.gov', updateFrequency: 'Daily' },
  'EconomicData': { name: 'Economic Indicators', source: 'Federal Reserve (FRED)', updateFrequency: 'Varies by series' },
  'sec_filings': { name: 'SEC Filings', source: 'SEC EDGAR', updateFrequency: 'Real-time' },
  'sec_sentences': { name: 'SEC Sentiment', source: 'SEC EDGAR + FinBERT', updateFrequency: 'Real-time' },
  'sec_sections': { name: 'SEC Filing Sections', source: 'SEC EDGAR', updateFrequency: 'Real-time' },
  'options_flow': { name: 'Options Activity', source: 'Market Data Feed', updateFrequency: 'Daily (6PM ET)' },
  'futures_prices': { name: 'Commodity Futures', source: 'CME Group', updateFrequency: 'Daily' },
  'commodity_positions': { name: 'CFTC Positions', source: 'CFTC Commitments of Traders', updateFrequency: 'Weekly (Friday)' },
  'eia_crude_inventory': { name: 'Crude Oil Inventory', source: 'EIA Petroleum Status', updateFrequency: 'Weekly (Wed)' },
  'eia_natgas_storage': { name: 'Natural Gas Storage', source: 'EIA Natural Gas', updateFrequency: 'Weekly (Thu)' },
  'eia_natgas_production': { name: 'Gas Production', source: 'EIA Natural Gas', updateFrequency: 'Monthly' },
  'eia_lng_exports': { name: 'LNG Exports', source: 'EIA Natural Gas', updateFrequency: 'Monthly' },
  'prediction_markets_polymarket': { name: 'Polymarket', source: 'Polymarket API', updateFrequency: 'Hourly' },
  'prediction_markets_kalshi': { name: 'Kalshi', source: 'Kalshi API', updateFrequency: 'Hourly' },
  'polymarket_traders': { name: 'Polymarket Traders', source: 'Polymarket API', updateFrequency: 'Hourly' },
  'polymarket_positions': { name: 'Trader Positions', source: 'Polymarket API', updateFrequency: 'Hourly' },
  'polymarket_price_history': { name: 'Price History', source: 'Polymarket API', updateFrequency: 'Hourly' },
}

export default function DataSourceAttribution({ collectionsUsed = [], queryIntent, lastUpdated }: DataSourceAttributionProps) {
  const dataSources = useMemo(() => {
    if (!collectionsUsed || collectionsUsed.length === 0) {
      return []
    }

    return collectionsUsed
      .filter(coll => dataSourceInfo[coll])
      .map(coll => dataSourceInfo[coll])
  }, [collectionsUsed])

  const hasWebSearch = queryIntent === 'hybrid' || queryIntent === 'web_only'

  if (dataSources.length === 0 && !hasWebSearch) {
    return null
  }

  return (
    <div className="mt-4 pt-4 border-t border-gray-700/50">
      <div className="text-xs text-gray-500 space-y-2">
        <div className="flex items-start gap-2">
          <span className="font-semibold text-gray-400 shrink-0">Data Sources:</span>
          <div className="flex flex-wrap gap-2">
            {dataSources.map((source, idx) => (
              <span key={idx} className="bg-dark-800 px-2 py-1 rounded border border-gray-700/30">
                <strong className="text-gray-300">{source.name}</strong>
                <span className="text-gray-500"> via {source.source}</span>
                <span className="text-gray-600 ml-1">({source.updateFrequency})</span>
              </span>
            ))}
            {hasWebSearch && (
              <span className="bg-dark-800 px-2 py-1 rounded border border-blue-700/30">
                <strong className="text-blue-300">Web Search</strong>
                <span className="text-gray-500"> via Perplexity AI</span>
                <span className="text-gray-600 ml-1">(Real-time)</span>
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 text-gray-600">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>
            Data may be delayed. Not financial advice. See{' '}
            <a href="/disclaimer" className="text-gold hover:underline">disclaimer</a> for details.
          </span>
        </div>
      </div>
    </div>
  )
}
