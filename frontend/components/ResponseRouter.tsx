'use client'

import { classifyQuery, getQueryTypeDescription } from '@/lib/responseClassifier'
import type { QueryType } from '@/lib/responseClassifier'
import CompanyWorkup from './CompanyWorkup'
import InsiderTradingSignal from './InsiderTradingSignal'
import SentimentDivergence from './SentimentDivergence'
import ResultsTable from './ResultsTable'

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
  }
}

interface ResponseRouterProps {
  message: Message
  onCompare?: (ticker: string) => void
  peerData?: any
  comparisonMode?: boolean
}

/**
 * ResponseRouter - Routes query results to appropriate UI template
 * Based on query classification and result structure
 */
export default function ResponseRouter({
  message,
  onCompare,
  peerData,
  comparisonMode
}: ResponseRouterProps) {
  // Classify the query to determine best presentation
  const queryType: QueryType = classifyQuery(message)

  console.log('[RESPONSE ROUTER] Query classified as:', queryType)
  console.log('[RESPONSE ROUTER] Message metadata:', message.metadata)
  console.log('[RESPONSE ROUTER] Results count:', message.results?.length)

  // Route to appropriate component based on classification
  switch (queryType) {
    case 'company_deep_dive':
      // Single company comprehensive analysis
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            📊 {getQueryTypeDescription(queryType)}
          </div>
          <CompanyWorkup
            data={message.results![0]}
            onCompare={onCompare}
            peerData={peerData}
            comparisonMode={comparisonMode}
          />
        </div>
      )

    case 'sector_comparison':
      // Multiple companies in same sector
      // TODO: Phase 4 - Create SectorComparison component
      return (
        <div className="space-y-4">
          <div className="text-xs text-yellow-500 italic mb-2">
            🏢 Sector Comparison (Coming in Phase 4)
          </div>
          <div className="text-sm text-gray-400 mb-4">
            Comparing {message.results!.length} companies in {message.results![0]?.sector || 'sector'}
          </div>
          <ResultsTable data={message.results!} />
        </div>
      )

    case 'peer_analysis':
      // Multiple companies comparison
      // TODO: Phase 4 - Create PeerAnalysis component
      return (
        <div className="space-y-4">
          <div className="text-xs text-yellow-500 italic mb-2">
            📈 Peer Analysis (Coming in Phase 4)
          </div>
          <div className="text-sm text-gray-400 mb-4">
            Comparing {message.results!.length} companies
          </div>
          <ResultsTable data={message.results!} />
        </div>
      )

    case 'time_series':
      // Historical trend analysis
      // TODO: Phase 5 - Create TimeSeriesView component
      return (
        <div className="space-y-4">
          <div className="text-xs text-yellow-500 italic mb-2">
            📉 Time Series Analysis (Coming in Phase 5)
          </div>
          <div className="text-sm text-gray-400 mb-4">
            Chart-first view with trend analysis
          </div>
          <ResultsTable data={message.results!} />
        </div>
      )

    case 'metric_focused':
      // Specific metric deep dive
      // TODO: Phase 5 - Create MetricFocusedView component
      return (
        <div className="space-y-4">
          <div className="text-xs text-yellow-500 italic mb-2">
            🎯 Metric Focus (Coming in Phase 5)
          </div>
          <div className="text-sm text-gray-400 mb-4">
            Detailed analysis of specific financial metric
          </div>
          <ResultsTable data={message.results!} />
        </div>
      )

    case 'insider_activity':
      // Options flow and insider trading
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            🔍 {getQueryTypeDescription(queryType)}
          </div>
          <InsiderTradingSignal signals={message.results!} />
        </div>
      )

    case 'news_sentiment':
      // SEC filings sentiment analysis
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            📰 {getQueryTypeDescription(queryType)}
          </div>
          <SentimentDivergence signals={message.results!} />
        </div>
      )

    case 'general_query':
    default:
      // Fallback to generic table display
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            📋 {getQueryTypeDescription(queryType)}
          </div>
          <ResultsTable data={message.results!} />
        </div>
      )
  }
}
