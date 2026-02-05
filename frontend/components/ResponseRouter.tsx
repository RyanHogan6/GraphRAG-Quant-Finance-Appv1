'use client'

import { classifyQuery, getQueryTypeDescription } from '@/lib/responseClassifier'
import type { QueryType } from '@/lib/responseClassifier'
import CompanyWorkup from './CompanyWorkup'
import InsiderTradingSignal from './InsiderTradingSignal'
import SentimentDivergence from './SentimentDivergence'
import ResultsTable from './ResultsTable'
import SectorComparison from './SectorComparison'
import PeerAnalysis from './PeerAnalysis'
import MetricFocusedView from './MetricFocusedView'
import TimeSeriesView from './TimeSeriesView'
import { extractMetric } from '@/lib/responseClassifier'

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
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            🏢 {getQueryTypeDescription(queryType)}
          </div>
          <SectorComparison companies={message.results!} />
        </div>
      )

    case 'peer_analysis':
      // Multiple companies comparison
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            📈 {getQueryTypeDescription(queryType)}
          </div>
          <PeerAnalysis companies={message.results!} />
        </div>
      )

    case 'time_series':
      // Historical trend analysis
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            📉 {getQueryTypeDescription(queryType)}
          </div>
          <TimeSeriesView
            data={message.results!}
            chartData={message.queryPlan?.chart_data}
          />
        </div>
      )

    case 'metric_focused':
      // Specific metric deep dive
      return (
        <div className="space-y-4">
          <div className="text-xs text-gray-500 italic mb-2">
            🎯 {getQueryTypeDescription(queryType)}
          </div>
          <MetricFocusedView
            metric={extractMetric(message.content)}
            data={message.results!}
          />
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
