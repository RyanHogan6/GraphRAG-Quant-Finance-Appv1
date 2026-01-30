'use client'

import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { AvailableConnection } from '@/lib/journey-types'
import { GRAPH_SCHEMA } from '@/lib/schema'

interface ConnectionSuggestionsProps {
  currentCollectionKey: string
  onAddConnection: (targetKey: string, edge: string, direction: 'OUTBOUND' | 'INBOUND') => void
}

export default function ConnectionSuggestions({ currentCollectionKey, onAddConnection }: ConnectionSuggestionsProps) {
  // Get available connections from schema
  const suggestions = useMemo(() => {
    const currentSchema = GRAPH_SCHEMA[currentCollectionKey]
    if (!currentSchema) return []

    const connections: AvailableConnection[] = currentSchema.connections.map(conn => {
      const targetSchema = GRAPH_SCHEMA[conn.target]

      // Determine rarity based on collection type (for now, hardcoded)
      let rarity: 'common' | 'rare' | 'very-rare' = 'common'
      let isNew = false

      // Mark new/rare connections
      if (conn.target === 'options') {
        isNew = true
        rarity = 'rare'
      }
      if (conn.target === 'futures' || conn.target === 'eia_crude' || conn.target === 'eia_natgas_storage') {
        rarity = 'very-rare'
      }
      if (conn.target === 'sec_exhibits' || conn.target === 'sec_xbrl_data') {
        isNew = true
      }

      return {
        targetKey: conn.target,
        targetLabel: targetSchema?.name || conn.target,
        edge: conn.edge,
        direction: conn.direction,
        description: getConnectionDescription(currentCollectionKey, conn.target, conn.edge),
        rarity,
        isNew
      }
    })

    return connections
  }, [currentCollectionKey])

  if (suggestions.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center p-8">
        <div className="text-center text-gray-500 text-sm">
          <div>No available connections</div>
          <div className="text-xs mt-1">from this collection</div>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full overflow-y-auto p-6 space-y-3">
      <div className="text-sm font-medium text-gray-400 mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
        What can you connect to?
      </div>

      {suggestions.map((suggestion, index) => (
        <motion.button
          key={`${suggestion.targetKey}-${suggestion.edge}`}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05 }}
          onClick={() => onAddConnection(suggestion.targetKey, suggestion.edge, suggestion.direction)}
          className="w-full p-4 bg-dark-800/50 hover:bg-dark-800 border border-green-500/20 hover:border-green-500/40
                   rounded-lg transition-all duration-200 group text-left"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                {/* Arrow icon */}
                <svg className="w-4 h-4 text-green-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>

                {/* Target name */}
                <span className="font-medium text-white group-hover:text-green-400 transition-colors">
                  {suggestion.targetLabel}
                </span>

                {/* Badges */}
                <div className="flex items-center gap-1 ml-auto">
                  {suggestion.isNew && (
                    <span className="px-2 py-0.5 text-xs font-bold bg-green-500/20 text-green-400 rounded-full border border-green-500/30">
                      NEW
                    </span>
                  )}
                  {suggestion.rarity === 'very-rare' && (
                    <span className="px-2 py-0.5 text-xs font-bold bg-amber-500/20 text-amber-400 rounded-full border border-amber-500/30">
                      🔥 RARE
                    </span>
                  )}
                  {suggestion.rarity === 'rare' && suggestion.targetKey === 'options' && (
                    <span className="px-2 py-0.5 text-xs font-bold bg-purple-500/20 text-purple-400 rounded-full border border-purple-500/30">
                      ⚡ HOT
                    </span>
                  )}
                </div>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">
                {suggestion.description}
              </p>

              {/* Edge info */}
              <div className="flex items-center gap-2 mt-2">
                <span className="text-xs text-gray-500 font-mono">
                  {suggestion.edge}
                </span>
                <span className="text-xs text-gray-600">•</span>
                <span className="text-xs text-gray-500">
                  {suggestion.direction === 'OUTBOUND' ? 'Forward link' : 'Reverse link'}
                </span>
              </div>
            </div>

            {/* Add button */}
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500/10 group-hover:bg-green-500/20
                          border border-green-500/30 group-hover:border-green-500/50 flex items-center justify-center">
              <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </div>
          </div>
        </motion.button>
      ))}
    </div>
  )
}

// Helper function to generate connection descriptions
function getConnectionDescription(fromKey: string, toKey: string, edge: string): string {
  const descriptions: Record<string, string> = {
    'company-awards': 'See government contracts awarded to this company',
    'company-marketdata': 'View stock price and technical indicators',
    'company-sec': 'Read SEC filings and disclosures',
    'company-options': 'Analyze options flow and unusual activity',
    'awards-options': 'Find unusual options trading before contract announcements',
    'marketdata-options': 'Correlate stock movement with options activity',
    'options-awards': 'Detect potential insider knowledge before awards',
    'options-sec': 'Spot unusual trading before SEC filings (8-Ks)',
    'company-futures': 'Connect to commodity prices via exposure',
    'futures-eia_crude': 'See crude oil inventory levels affecting prices',
    'futures-eia_natgas_storage': 'See natural gas storage affecting prices',
    'economicdata-futures': 'See macro indicators affecting commodity prices',
    'sec-sec_exhibits': 'View material contracts and exhibits',
    'sec-sec_xbrl_data': 'Analyze structured financial breakdowns',
    'sec-sec_sentences': 'Search filing text semantically',
    'predictionmarkets-company': 'See prediction markets mentioning this company',
  }

  const key = `${fromKey}-${toKey}`
  return descriptions[key] || `Connect via ${edge}`
}
